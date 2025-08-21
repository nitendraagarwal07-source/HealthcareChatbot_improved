import os
import time
import tempfile
from pathlib import Path
import streamlit as st

from config import settings
from loaders import load_and_split
from rag_pipeline import (
    get_embeddings,
    build_store,
    load_store,
    upsert_documents,
    get_retriever,
    get_llm,
    run_qa,
    to_messages,
)
from rules import is_allowed, BlockReason
from audit import log_query, log_response
from safety import outside_scope, grounded_enough, REFUSAL_OOS, REFUSAL_EMPTY

HCL_BLUE = "#0D47A1"
WHITE = "#FFFFFF"

st.set_page_config(page_title="HCL Healthcare RAG Chatbot", page_icon="🩺", layout="wide")

# --- Global CSS ---
st.markdown(
    f"""
    <style>
      .stApp {{ background-color: {WHITE}; }}
      .hcl-header {{
          background: {HCL_BLUE}; color: {WHITE};
          padding: 18px 20px; border-radius: 14px; margin-bottom: 14px;
          display: flex; align-items: center; justify-content: space-between;
      }}
      .hcl-title {{ font-size: 1.25rem; font-weight: 700; margin: 0; }}
      .hcl-subtitle {{ margin: 4px 0 0 0; opacity: 0.9; font-size: 0.95rem; }}
      .hcl-card {{
          background: #F8FAFF; border: 1px solid #E3ECFF; border-radius: 14px;
          padding: 16px 16px 10px 16px; margin-bottom: 12px;
      }}
      .hcl-chip {{
          display: inline-block; padding: 4px 10px; font-size: 12px; border-radius: 999px;
          border: 1px solid #cbd6ff; background: #EDF2FF; color: #2941C0; margin-left: 8px;
      }}
      .chat-window {{
          border: 1px solid #E3ECFF; border-radius: 14px; padding: 12px;
          height: 62vh; overflow-y: auto; background: #FFFFFF;
      }}
      .msg {{ margin: 8px 0; display: flex; }}
      .user-bubble {{
          margin-left: auto; max-width: 82%;
          background: #E8F0FF; border: 1px solid #D5E2FF; color: #0B2D73;
          padding: 10px 12px; border-radius: 12px 12px 2px 12px; white-space: pre-wrap;
      }}
      .bot-bubble {{
          margin-right: auto; max-width: 82%;
          background: #F7F9FF; border: 1px solid #E3ECFF; color: #0B2D73;
          padding: 10px 12px; border-radius: 12px 12px 12px 2px; white-space: pre-wrap;
      }}
      .src-list {{ font-size: 12px; color: #405089; margin-top: 6px; }}
      .src-list code {{ background: transparent; color: #1f2f6b; }}
      .stTextInput>div>div>input {{ border-radius: 12px; }}
      .stChatInput textarea {{ border-radius: 12px; }}
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Session init ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "embeddings" not in st.session_state:
    st.session_state.embeddings = get_embeddings(
        backend=settings.EMBEDDING_BACKEND,
        st_model=settings.EMBEDDING_MODEL,
        oa_model=settings.OPENAI_EMBED_MODEL,
        oa_key=settings.OPENAI_API_KEY,
    )

# --- Header & Reset ---
with st.container():
    left, right = st.columns([0.72, 0.28])
    with left:
        st.markdown(
            f"""
            <div class="hcl-header">
              <div>
                <div class="hcl-title">🩺 HCL Healthcare – RAG Chatbot (Claude + FAISS)</div>
                <div class="hcl-subtitle">Answers strictly from your uploaded documents. If not found, I will say:
                  <em>“I’m sorry, I don’t have that information in my knowledge base.”</em>
                </div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        if st.button("🔁 Reset Conversation", use_container_width=True):
            st.session_state.chat_history = []
            st.toast("Conversation reset", icon="🧼")

# --- Upload Section (Top) ---
with st.container():
    st.markdown('<div class="hcl-card">', unsafe_allow_html=True)
    up_left, up_mid, up_right = st.columns([0.60, 0.20, 0.20])
    with up_left:
        uploads = st.file_uploader("📥 Upload PDFs / Text", type=["pdf", "txt", "md"], accept_multiple_files=True)
    with up_mid:
        ingest_btn = st.button("➕ Add to Knowledge Base", use_container_width=True)
    with up_right:
        clear_btn = st.button("🗑️ Clear KB", type="secondary", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

# --- Sidebar Settings ---
st.sidebar.title("⚙️ Settings")
anthropic_key = st.sidebar.text_input("Anthropic API Key", value=os.environ.get("ANTHROPIC_API_KEY", ""), type="password")
if anthropic_key:
    os.environ["ANTHROPIC_API_KEY"] = anthropic_key

if settings.EMBEDDING_BACKEND == "openai":
    openai_key = st.sidebar.text_input("OpenAI API Key", value=os.environ.get("OPENAI_API_KEY", ""), type="password")
    if openai_key:
        os.environ["OPENAI_API_KEY"] = openai_key

llm_model = st.sidebar.text_input("LLM Model", value=settings.LLM_MODEL)
score_threshold = st.sidebar.slider("Score threshold (higher = stricter)", 0.0, 0.9, 0.30, 0.01)

index_dir = settings.INDEX_DIR
Path(index_dir).mkdir(parents=True, exist_ok=True)

# Load existing index on first run
if st.session_state.vectorstore is None:
    with st.spinner("Loading existing index (if any)..."):
        st.session_state.vectorstore = load_store(st.session_state.embeddings, index_dir)

# KB clear
if clear_btn:
    st.session_state.vectorstore = None
    if Path(index_dir).exists():
        for p in Path(index_dir).glob("*"):
            try:
                p.unlink()
            except:
                pass
    st.success("Knowledge base cleared.")

# Ingest
if ingest_btn and uploads:
    with st.spinner("Ingesting & indexing documents..."):
        temp_dir = tempfile.mkdtemp(prefix="hclhc_")
        all_chunks = []
        for f in uploads:
            dest = Path(temp_dir) / f.name
            with open(dest, "wb") as out:
                out.write(f.read())
            chunks = load_and_split(str(dest))
            for c in chunks:
                c.metadata["source"] = str(dest)
            all_chunks.extend(chunks)

        if not all_chunks:
            st.warning("No content extracted. Please check your files.")
        else:
            if st.session_state.vectorstore is None:
                st.session_state.vectorstore = build_store(all_chunks, st.session_state.embeddings, index_dir)
            else:
                upsert_documents(st.session_state.vectorstore, all_chunks, index_dir)
            st.success(f"Ingested {len(all_chunks)} chunks.")
            time.sleep(0.25)

# Status chips
kb_status = "Loaded" if st.session_state.vectorstore is not None else "Empty"
st.markdown(
    f"**Knowledge Base:** <span class='hcl-chip'>{kb_status}</span> "
    f"<span class='hcl-chip'>Embeddings: {settings.EMBEDDING_BACKEND}</span> "
    f"<span class='hcl-chip'>Index: {Path(index_dir).name}</span>",
    unsafe_allow_html=True,
)

st.write("")  # spacer

# --- Chat Window ---
st.markdown("#### 💬 Chat")
with st.container():
    st.markdown('<div class="chat-window">', unsafe_allow_html=True)

    # Render history
    for role, text in st.session_state.chat_history:
        if role == "human":
            st.markdown(f'<div class="msg"><div class="user-bubble">{text}</div></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="msg"><div class="bot-bubble">{text}</div></div>', unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

user_query = st.chat_input("Ask a question about the uploaded documents…")

# --- Chat Handling with Guardrails & Logging ---
if user_query:
    st.session_state.chat_history.append(("human", user_query))

    # Category guard
    rule = is_allowed(user_query)
    if not rule.allowed:
        refusal = "This question is outside the scope of my knowledge base."
        st.session_state.chat_history.append(("ai", refusal))
        st.markdown(f'<div class="msg"><div class="bot-bubble">{refusal}</div></div>', unsafe_allow_html=True)
        log_query(query=user_query, blocked=True, reason=str(rule.reason), details=rule.details)
        log_response(answer=refusal, grounded=False, docs=[], refusal_type="category_block")
    else:
        log_query(query=user_query, blocked=False, reason="none")
        if st.session_state.vectorstore is None:
            reply = "Please upload and ingest documents first."
            st.session_state.chat_history.append(("ai", reply))
            st.markdown(f'<div class="msg"><div class="bot-bubble">{reply}</div></div>', unsafe_allow_html=True)
            log_response(answer=reply, grounded=False, docs=[], refusal_type="no_kb")
        else:
            retriever = get_retriever(st.session_state.vectorstore, score_threshold=score_threshold)  # k=3 enforced
            llm = get_llm(llm_model, os.environ.get("ANTHROPIC_API_KEY", ""))

            with st.spinner("Thinking…"):
                answer, docs = run_qa(llm, retriever, user_query, to_messages(st.session_state.chat_history))

            if outside_scope(docs) or not grounded_enough(answer, docs):
                refusal = "This question is outside the scope of my knowledge base."
                st.session_state.chat_history.append(("ai", refusal))
                st.markdown(f'<div class="msg"><div class="bot-bubble">{refusal}</div></div>', unsafe_allow_html=True)
                log_response(answer=refusal, grounded=False, docs=[d.metadata for d in docs], refusal_type="oos_or_not_grounded")
            else:
                st.session_state.chat_history.append(("ai", answer))
                st.markdown(f'<div class="msg"><div class="bot-bubble">{answer}</div></div>', unsafe_allow_html=True)
                log_response(
                    answer=answer,
                    grounded=True,
                    docs=[{"source": Path(d.metadata.get("source","")).name, "page": d.metadata.get("page")} for d in docs],
                    refusal_type=None,
                )
                if docs:
                    src_lines = []
                    for i, d in enumerate(docs[:3], 1):
                        src = Path(d.metadata.get("source", "uploaded")).name
                        page = d.metadata.get("page", None)
                        page_str = f"(p.{page+1})" if isinstance(page, int) else ""
                        src_lines.append(f"**{i}.** `{src}` {page_str}")
                    st.markdown(f"<div class='src-list'>{'<br>'.join(src_lines)}</div>", unsafe_allow_html=True)
