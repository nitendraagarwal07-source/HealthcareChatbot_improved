import os
from typing import List, Tuple
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import Document, HumanMessage, AIMessage

def get_embeddings(backend: str, st_model: str, oa_model: str, oa_key: str):
    """
    Create embeddings using either SentenceTransformers or OpenAI.
    """
    backend = (backend or "sentencetransformers").lower()
    if backend == "openai":
        return OpenAIEmbeddings(model=oa_model, api_key=oa_key)
    return HuggingFaceEmbeddings(
        model_name=st_model,
        encode_kwargs={"normalize_embeddings": True},
    )

def build_store(docs: List[Document], embeddings, index_dir: str) -> FAISS:
    os.makedirs(index_dir, exist_ok=True)
    vs = FAISS.from_documents(docs, embeddings)
    vs.save_local(index_dir)
    return vs

def load_store(embeddings, index_dir: str) -> FAISS | None:
    if os.path.isdir(index_dir) and os.listdir(index_dir):
        return FAISS.load_local(index_dir, embeddings, allow_dangerous_deserialization=True)
    return None

def upsert_documents(store: FAISS, docs: List[Document], index_dir: str):
    store.add_documents(docs)
    store.save_local(index_dir)

def get_retriever(store: FAISS, score_threshold: float = 0.30):
    """Always return up to top-3 results above threshold."""
    return store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 3, "score_threshold": score_threshold},
    )

def get_llm(model_name: str, api_key: str):
    return ChatAnthropic(model=model_name, temperature=0, anthropic_api_key=api_key, timeout=90)

def make_prompt():
    system = (
        "You are a careful assistant for Q&A using ONLY the provided context.\n"
        "- If the context does not contain the answer, respond exactly:\n"
        '"I’m sorry, I don’t have that information in my knowledge base."\n'
        "- Do not use outside knowledge.\n"
        "- Be concise. Include inline (source: <file> p.<page>) when possible."
    )
    return ChatPromptTemplate.from_messages(
        [
            ("system", system),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "Question: {question}\n\nContext:\n{context}\n\nAnswer:"),
        ]
    )

def _format_doc(d: Document) -> str:
    import os as _os
    src = d.metadata.get("source", "uploaded")
    page = d.metadata.get("page", None)
    page_str = f" p.{page+1}" if isinstance(page, int) else ""
    return f"(source: {_os.path.basename(src)}{page_str})\n{d.page_content}"

def run_qa(llm, retriever, question: str, chat_history_msgs: List) -> tuple[str, List[Document]]:
    docs = retriever.get_relevant_documents(question)
    if not docs:
        return "I’m sorry, I don’t have that information in my knowledge base.", []
    docs = docs[:3]  # hard-cap
    context = "\n\n---\n\n".join(_format_doc(d) for d in docs)
    prompt = make_prompt()
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"question": question, "context": context, "chat_history": chat_history_msgs})
    return answer, docs

def to_messages(history: List[Tuple[str, str]]) -> List:
    msgs = []
    for role, text in history:
        if role == "human":
            msgs.append(HumanMessage(content=text))
        else:
            msgs.append(AIMessage(content=text))
    return msgs
