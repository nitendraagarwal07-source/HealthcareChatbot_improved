# HCL Healthcare RAG Chatbot (Claude + FAISS)

A production-ready Retrieval-Augmented Generation chatbot:
- **Streamlit** UI with HCL blue/white styling
- **Claude (Anthropic)** as the LLM
- **FAISS** vector store
- **PDF/Text ingestion** via LangChain loaders
- **Top-3 retrieval** per query
- Guardrails: scope-only answers, category blocks, audit logging

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.sample .env
# Edit .env with your keys / model choices
streamlit run app.py
