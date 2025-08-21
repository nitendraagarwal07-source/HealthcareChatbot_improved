from pathlib import Path
from typing import List
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

def load_and_split(file_path: str) -> List[Document]:
    """Load a file with LangChain loaders and split into chunks."""
    ext = Path(file_path).suffix.lower()
    if ext == ".pdf":
        docs = PyPDFLoader(file_path).load()
    else:
        docs = TextLoader(file_path, encoding="utf-8").load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=150,
        add_start_index=True,
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.split_documents(docs)
