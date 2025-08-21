from dataclasses import dataclass
from dotenv import load_dotenv
import os

load_dotenv()

@dataclass
class Settings:
    ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "claude-3-5-sonnet-2024-10-22")

    EMBEDDING_BACKEND: str = os.getenv("EMBEDDING_BACKEND", "sentencetransformers").lower()

    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_EMBED_MODEL: str = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")

    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

    INDEX_DIR: str = os.getenv("INDEX_DIR", "storage/faiss_index")

settings = Settings()
