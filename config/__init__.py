from pydantic_settings import BaseSettings
from typing import Optional
import os
from pathlib import Path
from dotenv import load_dotenv

# Force load .env file first, overriding any existing environment variables
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path, override=True)

class Settings(BaseSettings):
    # API Configuration - from .env
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", "8000"))
    API_TITLE: str = "Multilingual RAG System"
    API_VERSION: str = "1.0.0"
    
    # OpenAI API Configuration - from .env
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    
    # LLM Configuration - from .env
    LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-3.5-turbo")
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.1"))
    MAX_TOKENS: int = int(os.getenv("MAX_TOKENS", "1000"))
    
    # Fine-tuning Configuration - from .env
    ENABLE_FINE_TUNING: bool = os.getenv("ENABLE_FINE_TUNING", "false").lower() == "true"
    FINE_TUNED_MODEL_ID: Optional[str] = os.getenv("FINE_TUNED_MODEL_ID")
    
    # Embedding Configuration - from .env
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    EMBEDDING_DIMENSION: int = int(os.getenv("EMBEDDING_DIMENSION", "384"))
    
    # Vector Store Configuration - from .env
    VECTOR_STORE_TYPE: str = os.getenv("VECTOR_STORE_TYPE", "chromadb")
    CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma_db")
    FAISS_INDEX_PATH: str = "./data/faiss_index"
    
    # Chunking Configuration - from .env
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "512"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "50"))
    
    # Retrieval Configuration - from .env
    TOP_K_RETRIEVAL: int = int(os.getenv("TOP_K_RETRIEVAL", "5"))
    SIMILARITY_THRESHOLD: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.7"))
    
    # Memory Configuration - from .env
    MAX_CHAT_HISTORY: int = int(os.getenv("MAX_CHAT_HISTORY", "10"))
    CONTEXT_WINDOW_SIZE: int = int(os.getenv("CONTEXT_WINDOW_SIZE", "4000"))
    
    # Data Paths (computed from base paths)
    DATA_DIR: str = "./data"
    RAW_DATA_DIR: str = "./data/raw"
    PROCESSED_DATA_DIR: str = "./data/processed"
    
    class Config:
        env_file = str(Path(__file__).parent.parent / ".env")
        env_file_encoding = 'utf-8'
        case_sensitive = False

settings = Settings()
