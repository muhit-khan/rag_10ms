# config.py
"""
Central configuration loader for RAG system.
Simple configuration loader that doesn't rely on pydantic to avoid version conflicts.
"""
from pathlib import Path
from dotenv import load_dotenv
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("config")

# Load environment variables from .env file
env_path = Path(__file__).parent / ".env"
load_dotenv(env_path, override=True)
logger.info(f"Loaded environment variables from {env_path}")

class Config:
    """Simple configuration class that loads values from environment variables."""
    
    def __init__(self):
        # OpenAI
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
        
        # ChromaDB
        self.CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", str(Path("data/processed/chroma")))
        self.CHROMA_COLLECTION = os.getenv("CHROMA_COLLECTION", "rag_collection")
        
        # Redis
        self.REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
        self.REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
        self.REDIS_DB = int(os.getenv("REDIS_DB", "0"))
        
        # FastAPI
        self.API_HOST = os.getenv("API_HOST", "0.0.0.0")
        self.API_PORT = int(os.getenv("API_PORT", "8000"))
        self.API_RELOAD = os.getenv("API_RELOAD", "true").lower() == "true"
        
        # Other
        self.JWT_SECRET = os.getenv("JWT_SECRET", "change-me")
        self.LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2", "False").lower() == "true"
        self.LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY", "")
        
        # PDF Ingest
        self.PDF_PATH = os.getenv("PDF_PATH", "data/raw/")
        
        # Embedding
        self.EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        
        # LLM
        self.LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4.1-mini-2025-04-14")
        self.TEMPERATURE = float(os.getenv("TEMPERATURE", "0.1"))
        self.MAX_TOKENS = int(os.getenv("MAX_TOKENS", "500"))
        
        # Evaluation
        self.GROUND_SCORE_THRESHOLD = float(os.getenv("GROUND_SCORE_THRESHOLD", "0.25"))
        self.COSINE_THRESHOLD = float(os.getenv("COSINE_THRESHOLD", "0.8"))
        
        # Rate limit
        self.RATE_LIMIT = int(os.getenv("RATE_LIMIT", "20"))
        
        logger.info("Configuration loaded successfully")

# Create a singleton instance
config = Config()
