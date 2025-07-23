# config.py
"""
Central configuration loader for RAG system.
Loads all parameters from .env using pydantic-settings.
"""
from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path
from dotenv import load_dotenv
import os

# Load environment variables from .env file
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path, override=True)

# Define the configuration class using pydantic
# This class will automatically load environment variables defined in .env file
class Config(BaseSettings):
    # OpenAI
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    # ChromaDB
    CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", str(Path("data/processed/chroma")))
    CHROMA_COLLECTION: str = os.getenv("CHROMA_COLLECTION", "rag_collection")
    # Redis
    REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", 6379))
    REDIS_DB: int = int(os.getenv("REDIS_DB", 0))
    # FastAPI
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", 8000))
    API_RELOAD: bool = os.getenv("API_RELOAD", "true").lower() == "true"
    # Other
    JWT_SECRET: str = os.getenv("JWT_SECRET", "change-me")
    LANGCHAIN_TRACING_V2: bool = os.getenv("LANGCHAIN_TRACING_V2", "False").lower() == "true"
    LANGSMITH_API_KEY: str = os.getenv("LANGSMITH_API_KEY", "")
    # PDF Ingest
    PDF_PATH: str = os.getenv("PDF_PATH", "data/raw/")
    # Embedding
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    # LLM
    LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-4.1-mini-2025-04-14")
    # Evaluation
    GROUND_SCORE_THRESHOLD: float = float(os.getenv("GROUND_SCORE_THRESHOLD", 0.25))
    COSINE_THRESHOLD: float = float(os.getenv("COSINE_THRESHOLD", 0.8))
    # Rate limit
    RATE_LIMIT: int = int(os.getenv("RATE_LIMIT", 20))
    # ...add more as needed

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

config = Config()
