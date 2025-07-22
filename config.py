"""
Simplified configuration for RAG system
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=True)

# API Configuration
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))

# OpenAI Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# LLM Configuration
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4.1-mini-2025-04-14")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "1000"))

# Embedding Configuration
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# Vector Store Configuration
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma_db")

# Processing Configuration
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "600"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))
TOP_K_RETRIEVAL = int(os.getenv("TOP_K_RETRIEVAL", "5"))
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.6"))

# Data Paths
DATA_DIR = "./data"
RAW_DATA_DIR = "./data/raw"
