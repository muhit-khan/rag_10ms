"""
ChromaDB client initializer
"""
from chromadb import PersistentClient
from config import config
from pathlib import Path

def get_chroma_client():
    persist_dir = Path(config.CHROMA_PERSIST_DIR)
    persist_dir.mkdir(parents=True, exist_ok=True)
    client = PersistentClient(path=str(persist_dir))
    return client

def get_collection():
    client = get_chroma_client()
    collection = client.get_or_create_collection(config.CHROMA_COLLECTION)
    return collection
