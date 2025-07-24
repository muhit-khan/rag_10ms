"""
Embedding creation logic
"""
from openai import OpenAI
from config import config
from typing import List
import logging

logger = logging.getLogger("ingest.embedding")

def create_embeddings(chunks: List[str], batch_size: int = 96) -> List:
    """
    Create embeddings for text chunks using OpenAI API.
    Processes chunks in batches to optimize API usage.
    Returns a list of embeddings compatible with ChromaDB.
    """
    client = OpenAI(api_key=config.OPENAI_API_KEY)
    embeddings = []
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        try:
            response = client.embeddings.create(
                input=batch,
                model=config.EMBEDDING_MODEL
            )
            batch_embeddings = [item.embedding for item in response.data]
            embeddings.extend(batch_embeddings)
            logger.info(f"Created embeddings for batch {i//batch_size + 1}/{(len(chunks)-1)//batch_size + 1}")
        except Exception as e:
            logger.error(f"Error creating embeddings for batch {i//batch_size + 1}: {str(e)}")
            embeddings.extend([None] * len(batch))
    return embeddings
