"""
ChromaDB storage logic
"""
from typing import List
import logging

logger = logging.getLogger("ingest.storage")

def store_chunks(collection, all_chunks: List[str], embeddings: List, all_metadata: List[dict], all_ids: List[str], batch_size: int = 100):
    """
    Store chunks, embeddings, and metadata in ChromaDB in batches.
    """
    logger.info(f"Storing {len(all_chunks)} chunks in ChromaDB")
    for i in range(0, len(all_chunks), batch_size):
        end_idx = min(i + batch_size, len(all_chunks))
        try:
            collection.add(
                documents=all_chunks[i:end_idx],
                embeddings=embeddings[i:end_idx],
                metadatas=all_metadata[i:end_idx],
                ids=all_ids[i:end_idx]
            )
            logger.info(f"Stored batch {i//batch_size + 1}/{(len(all_chunks)-1)//batch_size + 1}")
        except Exception as e:
            logger.error(f"Error storing batch {i//batch_size + 1}: {str(e)}")
