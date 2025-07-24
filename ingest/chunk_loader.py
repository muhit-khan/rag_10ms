"""
Chunking logic
"""
import re
import logging

logger = logging.getLogger("ingest.chunk_loader")

def chunk_text(text: str, chunk_size: int = 250, chunk_overlap: int = 30):
    """
    Custom implementation of text chunking.
    Splits text on separators and combines chunks to respect size constraints.
    
    Args:
        text: Text to split into chunks
        chunk_size: Maximum size of each chunk
        chunk_overlap: Number of characters to overlap between chunks
        
    Returns:
        List of text chunks
    """
    # Define separators in order of priority
    separators = ["\n", ".", "?", "!", "।"]
    
    # Function to split text on a separator
    def split_on_separator(text, separator):
        return [chunk + separator for chunk in text.split(separator) if chunk]
    
    # Split text on each separator
    chunks = [text]
    for separator in separators:
        new_chunks = []
        for chunk in chunks:
            if len(chunk) <= chunk_size:
                new_chunks.append(chunk)
            else:
                new_chunks.extend(split_on_separator(chunk, separator))
        chunks = new_chunks
    
    # Combine small chunks to respect chunk_size
    result = []
    current_chunk = ""
    
    for chunk in chunks:
        if len(current_chunk) + len(chunk) <= chunk_size:
            current_chunk += chunk
        else:
            if current_chunk:
                result.append(current_chunk)
            current_chunk = chunk
    
    if current_chunk:
        result.append(current_chunk)
    
    # Add overlapping content
    if chunk_overlap > 0 and len(result) > 1:
        overlapped_result = [result[0]]
        for i in range(1, len(result)):
            prev_chunk = result[i-1]
            current_chunk = result[i]
            
            # Add overlap from previous chunk if possible
            overlap_size = min(chunk_overlap, len(prev_chunk))
            if overlap_size > 0:
                overlapped_chunk = prev_chunk[-overlap_size:] + current_chunk
                overlapped_result.append(overlapped_chunk)
            else:
                overlapped_result.append(current_chunk)
        
        result = overlapped_result
    
    logger.info(f"Split text into {len(result)} chunks")
    return result
