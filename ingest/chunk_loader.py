"""
Chunking logic
"""
from langchain.text_splitter import RecursiveCharacterTextSplitter

def chunk_text(text: str, chunk_size: int = 250, chunk_overlap: int = 30):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n", ".", "?", "!", "।"]
    )
    return splitter.split_text(text)
