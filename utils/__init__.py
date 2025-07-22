"""Utility module for the RAG system"""

from .logger import setup_logging
from .helpers import (
    ensure_directory_exists,
    detect_language, 
    clean_whitespace,
    is_valid_text_chunk,
    normalize_query
)

__all__ = [
    "setup_logging",
    "ensure_directory_exists", 
    "detect_language",
    "clean_whitespace",
    "is_valid_text_chunk", 
    "normalize_query"
]
