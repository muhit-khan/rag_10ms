"""Utility functions for the RAG system"""

import re
from typing import List, Optional
from pathlib import Path

def ensure_directory_exists(path: str) -> Path:
    """Create directory if it doesn't exist"""
    dir_path = Path(path)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path

def detect_language(text: str) -> str:
    """Detect if text is primarily Bengali or English"""
    bangla_chars = len(re.findall(r'[\u0980-\u09FF]', text))
    total_chars = len(re.findall(r'[^\s\n\r]', text))
    
    if total_chars == 0:
        return "unknown"
    
    bangla_ratio = bangla_chars / total_chars
    
    if bangla_ratio > 0.3:
        return "bn"
    else:
        return "en"

def clean_whitespace(text: str) -> str:
    """Clean excessive whitespace from text"""
    # Replace multiple whitespace with single space
    text = re.sub(r'\s+', ' ', text)
    # Remove leading/trailing whitespace
    text = text.strip()
    return text

def is_valid_text_chunk(text: str, min_length: int = 10) -> bool:
    """Check if text chunk is valid for processing"""
    if not text or len(text.strip()) < min_length:
        return False
    
    # Check if chunk has meaningful content (not just punctuation/whitespace)
    meaningful_chars = re.findall(r'[\u0980-\u09FF\w]', text)
    return len(meaningful_chars) >= min_length // 2

def normalize_query(query: str) -> str:
    """Normalize user query for better matching"""
    # Clean whitespace
    query = clean_whitespace(query)
    
    # Remove question marks and exclamation points for processing
    query = re.sub(r'[?!।]+$', '', query)
    
    return query.strip()
