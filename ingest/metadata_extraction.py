"""
Metadata extraction logic for PDF documents
"""
import logging
import re
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

logger = logging.getLogger("ingest.metadata_extraction")

def extract_metadata(pdf_path: Path, text: str) -> Dict[str, Any]:
    """
    Extract metadata from PDF file and its text content.
    
    Args:
        pdf_path: Path to the PDF file
        text: Extracted text content from the PDF
        
    Returns:
        Dict[str, Any]: Metadata dictionary
    """
    metadata = {}
    
    try:
        # Basic file metadata
        stat = pdf_path.stat()
        metadata.update({
            "source": str(pdf_path),
            "filename": pdf_path.name,
            "file_size": stat.st_size,
            "created_time": datetime.fromtimestamp(stat.st_ctime).isoformat(),
            "modified_time": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "extraction_time": datetime.now().isoformat()
        })
        
        # Text-based metadata
        metadata.update(_extract_text_metadata(text))
        
        # Document structure metadata
        metadata.update(_extract_structure_metadata(text))
        
        # Language detection (basic)
        metadata["language"] = _detect_language(text)
        
        logger.info(f"Extracted metadata for {pdf_path.name}")
        return metadata
        
    except Exception as e:
        logger.error(f"Error extracting metadata from {pdf_path}: {str(e)}")
        return {
            "source": str(pdf_path),
            "filename": pdf_path.name,
            "error": str(e),
            "extraction_time": datetime.now().isoformat()
        }

def _extract_text_metadata(text: str) -> Dict[str, Any]:
    """
    Extract metadata from text content.
    
    Args:
        text: Text content
        
    Returns:
        Dict[str, Any]: Text-based metadata
    """
    metadata = {}
    
    try:
        # Basic text statistics
        metadata["char_count"] = len(text)
        metadata["word_count"] = len(text.split())
        metadata["line_count"] = len(text.split('\n'))
        metadata["paragraph_count"] = len([p for p in text.split('\n\n') if p.strip()])
        
        # Extract potential title (first non-empty line)
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        if lines:
            metadata["potential_title"] = lines[0][:100]  # First 100 chars
        
        # Extract potential author/subject patterns
        author_patterns = [
            r'লেখক[:\s]*([^\n]+)',
            r'রচনা[:\s]*([^\n]+)',
            r'Author[:\s]*([^\n]+)',
            r'By[:\s]*([^\n]+)'
        ]
        
        for pattern in author_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                metadata["potential_author"] = match.group(1).strip()[:50]
                break
        
        # Extract chapter/section information
        chapter_patterns = [
            r'অধ্যায়[:\s]*(\d+)',
            r'পরিচ্ছেদ[:\s]*(\d+)',
            r'Chapter[:\s]*(\d+)',
            r'Section[:\s]*(\d+)'
        ]
        
        chapters = []
        for pattern in chapter_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            chapters.extend(matches)
        
        if chapters:
            metadata["chapters"] = list(set(chapters))
            metadata["chapter_count"] = len(set(chapters))
        
        return metadata
        
    except Exception as e:
        logger.error(f"Error extracting text metadata: {str(e)}")
        return {"text_metadata_error": str(e)}

def _extract_structure_metadata(text: str) -> Dict[str, Any]:
    """
    Extract document structure metadata.
    
    Args:
        text: Text content
        
    Returns:
        Dict[str, Any]: Structure metadata
    """
    metadata = {}
    
    try:
        # Count different types of punctuation (indicators of structure)
        metadata["sentence_endings"] = len(re.findall(r'[।.!?]', text))
        metadata["question_marks"] = len(re.findall(r'[?]', text))
        metadata["exclamations"] = len(re.findall(r'[!]', text))
        metadata["bengali_sentence_endings"] = len(re.findall(r'[।]', text))
        
        # Count numbered lists
        numbered_lists = len(re.findall(r'^\s*\d+[.)]\s', text, re.MULTILINE))
        metadata["numbered_lists"] = numbered_lists
        
        # Count bullet points
        bullet_points = len(re.findall(r'^\s*[•·▪▫-]\s', text, re.MULTILINE))
        metadata["bullet_points"] = bullet_points
        
        # Estimate reading time (assuming 200 words per minute)
        word_count = len(text.split())
        metadata["estimated_reading_time_minutes"] = round(word_count / 200, 1)
        
        return metadata
        
    except Exception as e:
        logger.error(f"Error extracting structure metadata: {str(e)}")
        return {"structure_metadata_error": str(e)}

def _detect_language(text: str) -> str:
    """
    Basic language detection for Bengali and English.
    
    Args:
        text: Text content
        
    Returns:
        str: Detected language ('bengali', 'english', 'mixed', or 'unknown')
    """
    try:
        # Count Bengali characters (Unicode range for Bengali)
        bengali_chars = len(re.findall(r'[\u0980-\u09FF]', text))
        
        # Count English characters
        english_chars = len(re.findall(r'[a-zA-Z]', text))
        
        total_chars = bengali_chars + english_chars
        
        if total_chars == 0:
            return "unknown"
        
        bengali_ratio = bengali_chars / total_chars
        english_ratio = english_chars / total_chars
        
        if bengali_ratio > 0.7:
            return "bengali"
        elif english_ratio > 0.7:
            return "english"
        elif bengali_ratio > 0.3 and english_ratio > 0.3:
            return "mixed"
        else:
            return "unknown"
            
    except Exception as e:
        logger.error(f"Error detecting language: {str(e)}")
        return "unknown"

def extract_page_metadata(text: str, page_number: Optional[int] = None) -> Dict[str, Any]:
    """
    Extract metadata for a specific page.
    
    Args:
        text: Page text content
        page_number: Page number (if known)
        
    Returns:
        Dict[str, Any]: Page metadata
    """
    metadata = {}
    
    try:
        if page_number is not None:
            metadata["page_number"] = page_number
        
        # Basic page statistics
        metadata["page_char_count"] = len(text)
        metadata["page_word_count"] = len(text.split())
        metadata["page_line_count"] = len(text.split('\n'))
        
        # Check if page contains specific content types
        metadata["has_questions"] = bool(re.search(r'[?]', text))
        metadata["has_numbers"] = bool(re.search(r'\d+', text))
        metadata["has_bengali"] = bool(re.search(r'[\u0980-\u09FF]', text))
        metadata["has_english"] = bool(re.search(r'[a-zA-Z]', text))
        
        return metadata
        
    except Exception as e:
        logger.error(f"Error extracting page metadata: {str(e)}")
        return {"page_metadata_error": str(e)}
