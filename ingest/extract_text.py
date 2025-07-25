"""
Document text extraction logic
"""
import pdfminer.high_level
import pdfminer.layout
import logging
from pathlib import Path

logger = logging.getLogger("ingest.extract_text")

def extract_text_from_pdf(file_path: str) -> str:
    """
    Extract text from PDF or text file with Bengali support.
    
    Args:
        file_path: Path to the document file
        
    Returns:
        str: Extracted text content
    """
    file_path_obj = Path(file_path)
    
    # Handle text files directly
    if file_path_obj.suffix.lower() in ['.txt']:
        return extract_text_from_txt(file_path)
    
    # Handle PDF files
    try:
        # Configure LAParams for better Bengali text extraction
        laparams = pdfminer.layout.LAParams(
            detect_vertical=True,  # Important for Bengali text
            word_margin=0.1,
            char_margin=2.0,
            line_margin=0.5,
            boxes_flow=0.5
        )
        
        # Extract text with custom parameters
        text = pdfminer.high_level.extract_text(
            file_path,
            laparams=laparams,
            codec='utf-8'
        )
        
        if text and text.strip():
            logger.info(f"Successfully extracted {len(text)} characters from {file_path}")
            return text
        else:
            logger.warning(f"No text extracted from {file_path}")
            return ""
            
    except Exception as e:
        logger.error(f"Error extracting text from {file_path}: {str(e)}")
        return ""

def extract_text_from_txt(file_path: str) -> str:
    """
    Extract text from a text file with proper encoding handling.
    
    Args:
        file_path: Path to the text file
        
    Returns:
        str: Text content
    """
    try:
        # Try UTF-8 first
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
            logger.info(f"Successfully read {len(text)} characters from {file_path}")
            return text
    except UnicodeDecodeError:
        # Fallback to other encodings
        encodings = ['utf-8-sig', 'latin-1', 'cp1252']
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    text = f.read()
                    logger.info(f"Successfully read {len(text)} characters from {file_path} using {encoding}")
                    return text
            except UnicodeDecodeError:
                continue
        
        logger.error(f"Could not decode text file {file_path} with any encoding")
        return ""
    except Exception as e:
        logger.error(f"Error reading text file {file_path}: {str(e)}")
        return ""

def extract_text_with_fallback(file_path: str) -> str:
    """
    Extract text with multiple fallback methods.
    
    Args:
        file_path: Path to the document file
        
    Returns:
        str: Extracted text content
    """
    # Primary method: format-specific extraction
    text = extract_text_from_pdf(file_path)
    
    if text and text.strip():
        return text
    
    # For PDFs, try fallback without LAParams
    if Path(file_path).suffix.lower() in ['.pdf']:
        try:
            text = pdfminer.high_level.extract_text(file_path)
            if text and text.strip():
                logger.info(f"Fallback extraction successful for {file_path}")
                return text
        except Exception as e:
            logger.error(f"Fallback extraction failed for {file_path}: {str(e)}")
    
    # If all methods fail, return empty string
    logger.error(f"All text extraction methods failed for {file_path}")
    return ""
