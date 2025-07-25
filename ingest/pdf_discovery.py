"""
PDF file discovery logic
"""
import logging
from pathlib import Path
from typing import List

logger = logging.getLogger("ingest.pdf_discovery")

def find_pdf_files(directory: str) -> List[Path]:
    """
    Find all PDF and text files in the given directory.
    
    Args:
        directory: Directory path to search for document files
        
    Returns:
        List[Path]: List of document file paths
    """
    directory_path = Path(directory)
    
    if not directory_path.exists():
        logger.error(f"Directory does not exist: {directory}")
        return []
    
    if not directory_path.is_dir():
        logger.error(f"Path is not a directory: {directory}")
        return []
    
    # Find all PDF files recursively
    pdf_files = list(directory_path.rglob("*.pdf"))
    
    # Also check for PDF files with uppercase extension
    pdf_files.extend(list(directory_path.rglob("*.PDF")))
    
    # Add text files for testing
    pdf_files.extend(list(directory_path.rglob("*.txt")))
    pdf_files.extend(list(directory_path.rglob("*.TXT")))
    
    # Remove duplicates and sort
    pdf_files = sorted(list(set(pdf_files)))
    
    logger.info(f"Found {len(pdf_files)} document files in {directory}")
    for pdf_file in pdf_files:
        logger.info(f"  - {pdf_file}")
    
    return pdf_files

def validate_pdf_file(pdf_path: Path) -> bool:
    """
    Validate if a file is a readable PDF.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        bool: True if file is a valid PDF, False otherwise
    """
    try:
        if not pdf_path.exists():
            logger.warning(f"PDF file does not exist: {pdf_path}")
            return False
        
        if not pdf_path.is_file():
            logger.warning(f"Path is not a file: {pdf_path}")
            return False
        
        # Check file size (should be > 0)
        if pdf_path.stat().st_size == 0:
            logger.warning(f"PDF file is empty: {pdf_path}")
            return False
        
        # Basic PDF header check
        with open(pdf_path, 'rb') as f:
            header = f.read(4)
            if header != b'%PDF':
                logger.warning(f"File does not have PDF header: {pdf_path}")
                return False
        
        logger.info(f"PDF file validated: {pdf_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error validating PDF file {pdf_path}: {str(e)}")
        return False

def get_pdf_info(pdf_path: Path) -> dict:
    """
    Get basic information about a PDF file.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        dict: PDF file information
    """
    try:
        stat = pdf_path.stat()
        return {
            "filename": pdf_path.name,
            "path": str(pdf_path),
            "size_bytes": stat.st_size,
            "size_mb": round(stat.st_size / (1024 * 1024), 2),
            "modified_time": stat.st_mtime,
            "is_valid": validate_pdf_file(pdf_path)
        }
    except Exception as e:
        logger.error(f"Error getting PDF info for {pdf_path}: {str(e)}")
        return {
            "filename": pdf_path.name,
            "path": str(pdf_path),
            "error": str(e),
            "is_valid": False
        }
