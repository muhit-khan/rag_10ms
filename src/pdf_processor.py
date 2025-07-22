try:
    import fitz  
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False
    fitz = None

import pdfplumber
from typing import List, Dict, Optional
import re
import logging
from pathlib import Path

from utils.helpers import clean_whitespace, detect_language, is_valid_text_chunk

logger = logging.getLogger(__name__)

class PDFProcessor:
    """Extract and preprocess text from PDF documents"""
    
    def __init__(self):
        self.bangla_pattern = re.compile(r'[\u0980-\u09FF]+')
        
    def extract_with_pymupdf(self, pdf_path: str) -> List[Dict]:
        """Extract text using PyMuPDF - better for Bengali fonts"""
        if not PYMUPDF_AVAILABLE or fitz is None:
            logger.warning("PyMuPDF not available, skipping PyMuPDF extraction")
            return []
            
        try:
            doc = fitz.open(pdf_path)
            pages_content = []
            
            logger.info(f"Extracting text from {len(doc)} pages using PyMuPDF")
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                
                # Get text with font information
                text_dict = page.get_text("dict")
                page_text = ""
                
                for block in text_dict["blocks"]:
                    if "lines" in block:
                        for line in block["lines"]:
                            for span in line["spans"]:
                                page_text += span["text"] + " "
                
                # Clean and validate text
                page_text = clean_whitespace(page_text)
                
                if is_valid_text_chunk(page_text, min_length=20):
                    pages_content.append({
                        "page": page_num + 1,
                        "text": page_text,
                        "language": detect_language(page_text),
                        "word_count": len(page_text.split())
                    })
                    
            doc.close()
            logger.info(f"Successfully extracted {len(pages_content)} pages")
            return pages_content
            
        except Exception as e:
            logger.error(f"PyMuPDF extraction failed: {e}")
            return []
    
    def extract_with_pdfplumber(self, pdf_path: str) -> List[Dict]:
        """Alternative extraction using pdfplumber"""
        try:
            pages_content = []
            
            with pdfplumber.open(pdf_path) as pdf:
                logger.info(f"Extracting text from {len(pdf.pages)} pages using pdfplumber")
                
                for page_num, page in enumerate(pdf.pages):
                    text = page.extract_text()
                    
                    if text:
                        text = clean_whitespace(text)
                        
                        if is_valid_text_chunk(text, min_length=20):
                            pages_content.append({
                                "page": page_num + 1,
                                "text": text,
                                "language": detect_language(text),
                                "word_count": len(text.split())
                            })
                            
            logger.info(f"Successfully extracted {len(pages_content)} pages")
            return pages_content
            
        except Exception as e:
            logger.error(f"pdfplumber extraction failed: {e}")
            return []
    
    def extract_text(self, pdf_path: str, method: str = "pdfplumber") -> List[Dict]:
        """Main extraction method with fallback"""
        if not Path(pdf_path).exists():
            logger.error(f"PDF file not found: {pdf_path}")
            return []
        
        logger.info(f"Starting PDF extraction from: {pdf_path}")
        
        # Try primary method - prioritize pdfplumber on Windows
        if method == "pymupdf" and PYMUPDF_AVAILABLE:
            pages = self.extract_with_pymupdf(pdf_path)
            # Fallback to pdfplumber if PyMuPDF fails
            if not pages:
                logger.warning("PyMuPDF failed, trying pdfplumber...")
                pages = self.extract_with_pdfplumber(pdf_path)
        else:
            # Use pdfplumber as primary method
            pages = self.extract_with_pdfplumber(pdf_path)
            # Fallback to PyMuPDF if available and pdfplumber fails
            if not pages and PYMUPDF_AVAILABLE:
                logger.warning("pdfplumber failed, trying PyMuPDF...")
                pages = self.extract_with_pymupdf(pdf_path)
        
        if not pages:
            logger.error("All PDF extraction methods failed")
            return []
        
        # Log extraction summary
        total_words = sum(page["word_count"] for page in pages)
        bangla_pages = sum(1 for page in pages if page["language"] == "bn")
        
        logger.info(f"Extraction complete: {len(pages)} pages, {total_words} words, {bangla_pages} Bengali pages")
        
        return pages
    
    def save_extracted_text(self, pages: List[Dict], output_path: str) -> None:
        """Save extracted text to file for inspection"""
        try:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                for page in pages:
                    f.write(f"=== Page {page['page']} ({page['language']}) ===\n")
                    f.write(page['text'])
                    f.write(f"\n{'='*50}\n\n")
                    
            logger.info(f"Extracted text saved to: {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to save extracted text: {e}")
