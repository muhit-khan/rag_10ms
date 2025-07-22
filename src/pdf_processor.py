try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False
    fitz = None  # type: ignore

import pdfplumber
from typing import List, Dict, Optional, Any
import re
import logging
from pathlib import Path
import unicodedata

# Additional imports for better Bengali processing
try:
    import cv2
    import numpy as np
    from PIL import Image
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False

from utils.helpers import clean_whitespace, detect_language, is_valid_text_chunk

logger = logging.getLogger(__name__)

class BengaliPDFProcessor:
    """Enhanced PDF processor specifically designed for Bengali text extraction"""
    
    def __init__(self):
        self.bangla_pattern = re.compile(r'[\u0980-\u09FF]+')
        
        # Bengali font corrections - common OCR/extraction errors
        self.bengali_corrections = {
            # Common character confusion fixes
            'ও': 'ও',  # Normalize Ou vowel
            'া': 'া',   # Normalize AA vowel
            'ি': 'ি',   # Normalize I vowel
            'ু': 'ু',   # Normalize U vowel
            'ূ': 'ূ',   # Normalize UU vowel
            'ে': 'ে',   # Normalize E vowel
            'ো': 'ো',   # Normalize O vowel
        }
        
        # Common Bengali word patterns for validation
        self.common_bengali_words = {
            'এবং', 'যে', 'এই', 'সে', 'না', 'কি', 'আর', 'হয়', 'বলে', 'করে',
            'দিয়ে', 'নিয়ে', 'থেকে', 'পরে', 'আগে', 'সাথে', 'মতো', 'জন্য',
            'কেন', 'কোথায়', 'কখন', 'কিভাবে', 'কী', 'কেউ', 'কিছু', 'সব',
            'অনুপম', 'কল্যাণী', 'শম্ভুনাথ', 'সম্পত্তি', 'বিয়ে', 'বয়স'
        }
    
    def normalize_bengali_text(self, text: str) -> str:
        """Normalize Bengali text to fix common OCR/extraction issues"""
        if not text:
            return text
        
        # Unicode normalization
        text = unicodedata.normalize('NFC', text)
        
        # Apply Bengali character corrections
        for wrong_char, correct_char in self.bengali_corrections.items():
            text = text.replace(wrong_char, correct_char)
        
        # Fix common spacing issues around Bengali punctuation
        text = re.sub(r'\s*।\s*', '। ', text)  # Bengali period
        text = re.sub(r'\s*,\s*', ', ', text)  # Comma
        text = re.sub(r'\s*;\s*', '; ', text)  # Semicolon
        text = re.sub(r'\s*:\s*', ': ', text)  # Colon
        
        # Remove excessive whitespace but preserve paragraph breaks
        text = re.sub(r'[ \t]+', ' ', text)  # Multiple spaces/tabs -> single space
        text = re.sub(r'\n[ \t]+', '\n', text)  # Remove leading whitespace on new lines
        text = re.sub(r'[ \t]+\n', '\n', text)  # Remove trailing whitespace before newlines
        text = re.sub(r'\n{3,}', '\n\n', text)  # Max 2 consecutive newlines
        
        return text.strip()
    
    def validate_bengali_content(self, text: str) -> Dict[str, Any]:
        """Validate and score Bengali text quality"""
        if not text:
            return {"score": 0, "reason": "empty_text", "bengali_ratio": 0}
        
        # Count Bengali characters
        bengali_chars = len(self.bangla_pattern.findall(text))
        total_chars = len([c for c in text if c.isalnum()])
        
        if total_chars == 0:
            return {"score": 0, "reason": "no_alphanumeric", "bengali_ratio": 0}
        
        bengali_ratio = bengali_chars / total_chars
        
        # Check for common Bengali words
        words = text.split()
        common_word_count = sum(1 for word in words if word in self.common_bengali_words)
        common_word_ratio = common_word_count / len(words) if words else 0
        
        # Calculate quality score
        quality_score = 0
        
        # Bengali character ratio (0-40 points)
        if bengali_ratio > 0.7:
            quality_score += 40
        elif bengali_ratio > 0.5:
            quality_score += 30
        elif bengali_ratio > 0.3:
            quality_score += 20
        elif bengali_ratio > 0.1:
            quality_score += 10
        
        # Common words presence (0-30 points)
        quality_score += min(30, common_word_ratio * 100)
        
        # Text length (0-20 points)
        if len(text) > 100:
            quality_score += 20
        elif len(text) > 50:
            quality_score += 15
        elif len(text) > 20:
            quality_score += 10
        
        # Coherence check (0-10 points)
        if not re.search(r'[^\u0980-\u09FF\s\.\,\;\:\!\?\(\)\[\]\"\'0-9a-zA-Z]', text):
            quality_score += 10  # No garbage characters
        
        return {
            "score": quality_score,
            "bengali_ratio": bengali_ratio,
            "common_word_ratio": common_word_ratio,
            "length": len(text),
            "reason": "good_quality" if quality_score > 50 else "low_quality"
        }
    
    def extract_with_enhanced_pymupdf(self, pdf_path: str) -> List[Dict]:
        """Enhanced PyMuPDF extraction with Bengali font handling"""
        if not PYMUPDF_AVAILABLE or fitz is None:
            logger.warning("PyMuPDF not available")
            return []
            
        try:
            doc = fitz.open(pdf_path)
            pages_content = []
            
            logger.info(f"Extracting text from {len(doc)} pages using Enhanced PyMuPDF")
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                
                # Method 1: Try structured text extraction with font info
                text_dict = page.get_text("dict")
                structured_text = self._extract_from_text_dict(text_dict)
                
                # Method 2: Try simple text extraction as fallback
                simple_text = page.get_text()
                
                # Method 3: Try blocks extraction
                blocks_text = page.get_text("blocks")
                blocks_combined = " ".join([block[4] for block in blocks_text if len(block) > 4])
                
                # Choose best extraction
                candidates = [
                    ("structured", structured_text),
                    ("simple", simple_text),
                    ("blocks", blocks_combined)
                ]
                
                best_text = ""
                best_score = 0
                best_method = "none"
                
                for method, text in candidates:
                    if text:
                        normalized_text = self.normalize_bengali_text(text)
                        validation = self.validate_bengali_content(normalized_text)
                        
                        if validation["score"] > best_score:
                            best_text = normalized_text
                            best_score = validation["score"]
                            best_method = method
                
                if best_text and best_score > 30:  # Minimum quality threshold
                    pages_content.append({
                        "page": page_num + 1,
                        "text": best_text,
                        "language": detect_language(best_text),
                        "word_count": len(best_text.split()),
                        "extraction_method": f"pymupdf_{best_method}",
                        "quality_score": best_score,
                        "bengali_ratio": self.validate_bengali_content(best_text)["bengali_ratio"]
                    })
                    
                    logger.debug(f"Page {page_num + 1}: {best_method} method, score={best_score:.1f}")
                else:
                    logger.warning(f"Page {page_num + 1}: Low quality text (score={best_score:.1f})")
                    
            doc.close()
            logger.info(f"Successfully extracted {len(pages_content)} pages with Enhanced PyMuPDF")
            return pages_content
            
        except Exception as e:
            logger.error(f"Enhanced PyMuPDF extraction failed: {e}")
            return []
    
    def _extract_from_text_dict(self, text_dict: Dict) -> str:
        """Extract text from PyMuPDF text dictionary with font consideration"""
        text_parts = []
        
        for block in text_dict.get("blocks", []):
            if "lines" not in block:
                continue
                
            for line in block["lines"]:
                line_text = ""
                for span in line.get("spans", []):
                    span_text = span.get("text", "")
                    font_name = span.get("font", "").lower()
                    
                    # Prioritize spans that likely contain Bengali fonts
                    if any(font_hint in font_name for font_hint in 
                           ["bengali", "bangla", "kalpurush", "solaimanlipi", "nikosh", "siyamrupali"]):
                        line_text = span_text + " " + line_text  # Prioritize Bengali fonts
                    else:
                        line_text += span_text + " "
                        
                text_parts.append(line_text.strip())
                
        return "\n".join(text_parts)
    
    def extract_with_enhanced_pdfplumber(self, pdf_path: str) -> List[Dict]:
        """Enhanced pdfplumber extraction with Bengali text validation"""
        try:
            pages_content = []
            
            with pdfplumber.open(pdf_path) as pdf:
                logger.info(f"Extracting text from {len(pdf.pages)} pages using Enhanced pdfplumber")
                
                for page_num, page in enumerate(pdf.pages):
                    # Try multiple extraction methods
                    methods = []
                    
                    # Method 1: Standard text extraction
                    standard_text = page.extract_text()
                    if standard_text:
                        methods.append(("standard", standard_text))
                    
                    # Method 2: Extract text with layout
                    try:
                        layout_text = page.extract_text(layout=True)
                        if layout_text and layout_text != standard_text:
                            methods.append(("layout", layout_text))
                    except:
                        pass
                    
                    # Method 3: Extract from words and reconstruct
                    try:
                        words = page.extract_words()
                        if words:
                            word_text = " ".join([word["text"] for word in words])
                            methods.append(("words", word_text))
                    except:
                        pass
                    
                    # Choose best extraction
                    best_text = ""
                    best_score = 0
                    best_method = "none"
                    
                    for method, text in methods:
                        if text:
                            normalized_text = self.normalize_bengali_text(text)
                            validation = self.validate_bengali_content(normalized_text)
                            
                            if validation["score"] > best_score:
                                best_text = normalized_text
                                best_score = validation["score"]
                                best_method = method
                    
                    if best_text and best_score > 30:
                        pages_content.append({
                            "page": page_num + 1,
                            "text": best_text,
                            "language": detect_language(best_text),
                            "word_count": len(best_text.split()),
                            "extraction_method": f"pdfplumber_{best_method}",
                            "quality_score": best_score,
                            "bengali_ratio": self.validate_bengali_content(best_text)["bengali_ratio"]
                        })
                        
                        logger.debug(f"Page {page_num + 1}: {best_method} method, score={best_score:.1f}")
                    else:
                        logger.warning(f"Page {page_num + 1}: Low quality text (score={best_score:.1f})")
                        
            logger.info(f"Successfully extracted {len(pages_content)} pages with Enhanced pdfplumber")
            return pages_content
            
        except Exception as e:
            logger.error(f"Enhanced pdfplumber extraction failed: {e}")
            return []
    
    def extract_text(self, pdf_path: str, method: str = "auto") -> List[Dict]:
        """Main extraction method with intelligent method selection"""
        if not Path(pdf_path).exists():
            logger.error(f"PDF file not found: {pdf_path}")
            return []
            
        logger.info(f"Extracting text from {pdf_path} using {method} method...")
        
        if method == "auto":
            # Try both methods and choose the better result
            logger.info("Auto-mode: trying both extraction methods...")
            
            pymupdf_result = self.extract_with_enhanced_pymupdf(pdf_path)
            pdfplumber_result = self.extract_with_enhanced_pdfplumber(pdf_path)
            
            # Compare results and choose better one
            pymupdf_score = sum(page.get("quality_score", 0) for page in pymupdf_result)
            pdfplumber_score = sum(page.get("quality_score", 0) for page in pdfplumber_result)
            
            if pymupdf_score >= pdfplumber_score and pymupdf_result:
                logger.info(f"Chose PyMuPDF (score: {pymupdf_score:.1f} vs {pdfplumber_score:.1f})")
                return pymupdf_result
            elif pdfplumber_result:
                logger.info(f"Chose pdfplumber (score: {pdfplumber_score:.1f} vs {pymupdf_score:.1f})")
                return pdfplumber_result
            else:
                logger.warning("Both methods failed, returning empty result")
                return []
                
        elif method == "pymupdf":
            return self.extract_with_enhanced_pymupdf(pdf_path)
        else:
            return self.extract_with_enhanced_pdfplumber(pdf_path)

    async def process_pdf(self, pdf_path: str) -> List[Dict]:
        """Asynchronously process a PDF to extract and clean text."""
        logger.info(f"Processing PDF: {pdf_path}")
        return self.extract_text(pdf_path, method="auto")
        
    def save_extracted_text(self, pages_content: List[Dict], save_path: str):
        """Save extracted text to a file for inspection with quality metrics"""
        try:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            
            total_pages = len(pages_content)
            total_score = sum(page.get("quality_score", 0) for page in pages_content)
            avg_bengali_ratio = sum(page.get("bengali_ratio", 0) for page in pages_content) / total_pages if total_pages > 0 else 0
            
            with open(save_path, 'w', encoding='utf-8') as f:
                # Write summary
                f.write(f"=== EXTRACTION SUMMARY ===\n")
                f.write(f"Total Pages: {total_pages}\n")
                f.write(f"Average Quality Score: {total_score/total_pages:.1f}/100\n")
                f.write(f"Average Bengali Ratio: {avg_bengali_ratio:.2%}\n")
                f.write(f"Extraction Methods Used: {', '.join(set(page.get('extraction_method', 'unknown') for page in pages_content))}\n")
                f.write(f"{'='*60}\n\n")
                
                # Write page content
                for page in pages_content:
                    f.write(f"=== Page {page['page']} ===\n")
                    f.write(f"Language: {page['language']}\n")
                    f.write(f"Method: {page.get('extraction_method', 'unknown')}\n")
                    f.write(f"Quality Score: {page.get('quality_score', 0):.1f}/100\n")
                    f.write(f"Bengali Ratio: {page.get('bengali_ratio', 0):.2%}\n")
                    f.write(f"Word Count: {page['word_count']}\n")
                    f.write(f"{'-'*50}\n")
                    f.write(page['text'])
                    f.write(f"\n{'='*50}\n\n")
                    
            logger.info(f"Enhanced extracted text saved to: {save_path}")
            
        except Exception as e:
            logger.error(f"Failed to save extracted text: {e}")
    
    def get_extraction_stats(self, pages_content: List[Dict]) -> Dict:
        """Get detailed extraction statistics"""
        if not pages_content:
            return {"status": "no_content"}
        
        total_pages = len(pages_content)
        quality_scores = [page.get("quality_score", 0) for page in pages_content]
        bengali_ratios = [page.get("bengali_ratio", 0) for page in pages_content]
        methods_used = [page.get("extraction_method", "unknown") for page in pages_content]
        
        return {
            "total_pages": total_pages,
            "avg_quality_score": sum(quality_scores) / total_pages,
            "min_quality_score": min(quality_scores),
            "max_quality_score": max(quality_scores),
            "avg_bengali_ratio": sum(bengali_ratios) / total_pages,
            "high_quality_pages": len([s for s in quality_scores if s > 70]),
            "medium_quality_pages": len([s for s in quality_scores if 40 <= s <= 70]),
            "low_quality_pages": len([s for s in quality_scores if s < 40]),
            "methods_used": list(set(methods_used)),
            "total_words": sum(page.get("word_count", 0) for page in pages_content),
            "languages_detected": list(set(page.get("language", "unknown") for page in pages_content))
        }


# Backward compatibility - create an alias for the old class name
class PDFProcessor(BengaliPDFProcessor):
    """Alias for backward compatibility"""
    pass
