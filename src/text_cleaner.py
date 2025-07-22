import re
from typing import List, Dict
import logging

from utils.helpers import clean_whitespace, detect_language, is_valid_text_chunk

logger = logging.getLogger(__name__)

class BengaliTextCleaner:
    """Clean and preprocess Bengali and English text"""
    
    def __init__(self):
        # Bengali-specific patterns
        self.bangla_sentence_delims = ['।', '!', '?', '\n\n']
        
        # OCR error patterns for Bengali
        self.ocr_corrections = {
            # Common OCR mistakes in Bengali
            'ৈ': 'ে',  # Fix vowel marks
            'ৌ': 'ো',
            '।।': '।',  # Duplicate sentence endings
            '??': '?',
            '!!': '!',
        }
        
        # Noise patterns to remove
        self.noise_patterns = [
            r'\s+',  # Multiple whitespace
            r'[^\u0980-\u09FF\u0020-\u007E\u00A0-\u00FF\n\r।!?.,;:]',  # Invalid chars
            r'(?<=[।!?])\s*(?=[।!?])',  # Duplicate punctuation spaces
        ]
    
    def fix_ocr_errors(self, text: str) -> str:
        """Fix common OCR errors in Bengali text"""
        for wrong, correct in self.ocr_corrections.items():
            text = text.replace(wrong, correct)
        return text
    
    def normalize_punctuation(self, text: str) -> str:
        """Normalize Bengali punctuation"""
        # Fix multiple sentence delimiters
        text = re.sub(r'[।]{2,}', '।', text)
        text = re.sub(r'[!]{2,}', '!', text)
        text = re.sub(r'[?]{2,}', '?', text)
        
        # Ensure proper spacing after punctuation
        text = re.sub(r'([।!?])([^\s])', r'\1 \2', text)
        
        return text
    
    def clean_text(self, text: str) -> str:
        """Main text cleaning method"""
        if not text:
            return ""
        
        # Basic cleaning
        text = clean_whitespace(text)
        
        # Fix OCR errors
        text = self.fix_ocr_errors(text)
        
        # Normalize punctuation
        text = self.normalize_punctuation(text)
        
        # Remove excessive newlines but preserve paragraph breaks
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def segment_sentences(self, text: str) -> List[str]:
        """Split text into sentences (Bengali-aware)"""
        if not text:
            return []
        
        sentences = []
        current_sentence = ""
        
        i = 0
        while i < len(text):
            char = text[i]
            current_sentence += char
            
            # Check if we hit a sentence delimiter
            if char in self.bangla_sentence_delims:
                # Clean and add sentence if it's valid
                sentence = current_sentence.strip()
                if is_valid_text_chunk(sentence, min_length=5):
                    sentences.append(sentence)
                current_sentence = ""
            
            i += 1
        
        # Add remaining text if valid
        if current_sentence.strip() and is_valid_text_chunk(current_sentence.strip(), min_length=5):
            sentences.append(current_sentence.strip())
        
        return sentences
    
    def segment_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs"""
        if not text:
            return []
        
        # Split by double newlines or multiple sentence delimiters
        paragraphs = re.split(r'\n\n+|[।!?]{2,}', text)
        
        valid_paragraphs = []
        for para in paragraphs:
            para = para.strip()
            if is_valid_text_chunk(para, min_length=20):
                valid_paragraphs.append(para)
        
        return valid_paragraphs
    
    def preprocess_page(self, page_data: Dict) -> Dict:
        """Preprocess a single page of text"""
        try:
            original_text = page_data.get("text", "")
            
            # Clean the text
            cleaned_text = self.clean_text(original_text)
            
            if not cleaned_text:
                logger.warning(f"Page {page_data.get('page', 'unknown')} became empty after cleaning")
                return None
            
            # Segment into sentences and paragraphs
            sentences = self.segment_sentences(cleaned_text)
            paragraphs = self.segment_paragraphs(cleaned_text)
            
            return {
                "page": page_data.get("page"),
                "original_text": original_text,
                "cleaned_text": cleaned_text,
                "sentences": sentences,
                "paragraphs": paragraphs,
                "language": detect_language(cleaned_text),
                "word_count": len(cleaned_text.split()),
                "sentence_count": len(sentences),
                "paragraph_count": len(paragraphs)
            }
            
        except Exception as e:
            logger.error(f"Error preprocessing page {page_data.get('page', 'unknown')}: {e}")
            return None
    
    def preprocess_document(self, pages_data: List[Dict]) -> List[Dict]:
        """Preprocess all pages in a document"""
        if not pages_data:
            logger.warning("No pages to preprocess")
            return []
        
        logger.info(f"Preprocessing {len(pages_data)} pages...")
        
        preprocessed_pages = []
        for page_data in pages_data:
            processed = self.preprocess_page(page_data)
            if processed:
                preprocessed_pages.append(processed)
        
        logger.info(f"Successfully preprocessed {len(preprocessed_pages)} pages")
        
        return preprocessed_pages
