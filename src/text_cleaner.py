import re
from typing import List, Dict
import logging
import unicodedata

from utils.helpers import clean_whitespace, detect_language, is_valid_text_chunk

logger = logging.getLogger(__name__)

class BengaliTextCleaner:
    """Enhanced Bengali and English text cleaner with OCR error correction"""
    
    def __init__(self):
        # Bengali-specific patterns
        self.bangla_sentence_delims = ['।', '!', '?', '\n\n']
        
        # Enhanced OCR error corrections for Bengali
        self.ocr_corrections = {
            # Common OCR mistakes in Bengali characters
            'ৈ': 'ে',  # Fix vowel marks
            'ৌ': 'ো',
            '।।': '।',  # Duplicate sentence endings
            '??': '?',
            '!!': '!',
            
            # Character confusion fixes
            'অা': 'আ',  # Combine A + AA = AA
            'েি': 'ৈ',  # e + i = ai
            'েো': 'ৌ',  # e + o = ou
            
            # Common word-level corrections
            'অনুপমম': 'অনুপম',
            'কল্যানী': 'কল্যাণী',
            'কল্যানি': 'কল্যাণী',
            'সম্পত্তি': 'সম্পত্তি',
            'সমপত্তি': 'সম্পত্তি',
            'বিযে': 'বিয়ে',
            'বিয়ের': 'বিয়ে',
            'শমভুনাথ': 'শম্ভুনাথ',
            'সমভুনাথ': 'শম্ভুনাথ',
        }
        
        # Enhanced noise patterns to remove
        self.noise_patterns = [
            r'[^\u0980-\u09FF\u0020-\u007E\u00A0-\u00FF\n\r।!?.,;:()\[\]"\'-]',  # Keep only valid chars
            r'(?<=[।!?])\s*(?=[।!?])',  # Duplicate punctuation spaces
            r'\s*([।!?])\s*\1+\s*',  # Multiple same punctuation
            r'^\s*[\-_=~]+\s*$',  # Lines with only symbols
            r'Page\s*\d+',  # Page numbers
            r'\d+\s*।',  # Numbers before Bengali period (likely page numbers)
        ]
        
        # Bengali word validation patterns
        self.valid_bengali_word_pattern = re.compile(r'^[\u0980-\u09FF]+$')
        
        # Common Bengali stopwords and important words for content validation
        self.bengali_content_words = {
            'অনুপম', 'কল্যাণী', 'শম্ভুনাথ', 'বিয়ে', 'সম্পত্তি', 'বয়স', 'পিতা', 'মা', 'ভাই', 'বোন',
            'গল্প', 'উপন্যাস', 'চরিত্র', 'ঘটনা', 'সময়', 'বছর', 'দিন', 'রাত', 'সকাল', 'বিকাল',
            'বলা', 'বলে', 'বলেন', 'বললেন', 'জানা', 'জানে', 'জানেন', 'জানালেন', 'হওয়া', 'হয়', 'হল',
            'করা', 'করে', 'করেন', 'করলেন', 'দেওয়া', 'দেয়', 'দেন', 'দিলেন', 'নেওয়া', 'নেয়', 'নিলেন'
        }
    
    def fix_unicode_issues(self, text: str) -> str:
        """Fix Unicode normalization issues"""
        # Normalize Unicode to canonical form
        text = unicodedata.normalize('NFC', text)
        
        # Fix common Unicode issues in Bengali
        text = text.replace('\u200c', '')  # Remove zero-width non-joiner
        text = text.replace('\u200d', '')  # Remove zero-width joiner if misplaced
        text = text.replace('\ufeff', '')  # Remove BOM
        
        return text
    
    def fix_ocr_errors(self, text: str) -> str:
        """Fix common OCR errors in Bengali text"""
        # Apply character-level corrections
        for wrong, correct in self.ocr_corrections.items():
            text = text.replace(wrong, correct)
        
        # Fix common spacing issues around Bengali characters
        text = re.sub(r'([।!?])\s*([।!?])', r'\1 \2', text)  # Space between different punct
        text = re.sub(r'([।!?])\1+', r'\1', text)  # Remove duplicate punctuation
        
        return text
    
    def normalize_punctuation(self, text: str) -> str:
        """Enhanced Bengali punctuation normalization"""
        # Fix multiple sentence delimiters
        text = re.sub(r'[।]{2,}', '।', text)
        text = re.sub(r'[!]{2,}', '!', text)
        text = re.sub(r'[?]{2,}', '?', text)
        
        # Ensure proper spacing after punctuation
        text = re.sub(r'([।!?,;:])([^\s\n])', r'\1 \2', text)
        
        # Fix spacing before punctuation (remove extra spaces)
        text = re.sub(r'\s+([।!?,;:])', r'\1', text)
        
        # Handle quotation marks
        text = re.sub(r'[""]', '"', text)
        text = re.sub(r'['']', "'", text)
        
        return text
    
    def remove_noise(self, text: str) -> str:
        """Remove noise patterns from text"""
        original_length = len(text)
        
        for pattern in self.noise_patterns:
            text = re.sub(pattern, ' ', text)
        
        # Additional noise removal
        text = re.sub(r'^[\s\-_=~]*$', '', text, flags=re.MULTILINE)  # Empty decorative lines
        text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)  # Max 2 newlines
        text = re.sub(r'[ \t]+', ' ', text)  # Multiple spaces to single space
        
        cleaned_length = len(text)
        if original_length > 0 and (original_length - cleaned_length) / original_length > 0.5:
            logger.warning(f"Removed {(original_length - cleaned_length) / original_length:.1%} of text as noise")
        
        return text.strip()
    
    def validate_bengali_content(self, text: str) -> Dict[str, float]:
        """Validate and score Bengali content quality"""
        if not text:
            return {"quality_score": 0.0, "bengali_ratio": 0.0, "content_words": 0}
        
        words = text.split()
        if not words:
            return {"quality_score": 0.0, "bengali_ratio": 0.0, "content_words": 0}
        
        # Count Bengali characters
        bengali_chars = len(re.findall(r'[\u0980-\u09FF]', text))
        total_chars = len(text.replace(' ', '').replace('\n', ''))
        bengali_ratio = bengali_chars / total_chars if total_chars > 0 else 0
        
        # Count content words
        content_words = sum(1 for word in words if word in self.bengali_content_words)
        content_word_ratio = content_words / len(words) if words else 0
        
        # Count valid Bengali words (containing Bengali characters)
        valid_bengali_words = sum(1 for word in words if re.search(r'[\u0980-\u09FF]', word))
        valid_word_ratio = valid_bengali_words / len(words) if words else 0
        
        # Calculate quality score (0-100)
        quality_score = (
            bengali_ratio * 40 +  # Bengali character ratio
            content_word_ratio * 30 +  # Important content words
            valid_word_ratio * 30  # Valid Bengali words
        ) * 100
        
        return {
            "quality_score": min(100.0, quality_score),
            "bengali_ratio": bengali_ratio,
            "content_words": content_words,
            "valid_bengali_words": valid_bengali_words,
            "total_words": len(words)
        }
    
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
