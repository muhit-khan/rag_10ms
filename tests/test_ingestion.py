import pytest
import asyncio
from pathlib import Path

from src.pdf_processor import PDFProcessor
from src.text_cleaner import BengaliTextCleaner
from src.chunking import DocumentChunker

class TestIngestion:
    """Test PDF ingestion and preprocessing"""
    
    @pytest.fixture
    def pdf_processor(self):
        return PDFProcessor()
    
    @pytest.fixture
    def text_cleaner(self):
        return BengaliTextCleaner()
    
    @pytest.fixture
    def chunker(self):
        return DocumentChunker(chunk_size=256, overlap=25)
    
    def test_pdf_processor_initialization(self, pdf_processor):
        """Test PDF processor initialization"""
        assert pdf_processor is not None
        assert hasattr(pdf_processor, 'bangla_pattern')
    
    def test_text_cleaner_initialization(self, text_cleaner):
        """Test text cleaner initialization"""
        assert text_cleaner is not None
        assert hasattr(text_cleaner, 'bangla_sentence_delims')
    
    def test_bangla_text_cleaning(self, text_cleaner):
        """Test Bengali text cleaning"""
        test_text = "এটি   একটি   পরীক্ষা।।   দুইটি বাক্য।"
        cleaned = text_cleaner.clean_text(test_text)
        
        assert cleaned is not None
        assert "।।" not in cleaned  # Double punctuation should be fixed
        assert cleaned.count(" ") < test_text.count(" ")  # Excessive spaces removed
    
    def test_sentence_segmentation(self, text_cleaner):
        """Test Bengali sentence segmentation"""
        test_text = "প্রথম বাক্য। দ্বিতীয় বাক্য! তৃতীয় বাক্য?"
        sentences = text_cleaner.segment_sentences(test_text)
        
        assert len(sentences) == 3
        assert all(sentence.strip() for sentence in sentences)
    
    def test_chunking_by_sentences(self, chunker):
        """Test sentence-based chunking"""
        sentences = [
            "এটি প্রথম বাক্য।",
            "এটি দ্বিতীয় বাক্য।", 
            "এটি তৃতীয় বাক্য।"
        ]
        
        metadata = {"page": 1, "language": "bn"}
        chunks = chunker.chunk_by_sentences(sentences, metadata)
        
        assert len(chunks) >= 1
        assert all(chunk["metadata"] == metadata for chunk in chunks)
        assert all(chunk["word_count"] > 0 for chunk in chunks)
    
    def test_chunking_stats(self, chunker):
        """Test chunking statistics"""
        chunks = [
            {"word_count": 50, "metadata": {"language": "bn", "page": 1}},
            {"word_count": 75, "metadata": {"language": "bn", "page": 2}},
        ]
        
        stats = chunker.get_chunking_stats(chunks)
        
        assert stats["total_chunks"] == 2
        assert stats["total_words"] == 125
        assert stats["avg_chunk_size"] == 62.5
        assert "bn" in stats["languages"]
