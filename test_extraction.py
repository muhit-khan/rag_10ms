#!/usr/bin/env python3
"""
Test script to verify enhanced Bengali text extraction
"""
import sys
import os
import asyncio
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

from src.pdf_processor import BengaliPDFProcessor
from src.text_cleaner import BengaliTextCleaner
from utils import setup_logging
import logging

async def test_bengali_extraction():
    """Test the enhanced Bengali PDF extraction"""
    
    # Setup logging
    setup_logging("INFO")
    logger = logging.getLogger(__name__)
    
    # Initialize processors
    pdf_processor = BengaliPDFProcessor()
    text_cleaner = BengaliTextCleaner()
    
    # Test PDF path
    pdf_path = "data/raw/hsc26_bangla_1st_paper.pdf"
    
    if not Path(pdf_path).exists():
        logger.error(f"PDF not found: {pdf_path}")
        return
    
    logger.info("=" * 60)
    logger.info("TESTING ENHANCED BENGALI TEXT EXTRACTION")
    logger.info("=" * 60)
    
    # Extract text
    logger.info("Extracting text from PDF...")
    extracted_pages = await pdf_processor.process_pdf(pdf_path)
    
    if not extracted_pages:
        logger.error("No pages extracted!")
        return
    
    # Get extraction statistics
    stats = pdf_processor.get_extraction_stats(extracted_pages)
    
    logger.info("\n" + "=" * 40)
    logger.info("EXTRACTION STATISTICS")
    logger.info("=" * 40)
    logger.info(f"Total Pages: {stats['total_pages']}")
    logger.info(f"Average Quality Score: {stats['avg_quality_score']:.1f}/100")
    logger.info(f"Average Bengali Ratio: {stats['avg_bengali_ratio']:.1%}")
    logger.info(f"High Quality Pages: {stats['high_quality_pages']}")
    logger.info(f"Medium Quality Pages: {stats['medium_quality_pages']}")
    logger.info(f"Low Quality Pages: {stats['low_quality_pages']}")
    logger.info(f"Total Words: {stats['total_words']}")
    logger.info(f"Methods Used: {', '.join(stats['methods_used'])}")
    logger.info(f"Languages Detected: {', '.join(stats['languages_detected'])}")
    
    # Show sample pages
    logger.info("\n" + "=" * 40)
    logger.info("SAMPLE EXTRACTED CONTENT")
    logger.info("=" * 40)
    
    for i, page in enumerate(extracted_pages[:3]):  # Show first 3 pages
        logger.info(f"\n--- PAGE {page['page']} ---")
        logger.info(f"Method: {page.get('extraction_method', 'unknown')}")
        logger.info(f"Quality: {page.get('quality_score', 0):.1f}/100")
        logger.info(f"Bengali Ratio: {page.get('bengali_ratio', 0):.1%}")
        logger.info(f"Words: {page['word_count']}")
        logger.info(f"Content Preview:")
        logger.info(f"{page['text'][:200]}...")
        
        # Test text cleaning
        cleaned = text_cleaner.clean_text(page['text'])
        validation = text_cleaner.validate_bengali_content(cleaned)
        
        logger.info(f"After Cleaning - Quality: {validation['quality_score']:.1f}/100")
        logger.info(f"Content Words Found: {validation['content_words']}")
        logger.info(f"Valid Bengali Words: {validation['valid_bengali_words']}")
    
    # Save enhanced extraction
    save_path = "data/processed/enhanced_extracted_text.txt"
    pdf_processor.save_extracted_text(extracted_pages, save_path)
    logger.info(f"\nEnhanced extraction saved to: {save_path}")
    
    # Test specific queries
    logger.info("\n" + "=" * 40)
    logger.info("TESTING CONTENT SEARCH")
    logger.info("=" * 40)
    
    search_terms = ['অনুপম', 'কল্যাণী', 'বিয়ে', 'বয়স', 'শম্ভুনাথ']
    
    for term in search_terms:
        found_pages = []
        for page in extracted_pages:
            if term in page['text']:
                found_pages.append(page['page'])
        
        if found_pages:
            logger.info(f"'{term}' found in pages: {', '.join(map(str, found_pages))}")
        else:
            logger.warning(f"'{term}' not found in any page")
    
    logger.info("\n" + "=" * 60)
    logger.info("EXTRACTION TEST COMPLETED")
    logger.info("=" * 60)

if __name__ == "__main__":
    asyncio.run(test_bengali_extraction())
