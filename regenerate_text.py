#!/usr/bin/env python3
"""
Force regenerate enhanced Bengali text extraction
"""
import sys
import os
import asyncio
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

from src.rag_pipeline import RAGPipeline
from utils import setup_logging
from config import settings
import logging

async def regenerate_extraction():
    """Force regenerate the enhanced Bengali text extraction"""
    
    setup_logging("INFO")
    logger = logging.getLogger(__name__)
    
    logger.info("🔄 REGENERATING ENHANCED BENGALI TEXT EXTRACTION")
    logger.info("=" * 60)
    
    # Initialize RAG pipeline
    rag_pipeline = RAGPipeline()
    
    # Get PDF path
    pdf_path = os.path.join(settings.RAW_DATA_DIR, "hsc26_bangla_1st_paper.pdf")
    
    if not Path(pdf_path).exists():
        logger.error(f"❌ PDF not found: {pdf_path}")
        return
    
    logger.info(f"📄 Processing: {pdf_path}")
    
    # Force extract and save enhanced text
    try:
        # Extract with enhanced processor
        pages_data = await rag_pipeline.pdf_processor.process_pdf(pdf_path)
        
        if not pages_data:
            logger.error("❌ No pages extracted!")
            return
        
        # Save enhanced extraction
        enhanced_path = os.path.join(settings.PROCESSED_DATA_DIR, "enhanced_extracted_text.txt")
        rag_pipeline.pdf_processor.save_extracted_text(pages_data, enhanced_path)
        
        # Get and display statistics
        stats = rag_pipeline.pdf_processor.get_extraction_stats(pages_data)
        
        logger.info("📊 ENHANCED EXTRACTION STATISTICS:")
        logger.info(f"   📄 Total Pages: {stats['total_pages']}")
        logger.info(f"   ⭐ Average Quality Score: {stats['avg_quality_score']:.1f}/100")
        logger.info(f"   🔤 Average Bengali Ratio: {stats['avg_bengali_ratio']:.1%}")
        logger.info(f"   ✅ High Quality Pages: {stats['high_quality_pages']}")
        logger.info(f"   ⚠️ Medium Quality Pages: {stats['medium_quality_pages']}")
        logger.info(f"   ❌ Low Quality Pages: {stats['low_quality_pages']}")
        logger.info(f"   📝 Total Words: {stats['total_words']:,}")
        logger.info(f"   🔧 Methods Used: {', '.join(stats['methods_used'])}")
        logger.info(f"   🌐 Languages: {', '.join(stats['languages_detected'])}")
        
        logger.info(f"\n✅ Enhanced text saved to: {enhanced_path}")
        
        # Also save a simple version for comparison
        simple_path = os.path.join(settings.PROCESSED_DATA_DIR, "extracted_text.txt")
        with open(simple_path, 'w', encoding='utf-8') as f:
            for page in pages_data:
                f.write(f"=== Page {page['page']} ===\n")
                f.write(page['text'])
                f.write(f"\n{'='*50}\n\n")
        
        logger.info(f"💾 Simple version saved to: {simple_path}")
        
        # Show sample content
        logger.info("\n📖 SAMPLE ENHANCED CONTENT:")
        logger.info("-" * 40)
        for i, page in enumerate(pages_data[:2]):  # Show first 2 pages
            logger.info(f"Page {page['page']} ({page.get('quality_score', 0):.1f} quality):")
            logger.info(f"{page['text'][:300]}...")
            logger.info("-" * 40)
        
        # Test search for key terms
        logger.info("\n🔍 CONTENT VERIFICATION:")
        search_terms = ['অনুপম', 'কল্যাণী', 'বিয়ে', 'বয়স', 'শম্ভুনাথ']
        
        for term in search_terms:
            found_pages = [page['page'] for page in pages_data if term in page['text']]
            if found_pages:
                logger.info(f"   ✅ '{term}' found in {len(found_pages)} pages: {', '.join(map(str, found_pages[:5]))}")
            else:
                logger.info(f"   ❌ '{term}' not found")
        
    except Exception as e:
        logger.error(f"❌ Error during extraction: {e}")
        return
    
    logger.info("\n🎉 ENHANCED EXTRACTION COMPLETED SUCCESSFULLY!")
    logger.info("=" * 60)

if __name__ == "__main__":
    asyncio.run(regenerate_extraction())
