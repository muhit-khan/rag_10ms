#!/usr/bin/env python3
"""
Simple test for Bengali PDF extraction
"""
import os
import asyncio
from src.pdf_processor import BengaliPDFProcessor

async def main():
    processor = BengaliPDFProcessor()
    
    # Extract from the PDF
    pdf_path = "data/raw/hsc26_bangla_1st_paper.pdf"
    
    print("🔄 Starting Bengali text extraction...")
    pages = await processor.process_pdf(pdf_path)
    
    print(f"✅ Extracted {len(pages)} pages")
    
    # Save to file
    output_path = "data/processed/enhanced_extracted_text.txt"
    processor.save_extracted_text(pages, output_path)
    
    print(f"💾 Saved to: {output_path}")
    
    # Show stats
    if pages:
        stats = processor.get_extraction_stats(pages)
        print(f"📊 Stats - Avg Quality: {stats['avg_quality_score']:.1f}")
        print(f"📊 Bengali Ratio: {stats['avg_bengali_ratio']:.1%}")
        
        # Show sample
        print(f"📖 Sample from Page 1:")
        print(f"{pages[0]['text'][:200]}...")

if __name__ == "__main__":
    asyncio.run(main())
