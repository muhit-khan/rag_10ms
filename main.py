import asyncio
import logging
import os
from pathlib import Path

from src.rag_pipeline import RAGPipeline
from config import settings
from utils import setup_logging

async def main():
    """Main entry point for the RAG system"""
    
    # Setup logging
    setup_logging("INFO")
    logger = logging.getLogger(__name__)
    
    logger.info("=== Multilingual RAG System ===")
    logger.info("Initializing system...")
    
    try:
        # Initialize RAG pipeline
        rag_pipeline = RAGPipeline()
        
        # Check for PDF in data/raw directory
        pdf_path = Path(settings.RAW_DATA_DIR) / "hsc26_bangla_1st_paper.pdf"
        
        if not pdf_path.exists():
            logger.warning(f"PDF not found at: {pdf_path}")
            logger.info("Please place the HSC26 Bangla 1st paper PDF in the data/raw/ directory")
            
            # Create sample instructions
            instructions_path = Path(settings.RAW_DATA_DIR) / "README.md"
            instructions_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(instructions_path, 'w', encoding='utf-8') as f:
                f.write("""# Data Directory

Place your PDF files here for processing.

## Expected file:
- `hsc26_bangla_1st_paper.pdf` - The HSC26 Bangla 1st paper textbook

## Usage:
1. Copy your PDF file to this directory
2. Run the main script: `python main.py`
3. The system will automatically process the PDF and create vector embeddings
""")
            
            logger.info(f"Created instructions at: {instructions_path}")
            return
        
        # Initialize with the PDF
        await rag_pipeline.initialize(str(pdf_path))
        
        if not rag_pipeline.is_initialized:
            logger.error("Failed to initialize RAG pipeline")
            return
        
        # Get system stats
        stats = rag_pipeline.get_system_stats()
        logger.info("System initialized successfully!")
        logger.info(f"Knowledge base: {stats['total_chunks']} chunks")
        logger.info(f"Languages: {stats['languages']}")
        
        # Interactive demo
        logger.info("\n=== Interactive Demo ===")
        print("Sample questions you can ask:")
        print("1. অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?")
        print("2. কাকে অনুপমের ভাগ্য দেবতা বলে উল্লেখ করা হয়েছে?")
        print("3. বিয়ের সময় কল্যাণীর প্রকৃত বয়স কত ছিল?")
        
        # Simple CLI interface
        session_id = "demo_session"
        
        while True:
            try:
                query = input("\nআপনার প্রশ্ন (বা 'exit' টাইপ করুন): ")
                
                if query.lower() in ['exit', 'quit', 'বন্ধ', 'বের']:
                    logger.info("Thank you for using the RAG system!")
                    break
                
                if query.strip():
                    print("\nProcessing your question...")
                    
                    result = await rag_pipeline.process_query(
                        query=query,
                        session_id=session_id
                    )
                    
                    print(f"\nAnswer: {result['answer']}")
                    print(f"Language: {result['language_detected']}")
                    print(f"Sources found: {len(result['sources'])}")
                    
                    if result.get('sources'):
                        print("\nSource pages:", [s['page'] for s in result['sources']])
                
            except KeyboardInterrupt:
                logger.info("\nExiting...")
                break
            except Exception as e:
                logger.error(f"Error processing query: {e}")
        
    except Exception as e:
        logger.error(f"System initialization failed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
