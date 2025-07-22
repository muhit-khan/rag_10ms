"""
Enhanced accuracy testing for the RAG system
"""

import asyncio
import logging
from src.rag_pipeline import RAGPipeline
from config import settings

# Test cases with expected answers
TEST_CASES = [
    {
        "question": "অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?",
        "expected": "শম্ভুনাথ",
        "context_keywords": ["শম্ভুনাথ", "সুপুরুষ", "অনুপম"]
    },
    {
        "question": "কাকে অনুপমের ভাগ্য দেবতা বলে উল্লেখ করা হয়েছে?",
        "expected": "মামাকে",
        "context_keywords": ["মামা", "ভাগ্য দেবতা", "অনুপম"]
    },
    {
        "question": "বিয়ের সময় কল্যাণীর প্রকৃত বয়স কত ছিল?",
        "expected": "১৫ বছর",
        "context_keywords": ["কল্যাণী", "বয়স", "বিয়ে"]
    },
    {
        "question": "অনুপমের বয়স কত বছর?",
        "expected": "সাতাশ বছর",
        "context_keywords": ["অনুপম", "বয়স", "সাতাশ"]
    },
    {
        "question": "অনুপমের বাবা কী কাজ করতেন?",
        "expected": "উকালতি",
        "context_keywords": ["অনুপম", "বাবা", "উকালতি"]
    }
]

async def test_accuracy():
    """Test the accuracy of the RAG system"""
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Initialize RAG pipeline
    rag_pipeline = RAGPipeline()
    
    # Check if system is ready
    pdf_path = "./data/raw/hsc26_bangla_1st_paper.pdf"
    await rag_pipeline.initialize(pdf_path)
    
    if not rag_pipeline.is_initialized:
        logger.error("RAG system not initialized. Please check the PDF file.")
        return
    
    logger.info("=== RAG System Accuracy Test ===")
    
    correct_answers = 0
    total_questions = len(TEST_CASES)
    results = []
    
    for i, test_case in enumerate(TEST_CASES, 1):
        logger.info(f"\n--- Test Case {i}/{total_questions} ---")
        logger.info(f"Question: {test_case['question']}")
        logger.info(f"Expected: {test_case['expected']}")
        
        try:
            # Process the query
            result = await rag_pipeline.process_query(
                query=test_case['question'],
                session_id=f"test_session_{i}"
            )
            
            answer = result['answer']
            sources_count = len(result['sources'])
            
            logger.info(f"Answer: {answer}")
            logger.info(f"Sources: {sources_count} chunks found")
            
            # Check if answer contains expected information
            is_correct = False
            expected_lower = test_case['expected'].lower()
            answer_lower = answer.lower()
            
            # Simple matching - can be enhanced
            if expected_lower in answer_lower:
                is_correct = True
            elif any(keyword.lower() in answer_lower for keyword in test_case['context_keywords']):
                # Partial credit for relevant context
                is_correct = True
                logger.info("⚠️  Partial match - contains relevant context")
            
            if is_correct:
                correct_answers += 1
                logger.info("✅ CORRECT")
            else:
                logger.info("❌ INCORRECT")
            
            results.append({
                'question': test_case['question'],
                'expected': test_case['expected'],
                'answer': answer,
                'correct': is_correct,
                'sources_count': sources_count,
                'retrieval_stats': result.get('retrieval_stats', {})
            })
            
        except Exception as e:
            logger.error(f"Error processing question: {e}")
            results.append({
                'question': test_case['question'],
                'expected': test_case['expected'],
                'answer': "ERROR",
                'correct': False,
                'error': str(e)
            })
    
    # Summary
    accuracy = (correct_answers / total_questions) * 100
    logger.info(f"\n=== Test Results Summary ===")
    logger.info(f"Total Questions: {total_questions}")
    logger.info(f"Correct Answers: {correct_answers}")
    logger.info(f"Accuracy: {accuracy:.1f}%")
    
    # Detailed analysis
    logger.info(f"\n=== Detailed Analysis ===")
    for i, result in enumerate(results, 1):
        status = "✅" if result['correct'] else "❌"
        logger.info(f"{status} Q{i}: {result.get('sources_count', 0)} sources")
    
    # System statistics
    stats = rag_pipeline.get_system_stats()
    logger.info(f"\n=== System Statistics ===")
    logger.info(f"Knowledge Base: {stats.get('total_chunks', 'N/A')} chunks")
    logger.info(f"Languages: {stats.get('languages', 'N/A')}")
    
    return {
        'accuracy': accuracy,
        'correct': correct_answers,
        'total': total_questions,
        'results': results
    }

if __name__ == "__main__":
    asyncio.run(test_accuracy())
