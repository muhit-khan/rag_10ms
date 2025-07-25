"""
Test cases for Bengali queries as specified in the technical assessment.

This module contains the specific test cases mentioned in the AI Engineer
technical assessment requirements.
"""
import pytest
from services.rag_service import RAGService
from services.eval_service import EvalService


class TestBengaliQueries:
    """Test class for Bengali query evaluation."""
    
    @pytest.fixture
    def rag_service(self):
        """Create a RAG service instance for testing."""
        return RAGService("test_user")
    
    @pytest.fixture
    def eval_service(self):
        """Create an evaluation service instance for testing."""
        return EvalService("test_user")
    
    def test_bengali_query_1(self, rag_service):
        """
        Test Case 1: অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?
        Expected Answer: শুম্ভুনাথ
        """
        query = "অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?"
        expected_answer = "শুম্ভুনাথ"
        
        answer, docs = rag_service.generate_answer(query)
        
        # Basic assertions
        assert answer is not None
        assert isinstance(answer, str)
        assert len(answer.strip()) > 0
        
        # Check if the expected answer is mentioned in the response
        assert expected_answer in answer or "শুম্ভুনাথ" in answer
        
        # Check if sources were retrieved
        assert docs is not None
        assert "documents" in docs
        assert len(docs["documents"][0]) > 0
    
    def test_bengali_query_2(self, rag_service):
        """
        Test Case 2: কাকে অনুপমের ভাগ্য দেবতা বলে উল্লেখ করা হয়েছে?
        Expected Answer: মামাকে
        """
        query = "কাকে অনুপমের ভাগ্য দেবতা বলে উল্লেখ করা হয়েছে?"
        expected_answer = "মামাকে"
        
        answer, docs = rag_service.generate_answer(query)
        
        # Basic assertions
        assert answer is not None
        assert isinstance(answer, str)
        assert len(answer.strip()) > 0
        
        # Check if the expected answer is mentioned in the response
        assert expected_answer in answer or "মামা" in answer
        
        # Check if sources were retrieved
        assert docs is not None
        assert "documents" in docs
        assert len(docs["documents"][0]) > 0
    
    def test_bengali_query_3(self, rag_service):
        """
        Test Case 3: বিয়ের সময় কল্যাণীর প্রকৃত বয়স কত ছিল?
        Expected Answer: ১৫ বছর
        """
        query = "বিয়ের সময় কল্যাণীর প্রকৃত বয়স কত ছিল?"
        expected_answer = "১৫ বছর"
        
        answer, docs = rag_service.generate_answer(query)
        
        # Basic assertions
        assert answer is not None
        assert isinstance(answer, str)
        assert len(answer.strip()) > 0
        
        # Check if the expected answer is mentioned in the response
        assert "১৫" in answer or "পনের" in answer or "fifteen" in answer.lower()
        
        # Check if sources were retrieved
        assert docs is not None
        assert "documents" in docs
        assert len(docs["documents"][0]) > 0
    
    def test_batch_evaluation(self, eval_service):
        """Test batch evaluation with all three Bengali test cases."""
        qa_pairs = [
            {
                "query": "অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?",
                "expected_answer": "শুম্ভুনাথ"
            },
            {
                "query": "কাকে অনুপমের ভাগ্য দেবতা বলে উল্লেখ করা হয়েছে?",
                "expected_answer": "মামাকে"
            },
            {
                "query": "বিয়ের সময় কল্যাণীর প্রকৃত বয়স কত ছিল?",
                "expected_answer": "১৫ বছর"
            }
        ]
        
        results = eval_service.batch_eval(qa_pairs)
        
        # Check that we got results for all queries
        assert len(results) == 3
        
        # Check each result
        for i, result in enumerate(results):
            assert "query" in result
            assert "answer" in result
            assert "grounded" in result
            assert "groundedness_score" in result
            assert "relevance_score" in result
            
            # Check that the query matches
            assert result["query"] == qa_pairs[i]["query"]
            
            # Check that we got a non-empty answer
            assert result["answer"] is not None
            assert len(result["answer"].strip()) > 0
            
            # Check evaluation metrics
            assert isinstance(result["grounded"], bool)
            assert isinstance(result["groundedness_score"], (int, float))
            assert isinstance(result["relevance_score"], (int, float))
            assert 0 <= result["groundedness_score"] <= 1
            assert 0 <= result["relevance_score"] <= 1
    
    def test_mixed_language_query(self, rag_service):
        """Test a mixed Bengali-English query."""
        query = "What is the meaning of অনুপম in Bengali literature?"
        
        answer, docs = rag_service.generate_answer(query)
        
        # Basic assertions
        assert answer is not None
        assert isinstance(answer, str)
        assert len(answer.strip()) > 0
        
        # Check if sources were retrieved
        assert docs is not None
        assert "documents" in docs
    
    def test_english_query_about_bengali_content(self, rag_service):
        """Test an English query about Bengali literature content."""
        query = "Who is the main character in the Bengali story?"
        
        answer, docs = rag_service.generate_answer(query)
        
        # Basic assertions
        assert answer is not None
        assert isinstance(answer, str)
        assert len(answer.strip()) > 0
        
        # Check if sources were retrieved
        assert docs is not None
        assert "documents" in docs
    
    @pytest.mark.parametrize("query,expected_keywords", [
        ("অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?", ["শুম্ভুনাথ"]),
        ("কাকে অনুপমের ভাগ্য দেবতা বলে উল্লেখ করা হয়েছে?", ["মামা"]),
        ("বিয়ের সময় কল্যাণীর প্রকৃত বয়স কত ছিল?", ["১৫", "পনের"]),
    ])
    def test_parametrized_bengali_queries(self, rag_service, query, expected_keywords):
        """Parametrized test for Bengali queries with expected keywords."""
        answer, docs = rag_service.generate_answer(query)
        
        # Basic assertions
        assert answer is not None
        assert isinstance(answer, str)
        assert len(answer.strip()) > 0
        
        # Check if at least one expected keyword is present
        answer_lower = answer.lower()
        keyword_found = any(keyword.lower() in answer_lower for keyword in expected_keywords)
        assert keyword_found, f"None of the expected keywords {expected_keywords} found in answer: {answer}"
        
        # Check if sources were retrieved
        assert docs is not None
        assert "documents" in docs
        assert len(docs["documents"][0]) > 0


class TestEvaluationMetrics:
    """Test class for evaluation metrics."""
    
    @pytest.fixture
    def eval_service(self):
        """Create an evaluation service instance for testing."""
        return EvalService("test_user")
    
    def test_groundedness_evaluation(self, eval_service):
        """Test groundedness evaluation functionality."""
        query = "অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?"
        answer = "শুম্ভুনাথকে অনুপমের ভাষায় সুপুরুষ বলা হয়েছে।"
        docs = {
            "documents": [["অনুপম শুম্ভুনাথকে সুপুরুষ বলে উল্লেখ করেছেন।"]]
        }
        
        is_grounded, score, metrics = eval_service.evaluate_groundedness(query, answer, docs)
        
        # Check return types
        assert isinstance(is_grounded, bool)
        assert isinstance(score, (int, float))
        assert isinstance(metrics, dict)
        
        # Check score range
        assert 0 <= score <= 1
        
        # Check metrics structure
        assert "max_similarity" in metrics
        assert "avg_similarity" in metrics
        assert "groundedness_score" in metrics
    
    def test_relevance_evaluation(self, eval_service):
        """Test relevance evaluation functionality."""
        query = "অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?"
        answer = "শুম্ভুনাথকে অনুপমের ভাষায় সুপুরুষ বলা হয়েছে।"
        
        relevance_score, metrics = eval_service.evaluate_relevance(query, answer)
        
        # Check return types
        assert isinstance(relevance_score, (int, float))
        assert isinstance(metrics, dict)
        
        # Check score range
        assert 0 <= relevance_score <= 1
        
        # Check metrics structure
        assert "query_answer_similarity" in metrics


# Integration test data for API testing
BENGALI_TEST_CASES = [
    {
        "query": "অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?",
        "expected_answer": "শুম্ভুনাথ",
        "description": "Test case 1 from technical assessment"
    },
    {
        "query": "কাকে অনুপমের ভাগ্য দেবতা বলে উল্লেখ করা হয়েছে?",
        "expected_answer": "মামাকে",
        "description": "Test case 2 from technical assessment"
    },
    {
        "query": "বিয়ের সময় কল্যাণীর প্রকৃত বয়স কত ছিল?",
        "expected_answer": "১৫ বছর",
        "description": "Test case 3 from technical assessment"
    }
]