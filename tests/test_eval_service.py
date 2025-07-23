"""
Tests for the evaluation service.

This module contains unit and integration tests for the evaluation service,
which is responsible for evaluating the quality of RAG-generated answers.
"""
import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import pytest

from services.eval_service import EvalService


class TestEvalService(unittest.TestCase):
    """Test cases for the evaluation service."""

    def setUp(self):
        """Set up test fixtures."""
        # Mock the dependencies
        self.rag_mock = MagicMock()
        self.openai_mock = MagicMock()
        
        # Create patches
        self.rag_patch = patch('services.eval_service.RAGService', return_value=self.rag_mock)
        self.openai_patch = patch('services.eval_service.OpenAI', return_value=self.openai_mock)
        
        # Start patches
        self.rag_patch.start()
        self.openai_patch.start()
        
        # Create evaluation service
        self.eval_service = EvalService(user_id="test_user")
    
    def tearDown(self):
        """Tear down test fixtures."""
        # Stop patches
        self.rag_patch.stop()
        self.openai_patch.stop()
    
    def test_init(self):
        """Test initialization of evaluation service."""
        self.assertEqual(self.eval_service.rag, self.rag_mock)
        self.assertEqual(self.eval_service.openai, self.openai_mock)
        self.assertIsNotNone(self.eval_service.ground_score_threshold)
        self.assertIsNotNone(self.eval_service.cosine_threshold)
    
    def test_get_embedding(self):
        """Test getting an embedding."""
        # Mock the OpenAI embeddings response
        embedding_mock = MagicMock()
        embedding_mock.data = [MagicMock(embedding=[0.1, 0.2, 0.3])]
        self.openai_mock.embeddings.create.return_value = embedding_mock
        
        # Call the method
        result = self.eval_service._get_embedding("test text")
        
        # Check the result
        self.assertEqual(result, [0.1, 0.2, 0.3])
        
        # Check that the OpenAI API was called correctly
        self.openai_mock.embeddings.create.assert_called_once()
        call_args = self.openai_mock.embeddings.create.call_args
        self.assertEqual(call_args[1]['input'], "test text")
    
    def test_compute_similarity(self):
        """Test computing similarity between vectors."""
        # Create test vectors
        vec1 = [1.0, 0.0, 0.0]
        vec2 = [0.0, 1.0, 0.0]
        
        # Call the method
        result = self.eval_service._compute_similarity(vec1, vec2)
        
        # Check the result (orthogonal vectors have similarity 0)
        self.assertEqual(result, 0.0)
        
        # Test with identical vectors (similarity 1)
        result = self.eval_service._compute_similarity(vec1, vec1)
        self.assertEqual(result, 1.0)
    
    def test_extract_citations(self):
        """Test extracting citations from text."""
        # Test with numbered citations
        answer = "This is a test [1] with multiple [2] citations [3]."
        result = self.eval_service._extract_citations(answer)
        self.assertEqual(set(result), {"1", "2", "3"})
        
        # Test with "According to" citations
        answer = "According to Smith, this is true. According to Jones, this is false."
        result = self.eval_service._extract_citations(answer)
        self.assertEqual(set(result), {"Smith", "Jones"})
        
        # Test with mixed citations
        answer = "According to Smith [1], this is cited properly."
        result = self.eval_service._extract_citations(answer)
        self.assertEqual(set(result), {"Smith", "1"})
    
    def test_evaluate_groundedness(self):
        """Test evaluating groundedness."""
        # Mock the dependencies
        self.eval_service._get_embedding = MagicMock(side_effect=[
            [1.0, 0.0, 0.0],  # Answer embedding
            [0.9, 0.1, 0.0],  # Doc 1 embedding (similar to answer)
            [0.1, 0.9, 0.0]   # Doc 2 embedding (different from answer)
        ])
        
        # Create test data
        query = "test query"
        answer = "test answer"
        docs = {
            "documents": [["doc1", "doc2"]],
            "metadatas": [[{"source": "test1"}, {"source": "test2"}]],
            "distances": [[0.1, 0.9]]
        }
        
        # Call the method
        is_grounded, score, metrics = self.eval_service.evaluate_groundedness(query, answer, docs)
        
        # Check the result
        self.assertTrue(is_grounded)  # Should be grounded due to high similarity with doc1
        self.assertGreater(score, 0.5)  # Score should be high
        
        # Check metrics
        self.assertIn("max_similarity", metrics)
        self.assertIn("avg_similarity", metrics)
        self.assertIn("similarities", metrics)
        self.assertIn("has_citations", metrics)
        self.assertIn("citations", metrics)
        self.assertIn("threshold", metrics)
        self.assertIn("groundedness_score", metrics)
    
    def test_evaluate_relevance(self):
        """Test evaluating relevance."""
        # Mock the dependencies
        self.eval_service._get_embedding = MagicMock(side_effect=[
            [1.0, 0.0, 0.0],  # Query embedding
            [0.8, 0.2, 0.0]   # Answer embedding (similar to query)
        ])
        
        # Create test data
        query = "test query"
        answer = "test answer"
        
        # Call the method
        score, metrics = self.eval_service.evaluate_relevance(query, answer)
        
        # Check the result
        self.assertGreater(score, 0.5)  # Score should be high due to similarity
        
        # Check metrics
        self.assertIn("query_answer_similarity", metrics)
        self.assertIn("threshold", metrics)
    
    def test_batch_eval(self):
        """Test batch evaluation."""
        # Mock the dependencies
        self.rag_mock.generate_answer.return_value = ("test answer", {"documents": [["doc1"]]})
        self.eval_service.evaluate_groundedness = MagicMock(return_value=(True, 0.8, {}))
        self.eval_service.evaluate_relevance = MagicMock(return_value=(0.9, {}))
        
        # Create test data
        qa_pairs = [
            {"query": "test query 1"},
            {"query": "test query 2", "expected_answer": "expected answer"}
        ]
        
        # Call the method
        results = self.eval_service.batch_eval(qa_pairs)
        
        # Check the result
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["query"], "test query 1")
        self.assertEqual(results[0]["answer"], "test answer")
        self.assertTrue(results[0]["grounded"])
        self.assertEqual(results[0]["groundedness_score"], 0.8)
        self.assertEqual(results[0]["relevance_score"], 0.9)
        
        # Check that the second result has expected_answer
        self.assertEqual(results[1]["query"], "test query 2")
        self.assertIn("expected_answer", results[1])


@pytest.mark.integration
class TestEvalServiceIntegration:
    """Integration tests for the evaluation service."""
    
    @pytest.fixture
    def eval_service(self):
        """Create an evaluation service for testing."""
        # Create evaluation service
        return EvalService(user_id="test_user")
    
    def test_compute_similarity_integration(self, eval_service):
        """Test computing similarity with real numpy operations."""
        # Create test vectors
        vec1 = np.array([1.0, 0.0, 0.0])
        vec2 = np.array([0.0, 1.0, 0.0])
        
        # Call the method
        result = eval_service._compute_similarity(vec1, vec2)
        
        # Check the result (orthogonal vectors have similarity 0)
        assert result == 0.0
        
        # Test with identical vectors (similarity 1)
        result = eval_service._compute_similarity(vec1, vec1)
        assert result == 1.0
    
    def test_extract_citations_integration(self, eval_service):
        """Test extracting citations with real regex operations."""
        # Test with numbered citations
        answer = "This is a test [1] with multiple [2] citations [3]."
        result = eval_service._extract_citations(answer)
        assert set(result) == {"1", "2", "3"}
        
        # Test with "According to" citations
        answer = "According to Smith, this is true. According to Jones, this is false."
        result = eval_service._extract_citations(answer)
        assert set(result) == {"Smith", "Jones"}


if __name__ == "__main__":
    unittest.main()