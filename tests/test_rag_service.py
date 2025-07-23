"""
Tests for the RAG service.

This module contains unit and integration tests for the RAG service,
which is responsible for generating answers to user queries.
"""
import os
import unittest
from unittest.mock import MagicMock, patch

import pytest

from services.rag_service import RAGService


class TestRAGService(unittest.TestCase):
    """Test cases for the RAG service."""

    def setUp(self):
        """Set up test fixtures."""
        # Mock the dependencies
        self.collection_mock = MagicMock()
        self.openai_mock = MagicMock()
        self.memory_mock = MagicMock()
        
        # Create patches
        self.get_collection_patch = patch('services.rag_service.get_collection', return_value=self.collection_mock)
        self.openai_patch = patch('services.rag_service.OpenAI', return_value=self.openai_mock)
        self.memory_patch = patch('services.rag_service.RedisWindow', return_value=self.memory_mock)
        
        # Start patches
        self.get_collection_patch.start()
        self.openai_patch.start()
        self.memory_patch.start()
        
        # Create RAG service
        self.rag_service = RAGService(user_id="test_user")
    
    def tearDown(self):
        """Tear down test fixtures."""
        # Stop patches
        self.get_collection_patch.stop()
        self.openai_patch.stop()
        self.memory_patch.stop()
    
    def test_init(self):
        """Test initialization of RAG service."""
        self.assertEqual(self.rag_service.collection, self.collection_mock)
        self.assertEqual(self.rag_service.openai, self.openai_mock)
        self.assertEqual(self.rag_service.memory, self.memory_mock)
    
    def test_embed_query(self):
        """Test embedding a query."""
        # Mock the OpenAI embeddings response
        embedding_mock = MagicMock()
        embedding_mock.data = [MagicMock(embedding=[0.1, 0.2, 0.3])]
        self.openai_mock.embeddings.create.return_value = embedding_mock
        
        # Call the method
        result = self.rag_service.embed_query("test query")
        
        # Check the result
        self.assertEqual(result, [0.1, 0.2, 0.3])
        
        # Check that the OpenAI API was called correctly
        self.openai_mock.embeddings.create.assert_called_once()
        call_args = self.openai_mock.embeddings.create.call_args
        self.assertEqual(call_args[1]['input'], "test query")
        # We don't check the model name as it comes from config
    
    def test_search(self):
        """Test searching for documents."""
        # Mock the embed_query method
        self.rag_service.embed_query = MagicMock(return_value=[0.1, 0.2, 0.3])
        
        # Mock the collection query response
        query_result = {"documents": [["doc1", "doc2"]], "metadatas": [[{"source": "test"}]], "distances": [[0.1, 0.2]]}
        self.collection_mock.query.return_value = query_result
        
        # Call the method
        result = self.rag_service.search("test query", k=2)
        
        # Check the result
        self.assertEqual(result, query_result)
        
        # Check that the collection query was called correctly
        self.collection_mock.query.assert_called_once_with(
            query_embeddings=[[0.1, 0.2, 0.3]],
            n_results=2,
            where={}
        )
    
    def test_generate_answer(self):
        """Test generating an answer."""
        # Mock the search method
        search_result = {"documents": [["doc1", "doc2"]], "metadatas": [[{"source": "test"}]], "distances": [[0.1, 0.2]]}
        self.rag_service.search = MagicMock(return_value=search_result)
        
        # Mock the OpenAI chat completions response
        chat_mock = MagicMock()
        chat_mock.choices = [MagicMock(message=MagicMock(content="test answer"))]
        self.openai_mock.chat.completions.create.return_value = chat_mock
        
        # Call the method
        answer, docs = self.rag_service.generate_answer("test query")
        
        # Check the result
        self.assertEqual(answer, "test answer")
        self.assertEqual(docs, search_result)
        
        # Check that the OpenAI API was called correctly
        self.openai_mock.chat.completions.create.assert_called_once()
        
        # Check that the memory was updated
        self.memory_mock.add_message.assert_any_call("user", "test query")
        self.memory_mock.add_message.assert_any_call("assistant", "test answer")


@pytest.mark.integration
class TestRAGServiceIntegration:
    """Integration tests for the RAG service."""
    
    @pytest.fixture
    def rag_service(self):
        """Create a RAG service for testing."""
        # Skip if no OpenAI API key
        if not os.environ.get("OPENAI_API_KEY"):
            pytest.skip("OpenAI API key not found")
        
        # Create RAG service
        return RAGService(user_id="test_user")
    
    def test_embed_query_integration(self, rag_service):
        """Test embedding a query with the real OpenAI API."""
        # Call the method
        result = rag_service.embed_query("test query")
        
        # Check the result
        assert isinstance(result, list)
        assert len(result) > 0
        assert all(isinstance(x, float) for x in result)
    
    def test_search_integration(self, rag_service):
        """Test searching for documents with the real ChromaDB."""
        # Call the method
        result = rag_service.search("test query", k=2)
        
        # Check the result
        assert isinstance(result, dict)
        assert "documents" in result
    
    def test_generate_answer_integration(self, rag_service):
        """Test generating an answer with the real OpenAI API."""
        # Call the method
        answer, docs = rag_service.generate_answer("What is the capital of France?")
        
        # Check the result
        assert isinstance(answer, str)
        assert len(answer) > 0
        assert isinstance(docs, dict)


if __name__ == "__main__":
    unittest.main()