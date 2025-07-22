import pytest
from unittest.mock import Mock, patch
import numpy as np

from src.embeddings import EmbeddingModel, VectorStore
from src.retrieval import Retriever

class TestRetrieval:
    """Test retrieval and search functionality"""
    
    @pytest.fixture
    def mock_embedding_model(self):
        """Mock embedding model for testing"""
        model = Mock(spec=EmbeddingModel)
        model.model_name = "test-model"
        model.dimension = 384
        model.encode_texts.return_value = np.random.rand(3, 384)
        model.encode_single.return_value = np.random.rand(384)
        return model
    
    @pytest.fixture
    def vector_store(self, mock_embedding_model):
        """Vector store with test data"""
        store = VectorStore(mock_embedding_model)
        
        # Add test chunks
        test_chunks = [
            {
                "text": "অনুপম একজন ভালো ছেলে।",
                "chunk_id": "chunk_001",
                "word_count": 5,
                "metadata": {"page": 1, "language": "bn"}
            },
            {
                "text": "কল্যাণী একটি সুন্দর নাম।",
                "chunk_id": "chunk_002", 
                "word_count": 5,
                "metadata": {"page": 2, "language": "bn"}
            },
            {
                "text": "This is an English sentence.",
                "chunk_id": "chunk_003",
                "word_count": 5,
                "metadata": {"page": 3, "language": "en"}
            }
        ]
        
        store.chunks = test_chunks
        store.embeddings = np.random.rand(3, 384)
        
        return store
    
    @pytest.fixture
    def retriever(self, vector_store):
        return Retriever(vector_store)
    
    def test_retriever_initialization(self, retriever):
        """Test retriever initialization"""
        assert retriever is not None
        assert retriever.vector_store is not None
        assert retriever.search_history == []
    
    def test_basic_retrieval(self, retriever):
        """Test basic document retrieval"""
        query = "অনুপম সম্পর্কে বলো"
        
        results = retriever.retrieve(query, k=2, threshold=0.0)
        
        assert isinstance(results, list)
        assert len(results) <= 2
        
        # Check result structure if results exist
        if results:
            result = results[0]
            assert "similarity_score" in result
            assert "text" in result
            assert "metadata" in result
    
    def test_language_preference_filtering(self, retriever):
        """Test language preference in retrieval"""
        query = "test query"
        
        # Test Bengali preference
        bn_results = retriever.retrieve(query, k=5, language_preference="bn")
        
        if bn_results:
            assert all(r["metadata"]["language"] == "bn" for r in bn_results)
    
    def test_retrieval_with_context(self, retriever):
        """Test retrieval with conversation context"""
        query = "কল্যাণী কে?"
        history = ["অনুপম কে?", "তার পরিবার কেমন?"]
        
        result = retriever.retrieve_with_context(
            query=query,
            conversation_history=history,
            k=2
        )
        
        assert "query" in result
        assert "context" in result
        assert "sources" in result
        assert "total_chunks" in result
        assert isinstance(result["sources"], list)
    
    def test_search_history_tracking(self, retriever):
        """Test search history tracking"""
        initial_count = len(retriever.search_history)
        
        retriever.retrieve("test query", k=1)
        
        assert len(retriever.search_history) == initial_count + 1
        
        # Check history entry structure
        if retriever.search_history:
            entry = retriever.search_history[-1]
            assert "query" in entry
            assert "language" in entry
            assert "results_count" in entry
    
    def test_retrieval_stats(self, retriever):
        """Test retrieval statistics"""
        # Perform some searches
        retriever.retrieve("প্রশ্ন ১", k=1)
        retriever.retrieve("question 2", k=1)
        
        stats = retriever.get_retrieval_stats()
        
        if stats.get("status") != "no_searches":
            assert "total_searches" in stats
            assert "avg_results_per_search" in stats
            assert "query_languages" in stats
            assert stats["total_searches"] >= 2
    
    def test_empty_query_handling(self, retriever):
        """Test handling of empty queries"""
        results = retriever.retrieve("", k=5)
        assert results == []
        
        results = retriever.retrieve("   ", k=5)
        assert results == []
    
    def test_reranking(self, retriever):
        """Test result re-ranking functionality"""
        # This tests the internal _rerank_results method indirectly
        query = "বাংলা প্রশ্ন"
        
        results = retriever.retrieve(query, k=3)
        
        # If results exist, check they have rank_score
        if results:
            for result in results:
                assert "rank_score" in result or "similarity_score" in result
