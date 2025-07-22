from typing import List, Dict, Optional
import logging
from sentence_transformers.cross_encoder import CrossEncoder

from src.embeddings import VectorStore
from utils.helpers import normalize_query, detect_language

logger = logging.getLogger(__name__)

class Retriever:
    """Document retrieval system"""
    
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store
        self.search_history = []  # For tracking search patterns
        # Load the cross-encoder model. This will be downloaded on first use.
        self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2')
    
    def retrieve(self, 
                query: str, 
                k: int = 5, 
                threshold: float = 0.3,
                language_preference: Optional[str] = None) -> List[Dict]:
        """Retrieve relevant documents for a query"""
        
        if not query.strip():
            logger.warning("Empty query provided")
            return []
        
        # Normalize query
        normalized_query = normalize_query(query)
        query_language = detect_language(normalized_query)
        
        logger.info(f"Retrieving documents for query (lang: {query_language}): {normalized_query[:100]}...")
        
        # Basic similarity search
        results = self.vector_store.similarity_search(
            normalized_query, 
            k=k * 2,  # Get more results for filtering
            threshold=threshold
        )
        
        # Apply language filtering if requested
        if language_preference:
            results = [r for r in results if r["metadata"]["language"] == language_preference]
        
        # Re-rank results using the cross-encoder for better relevance
        results = self._rerank_with_cross_encoder(results, normalized_query)
        
        # Take top-k after re-ranking
        final_results = results[:k]
        
        # Store search info
        self.search_history.append({
            "query": normalized_query,
            "language": query_language,
            "results_count": len(final_results),
            "top_score": float(final_results[0]["cross_encoder_score"]) if final_results else 0.0
        })
        
        logger.info(f"Retrieved {len(final_results)} documents after re-ranking")
        return final_results
    
    def _rerank_with_cross_encoder(self, results: List[Dict], query: str) -> List[Dict]:
        """Re-ranks results using a CrossEncoder model for greater accuracy."""
        if not results or not query:
            return []

        logger.info(f"Re-ranking {len(results)} documents with cross-encoder...")
        
        # Create pairs of [query, document_text] for the cross-encoder
        pairs = [[query, result['text']] for result in results]
        
        # Predict the relevance scores
        scores = self.cross_encoder.predict(pairs)
        
        # Add the scores to the results and sort
        for i, result in enumerate(results):
            result['cross_encoder_score'] = float(scores[i])
            
        # Sort results by the new cross-encoder score in descending order
        reranked_results = sorted(results, key=lambda x: x['cross_encoder_score'], reverse=True)
        
        logger.info("Re-ranking complete.")
        return reranked_results
    
    def retrieve_with_context(self, 
                            query: str, 
                            conversation_history: Optional[List[str]] = None,
                            k: int = 5) -> Dict:
        """Enhanced retrieve with conversation context and cross-encoder re-ranking."""
        
        # Build expanded query with conversation context
        if conversation_history:
            recent_context = " ".join(conversation_history[-2:])  # Last 2 turns
            expanded_query = f"{query} {recent_context}"
            logger.info("Using conversation context for retrieval")
        else:
            expanded_query = query
        
        logger.info(f"Retrieving documents for query: {expanded_query[:100]}...")
        
        # Get more candidates for better selection
        initial_results = self.vector_store.similarity_search(
            expanded_query,
            k=k * 3,  # Get more results for re-ranking
            threshold=0.2  # Lower threshold for inclusiveness
        )
        
        # Re-rank using the cross-encoder
        reranked_results = self._rerank_with_cross_encoder(initial_results, query)
        
        # Select top-k results after re-ranking
        final_results = reranked_results[:k]
        
        # Prepare comprehensive context
        context_texts = []
        source_info = []
        
        for result in final_results:
            page_ref = f"[Page {result['metadata']['page']}]"
            context_texts.append(f"{page_ref} {result['text']}")
            
            source_info.append({
                "chunk_id": result.get("chunk_id", "unknown"),
                "page": result["metadata"]["page"],
                "language": result["metadata"]["language"],
                "similarity_score": result.get("similarity_score", 0),
                "cross_encoder_score": result.get("cross_encoder_score", 0.0)
            })
        
        combined_context = "\n\n".join(context_texts)
        
        return {
            "query": query,
            "context": combined_context,
            "sources": source_info,
            "total_chunks": len(final_results),
            "avg_score": sum(r["cross_encoder_score"] for r in source_info) / len(source_info) if source_info else 0.0
        }
    
    def get_retrieval_stats(self) -> Dict:
        """Get statistics about retrieval performance"""
        if not self.search_history:
            return {"status": "no_searches"}
        
        total_searches = len(self.search_history)
        avg_results = sum(s["results_count"] for s in self.search_history) / total_searches
        avg_score = sum(s["top_score"] for s in self.search_history) / total_searches
        
        languages = [s["language"] for s in self.search_history]
        
        return {
            "total_searches": total_searches,
            "avg_results_per_search": avg_results,
            "avg_top_similarity_score": avg_score,
            "query_languages": list(set(languages)),
            "language_distribution": {lang: languages.count(lang) for lang in set(languages)}
        }
