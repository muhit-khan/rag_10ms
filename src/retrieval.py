from typing import List, Dict, Optional
import logging

from src.embeddings import VectorStore
from utils.helpers import normalize_query, detect_language

logger = logging.getLogger(__name__)

class Retriever:
    """Document retrieval system"""
    
    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store
        self.search_history = []  # For tracking search patterns
    
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
        
        # Re-rank results (simple implementation)
        results = self._rerank_results(results, normalized_query, query_language)
        
        # Take top-k after re-ranking
        final_results = results[:k]
        
        # Store search info
        self.search_history.append({
            "query": normalized_query,
            "language": query_language,
            "results_count": len(final_results),
            "top_score": final_results[0]["similarity_score"] if final_results else 0.0
        })
        
        logger.info(f"Retrieved {len(final_results)} documents")
        return final_results
    
    def _rerank_results(self, results: List[Dict], query: str, query_lang: str) -> List[Dict]:
        """Simple re-ranking based on language match and other factors"""
        if not results:
            return results
        
        def calculate_rank_score(result: Dict) -> float:
            base_score = result["similarity_score"]
            
            # Boost score for language match
            if result["metadata"]["language"] == query_lang:
                base_score *= 1.2
            
            # Boost for longer chunks (more context)
            word_count = result.get("word_count", 0)
            if word_count > 100:  # Substantial content
                base_score *= 1.1
            
            # Small penalty for very short chunks
            if word_count < 20:
                base_score *= 0.9
            
            return base_score
        
        # Calculate new scores and sort
        for result in results:
            result["rank_score"] = calculate_rank_score(result)
        
        return sorted(results, key=lambda x: x["rank_score"], reverse=True)
    
    def retrieve_with_context(self, 
                            query: str, 
                            conversation_history: List[str] = None,
                            k: int = 5) -> Dict:
        """Retrieve with conversation context"""
        
        # If we have conversation history, expand the query
        if conversation_history:
            # Simple context expansion - append recent queries
            recent_context = " ".join(conversation_history[-3:])  # Last 3 turns
            expanded_query = f"{recent_context} {query}"
            logger.info("Using conversation context for retrieval")
        else:
            expanded_query = query
        
        # Retrieve documents
        results = self.retrieve(expanded_query, k=k)
        
        # Prepare context for LLM
        context_texts = []
        source_info = []
        
        for result in results:
            context_texts.append(result["text"])
            source_info.append({
                "chunk_id": result.get("chunk_id", "unknown"),
                "page": result["metadata"]["page"],
                "language": result["metadata"]["language"],
                "similarity_score": result["similarity_score"]
            })
        
        combined_context = "\n\n".join(context_texts)
        
        return {
            "query": query,
            "context": combined_context,
            "sources": source_info,
            "total_chunks": len(results),
            "avg_similarity": sum(r["similarity_score"] for r in results) / len(results) if results else 0.0
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
