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
                            conversation_history: Optional[List[str]] = None,
                            k: int = 5) -> Dict:
        """Enhanced retrieve with conversation context and better ranking"""
        
        # Build expanded query with conversation context
        if conversation_history:
            # Use more sophisticated context integration
            recent_context = " ".join(conversation_history[-2:])  # Last 2 turns
            expanded_query = f"{query} {recent_context}"
            logger.info("Using conversation context for retrieval")
            logger.info(f"Retrieving documents for query (lang: {detect_language(query)}): {expanded_query[:100]}...")
        else:
            expanded_query = query
            logger.info(f"Retrieving documents for query (lang: {detect_language(query)}): {expanded_query[:100]}...")
        
        # Get more candidates for better selection
        initial_results = self.vector_store.similarity_search(
            expanded_query,
            k=k * 2,  # Get more results for filtering
            threshold=0.3  # Lower threshold for inclusiveness
        )
        
        # Enhanced ranking with multiple factors
        enhanced_results = []
        query_keywords = self._extract_key_terms(query)
        
        for result in initial_results:
            # Calculate enhanced score
            base_score = result["similarity_score"] if "similarity_score" in result else result.get("similarity", 0)
            
            # Keyword matching bonus
            text_lower = result["text"].lower()
            keyword_matches = sum(1 for keyword in query_keywords if keyword.lower() in text_lower)
            keyword_bonus = keyword_matches * 0.05
            
            # Content quality bonus (prefer substantial content)
            word_count = len(result["text"].split())
            quality_bonus = 0.02 if word_count > 30 else 0
            
            # Character name bonus (important for literature questions)
            char_names = ['অনুপম', 'কল্যাণী', 'শম্ভুনাথ', 'মামা', 'হরিশ']
            char_bonus = sum(0.03 for name in char_names if name in result["text"]) 
            
            enhanced_score = base_score + keyword_bonus + quality_bonus + char_bonus
            
            result["enhanced_score"] = enhanced_score
            enhanced_results.append(result)
        
        # Sort by enhanced score and select top results
        enhanced_results.sort(key=lambda x: x["enhanced_score"], reverse=True)
        final_results = enhanced_results[:k]
        
        # Prepare comprehensive context
        context_texts = []
        source_info = []
        
        for result in final_results:
            # Add page context for better reference
            page_ref = f"[পৃষ্ঠা {result['metadata']['page']}]"
            context_texts.append(f"{page_ref} {result['text']}")
            
            source_info.append({
                "chunk_id": result.get("chunk_id", "unknown"),
                "page": result["metadata"]["page"],
                "language": result["metadata"]["language"],
                "similarity_score": result.get("similarity_score", result.get("similarity", 0)),
                "enhanced_score": result["enhanced_score"]
            })
        
        combined_context = "\n\n".join(context_texts)
        
        return {
            "query": query,
            "context": combined_context,
            "sources": source_info,
            "total_chunks": len(final_results),
            "avg_similarity": sum(r.get("similarity_score", r.get("similarity", 0)) for r in final_results) / len(final_results) if final_results else 0.0
        }
    
    def _extract_key_terms(self, query: str) -> List[str]:
        """Extract key terms from query for better matching"""
        import re
        
        # Bengali stop words
        stop_words = {
            'কি', 'করে', 'কে', 'কী', 'কোন', 'কোথায়', 'কেন', 'কিভাবে', 'কত', 'কার', 
            'যে', 'যা', 'এর', 'ের', 'তে', 'এই', 'সেই', 'তিনি', 'তার', 'তাহার', 
            'হয়', 'হয়েছে', 'ছিল', 'আছে', 'বলা', 'বলে', 'দিয়ে', 'নিয়ে'
        }
        
        # Extract meaningful words
        words = re.findall(r'[\u0980-\u09FF]+|\w+', query.lower())
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        
        return keywords
    
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
