"""
Evaluation logic for RAG system.

This module provides functionality to evaluate the quality of RAG-generated answers,
focusing on groundedness, relevance, and factual consistency.
"""
import logging
import re
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity

from config import config
from services.rag_service import RAGService

# Configure logging
logger = logging.getLogger("eval_service")


class EvalService:
    """
    Service for evaluating RAG-generated answers.
    
    This service provides methods to evaluate the quality of answers generated
    by the RAG system, focusing on groundedness, relevance, and factual consistency.
    """
    
    def __init__(self, user_id: str):
        """
        Initialize the evaluation service.
        
        Args:
            user_id: User identifier for the RAG service
        """
        self.rag = RAGService(user_id)
        self.openai = OpenAI(api_key=config.OPENAI_API_KEY)
        self.ground_score_threshold = config.GROUND_SCORE_THRESHOLD
        self.cosine_threshold = config.COSINE_THRESHOLD
    
    def _get_embedding(self, text: str) -> List[float]:
        """
        Get embedding for a text using OpenAI's embedding model.
        
        Args:
            text: Text to embed
            
        Returns:
            List[float]: Embedding vector
            
        Raises:
            Exception: If embedding generation fails
        """
        try:
            response = self.openai.embeddings.create(
                input=text,
                model=config.EMBEDDING_MODEL
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            raise
    
    def _compute_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        Compute cosine similarity between two vectors.
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            float: Cosine similarity score (0-1)
        """
        # Reshape vectors for sklearn's cosine_similarity
        v1 = np.array(vec1).reshape(1, -1)
        v2 = np.array(vec2).reshape(1, -1)
        
        return float(cosine_similarity(v1, v2)[0][0])
    
    def _extract_citations(self, answer: str) -> List[str]:
        """
        Extract citation references from the answer.
        
        Args:
            answer: Generated answer text
            
        Returns:
            List[str]: List of citation references
        """
        # Look for citation patterns like [1], [2], etc.
        citation_pattern = r'\[(\d+)\]'
        citations = re.findall(citation_pattern, answer)
        
        # Also look for "According to..." patterns
        source_pattern = r'According to\s+([^,.]+)'
        sources = re.findall(source_pattern, answer)
        
        return list(set(citations + sources))
    
    def evaluate_groundedness(
        self,
        query: str,
        answer: str,
        docs: Any
    ) -> Tuple[bool, float, Dict[str, Any]]:
        """
        Evaluate if the answer is grounded in the source documents.
        
        This method computes embeddings for the answer and source documents,
        calculates cosine similarity, and determines if the answer is grounded
        based on a threshold.
        
        Args:
            query: User query
            answer: Generated answer
            docs: Source documents or ChromaDB results
            
        Returns:
            Tuple containing:
                bool: Whether the answer is grounded
                float: Groundedness score
                Dict: Detailed evaluation metrics
        """
        logger.info(f"Evaluating groundedness for query: '{query}'")
        
        try:
            # Extract documents from ChromaDB results if needed
            documents = []
            if isinstance(docs, dict) and "documents" in docs and docs["documents"]:
                documents = docs["documents"][0]
            elif isinstance(docs, list):
                documents = docs
            
            if not documents:
                logger.warning("No documents provided for groundedness evaluation")
                return False, 0.0, {"error": "No documents provided"}
            
            # Get embeddings
            answer_embedding = self._get_embedding(answer)
            doc_embeddings = [self._get_embedding(doc) for doc in documents]
            
            # Calculate similarities
            similarities = [
                self._compute_similarity(answer_embedding, doc_embedding)
                for doc_embedding in doc_embeddings
            ]
            
            # Calculate max and average similarity
            max_similarity = max(similarities) if similarities else 0.0
            avg_similarity = sum(similarities) / len(similarities) if similarities else 0.0
            
            # Check if any document has similarity above threshold
            is_grounded = max_similarity >= self.cosine_threshold
            
            # Extract citations
            citations = self._extract_citations(answer)
            has_citations = len(citations) > 0
            
            # Calculate overall groundedness score (weighted average)
            groundedness_score = 0.7 * max_similarity + 0.2 * avg_similarity + 0.1 * float(has_citations)
            
            # Prepare detailed metrics
            metrics = {
                "max_similarity": max_similarity,
                "avg_similarity": avg_similarity,
                "similarities": similarities,
                "has_citations": has_citations,
                "citations": citations,
                "threshold": self.cosine_threshold,
                "groundedness_score": groundedness_score
            }
            
            logger.info(f"Groundedness evaluation: score={groundedness_score:.2f}, is_grounded={is_grounded}")
            return is_grounded, groundedness_score, metrics
            
        except Exception as e:
            logger.error(f"Error evaluating groundedness: {str(e)}", exc_info=True)
            # Default to False in case of error
            return False, 0.0, {"error": str(e)}
    
    def evaluate_relevance(self, query: str, answer: str) -> Tuple[float, Dict[str, Any]]:
        """
        Evaluate the relevance of the answer to the query.
        
        Args:
            query: User query
            answer: Generated answer
            
        Returns:
            Tuple containing:
                float: Relevance score (0-1)
                Dict: Detailed evaluation metrics
        """
        try:
            # Get embeddings
            query_embedding = self._get_embedding(query)
            answer_embedding = self._get_embedding(answer)
            
            # Calculate similarity
            similarity = self._compute_similarity(query_embedding, answer_embedding)
            
            # Prepare metrics
            metrics = {
                "query_answer_similarity": similarity,
                "threshold": 0.5  # Arbitrary threshold for relevance
            }
            
            logger.info(f"Relevance evaluation: score={similarity:.2f}")
            return similarity, metrics
            
        except Exception as e:
            logger.error(f"Error evaluating relevance: {str(e)}", exc_info=True)
            return 0.0, {"error": str(e)}
    
    def batch_eval(self, qa_pairs: List[dict]) -> List[Dict[str, Any]]:
        """
        Evaluate a batch of QA pairs.
        
        Args:
            qa_pairs: List of dictionaries containing queries and optional expected answers
            
        Returns:
            List[Dict]: Evaluation results for each QA pair
        """
        logger.info(f"Batch evaluating {len(qa_pairs)} QA pairs")
        results = []
        
        for pair in qa_pairs:
            try:
                query = pair["query"]
                expected_answer = pair.get("expected_answer")
                
                # Generate answer using RAG
                answer, docs = self.rag.generate_answer(query)
                
                # Handle potential None answer
                if answer is None:
                    answer = ""
                    logger.warning(f"Received None answer for query: '{query}'")
                
                # Evaluate groundedness
                is_grounded, groundedness_score, ground_metrics = self.evaluate_groundedness(
                    query, answer, docs
                )
                
                # Evaluate relevance
                relevance_score, relevance_metrics = self.evaluate_relevance(query, answer)
                
                # Prepare result
                result = {
                    "query": query,
                    "answer": answer,
                    "grounded": is_grounded,
                    "groundedness_score": groundedness_score,
                    "relevance_score": relevance_score,
                    "metrics": {
                        "groundedness": ground_metrics,
                        "relevance": relevance_metrics
                    }
                }
                
                # Add comparison with expected answer if available
                if expected_answer:
                    try:
                        # Calculate similarity with expected answer
                        expected_embedding = self._get_embedding(expected_answer)
                        answer_embedding = self._get_embedding(answer)
                        similarity = self._compute_similarity(expected_embedding, answer_embedding)
                        
                        result["expected_answer"] = expected_answer
                        result["expected_similarity"] = similarity
                        result["metrics"]["expected"] = {"similarity": similarity}
                    except Exception as e:
                        logger.error(f"Error comparing with expected answer: {str(e)}")
                        result["metrics"]["expected"] = {"error": str(e)}
                
                results.append(result)
                logger.info(f"Evaluated query: '{query[:50]}...' - grounded: {is_grounded}")
                
            except Exception as e:
                logger.error(f"Error evaluating QA pair: {str(e)}", exc_info=True)
                results.append({
                    "query": pair["query"],
                    "error": str(e),
                    "grounded": False,
                    "groundedness_score": 0.0,
                    "relevance_score": 0.0
                })
        
        return results
