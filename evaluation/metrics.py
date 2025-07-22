from typing import List, Dict, Optional
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import logging

logger = logging.getLogger(__name__)

class RAGEvaluator:
    """Evaluation metrics for RAG system"""
    
    def __init__(self):
        self.evaluation_history = []
    
    def evaluate_groundedness(self, 
                            answer: str, 
                            retrieved_contexts: List[str],
                            threshold: float = 0.3) -> Dict:
        """
        Evaluate if the answer is grounded in retrieved context
        
        Simple approach: Check if answer content appears in contexts
        """
        if not answer or not retrieved_contexts:
            return {
                "groundedness_score": 0.0,
                "is_grounded": False,
                "explanation": "Empty answer or no context"
            }
        
        # Tokenize answer into meaningful parts
        answer_words = set(answer.lower().split())
        
        # Check overlap with each context
        context_overlaps = []
        for context in retrieved_contexts:
            context_words = set(context.lower().split())
            
            # Calculate word overlap
            overlap = len(answer_words.intersection(context_words))
            overlap_ratio = overlap / len(answer_words) if answer_words else 0
            context_overlaps.append(overlap_ratio)
        
        # Best overlap across all contexts
        max_overlap = max(context_overlaps) if context_overlaps else 0.0
        is_grounded = max_overlap >= threshold
        
        return {
            "groundedness_score": max_overlap,
            "is_grounded": is_grounded,
            "context_overlaps": context_overlaps,
            "threshold": threshold,
            "explanation": f"Max overlap: {max_overlap:.2f}, Threshold: {threshold}"
        }
    
    def evaluate_relevance(self, 
                          query: str, 
                          retrieved_contexts: List[str],
                          similarity_scores: List[float]) -> Dict:
        """
        Evaluate relevance of retrieved contexts to the query
        """
        if not retrieved_contexts or not similarity_scores:
            return {
                "relevance_score": 0.0,
                "avg_similarity": 0.0,
                "explanation": "No contexts or scores provided"
            }
        
        # Average similarity score
        avg_similarity = np.mean(similarity_scores)
        
        # Count high-relevance contexts (similarity > 0.5)
        high_relevance_count = sum(1 for score in similarity_scores if score > 0.5)
        relevance_ratio = high_relevance_count / len(similarity_scores)
        
        # Combined relevance score
        relevance_score = (avg_similarity * 0.7) + (relevance_ratio * 0.3)
        
        return {
            "relevance_score": relevance_score,
            "avg_similarity": avg_similarity,
            "high_relevance_count": high_relevance_count,
            "total_contexts": len(retrieved_contexts),
            "relevance_ratio": relevance_ratio,
            "explanation": f"Avg similarity: {avg_similarity:.3f}, High relevance: {high_relevance_count}/{len(similarity_scores)}"
        }
    
    def evaluate_answer_quality(self, 
                               query: str,
                               answer: str,
                               language: str) -> Dict:
        """
        Simple heuristic evaluation of answer quality
        """
        if not answer.strip():
            return {
                "quality_score": 0.0,
                "explanation": "Empty answer"
            }
        
        quality_factors = {
            "length": 0.0,
            "language_match": 0.0,
            "completeness": 0.0
        }
        
        # Length factor (not too short, not too long)
        answer_length = len(answer.split())
        if 3 <= answer_length <= 50:  # Reasonable length
            quality_factors["length"] = 1.0
        elif answer_length < 3:
            quality_factors["length"] = 0.3
        else:  # Too long
            quality_factors["length"] = 0.7
        
        # Language match (simple check)
        if language == "bn":
            # Check if answer contains Bengali characters
            import re
            bengali_chars = len(re.findall(r'[\u0980-\u09FF]', answer))
            if bengali_chars > 0:
                quality_factors["language_match"] = 1.0
        else:  # English
            # Check if answer is primarily English
            quality_factors["language_match"] = 0.8  # Assume reasonable for now
        
        # Completeness (has sentence structure)
        has_punctuation = any(p in answer for p in ['.', '।', '!', '?'])
        quality_factors["completeness"] = 1.0 if has_punctuation else 0.6
        
        # Weighted average
        quality_score = (
            quality_factors["length"] * 0.3 +
            quality_factors["language_match"] * 0.4 +
            quality_factors["completeness"] * 0.3
        )
        
        return {
            "quality_score": quality_score,
            "factors": quality_factors,
            "explanation": f"Length: {quality_factors['length']:.2f}, Lang: {quality_factors['language_match']:.2f}, Complete: {quality_factors['completeness']:.2f}"
        }
    
    def comprehensive_evaluation(self,
                               query: str,
                               answer: str, 
                               retrieved_contexts: List[str],
                               similarity_scores: List[float],
                               language: str) -> Dict:
        """
        Comprehensive evaluation combining all metrics
        """
        # Individual evaluations
        groundedness = self.evaluate_groundedness(answer, retrieved_contexts)
        relevance = self.evaluate_relevance(query, retrieved_contexts, similarity_scores)
        quality = self.evaluate_answer_quality(query, answer, language)
        
        # Overall score (weighted average)
        overall_score = (
            groundedness["groundedness_score"] * 0.4 +
            relevance["relevance_score"] * 0.3 +
            quality["quality_score"] * 0.3
        )
        
        evaluation_result = {
            "query": query,
            "answer": answer,
            "language": language,
            "overall_score": overall_score,
            "groundedness": groundedness,
            "relevance": relevance,
            "quality": quality,
            "retrieved_contexts_count": len(retrieved_contexts),
            "timestamp": None  # Could add timestamp
        }
        
        # Store for analytics
        self.evaluation_history.append(evaluation_result)
        
        logger.info(f"Evaluation complete. Overall score: {overall_score:.3f}")
        
        return evaluation_result
    
    def get_evaluation_stats(self) -> Dict:
        """Get statistics from all evaluations"""
        if not self.evaluation_history:
            return {"status": "no_evaluations"}
        
        scores = [eval_result["overall_score"] for eval_result in self.evaluation_history]
        groundedness_scores = [eval_result["groundedness"]["groundedness_score"] for eval_result in self.evaluation_history]
        relevance_scores = [eval_result["relevance"]["relevance_score"] for eval_result in self.evaluation_history]
        quality_scores = [eval_result["quality"]["quality_score"] for eval_result in self.evaluation_history]
        
        return {
            "total_evaluations": len(self.evaluation_history),
            "overall_stats": {
                "avg_score": np.mean(scores),
                "min_score": np.min(scores),
                "max_score": np.max(scores)
            },
            "groundedness_stats": {
                "avg": np.mean(groundedness_scores),
                "grounded_percentage": sum(1 for eval_result in self.evaluation_history if eval_result["groundedness"]["is_grounded"]) / len(self.evaluation_history)
            },
            "relevance_stats": {
                "avg": np.mean(relevance_scores)
            },
            "quality_stats": {
                "avg": np.mean(quality_scores)
            },
            "language_distribution": {
                lang: sum(1 for eval_result in self.evaluation_history if eval_result["language"] == lang)
                for lang in set(eval_result["language"] for eval_result in self.evaluation_history)
            }
        }
    
    def create_evaluation_report(self, output_path: str) -> None:
        """Create a detailed evaluation report"""
        stats = self.get_evaluation_stats()
        
        if stats.get("status") == "no_evaluations":
            logger.warning("No evaluations to report")
            return
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("# RAG System Evaluation Report\n\n")
                
                # Summary statistics
                f.write("## Summary Statistics\n")
                f.write(f"- Total Evaluations: {stats['total_evaluations']}\n")
                f.write(f"- Average Overall Score: {stats['overall_stats']['avg_score']:.3f}\n")
                f.write(f"- Score Range: {stats['overall_stats']['min_score']:.3f} - {stats['overall_stats']['max_score']:.3f}\n\n")
                
                # Detailed metrics
                f.write("## Detailed Metrics\n\n")
                
                f.write("### Groundedness\n")
                f.write(f"- Average Score: {stats['groundedness_stats']['avg']:.3f}\n")
                f.write(f"- Grounded Answers: {stats['groundedness_stats']['grounded_percentage']:.1%}\n\n")
                
                f.write("### Relevance\n")
                f.write(f"- Average Score: {stats['relevance_stats']['avg']:.3f}\n\n")
                
                f.write("### Quality\n")
                f.write(f"- Average Score: {stats['quality_stats']['avg']:.3f}\n\n")
                
                # Language breakdown
                f.write("### Language Distribution\n")
                for lang, count in stats['language_distribution'].items():
                    f.write(f"- {lang}: {count} evaluations\n")
                
                f.write("\n## Individual Evaluations\n\n")
                
                # Individual evaluation details
                for i, eval_result in enumerate(self.evaluation_history[-10:], 1):  # Last 10
                    f.write(f"### Evaluation {i}\n")
                    f.write(f"**Query**: {eval_result['query']}\n\n")
                    f.write(f"**Answer**: {eval_result['answer'][:200]}{'...' if len(eval_result['answer']) > 200 else ''}\n\n")
                    f.write(f"**Overall Score**: {eval_result['overall_score']:.3f}\n\n")
                    f.write(f"- Groundedness: {eval_result['groundedness']['groundedness_score']:.3f} ({'✓' if eval_result['groundedness']['is_grounded'] else '✗'})\n")
                    f.write(f"- Relevance: {eval_result['relevance']['relevance_score']:.3f}\n")
                    f.write(f"- Quality: {eval_result['quality']['quality_score']:.3f}\n\n")
                    f.write("---\n\n")
            
            logger.info(f"Evaluation report saved to: {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to create evaluation report: {e}")


# Sample evaluation test cases for the assessment
SAMPLE_TEST_CASES = [
    {
        "query": "অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?",
        "expected_answer": "শুম্ভুনাথ",
        "language": "bn"
    },
    {
        "query": "কাকে অনুপমের ভাগ্য দেবতা বলে উল্লেখ করা হয়েছে?",
        "expected_answer": "মামাকে",
        "language": "bn"
    },
    {
        "query": "বিয়ের সময় কল্যাণীর প্রকৃত বয়স কত ছিল?",
        "expected_answer": "১৫ বছর",
        "language": "bn"
    }
]
