"""
Evaluation logic
"""
from services.rag_service import RAGService
from config import config
from typing import List

class EvalService:
    def __init__(self, user_id: str):
        self.rag = RAGService(user_id)

    def evaluate_groundedness(self, query: str, answer: str, docs: List[str]) -> bool:
        # Dummy cosine similarity check
        # Replace with actual embedding comparison
        return True

    def batch_eval(self, qa_pairs: List[dict]):
        results = []
        for pair in qa_pairs:
            answer, docs = self.rag.generate_answer(pair["query"])
            grounded = self.evaluate_groundedness(pair["query"], answer, docs)
            results.append({"query": pair["query"], "answer": answer, "grounded": grounded})
        return results
