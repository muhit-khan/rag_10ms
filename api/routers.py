"""
FastAPI routers: /ask, /health
"""
from fastapi import APIRouter, Request
from services.rag_service import RAGService
from services.eval_service import EvalService

router = APIRouter()

@router.get("/health")
async def health():
    return {"status": "ok"}

@router.post("/ask")
async def ask(request: Request):
    data = await request.json()
    user_id = data.get("user_id", "default")
    query = data["query"]
    rag = RAGService(user_id)
    answer, docs = rag.generate_answer(query)
    return {"answer": answer, "sources": docs}

@router.post("/evaluate")
async def evaluate(request: Request):
    data = await request.json()
    user_id = data.get("user_id", "default")
    qa_pairs = data["qa_pairs"]
    eval_service = EvalService(user_id)
    results = eval_service.batch_eval(qa_pairs)
    return {"results": results}
