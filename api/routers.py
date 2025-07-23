"""
FastAPI routers: /ask, /health
"""
from fastapi import APIRouter

router = APIRouter()

@router.get("/health")
def health():
    return {"status": "ok"}

# ... /ask endpoint will be added later
