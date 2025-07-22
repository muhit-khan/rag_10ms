from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
import logging
import os

from src.rag_pipeline import RAGPipeline
from config import settings
from utils import setup_logging

# Setup logging
setup_logging("INFO")
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title=settings.API_TITLE,
    description="Multilingual RAG system for Bengali literature questions",
    version=settings.API_VERSION,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global RAG pipeline instance
rag_pipeline = None

# Pydantic models
class ChatRequest(BaseModel):
    query: str
    session_id: Optional[str] = "default"
    language: Optional[str] = None

class ChatResponse(BaseModel):
    answer: str
    sources: List[Dict]
    session_id: str
    language_detected: str
    retrieval_stats: Optional[Dict] = None

class DocumentUploadRequest(BaseModel):
    pdf_path: str

class SystemStats(BaseModel):
    pipeline_status: str
    knowledge_base_loaded: bool
    total_chunks: Optional[int] = None
    languages: Optional[List[str]] = None
    active_sessions: Optional[int] = None

# API Events
@app.on_event("startup")
async def startup_event():
    """Initialize the RAG pipeline on startup"""
    global rag_pipeline
    try:
        logger.info("Starting RAG system...")
        rag_pipeline = RAGPipeline()
        
        # Check if there's a default PDF to load
        default_pdf_path = os.path.join(settings.RAW_DATA_DIR, "hsc26_bangla_1st_paper.pdf")
        
        await rag_pipeline.initialize(pdf_path=default_pdf_path if os.path.exists(default_pdf_path) else None)
        
        logger.info("RAG system startup completed")
        
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        # Continue startup even if initialization fails

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("RAG system shutting down...")

# API Endpoints
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Multilingual RAG API for Bengali Literature",
        "version": settings.API_VERSION,
        "status": "running",
        "endpoints": {
            "chat": "/chat",
            "upload": "/upload-document", 
            "stats": "/stats",
            "health": "/health",
            "docs": "/docs"
        },
        "sample_queries": [
            "অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?",
            "কাকে অনুপমের ভাগ্য দেবতা বলে উল্লেখ করা হয়েছে?",
            "বিয়ের সময় কল্যাণীর প্রকৃত বয়স কত ছিল?"
        ]
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    global rag_pipeline
    
    status = "healthy"
    details = {"api": "running"}
    
    if rag_pipeline:
        details["pipeline"] = "initialized" if rag_pipeline.is_initialized else "not_ready"
        details["knowledge_base"] = "loaded" if rag_pipeline.knowledge_base_loaded else "empty"
    else:
        status = "unhealthy"
        details["pipeline"] = "not_initialized"
    
    return {
        "status": status,
        "details": details
    }

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """Main chat endpoint for RAG queries"""
    global rag_pipeline
    
    if not rag_pipeline:
        raise HTTPException(
            status_code=503, 
            detail="RAG pipeline not initialized"
        )
    
    if not request.query.strip():
        raise HTTPException(
            status_code=400, 
            detail="Query cannot be empty"
        )
    
    try:
        logger.info(f"Processing chat request: {request.query[:100]}...")
        
        result = await rag_pipeline.process_query(
            query=request.query,
            session_id=request.session_id,
            language=request.language
        )
        
        return ChatResponse(**result)
        
    except Exception as e:
        logger.error(f"Chat processing failed: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Error processing query: {str(e)}"
        )

@app.post("/upload-document")
async def upload_document(request: DocumentUploadRequest, background_tasks: BackgroundTasks):
    """Upload and process a new PDF document"""
    global rag_pipeline
    
    if not rag_pipeline:
        raise HTTPException(
            status_code=503, 
            detail="RAG pipeline not initialized"
        )
    
    if not os.path.exists(request.pdf_path):
        raise HTTPException(
            status_code=404, 
            detail=f"PDF file not found: {request.pdf_path}"
        )
    
    try:
        # Process document in background
        background_tasks.add_task(rag_pipeline.add_document, request.pdf_path)
        
        return {
            "message": "Document upload initiated",
            "pdf_path": request.pdf_path,
            "status": "processing"
        }
        
    except Exception as e:
        logger.error(f"Document upload failed: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Error uploading document: {str(e)}"
        )

@app.get("/stats", response_model=SystemStats)
async def get_system_stats():
    """Get system statistics and status"""
    global rag_pipeline
    
    if not rag_pipeline:
        return SystemStats(
            pipeline_status="not_initialized",
            knowledge_base_loaded=False
        )
    
    try:
        stats = rag_pipeline.get_system_stats()
        return SystemStats(**stats)
        
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Error retrieving statistics: {str(e)}"
        )

@app.delete("/sessions/{session_id}")
async def clear_session(session_id: str):
    """Clear conversation history for a session"""
    global rag_pipeline
    
    if not rag_pipeline:
        raise HTTPException(
            status_code=503, 
            detail="RAG pipeline not initialized"
        )
    
    try:
        rag_pipeline.clear_session(session_id)
        
        return {
            "message": f"Session {session_id} cleared",
            "session_id": session_id
        }
        
    except Exception as e:
        logger.error(f"Error clearing session: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Error clearing session: {str(e)}"
        )

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return {
        "error": "Not found",
        "message": "The requested endpoint does not exist",
        "available_endpoints": ["/", "/chat", "/health", "/stats", "/docs"]
    }

if __name__ == "__main__":
    import uvicorn
    
    logger.info(f"Starting server on {settings.API_HOST}:{settings.API_PORT}")
    
    uvicorn.run(
        "api.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=True,
        log_level="info"
    )
