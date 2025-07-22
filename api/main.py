from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import List, Optional, Dict
import logging
import os
from pathlib import Path

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

# Mount static files
static_dir = Path(__file__).parent.parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Global RAG pipeline instance
rag_pipeline = None

# Pydantic models
class ChatRequest(BaseModel):
    message: str  # Frontend sends 'message'
    session_id: Optional[str] = "default"
    language: Optional[str] = None

class ChatResponse(BaseModel):
    response: str  # Frontend expects 'response'
    context: List[Dict] = []  # Frontend expects 'context'
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
            "docs": "/docs",
            "chat_interface": "/chat-interface"
        },
        "sample_queries": [
            "অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?",
            "কাকে অনুপমের ভাগ্য দেবতা বলে উল্লেখ করা হয়েছে?",
            "বিয়ের সময় কল্যাণীর প্রকৃত বয়স কত ছিল?"
        ]
    }

@app.get("/chat-interface")
async def chat_interface():
    """Serve the chat interface"""
    static_dir = Path(__file__).parent.parent / "static"
    index_file = static_dir / "index.html"
    
    if index_file.exists():
        return FileResponse(index_file)
    else:
        raise HTTPException(status_code=404, detail="Chat interface not found")

@app.get("/favicon.ico")
async def favicon():
    """Serve a simple favicon"""
    from fastapi.responses import Response
    # Simple 1x1 transparent PNG
    favicon_bytes = bytes([
        0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0x00, 0x00, 0x00, 0x0D,
        0x49, 0x48, 0x44, 0x52, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01,
        0x08, 0x06, 0x00, 0x00, 0x00, 0x1F, 0x15, 0xC4, 0x89, 0x00, 0x00, 0x00,
        0x0B, 0x49, 0x44, 0x41, 0x54, 0x78, 0xDA, 0x63, 0x60, 0x00, 0x02, 0x00,
        0x00, 0x05, 0x00, 0x01, 0xE2, 0x26, 0x05, 0x9B, 0x00, 0x00, 0x00, 0x00,
        0x49, 0x45, 0x4E, 0x44, 0xAE, 0x42, 0x60, 0x82
    ])
    return Response(content=favicon_bytes, media_type="image/png")

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
    
    if not request.message.strip():
        raise HTTPException(
            status_code=400, 
            detail="Message cannot be empty"
        )
    
    try:
        logger.info(f"Processing chat request: {request.message[:100]}...")
        
        result = await rag_pipeline.process_query(
            query=request.message,
            session_id=request.session_id or "default",
            language=request.language
        )
        
        # Transform the result to match frontend expectations
        response_data = {
            "response": result.get("answer", ""),
            "context": result.get("sources", []),
            "session_id": result.get("session_id", "default"),
            "language_detected": result.get("language_detected", "en"),
            "retrieval_stats": result.get("retrieval_stats")
        }
        
        return ChatResponse(**response_data)
        
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
