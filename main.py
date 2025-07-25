"""
Main entry point for RAG system

This module provides two main functionalities:
1. A complete RAG workflow (PDF ingestion to query answering)
2. A FastAPI application for serving the RAG system via API

The complete workflow follows these steps:
- PDF → Text extraction → Cleaning → Chunking → Embedding → Persist to ChromaDB
- User query → Embed query → ChromaDB search → Merge with Redis memory → Prompt template → LLM call → Answer with citations
"""
import argparse
import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from typing import Dict, List, Optional, Tuple

import uvicorn
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address

from api.auth import AuthMiddleware
from api.routers import router, limiter
from config import config
from ingest import find_pdf_files, process_pdf, create_embeddings
from db.chroma_client import get_collection
from services.rag_service import RAGService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f"api_{time.strftime('%Y%m%d')}.log"),
    ],
)
logger = logging.getLogger("main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan context manager for FastAPI app.
    
    Handles startup and shutdown events.
    """
    # Startup
    logger.info("Starting up RAG system...")
    # You could initialize resources here (e.g., database connections)
    
    yield
    
    # Shutdown
    logger.info("Shutting down RAG system...")
    # You could clean up resources here


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.
    
    Returns:
        FastAPI: Configured FastAPI application
    """
    # Create FastAPI app with custom title, description, and version
    app = FastAPI(
        title="Multilingual RAG System",
        description="A Bengali-English Retrieval-Augmented Generation (RAG) system for answering textbook questions with grounded citations.",
        version="1.0.0",
        docs_url=None,  # We'll customize the docs URL
        redoc_url=None,  # We'll customize the redoc URL
        lifespan=lifespan,
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # In production, replace with specific origins
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Configure rate limiting
    app.state.limiter = limiter
    
    # Custom rate limit exception handler with correct signature
    @app.exception_handler(RateLimitExceeded)
    async def custom_rate_limit_exceeded_handler(request: Request, exc: Exception):
        """Handle rate limit exceeded exceptions."""
        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content={"detail": "Rate limit exceeded. Please try again later."}
        )
    
    # Add middlewares
    app.add_middleware(SlowAPIMiddleware)
    app.add_middleware(AuthMiddleware)
    
    # Add custom exception handlers
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        """Handle HTTP exceptions with custom format."""
        logger.error(f"HTTP error: {exc.status_code} - {exc.detail}")
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail}
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        """Handle general exceptions with custom format."""
        logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "An internal server error occurred"}
        )
    
    # Include routers
    app.include_router(router)
    
    # Custom OpenAPI and documentation endpoints
    @app.get("/docs", include_in_schema=False)
    async def custom_swagger_ui_html():
        """Serve custom Swagger UI."""
        return get_swagger_ui_html(
            openapi_url="/openapi.json",
            title=app.title + " - API Documentation",
            swagger_js_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui-bundle.js",
            swagger_css_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui.css",
        )
    
    @app.get("/openapi.json", include_in_schema=False)
    async def get_openapi_endpoint():
        """Serve OpenAPI schema."""
        return get_openapi(
            title=app.title,
            version=app.version,
            description=app.description,
            routes=app.routes,
        )
    
    # Add a simple redirect from root to docs
    @app.get("/", include_in_schema=False)
    async def root():
        """Redirect root to documentation."""
        return {"message": "Welcome to the Multilingual RAG System API", "docs_url": "/docs"}
    
    logger.info("FastAPI app created and configured")
    return app


def run_ingestion(pdf_path: str, clean: bool = False) -> None:
    """
    Run the ingestion pipeline.
    
    Args:
        pdf_path: Path to the directory containing PDF files
        clean: Whether to clear the existing collection before ingestion
    """
    logger.info("Starting ingestion pipeline...")
    
    # Get ChromaDB collection
    collection = get_collection()
    
    # Clear collection if requested
    if clean:
        logger.warning("Clearing existing collection")
        collection.delete(where={})
    
    # Find PDF files
    pdf_files = find_pdf_files(pdf_path)
    if not pdf_files:
        logger.error(f"No PDF files found in {pdf_path}")
        return
    
    # Process each PDF
    all_chunks = []
    all_metadata = []
    all_ids = []
    
    for pdf_file in pdf_files:
        logger.info(f"Processing {pdf_file}")
        chunks, metadata_list = process_pdf(pdf_file)
        
        if not chunks:
            continue
        
        # Generate IDs for chunks
        ids = [f"{pdf_file.stem}_{i}" for i in range(len(chunks))]
        
        all_chunks.extend(chunks)
        all_metadata.extend(metadata_list)
        all_ids.extend(ids)
    
    if not all_chunks:
        logger.error("No chunks generated from any PDF")
        return
    
    # Create embeddings
    logger.info(f"Creating embeddings for {len(all_chunks)} chunks")
    embeddings = create_embeddings(all_chunks)
    
    # Filter out chunks with failed embeddings
    valid_indices = [i for i, emb in enumerate(embeddings) if emb is not None]
    if len(valid_indices) < len(embeddings):
        logger.warning(f"Filtering out {len(embeddings) - len(valid_indices)} chunks with failed embeddings")
        all_chunks = [all_chunks[i] for i in valid_indices]
        all_metadata = [all_metadata[i] for i in valid_indices]
        all_ids = [all_ids[i] for i in valid_indices]
        embeddings = [embeddings[i] for i in valid_indices]
    
    # Store in ChromaDB
    logger.info(f"Storing {len(all_chunks)} chunks in ChromaDB")
    
    # Process in batches to avoid memory issues
    batch_size = 100
    for i in range(0, len(all_chunks), batch_size):
        end_idx = min(i + batch_size, len(all_chunks))
        try:
            collection.add(
                documents=all_chunks[i:end_idx],
                embeddings=embeddings[i:end_idx],
                metadatas=all_metadata[i:end_idx],
                ids=all_ids[i:end_idx]
            )
            logger.info(f"Stored batch {i//batch_size + 1}/{(len(all_chunks)-1)//batch_size + 1}")
        except Exception as e:
            logger.error(f"Error storing batch {i//batch_size + 1}: {str(e)}")
    
    logger.info("Ingestion complete")
    logger.info(f"Total documents: {collection.count()}")


def run_query(query: str, user_id: str = "default_user") -> Tuple[str, Dict]:
    """
    Run a query through the RAG pipeline.
    
    Args:
        query: The user's query
        user_id: User identifier for conversation history
        
    Returns:
        Tuple containing the answer and document citations
    """
    logger.info(f"Processing query: {query}")
    
    # Initialize RAG service
    rag_service = RAGService(user_id)
    
    # Generate answer
    answer, docs = rag_service.generate_answer(query)
    
    # Format citations
    citations = {}
    
    # Ensure docs has the expected structure
    if (docs and isinstance(docs, dict) and
        "documents" in docs and docs["documents"] is not None and
        "metadatas" in docs and docs["metadatas"] is not None):
        
        documents = docs["documents"]
        metadatas = docs["metadatas"]
        
        if (len(documents) > 0 and isinstance(documents[0], list) and
            len(metadatas) > 0 and isinstance(metadatas[0], list) and
            len(documents[0]) > 0 and len(metadatas[0]) > 0):
            
            for i, (doc, metadata) in enumerate(zip(documents[0], metadatas[0])):
                source = metadata.get("source", f"Unknown Source {i}")
                if source not in citations:
                    citations[source] = []
                citations[source].append({
                    "text": doc[:100] + "..." if len(doc) > 100 else doc,
                    "metadata": metadata
                })
    
    # Ensure answer is a string
    if answer is None:
        answer = "Sorry, I couldn't generate an answer based on the available information."
    
    logger.info(f"Generated answer with {len(citations)} citation sources")
    return answer, citations


def run_complete_workflow(args: argparse.Namespace) -> None:
    """
    Run the complete RAG workflow (ingestion + query).
    
    Args:
        args: Command line arguments
    """
    # Run ingestion if requested
    if args.ingest:
        run_ingestion(args.pdf_path, args.clean)
    
    # Run query if provided
    if args.query:
        answer, citations = run_query(args.query, args.user_id)
        
        # Print the answer
        print("\n" + "="*80)
        print("ANSWER:")
        print(answer)
        
        # Print citations
        print("\n" + "="*80)
        print("CITATIONS:")
        for source, citations_list in citations.items():
            print(f"\nSource: {source}")
            for i, citation in enumerate(citations_list):
                print(f"  [{i+1}] {citation['text']}")
                if args.verbose:
                    print(f"      Metadata: {citation['metadata']}")
        print("="*80 + "\n")


def setup_argparse() -> argparse.Namespace:
    """Parse command line arguments for the RAG workflow."""
    parser = argparse.ArgumentParser(
        description="Multilingual RAG System",
        epilog="Example: python main.py --ingest --clean --pdf_path data/raw/ --query 'What is machine learning?'"
    )
    
    # Server mode
    parser.add_argument(
        "--server",
        action="store_true",
        help="Run in server mode (FastAPI)",
    )
    
    # Ingestion options
    parser.add_argument(
        "--ingest",
        action="store_true",
        help="Run the ingestion pipeline",
    )
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Clear existing collection before ingestion",
    )
    parser.add_argument(
        "--pdf_path",
        type=str,
        default=config.PDF_PATH,
        help=f"Path to PDF directory (default: {config.PDF_PATH})",
    )
    
    # Query options
    parser.add_argument(
        "--query",
        type=str,
        help="Query to run through the RAG pipeline",
    )
    parser.add_argument(
        "--user_id",
        type=str,
        default="default_user",
        help="User ID for conversation history",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed citation metadata",
    )
    
    return parser.parse_args()



# --- Chat Endpoints ---
from fastapi import APIRouter
chat_router = APIRouter()

@chat_router.get("/static/{filename}")
async def static_files(filename: str):
    return FileResponse(f"static/{filename}")

@chat_router.get("/chat")
async def chat_page():
    return FileResponse("static/chat.html")

@chat_router.post("/chat/")
async def chat_api(request: Request):
    data = await request.json()
    prompt = data.get("prompt", "")
    answer, _ = run_query(prompt)
    return {"answer": answer}

# Create the FastAPI app
app = create_app()
app.include_router(chat_router)

if __name__ == "__main__":
    args = setup_argparse()
    
    if args.server:
        # Run in server mode
        logger.info(f"Starting server on {config.API_HOST}:{config.API_PORT}")
        uvicorn.run(
            "main:app",
            host=config.API_HOST,
            port=config.API_PORT,
            reload=config.API_RELOAD,
            log_level="info",
        )
    else:
        # Run the complete workflow
        run_complete_workflow(args)
