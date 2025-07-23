"""
FastAPI routers: /ask, /health, /evaluate

This module defines the API endpoints for the RAG system with proper
input validation, error handling, and documentation.
"""
import logging
import time
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from slowapi import Limiter
from slowapi.util import get_remote_address

from api.auth import Token, User, generate_test_token, get_current_user
from config import config
from services.rag_service import RAGService
from services.eval_service import EvalService

# Configure logging
logger = logging.getLogger("api")

# Configure rate limiter
limiter = Limiter(key_func=get_remote_address)

router = APIRouter()

# Define request and response models for better validation and documentation
class HealthResponse(BaseModel):
    """Health check response model."""
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")


class AskRequest(BaseModel):
    """Request model for the /ask endpoint."""
    query: str = Field(..., min_length=1, max_length=1000, description="The question to answer")
    user_id: Optional[str] = Field("default", description="User identifier for conversation context")

    @validator("query")
    def query_not_empty(cls, v):
        """Validate that query is not empty or just whitespace."""
        if not v.strip():
            raise ValueError("Query cannot be empty or just whitespace")
        return v.strip()


class Source(BaseModel):
    """Model for a source document."""
    document: str = Field(..., description="Source document text")
    metadata: Dict[str, Any] = Field(..., description="Document metadata")
    score: Optional[float] = Field(None, description="Relevance score")


class AskResponse(BaseModel):
    """Response model for the /ask endpoint."""
    answer: str = Field(..., description="Generated answer")
    sources: List[Source] = Field(..., description="Source documents used for the answer")
    processing_time: float = Field(..., description="Processing time in seconds")


class QAPair(BaseModel):
    """Model for a question-answer pair."""
    query: str = Field(..., min_length=1, description="The question to evaluate")
    expected_answer: Optional[str] = Field(None, description="Expected answer (if available)")


class EvaluateRequest(BaseModel):
    """Request model for the /evaluate endpoint."""
    qa_pairs: List[QAPair] = Field(..., description="List of QA pairs to evaluate")
    
    @validator("qa_pairs")
    def validate_qa_pairs(cls, v):
        if not v or len(v) < 1:
            raise ValueError("At least one QA pair is required")
        return v
    user_id: Optional[str] = Field("default", description="User identifier")


class EvaluationResult(BaseModel):
    """Model for an evaluation result."""
    query: str = Field(..., description="The evaluated question")
    answer: str = Field(..., description="Generated answer")
    grounded: bool = Field(..., description="Whether the answer is grounded in the sources")
    score: Optional[float] = Field(None, description="Evaluation score")


class EvaluateResponse(BaseModel):
    """Response model for the /evaluate endpoint."""
    results: List[EvaluationResult] = Field(..., description="Evaluation results")
    processing_time: float = Field(..., description="Processing time in seconds")


@router.get("/health", response_model=HealthResponse, tags=["System"])
async def health():
    """
    Health check endpoint.
    
    Returns:
        dict: Status information
    """
    logger.info("Health check requested")
    return {
        "status": "ok",
        "version": "1.0.0"
    }


# Authentication models
class LoginRequest(BaseModel):
    """Request model for the /auth/token endpoint."""
    username: str = Field(..., min_length=1, description="Username")
    password: str = Field(..., min_length=1, description="Password")


@router.post(
    "/auth/token",
    response_model=Token,
    tags=["Authentication"],
    responses={
        200: {"description": "Successfully generated token"},
        401: {"description": "Invalid credentials"},
        422: {"description": "Validation error"}
    }
)
@limiter.limit(f"{config.RATE_LIMIT}/minute")
async def login(request: Request, login_request: LoginRequest):
    """
    Generate a JWT token for authentication.
    
    In a real application, this would validate credentials against a database.
    For this example, we'll accept any username/password and generate a token.
    
    Args:
        request: FastAPI request object
        login_request: Login credentials
        
    Returns:
        Token: JWT token
        
    Raises:
        HTTPException: If authentication fails
    """
    # In a real application, validate credentials here
    # For this example, we'll accept any username/password
    
    # Generate token
    token = generate_test_token(login_request.username)
    
    logger.info(f"Generated token for user: {login_request.username}")
    return token


@router.get(
    "/auth/me",
    response_model=User,
    tags=["Authentication"],
    responses={
        200: {"description": "Current user information"},
        401: {"description": "Not authenticated"}
    }
)
async def get_me(current_user: User = Depends(get_current_user)):
    """
    Get information about the currently authenticated user.
    
    Args:
        current_user: Current authenticated user (from dependency)
        
    Returns:
        User: Current user information
    """
    return current_user


@router.post(
    "/ask",
    response_model=AskResponse,
    tags=["RAG"],
    responses={
        200: {"description": "Successful response"},
        400: {"description": "Bad request"},
        401: {"description": "Not authenticated"},
        422: {"description": "Validation error"},
        429: {"description": "Rate limit exceeded"},
        500: {"description": "Internal server error"}
    }
)
@limiter.limit(f"{config.RATE_LIMIT}/minute")
async def ask(
    request: Request,
    ask_request: AskRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Generate an answer for a given query using RAG.
    
    Args:
        request: FastAPI request object
        ask_request: Query and user information
        
    Returns:
        dict: Answer and source documents
        
    Raises:
        HTTPException: If an error occurs during processing
    """
    start_time = time.time()
    
    try:
        logger.info(f"Processing query: '{ask_request.query}' for user: {current_user.user_id}")
        
        # Use authenticated user ID
        rag = RAGService(current_user.user_id)
        answer, docs = rag.generate_answer(ask_request.query)
        
        # Format the sources
        sources = []
        if docs and "documents" in docs and docs["documents"]:
            for i, doc in enumerate(docs["documents"][0]):
                source = {
                    "document": doc,
                    "metadata": docs["metadatas"][0][i] if "metadatas" in docs and docs["metadatas"] else {},
                    "score": docs["distances"][0][i] if "distances" in docs and docs["distances"] else None
                }
                sources.append(source)
        
        processing_time = time.time() - start_time
        logger.info(f"Query processed in {processing_time:.2f}s")
        
        return {
            "answer": answer,
            "sources": sources,
            "processing_time": processing_time
        }
    
    except KeyError as e:
        logger.error(f"Missing required field: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Missing required field: {str(e)}"
        )
    
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while processing your query"
        )


@router.post(
    "/evaluate",
    response_model=EvaluateResponse,
    tags=["Evaluation"],
    responses={
        200: {"description": "Successful response"},
        400: {"description": "Bad request"},
        401: {"description": "Not authenticated"},
        422: {"description": "Validation error"},
        429: {"description": "Rate limit exceeded"},
        500: {"description": "Internal server error"}
    }
)
@limiter.limit(f"{config.RATE_LIMIT}/minute")
async def evaluate(
    request: Request,
    eval_request: EvaluateRequest,
    current_user: User = Depends(get_current_user)
):
    """
    Evaluate a batch of queries using the RAG system.
    
    Args:
        request: FastAPI request object
        eval_request: List of QA pairs to evaluate
        
    Returns:
        dict: Evaluation results
        
    Raises:
        HTTPException: If an error occurs during processing
    """
    start_time = time.time()
    
    try:
        logger.info(f"Evaluating {len(eval_request.qa_pairs)} QA pairs for user: {current_user.user_id}")
        
        # Convert Pydantic models to dictionaries
        qa_pairs = [qa_pair.dict() for qa_pair in eval_request.qa_pairs]
        
        # Use authenticated user ID
        eval_service = EvalService(current_user.user_id)
        results = eval_service.batch_eval(qa_pairs)
        
        processing_time = time.time() - start_time
        logger.info(f"Evaluation completed in {processing_time:.2f}s")
        
        return {
            "results": results,
            "processing_time": processing_time
        }
    
    except KeyError as e:
        logger.error(f"Missing required field: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Missing required field: {str(e)}"
        )
    
    except Exception as e:
        logger.error(f"Error during evaluation: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred during evaluation"
        )


# Note: The rate limit exception handler should be added to the main FastAPI app
# in main.py, not here in the router. The implementation will be moved there.
