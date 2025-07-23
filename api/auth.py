"""
Authentication middleware and utilities for the RAG API.

This module provides JWT-based authentication for the API endpoints,
including token generation, validation, and middleware.
"""
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, Optional, Union

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from pydantic import BaseModel, Field

from config import config

# Configure logging
logger = logging.getLogger("auth")

# Security scheme for Swagger UI
security = HTTPBearer(auto_error=False)

# Token settings
ACCESS_TOKEN_EXPIRE_MINUTES = 30
ALGORITHM = "HS256"
SECRET_KEY = config.JWT_SECRET


class Token(BaseModel):
    """Token response model."""
    access_token: str = Field(..., description="JWT access token")
    token_type: str = Field("bearer", description="Token type")
    expires_at: int = Field(..., description="Token expiration timestamp (Unix)")


class TokenData(BaseModel):
    """Token data model for decoded JWT payload."""
    sub: str = Field(..., description="Subject (user ID)")
    exp: int = Field(..., description="Expiration timestamp")
    iat: int = Field(..., description="Issued at timestamp")
    scopes: list = Field(default_factory=list, description="Token scopes/permissions")


class User(BaseModel):
    """User model."""
    user_id: str = Field(..., description="User ID")
    disabled: bool = Field(False, description="Whether the user is disabled")


def create_access_token(
    data: Dict,
    expires_delta: Optional[timedelta] = None
) -> str:
    """
    Create a JWT access token.
    
    Args:
        data: Data to encode in the token
        expires_delta: Token expiration time
        
    Returns:
        str: Encoded JWT token
    """
    to_encode = data.copy()
    
    # Set expiration time
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    
    # Add standard claims
    to_encode.update({
        "exp": expire.timestamp(),
        "iat": datetime.utcnow().timestamp()
    })
    
    # Encode token
    try:
        encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
        return encoded_jwt
    except Exception as e:
        logger.error(f"Error creating access token: {str(e)}")
        raise


def decode_token(token: str) -> TokenData:
    """
    Decode and validate a JWT token.
    
    Args:
        token: JWT token to decode
        
    Returns:
        TokenData: Decoded token data
        
    Raises:
        HTTPException: If token is invalid or expired
    """
    try:
        # Decode token
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        
        # Create token data
        token_data = TokenData(
            sub=payload.get("sub", ""),
            exp=payload.get("exp", 0),
            iat=payload.get("iat", 0),
            scopes=payload.get("scopes", [])
        )
        
        # Check if token is expired
        if datetime.fromtimestamp(token_data.exp) < datetime.utcnow():
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token expired",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        return token_data
    
    except JWTError as e:
        logger.error(f"JWT error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except Exception as e:
        logger.error(f"Error decoding token: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication error",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> User:
    """
    Get the current authenticated user from the request.
    
    This function is used as a dependency in protected endpoints.
    
    Args:
        credentials: HTTP Authorization credentials
        
    Returns:
        User: Current authenticated user
        
    Raises:
        HTTPException: If authentication fails
    """
    # Check if credentials are provided
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Decode token
    token_data = decode_token(credentials.credentials)
    
    # In a real application, you would fetch the user from a database
    # For this example, we'll just create a user object from the token data
    user = User(user_id=token_data.sub, disabled=False)
    
    return user


def get_optional_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> Optional[User]:
    """
    Get the current user if authenticated, or None if not.
    
    This function is used for endpoints that support both authenticated
    and unauthenticated access.
    
    Args:
        credentials: HTTP Authorization credentials
        
    Returns:
        Optional[User]: Current user or None
    """
    if credentials is None:
        return None
    
    try:
        token_data = decode_token(credentials.credentials)
        return User(user_id=token_data.sub, disabled=False)
    except HTTPException:
        return None


class AuthMiddleware:
    """
    Middleware for JWT authentication.
    
    This middleware extracts the user from the request and adds it to the request state.
    It does not enforce authentication, which is handled by the endpoint dependencies.
    """
    
    def __init__(self, app):
        """Initialize the middleware with the ASGI app."""
        self.app = app
    
    async def __call__(self, scope, receive, send):
        """Process the ASGI request."""
        if scope["type"] != "http":
            # If not an HTTP request, just pass it through
            await self.app(scope, receive, send)
            return
        
        # Create a request object
        request = Request(scope, receive=receive)
        
        # Extract token from Authorization header
        auth_header = request.headers.get("Authorization")
        user = None
        
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header.replace("Bearer ", "")
            try:
                token_data = decode_token(token)
                user = User(user_id=token_data.sub, disabled=False)
            except HTTPException:
                # Don't raise an exception here, just set user to None
                pass
        
        # Add user to request state
        request.state.user = user
        
        # Process the request
        await self.app(scope, receive, send)


# Helper function to generate a token for testing
def generate_test_token(user_id: str) -> Token:
    """
    Generate a test token for the given user ID.
    
    Args:
        user_id: User ID to include in the token
        
    Returns:
        Token: Generated token
    """
    expires_delta = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    expire_time = datetime.utcnow() + expires_delta
    
    access_token = create_access_token(
        data={"sub": user_id, "scopes": ["read", "write"]},
        expires_delta=expires_delta
    )
    
    return Token(
        access_token=access_token,
        token_type="bearer",
        expires_at=int(expire_time.timestamp())
    )