"""
Tests for the authentication system.

This module contains unit and integration tests for the authentication system,
including token generation, validation, and middleware.
"""
import time
import unittest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException, Request
from jose import jwt

from api.auth import (
    ALGORITHM,
    SECRET_KEY,
    AuthMiddleware,
    Token,
    TokenData,
    User,
    create_access_token,
    decode_token,
    generate_test_token,
    get_current_user,
    get_optional_user,
)


class TestAuth(unittest.TestCase):
    """Test cases for the authentication system."""

    def test_create_access_token(self):
        """Test creating an access token."""
        # Create a token with test data
        data = {"sub": "test_user", "scopes": ["read", "write"]}
        token = create_access_token(data)
        
        # Decode the token
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        
        # Check the payload
        self.assertEqual(payload["sub"], "test_user")
        self.assertEqual(payload["scopes"], ["read", "write"])
        self.assertIn("exp", payload)
        self.assertIn("iat", payload)
    
    def test_create_access_token_with_expiry(self):
        """Test creating an access token with custom expiry."""
        # Create a token with test data and custom expiry
        data = {"sub": "test_user"}
        expires_delta = timedelta(minutes=5)
        token = create_access_token(data, expires_delta)
        
        # Decode the token
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        
        # Check the expiry
        exp_time = datetime.fromtimestamp(payload["exp"])
        iat_time = datetime.fromtimestamp(payload["iat"])
        self.assertAlmostEqual((exp_time - iat_time).total_seconds(), 300, delta=10)
    
    def test_decode_token(self):
        """Test decoding a token."""
        # Create a token with test data
        data = {"sub": "test_user", "scopes": ["read", "write"]}
        token = create_access_token(data)
        
        # Decode the token
        token_data = decode_token(token)
        
        # Check the token data
        self.assertEqual(token_data.sub, "test_user")
        self.assertEqual(token_data.scopes, ["read", "write"])
        self.assertIsNotNone(token_data.exp)
        self.assertIsNotNone(token_data.iat)
    
    def test_decode_token_expired(self):
        """Test decoding an expired token."""
        # Create a token with test data and negative expiry
        data = {"sub": "test_user"}
        expires_delta = timedelta(minutes=-5)
        token = create_access_token(data, expires_delta)
        
        # Try to decode the token
        with self.assertRaises(HTTPException) as context:
            decode_token(token)
        
        # Check the exception
        self.assertEqual(context.exception.status_code, 401)
        self.assertEqual(context.exception.detail, "Token expired")
    
    def test_decode_token_invalid(self):
        """Test decoding an invalid token."""
        # Try to decode an invalid token
        with self.assertRaises(HTTPException) as context:
            decode_token("invalid_token")
        
        # Check the exception
        self.assertEqual(context.exception.status_code, 401)
        self.assertEqual(context.exception.detail, "Could not validate credentials")
    
    @patch('api.auth.decode_token')
    async def test_get_current_user(self, mock_decode_token):
        """Test getting the current user."""
        # Mock the decode_token function
        mock_decode_token.return_value = TokenData(sub="test_user", exp=int(time.time()) + 300, iat=int(time.time()), scopes=["read", "write"])
        
        # Create mock credentials
        credentials = MagicMock()
        credentials.credentials = "test_token"
        
        # Get the current user
        user = await get_current_user(credentials)
        
        # Check the user
        self.assertEqual(user.user_id, "test_user")
        self.assertFalse(user.disabled)
    
    @patch('api.auth.decode_token')
    async def test_get_current_user_no_credentials(self, mock_decode_token):
        """Test getting the current user with no credentials."""
        # Try to get the current user with no credentials
        with self.assertRaises(HTTPException) as context:
            await get_current_user(None)
        
        # Check the exception
        self.assertEqual(context.exception.status_code, 401)
        self.assertEqual(context.exception.detail, "Not authenticated")
    
    @patch('api.auth.decode_token')
    def test_get_optional_user(self, mock_decode_token):
        """Test getting an optional user."""
        # Mock the decode_token function
        mock_decode_token.return_value = TokenData(sub="test_user", exp=int(time.time()) + 300, iat=int(time.time()), scopes=["read", "write"])
        
        # Create mock credentials
        credentials = MagicMock()
        credentials.credentials = "test_token"
        
        # Get the optional user
        user = get_optional_user(credentials)
        
        # Check the user
        self.assertEqual(user.user_id, "test_user")
        self.assertFalse(user.disabled)
    
    @patch('api.auth.decode_token')
    def test_get_optional_user_no_credentials(self, mock_decode_token):
        """Test getting an optional user with no credentials."""
        # Get the optional user with no credentials
        user = get_optional_user(None)
        
        # Check the user
        self.assertIsNone(user)
    
    @patch('api.auth.decode_token')
    def test_get_optional_user_invalid_token(self, mock_decode_token):
        """Test getting an optional user with an invalid token."""
        # Mock the decode_token function to raise an exception
        mock_decode_token.side_effect = HTTPException(status_code=401, detail="Invalid token")
        
        # Create mock credentials
        credentials = MagicMock()
        credentials.credentials = "invalid_token"
        
        # Get the optional user
        user = get_optional_user(credentials)
        
        # Check the user
        self.assertIsNone(user)
    
    def test_generate_test_token(self):
        """Test generating a test token."""
        # Generate a test token
        token = generate_test_token("test_user")
        
        # Check the token
        self.assertIsInstance(token, Token)
        self.assertEqual(token.token_type, "bearer")
        self.assertIsNotNone(token.access_token)
        self.assertIsNotNone(token.expires_at)


@pytest.mark.integration
class TestAuthIntegration:
    """Integration tests for the authentication system."""
    
    def test_create_decode_token_integration(self):
        """Test creating and decoding a token with real JWT operations."""
        # Create a token with test data
        data = {"sub": "test_user", "scopes": ["read", "write"]}
        token = create_access_token(data)
        
        # Decode the token
        token_data = decode_token(token)
        
        # Check the token data
        assert token_data.sub == "test_user"
        assert token_data.scopes == ["read", "write"]
        assert token_data.exp is not None
        assert token_data.iat is not None
    
    @pytest.mark.asyncio
    async def test_auth_middleware_integration(self):
        """Test the authentication middleware with a real request."""
        # Create a middleware
        async def mock_app(scope, receive, send):
            return None
            
        middleware = AuthMiddleware(app=mock_app)
        
        # Create a mock request
        scope = {
            "type": "http",
            "headers": [(b"authorization", b"Bearer " + create_access_token({"sub": "test_user"}).encode())]
        }
        
        # Process the request (this should not raise an exception)
        await middleware(scope, None, None)


if __name__ == "__main__":
    unittest.main()