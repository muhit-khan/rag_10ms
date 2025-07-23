"""
Tests for the API endpoints.

This module contains unit and integration tests for the API endpoints,
including authentication, ask, and evaluate endpoints.
"""
import json
import unittest
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from api.auth import create_access_token
from api.routers import router
from main import create_app


class TestAPI(unittest.TestCase):
    """Test cases for the API endpoints."""

    def setUp(self):
        """Set up test fixtures."""
        # Create a test client
        self.app = create_app()
        self.client = TestClient(self.app)
        
        # Create a test token
        self.token = create_access_token({"sub": "test_user"})
        self.headers = {"Authorization": f"Bearer {self.token}"}
        
        # Mock the RAG service
        self.rag_mock = MagicMock()
        self.rag_mock.generate_answer.return_value = (
            "test answer",
            {
                "documents": [["doc1"]],
                "metadatas": [[{"source": "test"}]],
                "distances": [[0.1]]
            }
        )
        
        # Mock the evaluation service
        self.eval_mock = MagicMock()
        self.eval_mock.batch_eval.return_value = [
            {
                "query": "test query",
                "answer": "test answer",
                "grounded": True,
                "groundedness_score": 0.8,
                "relevance_score": 0.9
            }
        ]
        
        # Create patches
        self.rag_patch = patch('api.routers.RAGService', return_value=self.rag_mock)
        self.eval_patch = patch('api.routers.EvalService', return_value=self.eval_mock)
        
        # Start patches
        self.rag_patch.start()
        self.eval_patch.start()
    
    def tearDown(self):
        """Tear down test fixtures."""
        # Stop patches
        self.rag_patch.stop()
        self.eval_patch.stop()
    
    def test_health(self):
        """Test the health endpoint."""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")
    
    def test_auth_token(self):
        """Test the auth token endpoint."""
        response = self.client.post(
            "/auth/token",
            json={"username": "test_user", "password": "test_password"}
        )
        self.assertEqual(response.status_code, 200)
        self.assertIn("access_token", response.json())
        self.assertEqual(response.json()["token_type"], "bearer")
    
    def test_auth_me(self):
        """Test the auth me endpoint."""
        # Test with valid token
        response = self.client.get("/auth/me", headers=self.headers)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["user_id"], "test_user")
        
        # Test without token
        response = self.client.get("/auth/me")
        self.assertEqual(response.status_code, 401)
    
    def test_ask(self):
        """Test the ask endpoint."""
        # Test with valid token and query
        response = self.client.post(
            "/ask",
            headers=self.headers,
            json={"query": "test query"}
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["answer"], "test answer")
        self.assertEqual(len(response.json()["sources"]), 1)
        
        # Test without token
        response = self.client.post(
            "/ask",
            json={"query": "test query"}
        )
        self.assertEqual(response.status_code, 401)
        
        # Test with invalid query
        response = self.client.post(
            "/ask",
            headers=self.headers,
            json={"query": ""}
        )
        self.assertEqual(response.status_code, 422)
    
    def test_evaluate(self):
        """Test the evaluate endpoint."""
        # Test with valid token and query
        response = self.client.post(
            "/evaluate",
            headers=self.headers,
            json={
                "qa_pairs": [
                    {"query": "test query"}
                ]
            }
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.json()["results"]), 1)
        self.assertEqual(response.json()["results"][0]["answer"], "test answer")
        
        # Test without token
        response = self.client.post(
            "/evaluate",
            json={
                "qa_pairs": [
                    {"query": "test query"}
                ]
            }
        )
        self.assertEqual(response.status_code, 401)
        
        # Test with invalid query
        response = self.client.post(
            "/evaluate",
            headers=self.headers,
            json={
                "qa_pairs": []
            }
        )
        self.assertEqual(response.status_code, 422)


@pytest.mark.integration
class TestAPIIntegration:
    """Integration tests for the API endpoints."""
    
    @pytest.fixture
    def client(self):
        """Create a test client."""
        app = create_app()
        return TestClient(app)
    
    @pytest.fixture
    def token(self):
        """Create a test token."""
        return create_access_token({"sub": "test_user"})
    
    def test_health_integration(self, client):
        """Test the health endpoint with a real FastAPI instance."""
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"
    
    def test_auth_token_integration(self, client):
        """Test the auth token endpoint with a real FastAPI instance."""
        response = client.post(
            "/auth/token",
            json={"username": "test_user", "password": "test_password"}
        )
        assert response.status_code == 200
        assert "access_token" in response.json()
        assert response.json()["token_type"] == "bearer"
    
    def test_auth_me_integration(self, client, token):
        """Test the auth me endpoint with a real FastAPI instance."""
        headers = {"Authorization": f"Bearer {token}"}
        response = client.get("/auth/me", headers=headers)
        assert response.status_code == 200
        assert response.json()["user_id"] == "test_user"