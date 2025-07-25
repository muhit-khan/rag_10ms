"""
Basic health endpoint test
"""
from fastapi.testclient import TestClient
from fastapi import FastAPI
from api.routers import router

app = FastAPI()
app.include_router(router)

def test_health():
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
