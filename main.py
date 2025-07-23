"""
Main entry point for RAG system
"""
from fastapi import FastAPI
from api.routers import router
import uvicorn
from config import config

def create_app():
    app = FastAPI(title="Multilingual RAG System")
    app.include_router(router)
    return app


app = create_app()

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=config.API_HOST,
        port=config.API_PORT,
        reload=True
    )
