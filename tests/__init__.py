"""Test module for RAG system"""

# Test configuration
import os
import sys

# Add project root to Python path for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

__all__ = ["test_ingestion", "test_retrieval", "test_multilingual"]
