#!/usr/bin/env python3
"""
Main entry point for the ingestion module.

This module can be run directly to perform PDF ingestion:
    python -m ingest [--clean] [--pdf_path PATH]
"""
import sys
from pathlib import Path

# Add the parent directory to the path so we can import from the root
sys.path.insert(0, str(Path(__file__).parent.parent))

from ingest import main

if __name__ == "__main__":
    main()