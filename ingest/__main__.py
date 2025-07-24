#!/usr/bin/env python3
"""
Main entry point for the ingest package when run as a module.

This file enables running the ingest package with:
    python -m ingest [--clean] [--pdf_path PATH]

This is the proper Python way to make a package executable.
"""
from ingest import main

if __name__ == "__main__":
    main()