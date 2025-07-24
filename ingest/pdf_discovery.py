"""
PDF discovery logic
"""
from pathlib import Path
from typing import List
import logging

logger = logging.getLogger("ingest.pdf_discovery")

def find_pdf_files(pdf_path: str) -> List[Path]:
    """Find all PDF files in the given directory."""
    pdf_dir = Path(pdf_path)
    if not pdf_dir.exists():
        logger.error(f"PDF directory not found: {pdf_dir}")
        pdf_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created PDF directory: {pdf_dir}")
        return []

    pdf_files = list(pdf_dir.glob("**/*.pdf"))
    logger.info(f"Found {len(pdf_files)} PDF files in {pdf_dir}")
    return pdf_files
