#!/usr/bin/env python3
"""
Simplified ingestion pipeline for RAG system.

This script provides a simplified version of the ingestion pipeline
that doesn't rely on ChromaDB or LangChain to avoid pydantic version conflicts.

Usage:
    python ingest/simple_run.py [--clean] [--pdf_path PATH]

Options:
    --clean     Clear existing collection before ingestion
    --pdf_path  Override the default PDF path from config
"""
import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from openai import OpenAI
from tqdm import tqdm

from config import config
from ingest.chunk_loader import chunk_text
from ingest.extract_text import extract_text_from_pdf
from ingest.pdf_discovery import find_pdf_files
from ingest.text_cleaning import clean_text
from ingest.metadata_extraction import extract_metadata
from ingest.embedding import create_embeddings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f"simple_ingest_{time.strftime('%Y%m%d_%H%M%S')}.log"),
    ],
)
logger = logging.getLogger("simple_ingest")


def setup_argparse() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Simplified PDF ingestion")
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Clear existing collection before ingestion",
    )
    parser.add_argument(
        "--pdf_path",
        type=str,
        help=f"Path to PDF directory (default: {config.PDF_PATH})",
    )
    return parser.parse_args()


def process_pdf(pdf_path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    """
    Process a single PDF file.
    
    Returns chunks and their metadata.
    """
    logger.info(f"Processing {pdf_path}")
    
    try:
        # Extract text
        text = extract_text_from_pdf(str(pdf_path))
        if not text.strip():
            logger.warning(f"No text extracted from {pdf_path}")
            return [], []
        
        # Clean text
        text = clean_text(text)
        logger.info(f"Extracted and cleaned {len(text)} characters from {pdf_path}")
        
        # Extract metadata
        base_metadata = extract_metadata(pdf_path, text)
        
        # Chunk text
        chunks = chunk_text(text)
        logger.info(f"Created {len(chunks)} chunks from {pdf_path}")
        
        # Create metadata for each chunk
        metadata_list = []
        for i, chunk in enumerate(chunks):
            chunk_metadata = base_metadata.copy()
            chunk_metadata["chunk_id"] = str(i)
            chunk_metadata["chunk_index"] = str(i)
            chunk_metadata["chunk_total"] = str(len(chunks))
            metadata_list.append(chunk_metadata)
        
        return chunks, metadata_list
    
    except Exception as e:
        logger.error(f"Error processing {pdf_path}: {str(e)}")
        return [], []


def store_to_json(chunks: List[str], embeddings: List, metadata_list: List[Dict], ids: List[str], output_dir: str):
    """
    Store chunks, embeddings, and metadata to JSON files.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Store chunks and metadata
    data = []
    for i, (chunk, metadata, chunk_id) in enumerate(zip(chunks, metadata_list, ids)):
        item = {
            "id": chunk_id,
            "chunk": chunk,
            "metadata": metadata,
            "embedding": embeddings[i] if i < len(embeddings) and embeddings[i] is not None else None
        }
        data.append(item)
    
    # Write to JSON file
    output_file = output_path / "chunks.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Stored {len(data)} chunks to {output_file}")


def main():
    """Main entry point for simplified ingestion pipeline."""
    args = setup_argparse()
    pdf_path = args.pdf_path or config.PDF_PATH
    
    # Create output directory
    output_dir = Path("data/processed/simple")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Clear existing data if requested
    if args.clean:
        logger.warning("Clearing existing data")
        for file in output_dir.glob("*.json"):
            file.unlink()
    
    # Find PDF files
    pdf_files = find_pdf_files(pdf_path)
    if not pdf_files:
        logger.error(f"No PDF files found in {pdf_path}")
        return
    
    # Process each PDF
    all_chunks = []
    all_metadata = []
    all_ids = []
    
    for pdf_file in tqdm(pdf_files, desc="Processing PDFs"):
        chunks, metadata_list = process_pdf(pdf_file)
        
        if not chunks:
            continue
        
        # Generate IDs for chunks
        ids = [f"{pdf_file.stem}_{i}" for i in range(len(chunks))]
        
        all_chunks.extend(chunks)
        all_metadata.extend(metadata_list)
        all_ids.extend(ids)
    
    if not all_chunks:
        logger.error("No chunks generated from any PDF")
        return
    
    # Create embeddings
    logger.info(f"Creating embeddings for {len(all_chunks)} chunks")
    embeddings = create_embeddings(all_chunks)
    
    # Filter out chunks with failed embeddings
    valid_indices = [i for i, emb in enumerate(embeddings) if emb is not None]
    if len(valid_indices) < len(embeddings):
        logger.warning(f"Filtering out {len(embeddings) - len(valid_indices)} chunks with failed embeddings")
        all_chunks = [all_chunks[i] for i in valid_indices]
        all_metadata = [all_metadata[i] for i in valid_indices]
        all_ids = [all_ids[i] for i in valid_indices]
        embeddings = [embeddings[i] for i in valid_indices]
    
    # Store to JSON
    store_to_json(all_chunks, embeddings, all_metadata, all_ids, str(output_dir))
    logger.info("Ingestion complete")


if __name__ == "__main__":
    main()