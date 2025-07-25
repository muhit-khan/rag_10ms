#!/usr/bin/env python3
"""
Complete ingestion pipeline for RAG system.

This module provides the complete ingestion pipeline that stores data in ChromaDB.

Usage:
    python -m ingest [--clean] [--pdf_path PATH]
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
from ingest.text_cleaning import clean_text, save_text_to_file
from ingest.metadata_extraction import extract_metadata
from ingest.embedding import create_embeddings
from db.chroma_client import get_collection, get_chroma_client

# Configure logging
log_dir = Path("logs/ingest_logs")
log_dir.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f"logs/ingest_logs/ingest_{time.strftime('%Y%m%d_%H%M%S')}.log"),
    ],
)
logger = logging.getLogger("ingest")


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
        if not text or not text.strip():
            logger.warning(f"No text extracted from {pdf_path}")
            return [], []
        
        # Clean text
        text = clean_text(text)
        logger.info(f"Extracted and cleaned {len(text)} characters from {pdf_path}")
        
        # Save extracted & cleaned text as .txt file
        save_text_to_file(text, pdf_path.stem)
        
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


def normalize_metadata_for_chromadb(metadata: Dict):
    """
    Normalize metadata for ChromaDB compatibility.
    ChromaDB only accepts string, int, float, or bool values.
    """
    normalized = {}
    
    for key, value in metadata.items():
        if value is None:
            continue
        elif isinstance(value, (str, int, float, bool)):
            normalized[key] = value
        elif isinstance(value, list):
            # Convert lists to comma-separated strings
            normalized[key] = ", ".join(str(item) for item in value)
        else:
            # Convert everything else to string
            normalized[key] = str(value)
    
    return normalized


def store_to_chromadb(chunks: List[str], embeddings: List, metadata_list: List[Dict], ids: List[str], clean: bool = False):
    """
    Store chunks, embeddings, and metadata to ChromaDB.
    """
    logger.info("Initializing ChromaDB...")
    
    # Get ChromaDB collection
    collection = get_collection()
    
    # Clear collection if requested
    if clean:
        logger.warning("Clearing existing ChromaDB collection")
        try:
            # Get all IDs first, then delete them
            all_data = collection.get()
            if all_data and all_data.get('ids'):
                collection.delete(ids=all_data['ids'])
                logger.info(f"Deleted {len(all_data['ids'])} existing documents")
            else:
                logger.info("Collection is already empty")
        except Exception as e:
            logger.warning(f"Error clearing collection: {str(e)}")
            # Try to delete and recreate the collection
            try:
                client = get_chroma_client()
                client.delete_collection(config.CHROMA_COLLECTION)
                collection = client.get_or_create_collection(config.CHROMA_COLLECTION)
                logger.info("Recreated collection after deletion")
            except Exception as e2:
                logger.error(f"Failed to recreate collection: {str(e2)}")
    
    # Normalize metadata for ChromaDB compatibility
    logger.info("Normalizing metadata for ChromaDB...")
    normalized_metadata = [normalize_metadata_for_chromadb(metadata) for metadata in metadata_list]
    
    # Store in ChromaDB in batches
    logger.info(f"Storing {len(chunks)} chunks in ChromaDB...")
    batch_size = 100
    
    for i in range(0, len(chunks), batch_size):
        end_idx = min(i + batch_size, len(chunks))
        try:
            collection.add(
                documents=chunks[i:end_idx],
                embeddings=embeddings[i:end_idx],
                metadatas=normalized_metadata[i:end_idx],
                ids=ids[i:end_idx]
            )
            logger.info(f"Stored batch {i//batch_size + 1}/{(len(chunks)-1)//batch_size + 1}")
        except Exception as e:
            logger.error(f"Error storing batch {i//batch_size + 1}: {str(e)}")
    
    # Verify storage
    total_count = collection.count()
    logger.info(f"ChromaDB now contains {total_count} total documents")


def main():
    """Main entry point for complete ingestion pipeline."""
    args = setup_argparse()
    pdf_path = args.pdf_path or config.PDF_PATH
    
    logger.info("Starting ChromaDB ingestion pipeline...")
    logger.info(f"PDF path: {pdf_path}")
    logger.info(f"ChromaDB persist directory: {config.CHROMA_PERSIST_DIR}")
    logger.info(f"ChromaDB collection: {config.CHROMA_COLLECTION}")
    
    # Find PDF files
    pdf_files = find_pdf_files(pdf_path)
    if not pdf_files:
        logger.error(f"No PDF files found in {pdf_path}")
        return
    
    logger.info(f"Found {len(pdf_files)} PDF files to process")
    
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
    
    logger.info(f"Generated {len(all_chunks)} total chunks from {len(pdf_files)} PDFs")
    
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
    
    logger.info(f"Successfully created embeddings for {len(all_chunks)} chunks")
    
    # Store to ChromaDB
    store_to_chromadb(all_chunks, embeddings, all_metadata, all_ids, args.clean)
    logger.info("Ingestion pipeline completed successfully!")


if __name__ == "__main__":
    main()