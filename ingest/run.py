#!/usr/bin/env python3
"""
Ingestion pipeline for RAG system.

This script orchestrates the full ingestion process:
1. Find all PDF files in the configured directory
2. Extract text from each PDF
3. Clean and normalize the text
4. Split text into chunks
5. Generate embeddings for each chunk
6. Store chunks and embeddings in ChromaDB with metadata

Usage:
    python ingest/run.py [--clean] [--pdf_path PATH]

Options:
    --clean     Clear existing collection before ingestion
    --pdf_path  Override the default PDF path from config
"""
import argparse
import glob
import logging
import os
import re
import sys
import time
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from openai import OpenAI
from tqdm import tqdm

from config import config
from db.chroma_client import get_collection
from ingest.chunk_loader import chunk_text
from ingest.extract_text import extract_text_from_pdf

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f"ingest_{time.strftime('%Y%m%d_%H%M%S')}.log"),
    ],
)
logger = logging.getLogger("ingest")


def setup_argparse() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Ingest PDFs into ChromaDB")
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


def clean_text(text: str) -> str:
    """
    Clean and normalize text.
    
    - Normalize Unicode (NFKC)
    - Remove headers and footers
    - Remove page numbers
    - Remove excessive whitespace
    """
    # Normalize Unicode
    text = unicodedata.normalize("NFKC", text)
    
    # Remove headers/footers (simplified approach - adjust as needed)
    lines = text.split("\n")
    cleaned_lines = []
    for line in lines:
        # Skip page numbers
        if re.match(r"^\s*\d+\s*$", line):
            continue
        # Skip headers/footers (adjust patterns as needed)
        if re.match(r"^\s*(header|footer|www\.|http|copyright)", line.lower()):
            continue
        cleaned_lines.append(line)
    
    # Join lines and remove excessive whitespace
    text = "\n".join(cleaned_lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)
    
    return text.strip()


def extract_metadata(pdf_path: Path, text: str) -> Dict[str, str]:
    """
    Extract metadata from PDF path and content.
    
    Attempts to identify chapter, section, and page information.
    Falls back to filename-based metadata if extraction fails.
    """
    metadata = {
        "source": pdf_path.name,
        "path": str(pdf_path),
    }
    
    # Extract chapter/section from filename (assuming format like "chapter1.pdf" or "section2_3.pdf")
    filename = pdf_path.stem
    chapter_match = re.search(r"chapter[_-]?(\d+)", filename, re.IGNORECASE)
    if chapter_match:
        metadata["chapter"] = chapter_match.group(1)
    
    section_match = re.search(r"section[_-]?(\d+)", filename, re.IGNORECASE)
    if section_match:
        metadata["section"] = section_match.group(1)
    
    # Try to extract page numbers from text (simplified approach)
    page_matches = re.findall(r"page\s+(\d+)", text, re.IGNORECASE)
    if page_matches:
        metadata["page"] = page_matches[0]
    
    return metadata


def create_embeddings(chunks: List[str], batch_size: int = 96) -> List:
    """
    Create embeddings for text chunks using OpenAI API.
    
    Processes chunks in batches to optimize API usage.
    Returns a list of embeddings compatible with ChromaDB.
    """
    client = OpenAI(api_key=config.OPENAI_API_KEY)
    embeddings = []
    
    # Process in batches
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        try:
            response = client.embeddings.create(
                input=batch,
                model=config.EMBEDDING_MODEL
            )
            batch_embeddings = [item.embedding for item in response.data]
            embeddings.extend(batch_embeddings)
            logger.info(f"Created embeddings for batch {i//batch_size + 1}/{(len(chunks)-1)//batch_size + 1}")
        except Exception as e:
            logger.error(f"Error creating embeddings for batch {i//batch_size + 1}: {str(e)}")
            # Return None for failed embeddings to maintain alignment
            embeddings.extend([None] * len(batch))
    
    return embeddings


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


def main():
    """Main entry point for ingestion pipeline."""
    args = setup_argparse()
    pdf_path = args.pdf_path or config.PDF_PATH
    
    # Get ChromaDB collection
    collection = get_collection()
    
    # Clear collection if requested
    if args.clean:
        logger.warning("Clearing existing collection")
        collection.delete(where={})
    
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
    
    # Store in ChromaDB
    logger.info(f"Storing {len(all_chunks)} chunks in ChromaDB")
    
    # Process in batches to avoid memory issues
    batch_size = 100
    for i in range(0, len(all_chunks), batch_size):
        end_idx = min(i + batch_size, len(all_chunks))
        try:
            # Use add method instead of upsert for better type compatibility
            collection.add(
                documents=all_chunks[i:end_idx],
                embeddings=embeddings[i:end_idx],
                metadatas=all_metadata[i:end_idx],
                ids=all_ids[i:end_idx]
            )
            logger.info(f"Stored batch {i//batch_size + 1}/{(len(all_chunks)-1)//batch_size + 1}")
        except Exception as e:
            logger.error(f"Error storing batch {i//batch_size + 1}: {str(e)}")
    
    logger.info("Ingestion complete")
    logger.info(f"Total documents: {collection.count()}")


if __name__ == "__main__":
    main()