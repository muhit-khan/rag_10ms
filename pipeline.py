#!/usr/bin/env python3
"""
Unified entry point for the complete RAG pipeline.

This script provides a single command to run the complete pipeline:
1. Ingest PDFs into ChromaDB
2. Start the FastAPI server
3. Open the chat interface

Usage:
    python pipeline.py [--clean] [--pdf_path PATH] [--port PORT]
"""
import argparse
import logging
import os
import subprocess
import sys
import time
import webbrowser
from pathlib import Path
from threading import Thread

from config import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f"pipeline_{time.strftime('%Y%m%d_%H%M%S')}.log"),
    ],
)
logger = logging.getLogger("pipeline")


def run_ingestion(pdf_path: str, clean: bool = False):
    """Run the ingestion pipeline."""
    logger.info("=" * 60)
    logger.info("STEP 1: Running PDF Ingestion Pipeline")
    logger.info("=" * 60)
    
    try:
        cmd = [sys.executable, "-m", "ingest"]
        if clean:
            cmd.append("--clean")
        if pdf_path:
            cmd.extend(["--pdf_path", pdf_path])
        
        logger.info(f"Running command: {' '.join(cmd)}")
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        
        logger.info("Ingestion completed successfully!")
        logger.info("Ingestion output:")
        for line in result.stdout.split('\n'):
            if line.strip():
                logger.info(f"  {line}")
        
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Ingestion failed with exit code {e.returncode}")
        logger.error(f"Error output: {e.stderr}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error during ingestion: {str(e)}")
        return False


def start_server(port: int = 8000):
    """Start the FastAPI server."""
    logger.info("=" * 60)
    logger.info("STEP 2: Starting FastAPI Server")
    logger.info("=" * 60)
    
    try:
        cmd = [
            sys.executable, "-m", "uvicorn", 
            "main:app", 
            "--host", "0.0.0.0", 
            "--port", str(port), 
            "--reload"
        ]
        
        logger.info(f"Running command: {' '.join(cmd)}")
        logger.info(f"Server will be available at: http://localhost:{port}")
        logger.info(f"Chat interface will be available at: http://localhost:{port}/chat")
        
        # Start server in a separate process
        process = subprocess.Popen(cmd)
        
        # Wait a bit for server to start
        time.sleep(3)
        
        return process
        
    except Exception as e:
        logger.error(f"Failed to start server: {str(e)}")
        return None


def open_chat_interface(port: int = 8000, delay: int = 5):
    """Open the chat interface in the default browser."""
    def delayed_open():
        time.sleep(delay)
        chat_url = f"http://localhost:{port}/chat"
        logger.info("=" * 60)
        logger.info("STEP 3: Opening Chat Interface")
        logger.info("=" * 60)
        logger.info(f"Opening chat interface at: {chat_url}")
        
        try:
            webbrowser.open(chat_url)
            logger.info("Chat interface opened in your default browser!")
        except Exception as e:
            logger.error(f"Failed to open browser: {str(e)}")
            logger.info(f"Please manually open: {chat_url}")
    
    # Start in a separate thread
    thread = Thread(target=delayed_open)
    thread.daemon = True
    thread.start()


def check_prerequisites():
    """Check if all prerequisites are met."""
    logger.info("Checking prerequisites...")
    
    # Check if OpenAI API key is set
    if not config.OPENAI_API_KEY or config.OPENAI_API_KEY.strip() == "":
        logger.error("OpenAI API key is not set in .env file!")
        return False
    
    # Check if PDF directory exists
    pdf_path = Path(config.PDF_PATH)
    if not pdf_path.exists():
        logger.warning(f"PDF directory {pdf_path} does not exist. Creating it...")
        pdf_path.mkdir(parents=True, exist_ok=True)
    
    # Check if there are PDF files
    pdf_files = list(pdf_path.glob("*.pdf"))
    if not pdf_files:
        logger.warning(f"No PDF files found in {pdf_path}")
        logger.warning("Please add some PDF files to the directory before running the pipeline.")
        return False
    
    logger.info(f"Found {len(pdf_files)} PDF files in {pdf_path}")
    
    # Check if required directories exist
    chroma_dir = Path(config.CHROMA_PERSIST_DIR)
    chroma_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("All prerequisites checked!")
    return True


def setup_argparse():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Unified RAG Pipeline - Complete workflow from PDF ingestion to chat interface",
        epilog="Example: python pipeline.py --clean --pdf_path data/raw/ --port 8000"
    )
    
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Clear existing ChromaDB collection before ingestion",
    )
    parser.add_argument(
        "--pdf_path",
        type=str,
        default=config.PDF_PATH,
        help=f"Path to PDF directory (default: {config.PDF_PATH})",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for the FastAPI server (default: 8000)",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Don't automatically open the chat interface in browser",
    )
    parser.add_argument(
        "--ingestion-only",
        action="store_true",
        help="Run only the ingestion pipeline, don't start server",
    )
    
    return parser.parse_args()


def main():
    """Main entry point for the unified pipeline."""
    print("=" * 80)
    print("🚀 RAG SYSTEM - UNIFIED PIPELINE")
    print("=" * 80)
    print("This script will:")
    print("1. 📚 Ingest PDF files into ChromaDB")
    print("2. 🌐 Start the FastAPI server")
    print("3. 💬 Open the chat interface in your browser")
    print("=" * 80)
    
    args = setup_argparse()
    
    # Check prerequisites
    if not check_prerequisites():
        logger.error("Prerequisites check failed. Please fix the issues above.")
        sys.exit(1)
    
    try:
        # Step 1: Run ingestion
        success = run_ingestion(args.pdf_path, args.clean)
        if not success:
            logger.error("Ingestion failed. Stopping pipeline.")
            sys.exit(1)
        
        if args.ingestion_only:
            logger.info("Ingestion completed. Exiting as requested.")
            return
        
        # Step 2: Start server
        server_process = start_server(args.port)
        if not server_process:
            logger.error("Failed to start server. Stopping pipeline.")
            sys.exit(1)
        
        # Step 3: Open chat interface
        if not args.no_browser:
            open_chat_interface(args.port)
        
        # Keep the script running
        logger.info("=" * 60)
        logger.info("🎉 PIPELINE STARTED SUCCESSFULLY!")
        logger.info("=" * 60)
        logger.info(f"📊 API Documentation: http://localhost:{args.port}/docs")
        logger.info(f"💬 Chat Interface: http://localhost:{args.port}/chat")
        logger.info("=" * 60)
        logger.info("Press Ctrl+C to stop the server...")
        
        try:
            server_process.wait()
        except KeyboardInterrupt:
            logger.info("\nShutting down server...")
            server_process.terminate()
            server_process.wait()
            logger.info("Server stopped. Goodbye!")
    
    except Exception as e:
        logger.error(f"Pipeline failed with error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()