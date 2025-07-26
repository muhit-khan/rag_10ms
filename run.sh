#!/bin/bash
# Central script to run the RAG app with all options

set -e

echo "Select an option:"
echo "1) 🚀 Run COMPLETE PIPELINE (Ingest + Server + Chat) - RECOMMENDED"
echo "2) Run main.py (central entry point)"
echo "3) Start FastAPI server"
echo "4) Run ingestion pipeline"
echo "5) Run Clean ingestion pipeline"
echo "6) Run tests"
echo "7) Run evaluation harness"
echo "8) Start RAG Chat Interface"
read -p "Enter option [1-8]: " opt

case "$opt" in
  1) echo "🚀 Running COMPLETE PIPELINE..."; python complete_pipeline.py ;;
  2) echo "Running main.py..."; python main.py ;;
  3) echo "Starting FastAPI server..."; uvicorn main:app --host 0.0.0.0 --port 8000 --reload ;;
  4) echo "Running ingestion pipeline..."; python -m ingest ;;
  5) echo "Running ingestion pipeline..."; python -m ingest --clean ;;
  6) echo "Running tests..."; pytest --maxfail=1 --disable-warnings ;;
  7) echo "Running evaluation harness..."; python services/eval_service.py ;;
  8) echo "Starting RAG Chat Interface..."; uvicorn main:app --host 0.0.0.0 --port 8000 --reload; echo "Open http://localhost:8000/chat" ;;
  *) echo "Invalid option"; exit 1 ;;
esac
