#!/bin/bash
# Central script to run the RAG app with all options

set -e

echo "Select an option:"
echo "1) Run main.py (central entry point)"
echo "2) Start FastAPI server"
echo "3) Run ingestion pipeline"
echo "4) Run Clean ingestion pipeline"
echo "5) Run tests"
echo "6) Run evaluation harness"
echo "7) Start RAG Chat Interface"
read -p "Enter option [1-7]: " opt

case "$opt" in
  1) echo "Running main.py..."; python main.py ;;
  2) echo "Starting FastAPI server..."; uvicorn main:app --host 0.0.0.0 --port 8000 --reload ;;
  3) echo "Running ingestion pipeline..."; python -m ingest ;;
  4) echo "Running ingestion pipeline..."; python -m ingest --clean ;;
  5) echo "Running tests..."; pytest --maxfail=1 --disable-warnings ;;
  6) echo "Running evaluation harness..."; python services/eval_service.py ;;
  7) echo "Starting RAG Chat Interface..."; uvicorn main:app --host 0.0.0.0 --port 8000 --reload; echo "Open http://localhost:8000/chat" ;;
  *) echo "Invalid option"; exit 1 ;;
esac
