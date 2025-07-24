#!/bin/bash
# Central script to run the RAG app with all options

set -e

usage() {
  echo "Usage: $0 [main|api|ingest|test|eval]"
  echo "  main   - Run main.py (central entry point)"
  echo "  api    - Start FastAPI server (default)"
  echo "  ingest - Run ingestion pipeline (PDF to ChromaDB)"
  echo "  test   - Run all tests"
  echo "  eval   - Run evaluation harness"
  exit 1
}

if [ $# -eq 0 ]; then
  CMD="api"
else
  CMD="$1"
fi

case "$CMD" in
  main)
    echo "Running main.py (central entry point)..."
    python main.py
    ;;
  api)
    echo "Starting FastAPI server..."
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload
    ;;
  ingest)
    echo "Running ingestion pipeline..."
    python -m ingest
    ;;
  test)
    echo "Running tests..."
    pytest --maxfail=1 --disable-warnings
    ;;
  eval)
    echo "Running evaluation harness..."
    python services/eval_service.py
    ;;
  *)
    usage
    ;;
esac
