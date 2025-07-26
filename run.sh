#!/bin/bash
# Central script to run the RAG app with all options

set -e

echo "🤖 RAG System - Select an option:"
echo "1) 🚀 Run COMPLETE PIPELINE (Ingest + Server + Chat) - RECOMMENDED"
echo "2) 📚 Run ingestion only (clean)"
echo "3) 🌐 Start API server only"
echo "4) 🧪 Run tests"
echo "5) 📊 Run evaluation"
read -p "Enter option [1-5]: " opt

case "$opt" in
  1) echo "🚀 Running COMPLETE PIPELINE..."; python complete_pipeline.py --clean ;;
  2) echo "📚 Running ingestion pipeline..."; python -m ingest --clean ;;
  3) echo "🌐 Starting FastAPI server..."; uvicorn main:app --host 0.0.0.0 --port 8000 --reload ;;
  4) echo "🧪 Running tests..."; pytest --maxfail=1 --disable-warnings ;;
  5) echo "📊 Running evaluation..."; python -c "from services.eval_service import EvalService; from tests.test_bengali_queries import BENGALI_TEST_CASES; eval_svc = EvalService('test_user'); results = eval_svc.batch_eval(BENGALI_TEST_CASES); print('Evaluation Results:'); [print(f'Query: {r[\"query\"][:50]}... | Grounded: {r[\"grounded\"]} | Score: {r[\"groundedness_score\"]:.2f}') for r in results]" ;;
  *) echo "❌ Invalid option"; exit 1 ;;
esac
