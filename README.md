# Multilingual RAG System for Bengali Literature

A Retrieval-Augmented Generation (RAG) system designed to answer questions about Bengali literature, specifically the HSC26 Bangla 1st paper textbook. The system handles queries in both Bengali and English.

## 🎯 Features

- **Multilingual Support**: Handles Bengali (বাংলা) and English queries
- **PDF Processing**: Extracts and processes Bengali text from PDF documents
- **Vector Search**: Semantic search using multilingual embeddings
- **Conversational Memory**: Maintains short-term chat history and long-term document knowledge
- **REST API**: FastAPI-based web interface
- **Evaluation Metrics**: Built-in evaluation for groundedness and relevance

## 📁 Project Structure

```
rag_10ms/
├── config/
│   └── __init__.py          # Configuration settings
├── src/
│   ├── pdf_processor.py     # PDF text extraction
│   ├── text_cleaner.py      # Bengali text preprocessing
│   ├── chunking.py          # Document chunking strategies
│   ├── embeddings.py        # Multilingual embeddings & vector store
│   ├── retrieval.py         # Document retrieval logic
│   ├── llm_client.py        # LLM integration & conversation memory
│   └── rag_pipeline.py      # Main RAG orchestration
├── api/
│   └── main.py              # FastAPI web interface
├── evaluation/
│   └── metrics.py           # RAG evaluation metrics
├── tests/                   # Unit tests
├── utils/                   # Helper utilities
├── data/
│   ├── raw/                 # Place PDF files here
│   └── processed/           # Generated embeddings & chunks
└── main.py                  # CLI interface
```

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Activate (Linux/Mac)
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Environment Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit .env file with your settings
# Add OpenAI API key if using GPT models
OPENAI_API_KEY=your_api_key_here
```

### 3. Add Your PDF Document

Place the HSC26 Bangla 1st paper PDF in the `data/raw/` directory:

```
data/raw/hsc26_bangla_1st_paper.pdf
```

### 4. Run the System

#### CLI Interface (Recommended for testing):

```bash
python main.py
```

#### Web API:

```bash
# Start the API server
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# Access API documentation
# http://localhost:8000/docs
```

## 🧪 Sample Queries

Test the system with these sample questions:

**Bengali:**

- অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?
- কাকে অনুপমের ভাগ্য দেবতা বলে উল্লেখ করা হয়েছে?
- বিয়ের সময় কল্যাণীর প্রকৃত বয়স কত ছিল?

**English:**

- Who is described as a good person in Anupam's language?
- Who is mentioned as Anupam's fortune deity?

## 🔧 API Usage

### Chat Endpoint

```bash
curl -X POST "http://localhost:8000/chat" \
     -H "Content-Type: application/json" \
     -d '{
       "query": "অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?",
       "session_id": "user123"
     }'
```

### Response Format

```json
{
  "answer": "শুম্ভুনাথ",
  "sources": [
    {
      "chunk_id": "chunk_0001",
      "page": 15,
      "language": "bn",
      "similarity_score": 0.87
    }
  ],
  "session_id": "user123",
  "language_detected": "bn"
}
```

### Other Endpoints

- `GET /health` - Health check
- `GET /stats` - System statistics
- `POST /upload-document` - Add new document
- `DELETE /sessions/{session_id}` - Clear chat history

## 🛠️ Tools & Libraries Used

### Core Framework

- **FastAPI**: Modern Python web framework for APIs
- **Pydantic**: Data validation and settings management
- **Uvicorn**: ASGI server for FastAPI

### PDF Processing

- **PyMuPDF (fitz)**: Primary PDF text extraction (better Bengali font support)
- **pdfplumber**: Fallback PDF processing
- **pdfminer**: Alternative PDF parsing

### Text Processing & NLP

- **sentence-transformers**: Multilingual embeddings
  - Model: `paraphrase-multilingual-MiniLM-L12-v2`
- **NLTK**: Natural language processing utilities
- **bnlp-toolkit**: Bengali language processing
- **indic-nlp-library**: Indic languages support

### Vector Storage & Retrieval

- **ChromaDB**: Primary vector database
- **FAISS**: Alternative vector search
- **NumPy**: Numerical computations
- **scikit-learn**: ML utilities and metrics

### LLM Integration

- **OpenAI API**: GPT models for response generation
- **langchain**: LLM orchestration framework

### Development & Testing

- **pytest**: Testing framework
- **black**: Code formatting
- **python-dotenv**: Environment management

## 📊 Technical Implementation Details

### 1. PDF Text Extraction

**Method**: PyMuPDF (primary) with pdfplumber fallback
**Challenge**: Bengali fonts often require specialized handling
**Solution**: Multiple extraction methods with font-aware processing

### 2. Chunking Strategy

**Approach**: Sentence-based chunking with overlap

- **Chunk size**: 512 words
- **Overlap**: 50 words
- **Rationale**: Maintains semantic coherence while ensuring retrievable context

### 3. Embedding Model

**Model**: `paraphrase-multilingual-MiniLM-L12-v2`
**Why**:

- Multilingual support (Bengali + English)
- Good performance on semantic similarity
- Reasonable size (384 dimensions)
- Proven performance on cross-lingual tasks

### 4. Vector Storage & Similarity

**Method**: ChromaDB with cosine similarity
**Storage**: Persistent local database
**Search**: Top-k retrieval with similarity threshold (0.7)

### 5. Context Handling

**Short-term**: Conversation history (last 10 exchanges)
**Long-term**: Document chunks in vector database
**Memory management**: Session-based isolation

## 📈 Evaluation Metrics

### Groundedness

- Measures if answers are supported by retrieved context
- Method: Word overlap analysis between answer and source chunks
- Threshold: 30% minimum overlap

### Relevance

- Evaluates quality of retrieved documents
- Metrics: Average similarity scores, high-relevance chunk count
- Threshold: 0.5 similarity for high relevance

### Answer Quality

- Heuristic evaluation of response quality
- Factors: Length appropriateness, language consistency, completeness

### Sample Evaluation

```python
from evaluation import RAGEvaluator

evaluator = RAGEvaluator()
result = evaluator.comprehensive_evaluation(
    query="অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?",
    answer="শুম্ভুনাথ",
    retrieved_contexts=[...],
    similarity_scores=[0.85, 0.72, 0.68],
    language="bn"
)
```

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test category
pytest tests/test_multilingual.py -v

# Run with coverage
pytest --cov=src tests/
```

## 📝 Questions & Answers (Technical Assessment)

### Q1: Text Extraction Method

**Method Used**: PyMuPDF (fitz) as primary method with pdfplumber as fallback

**Why**:

- Better handling of Bengali Unicode characters
- Superior font rendering for complex scripts
- More reliable text positioning

**Challenges Faced**:

- OCR errors in Bengali text required correction patterns
- Font encoding issues needed specialized handling
- Page layout preservation for context

### Q2: Chunking Strategy

**Strategy**: Sentence-based chunking with 50-word overlap

**Why**:

- Maintains semantic coherence better than character-based
- Bengali sentence delimiters (।!?) properly handled
- Overlap ensures context continuity across chunks
- 512-word chunks provide sufficient context for LLM

### Q3: Embedding Model

**Model**: `paraphrase-multilingual-MiniLM-L12-v2`

**Why**:

- Excellent cross-lingual performance (Bengali-English)
- Trained on paraphrase data, good for Q&A tasks
- Balanced size (384D) - performance vs efficiency
- Strong semantic understanding for literature content

### Q4: Similarity & Storage

**Method**: ChromaDB with cosine similarity

**Why**:

- Native support for metadata filtering
- Persistent storage with easy querying
- Cosine similarity works well with normalized embeddings
- Built-in support for adding/updating documents

### Q5: Query-Document Matching

**Approach**:

- Query normalization (whitespace, punctuation)
- Multilingual embedding in same vector space
- Top-k retrieval with re-ranking
- Language preference filtering when needed

**Handling Vague Queries**:

- Conversation context expansion
- Similarity threshold filtering
- Fallback responses for low-confidence matches

### Q6: Result Quality & Improvements

**Current Performance**: Good for direct factual questions

**Improvements Needed**:

- Better chunking for complex narratives
- Fine-tuned embedding model on Bengali literature
- Larger document corpus for better coverage
- Advanced re-ranking with cross-encoder models

## 🚧 Future Enhancements

1. **Fine-tuned Embedding Models**: Train on Bengali literature corpus
2. **Advanced Chunking**: Paragraph-aware semantic chunking
3. **Multi-document Support**: Handle multiple textbooks
4. **Better Evaluation**: Human evaluation with domain experts
5. **Caching**: Redis for faster repeated queries
6. **Monitoring**: Logging and analytics dashboard

## 📄 License

This project is developed for educational purposes as part of a technical assessment.

## 🤝 Contributing

This is an assessment project. For questions or suggestions, please contact the developer.

---

**Note**: This system is designed specifically for the HSC26 Bangla 1st paper textbook and optimized for Bengali literature questions. Performance may vary with other document types or domains.
