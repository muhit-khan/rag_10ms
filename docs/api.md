# API Documentation

## Overview

The Multilingual RAG System provides a RESTful API for querying Bengali and English documents. The API is built with FastAPI and includes authentication, rate limiting, and comprehensive error handling.

## Base URL

```
http://localhost:8000
```

## Authentication

The API uses JWT (JSON Web Token) authentication. You need to obtain a token before making requests to protected endpoints.

### Generate Token

**Endpoint:** `POST /auth/token`

**Request:**

```json
{
  "username": "string",
  "password": "string"
}
```

**Response:**

```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer"
}
```

**Example:**

```bash
curl -X POST "http://localhost:8000/auth/token" \
  -H "Content-Type: application/json" \
  -d '{"username": "test", "password": "test"}'
```

### Get Current User

**Endpoint:** `GET /auth/me`

**Headers:**

```
Authorization: Bearer <your_token>
```

**Response:**

```json
{
  "user_id": "test",
  "username": "test"
}
```

## Core Endpoints

### Ask Question

**Endpoint:** `POST /ask`

**Description:** Submit a question in Bengali or English and receive an answer with source citations.

**Headers:**

```
Authorization: Bearer <your_token>
Content-Type: application/json
```

**Request:**

```json
{
  "query": "string (1-1000 characters)",
  "user_id": "string (optional, defaults to 'default')"
}
```

**Response:**

```json
{
  "answer": "string",
  "sources": [
    {
      "document": "string",
      "metadata": {
        "source": "string",
        "filename": "string",
        "chunk_id": "string",
        "language": "string"
      },
      "score": 0.85
    }
  ],
  "processing_time": 1.23
}
```

**Example - Bengali Query:**

```bash
curl -X POST "http://localhost:8000/ask" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?",
    "user_id": "user123"
  }'
```

**Example Response:**

```json
{
  "answer": "অনুপমের ভাষায় শুম্ভুনাথকে সুপুরুষ বলা হয়েছে।",
  "sources": [
    {
      "document": "অনুপম তার বন্ধু শুম্ভুনাথকে একজন সুপুরুষ হিসেবে বর্ণনা করেছেন...",
      "metadata": {
        "source": "data/raw/hsc_bangla_1st_paper.pdf",
        "filename": "hsc_bangla_1st_paper.pdf",
        "chunk_id": "42",
        "language": "bengali"
      },
      "score": 0.92
    }
  ],
  "processing_time": 1.45
}
```

**Example - English Query:**

```bash
curl -X POST "http://localhost:8000/ask" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the main theme of Bengali literature?",
    "user_id": "user123"
  }'
```

### Health Check

**Endpoint:** `GET /health`

**Description:** Check the health status of the API.

**Response:**

```json
{
  "status": "ok",
  "version": "1.0.0"
}
```

**Example:**

```bash
curl -X GET "http://localhost:8000/health"
```

### Batch Evaluation

**Endpoint:** `POST /evaluate`

**Description:** Evaluate multiple question-answer pairs for quality assessment.

**Headers:**

```
Authorization: Bearer <your_token>
Content-Type: application/json
```

**Request:**

```json
{
  "qa_pairs": [
    {
      "query": "string",
      "expected_answer": "string (optional)"
    }
  ],
  "user_id": "string (optional)"
}
```

**Response:**

```json
{
  "results": [
    {
      "query": "string",
      "answer": "string",
      "grounded": true,
      "groundedness_score": 0.85,
      "relevance_score": 0.92,
      "expected_similarity": 0.78,
      "metrics": {
        "groundedness": {
          "max_similarity": 0.85,
          "avg_similarity": 0.72,
          "has_citations": true
        },
        "relevance": {
          "query_answer_similarity": 0.92
        }
      }
    }
  ],
  "processing_time": 3.45
}
```

**Example:**

```bash
curl -X POST "http://localhost:8000/evaluate" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "qa_pairs": [
      {
        "query": "অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?",
        "expected_answer": "শুম্ভুনাথ"
      },
      {
        "query": "কাকে অনুপমের ভাগ্য দেবতা বলে উল্লেখ করা হয়েছে?",
        "expected_answer": "মামাকে"
      }
    ]
  }'
```

## Interactive Documentation

The API provides interactive documentation at:

- **Swagger UI:** `http://localhost:8000/docs`
- **ReDoc:** `http://localhost:8000/redoc`

## Rate Limiting

The API implements rate limiting to ensure fair usage:

- **Default Limit:** 20 requests per minute per IP address
- **Rate Limit Headers:** Included in responses
- **Exceeded Response:** HTTP 429 with retry information

## Error Handling

### HTTP Status Codes

| Code | Description                              |
| ---- | ---------------------------------------- |
| 200  | Success                                  |
| 400  | Bad Request - Invalid input              |
| 401  | Unauthorized - Invalid or missing token  |
| 422  | Validation Error - Request format issues |
| 429  | Too Many Requests - Rate limit exceeded  |
| 500  | Internal Server Error                    |

### Error Response Format

```json
{
  "detail": "Error description"
}
```

### Common Errors

**Invalid Token:**

```json
{
  "detail": "Could not validate credentials"
}
```

**Rate Limit Exceeded:**

```json
{
  "detail": "Rate limit exceeded. Please try again later."
}
```

**Validation Error:**

```json
{
  "detail": [
    {
      "loc": ["body", "query"],
      "msg": "ensure this value has at least 1 characters",
      "type": "value_error.any_str.min_length"
    }
  ]
}
```

## Sample Test Cases

### Bengali Literature Questions

```bash
# Test Case 1: Character identification
curl -X POST "http://localhost:8000/ask" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"query": "অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?"}'

# Test Case 2: Relationship identification
curl -X POST "http://localhost:8000/ask" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"query": "কাকে অনুপমের ভাগ্য দেবতা বলে উল্লেখ করা হয়েছে?"}'

# Test Case 3: Factual information
curl -X POST "http://localhost:8000/ask" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"query": "বিয়ের সময় কল্যাণীর প্রকৃত বয়স কত ছিল?"}'
```

### Mixed Language Queries

```bash
# Bengali-English mixed query
curl -X POST "http://localhost:8000/ask" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the meaning of অনুপম in this story?"}'

# English query about Bengali content
curl -X POST "http://localhost:8000/ask" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"query": "Who are the main characters in the Bengali story?"}'
```

## Evaluation Metrics

### Groundedness Evaluation

The system evaluates whether answers are grounded in the source documents using:

- **Cosine Similarity:** Semantic similarity between answer and sources
- **Citation Analysis:** Presence of proper source references
- **Threshold:** 0.8 cosine similarity for groundedness determination

### Relevance Evaluation

Measures how well the answer addresses the query:

- **Query-Answer Similarity:** Semantic alignment between question and response
- **Context Relevance:** Quality of retrieved document chunks

### Performance Metrics

- **Processing Time:** End-to-end response latency
- **Retrieval Accuracy:** Quality of document retrieval
- **Answer Quality:** Factual correctness and completeness

## Client Libraries

### Python Client Example

```python
import requests
import json

class RAGClient:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.token = None

    def authenticate(self, username, password):
        response = requests.post(
            f"{self.base_url}/auth/token",
            json={"username": username, "password": password}
        )
        if response.status_code == 200:
            self.token = response.json()["access_token"]
            return True
        return False

    def ask(self, query, user_id="default"):
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json"
        }
        response = requests.post(
            f"{self.base_url}/ask",
            json={"query": query, "user_id": user_id},
            headers=headers
        )
        return response.json()

    def evaluate(self, qa_pairs, user_id="default"):
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json"
        }
        response = requests.post(
            f"{self.base_url}/evaluate",
            json={"qa_pairs": qa_pairs, "user_id": user_id},
            headers=headers
        )
        return response.json()

# Usage
client = RAGClient()
client.authenticate("test", "test")

# Ask a Bengali question
result = client.ask("অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?")
print(f"Answer: {result['answer']}")
```

### JavaScript Client Example

```javascript
class RAGClient {
  constructor(baseUrl = "http://localhost:8000") {
    this.baseUrl = baseUrl;
    this.token = null;
  }

  async authenticate(username, password) {
    const response = await fetch(`${this.baseUrl}/auth/token`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ username, password }),
    });

    if (response.ok) {
      const data = await response.json();
      this.token = data.access_token;
      return true;
    }
    return false;
  }

  async ask(query, userId = "default") {
    const response = await fetch(`${this.baseUrl}/ask`, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${this.token}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ query, user_id: userId }),
    });

    return await response.json();
  }
}

// Usage
const client = new RAGClient();
await client.authenticate("test", "test");

const result = await client.ask("অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?");
console.log("Answer:", result.answer);
```

## Deployment Considerations

### Environment Variables

```bash
# Required
OPENAI_API_KEY=your_openai_api_key

# Optional (with defaults)
API_HOST=0.0.0.0
API_PORT=8000
RATE_LIMIT=20
JWT_SECRET=your_jwt_secret
REDIS_HOST=localhost
REDIS_PORT=6379
```

### Docker Deployment

```bash
# Start all services
docker-compose up -d

# Check API health
curl http://localhost:8000/health

# View logs
docker-compose logs -f api
```

### Production Settings

```bash
# Disable auto-reload
API_RELOAD=false

# Increase rate limits
RATE_LIMIT=100

# Use secure JWT secret
JWT_SECRET=your_secure_random_secret_key
```

## Monitoring and Logging

### Health Monitoring

```bash
# Basic health check
curl http://localhost:8000/health

# Detailed system status (if implemented)
curl http://localhost:8000/status
```

### Log Levels

- **INFO:** Normal operations, request processing
- **WARNING:** Non-critical issues, fallbacks
- **ERROR:** Request failures, system errors
- **DEBUG:** Detailed debugging information

### Metrics Collection

The system can be configured to export metrics to:

- **Prometheus:** For monitoring and alerting
- **LangSmith:** For LLM operation tracking
- **OpenTelemetry:** For distributed tracing

---

For more information, visit the interactive documentation at `http://localhost:8000/docs` when the server is running.
