# Multilingual RAG System API Documentation

This document provides comprehensive documentation for the Multilingual RAG System API endpoints, including request/response formats, authentication, and examples.

## Base URL

```
http://localhost:8000
```

## Authentication

The API uses JWT (JSON Web Token) authentication. To access protected endpoints, you need to:

1. Obtain a token using the `/auth/token` endpoint
2. Include the token in the `Authorization` header of your requests:

```
Authorization: Bearer <your_token>
```

### Obtaining a Token

**Endpoint:** `POST /auth/token`

**Request:**

```json
{
  "username": "your_username",
  "password": "your_password"
}
```

**Response:**

```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_at": 1627484400
}
```

### Getting Current User

**Endpoint:** `GET /auth/me`

**Headers:**

```
Authorization: Bearer <your_token>
```

**Response:**

```json
{
  "user_id": "your_username",
  "disabled": false
}
```

## Endpoints

### Health Check

**Endpoint:** `GET /health`

**Description:** Check if the API is running.

**Authentication:** Not required

**Response:**

```json
{
  "status": "ok",
  "version": "1.0.0"
}
```

### Ask a Question

**Endpoint:** `POST /ask`

**Description:** Generate an answer for a given query using RAG.

**Authentication:** Required

**Request:**

```json
{
  "query": "অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?"
}
```

**Response:**

```json
{
  "answer": "অনুপমের ভাষায়, যে ব্যক্তি নিজের কাজ নিজে করে এবং অন্যের উপর নির্ভর করে না, তাকে সুপুরুষ বলা হয়েছে।",
  "sources": [
    {
      "document": "অনুপম বলেছেন, 'যে নিজের কাজ নিজে করে, অন্যের উপর নির্ভর করে না, সেই সুপুরুষ।'",
      "metadata": {
        "source": "chapter1.pdf",
        "page": "15",
        "chunk_id": "3"
      },
      "score": 0.92
    }
  ],
  "processing_time": 1.25
}
```

### Batch Evaluation

**Endpoint:** `POST /evaluate`

**Description:** Evaluate a batch of queries using the RAG system.

**Authentication:** Required

**Request:**

```json
{
  "qa_pairs": [
    {
      "query": "অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?",
      "expected_answer": "যে নিজের কাজ নিজে করে, অন্যের উপর নির্ভর করে না।"
    },
    {
      "query": "বাংলা সাহিত্যে রবীন্দ্রনাথ ঠাকুরের অবদান কী?",
      "expected_answer": "বাংলা সাহিত্যে রবীন্দ্রনাথ ঠাকুর কবিতা, গান, উপন্যাস, ছোটগল্প, নাটক ইত্যাদি বিভিন্ন ক্ষেত্রে অসামান্য অবদান রেখেছেন।"
    }
  ]
}
```

**Response:**

```json
{
  "results": [
    {
      "query": "অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?",
      "answer": "অনুপমের ভাষায়, যে ব্যক্তি নিজের কাজ নিজে করে এবং অন্যের উপর নির্ভর করে না, তাকে সুপুরুষ বলা হয়েছে।",
      "grounded": true,
      "groundedness_score": 0.85,
      "relevance_score": 0.92,
      "expected_similarity": 0.88,
      "metrics": {
        "groundedness": {
          "max_similarity": 0.92,
          "avg_similarity": 0.75,
          "has_citations": true
        },
        "relevance": {
          "query_answer_similarity": 0.92
        },
        "expected": {
          "similarity": 0.88
        }
      }
    },
    {
      "query": "বাংলা সাহিত্যে রবীন্দ্রনাথ ঠাকুরের অবদান কী?",
      "answer": "বাংলা সাহিত্যে রবীন্দ্রনাথ ঠাকুর কবিতা, গান, উপন্যাস, ছোটগল্প, নাটক ইত্যাদি বিভিন্ন ক্ষেত্রে অসামান্য অবদান রেখেছেন।",
      "grounded": true,
      "groundedness_score": 0.78,
      "relevance_score": 0.95,
      "expected_similarity": 0.96,
      "metrics": {
        "groundedness": {
          "max_similarity": 0.85,
          "avg_similarity": 0.72,
          "has_citations": true
        },
        "relevance": {
          "query_answer_similarity": 0.95
        },
        "expected": {
          "similarity": 0.96
        }
      }
    }
  ],
  "processing_time": 3.45
}
```

## Error Responses

The API returns standard HTTP status codes and error messages:

### 400 Bad Request

```json
{
  "detail": "Missing required field: query"
}
```

### 401 Unauthorized

```json
{
  "detail": "Not authenticated"
}
```

### 422 Validation Error

```json
{
  "detail": [
    {
      "loc": ["body", "query"],
      "msg": "field required",
      "type": "value_error.missing"
    }
  ]
}
```

### 429 Too Many Requests

```json
{
  "detail": "Rate limit exceeded. Please try again later."
}
```

### 500 Internal Server Error

```json
{
  "detail": "An error occurred while processing your query"
}
```

## Rate Limiting

The API has a rate limit of 20 requests per minute per IP address. If you exceed this limit, you will receive a 429 Too Many Requests response.

## Using the API with cURL

### Health Check

```bash
curl -X GET http://localhost:8000/health
```

### Authentication

```bash
curl -X POST http://localhost:8000/auth/token \
  -H "Content-Type: application/json" \
  -d '{"username": "your_username", "password": "your_password"}'
```

### Ask a Question

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d '{"query": "অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?"}'
```

### Batch Evaluation

```bash
curl -X POST http://localhost:8000/evaluate \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d '{
    "qa_pairs": [
      {
        "query": "অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?",
        "expected_answer": "যে নিজের কাজ নিজে করে, অন্যের উপর নির্ভর করে না।"
      }
    ]
  }'
```

## Using the API with Python

```python
import requests
import json

BASE_URL = "http://localhost:8000"

# Authenticate
auth_response = requests.post(
    f"{BASE_URL}/auth/token",
    json={"username": "your_username", "password": "your_password"}
)
token = auth_response.json()["access_token"]

# Set headers with token
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {token}"
}

# Ask a question
ask_response = requests.post(
    f"{BASE_URL}/ask",
    headers=headers,
    json={"query": "অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?"}
)

print(json.dumps(ask_response.json(), indent=2, ensure_ascii=False))
```
