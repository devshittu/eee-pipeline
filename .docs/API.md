
# REST API Documentation

Complete reference for all EEE Pipeline REST endpoints.

## Base URL

```
http://localhost:8000
```

## Authentication

Currently, the API does not require authentication. For production deployments, implement API keys or OAuth2.

---

## Table of Contents

1. [Health & Status Endpoints](#health--status-endpoints)
2. [Document Processing Endpoints](#document-processing-endpoints)
3. [Batch Processing Endpoints](#batch-processing-endpoints)
4. [Job Management Endpoints](#job-management-endpoints)
5. [Microservice APIs](#microservice-apis)

---

## Health & Status Endpoints

### GET /health

Check overall system health and dependency status.

**Response 200 OK**:

```json
{
  "status": "ok",
  "dependencies_ok": true,
  "ner_service": {"status": "healthy"},
  "dp_service": {"status": "healthy"},
  "event_llm_service": {"status": "healthy"},
  "celery_broker_reachable": true
}
```

**Example**:

```bash
curl http://localhost:8000/health
```

---

## Document Processing Endpoints

### POST /v1/documents

Process a single text document synchronously (simple mode).

**⚠️ Use Case**: Short texts (< 500 characters) that require fast response.

**Request Body**:

```json
{
  "text": "Apple CEO Tim Cook announced iPhone 15 at Apple Park yesterday."
}
```

**Response 200 OK**:

```json
{
  "events": [
    {
      "event_type": "product_announcement",
      "trigger": {
        "text": "announced",
        "start_char": 19,
        "end_char": 28
      },
      "arguments": [
        {
          "argument_role": "agent",
          "entity": {
            "text": "Tim Cook",
            "type": "PER",
            "start_char": 10,
            "end_char": 18
          }
        },
        {
          "argument_role": "product",
          "entity": {
            "text": "iPhone 15",
            "type": "PRODUCT",
            "start_char": 29,
            "end_char": 38
          }
        }
      ],
      "metadata": {
        "sentiment": "neutral",
        "causality": "Tim Cook announced the iPhone 15 product launch."
      }
    }
  ],
  "extracted_entities": [
    {
      "text": "Apple",
      "type": "ORG",
      "start_char": 0,
      "end_char": 5
    },
    {
      "text": "Tim Cook",
      "type": "PER",
      "start_char": 10,
      "end_char": 18
    }
  ],
  "extracted_soa_triplets": [
    {
      "subject": {"text": "Tim Cook", "start_char": 10, "end_char": 18},
      "action": {"text": "announced", "start_char": 19, "end_char": 28},
      "object": {"text": "iPhone 15", "start_char": 29, "end_char": 38}
    }
  ],
  "original_text": "Apple CEO Tim Cook announced iPhone 15 at Apple Park yesterday.",
  "job_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

**Example**:

```bash
curl -X POST http://localhost:8000/v1/documents \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Tesla opened a new Gigafactory in Berlin on March 22, 2024."
  }'
```

**Error Responses**:

**400 Bad Request** - Invalid input:

```json
{
  "detail": "Invalid data format: text field is required"
}
```

**503 Service Unavailable** - Downstream service error:

```json
{
  "detail": "Could not reach upstream service: http://ner-service:8001/predict"
}
```

---

### POST /v1/documents/enriched

Submit an enriched document for asynchronous processing.

**⚠️ Use Case**: Documents with metadata (title, author, date, etc.) requiring LLM processing.

**⏱️ Processing Time**: 5-10 minutes per document.

**Request Body** (flexible schema):

```json
{
  "document_id": "article_12345",
  "cleaned_text": "Breaking News! Dr. Evelyn Reed presented groundbreaking research on quantum entanglement at the International Science Conference in Geneva last Friday.",
  "cleaned_title": "Quantum Breakthrough at Science Conference",
  "cleaned_author": "Science Daily Reporter",
  "cleaned_publication_date": "2025-07-04",
  "cleaned_source_url": "http://example.com/news/quantum-breakthrough",
  "entities": [
    {
      "text": "Evelyn Reed",
      "type": "PERSON",
      "start_char": 19,
      "end_char": 30
    }
  ]
}
```

**Response 202 ACCEPTED**:

```json
{
  "job_id": "abc-123-def-456",
  "status": "SUBMITTED",
  "message": "Document submitted for async processing. Results will be available in 5-10 minutes.",
  "status_endpoint": "/v1/jobs/abc-123-def-456",
  "polling_command": "curl http://localhost:8000/v1/jobs/abc-123-def-456",
  "estimated_completion_minutes": 7
}
```

**Example**:

```bash
curl -X POST http://localhost:8000/v1/documents/enriched \
  -H "Content-Type: application/json" \
  -d @data/enriched_doc.json
```

---

### POST /v1/documents/enriched/upload

Upload a JSON or JSONL file with enriched documents.

**⚠️ Supported Formats**:

- **JSON**: Single document
- **JSONL**: Multiple documents (one per line)

**Request**:

```bash
curl -X POST http://localhost:8000/v1/documents/enriched/upload \
  -F "file=@data/articles.jsonl"
```

**Response 202 ACCEPTED**:

```json
{
  "job_id": "batch-xyz-789",
  "status": "SUBMITTED",
  "documents_count": 50,
  "message": "Submitted 50 document(s) for async processing.",
  "status_endpoint": "/v1/jobs/batch-xyz-789",
  "polling_command": "curl http://localhost:8000/v1/jobs/batch-xyz-789",
  "estimated_completion_minutes": 350
}
```

**JSONL Example** (`articles.jsonl`):

```jsonl
{"document_id": "doc1", "cleaned_text": "Apple released iOS 18.", "cleaned_title": "iOS 18 Launch"}
{"document_id": "doc2", "cleaned_text": "Tesla opened Berlin factory.", "cleaned_title": "Tesla Expansion"}
{"document_id": "doc3", "cleaned_text": "NASA landed on Mars.", "cleaned_title": "Mars Mission Success"}
```

---

## Batch Processing Endpoints

### POST /v1/documents/batch

Submit a batch of simple text documents for asynchronous processing.

**Request Body**:

```json
{
  "texts": [
    "Apple released iOS 18 with new AI features.",
    "Microsoft acquired GitHub for $7.5 billion.",
    "Tesla stock rose 15% after earnings report."
  ],
  "job_id": "custom-job-id-123"
}
```

**Note**: `job_id` is optional. If not provided, a UUID will be auto-generated.

**Response 202 ACCEPTED**:

```json
{
  "job_id": "custom-job-id-123",
  "message": "Batch processing initiated. Use GET /v1/jobs/custom-job-id-123 to track progress.",
  "status_endpoint": "/v1/jobs/custom-job-id-123"
}
```

**Example**:

```bash
curl -X POST http://localhost:8000/v1/documents/batch \
  -H "Content-Type: application/json" \
  -d '{
    "texts": [
      "Google announced Gemini 2.0 AI model.",
      "Amazon opened new fulfillment center in Texas.",
      "Netflix subscriber count reached 250 million."
    ]
  }'
```

---

## Job Management Endpoints

### GET /v1/jobs/{job_id}

Check the status and results of an asynchronous job.

**Parameters**:

- `job_id` (path): The job ID returned from submission

**Response 200 OK** (In Progress):

```json
{
  "job_id": "abc-123-def-456",
  "status": "PENDING",
  "progress": 0.4,
  "total_items": 10,
  "processed_items": 4,
  "results": null,
  "error": null
}
```

**Response 200 OK** (Completed):

```json
{
  "job_id": "abc-123-def-456",
  "status": "SUCCESS",
  "progress": 1.0,
  "total_items": 10,
  "processed_items": 10,
  "results": [
    {
      "events": [...],
      "extracted_entities": [...],
      "extracted_soa_triplets": [...],
      "job_id": "abc-123-def-456-0",
      "source_document": {...}
    },
    // ... 9 more results
  ],
  "error": null
}
```

**Response 404 NOT FOUND**:

```json
{
  "detail": "Job ID abc-123-def-456 not found."
}
```

**Example - Poll for results**:

```bash
# Poll immediately
curl http://localhost:8000/v1/jobs/abc-123-def-456

# Poll with watch (every 30 seconds)
watch -n 30 curl -s http://localhost:8000/v1/jobs/abc-123-def-456
```

**Shell Script for Polling**:

```bash
#!/bin/bash
JOB_ID=$1
while true; do
  STATUS=$(curl -s http://localhost:8000/v1/jobs/$JOB_ID | jq -r '.status')
  echo "Status: $STATUS"
  if [ "$STATUS" = "SUCCESS" ] || [ "$STATUS" = "FAILURE" ]; then
    break
  fi
  sleep 30
done
echo "Job completed!"
curl -s http://localhost:8000/v1/jobs/$JOB_ID | jq '.'
```

---

## Microservice APIs

### NER Service API

**Base URL**: `http://localhost:8001`

#### POST /predict

Extract named entities from text.

**Request**:

```json
{
  "text": "Apple CEO Tim Cook visited Paris last week."
}
```

**Response**:

```json
{
  "entities": [
    {"text": "Apple", "type": "ORG", "start_char": 0, "end_char": 5},
    {"text": "Tim Cook", "type": "PER", "start_char": 10, "end_char": 18},
    {"text": "Paris", "type": "LOC", "start_char": 27, "end_char": 32}
  ],
  "text": "Apple CEO Tim Cook visited Paris last week."
}
```

---

### DP Service API

**Base URL**: `http://localhost:8002`

#### POST /extract-soa

Extract Subject-Object-Action triplets from text.

**Request**:

```json
{
  "text": "Microsoft acquired LinkedIn for $26 billion."
}
```

**Response**:

```json
{
  "soa_triplets": [
    {
      "subject": {"text": "Microsoft", "start_char": 0, "end_char": 9},
      "action": {"text": "acquired", "start_char": 10, "end_char": 18},
      "object": {"text": "LinkedIn", "start_char": 19, "end_char": 27}
    }
  ],
  "text": "Microsoft acquired LinkedIn for $26 billion."
}
```

---

### Event LLM Service API

**Base URL**: `http://localhost:8003`

#### POST /generate-events

Generate structured events using LLM.

**Request**:

```json
{
  "text": "Tesla opened a new Gigafactory in Berlin.",
  "ner_entities": [
    {"text": "Tesla", "type": "ORG", "start_char": 0, "end_char": 5},
    {"text": "Berlin", "type": "LOC", "start_char": 34, "end_char": 40}
  ],
  "soa_triplets": [
    {
      "subject": {"text": "Tesla", "start_char": 0, "end_char": 5},
      "action": {"text": "opened", "start_char": 6, "end_char": 12},
      "object": {"text": "Gigafactory", "start_char": 19, "end_char": 30}
    }
  ],
  "context_metadata": {
    "title": "Tesla Expansion",
    "date": "2024-03-22"
  }
}
```

**Response**:

```json
{
  "events": [
    {
      "event_type": "facility_opening",
      "trigger": {"text": "opened", "start_char": 6, "end_char": 12},
      "arguments": [
        {
          "argument_role": "agent",
          "entity": {"text": "Tesla", "type": "ORG", "start_char": 0, "end_char": 5}
        },
        {
          "argument_role": "facility",
          "entity": {"text": "Gigafactory", "type": "FAC", "start_char": 19, "end_char": 30}
        },
        {
          "argument_role": "location",
          "entity": {"text": "Berlin", "type": "LOC", "start_char": 34, "end_char": 40}
        }
      ],
      "metadata": {
        "sentiment": "positive",
        "causality": "Tesla expanded its manufacturing capacity by opening a new facility."
      }
    }
  ],
  "extracted_entities": [...],
  "extracted_soa_triplets": [...],
  "job_id": "...",
  "context_metadata": {"title": "Tesla Expansion", "date": "2024-03-22"}
}
```

---

## Rate Limits

Currently, no rate limits are enforced. For production:

- Recommended: 100 requests/minute per IP
- Batch endpoint: 10 requests/minute per IP
- Implement using Redis + middleware

---

## Error Codes

| Code | Meaning | Common Causes |
|------|---------|---------------|
| 400 | Bad Request | Invalid JSON, missing fields |
| 404 | Not Found | Invalid job_id |
| 422 | Validation Error | Schema violation |
| 500 | Internal Server Error | Unexpected service error |
| 503 | Service Unavailable | Microservice down |
| 504 | Gateway Timeout | LLM processing took too long |

---

## Webhook Support (Future)

Coming soon: Register webhooks to receive notifications when jobs complete.

```json
POST /v1/webhooks
{
  "url": "https://your-app.com/webhook",
  "events": ["job.completed", "job.failed"]
}
```

---

## Versioning

The API uses URL versioning (`/v1/`). Breaking changes will be released under new versions (`/v2/`).

Current version: **v1.0.0**

---

## OpenAPI/Swagger

Interactive API documentation available at:

- **Swagger UI**: <http://localhost:8000/docs>
- **ReDoc**: <http://localhost:8000/redoc>
- **OpenAPI JSON**: <http://localhost:8000/openapi.json>
