
# User Guide

Complete guide for using the EEE Pipeline to extract events and entities from text.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Processing Simple Text](#processing-simple-text)
3. [Processing Enriched Documents](#processing-enriched-documents)
4. [Batch Processing](#batch-processing)
5. [Understanding Results](#understanding-results)
6. [Best Practices](#best-practices)
7. [Common Use Cases](#common-use-cases)

---

## Getting Started

### Prerequisites

Ensure the EEE Pipeline is running:

```bash
# Check system health
curl http://localhost:8000/health

# Expected response:
# {"status": "ok", "dependencies_ok": true, ...}
```

### Tools You'll Need

- **curl** (command-line)
- **Postman** or **Insomnia** (GUI)
- **Python requests** (programmatic access)
- **httpie** (user-friendly CLI)

---

## Processing Simple Text

### When to Use

- Short text snippets (< 500 characters)
- News headlines
- Social media posts
- Quick prototyping

### Basic Example

```bash
curl -X POST http://localhost:8000/v1/documents \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Amazon CEO Andy Jassy announced a $1 billion investment in renewable energy projects across Europe."
  }'
```

### Response Structure

```json
{
  "events": [
    {
      "event_type": "investment_announcement",
      "trigger": {
        "text": "announced",
        "start_char": 20,
        "end_char": 29
      },
      "arguments": [
        {
          "argument_role": "agent",
          "entity": {
            "text": "Andy Jassy",
            "type": "PER",
            "start_char": 11,
            "end_char": 21
          }
        },
        {
          "argument_role": "amount",
          "entity": {
            "text": "$1 billion",
            "type": "MONEY",
            "start_char": 32,
            "end_char": 42
          }
        }
      ],
      "metadata": {
        "sentiment": "positive",
        "causality": "Amazon announced a major investment in renewable energy."
      }
    }
  ],
  "extracted_entities": [
    {"text": "Amazon", "type": "ORG", "start_char": 0, "end_char": 6},
    {"text": "Andy Jassy", "type": "PER", "start_char": 11, "end_char": 21},
    {"text": "$1 billion", "type": "MONEY", "start_char": 32, "end_char": 42},
    {"text": "Europe", "type": "LOC", "start_char": 91, "end_char": 97}
  ],
  "extracted_soa_triplets": [
    {
      "subject": {"text": "Andy Jassy", "start_char": 11, "end_char": 21},
      "action": {"text": "announced", "start_char": 20, "end_char": 29},
      "object": {"text": "investment", "start_char": 43, "end_char": 53}
    }
  ],
  "job_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

### Python Example

```python
import requests

response = requests.post(
    "http://localhost:8000/v1/documents",
    json={
        "text": "Tesla stock surged 12% after Q4 earnings beat expectations."
    }
)

result = response.json()
print(f"Found {len(result['events'])} events")
for event in result['events']:
    print(f"  - {event['event_type']}: {event['trigger']['text']}")
```

---

## Processing Enriched Documents

### When to Use

- News articles with metadata
- Blog posts with author/date
- Research papers with citations
- Any document where context enhances extraction

### Document Structure

Your enriched document can have any structure. Configure field mapping in `config/settings.yaml`:

```yaml
document_field_mapping:
  text_field: "cleaned_text"  # Primary text source
  context_fields:
    - "cleaned_title"
    - "cleaned_author"
    - "cleaned_publication_date"
    - "cleaned_source_url"
```

### Example Document

```json
{
  "document_id": "article_001",
  "cleaned_text": "Dr. Emily Chen won the Nobel Prize in Physics for her work on quantum computing. The announcement was made in Stockholm on October 10, 2024.",
  "cleaned_title": "Physicist Wins Nobel Prize",
  "cleaned_author": "Science Reporter",
  "cleaned_publication_date": "2024-10-10",
  "cleaned_source_url": "https://news.example.com/nobel-2024",
  "entities": [
    {"text": "Emily Chen", "type": "PERSON", "start_char": 4, "end_char": 14}
  ]
}
```

### Submit for Processing

**Option 1: Direct JSON**

```bash
curl -X POST http://localhost:8000/v1/documents/enriched \
  -H "Content-Type: application/json" \
  -d @article.json
```

**Option 2: File Upload**

```bash
curl -X POST http://localhost:8000/v1/documents/enriched/upload \
  -F "file=@article.json"
```

### Response

```json
{
  "job_id": "nobel-article-123",
  "status": "SUBMITTED",
  "message": "Document submitted for async processing. Results will be available in 5-10 minutes.",
  "status_endpoint": "/v1/jobs/nobel-article-123",
  "polling_command": "curl http://localhost:8000/v1/jobs/nobel-article-123",
  "estimated_completion_minutes": 7
}
```

### Check Status

```bash
# Poll for results
curl http://localhost:8000/v1/jobs/nobel-article-123
```

**In Progress**:

```json
{
  "job_id": "nobel-article-123",
  "status": "PENDING",
  "progress": 0.3,
  "total_items": 1,
  "processed_items": 0
}
```

**Completed**:

```json
{
  "job_id": "nobel-article-123",
  "status": "SUCCESS",
  "progress": 1.0,
  "results": [
    {
      "events": [
        {
          "event_type": "award_win",
          "trigger": {"text": "won", "start_char": 15, "end_char": 18},
          "arguments": [
            {
              "argument_role": "winner",
              "entity": {"text": "Dr. Emily Chen", "type": "PER", "start_char": 0, "end_char": 14}
            },
            {
              "argument_role": "award",
              "entity": {"text": "Nobel Prize in Physics", "type": "AWARD", "start_char": 23, "end_char": 45}
            }
          ],
          "metadata": {
            "sentiment": "positive",
            "causality": "Dr. Emily Chen was recognized for her contributions to quantum computing."
          }
        }
      ],
      "extracted_entities": [...],
      "source_document": {
        "document_id": "article_001",
        "cleaned_title": "Physicist Wins Nobel Prize",
        "cleaned_author": "Science Reporter",
        ...
      }
    }
  ]
}
```

---

## Batch Processing

### When to Use

- Processing 10+ documents
- Daily news scraping
- Historical data analysis
- Large-scale content analysis

### Simple Text Batch

```bash
curl -X POST http://localhost:8000/v1/documents/batch \
  -H "Content-Type: application/json" \
  -d '{
    "texts": [
      "Apple launched the new MacBook Pro with M3 chip.",
      "Google announced Gemini AI model improvements.",
      "Microsoft released Windows 12 developer preview."
    ]
  }'
```

**Response**:

```json
{
  "job_id": "batch-abc-123",
  "message": "Batch processing initiated.",
  "status_endpoint": "/v1/jobs/batch-abc-123"
}
```

### Enriched Document Batch (JSONL Format)

**Create `articles.jsonl`**:

```jsonl
{"document_id": "doc1", "cleaned_text": "Apple released iOS 18 with new AI features on September 15, 2024.", "cleaned_title": "iOS 18 Launch", "cleaned_publication_date": "2024-09-15"}
{"document_id": "doc2", "cleaned_text": "Tesla opened a new Gigafactory in Austin, Texas.", "cleaned_title": "Tesla Expansion", "cleaned_publication_date": "2024-08-20"}
{"document_id": "doc3", "cleaned_text": "SpaceX successfully launched Starship to orbit.", "cleaned_title": "Starship Success", "cleaned_publication_date": "2024-07-10"}
```

**Submit Batch**:

```bash
curl -X POST http://localhost:8000/v1/documents/enriched/upload \
  -F "file=@articles.jsonl"
```

**Response**:

```json
{
  "job_id": "batch-enriched-456",
  "status": "SUBMITTED",
  "documents_count": 3,
  "message": "Submitted 3 document(s) for async processing.",
  "status_endpoint": "/v1/jobs/batch-enriched-456",
  "estimated_completion_minutes": 21
}
```

### Monitoring Batch Progress

**Shell Script** (`monitor_job.sh`):

```bash
#!/bin/bash
JOB_ID=$1
BASE_URL="http://localhost:8000"

echo "Monitoring job: $JOB_ID"
echo "================================"

while true; do
  RESPONSE=$(curl -s "$BASE_URL/v1/jobs/$JOB_ID")
  STATUS=$(echo $RESPONSE | jq -r '.status')
  PROGRESS=$(echo $RESPONSE | jq -r '.progress')
  PROCESSED=$(echo $RESPONSE | jq -r '.processed_items')
  TOTAL=$(echo $RESPONSE | jq -r '.total_items')
  
  echo "[$(date +%H:%M:%S)] Status: $STATUS | Progress: $(echo "$PROGRESS * 100" | bc)% | Items: $PROCESSED/$TOTAL"
  
  if [ "$STATUS" = "SUCCESS" ] || [ "$STATUS" = "FAILURE" ]; then
    echo "================================"
    echo "Job completed with status: $STATUS"
    break
  fi
  
  sleep 10
done

# Save results
curl -s "$BASE_URL/v1/jobs/$JOB_ID" | jq '.' > "results_$JOB_ID.json"
echo "Results saved to: results_$JOB_ID.json"
```

**Usage**:

```bash
chmod +x monitor_job.sh
./monitor_job.sh batch-enriched-456
```

### Python Batch Processing

```python
import requests
import time
import json

def submit_batch(documents):
    """Submit a batch of enriched documents."""
    response = requests.post(
        "http://localhost:8000/v1/documents/batch",
        json={"documents": documents}
    )
    return response.json()["job_id"]

def poll_job(job_id, interval=30):
    """Poll job status until completion."""
    url = f"http://localhost:8000/v1/jobs/{job_id}"
    
    while True:
        response = requests.get(url)
        result = response.json()
        
        status = result["status"]
        progress = result.get("progress", 0)
        
        print(f"Status: {status} | Progress: {progress*100:.1f}%")
        
        if status in ["SUCCESS", "FAILURE"]:
            return result
        
        time.sleep(interval)

# Example usage
documents = [
    {
        "document_id": "doc1",
        "cleaned_text": "Apple released iOS 18.",
        "cleaned_title": "iOS 18 Launch"
    },
    {
        "document_id": "doc2",
        "cleaned_text": "Tesla opened Berlin factory.",
        "cleaned_title": "Tesla Expansion"
    }
]

# Submit and monitor
job_id = submit_batch(documents)
print(f"Submitted job: {job_id}")

results = poll_job(job_id)
print(f"Completed! Found {len(results['results'])} results")

# Save results
with open(f"results_{job_id}.json", "w") as f:
    json.dump(results, f, indent=2)
```

---

## Understanding Results

### Event Structure

Each event contains:

```json
{
  "event_type": "product_launch",      // Type of event
  "trigger": {                          // Word/phrase that indicates the event
    "text": "released",
    "start_char": 10,
    "end_char": 18
  },
  "arguments": [                        // Entities involved in the event
    {
      "argument_role": "agent",         // Role: agent, patient, location, time, etc.
      "entity": {                       // Single entity
        "text": "Apple",
        "type": "ORG",
        "start_char": 0,
        "end_char": 5
      }
    },
    {
      "argument_role": "recipients",    // Role with multiple entities
      "entities": [                     // List of entities
        {"text": "customers", "type": "GROUP", "start_char": 50, "end_char": 59},
        {"text": "developers", "type": "GROUP", "start_char": 64, "end_char": 74}
      ]
    }
  ],
  "metadata": {
    "sentiment": "positive",            // positive, negative, neutral
    "causality": "Apple launched a new product to expand market share."
  }
}
```

### Common Event Types

| Event Type | Description | Example Triggers |
|------------|-------------|------------------|
| `product_launch` | Company releases new product | released, launched, unveiled |
| `acquisition` | Company acquires another | acquired, bought, purchased |
| `investment` | Financial investment | invested, funded, raised |
| `partnership` | Business collaboration | partnered, collaborated, joined |
| `appointment` | Leadership change | appointed, hired, promoted |
| `legal` | Legal proceedings | sued, charged, sentenced |
| `crime` | Criminal activity | arrested, convicted, charged |
| `research_presentation` | Scientific presentation | presented, published, announced |
| `facility_opening` | New facility | opened, inaugurated, launched |

### Entity Types

| Type | Description | Examples |
|------|-------------|----------|
| `PER` / `PERSON` | Person name | Tim Cook, Dr. Smith |
| `ORG` | Organization | Apple, UN, Tesla |
| `LOC` / `GPE` | Location | Paris, California, USA |
| `DATE` | Date/time | yesterday, March 2024 |
| `MONEY` | Monetary amount | $1 billion, €500 |
| `PRODUCT` | Product name | iPhone 15, Windows 11 |
| `EVENT` | Event name | World Cup, Olympics |
| `FAC` | Facility | Gigafactory, Airport |

### Character Offsets

All text spans include `start_char` and `end_char`:

```python
text = "Apple CEO Tim Cook visited Paris."
#       012345678901234567890123456789012
#       0         1         2         3

# Entity: "Tim Cook"
entity = {
  "text": "Tim Cook",
  "start_char": 10,
  "end_char": 18
}

# Extract from original text
extracted = text[entity["start_char"]:entity["end_char"]]
assert extracted == "Tim Cook"
```

---

## Best Practices

### 1. Text Preprocessing

**Clean your text** before submission:

```python
def clean_text(raw_text):
    """Prepare text for optimal extraction."""
    # Remove excessive whitespace
    text = " ".join(raw_text.split())
    
    # Remove special characters that confuse the model
    text = text.replace('\u200b', '')  # Zero-width space
    text = text.replace('\xa0', ' ')   # Non-breaking space
    
    # Normalize quotes
    text = text.replace('"', '"').replace('"', '"')
    text = text.replace(''', "'").replace(''', "'")
    
    return text

# Good
clean_text("  Apple    released   iOS  18.  ")
# Output: "Apple released iOS 18."

# Bad (don't send raw HTML)
"<p>Apple released <strong>iOS 18</strong>.</p>"
```

### 2. Context is Key

For enriched documents, **always include metadata**:

```json
{
  "cleaned_text": "The company announced record profits.",
  "cleaned_title": "Apple Q4 Earnings Beat Expectations",    // ← Helps identify "company" = Apple
  "cleaned_publication_date": "2024-10-15",                  // ← Temporal context
  "cleaned_source_url": "https://finance.yahoo.com/apple"    // ← Domain context
}
```

### 3. Batch Size Recommendations

| Document Type | Recommended Batch Size | Processing Time |
|---------------|------------------------|-----------------|
| Simple text (< 500 chars) | 50-100 documents | 5-10 minutes |
| Medium text (500-2000 chars) | 20-50 documents | 10-20 minutes |
| Enriched documents (2000+ chars) | 5-10 documents | 30-60 minutes |

### 4. Error Handling

**Always check job status**:

```python
def safe_poll(job_id, max_retries=3):
    """Poll with error handling."""
    retries = 0
    
    while retries < max_retries:
        try:
            response = requests.get(
                f"http://localhost:8000/v1/jobs/{job_id}",
                timeout=10
            )
            response.raise_for_status()
            
            result = response.json()
            
            if result["status"] == "FAILURE":
                print(f"Job failed: {result.get('error')}")
                return None
            
            if result["status"] == "SUCCESS":
                return result["results"]
            
            time.sleep(30)
            
        except requests.RequestException as e:
            print(f"Request error: {e}")
            retries += 1
            time.sleep(5)
    
    print("Max retries exceeded")
    return None
```

### 5. Performance Tips

**Optimize for speed**:

- Use **simple text endpoint** for short documents (< 500 chars)
- Use **batch processing** for 10+ documents (parallelization)
- Keep documents focused (shorter = faster)
- Remove unnecessary content (ads, navigation, footers)

**Don't**:

- Send entire web pages (extract article content first)
- Include duplicate documents in same batch
- Submit individual documents in a loop (use batch instead)

---

## Common Use Cases

### Use Case 1: News Monitoring

**Goal**: Extract events from daily news articles

```python
import feedparser
import requests

def process_news_feed(feed_url):
    """Process RSS feed articles."""
    feed = feedparser.parse(feed_url)
    documents = []
    
    for entry in feed.entries[:10]:  # Process 10 latest
        documents.append({
            "document_id": entry.id,
            "cleaned_text": entry.summary,
            "cleaned_title": entry.title,
            "cleaned_publication_date": entry.published,
            "cleaned_source_url": entry.link
        })
    
    # Submit batch
    response = requests.post(
        "http://localhost:8000/v1/documents/batch",
        json={"documents": documents}
    )
    
    job_id = response.json()["job_id"]
    print(f"Processing {len(documents)} articles. Job ID: {job_id}")
    
    return job_id

# Monitor multiple news sources
feeds = [
    "https://news.google.com/rss",
    "https://www.reuters.com/rssfeed",
    "https://feeds.bbci.co.uk/news/rss.xml"
]

for feed_url in feeds:
    job_id = process_news_feed(feed_url)
    # Poll and save results...
```

### Use Case 2: Social Media Analysis

**Goal**: Extract events from tweets/posts

```python
def process_social_media(posts):
    """Process social media posts."""
    # Filter out short/spam posts
    filtered = [
        p for p in posts 
        if len(p["text"]) > 50 and not is_spam(p["text"])
    ]
    
    # Simple text processing (fast)
    results = []
    for post in filtered:
        response = requests.post(
            "http://localhost:8000/v1/documents",
            json={"text": post["text"]}
        )
        
        if response.status_code == 200:
            result = response.json()
            if result["events"]:  # Only save if events found
                results.append({
                    "post_id": post["id"],
                    "events": result["events"]
                })
    
    return results

def is_spam(text):
    """Simple spam detection."""
    spam_keywords = ["buy now", "click here", "limited offer"]
    return any(kw in text.lower() for kw in spam_keywords)
```

### Use Case 3: Financial Document Analysis

**Goal**: Extract investment/acquisition events from earnings reports

```python
def analyze_earnings_report(pdf_text):
    """Extract events from earnings report."""
    # Split into sections
    sections = split_into_sections(pdf_text)
    
    # Focus on key sections
    relevant_sections = [
        sections.get("Business Highlights"),
        sections.get("Recent Developments"),
        sections.get("Forward Looking Statements")
    ]
    
    documents = []
    for i, section_text in enumerate(relevant_sections):
        if section_text:
            documents.append({
                "document_id": f"earnings_section_{i}",
                "cleaned_text": section_text,
                "cleaned_title": f"Earnings Report - Section {i}",
                "cleaned_publication_date": "2024-10-15"
            })
    
    # Process batch
    response = requests.post(
        "http://localhost:8000/v1/documents/enriched/upload",
        files={"file": json.dumps(documents)}
    )
    
    return response.json()["job_id"]

def split_into_sections(text):
    """Split document into sections."""
    # Implementation depends on document structure
    return {
        "Business Highlights": "...",
        "Recent Developments": "..."
    }
```

### Use Case 4: Academic Research

**Goal**: Extract research findings from papers

```python
def process_research_papers(papers_dir):
    """Process academic papers."""
    documents = []
    
    for paper_file in Path(papers_dir).glob("*.txt"):
        with open(paper_file) as f:
            text = f.read()
        
        # Extract metadata from filename or content
        metadata = extract_paper_metadata(paper_file)
        
        documents.append({
            "document_id": paper_file.stem,
            "cleaned_text": text,
            "cleaned_title": metadata["title"],
            "cleaned_author": metadata["authors"],
            "cleaned_publication_date": metadata["year"]
        })
    
    # Submit as JSONL
    jsonl_content = "\n".join(
        json.dumps(doc) for doc in documents
    )
    
    response = requests.post(
        "http://localhost:8000/v1/documents/enriched/upload",
        files={"file": ("papers.jsonl", jsonl_content)}
    )
    
    return response.json()["job_id"]
```

### Use Case 5: Real-time Event Detection

**Goal**: Process incoming text streams

```python
import asyncio
import aiohttp

async def stream_processor(text_stream):
    """Process streaming text in real-time."""
    buffer = []
    batch_size = 10
    
    async with aiohttp.ClientSession() as session:
        async for text in text_stream:
            buffer.append(text)
            
            if len(buffer) >= batch_size:
                # Submit batch
                job_id = await submit_batch_async(session, buffer)
                print(f"Submitted batch: {job_id}")
                
                # Start monitoring in background
                asyncio.create_task(monitor_job_async(session, job_id))
                
                buffer = []

async def submit_batch_async(session, texts):
    """Submit batch asynchronously."""
    async with session.post(
        "http://localhost:8000/v1/documents/batch",
        json={"texts": texts}
    ) as response:
        result = await response.json()
        return result["job_id"]

async def monitor_job_async(session, job_id):
    """Monitor job asynchronously."""
    while True:
        async with session.get(
            f"http://localhost:8000/v1/jobs/{job_id}"
        ) as response:
            result = await response.json()
            
            if result["status"] in ["SUCCESS", "FAILURE"]:
                # Save results
                save_results(job_id, result)
                break
        
        await asyncio.sleep(10)
```

---

## Troubleshooting

### Problem: Timeout on `/v1/documents`

**Cause**: Document too long for synchronous processing

**Solution**: Use enriched endpoint instead

```bash
# Don't do this for long documents
curl -X POST http://localhost:8000/v1/documents \
  -d '{"text": "... 5000 characters ..."}'

# Do this instead
curl -X POST http://localhost:8000/v1/documents/enriched \
  -d '{"cleaned_text": "... 5000 characters ..."}'
```

### Problem: No events extracted

**Possible causes**:

1. Text too vague (lacks specific events)
2. Text too short (< 20 characters)
3. Text in unsupported language

**Solutions**:

```python
# Add context for enriched documents
{
  "cleaned_text": "The company announced earnings.",  # Vague
  "cleaned_title": "Apple Q4 2024 Earnings",          # Context helps!
  "cleaned_publication_date": "2024-10-15"
}

# Ensure minimum length
if len(text) < 20:
    print("Text too short for meaningful extraction")
```

### Problem: Incorrect character offsets

**Cause**: Text preprocessing changed character positions

**Solution**: Keep track of original text

```python
original_text = "  Apple   released iOS 18.  "
cleaned_text = " ".join(original_text.split())  # "Apple released iOS 18."

# Offsets in cleaned_text won't match original_text
# Solution: Don't clean, or map offsets back
```

### Problem: Job stuck in PENDING

**Causes**:

1. Celery workers not running
2. Redis connection lost
3. LLM service crashed

**Check**:

```bash
# Check worker status
docker compose ps

# Check logs
docker compose logs celery-worker
docker compose logs event-llm-service

# Restart services
docker compose restart
```

---

## Next Steps

- **[API Documentation](API.md)** - Full API reference
- **[Configuration Guide](CONFIGURATION.md)** - Customize behavior
- **[Developer Guide](DEVELOPER_GUIDE.md)** - Extend the pipeline
- **[Troubleshooting](TROUBLESHOOTING.md)** - Common issues

---

**Need Help?**

- Check logs: `docker compose logs -f`
- File issue: [GitHub Issues](https://github.com/your-org/eee-pipeline/issues)
- Email: <support@your-org.com>
