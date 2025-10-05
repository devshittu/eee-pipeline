# EEE Pipeline Documentation

## Complete Guide to the Event & Entity Extraction System

---

## Table of Contents

1. [What is the EEE Pipeline?](#what-is-the-eee-pipeline)
2. [Quick Start (2 Minutes)](#quick-start-2-minutes)
3. [REST API Reference](#rest-api-reference)
4. [CLI Guide](#cli-guide)
5. [Handling Large Files](#handling-large-files)
6. [Configuration](#configuration)
7. [FAQ & Troubleshooting](#faq--troubleshooting)

---

## What is the EEE Pipeline?

The **Event & Entity Extraction (EEE) Pipeline** automatically analyzes text documents (like news articles, reports, or social media posts) to identify:

- **Entities**: People, organizations, locations, dates, and other named items
- **Events**: Actions or occurrences with their participants (who did what, when, where)
- **Relationships**: How entities interact through subject-action-object patterns

**Example Input**:

```
The UK government licensed its new AI tool to the US and Canada after a successful pilot.
```

**Example Output**:

```json
{
  "events": [
    {
      "event_type": "policy_change",
      "trigger": {"text": "licensed", "start_char": 29, "end_char": 37},
      "arguments": [
        {"argument_role": "agent", "entity": {"text": "UK government", "type": "ORG"}},
        {"argument_role": "recipients", "entities": [
          {"text": "US", "type": "LOC"},
          {"text": "Canada", "type": "LOC"}
        ]}
      ],
      "metadata": {"sentiment": "positive", "causality": "The licensing followed a successful pilot program."}
    }
  ],
  "extracted_entities": [
    {"text": "UK government", "type": "ORG", "start_char": 4, "end_char": 17},
    {"text": "US", "type": "LOC", "start_char": 47, "end_char": 49},
    {"text": "Canada", "type": "LOC", "start_char": 54, "end_char": 60}
  ]
}
```

**Use Cases**:

- News monitoring and analysis
- Social media intelligence
- Legal document processing
- Research paper analysis
- Business intelligence from reports

---

## Quick Start (2 Minutes)

### Prerequisites

- Docker installed (with GPU support for optimal performance)
- 8GB+ RAM available
- Internet connection (for model downloads on first run)

### Step 1: Start the System

```bash
# Clone the repository
git clone <your-repo-url>
cd eee-pipeline

# Start all services (production mode)
./run.sh start

# Wait ~2 minutes for services to initialize and download models
# Check status with:
./run.sh status
```

### Step 2: Process Your First Document

**Option A: Using CLI (Inside Container)**

```bash
./run.sh cli documents process "The UK government licensed its new AI tool to the US and Canada after a successful pilot."
```

**Option B: Using REST API (From Your Machine)**

```bash
curl -X POST http://localhost:8000/v1/documents \
  -H "Content-Type: application/json" \
  -d '{"text": "The UK government licensed its new AI tool to the US and Canada after a successful pilot."}'
```

**Expected Output**: JSON with extracted events, entities, and metadata (saved automatically to `/app/data/extracted_events_YYYY-MM-DD.jsonl` inside the container).

### Step 3: Check Service Health

```bash
# Via CLI
./run.sh cli admin health

# Or via API
curl http://localhost:8000/health
```

---

## REST API Reference

Base URL: `http://localhost:8000`

All endpoints return JSON. Version prefix: `/v1/`

---

### Health Check

**Endpoint**: `GET /health`

**Description**: Check if all services (NER, DP, Event LLM, Celery) are running.

**Example Request**:

```bash
curl http://localhost:8000/health
```

**Example Response**:

```json
{
  "status": "ok",
  "dependencies_ok": true,
  "ner_service": {"status": "ok", "model_loaded": true},
  "dp_service": {"status": "ok", "model_loaded": true},
  "event_llm_service": {"status": "ok", "model_loaded": true},
  "celery_broker_reachable": true
}
```

**Status Codes**:

- `200 OK`: All services healthy
- `503 Service Unavailable`: One or more services down

---

### Process Single Document (Synchronous)

**Endpoint**: `POST /v1/documents`

**Description**: Process a single text document and get results immediately. Best for interactive use or small documents (<10,000 characters).

**Request Body**:

```json
{
  "text": "Your document text here..."
}
```

**Example Request**:

```bash
curl -X POST http://localhost:8000/v1/documents \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Apple Inc. announced a new iPhone model in San Francisco yesterday. CEO Tim Cook unveiled the device at a packed event."
  }'
```

**Example Response** (abbreviated):

```json
{
  "events": [
    {
      "event_type": "product_launch",
      "trigger": {"text": "announced", "start_char": 11, "end_char": 20},
      "arguments": [
        {"argument_role": "agent", "entity": {"text": "Apple Inc.", "type": "ORG"}},
        {"argument_role": "product", "entity": {"text": "new iPhone model", "type": "PRODUCT"}},
        {"argument_role": "location", "entity": {"text": "San Francisco", "type": "LOC"}}
      ],
      "metadata": {"sentiment": "positive", "causality": "The announcement was made during a packed event."}
    }
  ],
  "extracted_entities": [
    {"text": "Apple Inc.", "type": "ORG", "start_char": 0, "end_char": 10},
    {"text": "San Francisco", "type": "LOC", "start_char": 40, "end_char": 53},
    {"text": "yesterday", "type": "DATE", "start_char": 54, "end_char": 63},
    {"text": "Tim Cook", "type": "PER", "start_char": 69, "end_char": 77}
  ],
  "original_text": "Apple Inc. announced...",
  "job_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
}
```

**Status Codes**:

- `200 OK`: Success
- `400 Bad Request`: Invalid JSON or missing `text` field
- `503 Service Unavailable`: Upstream service (NER/DP/LLM) unavailable
- `500 Internal Server Error`: Processing error

**Response Time**: ~5-30 seconds (depends on text length; GPU speeds this up significantly)

---

### Submit Batch Job (Asynchronous)

**Endpoint**: `POST /v1/documents/batch`

**Description**: Submit multiple documents for background processing. Returns immediately with a `job_id` for tracking. Best for 100+ documents.

**Request Body**:

```json
{
  "texts": ["Document 1 text...", "Document 2 text...", "Document 3 text..."],
  "job_id": "optional-custom-id"  // Auto-generated if omitted
}
```

**Example Request**:

```bash
curl -X POST http://localhost:8000/v1/documents/batch \
  -H "Content-Type: application/json" \
  -d '{
    "texts": [
      "Tesla announced record sales in Q4 2024.",
      "The Federal Reserve raised interest rates by 0.25%.",
      "Scientists discovered a new exoplanet in the Andromeda galaxy."
    ]
  }'
```

**Example Response**:

```json
{
  "job_id": "batch-2025-01-15-abc123",
  "message": "Batch processing initiated. Use GET /v1/jobs/batch-2025-01-15-abc123 to track progress.",
  "status_endpoint": "/v1/jobs/batch-2025-01-15-abc123"
}
```

**Status Codes**:

- `202 Accepted`: Job queued successfully
- `400 Bad Request`: Invalid request (empty `texts` array)
- `500 Internal Server Error`: Failed to queue job

**Processing Time**: Varies by batch size. Check status via `/v1/jobs/{job_id}`.

---

### Get Job Status

**Endpoint**: `GET /v1/jobs/{job_id}`

**Description**: Check the progress and results of a batch job.

**Example Request**:

```bash
curl http://localhost:8000/v1/jobs/batch-2025-01-15-abc123
```

**Example Response (In Progress)**:

```json
{
  "job_id": "batch-2025-01-15-abc123",
  "status": "PENDING",
  "progress": 0.45,
  "total_items": 100,
  "processed_items": 45,
  "results": [],
  "error": null
}
```

**Example Response (Completed)**:

```json
{
  "job_id": "batch-2025-01-15-abc123",
  "status": "SUCCESS",
  "progress": 1.0,
  "total_items": 3,
  "processed_items": 3,
  "results": [
    {
      "events": [...],
      "extracted_entities": [...],
      "original_text": "Tesla announced record sales...",
      "job_id": "batch-2025-01-15-abc123-0"
    },
    // ... 2 more results
  ],
  "error": null
}
```

**Status Values**:

- `PENDING`: Job is queued or in progress
- `SUCCESS`: Job completed successfully
- `FAILURE`: Job failed (see `error` field)

**Status Codes**:

- `200 OK`: Job found
- `404 Not Found`: Invalid `job_id`

**Polling Recommendation**: Poll every 5-10 seconds for jobs with <1000 documents, every 30 seconds for larger jobs.

---

### Legacy Endpoints (Deprecated)

The following endpoints still work but will be removed in v2.0:

- `POST /process-text` → Use `POST /v1/documents`
- `POST /process-batch` → Use `POST /v1/documents/batch`
- `GET /status/{job_id}` → Use `GET /v1/jobs/{job_id}`

---

## CLI Guide

The CLI provides a command-line interface to the same functionality as the REST API, with added convenience for file handling and scripting.

### Installation

**Inside Docker Container** (already available):

```bash
docker exec -it orchestrator-service bash
python -m src.cli.main --help
```

**On Your Local Machine**:

```bash
# Clone repo and install
cd eee-pipeline
pip install -e .

# Now available as system command
eee-cli --help
```

**Via run.sh Helper Script**:

```bash
./run.sh cli --help
```

---

### CLI Command Structure

```
eee-cli <group> <command> [options]
```

**Command Groups**:

- `documents`: Process single texts or batch files
- `jobs`: Manage batch processing jobs
- `admin`: System administration

---

### Documents Commands

#### Process Single Document

```bash
eee-cli documents process "TEXT_STRING"
```

**Example**:

```bash
eee-cli documents process "The UN announced new climate targets at COP29 in Dubai."
```

**Example via run.sh**:

```bash
./run.sh cli documents process "The UN announced new climate targets at COP29 in Dubai."
```

**Output**: JSON printed to stdout + saved to daily JSONL file (`/app/data/extracted_events_2025-01-15.jsonl`)

---

#### Process Batch File

```bash
eee-cli documents batch INPUT_FILE --output OUTPUT_FILE
```

**Input File Format** (JSONL - one JSON object per line):

```jsonl
{"text": "First document text here..."}
{"text": "Second document text here..."}
{"text": "Third document text here..."}
```

**Example**:

```bash
# Create sample input file
cat > input.jsonl << 'EOF'
{"text": "Apple released iOS 18 with new AI features."}
{"text": "The ECB kept interest rates unchanged at 4.5%."}
{"text": "NASA's Artemis mission returned safely to Earth."}
EOF

# Process batch
eee-cli documents batch input.jsonl --output results.jsonl

# Or via run.sh (mount file into container first)
docker cp input.jsonl orchestrator-service:/app/data/
./run.sh cli documents batch /app/data/input.jsonl --output /app/data/results.jsonl
docker cp orchestrator-service:/app/data/results.jsonl ./
```

**Process**:

1. Uploads batch to orchestrator
2. Returns `job_id`
3. Polls status every 5 seconds
4. Saves results to output file when complete

**Output File Format** (JSONL):

```jsonl
{"events": [...], "extracted_entities": [...], "original_text": "Apple released...", "job_id": "..."}
{"events": [...], "extracted_entities": [...], "original_text": "The ECB kept...", "job_id": "..."}
{"events": [...], "extracted_entities": [...], "original_text": "NASA's Artemis...", "job_id": "..."}
```

---

### Jobs Commands

#### Check Job Status

```bash
eee-cli jobs status JOB_ID
```

**Example**:

```bash
eee-cli jobs status batch-2025-01-15-abc123
```

**Output**:

```json
{
  "job_id": "batch-2025-01-15-abc123",
  "status": "PENDING",
  "progress": 0.67,
  "total_items": 100,
  "processed_items": 67,
  "results": [],
  "error": null
}
```

---

#### Retrieve Job Results

```bash
eee-cli jobs results JOB_ID [--output FILE]
```

**Example (print to stdout)**:

```bash
eee-cli jobs results batch-2025-01-15-abc123
```

**Example (save to file)**:

```bash
eee-cli jobs results batch-2025-01-15-abc123 --output results.jsonl
```

**Note**: Only works for completed jobs (`status: "SUCCESS"`).

---

### Admin Commands

#### Health Check

```bash
eee-cli admin health
```

**Output**:

```
=== EEE Pipeline Health Check ===
Orchestrator Status: OK
Dependencies OK: True

Service Status:
  - NER Service: ok
  - DP Service: ok
  - Event LLM Service: ok
  - Celery Broker: OK
```

---

#### Download Models (Pre-cache)

```bash
eee-cli admin download-models
```

**Note**: Models are automatically downloaded when services start. This command is for manual pre-caching or offline environments.

---

## Handling Large Files

### Batch Size Recommendations

| Document Count | Method | Expected Time (48-core CPU + GPU) |
|----------------|--------|-----------------------------------|
| 1-10 | Single API calls | <5 min |
| 10-100 | Batch API (small array) | 5-30 min |
| 100-10,000 | Batch API (JSONL file via CLI) | 30 min - 5 hours |
| 10,000+ | Split into multiple jobs | Varies (parallel processing) |

### Processing Very Large Datasets

**Strategy 1: Split Files**

```bash
# Split large JSONL into chunks of 5000 lines each
split -l 5000 huge_dataset.jsonl chunk_

# Process each chunk as separate job
for file in chunk_*; do
  eee-cli documents batch "$file" --output "results_$file.jsonl"
done

# Merge results
cat results_chunk_*.jsonl > final_results.jsonl
```

**Strategy 2: Use Dask Parallelism**

The system automatically uses Dask for parallel processing. Configure workers in `config/settings.yaml`:

```yaml
celery:
  dask_local_cluster_n_workers: 22  # Half your core count (recommended)
  dask_local_cluster_threads_per_worker: 1
  dask_local_cluster_memory_limit: "150GB"
```

**Strategy 3: Monitor Progress**

```bash
# Submit job
JOB_ID=$(eee-cli documents batch large_file.jsonl --output results.jsonl | grep "Job ID:" | awk '{print $3}')

# Poll status in separate terminal
watch -n 10 "eee-cli jobs status $JOB_ID"
```

### Memory Management

- **Single Document Processing**: ~2GB RAM per document (LLM inference)
- **Batch Processing**: Memory usage scales with `batch_processing_chunk_size` (default: 100 docs per worker)
- **Recommendation**: For 160GB RAM, process batches of 5000-10000 documents at once

---

## Configuration

Main config file: `config/settings.yaml`

### Key Settings

```yaml
# Model Selection (change for different domains)
event_llm_service:
  model_name: "mistralai/Mistral-7B-Instruct-v0.2"
  chunk_size_tokens: 2048  # Increase to 4096 for longer context
  max_new_tokens: 4096

# Batch Processing
orchestrator_service:
  batch_processing_chunk_size: 100  # Documents per Celery task
  batch_processing_job_results_ttl: 3600  # Results expire after 1 hour

# Storage Backends (enable multiple for redundancy)
storage:
  enabled_backends: ["jsonl", "postgresql"]  # Also: "elasticsearch"
  jsonl:
    output_path: "/app/data/extracted_events.jsonl"
  postgresql:
    host: "postgres"
    dbname: "eeedb"
    user: "user"
    password: "CHANGE_THIS_IN_PRODUCTION"
    table_name: "extracted_events"
```

### GPU Configuration

```yaml
general:
  gpu_enabled: true  # Set to false for CPU-only
```

**Note**: GPU acceleration reduces processing time by 5-10x for LLM inference.

---

## FAQ & Troubleshooting

### General Questions

**Q: How accurate is the extraction?**

A: Accuracy depends on domain and text quality:

- **Named Entity Recognition**: 85-95% F1 score
- **Event Extraction**: 70-85% F1 score (challenging task)
- **Relation Extraction**: 65-80% F1 score

Best results on: news articles, formal reports, structured text
Reduced accuracy on: social media, fragmented text, highly technical jargon

**Q: What languages are supported?**

A: Currently English only. The NER model (`wikineural-multilingual-ner`) supports multiple languages, but the LLM is English-focused. To add languages, fine-tune the event LLM on multilingual data.

**Q: Can I use custom entity types?**

A: Yes. Fine-tune the NER model on your domain-specific data. See [Hugging Face fine-tuning guide](https://huggingface.co/docs/transformers/tasks/token_classification).

**Q: How do I export results to my database?**

A: Enable storage backends in `config/settings.yaml`:

```yaml
storage:
  enabled_backends: ["postgresql"]  # or "elasticsearch"
```

Results are automatically persisted to all enabled backends.

---

### Troubleshooting

#### Services Won't Start

**Symptom**: `./run.sh start` fails with "service unhealthy"

**Solutions**:

1. Check Docker resource limits: `docker system info` (ensure >8GB RAM allocated)
2. View service logs: `./run.sh logs <service-name>` (e.g., `event-llm-service`)
3. Verify GPU access: `docker run --rm --gpus all nvidia/cuda:12.3.2-base-ubuntu22.04 nvidia-smi`

**Common Issue**: Model download timeout

```bash
# Check event-llm-service logs
./run.sh logs event-llm-service

# If "403 Forbidden" error, add Hugging Face token
# Create .env.dev file (see below)
```

---

#### Missing Hugging Face Token

**Symptom**: Event LLM service fails with "Repository not found" or "403 Forbidden"

**Solution**: Create `.env.dev` file in project root:

```bash
# .env.dev
HUGGINGFACE_TOKEN=hf_YOUR_TOKEN_HERE
```

Get token from: <https://huggingface.co/settings/tokens>

Then restart:

```bash
./run.sh clean
./run.sh rebuild event-llm-service
./run.sh start
```

---

#### Slow Processing

**Symptom**: Single document takes >60 seconds

**Diagnostics**:

```bash
# Check GPU usage
docker exec orchestrator-service nvidia-smi

# Check CPU usage
docker stats
```

**Solutions**:

1. **No GPU detected**: Verify `gpu_enabled: true` in `settings.yaml` and Docker GPU access
2. **CPU bottleneck**: Increase Dask workers: `dask_local_cluster_n_workers: 22` (half your core count)
3. **Memory bottleneck**: Reduce chunk size: `batch_processing_chunk_size: 50`

---

#### Empty Results

**Symptom**: API returns `{"events": [], "extracted_entities": []}`

**Causes**:

1. **Text too short**: Minimum ~20 words recommended
2. **No recognizable entities**: Check if text contains names, organizations, locations
3. **LLM generation failure**: Check logs: `./run.sh logs event-llm-service`

**Example of problematic input**:

```
Bad: "It was good."  # Too short, no entities
Good: "Apple CEO Tim Cook announced the new product yesterday."
```

---

#### Batch Job Stuck in PENDING

**Symptom**: `eee-cli jobs status <job-id>` shows `PENDING` indefinitely

**Diagnostics**:

```bash
# Check Celery worker status
docker exec celery-worker celery -A src.core.celery_tasks inspect active

# Check Redis connection
docker exec redis redis-cli ping
```

**Solutions**:

1. **Worker crash**: Restart worker: `docker restart celery-worker`
2. **Redis connection lost**: Restart Redis: `docker restart redis`
3. **Task stuck**: Revoke task and resubmit: `docker exec celery-worker celery -A src.core.celery_tasks revoke <task-id>`

---

#### Out of Memory

**Symptom**: Service crashes with "Killed" or "OOM"

**Solutions**:

1. **Reduce batch size**: `batch_processing_chunk_size: 25` (default: 100)
2. **Increase Docker memory limit**: Edit Docker Desktop settings (Mac/Windows) or `/etc/docker/daemon.json` (Linux)
3. **Use quantization**: LLM already uses 4-bit quantization; further reduction impacts quality

---

#### API Returns 503 Service Unavailable

**Symptom**: `curl` requests fail with 503 error

**Cause**: Upstream service (NER/DP/LLM) not ready

**Solution**:

```bash
# Wait for all services to be healthy
./run.sh status

# Check individual service health
curl http://localhost:8001/health  # NER
curl http://localhost:8002/health  # DP
curl http://localhost:8003/health  # Event LLM

# Restart specific service if unhealthy
docker restart ner-service
```

---

### Performance Tuning

**For Maximum Throughput** (your hardware: 48-core CPU, 160GB RAM, RTX A4000):

```yaml
# config/settings.yaml
celery:
  dask_local_cluster_n_workers: 22  # Half your cores
  dask_local_cluster_threads_per_worker: 1  # CPU-bound tasks
  dask_local_cluster_memory_limit: "140GB"  # Leave 20GB for OS

orchestrator_service:
  batch_processing_chunk_size: 200  # Larger chunks for your RAM

event_llm_service:
  chunk_size_tokens: 4096  # Larger context windows
  max_new_tokens: 4096
```

**Expected Performance**:

- Single document: 10-20 seconds (GPU accelerated)
- Batch of 1000 documents: 30-60 minutes
- Batch of 10,000 documents: 3-6 hours

---

### Getting Help

1. **Check logs**: `./run.sh logs <service-name>`
2. **Enable debug logging**: Set `log_level: DEBUG` in `settings.yaml`
3. **Search issues**: Check project repository for similar issues
4. **Report bugs**: Include service logs + `docker compose ps` output

---

## Next Steps

- **Production Deployment**: See `docker-compose.yml` for production settings (remove dev overrides)
- **Custom Models**: Fine-tune NER/LLM on your domain data
- **Monitoring**: Integrate with Prometheus/Grafana for observability
- **Scaling**: Deploy on Kubernetes for multi-node processing

---

**Last Updated**: 2025-01-15  
**Version**: 1.0.0  
**Maintainer**: EEE Pipeline Team
