
# Event & Entity Extraction (EEE) Pipeline

A production-ready, microservices-based NLP pipeline for extracting events, entities, and relationships from unstructured text using state-of-the-art deep learning models.

## 🎯 Overview

The EEE Pipeline orchestrates three specialized NLP services to transform raw text into structured event data:

1. **NER Service**: Named Entity Recognition (spaCy-based)
2. **DP Service**: Dependency Parsing & Subject-Object-Action extraction
3. **Event LLM Service**: Event extraction using Large Language Models (Mistral-7B)

### Key Features

- ✅ **Microservices Architecture**: Independently scalable services
- ✅ **Async Batch Processing**: Celery + Dask for parallel processing
- ✅ **Flexible Input**: Simple text or enriched documents with metadata
- ✅ **RESTful API**: OpenAPI/Swagger documentation
- ✅ **Multiple Storage Backends**: JSON, JSONL, PostgreSQL, MongoDB
- ✅ **Docker-native**: Full containerization with docker-compose
- ✅ **GPU-accelerated**: CUDA support for LLM inference
- ✅ **Production-ready**: Health checks, logging, error handling, retries

## 🏗️ Architecture

```
┌─────────────┐
│   Client    │
└──────┬──────┘
       │ HTTP/REST
       ▼
┌─────────────────────┐
│ Orchestrator Service│ ← Redis (Job tracking)
└──────┬──────────────┘
       │
       ├──→ NER Service (spaCy)
       ├──→ DP Service (spaCy)
       └──→ Event LLM Service (Mistral-7B)
              │
              └──→ Celery Workers (Async batch processing)
                      │
                      └──→ Dask (Parallel execution)
                            │
                            └──→ Storage Backends
```

## 📋 Prerequisites

### Hardware Requirements

**Minimum (CPU only)**:

- 8 CPU cores
- 32GB RAM
- 50GB disk space

**Recommended (with GPU)**:

- 16 CPU cores
- 64GB RAM
- NVIDIA GPU with 16GB VRAM (e.g., RTX A4000, RTX 4090)
- 100GB disk space

### Software Requirements

- Docker 24.0+
- Docker Compose v2.20+
- NVIDIA Container Toolkit (for GPU support)
- Python 3.10+ (for development)

## 🚀 Quick Start

### 1. Clone Repository

```bash
git clone https://github.com/your-org/eee-pipeline.git
cd eee-pipeline
```

### 2. Configuration

```bash
# Copy and edit configuration
cp config/settings.yaml.example config/settings.yaml

# Edit settings.yaml with your preferences
nano config/settings.yaml
```

### 3. Start Services

**CPU-only mode:**

```bash
docker compose up -d
```

**GPU-accelerated mode:**

```bash
docker compose -f docker-compose.gpu.yml up -d
```

### 4. Verify Health

```bash
curl http://localhost:8000/health
```

### 5. Process Your First Document

```bash
curl -X POST http://localhost:8000/v1/documents \
  -H "Content-Type: application/json" \
  -d '{"text": "Apple CEO Tim Cook announced the new iPhone 15 in Cupertino yesterday."}'
```

## 📖 Documentation

Comprehensive documentation is organized by concern:

- **[API Documentation](docs/API.md)** - REST API endpoints and examples
- **[User Guide](docs/USER_GUIDE.md)** - How to use the pipeline
- **[Deployment Guide](docs/DEPLOYMENT.md)** - Production deployment
- **[Configuration Guide](docs/CONFIGURATION.md)** - Settings and tuning
- **[Developer Guide](docs/DEVELOPER_GUIDE.md)** - Internal architecture and development
- **[Storage Guide](docs/STORAGE.md)** - Storage backends configuration
- **[Troubleshooting](docs/TROUBLESHOOTING.md)** - Common issues and solutions

## 🎓 Quick Examples

### Simple Text Processing

```bash
curl -X POST http://localhost:8000/v1/documents \
  -H "Content-Type: application/json" \
  -d '{"text": "Tesla opened a new factory in Berlin on March 22, 2024."}'
```

### Enriched Document Processing

```bash
curl -X POST http://localhost:8000/v1/documents/enriched/upload \
  -F "file=@data/article.json"
```

### Batch Processing

```bash
curl -X POST http://localhost:8000/v1/documents/batch \
  -H "Content-Type: application/json" \
  -d '{
    "texts": [
      "Apple released iOS 18.",
      "Microsoft acquired GitHub.",
      "Tesla stock rose 5%."
    ]
  }'
```

## 🔧 Development Setup

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run tests
pytest tests/

# Lint code
flake8 src/
black src/
```

## 📊 Performance

### Benchmarks (RTX A4000, 16GB VRAM)

| Document Type | Avg Time | Throughput |
|---------------|----------|------------|
| Simple text (< 500 chars) | 2-5 sec | 720 docs/hour |
| Medium text (500-2000 chars) | 5-15 sec | 240 docs/hour |
| Enriched document (2000+ chars) | 5-10 min | 8 docs/hour |

### Scaling

- **Horizontal**: Add more Celery workers
- **Vertical**: Use larger GPUs or multi-GPU setups
- **Optimization**: See [Performance Tuning](docs/PERFORMANCE.md)

## 🤝 Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## 📝 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [spaCy](https://spacy.io/) for NER and dependency parsing
- [HuggingFace Transformers](https://huggingface.co/transformers/) for LLM integration
- [Celery](https://docs.celeryproject.org/) for distributed task processing
- [Dask](https://dask.org/) for parallel execution

## 📧 Support

- **Issues**: [GitHub Issues](https://github.com/your-org/eee-pipeline/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-org/eee-pipeline/discussions)
- **Email**: <support@your-org.com>

## 🗺️ Roadmap

- [ ] vLLM integration for 10x faster inference
- [ ] WebSocket streaming for real-time progress
- [ ] Fine-tuned models for specific domains
- [ ] Multi-language support
- [ ] GraphQL API
- [ ] Cloud deployment templates (AWS, GCP, Azure)

---

**Version**: 1.0.0  
**Last Updated**: October 2025
