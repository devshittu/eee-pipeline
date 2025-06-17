# SOTA Production-Ready Event and Entity Extraction Pipeline

This repository contains a robust, scalable, and highly available Event and Entity Extraction (EEE) pipeline designed to process dynamic content at scale. It focuses on extracting precise Subject-Object-Action (S-O-A) triplets and generating a standardized, valid JSON schema. The solution employs a microservices architecture, CLI and API integration, comprehensive containerization, structured logging, resilient error handling, and externalized configuration.

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Setup and Installation](#setup-and-installation)
    - [Prerequisites](#prerequisites)
    - [Model Downloading](#model-downloading)
    - [Building and Running with Docker Compose](#building-and-running-with-docker-compose)
- [API Usage](#api-usage)
    - [NER Service](#ner-service)
    - [Dependency Parsing Service](#dependency-parsing-service)
    - [Event & Schema Generation Service](#event--schema-generation-service)
    - [Orchestrator Service](#orchestrator-service)
- [CLI Usage](#cli-usage)
    - [Process Single Text](#process-single-text)
    - [Process File](#process-file)
- [Configuration](#configuration)
- [Logging](#logging)
- [Error Handling](#error-handling)
- [Development Practices](#development-practices)

## Features

-   **Microservices Architecture**: Clear separation of concerns with dedicated services for NER, Dependency Parsing, Event/Schema Generation, and Orchestration.
-   **API Integration**: All services expose well-documented RESTful APIs using FastAPI.
-   **CLI Integration**: Unified command-line interface for common operations.
-   **Containerization**: Full Docker support for each service, orchestrated by Docker Compose, with GPU acceleration.
-   **Structured Logging**: Comprehensive JSON logging for easy monitoring and debugging.
-   **Robust Error Handling**: Centralized exception handling, consistent error responses, and health checks.
-   **Externalized Configuration**: All parameters managed via a centralized `settings.yaml` file.
-   **Pydantic Models**: Strict data validation and clear schema definitions.
-   **S-O-A Triplet Extraction**: Dedicated dependency parsing for Subject-Action-Object extraction.
-   **Generative LLM**: Fine-tuned LLM for event identification, argument mapping, metadata synthesis, and strict JSON schema generation.

## Architecture

The pipeline consists of four main microservices:

1.  **NER Service (`ner-service`)**: Performs Named Entity Recognition using `Babelscape/wikineural-multilingual-ner`.
2.  **Dependency Parsing Service (`dp-service`)**: Extracts Subject-Object-Action (S-O-A) triplets using spaCy.
3.  **Event & Schema Generation Service (`event-llm-service`)**: Utilizes a fine-tuned generative LLM (e.g., T5/Flan-T5 variant) to identify events, map arguments, synthesize metadata, and generate standardized JSON output.
4.  **Orchestration & Ingestion Service (`orchestrator-service`)**: The central entry point, orchestrating calls to other services, merging outputs, and providing a unified API for single and batch text processing.

## Setup and Installation

### Prerequisites

-   Docker and Docker Compose (v2)
-   NVIDIA Container Toolkit (for GPU support) if running on a GPU-enabled machine.

### Model Downloading

It is highly recommended to pre-download models to avoid repeated downloads within Docker builds or during service startup.

```bash
# From the root of the project
python src/main.py download-models
```
This command will download:
- `Babelscape/wikineural-multilingual-ner` Hugging Face model for NER.
- `en_core_web_trf` spaCy model for Dependency Parsing.
- Placeholder for the Event LLM (you would replace this with your fine-tuned model path/name).

### Building and Running with Docker Compose

1.  **Build the Docker images:**
    ```bash
    docker compose build
    ```

2.  **Start the services:**
    ```bash
    docker compose up -d
    ```

3.  **Verify service health:**
    ```bash
    docker compose ps
    docker compose logs
    ```
    You should see all services running and healthy.

## API Usage

All services expose RESTful APIs using FastAPI. Replace `localhost` with your Docker host IP if running remotely.

### NER Service (`ner-service`)

-   **Base URL**: `http://localhost:8001`
-   **Health Check**: `GET /health`
-   **Predict Entities**: `POST /predict`
    ```json
    // Request Body
    {
        "texts": ["Barack Obama visited London.", "Apple announced new products."]
    }
    // Response Body (Example)
    {
        "results": [
            {
                "text": "Barack Obama visited London.",
                "entities": [
                    {"text": "Barack Obama", "type": "PERSON", "start_char": 0, "end_char": 12},
                    {"text": "London", "type": "LOCATION", "start_char": 21, "end_char": 27}
                ]
            },
            // ... more results
        ]
    }
    ```

### Dependency Parsing Service (`dp-service`)

-   **Base URL**: `http://localhost:8002`
-   **Health Check**: `GET /health`
-   **Extract S-O-A**: `POST /extract-soa`
    ```json
    // Request Body
    {
        "texts": ["John ate an apple.", "The dog chased the ball."]
    }
    // Response Body (Example)
    {
        "results": [
            {
                "text": "John ate an apple.",
                "soa_triplets": [
                    {"subject": "John", "action": "ate", "object": "apple"}
                ]
            },
            // ... more results
        ]
    }
    ```

### Event & Schema Generation Service (`event-llm-service`)

-   **Base URL**: `http://localhost:8003`
-   **Health Check**: `GET /health`
-   **Generate Events**: `POST /generate-events`
    ```json
    // Request Body (Simplified Example)
    {
        "input_data": [
            {
                "raw_text": "Barack Obama visited London on Monday.",
                "ner_results": [
                    {"text": "Barack Obama", "type": "PERSON", "start_char": 0, "end_char": 12},
                    {"text": "London", "type": "LOCATION", "start_char": 21, "end_char": 27},
                    {"text": "Monday", "type": "DATE", "start_char": 31, "end_char": 37}
                ],
                "soa_triplets": [
                    {"subject": "Barack Obama", "action": "visited", "object": "London"}
                ]
            }
        ]
    }
    // Response Body (Example - adheres to StandardizedEventSchema)
    {
        "results": [
            {
                "events": [
                    {
                        "event_type": "Visit",
                        "trigger": "visited",
                        "arguments": [
                            {"role": "Agent", "entity": {"text": "Barack Obama", "type": "PERSON"}},
                            {"role": "Location", "entity": {"text": "London", "type": "LOCATION"}},
                            {"role": "Time", "entity": {"text": "Monday", "type": "DATE"}}
                        ],
                        "metadata": {
                            "sentiment": "neutral",
                            "causality": null,
                            "certainty": "high"
                        }
                    }
                ],
                "entities": [
                    {"text": "Barack Obama", "type": "PERSON"},
                    {"text": "London", "type": "LOCATION"},
                    {"text": "Monday", "type": "DATE"}
                ]
            }
            // ... more results
        ]
    }
    ```

### Orchestrator Service (`orchestrator-service`)

-   **Base URL**: `http://localhost:8000`
-   **Health Check**: `GET /health`
-   **Process Single Text**: `POST /process-text`
    ```json
    // Request Body
    {
        "text": "Barack Obama visited London on Monday."
    }
    // Response Body (Adheres to StandardizedEventSchema)
    {
        "events": [
            {
                "event_type": "Visit",
                "trigger": "visited",
                "arguments": [
                    {"role": "Agent", "entity": {"text": "Barack Obama", "type": "PERSON"}},
                    {"role": "Location", "entity": {"text": "London", "type": "LOCATION"}},
                    {"role": "Time", "entity": {"text": "Monday", "type": "DATE"}}
                ],
                "metadata": {
                    "sentiment": "neutral",
                    "causality": null,
                    "certainty": "high"
                }
            }
        ],
        "entities": [
            {"text": "Barack Obama", "type": "PERSON"},
            {"text": "London", "type": "LOCATION"},
            {"text": "Monday", "type": "DATE"}
        ]
    }
    ```
-   **Process Batch Texts**: `POST /process-batch` (Asynchronous)
    ```json
    // Request Body
    {
        "texts": [
            "Barack Obama visited London on Monday.",
            "The company announced record profits yesterday."
        ]
    }
    // Response Body
    {
        "job_id": "unique-job-id-123",
        "message": "Batch processing initiated. Use /status/{job_id} to check progress."
    }
    ```
-   **Get Batch Status/Results**: `GET /status/{job_id}`
    ```json
    // Response Body (In Progress)
    {
        "job_id": "unique-job-id-123",
        "status": "processing",
        "progress": "50%",
        "results": []
    }
    // Response Body (Completed)
    {
        "job_id": "unique-job-id-123",
        "status": "completed",
        "results": [
            { /* StandardizedEventSchema for text 1 */ },
            { /* StandardizedEventSchema for text 2 */ }
        ]
    }
    ```

## CLI Usage

The unified CLI entry point is `src/main.py`.

### Process Single Text

Processes a single text string through the full pipeline and prints the JSON output.

```bash
python src/main.py process-single --text "Barack Obama visited London on Monday."
```

### Process File

Processes a file containing multiple texts (JSONL format with `{"id": "...", "text": "..."}` objects) and writes the final standardized JSON output to a specified file.

```bash
# Example input.jsonl
# {"id": "doc1", "text": "John ate an apple."}
# {"id": "doc2", "text": "Mary went to the park."}

python src/main.py process-file --input_path data/input.jsonl --output_path data/output.jsonl
```

## Configuration

All configurable parameters are externalized into `config/settings.yaml`. You can modify this file to adjust model paths, service ports, logging levels, etc.

## Logging

Each service implements structured JSON logging to persistent files within a mounted `logs` directory.
- `logs/ner_service.jsonl`
- `logs/dp_service.jsonl`
- `logs/event_llm_service.jsonl`
- `logs/orchestrator_service.jsonl`

## Error Handling

The pipeline incorporates robust error handling with consistent HTTP status codes for API responses, graceful handling of runtime errors, and detailed logging of exceptions. Docker health checks ensure services are truly ready.

## Development Practices

-   **Type Hinting**: Extensive use of type hints for code clarity and maintainability.
-   **Docstrings**: Comprehensive docstrings for all modules, classes, and public methods.
-   **Code Formatting**: Adherence to PEP 8, with `black` and `isort` recommended for consistent formatting.
-   **Dependencies**: Pinned versions in `requirements.txt` files for each service.
-   **Modularity**: Highly modular design following DRY and SOLID principles.