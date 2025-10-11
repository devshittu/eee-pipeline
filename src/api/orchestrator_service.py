# src/api/orchestrator_service.py
# File path: src/api/orchestrator_service.py

import logging
import httpx
import uuid
import math
import redis
from fastapi import FastAPI, HTTPException, status, BackgroundTasks
from pydantic import ValidationError
from fastapi import UploadFile, File
import json
from typing import List, Dict, Any, Optional, Union
from urllib.parse import urljoin
from src.schemas.data_models import (
    ProcessTextRequest, ProcessTextResponse,
    EnrichedDocumentRequest, EnrichedDocumentResponse,
    EnrichedBatchRequest,
    NERPredictRequest, NERPredictResponse,
    DPExtractSOARequest, DPExtractSOAResponse,
    EventLLMInput, EventLLMGenerateResponse,
    ProcessBatchRequest, BatchJobStatusResponse,
    CeleryBatchProcessTaskPayload
)
from src.core.celery_tasks import celery_app, process_batch_task
from src.utils.config_manager import ConfigManager
from src.utils.logger import setup_logging
from src.utils.document_processor import DocumentProcessor

# Load settings and configure logging
settings = ConfigManager.get_settings()
setup_logging()
logger = logging.getLogger("orchestrator_service")

# Initialize FastAPI app
app = FastAPI(
    title="EEE Pipeline Orchestrator Service",
    description="Central entry point for the Event & Entity Extraction pipeline, orchestrating microservices.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Initialize HTTP client for internal service calls
http_client = httpx.AsyncClient(timeout=600.0)

# Initialize Redis client for job metadata
redis_client = redis.Redis.from_url(
    settings.celery.broker_url, decode_responses=True)

# Initialize document processor for enriched requests
document_processor = DocumentProcessor()


@app.on_event("startup")
async def startup_event():
    """Startup event for the Orchestrator service."""
    logger.info("Orchestrator Service starting up...")
    try:
        ner_url = urljoin(
            str(settings.orchestrator_service.ner_service_url), "health")
        dp_url = urljoin(
            str(settings.orchestrator_service.dp_service_url), "health")
        llm_url = urljoin(
            str(settings.orchestrator_service.event_llm_service_url), "health")

        await http_client.get(ner_url)
        await http_client.get(dp_url)
        await http_client.get(llm_url)
        logger.info("Successfully connected to dependent services.")
    except httpx.RequestError as e:
        logger.error(
            f"Failed to connect to dependent services at startup: {e}", exc_info=True)
        pass
    logger.info("Orchestrator Service startup complete.")


@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown event for the Orchestrator service."""
    logger.info("Orchestrator Service shutting down...")
    await http_client.aclose()
    redis_client.close()
    logger.info("Orchestrator Service shutdown complete.")


@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    """Health check endpoint."""
    try:
        ner_url = urljoin(
            str(settings.orchestrator_service.ner_service_url), "health")
        dp_url = urljoin(
            str(settings.orchestrator_service.dp_service_url), "health")
        llm_url = urljoin(
            str(settings.orchestrator_service.event_llm_service_url), "health")

        logger.info("Performing health check on dependent services...")
        logger.debug(
            f"Checking health: ner={ner_url}, dp={dp_url}, llm={llm_url}")

        ner_health = await http_client.get(ner_url)
        dp_health = await http_client.get(dp_url)
        llm_health = await http_client.get(llm_url)

        ner_health.raise_for_status()
        dp_health.raise_for_status()
        llm_health.raise_for_status()

        return {
            "status": "ok",
            "dependencies_ok": True,
            "ner_service": ner_health.json(),
            "dp_service": dp_health.json(),
            "event_llm_service": llm_health.json(),
            "celery_broker_reachable": True
        }
    except httpx.HTTPStatusError as e:
        logger.error(
            f"Dependent service returned error status: {e.response.url} - {e.response.status_code}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                            detail=f"Dependent service unavailable: {e.response.url} returned {e.response.status_code}")
    except httpx.RequestError as e:
        logger.error(
            f"Failed to connect to dependent service: {e.request.url} - {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                            detail=f"Failed to connect to dependent service: {e.request.url}")
    except Exception as e:
        logger.error(
            f"Unexpected error during health check: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail="Internal server error during health check.")


# ===========================
# V1 API Routes (RESTful)
# ===========================

@app.post("/v1/documents", response_model=ProcessTextResponse, status_code=status.HTTP_200_OK)
async def process_document(request: ProcessTextRequest):
    """
    Processes a single document (simple text string) through the full pipeline.
    """
    text = request.text
    logger.info(
        f"Received request to process single document (len: {len(text)}).")

    try:
        # 1. Call NER Service
        ner_url = urljoin(
            str(settings.orchestrator_service.ner_service_url), "predict")
        ner_response = await http_client.post(ner_url, json={"text": text})
        ner_response.raise_for_status()
        ner_results = NERPredictResponse.parse_obj(ner_response.json())
        logger.debug(
            f"NER service responded with {len(ner_results.entities)} entities.")

        # 2. Call DP Service
        dp_url = urljoin(
            str(settings.orchestrator_service.dp_service_url), "extract-soa")
        dp_response = await http_client.post(dp_url, json={"text": text})
        dp_response.raise_for_status()
        dp_results = DPExtractSOAResponse.parse_obj(dp_response.json())
        logger.debug(
            f"DP service responded with {len(dp_results.soa_triplets)} S-O-A triplets.")

        # 3. Prepare input for Event LLM Service (no context metadata for simple request)
        llm_input = EventLLMInput(
            text=text,
            ner_entities=ner_results.entities,
            soa_triplets=dp_results.soa_triplets,
            context_metadata=None
        )
        logger.debug("Prepared input for Event LLM service.")

        # 4. Call Event LLM Service
        llm_url = urljoin(
            str(settings.orchestrator_service.event_llm_service_url), "generate-events")
        llm_response = await http_client.post(llm_url, json=llm_input.dict())
        llm_response.raise_for_status()
        final_output = EventLLMGenerateResponse.parse_obj(llm_response.json())
        logger.info(
            f"Successfully processed single document (len: {len(text)}).")

        return ProcessTextResponse(**final_output.dict())

    except ValidationError as e:
        logger.warning(
            f"Invalid request/response payload: {e.errors()}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail=f"Invalid data format: {e.errors()}")
    except httpx.HTTPStatusError as e:
        logger.error(
            f"Error from upstream service {e.response.url}: {e.response.status_code} - {e.response.text}", exc_info=True)
        raise HTTPException(status_code=e.response.status_code,
                            detail=f"Upstream service error: {e.response.text}")
    except httpx.RequestError as e:
        logger.error(
            f"Network error calling upstream service {e.request.url}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                            detail=f"Could not reach upstream service: {e.request.url}")
    except Exception as e:
        logger.error(
            f"Internal server error during single document processing: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error.")


# ===========================
# CRITICAL NEW ENDPOINT: Enriched Document Processing
# ===========================

@app.post("/v1/documents/enriched", status_code=status.HTTP_202_ACCEPTED)
async def process_enriched_document(request: EnrichedDocumentRequest):
    """
    Processes a document with rich metadata structure asynchronously.
    Returns immediately with job tracking information.
    
    **CRITICAL**: Enriched documents require LLM processing which takes 5-10 minutes.
    This endpoint submits the job and returns a tracking URL immediately.
    
    Use GET /v1/jobs/{job_id} to poll for results.
    """
    # Generate job ID
    job_id = str(uuid.uuid4())

    # Convert single document to batch format
    batch_request = EnrichedBatchRequest(
        documents=[request.dict()],
        job_id=job_id
    )

    logger.info(
        f"Received enriched document, submitting as async job {job_id}")

    # Submit to batch processor
    background_tasks = BackgroundTasks()
    result = await submit_batch_job(batch_request, background_tasks)

    # Return with helpful polling instructions
    return {
        "job_id": job_id,
        "status": "SUBMITTED",
        "message": "Document submitted for async processing. Results will be available in 5-10 minutes.",
        "status_endpoint": f"/v1/jobs/{job_id}",
        "polling_command": f"curl http://localhost:8000/v1/jobs/{job_id}",
        "estimated_completion_minutes": 7
    }



@app.post("/v1/documents/enriched/upload", status_code=status.HTTP_202_ACCEPTED)
async def upload_enriched_document(file: UploadFile = File(...)):
    """
    Upload a JSON or JSONL file with enriched documents for async processing.
    
    Supports:
    - Single JSON file: {"document_id": "...", "cleaned_text": "...", ...}
    - JSONL file: Multiple documents, one JSON per line
    
    Returns immediately with job tracking information.
    """
    try:
        content = await file.read()
        content_str = content.decode('utf-8')

        # Detect format
        documents = []
        if file.filename.endswith('.jsonl'):
            # Parse JSONL
            for line in content_str.strip().split('\n'):
                if line.strip():
                    documents.append(json.loads(line))
            logger.info(f"Parsed {len(documents)} documents from JSONL file")
        else:
            # Parse single JSON
            doc = json.loads(content_str)
            documents = [doc]
            logger.info(f"Parsed single document from JSON file")

        # Submit as batch
        job_id = str(uuid.uuid4())
        batch_request = EnrichedBatchRequest(
            documents=documents,
            job_id=job_id
        )

        background_tasks = BackgroundTasks()
        await submit_batch_job(batch_request, background_tasks)

        return {
            "job_id": job_id,
            "status": "SUBMITTED",
            "documents_count": len(documents),
            "message": f"Submitted {len(documents)} document(s) for async processing.",
            "status_endpoint": f"/v1/jobs/{job_id}",
            "polling_command": f"curl http://localhost:8000/v1/jobs/{job_id}",
            "estimated_completion_minutes": len(documents) * 7
        }

    except json.JSONDecodeError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid JSON format: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Error processing file upload: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing upload: {str(e)}"
        )

@app.post("/v1/documents/batch", response_model=Dict[str, str], status_code=status.HTTP_202_ACCEPTED)
async def submit_batch_job(request: Union[ProcessBatchRequest, EnrichedBatchRequest], background_tasks: BackgroundTasks):
    """
    Submits a batch of documents for asynchronous processing.
    Supports both simple text arrays and enriched document arrays.
    
    **Route Convention**: POST /v1/documents/batch (RESTful batch resource creation)
    """
    job_id = request.job_id

    # Detect request type and prepare payload
    if isinstance(request, ProcessBatchRequest):
        # Simple text batch
        texts_to_process = [{"id": f"{job_id}-{i}", "text": text, "enriched": False}
                            for i, text in enumerate(request.texts)]
        logger.info(
            f"Received simple batch request for Job ID {job_id} with {len(request.texts)} texts.")
    else:
        # Enriched document batch
        texts_to_process = [{"id": f"{job_id}-{i}", "document": doc, "enriched": True}
                            for i, doc in enumerate(request.documents)]
        logger.info(
            f"Received enriched batch request for Job ID {job_id} with {len(request.documents)} documents.")

    chunk_size = settings.orchestrator_service.batch_processing_chunk_size
    total_chunks = math.ceil(len(texts_to_process) / chunk_size)

    task_ids = []
    try:
        for i in range(0, len(texts_to_process), chunk_size):
            chunk = texts_to_process[i:i + chunk_size]
            task_payload = CeleryBatchProcessTaskPayload(
                job_id=job_id,
                task_id=str(uuid.uuid4()),
                text_data=chunk
            )
            task_result = process_batch_task.apply_async(
                args=[task_payload.dict()],
                task_id=task_payload.task_id
            )
            task_ids.append(task_result.id)
            logger.debug(
                f"Dispatched Celery task {task_result.id} for chunk {i//chunk_size + 1}/{total_chunks} of Job ID {job_id}.")

        redis_key = f"job:{job_id}:tasks"
        redis_client.rpush(redis_key, *task_ids)
        redis_client.expire(
            redis_key, settings.orchestrator_service.batch_processing_job_results_ttl)
    except Exception as e:
        logger.error(
            f"Failed to dispatch tasks for Job ID {job_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to initiate batch processing: {str(e)}"
        )

    return {
        "job_id": job_id,
        "message": f"Batch processing initiated. Use GET /v1/jobs/{job_id} to track progress.",
        "status_endpoint": f"/v1/jobs/{job_id}"
    }


@app.get("/v1/jobs/{job_id}", response_model=BatchJobStatusResponse, status_code=status.HTTP_200_OK)
async def get_job_status(job_id: str):
    """
    Returns the status and results of a batch processing job.
    """
    logger.info(f"Received status request for Job ID {job_id}.")

    try:
        redis_key = f"job:{job_id}:tasks"
        task_ids = redis_client.lrange(redis_key, 0, -1)
        if not task_ids:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail=f"Job ID {job_id} not found."
            )

        total_items = 0
        processed_items = 0
        results = []
        errors = []
        all_success = True
        all_completed = True
        for task_id in task_ids:
            task_result = celery_app.AsyncResult(task_id)
            task_state = task_result.state

            if task_state == "SUCCESS":
                task_data = task_result.get()
                total_items += task_data.get("total_count", 0)
                processed_items += task_data.get("processed_count", 0)
                results.extend(task_data.get("processed_data", []))
                if task_data.get("error"):
                    errors.append(task_data.get("error"))
                    all_success = False
            elif task_state == "FAILURE":
                all_success = False
                all_completed = False
                errors.append(str(task_result.info))
            else:
                all_completed = False

        progress = processed_items / total_items if total_items > 0 else 0.0
        status_str = "SUCCESS" if all_success and all_completed else "FAILURE" if errors else "PENDING"

        # Parse results as either EventLLMGenerateResponse or EnrichedDocumentResponse
        parsed_results = []
        for r in results:
            try:
                if "source_document" in r:
                    parsed_results.append(
                        EnrichedDocumentResponse.parse_obj(r))
                else:
                    parsed_results.append(
                        EventLLMGenerateResponse.parse_obj(r))
            except Exception as e:
                logger.error(f"Failed to parse result: {e}", exc_info=True)
                continue

        return BatchJobStatusResponse(
            job_id=job_id,
            status=status_str,
            results=parsed_results,
            error="; ".join(errors) if errors else None,
            progress=progress,
            total_items=total_items,
            processed_items=processed_items
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            f"Error fetching status for Job ID {job_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error fetching job status: {str(e)}"
        )


# ===========================
# Legacy Routes (Deprecated - Redirect to V1)
# ===========================

@app.post("/process-text", response_model=ProcessTextResponse, status_code=status.HTTP_200_OK, deprecated=True)
async def legacy_process_text(request: ProcessTextRequest):
    """**DEPRECATED**: Use POST /v1/documents instead."""
    logger.warning(
        "Legacy endpoint /process-text called. Redirecting to /v1/documents.")
    return await process_document(request)


@app.post("/process-batch", response_model=Dict[str, str], status_code=status.HTTP_202_ACCEPTED, deprecated=True)
async def legacy_process_batch(request: ProcessBatchRequest, background_tasks: BackgroundTasks):
    """**DEPRECATED**: Use POST /v1/documents/batch instead."""
    logger.warning(
        "Legacy endpoint /process-batch called. Redirecting to /v1/documents/batch.")
    return await submit_batch_job(request, background_tasks)


@app.get("/status/{job_id}", response_model=BatchJobStatusResponse, status_code=status.HTTP_200_OK, deprecated=True)
async def legacy_get_status(job_id: str):
    """**DEPRECATED**: Use GET /v1/jobs/{job_id} instead."""
    logger.warning(
        f"Legacy endpoint /status/{job_id} called. Redirecting to /v1/jobs/{job_id}.")
    return await get_job_status(job_id)

# src/api/orchestrator_service.py
# File path: src/api/orchestrator_service.py
