# src/core/celery_tasks.py
# File path: src/core/celery_tasks.py

import logging
import httpx  # For making HTTP requests to other microservices
import asyncio
import os
from celery import Celery
# Keep for Dask client setup
from celery.signals import worker_process_init, worker_process_shutdown
from dask.distributed import Client, LocalCluster
from typing import List, Dict, Any, Optional
from src.schemas.data_models import (
    NERPredictRequest, NERPredictResponse,
    DPExtractSOARequest, DPExtractSOAResponse,
    EventLLMInput, EventLLMGenerateResponse,
    CeleryBatchProcessTaskPayload, CelerySingleDocumentProcessPayload,
    CeleryTaskResult
)
from src.utils.config_manager import ConfigManager
from src.utils.logger import setup_logging
from urllib.parse import urljoin  # Import urljoin for robust URL construction
# CRITICAL FIX: Import StorageBackendFactory
from src.storage.backends import StorageBackendFactory 

# Load settings and configure logging for Celery worker
settings = ConfigManager.get_settings()
setup_logging()
logger = logging.getLogger("celery_worker")

# Initialize Celery app
celery_app = Celery(
    "pipeline_tasks",
    broker=settings.celery.broker_url,
    backend=settings.celery.result_backend
)

# Celery configuration
celery_app.conf.update(
    task_acks_late=settings.celery.task_acks_late,
    worker_prefetch_multiplier=settings.celery.worker_prefetch_multiplier,
    broker_connection_retry_on_startup=True,
    result_expires=settings.orchestrator_service.batch_processing_job_results_ttl
)

# Global Dask client (initialized per Celery worker process).
# The http_client global is removed as it's now managed locally per Dask task.
dask_client: Optional[Client] = None


@worker_process_init.connect
def setup_dask_client(sender=None, **kwargs):
    """
    Initializes Dask LocalCluster for each Celery worker process.
    HTTP client initialization is now handled dynamically within
    `process_single_document_pipeline_async` to avoid serialization issues
    when using Dask's multiprocessing mode.
    """
    global dask_client
    logger.info(
        f"Setting up Dask LocalCluster for Celery worker process {os.getpid()}")

    dask_n_workers = settings.celery.dask_local_cluster_n_workers
    if dask_n_workers is None:
        dask_n_workers = os.cpu_count()
        logger.info(
            f"Dask local cluster n_workers auto-detected to {dask_n_workers} (CPU count).")

    try:
        cluster = LocalCluster(
            n_workers=dask_n_workers,
            threads_per_worker=settings.celery.dask_local_cluster_threads_per_worker,
            memory_limit=settings.celery.dask_local_cluster_memory_limit,
            # Use processes if threads_per_worker is 1 (recommended for CPU-bound tasks)
            processes=True if settings.celery.dask_local_cluster_threads_per_worker == 1 else False,
            dashboard_address=None  # No need for a dashboard per worker process
        )
        dask_client = Client(cluster)
        logger.info(
            f"Dask LocalCluster initialized: {dask_client.dashboard_link}")
    except Exception as e:
        logger.error(
            f"Failed to initialize Dask LocalCluster: {e}", exc_info=True)
        dask_client = None


@worker_process_shutdown.connect
def teardown_dask_client(sender=None, **kwargs):
    """
    Shuts down Dask LocalCluster when a Celery worker process exits.
    """
    global dask_client
    logger.info(
        f"Shutting down Dask LocalCluster for Celery worker process {os.getpid()}")
    if dask_client:
        dask_client.close()
        dask_client = None


async def call_ner_service(http_client: httpx.AsyncClient, text: str) -> NERPredictResponse:
    """
    Calls the NER microservice.
    Accepts an httpx.AsyncClient instance explicitly.
    Converts HttpUrl from settings to string for urljoin.
    """
    try:
        # Convert Pydantic's HttpUrl object to a string for urljoin
        url = urljoin(
            str(settings.orchestrator_service.ner_service_url), "/predict")
        response = await http_client.post(
            url,
            json={"text": text}
        )
        response.raise_for_status()
        return NERPredictResponse.parse_obj(response.json())
    except httpx.HTTPStatusError as e:
        logger.error(
            f"NER service HTTP error for text '{text[:50]}...': {e.response.status_code} - {e.response.text}", exc_info=True)
        raise
    except httpx.RequestError as e:
        logger.error(
            f"NER service request error for text '{text[:50]}...': {e}", exc_info=True)
        raise


async def call_dp_service(http_client: httpx.AsyncClient, text: str) -> DPExtractSOAResponse:
    """
    Calls the DP microservice.
    Accepts an httpx.AsyncClient instance explicitly.
    Converts HttpUrl from settings to string for urljoin.
    """
    try:
        # Convert Pydantic's HttpUrl object to a string for urljoin
        url = urljoin(
            str(settings.orchestrator_service.dp_service_url), "/extract-soa")
        response = await http_client.post(
            url,
            json={"text": text}
        )
        response.raise_for_status()
        return DPExtractSOAResponse.parse_obj(response.json())
    except httpx.HTTPStatusError as e:
        logger.error(
            f"DP service HTTP error for text '{text[:50]}...': {e.response.status_code} - {e.response.text}", exc_info=True)
        raise
    except httpx.RequestError as e:
        logger.error(
            f"DP service request error for text '{text[:50]}...': {e}", exc_info=True)
        raise


async def call_event_llm_service(http_client: httpx.AsyncClient, llm_input: EventLLMInput) -> EventLLMGenerateResponse:
    """
    Calls the Event LLM microservice.
    Accepts an httpx.AsyncClient instance explicitly.
    Converts HttpUrl from settings to string for urljoin.
    """
    try:
        # Convert Pydantic's HttpUrl object to a string for urljoin
        url = urljoin(
            str(settings.orchestrator_service.event_llm_service_url), "/generate-events")
        request_timeout = settings.event_llm_service.request_timeout_seconds
        response = await http_client.post(
            url,
            json=llm_input.dict(),
            timeout=request_timeout
        )
        response.raise_for_status()
        return EventLLMGenerateResponse.parse_obj(response.json())
    except httpx.HTTPStatusError as e:
        logger.error(
            f"Event LLM service HTTP error for text '{llm_input.text[:50]}...': {e.response.status_code} - {e.response.text}", exc_info=True)
        raise
    except httpx.RequestError as e:
        logger.error(
            f"Event LLM service request error for text '{llm_input.text[:50]}...': {e}", exc_info=True)
        raise


async def process_single_document_pipeline_async(document_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Orchestrates calls to NER, DP, and Event LLM services for a single document.
    Designed to be run asynchronously and will create its own httpx.AsyncClient
    for each invocation to manage resources properly in a Dask multiprocessing environment.
    """
    doc_id = document_data.get("id", "N/A")
    text = document_data.get("text", "")

    if not text:
        logger.warning(f"Document {doc_id} has no text, skipping processing.")
        return {"id": doc_id, "error": "No text provided", "success": False}

    try:
        logger.info(
            f"Processing document {doc_id} (text len: {len(text)}) for pipeline.")

        # Crucial change: Initialize httpx.AsyncClient locally within the async function.
        # This prevents serialization issues when Dask sends tasks across processes.
        # The client will be properly closed when the 'async with' block exits.
        # Use the request_timeout_seconds from orchestrator_service settings for the default client timeout.
        async with httpx.AsyncClient(timeout=settings.orchestrator_service.request_timeout_seconds) as http_client_local:
            # 1. Call NER Service, passing the local http_client_local
            ner_response = await call_ner_service(http_client_local, text)

            # 2. Call DP Service, passing the local http_client_local
            dp_response = await call_dp_service(http_client_local, text)

            # 3. Prepare input for Event LLM Service
            llm_input = EventLLMInput(
                text=text,
                ner_entities=ner_response.entities,
                soa_triplets=dp_response.soa_triplets
            )

            # 4. Call Event LLM Service, passing the local http_client_local
            event_llm_response = await call_event_llm_service(http_client_local, llm_input)

        logger.info(f"Successfully processed document {doc_id}.")
        # Use .dict() for Celery serialization
        return {"id": doc_id, "result": event_llm_response.dict(), "success": True}

    except Exception as e:
        logger.error(
            f"Error processing document {doc_id} (text len: {len(text)}) for pipeline: {e}", exc_info=True)
        return {"id": doc_id, "error": str(e), "success": False}


@celery_app.task(bind=True)
def process_batch_task(self, payload_dict: Dict[str, Any]):
    """
    Celery task to process a batch of documents.
    Leverages Dask for parallel execution of single-document pipelines.
    CRITICAL FIX: Persists successful results to storage backends.
    """
    payload = CeleryBatchProcessTaskPayload.parse_obj(payload_dict)
    job_id = payload.job_id
    task_id = payload.task_id
    text_data = payload.text_data
    total_docs_in_chunk = len(text_data)

    logger.info(
        f"Celery task {task_id} (Job ID: {job_id}) started processing {total_docs_in_chunk} documents.")

    if not dask_client:
        logger.warning(
            "Dask client not initialized. Cannot run tasks in parallel. Processing sequentially.")
        results = []
        for doc in text_data:
            results.append(asyncio.run(
                process_single_document_pipeline_async(doc)))
    else:
        logger.info(
            f"Dispatching {total_docs_in_chunk} documents to Dask for parallel processing.")

        futures = [dask_client.submit(
            process_single_document_pipeline_async, doc) for doc in text_data]

        results = dask_client.gather(futures)
        logger.info(
            f"Dask parallel processing completed for {total_docs_in_chunk} documents.")

    # Prepare results for persistence and return
    processed_results: List[EventLLMGenerateResponse] = []
    failed_count = 0
    
    # CRITICAL FIX: Collect successful Pydantic objects for persistence
    successful_pydantic_objects: List[EventLLMGenerateResponse] = []

    for res in results:
        if res.get("success") and isinstance(res["result"], dict):
            try:
                # Parse the raw dict result back into a Pydantic model
                llm_response_model = EventLLMGenerateResponse.parse_obj(res["result"])
                successful_pydantic_objects.append(llm_response_model)
                # Store the dictionary representation for Celery result serialization
                processed_results.append(res["result"])
            except Exception as e:
                logger.error(
                    f"Result validation failed for document {res.get('id')}: {e}", exc_info=True)
                failed_count += 1
        else:
            failed_count += 1
            logger.error(
                f"Document {res.get('id')} failed: {res.get('error') or 'Unknown error'}")

    # --- Persistence Step (CRITICAL FIX) ---
    if successful_pydantic_objects:
        try:
            backends = StorageBackendFactory.get_backends()
            for backend in backends:
                backend.save_batch(successful_pydantic_objects)
            logger.info(
                f"Celery task {task_id} successfully persisted {len(successful_pydantic_objects)} results.")
        except Exception as e:
            logger.error(
                f"Failed to persist batch results for Job ID {job_id}: {e}", exc_info=True)
            # Persistence failure is a serious warning but shouldn't fail the task result itself,
            # as the results are still available in the Celery backend for the orchestrator.

    # Final counts for the Celery metadata payload
    successful_count = len(processed_results)
    
    logger.info(
        f"Celery task {task_id} (Job ID: {job_id}) finished. Processed {successful_count} successful, {failed_count} failed.")

    # Return result to Celery backend
    return {
        "job_id": job_id,
        "task_id": task_id,
        # processed_data is a list of dictionaries (ready for JSON serialization)
        "processed_data": processed_results, 
        "success": failed_count == 0,
        "error": f"{failed_count} documents failed." if failed_count > 0 else None,
        "processed_count": successful_count,
        "total_count": total_docs_in_chunk
    }


@celery_app.task
def download_models_task():
    """Celery task to download models for all services."""
    logger.info(
        "Triggered model download task (note: models typically download on service startup).")
    pass

# src/core/celery_tasks.py
# File path: src/core/celery_tasks.py
