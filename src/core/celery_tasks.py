# src/core/celery_tasks.py
# File path: src/core/celery_tasks.py

import logging
import httpx
import asyncio
import os
from celery import Celery
from celery.signals import worker_process_init, worker_process_shutdown
from dask.distributed import Client, LocalCluster
from typing import List, Dict, Any, Optional
from src.schemas.data_models import (
    NERPredictRequest, NERPredictResponse,
    DPExtractSOARequest, DPExtractSOAResponse,
    EventLLMInput, EventLLMGenerateResponse,
    EnrichedDocumentResponse,
    CeleryBatchProcessTaskPayload
)
from src.utils.config_manager import ConfigManager
from src.utils.logger import setup_logging
from urllib.parse import urljoin
from src.storage.backends import StorageBackendFactory
from src.utils.document_processor import DocumentProcessor

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

# Global Dask client
dask_client: Optional[Client] = None

# Global document processor
document_processor = DocumentProcessor()


@worker_process_init.connect
def setup_dask_client(sender=None, **kwargs):
    """Initializes Dask LocalCluster for each Celery worker process."""
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
            processes=True if settings.celery.dask_local_cluster_threads_per_worker == 1 else False,
            dashboard_address=None
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
    """Shuts down Dask LocalCluster when a Celery worker process exits."""
    global dask_client
    logger.info(
        f"Shutting down Dask LocalCluster for Celery worker process {os.getpid()}")
    if dask_client:
        dask_client.close()
        dask_client = None


async def call_ner_service(http_client: httpx.AsyncClient, text: str) -> NERPredictResponse:
    """Calls the NER microservice."""
    try:
        url = urljoin(
            str(settings.orchestrator_service.ner_service_url), "/predict")
        response = await http_client.post(url, json={"text": text})
        response.raise_for_status()
        return NERPredictResponse.parse_obj(response.json())
    except httpx.HTTPStatusError as e:
        logger.error(
            f"NER service HTTP error: {e.response.status_code} - {e.response.text}", exc_info=True)
        raise
    except httpx.RequestError as e:
        logger.error(f"NER service request error: {e}", exc_info=True)
        raise


async def call_dp_service(http_client: httpx.AsyncClient, text: str) -> DPExtractSOAResponse:
    """Calls the DP microservice."""
    try:
        url = urljoin(
            str(settings.orchestrator_service.dp_service_url), "/extract-soa")
        response = await http_client.post(url, json={"text": text})
        response.raise_for_status()
        return DPExtractSOAResponse.parse_obj(response.json())
    except httpx.HTTPStatusError as e:
        logger.error(
            f"DP service HTTP error: {e.response.status_code} - {e.response.text}", exc_info=True)
        raise
    except httpx.RequestError as e:
        logger.error(f"DP service request error: {e}", exc_info=True)
        raise


async def call_event_llm_service(http_client: httpx.AsyncClient, llm_input: EventLLMInput) -> EventLLMGenerateResponse:
    """Calls the Event LLM microservice."""
    try:
        url = urljoin(
            str(settings.orchestrator_service.event_llm_service_url), "/generate-events")
        request_timeout = settings.event_llm_service.request_timeout_seconds
        response = await http_client.post(url, json=llm_input.dict(), timeout=request_timeout)
        response.raise_for_status()
        return EventLLMGenerateResponse.parse_obj(response.json())
    except httpx.HTTPStatusError as e:
        logger.error(
            f"Event LLM service HTTP error: {e.response.status_code} - {e.response.text}", exc_info=True)
        raise
    except httpx.RequestError as e:
        logger.error(f"Event LLM service request error: {e}", exc_info=True)
        raise


async def process_single_document_pipeline_async(document_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Orchestrates calls to NER, DP, and Event LLM services for a single document.

    CRITICAL ENHANCEMENT: Handles both simple text and enriched document structures.
    """
    doc_id = document_data.get("id", "N/A")
    is_enriched = document_data.get("enriched", False)

    try:
        # Branch based on document type
        if is_enriched:
            # Enriched document processing
            raw_document = document_data.get("document", {})
            logger.info(
                f"Processing enriched document {doc_id} with fields: {list(raw_document.keys())}")

            # Extract text and context using document processor
            enriched_input = document_processor.prepare_enriched_input(
                raw_document)
            text = enriched_input["text"]
            context_metadata = enriched_input["context_metadata"]
            upstream_entities = enriched_input["upstream_entities"]
            source_document = enriched_input["source_document"]
            document_id = enriched_input["document_id"]  # NEW

            logger.info(
                f"Extracted text (len: {len(text)}), context fields: {list(context_metadata.keys())}")

        else:
            # Simple text processing
            text = document_data.get("text", "")
            context_metadata = None
            upstream_entities = None
            source_document = None
            document_id = doc_id  # Use provided ID or N/A

            if not text:
                logger.warning(
                    f"Document {doc_id} has no text, skipping processing.")
                return {"id": doc_id, "error": "No text provided", "success": False}

            logger.info(
                f"Processing simple text document {doc_id} (len: {len(text)})")

        # Initialize HTTP client locally
        async with httpx.AsyncClient(timeout=settings.orchestrator_service.request_timeout_seconds) as http_client_local:
            # 1. Call NER Service
            ner_response = await call_ner_service(http_client_local, text)

            # Merge entities if enriched
            if is_enriched and upstream_entities:
                from src.schemas.data_models import Entity
                merged_entities = document_processor.merge_entities(
                    upstream_entities,
                    [e.dict() for e in ner_response.entities]
                )
                final_entities = [Entity.parse_obj(e) for e in merged_entities]
            else:
                final_entities = ner_response.entities

            # 2. Call DP Service
            dp_response = await call_dp_service(http_client_local, text)

            # 3. Prepare input for Event LLM Service
            llm_input = EventLLMInput(
                text=text,
                ner_entities=final_entities,
                soa_triplets=dp_response.soa_triplets,
                context_metadata=context_metadata  # Pass context for enriched docs
            )

            # 4. Call Event LLM Service
            event_llm_response = await call_event_llm_service(http_client_local, llm_input)

            # Inject document_id into response
            event_llm_response.document_id = document_id  # NEW

        logger.info(f"Successfully processed document {doc_id}.")

        # Format result based on document type
        if is_enriched:
            result = EnrichedDocumentResponse(
                **event_llm_response.dict(),
                source_document=source_document
            ).dict()
        else:
            result = event_llm_response.dict()

        return {"id": doc_id, "result": result, "success": True}

    except Exception as e:
        logger.error(f"Error processing document {doc_id}: {e}", exc_info=True)
        return {"id": doc_id, "error": str(e), "success": False}


@celery_app.task(bind=True)
def process_batch_task(self, payload_dict: Dict[str, Any]):
    """
    Celery task to process a batch of documents.
    Leverages Dask for parallel execution of single-document pipelines.

    CRITICAL ENHANCEMENT: Handles both simple texts and enriched documents.
    NEW: Performs event linking across batch to identify co-referent events.
    """
    payload = CeleryBatchProcessTaskPayload.parse_obj(payload_dict)
    job_id = payload.job_id
    task_id = payload.task_id
    text_data = payload.text_data
    total_docs_in_chunk = len(text_data)

    logger.info(
        f"Celery task {task_id} (Job ID: {job_id}) started processing {total_docs_in_chunk} documents.")

    # Step 1: Parallel document processing using Dask
    if not dask_client:
        logger.warning("Dask client not initialized. Processing sequentially.")
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

    # Step 2: Prepare results for persistence and validation
    processed_results: List[Dict[str, Any]] = []
    failed_count = 0
    successful_pydantic_objects: List[Any] = []

    for res in results:
        if res.get("success") and isinstance(res["result"], dict):
            try:
                # Detect result type and parse accordingly
                if "source_document" in res["result"]:
                    llm_response_model = EnrichedDocumentResponse.parse_obj(
                        res["result"])
                else:
                    llm_response_model = EventLLMGenerateResponse.parse_obj(
                        res["result"])

                successful_pydantic_objects.append(llm_response_model)
                processed_results.append(res["result"])
            except Exception as e:
                logger.error(
                    f"Result validation failed for document {res.get('id')}: {e}", exc_info=True)
                failed_count += 1
        else:
            failed_count += 1
            logger.error(
                f"Document {res.get('id')} failed: {res.get('error') or 'Unknown error'}")

    # Step 3: Event Linking - Link events across all documents in this batch
    if successful_pydantic_objects:
        try:
            from src.core.event_linker import EventLinker

            logger.info(
                f"Starting event linking for {len(successful_pydantic_objects)} documents...")

            # Initialize event linker with configurable parameters
            linker = EventLinker(
                entity_overlap_threshold=0.5,  # 50% entity overlap required
                temporal_window_days=7,         # Events within 7 days can be linked
                min_entities_for_match=2        # At least 2 shared entities required
            )

            # Collect all events with metadata from all documents
            events_with_metadata = []
            for response_obj in successful_pydantic_objects:
                # Ensure document_id exists (should be populated by pipeline)
                doc_id = getattr(response_obj, 'document_id', 'unknown')
                normalized_date = getattr(
                    response_obj, 'normalized_date', None)

                # Extract each event and create unique event_id
                for event_idx, event in enumerate(response_obj.events):
                    # Create deterministic event ID: document_id:trigger_text:start_char
                    event_id = f"{doc_id}:event_{event_idx}:{event.trigger.text}:{event.trigger.start_char}"

                    events_with_metadata.append({
                        "event": event,
                        "document_id": doc_id,
                        "normalized_date": normalized_date,
                        "event_id": event_id
                    })

            logger.info(
                f"Collected {len(events_with_metadata)} events from batch for linking.")

            # Perform event linking across batch
            linked_groups = linker.link_events_in_batch(events_with_metadata)

            logger.info(
                f"Event linking identified {len(linked_groups)} groups of related events.")

            # Step 4: Update event_references in response objects
            # Build reverse lookup: event_id -> response_obj
            event_to_response_map = {}
            for response_obj in successful_pydantic_objects:
                doc_id = getattr(response_obj, 'document_id', 'unknown')
                for event_idx, event in enumerate(response_obj.events):
                    event_id = f"{doc_id}:event_{event_idx}:{event.trigger.text}:{event.trigger.start_char}"
                    event_to_response_map[event_id] = (response_obj, event_idx)

            # Populate event_references field for linked events
            for canonical_event_id, related_event_ids in linked_groups.items():
                if canonical_event_id in event_to_response_map:
                    response_obj, event_idx = event_to_response_map[canonical_event_id]

                    # Store related event IDs (excluding self)
                    related_ids = [
                        eid for eid in related_event_ids if eid != canonical_event_id]

                    # Initialize event_references if not set
                    if not hasattr(response_obj, 'event_references') or response_obj.event_references is None:
                        response_obj.event_references = []

                    # Append related events to document-level event_references
                    # Note: This is a document-level field that aggregates all linked events
                    response_obj.event_references.extend(related_ids)

            # Deduplicate event_references (in case of multiple events in same doc)
            for response_obj in successful_pydantic_objects:
                if hasattr(response_obj, 'event_references') and response_obj.event_references:
                    response_obj.event_references = list(
                        set(response_obj.event_references))

            logger.info(
                f"Event linking complete. Updated event_references for {len(linked_groups)} documents.")

        except ImportError as e:
            logger.error(
                f"Failed to import EventLinker. Skipping event linking: {e}", exc_info=True)
        except Exception as e:
            logger.error(
                f"Error during event linking for Job ID {job_id}: {e}", exc_info=True)
            # Continue with persistence even if linking fails

    # Step 5: Persistence step - save to all configured backends
    if successful_pydantic_objects:
        try:
            backends = StorageBackendFactory.get_backends()
            for backend in backends:
                backend.save_batch(successful_pydantic_objects)
            logger.info(
                f"Celery task {task_id} successfully persisted {len(successful_pydantic_objects)} results to {len(backends)} backend(s).")
        except Exception as e:
            logger.error(
                f"Failed to persist batch results for Job ID {job_id}: {e}", exc_info=True)

    # Step 6: Prepare final task result
    successful_count = len(processed_results)
    logger.info(
        f"Celery task {task_id} (Job ID: {job_id}) finished. Processed {successful_count} successful, {failed_count} failed.")

    return {
        "job_id": job_id,
        "task_id": task_id,
        "processed_data": processed_results,
        "success": failed_count == 0,
        "error": f"{failed_count} documents failed." if failed_count > 0 else None,
        "processed_count": successful_count,
        "total_count": total_docs_in_chunk
    }



# @celery_app.task(bind=True)
# def process_batch_task(self, payload_dict: Dict[str, Any]):
#     """
#     Celery task to process a batch of documents.
#     Leverages Dask for parallel execution of single-document pipelines.

#     CRITICAL ENHANCEMENT: Handles both simple texts and enriched documents.
#     """
#     payload = CeleryBatchProcessTaskPayload.parse_obj(payload_dict)
#     job_id = payload.job_id
#     task_id = payload.task_id
#     text_data = payload.text_data
#     total_docs_in_chunk = len(text_data)

#     logger.info(
#         f"Celery task {task_id} (Job ID: {job_id}) started processing {total_docs_in_chunk} documents.")

#     if not dask_client:
#         logger.warning("Dask client not initialized. Processing sequentially.")
#         results = []
#         for doc in text_data:
#             results.append(asyncio.run(
#                 process_single_document_pipeline_async(doc)))
#     else:
#         logger.info(
#             f"Dispatching {total_docs_in_chunk} documents to Dask for parallel processing.")
#         futures = [dask_client.submit(
#             process_single_document_pipeline_async, doc) for doc in text_data]
#         results = dask_client.gather(futures)
#         logger.info(
#             f"Dask parallel processing completed for {total_docs_in_chunk} documents.")

#     # Prepare results for persistence and return
#     processed_results: List[Dict[str, Any]] = []
#     failed_count = 0
#     successful_pydantic_objects: List[Any] = []

#     for res in results:
#         if res.get("success") and isinstance(res["result"], dict):
#             try:
#                 # Detect result type and parse accordingly
#                 if "source_document" in res["result"]:
#                     llm_response_model = EnrichedDocumentResponse.parse_obj(
#                         res["result"])
#                 else:
#                     llm_response_model = EventLLMGenerateResponse.parse_obj(
#                         res["result"])

#                 successful_pydantic_objects.append(llm_response_model)
#                 processed_results.append(res["result"])
#             except Exception as e:
#                 logger.error(
#                     f"Result validation failed for document {res.get('id')}: {e}", exc_info=True)
#                 failed_count += 1
#         else:
#             failed_count += 1
#             logger.error(
#                 f"Document {res.get('id')} failed: {res.get('error') or 'Unknown error'}")

#     # Persistence step
#     if successful_pydantic_objects:
#         try:
#             backends = StorageBackendFactory.get_backends()
#             for backend in backends:
#                 backend.save_batch(successful_pydantic_objects)
#             logger.info(
#                 f"Celery task {task_id} successfully persisted {len(successful_pydantic_objects)} results.")
#         except Exception as e:
#             logger.error(
#                 f"Failed to persist batch results for Job ID {job_id}: {e}", exc_info=True)

#     successful_count = len(processed_results)
#     logger.info(
#         f"Celery task {task_id} (Job ID: {job_id}) finished. Processed {successful_count} successful, {failed_count} failed.")

#     return {
#         "job_id": job_id,
#         "task_id": task_id,
#         "processed_data": processed_results,
#         "success": failed_count == 0,
#         "error": f"{failed_count} documents failed." if failed_count > 0 else None,
#         "processed_count": successful_count,
#         "total_count": total_docs_in_chunk
#     }


@celery_app.task
def download_models_task():
    """Celery task to download models for all services."""
    logger.info("Triggered model download task.")
    pass


# src/core/celery_tasks.py
# File path: src/core/celery_tasks.py
