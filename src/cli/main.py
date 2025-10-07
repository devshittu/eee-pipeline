# src/cli/main.py
# File path: src/cli/main.py

"""
CLI for the Event and Entity Extraction (EEE) Pipeline.
Uses Click framework for intuitive command structure and developer experience.

CRITICAL ENHANCEMENT: Supports enriched document processing with flexible schemas.
"""

import asyncio
import json
import os
import logging
from typing import Optional
import click
import httpx
import dask.bag as db
from dask.distributed import Client, LocalCluster
from src.utils.config_manager import ConfigManager
from src.utils.logger import setup_logging
from src.schemas.data_models import (
    ProcessTextRequest,
    ProcessBatchRequest,
    EnrichedDocumentRequest,
    EnrichedBatchRequest,
    BatchJobStatusResponse,
    EventLLMGenerateResponse,
    EnrichedDocumentResponse
)
from src.storage.backends import StorageBackendFactory

# Configure logging for CLI
setup_logging()
logger = logging.getLogger("eee_cli")

# Load settings
settings = ConfigManager.get_settings()
orchestrator_url = f"http://localhost:{settings.orchestrator_service.port}"


# ===========================
# CLI Context & Utilities
# ===========================

@click.group()
@click.version_option(version="1.1.0", prog_name="eee-cli")
@click.pass_context
def cli(ctx):
    """
    Event & Entity Extraction (EEE) Pipeline CLI.

    Provides commands for processing documents, managing batch jobs,
    and administering the pipeline services.

    New in v1.1: Support for enriched documents with flexible schemas.
    """
    ctx.ensure_object(dict)
    ctx.obj['orchestrator_url'] = orchestrator_url


# ===========================
# Document Commands
# ===========================

@cli.group()
def documents():
    """Manage document processing (single and batch)."""
    pass


@documents.command(name="process")
@click.argument("text", type=str)
@click.pass_context
def process_document(ctx, text: str):
    """
    Process a single document synchronously (simple text).

    TEXT: The input text string to process through the EEE pipeline.

    Example:
        eee-cli documents process "The UK government licensed its new AI tool..."
    """
    asyncio.run(_process_single_async(ctx.obj['orchestrator_url'], text))


@documents.command(name="process-enriched")
@click.argument("json_file", type=click.Path(exists=True))
@click.pass_context
def process_enriched_document(ctx, json_file: str):
    """
    Process a single enriched document with metadata from JSON file.

    JSON_FILE: Path to a JSON file containing the enriched document structure.

    Example:
        eee-cli documents process-enriched enriched_doc.json
    """
    asyncio.run(_process_enriched_async(
        ctx.obj['orchestrator_url'], json_file))


@documents.command(name="batch")
@click.argument("input_file", type=click.Path(exists=True))
@click.option("--output", "-o", type=click.Path(), required=True,
              help="Path to save the aggregated results (JSONL format).")
@click.option("--enriched", is_flag=True, default=False,
              help="Treat input as enriched documents (JSONL with full document objects).")
@click.pass_context
def process_batch(ctx, input_file: str, output: str, enriched: bool):
    """
    Submit a batch of documents for asynchronous processing.

    INPUT_FILE: Path to a JSONL file where each line is:
        - Simple mode: {"text": "..."}
        - Enriched mode: {full document object with configured fields}

    Examples:
        Simple:   eee-cli documents batch input.jsonl --output results.jsonl
        Enriched: eee-cli documents batch enriched.jsonl --output results.jsonl --enriched
    """
    asyncio.run(_process_batch_async(
        ctx.obj['orchestrator_url'], input_file, output, enriched))


# ===========================
# Job Commands
# ===========================

@cli.group()
def jobs():
    """Manage batch processing jobs."""
    pass


@jobs.command(name="status")
@click.argument("job_id", type=str)
@click.pass_context
def job_status(ctx, job_id: str):
    """
    Check the status of a batch processing job.

    JOB_ID: The unique identifier for the batch job.

    Example:
        eee-cli jobs status abc123-def456
    """
    asyncio.run(_get_job_status_async(
        ctx.obj['orchestrator_url'], job_id, poll=False))


@jobs.command(name="results")
@click.argument("job_id", type=str)
@click.option("--output", "-o", type=click.Path(),
              help="Path to save results (JSONL format). If omitted, prints to stdout.")
@click.pass_context
def job_results(ctx, job_id: str, output: Optional[str]):
    """
    Retrieve results from a completed batch job.

    JOB_ID: The unique identifier for the batch job.

    Example:
        eee-cli jobs results abc123-def456 --output results.jsonl
    """
    asyncio.run(_get_job_results_async(
        ctx.obj['orchestrator_url'], job_id, output))


# ===========================
# Admin Commands
# ===========================

@cli.group()
def admin():
    """Administrative commands for the EEE pipeline."""
    pass


@admin.command(name="download-models")
def download_models():
    """Pre-download all required models for offline use."""
    click.echo(
        "Model download triggered. Models download during service startup (see Docker entrypoints).")
    logger.info("Model download command executed.")


@admin.command(name="health")
@click.pass_context
def health_check(ctx):
    """Check the health of all pipeline services."""
    asyncio.run(_health_check_async(ctx.obj['orchestrator_url']))


@admin.command(name="show-config")
def show_config():
    """Display current document field mapping configuration."""
    from src.utils.config_manager import ConfigManager
    config = ConfigManager.get_settings().document_field_mapping

    click.echo("=== Document Field Mapping Configuration ===")
    click.echo(f"Text Field: {config.text_field}")
    click.echo(f"Fallback Fields: {', '.join(config.text_field_fallbacks)}")
    click.echo(f"Context Fields: {', '.join(config.context_fields)}")
    click.echo(f"Preserve in Output: {', '.join(config.preserve_in_output)}")


# ===========================
# Async Helper Functions
# ===========================

async def _process_single_async(orchestrator_url: str, text: str):
    """Processes a single text via orchestrator and persists the result."""
    async with httpx.AsyncClient(timeout=600.0) as client:
        try:
            logger.info(
                f"Sending single document (len: {len(text)}) to orchestrator.")
            response = await client.post(
                f"{orchestrator_url}/v1/documents",
                json=ProcessTextRequest(text=text).model_dump()
            )
            response.raise_for_status()
            result_dict = response.json()

            final_output = EventLLMGenerateResponse.parse_obj(result_dict)
            backends = StorageBackendFactory.get_backends()

            for backend in backends:
                backend.save(final_output)

            logger.info(
                "Successfully processed and persisted single document.")
            click.echo(json.dumps(final_output.model_dump(),
                       indent=2, ensure_ascii=False))

        except httpx.HTTPStatusError as e:
            logger.error(
                f"HTTP error: {e.response.status_code} - {e.response.text}", exc_info=True)
            click.echo(
                f"Error: {e.response.status_code} - {e.response.text}", err=True)
        except httpx.RequestError as e:
            logger.error(f"Network error: {e}", exc_info=True)
            click.echo(f"Network error: {e}", err=True)
        except Exception as e:
            logger.error(f"Unexpected error: {e}", exc_info=True)
            click.echo(f"Unexpected error: {e}", err=True)


async def _process_enriched_async(orchestrator_url: str, json_file: str):
    """Processes an enriched document from JSON file."""
    async with httpx.AsyncClient(timeout=600.0) as client:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                document = json.load(f)

            logger.info(
                f"Sending enriched document from {json_file} to orchestrator.")
            response = await client.post(
                f"{orchestrator_url}/v1/documents/enriched",
                json=document
            )
            response.raise_for_status()
            result_dict = response.json()

            final_output = EnrichedDocumentResponse.parse_obj(result_dict)
            backends = StorageBackendFactory.get_backends()

            for backend in backends:
                backend.save(final_output)

            logger.info(
                "Successfully processed and persisted enriched document.")
            click.echo(json.dumps(final_output.model_dump(),
                       indent=2, ensure_ascii=False))

        except FileNotFoundError:
            click.echo(f"Error: File not found: {json_file}", err=True)
        except json.JSONDecodeError as e:
            click.echo(f"Error: Invalid JSON in file: {e}", err=True)
        except httpx.HTTPStatusError as e:
            logger.error(
                f"HTTP error: {e.response.status_code} - {e.response.text}", exc_info=True)
            click.echo(
                f"Error: {e.response.status_code} - {e.response.text}", err=True)
        except httpx.RequestError as e:
            logger.error(f"Network error: {e}", exc_info=True)
            click.echo(f"Network error: {e}", err=True)
        except Exception as e:
            logger.error(f"Unexpected error: {e}", exc_info=True)
            click.echo(f"Unexpected error: {e}", err=True)


async def _process_batch_async(orchestrator_url: str, input_path: str, output_path: str, enriched: bool):
    """Processes a batch file via orchestrator and polls for completion."""
    if not os.path.exists(input_path):
        click.echo(f"Error: Input file not found at {input_path}", err=True)
        return

    logger.info(
        f"Starting batch processing for file: {input_path} (enriched={enriched})")

    dask_n_workers = settings.celery.dask_local_cluster_n_workers or os.cpu_count()
    cluster = None
    client_obj = None

    try:
        cluster = LocalCluster(
            n_workers=dask_n_workers,
            threads_per_worker=settings.celery.dask_local_cluster_threads_per_worker,
            memory_limit=settings.celery.dask_local_cluster_memory_limit,
            processes=True if settings.celery.dask_local_cluster_threads_per_worker == 1 else False,
            dashboard_address=None
        )
        client_obj = Client(cluster)
        logger.info(f"Dask LocalCluster initialized for CLI")

        def parse_json_line(line):
            try:
                return json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning(
                    f"Skipping malformed JSON line: {line[:100]}... Error: {e}")
                return None

        documents = db.read_text(input_path).map(
            parse_json_line).filter(lambda x: x is not None).compute()

        if not documents:
            click.echo("No valid documents found in the input file.", err=True)
            return

        logger.info(f"Loaded {len(documents)} documents from file.")

        async with httpx.AsyncClient(timeout=600.0) as http_client:
            if enriched:
                # Enriched batch request
                batch_request = EnrichedBatchRequest(documents=documents)
                endpoint = f"{orchestrator_url}/v1/documents/batch"
            else:
                # Simple batch request - extract text field
                texts = [doc.get("text", "") for doc in documents]
                texts = [t for t in texts if t]  # Filter empty
                if not texts:
                    click.echo("No valid texts found in documents.", err=True)
                    return
                batch_request = ProcessBatchRequest(texts=texts)
                endpoint = f"{orchestrator_url}/v1/documents/batch"

            response = await http_client.post(endpoint, json=batch_request.model_dump())
            response.raise_for_status()
            batch_info = response.json()
            job_id = batch_info["job_id"]
            click.echo(f"Batch processing initiated. Job ID: {job_id}")
            click.echo(f"Track status with: eee-cli jobs status {job_id}")

            await _poll_batch_status(http_client, orchestrator_url, job_id, output_path)

    except httpx.HTTPStatusError as e:
        logger.error(
            f"HTTP error: {e.response.status_code} - {e.response.text}", exc_info=True)
        click.echo(
            f"Error: {e.response.status_code} - {e.response.text}", err=True)
    except httpx.RequestError as e:
        logger.error(f"Network error: {e}", exc_info=True)
        click.echo(f"Network error: {e}", err=True)
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        click.echo(f"Unexpected error: {e}", err=True)
    finally:
        if client_obj:
            client_obj.close()
        if cluster:
            cluster.close()
        logger.info("Dask LocalCluster closed.")


async def _poll_batch_status(client: httpx.AsyncClient, orchestrator_url: str, job_id: str, output_path: Optional[str]):
    """Polls orchestrator for batch job status and saves results upon completion."""
    while True:
        try:
            response = await client.get(f"{orchestrator_url}/v1/jobs/{job_id}")
            response.raise_for_status()
            status_response = BatchJobStatusResponse.model_validate(
                response.json())

            click.echo(
                f"Job ID: {status_response.job_id}, Status: {status_response.status}", nl=False)
            if status_response.progress is not None:
                click.echo(
                    f", Progress: {status_response.progress:.2%}", nl=False)
            if status_response.processed_items is not None and status_response.total_items is not None:
                click.echo(
                    f" ({status_response.processed_items}/{status_response.total_items} items)", nl=False)
            click.echo()

            if status_response.status == "SUCCESS":
                if status_response.results and output_path:
                    try:
                        with open(output_path, 'w', encoding='utf-8') as f:
                            for res in status_response.results:
                                f.write(json.dumps(res.model_dump(),
                                        ensure_ascii=False) + '\n')
                        click.echo(
                            f"Batch processing completed successfully. Results saved to {output_path}")
                    except Exception as e:
                        logger.error(
                            f"Failed to write results to {output_path}: {e}", exc_info=True)
                        click.echo(
                            f"Warning: Failed to write results to {output_path}. Check logs.", err=True)
                else:
                    click.echo(
                        "Batch processing completed successfully. Results persisted by Celery tasks.")
                break
            elif status_response.status == "FAILURE":
                click.echo(
                    f"Batch processing failed. Error: {status_response.error}", err=True)
                break

            await asyncio.sleep(5)

        except httpx.HTTPStatusError as e:
            logger.error(
                f"HTTP error polling status: {e.response.status_code}", exc_info=True)
            click.echo(
                f"Error polling status: {e.response.status_code} - {e.response.text}", err=True)
            break
        except httpx.RequestError as e:
            logger.error(f"Network error polling status: {e}", exc_info=True)
            click.echo(f"Network error: {e}", err=True)
            break
        except Exception as e:
            logger.error(
                f"Unexpected error polling status: {e}", exc_info=True)
            click.echo(f"Unexpected error: {e}", err=True)
            break


async def _get_job_status_async(orchestrator_url: str, job_id: str, poll: bool = False):
    """Retrieves job status once or polls until completion."""
    async with httpx.AsyncClient(timeout=600.0) as client:
        try:
            response = await client.get(f"{orchestrator_url}/v1/jobs/{job_id}")
            response.raise_for_status()
            status_response = BatchJobStatusResponse.model_validate(
                response.json())

            click.echo(json.dumps(status_response.model_dump(),
                       indent=2, ensure_ascii=False))

            if poll and status_response.status == "PENDING":
                click.echo("\nPolling for updates...")
                await _poll_batch_status(client, orchestrator_url, job_id, None)

        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error: {e.response.status_code} - {e.response.text}", exc_info=True,
                         click.echo(
                             f"Error: {e.response.status_code} - {e.response.text}", err=True)
        except httpx.RequestError as e:
            logger.error(f"Network error: {e}", exc_info=True)
            click.echo(f"Network error: {e}", err=True)
        except Exception as e:
            logger.error(f"Unexpected error: {e}", exc_info=True)
            click.echo(f"Unexpected error: {e}", err=True)


async def _get_job_results_async(orchestrator_url: str, job_id: str, output_path: Optional[str]):
    """Retrieves job results and saves to file or prints to stdout."""
    async with httpx.AsyncClient(timeout=600.0) as client:
        try:
            response=await client.get(f"{orchestrator_url}/v1/jobs/{job_id}")
            response.raise_for_status()
            status_response=BatchJobStatusResponse.model_validate(
                response.json())

            if status_response.status != "SUCCESS":
                click.echo(
                    f"Job {job_id} is not completed yet. Current status: {status_response.status}", err=True)
                return

            if not status_response.results:
                click.echo(
                    f"Job {job_id} completed but has no results.", err=True)
                return

            if output_path:
                with open(output_path, 'w', encoding='utf-8') as f:
                    for res in status_response.results:
                        f.write(json.dumps(res.model_dump(),
                                ensure_ascii=False) + '\n')
                click.echo(f"Results saved to {output_path}")
            else:
                for res in status_response.results:
                    click.echo(json.dumps(res.model_dump(),
                               indent=2, ensure_ascii=False))

        except httpx.HTTPStatusError as e:
            logger.error(
                f"HTTP error: {e.response.status_code} - {e.response.text}", exc_info=True)
            click.echo(
                f"Error: {e.response.status_code} - {e.response.text}", err=True)
        except httpx.RequestError as e:
            logger.error(f"Network error: {e}", exc_info=True)
            click.echo(f"Network error: {e}", err=True)
        except Exception as e:
            logger.error(f"Unexpected error: {e}", exc_info=True)
            click.echo(f"Unexpected error: {e}", err=True)


async def _health_check_async(orchestrator_url: str):
    """Checks health of all pipeline services."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response=await client.get(f"{orchestrator_url}/health")
            response.raise_for_status()
            health_data=response.json()

            click.echo("=== EEE Pipeline Health Check ===")
            click.echo(
                f"Orchestrator Status: {health_data.get('status', 'unknown').upper()}")
            click.echo(
                f"Dependencies OK: {health_data.get('dependencies_ok', False)}")
            click.echo("\nService Status:")
            click.echo(
                f"  - NER Service: {health_data.get('ner_service', {}).get('status', 'unknown')}")
            click.echo(
                f"  - DP Service: {health_data.get('dp_service', {}).get('status', 'unknown')}")
            click.echo(
                f"  - Event LLM Service: {health_data.get('event_llm_service', {}).get('status', 'unknown')}")
            click.echo(
                f"  - Celery Broker: {'OK' if health_data.get('celery_broker_reachable') else 'UNAVAILABLE'}")

        except httpx.HTTPStatusError as e:
            logger.error(
                f"HTTP error: {e.response.status_code} - {e.response.text}", exc_info=True)
            click.echo(
                f"Health check failed: {e.response.status_code} - {e.response.text}", err=True)
        except httpx.RequestError as e:
            logger.error(f"Network error: {e}", exc_info=True)
            click.echo(
                f"Network error: Cannot reach orchestrator at {orchestrator_url}", err=True)
        except Exception as e:
            logger.error(f"Unexpected error: {e}", exc_info=True)
            click.echo(f"Unexpected error: {e}", err=True)


# ===========================
# Entry Point
# ===========================

if __name__ == "__main__":
    cli(obj={})


# src/cli/main.py
# File path: src/cli/main.py
