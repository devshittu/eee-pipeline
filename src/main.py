# src/main.py
# File path: src/main.py

import argparse
import asyncio
import json
import os
import logging
from typing import List, Dict, Any, Optional
import httpx
from src.utils.config_manager import ConfigManager
from src.utils.logger import setup_logging
from src.schemas.data_models import ProcessTextRequest, ProcessBatchRequest, BatchJobStatusResponse, EventLLMGenerateResponse
import dask.bag as db
from dask.distributed import Client, LocalCluster
# Import factory for direct use
from src.storage.backends import StorageBackendFactory, JSONLStorageBackend

# Configure logging for CLI
setup_logging()
logger = logging.getLogger("cli_main")

# Load settings
settings = ConfigManager.get_settings()
orchestrator_url = f"http://localhost:{settings.orchestrator_service.port}"


async def process_single_cli(text: str):
    """
    Processes a single text string via the Orchestrator API.
    CRITICAL FIX: Persists the result using the StorageBackendFactory after receiving the output.
    """
    async with httpx.AsyncClient(timeout=600.0) as client:
        try:
            logger.info(
                f"Sending single text (len: {len(text)}) to Orchestrator service: {orchestrator_url}/process-text")
            response = await client.post(
                f"{orchestrator_url}/process-text",
                json=ProcessTextRequest(text=text).model_dump()
            )
            response.raise_for_status()
            result_dict = response.json()

            # --- Persistence of Single Result ---
            final_output = EventLLMGenerateResponse.parse_obj(result_dict)
            backends = StorageBackendFactory.get_backends()

            # Persist to all configured backends
            for backend in backends:
                backend.save(final_output)

            logger.info(
                "Successfully processed and persisted single text result.")
            print(json.dumps(final_output.model_dump(),
                  indent=2, ensure_ascii=False))

        except httpx.HTTPStatusError as e:
            logger.error(
                f"HTTP error processing single text: {e.response.status_code} - {e.response.text}", exc_info=True)
            print(f"Error: {e.response.status_code} - {e.response.text}")
        except httpx.RequestError as e:
            logger.error(
                f"Network error processing single text: {e}", exc_info=True)
            print(f"Network error: {e}")
        except Exception as e:
            logger.error(f"An unexpected error occurred: {e}", exc_info=True)
            print(f"An unexpected error occurred: {e}")


async def process_file_cli(input_path: str, output_path: str):
    """
    Processes a file containing multiple texts in bulk via the Orchestrator API.
    Dispatches to Celery, and initiates status polling.
    """
    if not os.path.exists(input_path):
        logger.error(f"Input file not found: {input_path}")
        print(f"Error: Input file not found at {input_path}")
        return

    logger.info(f"Starting batch processing for file: {input_path}")

    # Use Dask LocalCluster for parallel reading and preprocessing of the file
    dask_n_workers = settings.celery.dask_local_cluster_n_workers
    if dask_n_workers is None:
        dask_n_workers = os.cpu_count()  # Auto-detect CPU cores

    cluster = None
    client = None
    try:
        cluster = LocalCluster(
            n_workers=dask_n_workers,
            threads_per_worker=settings.celery.dask_local_cluster_threads_per_worker,
            memory_limit=settings.celery.dask_local_cluster_memory_limit,
            processes=True if settings.celery.dask_local_cluster_threads_per_worker == 1 else False,
            dashboard_address=None  # No need for dashboard in CLI
        )
        client = Client(cluster)
        logger.info(
            f"Dask LocalCluster initialized for CLI: {client.dashboard_link if client.dashboard_link else 'N/A'}")

        def parse_json_line(line):
            try:
                # Assumes input format is JSONL: {"id": "...", "text": "..."}
                data = json.loads(line)
                return data.get("text")
            except json.JSONDecodeError as e:
                logger.warning(
                    f"Skipping malformed JSON line in input file: {line[:100]}... Error: {e}")
                return None
            except Exception:
                # Skip if the 'text' field is missing or invalid
                return None

        texts_only = db.read_text(input_path).map(
            parse_json_line).filter(lambda x: x is not None).compute()

        if not texts_only:
            logger.warning("No valid texts found in the input file.")
            print("No valid texts found in the input file to process.")
            return

        logger.info(
            f"Initiating batch processing for {len(texts_only)} documents via Orchestrator API.")

        # Send the batch request to the Orchestrator Service
        async with httpx.AsyncClient(timeout=600.0) as http_client_orchestrator:
            batch_request = ProcessBatchRequest(texts=texts_only)
            response = await http_client_orchestrator.post(
                f"{orchestrator_url}/process-batch",
                json=batch_request.model_dump()
            )
            response.raise_for_status()
            batch_info = response.json()
            job_id = batch_info["job_id"]
            print(f"Batch processing initiated. Job ID: {job_id}")
            print(
                f"Track status with: python src/main.py status --job_id {job_id}")

            # Poll for status until completion. The output_path is passed here
            # to enable writing the aggregated JSONL file at the end.
            await poll_batch_status(job_id, http_client_orchestrator, output_path)

    except httpx.HTTPStatusError as e:
        logger.error(
            f"HTTP error during file batch processing: {e.response.status_code} - {e.response.text}", exc_info=True)
        print(f"Error: {e.response.status_code} - {e.response.text}")
    except httpx.RequestError as e:
        logger.error(
            f"Network error during file batch processing: {e}", exc_info=True)
        print(f"Network error: {e}")
    except Exception as e:
        logger.error(
            f"An unexpected error occurred during file batch processing: {e}", exc_info=True)
        print(f"An unexpected error occurred: {e}")
    finally:
        if client:
            client.close()
        if cluster:
            cluster.close()
        logger.info("Dask LocalCluster closed.")


async def poll_batch_status(job_id: str, client: httpx.AsyncClient, output_path: Optional[str]):
    """
    Polls the orchestrator for batch job status and, upon completion, 
    saves aggregated results to the specified output path using the JSONL backend logic.
    """
    while True:
        try:
            response = await client.get(f"{orchestrator_url}/status/{job_id}")
            response.raise_for_status()
            status_response = BatchJobStatusResponse.model_validate(
                response.json())

            print(
                f"Job ID: {status_response.job_id}, Status: {status_response.status}", end="")
            if status_response.progress is not None:
                print(f", Progress: {status_response.progress:.2%}", end="")
            if status_response.processed_items is not None and status_response.total_items is not None:
                print(
                    f" ({status_response.processed_items}/{status_response.total_items} items processed)", end="")
            print()

            if status_response.status == "SUCCESS":
                if status_response.results and output_path:
                    # CRITICAL FIX: Manually handle writing the aggregated file for the CLI here.
                    # This ensures the CLI output file matches the expected structure,
                    # while the Celery task handles persistence to databases/daily files.
                    try:
                        with open(output_path, 'w', encoding='utf-8') as f:
                            for res in status_response.results:
                                f.write(json.dumps(
                                    res.model_dump(), ensure_ascii=False) + '\n')
                        print(
                            f"Batch processing completed successfully. Aggregated results saved to {output_path}")
                    except Exception as e:
                        logger.error(
                            f"Failed to write aggregated results to {output_path}: {e}", exc_info=True)
                        print(
                            f"Warning: Failed to write aggregated results to {output_path}. Check logs.")
                else:
                    print(
                        "Batch processing completed successfully. Results were persisted by Celery tasks.")
                break
            elif status_response.status == "FAILURE":
                print(
                    f"Batch processing failed. Error: {status_response.error}")
                break

            await asyncio.sleep(5)  # Poll every 5 seconds

        except httpx.HTTPStatusError as e:
            logger.error(
                f"HTTP error polling status for job {job_id}: {e.response.status_code} - {e.response.text}", exc_info=True)
            print(
                f"Error polling status: {e.response.status_code} - {e.response.text}")
            break
        except httpx.RequestError as e:
            logger.error(
                f"Network error polling status for job {job_id}: {e}", exc_info=True)
            print(f"Network error polling status: {e}")
            break
        except Exception as e:
            logger.error(
                f"An unexpected error occurred while polling status for job {job_id}: {e}", exc_info=True)
            print(f"An unexpected error occurred: {e}")
            break


def download_models_cli():
    """Triggers model download for all services locally."""
    # Placeholder implementation remains the same.
    print("Model download command executed. Models are typically downloaded during service startup (Docker entrypoints).")


def main():
    parser = argparse.ArgumentParser(
        description="CLI for the Event and Entity Extraction Pipeline.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Process single text command
    process_single_parser = subparsers.add_parser(
        "process-single", help="Process a single text string and persist the result.")
    process_single_parser.add_argument(
        "--text", type=str, required=True, help="The input text string.")

    # Process file command
    process_file_parser = subparsers.add_parser(
        "process-file", help="Process a file containing multiple texts (JSONL format) and persist/output results.")
    process_file_parser.add_argument(
        "--input_path", type=str, required=True, help="Path to the input JSONL file.")
    process_file_parser.add_argument(
        "--output_path", type=str, required=True, help="Path to save the final aggregated output JSONL file.")

    # Download models command
    download_models_parser = subparsers.add_parser(
        "download-models", help="Pre-download all necessary models for offline use.")

    # Status command for batch jobs
    status_parser = subparsers.add_parser(
        "status", help="Get the status of a batch processing job.")
    status_parser.add_argument(
        "--job_id", type=str, required=True, help="The ID of the batch job to query.")

    args = parser.parse_args()

    if args.command == "process-single":
        asyncio.run(process_single_cli(args.text))
    elif args.command == "process-file":
        asyncio.run(process_file_cli(args.input_path, args.output_path))
    elif args.command == "download-models":
        download_models_cli()
    elif args.command == "status":
        # For status, output_path is not relevant, so we pass None
        asyncio.run(poll_batch_status(
            args.job_id, httpx.AsyncClient(timeout=600.0), None))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()


# src/main.py
# File path: src/main.py
