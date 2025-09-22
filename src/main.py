# src/main.py
# File path: src/main.py

import argparse
import asyncio
import json
import os
import logging
from typing import List, Dict, Any
import httpx
from src.utils.config_manager import ConfigManager
from src.utils.logger import setup_logging
from src.schemas.data_models import ProcessTextRequest, ProcessBatchRequest, BatchJobStatusResponse, EventLLMGenerateResponse
import dask.bag as db
from dask.distributed import Client, LocalCluster

# Configure logging for CLI
setup_logging()
logger = logging.getLogger("cli_main")

# Load settings
settings = ConfigManager.get_settings()
orchestrator_url = f"http://localhost:{settings.orchestrator_service.port}"


async def process_single_cli(text: str):
    """Processes a single text string via the Orchestrator API."""
    async with httpx.AsyncClient(timeout=600.0) as client:
        try:
            logger.info(
                f"Sending single text (len: {len(text)}) to Orchestrator service: {orchestrator_url}/process-text")
            response = await client.post(
                f"{orchestrator_url}/process-text",
                # FIXED: Use model_dump()
                json=ProcessTextRequest(text=text).model_dump()
            )
            response.raise_for_status()
            result = response.json()
            logger.info("Successfully processed single text.")
            print(json.dumps(result, indent=2, ensure_ascii=False))
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
    Processes a file containing multiple texts in bulk via the Orchestrator API
    using Dask for local parallelism before dispatching.
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

        # Read file using Dask Bag for parallelization
        # Assumes input_path is JSONL format: {"id": "...", "text": "..."}
        # If it's just plain text lines, modify db.read_text and parsing

        # Read the file and parse each line as JSON
        # Error handling for malformed JSON lines
        def parse_json_line(line):
            try:
                return json.loads(line)
            except json.JSONDecodeError as e:
                logger.warning(
                    f"Skipping malformed JSON line in input file: {line[:100]}... Error: {e}")
                return None

        texts_bag = db.read_text(input_path).map(
            parse_json_line).filter(lambda x: x is not None)

        # Collect all texts to send to Orchestrator (as a single batch request)
        # Or, chunk and send multiple batch requests to Orchestrator (more scalable for huge files)
        # For simplicity, sending as one large list if it fits.

        # For very large files, it's better to chunk the Dask Bag and send multiple
        # batch requests to the Orchestrator, rather than collecting all at once.
        # This will depend on the `batch_processing_chunk_size` of the Orchestrator.

        # Example: Collect all texts as a list (might be memory intensive for huge files)
        all_texts_data = texts_bag.compute()
        texts_only = [item["text"]
                      for item in all_texts_data if "text" in item]

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
                json=batch_request.model_dump()  # FIXED: Use model_dump()
            )
            response.raise_for_status()
            batch_info = response.json()
            job_id = batch_info["job_id"]
            print(f"Batch processing initiated. Job ID: {job_id}")
            print(
                f"Track status with: python src/main.py status --job_id {job_id}")

            # Optionally, poll for status until completion
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


async def poll_batch_status(job_id: str, client: httpx.AsyncClient, output_path: str):
    """Polls the orchestrator for batch job status and saves results."""
    while True:
        try:
            response = await client.get(f"{orchestrator_url}/status/{job_id}")
            response.raise_for_status()
            status_response = BatchJobStatusResponse.model_validate(
                response.json())  # FIXED: Use model_validate

            print(
                f"Job ID: {status_response.job_id}, Status: {status_response.status}", end="")
            if status_response.progress is not None:
                print(f", Progress: {status_response.progress:.2%}", end="")
            if status_response.processed_items is not None and status_response.total_items is not None:
                print(
                    f" ({status_response.processed_items}/{status_response.total_items} items processed)", end="")
            print()

            if status_response.status == "SUCCESS":
                if status_response.results:
                    with open(output_path, 'w', encoding='utf-8') as f:
                        for res in status_response.results:
                            f.write(json.dumps(
                                res.model_dump(), ensure_ascii=False) + '\n')  # FIXED: Use model_dump()
                    print(
                        f"Batch processing completed successfully. Results saved to {output_path}")
                else:
                    print(
                        "Batch processing completed successfully, but no results returned.")
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
    # This command is primarily for pre-caching models locally or within containers
    # For Docker Compose setup, this is largely handled by entrypoint scripts.
    # However, if running CLI without Docker Compose, this would be useful.
    logger.info("Initiating model download. This might take some time.")

    # Call a Celery task that coordinates model downloads
    # For now, we'll just log a message as actual download is in entrypoints
    # In a real scenario, you would have a Celery task that signals each service to download models
    # Or, the entrypoint scripts themselves are responsible for ensuring models exist.
    # Given the Docker setup, models download when their respective containers start.
    # If this CLI command is meant for a *local* dev environment without Docker,
    # then it would manually call the model download logic for each service:
    # from src.core.ner_logic import NERModel
    # NERModel() # This would trigger download
    # from src.core.dp_logic import DPModel
    # DPModel() # This would trigger download
    # from src.core.event_llm_logic import EventLLMModel
    # EventLLMModel() # This would trigger download

    # For now, just a placeholder. The primary model download mechanism is via Docker entrypoints.
    print("Model download command executed. Models are typically downloaded during service startup (Docker entrypoints).")
    print("For CLI-only local model download, you would need to implement direct calls to model loading logic for each service.")


def main():
    parser = argparse.ArgumentParser(
        description="CLI for the Event and Entity Extraction Pipeline.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Process single text command
    process_single_parser = subparsers.add_parser(
        "process-single", help="Process a single text string.")
    process_single_parser.add_argument(
        "--text", type=str, required=True, help="The input text string.")

    # Process file command
    process_file_parser = subparsers.add_parser(
        "process-file", help="Process a file containing multiple texts (JSONL format).")
    process_file_parser.add_argument(
        "--input_path", type=str, required=True, help="Path to the input JSONL file.")
    process_file_parser.add_argument(
        "--output_path", type=str, required=True, help="Path to save the output JSONL file.")

    # Download models command
    download_models_parser = subparsers.add_parser(
        "download-models", help="Pre-download all necessary Hugging Face and SpaCy/Stanza models for offline use.")

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
        # FIXED: For status only, no output_path needed; adjust call
        asyncio.run(poll_batch_status(
            args.job_id, httpx.AsyncClient(timeout=600.0), None))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()


# src/main.py
# File path: src/main.py
