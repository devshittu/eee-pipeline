# src/core/orchestration_logic.py
import httpx
import asyncio
from typing import List, Dict, Any, Tuple
from src.schemas.data_models import (
    NERResult, NERPredictResponse, DPExtractSOAResponse,
    LLMGenerateEventsRequest, LLMInputData, LLMGenerateEventsResponse,
    StandardizedEventSchema, Entity, SOATriplet, ArgumentEntity
)
from src.utils.logger import logger
from src.utils.common_utils import get_service_url


class Orchestrator:
    """
    Manages the orchestration of calls to NER, DP, and Event LLM services.
    """

    def __init__(self, settings: Any):
        self.settings = settings
        self.ner_service_url = get_service_url("ner-service", settings)
        self.dp_service_url = get_service_url("dp-service", settings)
        self.event_llm_service_url = get_service_url(
            "event-llm-service", settings)
        self.request_timeout = settings.orchestrator.request_timeout_seconds
        self.http_client = httpx.AsyncClient(timeout=self.request_timeout)

    async def _call_ner_service(self, texts: List[str], trace_id: str) -> List[NERResult]:
        """Calls the NER service to get named entities."""
        logger.info(f"Calling NER service for {len(texts)} texts.", extra_data={
                    "trace_id": trace_id})
        try:
            response = await self.http_client.post(
                f"{self.ner_service_url}/predict",
                json={"texts": texts},
                headers={"X-Trace-ID": trace_id}
            )
            response.raise_for_status()
            return NERPredictResponse.model_validate(response.json()).results
        except httpx.HTTPStatusError as e:
            logger.error(f"NER service responded with error status {e.response.status_code}: {e.response.text}",
                         extra_data={"trace_id": trace_id, "status_code": e.response.status_code, "response_body": e.response.text})
            raise RuntimeError(f"NER service failed: {e.response.text}")
        except httpx.RequestError as e:
            logger.error(f"Failed to connect to NER service: {e}", exc_info=True, extra_data={
                         "trace_id": trace_id})
            raise RuntimeError(f"Failed to connect to NER service: {e}")
        except Exception as e:
            logger.error(f"Unexpected error calling NER service: {e}", exc_info=True, extra_data={
                         "trace_id": trace_id})
            raise

    async def _call_dp_service(self, texts: List[str], trace_id: str) -> List[SOATriplet]:
        """Calls the DP service to get S-O-A triplets."""
        logger.info(f"Calling DP service for {len(texts)} texts.", extra_data={
                    "trace_id": trace_id})
        try:
            response = await self.http_client.post(
                f"{self.dp_service_url}/extract-soa",
                json={"texts": texts},
                headers={"X-Trace-ID": trace_id}
            )
            response.raise_for_status()
            # DP service returns List[DPResult] where each has text and soa_triplets
            # We need to extract just the triplets, potentially mapping back to original text index
            dp_results = DPExtractSOAResponse.model_validate(
                response.json()).results
            # This simplification assumes a 1:1 mapping and returns list of triplets for each text.
            # For `process_single` it's one text, for `process_batch` we need to manage lists of lists.
            # The current implementation returns a list of results, so we can unpack it.
            return [res.soa_triplets for res in dp_results]
        except httpx.HTTPStatusError as e:
            logger.error(f"DP service responded with error status {e.response.status_code}: {e.response.text}",
                         extra_data={"trace_id": trace_id, "status_code": e.response.status_code, "response_body": e.response.text})
            raise RuntimeError(f"DP service failed: {e.response.text}")
        except httpx.RequestError as e:
            logger.error(f"Failed to connect to DP service: {e}", exc_info=True, extra_data={
                         "trace_id": trace_id})
            raise RuntimeError(f"Failed to connect to DP service: {e}")
        except Exception as e:
            logger.error(f"Unexpected error calling DP service: {e}", exc_info=True, extra_data={
                         "trace_id": trace_id})
            raise

    async def _call_event_llm_service(self, llm_input_data: List[LLMInputData], trace_id: str) -> List[StandardizedEventSchema]:
        """Calls the Event LLM service to generate events and schema."""
        logger.info(f"Calling Event LLM service for {len(llm_input_data)} items.", extra_data={
                    "trace_id": trace_id})
        try:
            response = await self.http_client.post(
                f"{self.event_llm_service_url}/generate-events",
                json=LLMGenerateEventsRequest(
                    input_data=llm_input_data).model_dump(),
                headers={"X-Trace-ID": trace_id}
            )
            response.raise_for_status()
            return LLMGenerateEventsResponse.model_validate(response.json()).results
        except httpx.HTTPStatusError as e:
            logger.error(f"Event LLM service responded with error status {e.response.status_code}: {e.response.text}",
                         extra_data={"trace_id": trace_id, "status_code": e.response.status_code, "response_body": e.response.text})
            raise RuntimeError(f"Event LLM service failed: {e.response.text}")
        except httpx.RequestError as e:
            logger.error(f"Failed to connect to Event LLM service: {e}", exc_info=True, extra_data={
                         "trace_id": trace_id})
            raise RuntimeError(f"Failed to connect to Event LLM service: {e}")
        except Exception as e:
            logger.error(f"Unexpected error calling Event LLM service: {e}", exc_info=True, extra_data={
                         "trace_id": trace_id})
            raise

    async def process_single_text(self, text: str, trace_id: str) -> StandardizedEventSchema:
        """
        Orchestrates the pipeline for a single text input.
        """
        logger.info(f"Processing single text: '{text[:50]}...'", extra_data={
                    "trace_id": trace_id})

        # 1. Call NER Service
        ner_results_list = await self._call_ner_service(texts=[text], trace_id=trace_id)
        ner_results = ner_results_list[0].entities if ner_results_list else []
        logger.debug(f"NER results: {ner_results}",
                     extra_data={"trace_id": trace_id})

        # 2. Call DP Service
        dp_results_list_of_lists = await self._call_dp_service(texts=[text], trace_id=trace_id)
        # dp_results_list_of_lists will be like [[soa_triplet_1, soa_triplet_2], ...]
        # We need the inner list for a single text.
        soa_triplets = dp_results_list_of_lists[0] if dp_results_list_of_lists else [
        ]
        logger.debug(f"DP (SOA) results: {soa_triplets}", extra_data={
                     "trace_id": trace_id})

        # 3. Prepare input for Event LLM Service
        llm_input = LLMInputData(
            raw_text=text,
            ner_results=ner_results,
            soa_triplets=soa_triplets
        )

        # 4. Call Event LLM Service
        event_llm_results = await self._call_event_llm_service(input_data=[llm_input], trace_id=trace_id)
        final_output = event_llm_results[0] if event_llm_results else StandardizedEventSchema(
            events=[], entities=[])
        logger.info(f"Successfully processed single text: '{text[:50]}...'", extra_data={
                    "trace_id": trace_id})
        return final_output

    async def process_batch_texts(self, texts: List[str], trace_id: str) -> List[StandardizedEventSchema]:
        """
        Orchestrates the pipeline for a batch of text inputs.
        This version processes sequentially, but can be parallelized within a service.
        """
        logger.info(f"Processing batch of {len(texts)} texts.", extra_data={
                    "trace_id": trace_id})

        # Process NER and DP in parallel for the whole batch
        # This sends two batch requests to upstream services.
        ner_task = self._call_ner_service(texts=texts, trace_id=trace_id)
        dp_task = self._call_dp_service(texts=texts, trace_id=trace_id)

        ner_results_all: List[NERResult] = []
        dp_soa_triplets_all: List[List[SOATriplet]] = []

        try:
            ner_results_all, dp_soa_triplets_all = await asyncio.gather(ner_task, dp_task)
            logger.debug(f"Received NER and DP results for batch.",
                         extra_data={"trace_id": trace_id})
        except Exception as e:
            logger.error(
                f"Error during parallel NER/DP calls for batch: {e}", exc_info=True, extra_data={"trace_id": trace_id})
            raise RuntimeError(f"Batch processing failed during NER/DP: {e}")

        # Prepare input for Event LLM Service
        llm_input_data: List[LLMInputData] = []
        for i, text in enumerate(texts):
            # Ensure safe access, handle cases where a service might return fewer results (though they should match input size)
            current_ner_entities = ner_results_all[i].entities if i < len(
                ner_results_all) and ner_results_all[i] else []
            current_soa_triplets = dp_soa_triplets_all[i] if i < len(
                dp_soa_triplets_all) else []

            llm_input_data.append(
                LLMInputData(
                    raw_text=text,
                    ner_results=current_ner_entities,
                    soa_triplets=current_soa_triplets
                )
            )

        # Call Event LLM Service for the batch
        event_llm_results = await self._call_event_llm_service(input_data=llm_input_data, trace_id=trace_id)
        logger.info(f"Successfully processed batch of {len(texts)} texts.", extra_data={
                    "trace_id": trace_id})
        return event_llm_results

    async def close(self):
        """Closes the HTTP client session."""
        await self.http_client.aclose()
