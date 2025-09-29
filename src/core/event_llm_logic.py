# src/core/event_llm_logic.py
# File path: src/core/event_llm_logic.py

import logging
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from typing import List, Dict, Any, Optional, Literal
from src.schemas.data_models import EventLLMInput, EventLLMGenerateResponse, Event, EventArgument, ArgumentEntity, EventMetadata, TextSpan, Entity, SOATriplet
from src.utils.config_manager import ConfigManager
import json
import re
from pydantic import ValidationError
from collections import defaultdict
import uuid

# CRITICAL MODULARITY FIXES: Import logic from the new dedicated files
from src.core.llm_domains import determine_domain, get_domain_examples, get_domain_persona
# Import the new post-processor
from src.core.llm_postprocessor import LLMPostProcessor

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, before_sleep_log, RetryCallState

# Logger for the event LLM service
logger = logging.getLogger("event_llm_service")

# FIX: Define a simple custom exception for retry logic


class LLMGenerationError(Exception):
    """Custom exception raised when LLM generation or output parsing fails."""
    pass

# CRITICAL MODULARITY FIX: map_llm_output_fields is moved to LLMPostProcessor class.
# We retain _extract_first_json_block as a static utility for raw text parsing.


def _extract_first_json_block(text: str) -> Optional[str]:
    """
    Universally extracts the first complete, top-level JSON object from a string,
    handling trailing noise. Uses bracket counting for robustness.
    """
    start_index = text.find('{')
    if start_index == -1:
        return None

    balance = 0
    end_index = -1
    in_string = False
    escape = False

    for i, char in enumerate(text[start_index:]):

        if char == '\\':
            escape = not escape
            continue

        if not escape:
            if char == '"':
                in_string = not in_string
            elif not in_string:
                if char == '{':
                    balance += 1
                elif char == '}':
                    balance -= 1
                    if balance == 0:
                        end_index = start_index + i
                        break
        escape = False

    if end_index != -1:
        raw_json = text[start_index:end_index + 1]

        # Final validation and cleanup: ensure it parses
        try:
            # We parse and re-dump to ensure the returned string is clean, canonical, and guaranteed parsable
            return json.dumps(json.loads(raw_json))
        except (json.JSONDecodeError, Exception):
            return None  # Parsing failed, abandon this block

    return None


class EventLLMModel:
    """
    Singleton class to load and manage the Event LLM model.
    Ensures the model is loaded only once.
    """
    _instance = None
    # Reverting to original attribute names (model, tokenizer) for compatibility with API/health checks.
    model = None
    tokenizer = None
    _config_manager = None

    def __new__(cls, *args, **kwargs):
        """
        Ensures a single instance of EventLLMModel is created (Singleton pattern).
        """
        if cls._instance is None:
            cls._instance = super(EventLLMModel, cls).__new__(cls)
            cls._instance._init_model()
        return cls._instance

    def _init_model(self):
        """Initializes the LLM model and tokenizer."""
        try:
            # FIX: Correctly access the settings from the ConfigManager singleton
            model_config = ConfigManager.get_settings().event_llm_service
            general_config = ConfigManager.get_settings().general  # Get GPU flag

            logger.info("Loading model and tokenizer...")
            self.model_name = model_config.model_name
            self.model_path = model_config.model_path
            self.max_new_tokens = model_config.max_new_tokens
            self.temperature = model_config.temperature
            self.top_p = model_config.top_p
            self.generation_max_retries = model_config.generation_max_retries
            self.generation_retry_delay_seconds = model_config.generation_retry_delay_seconds
            # FIX: Initialize chunking parameters from config
            self.chunk_size_tokens = model_config.chunk_size_tokens
            self.overlap_size_tokens = model_config.overlap_size_tokens

            # --- Load Tokenizer and Model (Using public attributes for API compatibility) ---

            # FIX: Use use_fast=False to mitigate OverflowError/compatibility issues with fast tokenizers
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=model_config.model_cache_dir,
                use_fast=False  # Enforce slow tokenizer for robustness against OverflowError/protobuf issues
            )

            if torch.cuda.is_available() and general_config.gpu_enabled:
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.bfloat16
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    quantization_config=bnb_config,
                    device_map="auto",
                    cache_dir=model_config.model_cache_dir,
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    cache_dir=model_config.model_cache_dir,
                )

            # Ensure padding token is set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            logger.info(f"Model {self.model_name} loaded successfully.")

        except Exception as e:
            logger.error(f"Error initializing LLM model: {e}", exc_info=True)
            # CRITICAL FIX: Ensure the model/tokenizer are explicitly set to None
            # so subsequent calls fail fast and correctly.
            self.model = None
            self.tokenizer = None
            raise  # Re-raise to ensure the service startup failure is logged and handled

    def _get_prompt(self, input_data: EventLLMInput, mode: str = "chunk", aggregated_response: Optional[EventLLMGenerateResponse] = None) -> str:
        """
        Generates a structured few-shot prompt for the LLM.
        The prompt's instructions adapt based on the 'mode' ('chunk' or 'synthesis').
        """
        # CRITICAL CHECK: Ensure LLM is loaded before generating prompt
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("LLM model or tokenizer is not initialized.")

        # CRITICAL FIX: Dynamically determine the domain and select examples using the new module
        domain = determine_domain(input_data.text)
        examples_to_use = get_domain_examples(domain)
        persona = get_domain_persona(domain)

        # --- CRITICAL METADATA INSTRUCTION ---
        metadata_instruction = (
            "6.  **STRICT METADATA:** You **MUST NOT** use `null`, empty string (`\"\"`), or `None` values for the `sentiment` or `causality` fields in the `metadata` object. "
            "If sentiment cannot be determined (e.g., purely factual), use `neutral`. If causality is not explicit, infer the most likely cause/effect from the sentence. "
            "If you cannot infer a strong, unique causality, summarize the event itself as the causality. **NEVER** return `null` or `\"\"` for these fields."
        )

        if mode == "chunk":

            system_instruction = (
                f"You are {persona}. Your task is to analyze the provided text and "
                "extract all relevant events and entities based strictly on the content of the text. "
                "You must respond with a single, valid JSON object that adheres to the Pydantic schema for `EventLLMGenerateResponse`.\n\n"
                "Key rules:\n"
                "1.  **NO HALLUCINATION:** Only extract information explicitly mentioned in the text.\n"
                "2.  **STRICT JSON FORMAT:** The response must be a single JSON object matching the schema `EventLLMGenerateResponse`.\n"
                "3.  **ACCURATE OFFSETS:** Ensure all character offsets (`start_char`, `end_char`) are precise and correct.\n"
                "4.  **RELEVANT INFORMATION ONLY:** Do not include any information not present in the text.\n"
                "5.  **MULTI-ENTITY ARGUMENTS:** For argument roles that refer to multiple entities (e.g., 'recipients', 'parties'), use the `entities` field instead of the singular `entity` field in the JSON output.\n"
                f"{metadata_instruction}\n"
            )
            # Few-shot examples for chunk-level extraction
            examples_prompt = ""
            for example in examples_to_use:
                examples_prompt += f"Input:\n{json.dumps(example['input'], indent=2)}\n\n"
                examples_prompt += f"Output:\n{json.dumps(example['output'], indent=2)}\n\n"
            user_input_prompt = f"Input:\n{json.dumps(input_data.dict(exclude={'events', 'extracted_entities', 'extracted_soa_triplets'}), indent=2)}\n\nOutput:\n"
            return system_instruction + examples_prompt + user_input_prompt

        elif mode == "synthesis" and aggregated_response is not None:
            # The synthesis prompt
            synthesis_instruction = (
                "You are a sophisticated document summarization and event synthesis engine. Your task is to review the following "
                "aggregated data (low-level events and entities) and identify the **1 to 3 single, most significant main events** and **key entities** "
                "that define the core narrative. You must synthesize the result into a single, valid JSON object matching the `EventLLMGenerateResponse` schema. "
                "DO NOT include the original document text in the input to save tokens. \n\n"
                "Key rules:\n"
                "1. **HIGH-LEVEL SYNTHESIS:** Your output must represent the main narrative (1-3 events MAX).\n"
                "2. **STRICT JSON FORMAT:** The response must be a single JSON object matching the schema.\n"
                "3. **ACCURATE OFFSETS:** Retain original character offsets relative to the initial long document.\n"
                "4. **SOA TRIPLETS:** Omit extracted_soa_triplets from your final synthesized output.\n"
                "5. **MULTI-ENTITY ARGUMENTS:** For argument roles that refer to multiple entities (e.g., 'recipients', 'parties'), use the `entities` field instead of the singular `entity` field in the JSON output.\n"
                # Apply strict metadata rule here too
                f"{metadata_instruction}\n"
            )

            # FIX: Only pass aggregated data, NOT the entire original document text
            synthesis_data_for_llm = {
                "Extracted_Low_Level_Events": [e.dict() for e in aggregated_response.events],
                "Extracted_Entities": [e.dict() for e in aggregated_response.extracted_entities],
            }
            user_input_prompt = f"Review Data:\n{json.dumps(synthesis_data_for_llm, indent=2)}\n\nSynthesized Output (Events and Key Entities ONLY):\n"
            return synthesis_instruction + user_input_prompt

        else:
            raise ValueError(f"Invalid mode: {mode}")

    def _call_llm(self, prompt: str) -> str:
        """
        Calls the LLM with the generated prompt and returns the raw text response.
        This version is more robust in handling different output formats.
        """
        # CRITICAL CHECK: Ensure LLM is loaded before calling
        if self.model is None or self.tokenizer is None:
            raise RuntimeError(
                "LLM model or tokenizer is not initialized. Cannot call LLM.")

        inputs = self.tokenizer(
            prompt, return_tensors="pt", max_length=self.tokenizer.model_max_length, truncation=True).to(self.model.device)
        output = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,  # Use deterministic generation for IE
            temperature=self.temperature,
            top_p=self.top_p,
            pad_token_id=self.tokenizer.eos_token_id
        )
        response_text = self.tokenizer.decode(
            output[0], skip_special_tokens=True)

        # --- FINAL FIX: Universal JSON Block Extraction ---

        # 1. Search for JSON block surrounded by triple backticks (most reliable for clean output)
        json_match = re.search(
            r'```json\s*(\{.*?\})\s*```', response_text, re.DOTALL)
        if json_match:
            return json_match.group(1).strip()

        # 2. Isolate the prediction area using the final delimiter 'Output:\n'
        output_delimiter = "Output:\n"
        last_output_index = response_text.rfind(output_delimiter)

        search_text = response_text
        if last_output_index != -1:
            # If the delimiter is found, search only the content after it
            search_text = response_text[last_output_index +
                                        len(output_delimiter):].strip()

        # 3. Use the bracket counting function to extract the first complete top-level JSON block
        clean_json_str = _extract_first_json_block(search_text)

        if clean_json_str:
            return clean_json_str  # Already parsed and canonized inside _extract_first_json_block

        # 4. Final Fallback (Grab the cleanest JSON structure, whether it parses or not)
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            return json_match.group(0).strip()

        return response_text.strip()

    @retry(
        stop=stop_after_attempt(ConfigManager.get_settings(
        ).event_llm_service.generation_max_retries),
        wait=wait_exponential(multiplier=1, min=ConfigManager.get_settings(
        ).event_llm_service.generation_retry_delay_seconds, max=10),
        # Retry on both ValidationError and our custom error
        retry=retry_if_exception_type((ValidationError, LLMGenerationError)),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    def _generate_response_for_chunk(self, input_data: EventLLMInput) -> EventLLMGenerateResponse:
        """
        Generates events and entities for a SINGLE text chunk.
        This method is now a private helper with retry logic.
        """
        raw_response = ""
        try:
            prompt = self._get_prompt(input_data, mode="chunk")
            logger.info(
                f"Generating response for input chunk: {input_data.text[:100]}...")
            raw_response = self._call_llm(prompt)

            # Sanitize response (stripping non-json noise outside the LLM cleanup)
            raw_response = raw_response.strip().lstrip("```json").rstrip("```").strip()

            # CRITICAL FIX: Change log level from ERROR to DEBUG/INFO for the raw response.
            # We use DEBUG for the raw output dump, as this is purely a diagnostic log.
            logger.debug(f"Raw LLM response (pre-parse): {raw_response}")

            # FIX: Robust JSON loading
            try:
                json_data = json.loads(raw_response)
            # Catch general parsing errors, including OverflowError and JSONDecodeError sources
            except Exception as e:
                # We now raise our custom exception (LLMGenerationError), which is retryable
                error_msg = f"LLM output parsing failed ({type(e).__name__}: {e})"
                logger.error(error_msg)
                raise LLMGenerationError(error_msg) from e

            # --- CRITICAL FIX 3: Delegate post-processing, validation, and imputation ---
            # This is where we run the new LLMPostProcessor to map fields, validate, and fill nulls.
            # Imputation is removed from LLMPostProcessor, so it focuses on validation/cleanup.
            response_model = LLMPostProcessor.post_process_response(json_data)

            return response_model

        except ValidationError as e:
            # Re-raise ValidationError to trigger tenacity retry mechanism
            logger.error(f"Validation error for chunk: {e}")
            logger.error(f"Failing raw response: {raw_response}")
            raise
        except Exception as e:
            # This catch is for unexpected errors outside of Validation/JSON decoding
            error_msg = f"Unexpected fatal runtime error in LLM chunk processing: {type(e).__name__}: {e}"
            logger.error(error_msg, exc_info=True)

            # Since this is a final catch, we assume the parsing issues are frequent and retryable
            # If it's truly unexpected (like a memory error), tenacity will eventually stop.
            raise LLMGenerationError(error_msg) from e

    def _chunk_text(self, text: str) -> List[Dict[str, Any]]:
        """
        Splits a long text into smaller chunks for sequential processing.
        Uses a sliding window approach with a specified overlap to preserve context.
        """
        # CRITICAL CHECK: Check if tokenizer is None due to initialization failure
        if self.tokenizer is None:
            raise RuntimeError(
                "Tokenizer is None. LLM model initialization failed.")

        tokens = self.tokenizer.encode(text)
        total_tokens = len(tokens)
        chunks = []

        # FIX: Get chunk parameters from self (config)
        max_tokens = self.chunk_size_tokens
        overlap = self.overlap_size_tokens

        current_position = 0
        while current_position < total_tokens:
            end_position = min(current_position + max_tokens, total_tokens)
            chunk_tokens = tokens[current_position:end_position]
            chunk_text = self.tokenizer.decode(chunk_tokens)

            # Find the actual character start/end positions in the original text
            # This is an approximation/fallback for tokenizers that don't map tokens perfectly
            # For most models, the tokenizer.decode and text.find should work well.
            start_char = len(self.tokenizer.decode(
                tokens[:current_position], skip_special_tokens=True))
            end_char = start_char + len(chunk_text)

            # Re-check actual character offsets in the original text to be safe
            # Use `text.find` for the first occurrence of the chunk text, starting search after the previous chunk's start.
            if chunks:
                search_start = chunks[-1]['end_char_offset'] - overlap
            else:
                search_start = 0

            # Use a slightly more robust way to find the actual start in the original string
            try:
                # Find where the chunk_text starts in the original text
                start_char_candidate = text.find(
                    chunk_text.strip(), search_start)
                if start_char_candidate != -1 and start_char_candidate >= search_start:
                    start_char = start_char_candidate
                    # Adjust end_char based on the found start
                    end_char = start_char + len(chunk_text.strip())
            except:
                # Fallback to token-based offset if string search fails
                pass

            chunks.append({
                "text": chunk_text,
                "start_char_offset": start_char,
                "end_char_offset": end_char
            })

            if end_position == total_tokens:
                break
            current_position = end_position - overlap

        return chunks

    def _aggregate_results(self, chunked_responses: List[EventLLMGenerateResponse], original_text: str) -> EventLLMGenerateResponse:
        """
        Aggregates results from multiple chunks, resolving overlaps and de-duplicating events.
        It uses a fuzzy matching approach for events and entities to handle minor discrepancies.
        """
        aggregated_events = []
        aggregated_entities = []
        aggregated_soa_triplets = []  # Keep all SOA triplets, they are low-level details

        seen_event_keys = set()
        seen_entity_texts = set()
        seen_soa_keys = set()

        for chunk_response in chunked_responses:
            # Events aggregation
            for event in chunk_response.events:
                # Create a canonical key for de-duplication
                event_key = (
                    event.event_type.lower(),
                    event.trigger.text.lower(),
                    # Entities are important for de-duping, use a tuple of sorted argument texts
                    # CRITICAL FIX: Handle both `entity` and `entities` fields when creating the key
                    tuple(sorted(arg.entity.text.lower() if arg.entity else ""
                                 for arg in event.arguments))
                )
                if event_key not in seen_event_keys:
                    # Check for approximate overlap in trigger span to be safer
                    is_duplicate = False
                    for existing_event in aggregated_events:
                        # Simple overlap check: if event types match and triggers overlap significantly
                        if existing_event.event_type.lower() == event.event_type.lower():
                            trigger_overlap = min(existing_event.trigger.end_char, event.trigger.end_char) - max(
                                existing_event.trigger.start_char, event.trigger.start_char)
                            if trigger_overlap > 0:  # Overlap exists
                                overlap_ratio = trigger_overlap / \
                                    min(len(existing_event.trigger.text),
                                        len(event.trigger.text))
                                if overlap_ratio > 0.5:  # More than 50% overlap in trigger span
                                    is_duplicate = True
                                    break

                    if not is_duplicate:
                        aggregated_events.append(event)
                        seen_event_keys.add(event_key)

            # Entities aggregation
            for entity in chunk_response.extracted_entities:
                # Use text and type for de-duplication
                if (entity.text.lower(), entity.type.lower()) not in seen_entity_texts:
                    aggregated_entities.append(entity)
                    seen_entity_texts.add(
                        (entity.text.lower(), entity.type.lower()))

            # SOA triplets aggregation (keep for potential use in synthesis, although prompt says to discard in final output)
            for soa in chunk_response.extracted_soa_triplets:
                soa_key = (
                    soa.subject.text.lower(),
                    soa.action.text.lower(),
                    soa.object.text.lower() if soa.object else ""
                )
                if soa_key not in seen_soa_keys:
                    aggregated_soa_triplets.append(soa)
                    seen_soa_keys.add(soa_key)

        # Get job_id from the first successful chunk, or generate a new one if the list is empty
        first_job_id = chunked_responses[0].job_id if chunked_responses and chunked_responses[0].job_id else str(
            uuid.uuid4())

        return EventLLMGenerateResponse(
            events=aggregated_events,
            extracted_entities=aggregated_entities,
            extracted_soa_triplets=aggregated_soa_triplets,
            # FIX: Ensure job_id is correctly set (either from a chunk or generated).
            job_id=first_job_id,
            # FIX: Ensure original_text is always provided here.
            original_text=original_text
        )

    @retry(
        stop=stop_after_attempt(ConfigManager.get_settings(
        ).event_llm_service.generation_max_retries),
        wait=wait_exponential(multiplier=1, min=ConfigManager.get_settings(
        ).event_llm_service.generation_retry_delay_seconds, max=10),
        # Retry on both ValidationError and our custom error
        retry=retry_if_exception_type((ValidationError, LLMGenerationError)),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    def _synthesize_main_events(self, original_text: str, aggregated_response: EventLLMGenerateResponse) -> EventLLMGenerateResponse:
        """
        Synthesizes main events from the aggregated low-level events.
        This is the second, high-level pass of the pipeline.
        """
        # EventLLMInput is used to pass the aggregated data to the prompt generator
        synthesis_input = EventLLMInput(
            text=original_text,
            ner_entities=[],  # Not strictly needed here, but kept for schema consistency
            soa_triplets=[],
            # Pass all aggregated data to the prompt generation
            events=aggregated_response.events,
            extracted_entities=aggregated_response.extracted_entities
        )

        # FIX: Significantly simplify the synthesis prompt by removing the original document text.
        prompt = self._get_prompt(
            synthesis_input, mode="synthesis", aggregated_response=aggregated_response)
        logger.info("Performing main event synthesis...")
        raw_response = self._call_llm(prompt)

        try:
            raw_response = raw_response.strip().lstrip("```json").rstrip("```").strip()

            # Use safe JSON loading
            try:
                json_data = json.loads(raw_response)
            except Exception as e:
                error_msg = f"LLM synthesis output parsing failed ({type(e).__name__}: {e})"
                logger.error(error_msg)

                # Raise LLMGenerationError to signal tenacity for a retry
                raise LLMGenerationError(error_msg) from e

            # --- CRITICAL FIX 3: Delegate post-processing, validation, and imputation ---
            # This is where we run the new LLMPostProcessor to map fields, validate, and fill nulls.
            # We explicitly remove SOA triplets here as the LLM is instructed not to generate them in synthesis.
            if "extracted_soa_triplets" in json_data:
                del json_data["extracted_soa_triplets"]

            response_model = LLMPostProcessor.post_process_response(json_data)

            # Ensure the synthesized response has the original job_id and text from aggregation pass
            response_model.job_id = aggregated_response.job_id
            response_model.original_text = original_text

            # Additional check: ensure max 3 events
            if len(response_model.events) > 3:
                logger.warning(
                    f"Synthesis returned {len(response_model.events)} events. Truncating to 3 main events.")
                response_model.events = response_model.events[:3]

            return response_model

        except ValidationError as e:
            logger.error(
                f"Validation error for synthesis pass: {e}")
            logger.error(f"Failing raw response: {raw_response}")
            # Re-raise existing ValidationError to ensure tenacity retry is triggered
            raise
        except Exception as e:
            error_msg = f"Unexpected runtime error in LLM synthesis processing: {type(e).__name__}: {e}"
            logger.error(error_msg, exc_info=True)
            # Raise LLMGenerationError to signal tenacity for a retry
            raise LLMGenerationError(error_msg) from e

    def process_article_in_chunks(self, article_text: str, ner_entities: Optional[List[Entity]] = None) -> EventLLMGenerateResponse:
        """
        Main method to process a long article using a two-pass pipeline.
        It orchestrates the chunking, individual chunk processing, and final synthesis.

        NOTE: The ner_entities input is now expected to be a list of Pydantic Entity objects, 
        as received from the EventLLMInput validation layer.
        """
        if not article_text:
            logger.warning("Attempted to process an empty article.")
            # Ensure we return a valid response even for empty text
            return EventLLMGenerateResponse(events=[], extracted_entities=[], extracted_soa_triplets=[], job_id=str(uuid.uuid4()), original_text=article_text)

        # CRITICAL CHECK: Ensure LLM is loaded before starting pipeline
        if self.model is None or self.tokenizer is None:
            raise RuntimeError(
                "LLM model is not initialized. Cannot process article.")

        # Pass 1: Chunk-level extraction and aggregation
        # Parameters retrieved from config/self

        #
        logger.info(
            f"Initiating two-pass pipeline for article of length {len(article_text)} characters.")

        # FIX: Remove hardcoded chunk parameters
        chunks = self._chunk_text(article_text)
        chunked_responses = []

        # Ensure ner_entities is treated as an empty list if None
        ner_entities = ner_entities if ner_entities is not None else []

        for i, chunk in enumerate(chunks):
            # Filter NER entities relevant to the current chunk's character span
            # CRITICAL FIX: Use attribute access (dot notation) instead of subscripting
            # Pydantic models are passed in, which are not subscriptable.
            chunk_ner_entities = [
                entity for entity in ner_entities if
                entity.start_char >= chunk["start_char_offset"] and
                entity.end_char <= chunk["end_char_offset"]
            ]

            # Adjust character offsets of the NER entities to be relative to the *start of the chunk*
            # CRITICAL FIX: Convert the Pydantic Entity object to a dictionary for the LLM prompt,
            # as the prompt examples and internal schema conversion expect Dict[str, Any].
            adjusted_ner_entities = []
            offset = chunk["start_char_offset"]
            for entity in chunk_ner_entities:
                # Use .model_dump() or .dict() to convert Pydantic object to dict
                adjusted_entity = entity.model_dump()
                # Offsets for LLM input must be relative to the chunk text
                adjusted_entity["start_char"] -= offset
                adjusted_entity["end_char"] -= offset
                adjusted_ner_entities.append(adjusted_entity)

            chunk_input = EventLLMInput(
                text=chunk["text"],
                ner_entities=adjusted_ner_entities
                # SOA triplets are omitted here, allowed by schema change
            )

            try:
                # _generate_response_for_chunk has the retry logic baked in
                chunk_response = self._generate_response_for_chunk(chunk_input)

                # Adjust all character offsets in the response back to the original text's frame
                # The response object contains Pydantic models, so use attribute access again
                for event in chunk_response.events:
                    event.trigger.start_char += offset
                    event.trigger.end_char += offset
                    for arg in event.arguments:
                        if arg.entity:
                            arg.entity.start_char += offset
                            arg.entity.end_char += offset
                        if arg.entities:
                            for entity in arg.entities:
                                entity.start_char += offset
                                entity.end_char += offset
                for entity in chunk_response.extracted_entities:
                    entity.start_char += offset
                    entity.end_char += offset
                for soa in chunk_response.extracted_soa_triplets:
                    soa.subject.start_char += offset
                    soa.subject.end_char += offset
                    soa.action.start_char += offset
                    soa.action.end_char += offset
                    if soa.object:
                        soa.object.start_char += offset
                        soa.object.end_char += offset

                chunked_responses.append(chunk_response)
                logger.info(
                    f"Successfully processed chunk {i+1}/{len(chunks)}.")
            except Exception as e:
                # FIX: More robust logging of error object to avoid "No constructor defined"
                logger.error(
                    f"Failed to process chunk {i+1}/{len(chunks)}. Skipping. Error: {type(e).__name__}: {e}", exc_info=True)
                # Re-raise on chunk failure if needed, but skipping allows aggregation of partial results
                continue

        # Aggregate the raw results from all chunks
        aggregated_response = self._aggregate_results(
            chunked_responses, article_text)
        logger.info(
            f"Aggregation complete. Found {len(aggregated_response.events)} low-level events and {len(aggregated_response.extracted_entities)} entities.")

        # Pass 2: Main event synthesis
        # FINAL FIX: If only one chunk was processed, we bypass the synthesis step
        # and rely on the Pass 1 aggregation, as the synthesis is redundant and error-prone.
        if len(chunks) == 1 and len(chunked_responses) == 1:
            logger.info(
                "Single chunk processed successfully. Bypassing redundant synthesis step.")
            # Run post-processing/imputation to clean up any null fields even in a single chunk output
            # CRITICAL FIX: Only run post_process_response which cleans up empty strings, preserving explicit nulls.
            return LLMPostProcessor.post_process_response(aggregated_response.dict())

        # If all chunks failed, the aggregated response will be empty, skip synthesis
        if not aggregated_response.events:
            logger.warning(
                "No events extracted from any chunk. Skipping synthesis.")
            # CRITICAL FIX: Run final cleanup even on empty response to ensure metadata is clean.
            return LLMPostProcessor.post_process_response(aggregated_response.dict())

        # _synthesize_main_events has the retry logic baked in
        final_response = self._synthesize_main_events(
            article_text, aggregated_response)
        logger.info(
            f"Synthesis complete. Found {len(final_response.events)} main events.")

        return final_response

# src/core/event_llm_logic.py
# File path: src/core/event_llm_logic.py
