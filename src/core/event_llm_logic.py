

# src/core/event_llm_logic.py
# File path: src/core/event_llm_logic.py

import logging
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from typing import List, Dict, Any, Optional
from src.schemas.data_models import EventLLMInput, EventLLMGenerateResponse, Entity
from src.utils.config_manager import ConfigManager
import json
import re
from pydantic import ValidationError
import uuid

from src.core.llm_domains import determine_domain, get_domain_examples, get_domain_persona
from src.core.llm_postprocessor import LLMPostProcessor
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, before_sleep_log

logger = logging.getLogger("event_llm_service")


class LLMGenerationError(Exception):
    """Custom exception raised when LLM generation or output parsing fails."""
    pass


def _find_all_json_blocks(text: str) -> List[tuple]:
    """
    Finds ALL complete JSON objects in text and returns (start_pos, end_pos, json_str) tuples.
    Production-tested extraction that handles nested braces and strings correctly.
    """
    results = []
    i = 0
    text_len = len(text)

    while i < text_len:
        # Find next opening brace
        start = text.find('{', i)
        if start == -1:
            break

        # Track brace balance and string state
        balance = 0
        in_string = False
        escape = False
        end = -1

        for j in range(start, text_len):
            char = text[j]

            if escape:
                escape = False
                continue

            if char == '\\':
                escape = True
                continue

            if char == '"':
                in_string = not in_string
            elif not in_string:
                if char == '{':
                    balance += 1
                elif char == '}':
                    balance -= 1
                    if balance == 0:
                        end = j + 1
                        json_str = text[start:end]
                        # Validate it's actually parseable JSON
                        try:
                            json.loads(json_str)
                            results.append((start, end, json_str))
                        except json.JSONDecodeError:
                            pass  # Invalid JSON, skip it
                        break

        # Move past this block (or just past the '{' if no valid block found)
        i = end if end != -1 else start + 1

    return results


def _extract_response_json(text: str) -> Optional[str]:
    """
    Extracts the actual LLM response JSON from the full output.
    
    CRITICAL FIX: Returns the LARGEST JSON block after the last "Output:" marker,
    as the complete response object will be larger than individual nested objects.
    """
    # Find all occurrences of "Output:" (case-insensitive)
    output_markers = []
    for match in re.finditer(r'(?:^|\n)\s*Output:\s*', text, re.IGNORECASE):
        output_markers.append(match.end())

    if output_markers:
        # Search from the LAST "Output:" marker
        search_start = output_markers[-1]
        logger.debug(
            f"Found {len(output_markers)} 'Output:' marker(s), searching from position {search_start}")

        # Find all JSON blocks after this marker
        all_blocks = _find_all_json_blocks(text[search_start:])

        if all_blocks:
            # CRITICAL: Return the LARGEST block (complete response, not nested object)
            largest_block = max(all_blocks, key=lambda x: len(x[2]))
            actual_start, actual_end, json_str = largest_block

            # Validate it has the expected structure
            try:
                parsed = json.loads(json_str)
                if isinstance(parsed, dict) and ("events" in parsed or "extracted_entities" in parsed):
                    logger.info(
                        f"Extracted response JSON (length {len(json_str)}) from position {search_start + actual_start}")
                    return json_str
                else:
                    logger.warning(
                        f"Largest block doesn't have expected structure. Keys: {list(parsed.keys())}")
            except json.JSONDecodeError:
                pass

            # If largest doesn't work, try all blocks and find one with correct structure
            for start, end, json_str in sorted(all_blocks, key=lambda x: len(x[2]), reverse=True):
                try:
                    parsed = json.loads(json_str)
                    if isinstance(parsed, dict) and ("events" in parsed or "extracted_entities" in parsed):
                        logger.info(
                            f"Found valid response structure (length {len(json_str)})")
                        return json_str
                except json.JSONDecodeError:
                    continue

        logger.warning(
            f"No valid response JSON found after last 'Output:' marker")

    # Fallback: No "Output:" marker found, get the LARGEST JSON block in entire text
    logger.debug(
        "Fallback: searching for largest JSON block in entire response")
    all_blocks = _find_all_json_blocks(text)

    if all_blocks:
        # Try blocks from largest to smallest, looking for response structure
        for start, end, json_str in sorted(all_blocks, key=lambda x: len(x[2]), reverse=True):
            try:
                parsed = json.loads(json_str)
                if isinstance(parsed, dict) and ("events" in parsed or "extracted_entities" in parsed):
                    logger.info(
                        f"Fallback: found valid response structure (length {len(json_str)})")
                    return json_str
            except json.JSONDecodeError:
                continue

    logger.error(
        "No valid EventLLMGenerateResponse found anywhere in LLM output")
    return None


class EventLLMModel:
    """
    Singleton class to load and manage the Event LLM model.
    Ensures the model is loaded only once.
    """
    _instance = None
    model = None
    tokenizer = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(EventLLMModel, cls).__new__(cls)
            cls._instance._init_model()
        return cls._instance

    def _init_model(self):
        """Initializes the LLM model and tokenizer."""
        try:
            model_config = ConfigManager.get_settings().event_llm_service
            general_config = ConfigManager.get_settings().general

            logger.info("Loading model and tokenizer...")
            self.model_name = model_config.model_name
            self.model_path = model_config.model_path
            self.max_new_tokens = model_config.max_new_tokens
            self.temperature = model_config.temperature
            self.top_p = model_config.top_p
            self.generation_max_retries = model_config.generation_max_retries
            self.generation_retry_delay_seconds = model_config.generation_retry_delay_seconds
            self.chunk_size_tokens = model_config.chunk_size_tokens
            self.overlap_size_tokens = model_config.overlap_size_tokens

            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=model_config.model_cache_dir,
                use_fast=False
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

            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token

            logger.info(f"Model {self.model_name} loaded successfully.")

        except Exception as e:
            logger.error(f"Error initializing LLM model: {e}", exc_info=True)
            self.model = None
            self.tokenizer = None
            raise

    def _get_prompt(
        self,
        input_data: EventLLMInput,
        mode: str = "chunk",
        aggregated_response: Optional[EventLLMGenerateResponse] = None
    ) -> str:
        """
        Generates a structured few-shot prompt for the LLM.
        Context metadata is appended concisely after core instructions.
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("LLM model or tokenizer is not initialized.")

        domain = determine_domain(input_data.text)
        examples_to_use = get_domain_examples(domain)
        persona = get_domain_persona(domain)

        # Format context concisely
        context_section = ""
        if input_data.context_metadata:
            from src.utils.document_processor import DocumentProcessor
            processor = DocumentProcessor()
            context_line = processor.format_context_for_prompt(
                input_data.context_metadata)
            if context_line:
                context_section = f"\n{context_line}\n"

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
                "7.  **COMPREHENSIVE EXTRACTION:** Extract ALL events, entities, and relationships mentioned in the text. Do not limit yourself to only the most prominent ones.\n"
                "8.  **DO NOT include 'original_text' or 'job_id' fields in your output** - these are added by the system.\n"
            )

            examples_prompt = ""
            for example in examples_to_use:

                # Exclude original_text from example outputs to save tokens
                example_output = example['output'].copy() if isinstance(
                    example['output'], dict) else example['output']
                if isinstance(example_output, dict) and 'original_text' in example_output:
                    del example_output['original_text']
                examples_prompt += f"Input:\n{json.dumps(example['input'], indent=2)}\n\n"
                examples_prompt += f"Output:\n{json.dumps(example_output, indent=2)}\n\n"

            # Context placed with input text
            input_with_context = input_data.dict(
                exclude={'events', 'extracted_entities', 'extracted_soa_triplets'})
            input_text_section = f"Input:\n{context_section}{json.dumps(input_with_context, indent=2)}\n\nOutput:\n"

            return system_instruction + examples_prompt + input_text_section

        elif mode == "synthesis" and aggregated_response is not None:
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
                f"{metadata_instruction}\n"
            )

            synthesis_data_for_llm = {
                "Extracted_Low_Level_Events": [e.dict() for e in aggregated_response.events],
                "Extracted_Entities": [e.dict() for e in aggregated_response.extracted_entities],
            }

            context_line = ""
            if aggregated_response.context_metadata:
                from src.utils.document_processor import DocumentProcessor
                processor = DocumentProcessor()
                context_line = processor.format_context_for_prompt(
                    aggregated_response.context_metadata)

            user_input_prompt = f"{context_line}\nReview Data:\n{json.dumps(synthesis_data_for_llm, indent=2)}\n\nSynthesized Output (Events and Key Entities ONLY):\n"
            return synthesis_instruction + user_input_prompt

        else:
            raise ValueError(f"Invalid mode: {mode}")

    def _call_llm(self, prompt: str) -> str:
        """
        Calls the LLM with the generated prompt and returns clean JSON string.
        DIAGNOSTIC VERSION: Logs full output to understand generation issues.
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError(
                "LLM model or tokenizer is not initialized. Cannot call LLM.")

        inputs = self.tokenizer(
            prompt, return_tensors="pt", max_length=self.tokenizer.model_max_length, truncation=True).to(self.model.device)

        # Log prompt stats
        prompt_tokens = inputs['input_ids'].shape[1]
        logger.info(
            f"Prompt length: {len(prompt)} chars, {prompt_tokens} tokens")
        logger.debug(
            f"Prompt (last 500 chars before generation): ...{prompt[-500:]}")

        output = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            temperature=self.temperature,
            top_p=self.top_p,
            pad_token_id=self.tokenizer.eos_token_id
        )

        # Decode only the NEW tokens (not the prompt)
        generated_tokens = output[0][prompt_tokens:]
        generated_text = self.tokenizer.decode(
            generated_tokens, skip_special_tokens=True)

        logger.info(
            f"LLM generated {len(generated_tokens)} NEW tokens ({len(generated_text)} chars)")
        logger.info(
            f"=== FULL GENERATED OUTPUT (first 3000 chars) ===\n{generated_text[:3000]}\n=== END GENERATED OUTPUT ===")

        # Now work with the GENERATED text, not the full response
        response_text = generated_text

        # Extract the actual response JSON from generated output
        clean_json_str = _extract_response_json(response_text)

        if clean_json_str is None:
            logger.error(
                f"=== FULL GENERATED TEXT ===\n{generated_text}\n=== END ===")
            raise LLMGenerationError(
                "Failed to extract valid JSON from LLM generated output")

        return clean_json_str.strip()

    @retry(
        stop=stop_after_attempt(ConfigManager.get_settings(
        ).event_llm_service.generation_max_retries),
        wait=wait_exponential(multiplier=1, min=ConfigManager.get_settings(
        ).event_llm_service.generation_retry_delay_seconds, max=10),
        retry=retry_if_exception_type((ValidationError, LLMGenerationError)),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    def _generate_response_for_chunk(self, input_data: EventLLMInput) -> EventLLMGenerateResponse:
        """
        Generates events and entities for a SINGLE text chunk.
        Includes retry logic for robustness.
        """
        raw_response = ""
        try:
            prompt = self._get_prompt(input_data, mode="chunk")
            logger.info(
                f"Generating response for input chunk: {input_data.text[:100]}...")
            raw_response = self._call_llm(prompt)

            # Clean up any markdown artifacts
            raw_response = raw_response.strip().lstrip("```json").rstrip("```").strip()

            # Log sample for debugging
            logger.debug(
                f"Extracted JSON (first 500 chars): {raw_response[:500]}...")

            try:
                json_data = json.loads(raw_response)
            except json.JSONDecodeError as e:
                error_msg = f"JSON parsing failed ({type(e).__name__}: {e})"
                logger.error(error_msg)
                logger.error(
                    f"Problematic JSON (first 2000 chars): {raw_response[:2000]}")
                raise LLMGenerationError(error_msg) from e

            # Post-process and validate
            response_model = LLMPostProcessor.post_process_response(json_data)
            response_model.context_metadata = input_data.context_metadata

            # Log extraction stats
            logger.info(
                f"Extracted {len(response_model.events)} events, {len(response_model.extracted_entities)} entities, {len(response_model.extracted_soa_triplets)} SOA triplets")

            return response_model

        except ValidationError as e:
            logger.error(f"Validation error: {e}")
            logger.error(
                f"Response that failed validation (first 2000 chars): {raw_response[:2000]}")
            raise
        except Exception as e:
            error_msg = f"LLM chunk processing error: {type(e).__name__}: {e}"
            logger.error(error_msg, exc_info=True)
            raise LLMGenerationError(error_msg) from e

    def _chunk_text(self, text: str) -> List[Dict[str, Any]]:
        """
        Splits a long text into smaller chunks for sequential processing.
        Uses a sliding window approach with specified overlap to preserve context.
        """
        if self.tokenizer is None:
            raise RuntimeError(
                "Tokenizer is None. LLM model initialization failed.")

        tokens = self.tokenizer.encode(text)
        total_tokens = len(tokens)
        chunks = []

        max_tokens = self.chunk_size_tokens
        overlap = self.overlap_size_tokens

        current_position = 0
        while current_position < total_tokens:
            end_position = min(current_position + max_tokens, total_tokens)
            chunk_tokens = tokens[current_position:end_position]
            chunk_text = self.tokenizer.decode(chunk_tokens)

            start_char = len(self.tokenizer.decode(
                tokens[:current_position], skip_special_tokens=True))
            end_char = start_char + len(chunk_text)

            if chunks:
                search_start = chunks[-1]['end_char_offset'] - overlap
            else:
                search_start = 0

            try:
                start_char_candidate = text.find(
                    chunk_text.strip(), search_start)
                if start_char_candidate != -1 and start_char_candidate >= search_start:
                    start_char = start_char_candidate
                    end_char = start_char + len(chunk_text.strip())
            except:
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

    def _aggregate_results(
        self,
        chunked_responses: List[EventLLMGenerateResponse],
        original_text: str,
        document_id: Optional[str] = None
    ) -> EventLLMGenerateResponse:
        """
        Aggregates results from multiple chunks, resolving overlaps and de-duplicating events.
        """
        aggregated_events = []
        aggregated_entities = []
        aggregated_soa_triplets = []

        seen_event_keys = set()
        seen_entity_texts = set()
        seen_soa_keys = set()

        # Preserve context metadata from first chunk
        context_metadata = None
        if chunked_responses and chunked_responses[0].context_metadata:
            context_metadata = chunked_responses[0].context_metadata

        for chunk_response in chunked_responses:
            for event in chunk_response.events:
                event_key = (
                    event.event_type.lower(),
                    event.trigger.text.lower(),
                    tuple(sorted(arg.entity.text.lower()
                          if arg.entity else "" for arg in event.arguments))
                )
                if event_key not in seen_event_keys:
                    is_duplicate = False
                    for existing_event in aggregated_events:
                        if existing_event.event_type.lower() == event.event_type.lower():
                            trigger_overlap = min(existing_event.trigger.end_char, event.trigger.end_char) - max(
                                existing_event.trigger.start_char, event.trigger.start_char)
                            if trigger_overlap > 0:
                                overlap_ratio = trigger_overlap / \
                                    min(len(existing_event.trigger.text),
                                        len(event.trigger.text))
                                if overlap_ratio > 0.5:
                                    is_duplicate = True
                                    break

                    if not is_duplicate:
                        aggregated_events.append(event)
                        seen_event_keys.add(event_key)

            for entity in chunk_response.extracted_entities:
                if (entity.text.lower(), entity.type.lower()) not in seen_entity_texts:
                    aggregated_entities.append(entity)
                    seen_entity_texts.add(
                        (entity.text.lower(), entity.type.lower()))

            for soa in chunk_response.extracted_soa_triplets:
                soa_key = (
                    soa.subject.text.lower(),
                    soa.action.text.lower(),
                    soa.object.text.lower() if soa.object else ""
                )
                if soa_key not in seen_soa_keys:
                    aggregated_soa_triplets.append(soa)
                    seen_soa_keys.add(soa_key)

        first_job_id = chunked_responses[0].job_id if chunked_responses and chunked_responses[0].job_id else str(
            uuid.uuid4())

        # CRITICAL FIX: Pass through to post_process_response instead of direct instantiation
        aggregated_dict = {
            'events': aggregated_events,
            'extracted_entities': aggregated_entities,
            'extracted_soa_triplets': aggregated_soa_triplets,
            'job_id': first_job_id,
            'original_text': original_text,
            'context_metadata': context_metadata
        }

        return LLMPostProcessor.post_process_response(
            aggregated_dict,
            document_id=document_id
        )

    @retry(
        stop=stop_after_attempt(ConfigManager.get_settings(
        ).event_llm_service.generation_max_retries),
        wait=wait_exponential(multiplier=1, min=ConfigManager.get_settings(
        ).event_llm_service.generation_retry_delay_seconds, max=10),
        retry=retry_if_exception_type((ValidationError, LLMGenerationError)),
        before_sleep=before_sleep_log(logger, logging.WARNING),
    )
    def _synthesize_main_events(self, original_text: str, aggregated_response: EventLLMGenerateResponse) -> EventLLMGenerateResponse:
        """
        Synthesizes main events from the aggregated low-level events.
        This is the second, high-level pass of the pipeline.
        """
        synthesis_input = EventLLMInput(
            text=original_text,
            ner_entities=[],
            soa_triplets=[],
            events=aggregated_response.events,
            extracted_entities=aggregated_response.extracted_entities,
            context_metadata=aggregated_response.context_metadata
        )

        prompt = self._get_prompt(
            synthesis_input, mode="synthesis", aggregated_response=aggregated_response)
        logger.info("Performing main event synthesis...")
        raw_response = self._call_llm(prompt)

        try:
            raw_response = raw_response.strip().lstrip("```json").rstrip("```").strip()

            try:
                json_data = json.loads(raw_response)
            except json.JSONDecodeError as e:
                error_msg = f"Synthesis JSON parsing failed ({type(e).__name__}: {e})"
                logger.error(error_msg)
                raise LLMGenerationError(error_msg) from e

            if "extracted_soa_triplets" in json_data:
                del json_data["extracted_soa_triplets"]

            response_model = LLMPostProcessor.post_process_response(json_data)

            response_model.job_id = aggregated_response.job_id
            response_model.original_text = original_text
            response_model.context_metadata = aggregated_response.context_metadata

            if len(response_model.events) > 3:
                logger.warning(
                    f"Synthesis returned {len(response_model.events)} events. Truncating to 3 main events.")
                response_model.events = response_model.events[:3]

            return response_model

        except ValidationError as e:
            logger.error(f"Synthesis validation error: {e}")
            logger.error(f"Failing response: {raw_response[:1000]}")
            raise
        except Exception as e:
            error_msg = f"Synthesis processing error: {type(e).__name__}: {e}"
            logger.error(error_msg, exc_info=True)
            raise LLMGenerationError(error_msg) from e


    def process_article_in_chunks(
        self,
        article_text: str,
        ner_entities: Optional[List[Entity]] = None,
        context_metadata: Optional[Dict[str, Any]] = None,
        document_id: Optional[str] = None,
        normalized_date: Optional[str] = None
    ) -> EventLLMGenerateResponse:
        """
        Main method to process a long article using a two-pass pipeline.
        Orchestrates chunking, individual chunk processing, and final synthesis.
        """

        if not document_id:
            # Generate fallback ID
            import hashlib
            document_id = hashlib.sha256(
                article_text.encode()).hexdigest()[:16]

        if not article_text:
            logger.warning("Attempted to process an empty article.")
            # CRITICAL FIX: Use post_process_response for empty response
            return LLMPostProcessor.post_process_response(
                {
                    'events': [],
                    'extracted_entities': [],
                    'extracted_soa_triplets': [],
                    'job_id': str(uuid.uuid4()),
                    'original_text': article_text,
                    'context_metadata': context_metadata
                },
                document_id=document_id,
                normalized_date=normalized_date
            )

        if self.model is None or self.tokenizer is None:
            raise RuntimeError(
                "LLM model is not initialized. Cannot process article.")

        logger.info(
            f"Initiating two-pass pipeline for article of length {len(article_text)} characters.")

        chunks = self._chunk_text(article_text)
        chunked_responses = []

        ner_entities = ner_entities if ner_entities is not None else []

        for i, chunk in enumerate(chunks):
            chunk_ner_entities = [
                entity for entity in ner_entities if
                entity.start_char >= chunk["start_char_offset"] and
                entity.end_char <= chunk["end_char_offset"]
            ]

            adjusted_ner_entities = []
            offset = chunk["start_char_offset"]
            for entity in chunk_ner_entities:
                adjusted_entity = entity.model_dump()
                adjusted_entity["start_char"] -= offset
                adjusted_entity["end_char"] -= offset
                adjusted_ner_entities.append(adjusted_entity)

            chunk_input = EventLLMInput(
                text=chunk["text"],
                ner_entities=adjusted_ner_entities,
                context_metadata=context_metadata
            )

            try:
                chunk_response = self._generate_response_for_chunk(chunk_input)

                # Adjust offsets back to original text frame
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
                logger.error(
                    f"Failed to process chunk {i+1}/{len(chunks)}. Skipping. Error: {type(e).__name__}: {e}", exc_info=True)
                continue

        # CRITICAL FIX: Pass document_id to aggregation
        aggregated_response = self._aggregate_results(
            chunked_responses, article_text, document_id=document_id)
        logger.info(
            f"Aggregation complete. Found {len(aggregated_response.events)} low-level events and {len(aggregated_response.extracted_entities)} entities.")

        if len(chunks) == 1 and len(chunked_responses) == 1:
            logger.info(
                "Single chunk processed successfully. Bypassing redundant synthesis step.")
            # CRITICAL FIX: Use post_process_response with metadata
            return LLMPostProcessor.post_process_response(
                aggregated_response.dict(),
                document_id=document_id,
                normalized_date=normalized_date
            )

        if not aggregated_response.events:
            logger.warning(
                "No events extracted from any chunk. Skipping synthesis.")
            # CRITICAL FIX: Use post_process_response with metadata
            return LLMPostProcessor.post_process_response(
                aggregated_response.dict(),
                document_id=document_id,
                normalized_date=normalized_date
            )

        final_response = self._synthesize_main_events(
            article_text, aggregated_response)

        # Inject metadata fields
        final_response.document_id = document_id
        final_response.normalized_date = normalized_date
        logger.info(
            f"Synthesis complete. Found {len(final_response.events)} main events.")

        return final_response

# src/core/event_llm_logic.py
# File path: src/core/event_llm_logic.py
