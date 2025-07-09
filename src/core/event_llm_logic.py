# src/core/event_llm_logic.py
# File path: src/core/event_llm_logic.py

import logging
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from typing import List, Dict, Any, Optional
from src.schemas.data_models import EventLLMInput, EventLLMGenerateResponse, Event, EventArgument, ArgumentEntity, EventMetadata, TextSpan, Entity, SOATriplet
from src.utils.config_manager import ConfigManager
import json
import re
from pydantic import ValidationError

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, before_sleep_log, RetryCallState


logger = logging.getLogger("event_llm_service")


class EventLLMModel:
    """
    Singleton class to load and manage the Event LLM model.
    Ensures the model is loaded only once.
    """
    _instance = None

    # Define a list of diverse examples for few-shot prompting.
    # Each example contains an 'input' (EventLLMInput structure) and an 'output' (EventLLMGenerateResponse structure).
    # These examples are designed to strictly adhere to the Pydantic schemas.
    _DIVERSE_EXAMPLES = [
        {
            "input": {
                "text": "Ukrainian forces launched a counteroffensive near Kharkiv, reclaiming several villages.",
                "ner_entities": [
                    {"text": "Ukrainian", "type": "NORP",
                        "start_char": 0, "end_char": 9},
                    {"text": "Kharkiv", "type": "LOC",
                        "start_char": 38, "end_char": 45},
                    {"text": "villages", "type": "LOC",
                        "start_char": 64, "end_char": 72}
                ],
                "soa_triplets": [
                    {"subject": {"text": "Ukrainian forces", "start_char": 0, "end_char": 16}, "action": {"text": "launched",
                                                                                                          "start_char": 17, "end_char": 25}, "object": {"text": "counteroffensive", "start_char": 27, "end_char": 43}},
                    {"subject": {"text": "Ukrainian forces", "start_char": 0, "end_char": 16}, "action": {
                        "text": "reclaiming", "start_char": 53, "end_char": 63}, "object": {"text": "villages", "start_char": 64, "end_char": 72}}
                ]
            },
            "output": {
                "events": [
                    {
                        "event_type": "military_action",
                        "trigger": {"text": "launched", "start_char": 17, "end_char": 25},
                        "arguments": [
                            {"argument_role": "agent", "entity": {
                                "text": "Ukrainian forces", "type": "NORP", "start_char": 0, "end_char": 16}},
                            {"argument_role": "action_target", "entity": {
                                "text": "counteroffensive", "type": "MISC", "start_char": 27, "end_char": 43}},
                            {"argument_role": "location", "entity": {
                                "text": "Kharkiv", "type": "LOC", "start_char": 38, "end_char": 45}}
                        ],
                        "metadata": {"sentiment": "neutral", "causality": "Forces initiated a counterattack."}
                    },
                    {
                        "event_type": "territory_change",
                        "trigger": {"text": "reclaiming", "start_char": 53, "end_char": 63},
                        "arguments": [
                            {"argument_role": "agent", "entity": {
                                "text": "Ukrainian forces", "type": "NORP", "start_char": 0, "end_char": 16}},
                            {"argument_role": "territory", "entity": {
                                "text": "villages", "type": "LOC", "start_char": 64, "end_char": 72}}
                        ],
                        "metadata": {"sentiment": "positive", "causality": "Counteroffensive led to territory recovery."}
                    }
                ],
                "extracted_entities": [
                    {"text": "Ukrainian", "type": "NORP",
                        "start_char": 0, "end_char": 9},
                    {"text": "Kharkiv", "type": "LOC",
                        "start_char": 38, "end_char": 45},
                    {"text": "villages", "type": "LOC",
                        "start_char": 64, "end_char": 72}
                ],
                "extracted_soa_triplets": [
                    {"subject": {"text": "Ukrainian forces", "start_char": 0, "end_char": 16}, "action": {"text": "launched",
                                                                                                          "start_char": 17, "end_char": 25}, "object": {"text": "counteroffensive", "start_char": 27, "end_char": 43}},
                    {"subject": {"text": "Ukrainian forces", "start_char": 0, "end_char": 16}, "action": {
                        "text": "reclaiming", "start_char": 53, "end_char": 63}, "object": {"text": "villages", "start_char": 64, "end_char": 72}}
                ],
                "original_text": "Ukrainian forces launched a counteroffensive near Kharkiv, reclaiming several villages."
            }
        },
        {
            "input": {
                "text": "The Parliament debated the new climate bill, facing strong opposition from conservative parties.",
                "ner_entities": [
                    {"text": "Parliament", "type": "ORG",
                        "start_char": 4, "end_char": 14},
                    {"text": "climate bill", "type": "MISC",
                        "start_char": 30, "end_char": 42},
                    {"text": "conservative parties", "type": "ORG",
                        "start_char": 71, "end_char": 91}
                ],
                "soa_triplets": [
                    {"subject": {"text": "The Parliament", "start_char": 0, "end_char": 14}, "action": {"text": "debated",
                                                                                                        "start_char": 15, "end_char": 22}, "object": {"text": "the new climate bill", "start_char": 23, "end_char": 42}},
                    {"subject": {"text": "conservative parties", "start_char": 71, "end_char": 91}, "action": {"text": "facing",
                                                                                                               "start_char": 44, "end_char": 50}, "object": {"text": "strong opposition", "start_char": 51, "end_char": 68}}
                ]
            },
            "output": {
                "events": [
                    {
                        "event_type": "legislative_process",
                        "trigger": {"text": "debated", "start_char": 15, "end_char": 22},
                        "arguments": [
                            {"argument_role": "actor", "entity": {
                                "text": "Parliament", "type": "ORG", "start_char": 4, "end_char": 14}},
                            {"argument_role": "subject", "entity": {
                                "text": "climate bill", "type": "MISC", "start_char": 30, "end_char": 42}}
                        ],
                        "metadata": {"sentiment": "neutral", "causality": "Parliament engaged in discussion about the bill."}
                    },
                    {
                        "event_type": "political_opposition",
                        "trigger": {"text": "facing", "start_char": 44, "end_char": 50},
                        "arguments": [
                            {"argument_role": "actor", "entity": {
                                "text": "conservative parties", "type": "ORG", "start_char": 71, "end_char": 91}},
                            {"argument_role": "opposition_target", "entity": {
                                "text": "climate bill", "type": "MISC", "start_char": 30, "end_char": 42}}
                        ],
                        "metadata": {"sentiment": "negative", "causality": "The bill's content led to strong disagreement."}
                    }
                ],
                "extracted_entities": [
                    {"text": "Parliament", "type": "ORG",
                        "start_char": 4, "end_char": 14},
                    {"text": "climate bill", "type": "MISC",
                        "start_char": 30, "end_char": 42},
                    {"text": "conservative parties", "type": "ORG",
                        "start_char": 71, "end_char": 91}
                ],
                "extracted_soa_triplets": [
                    {"subject": {"text": "The Parliament", "start_char": 0, "end_char": 14}, "action": {"text": "debated",
                                                                                                        "start_char": 15, "end_char": 22}, "object": {"text": "the new climate bill", "start_char": 23, "end_char": 42}},
                    {"subject": {"text": "conservative parties", "start_char": 71, "end_char": 91}, "action": {"text": "facing",
                                                                                                               "start_char": 44, "end_char": 50}, "object": {"text": "strong opposition", "start_char": 51, "end_char": 68}}
                ],
                "original_text": "The Parliament debated the new climate bill, facing strong opposition from conservative parties."
            }
        },
        {
            "input": {
                "text": "Global stock markets reacted negatively to the central bank's interest rate hike, causing a sharp decline.",
                "ner_entities": [
                    {"text": "Global stock markets", "type": "ORG",
                        "start_char": 0, "end_char": 20},
                    {"text": "central bank", "type": "ORG",
                        "start_char": 40, "end_char": 52},
                    {"text": "interest rate hike", "type": "MISC",
                        "start_char": 55, "end_char": 73}
                ],
                "soa_triplets": [
                    {"subject": {"text": "Global stock markets", "start_char": 0, "end_char": 20}, "action": {
                        "text": "reacted", "start_char": 21, "end_char": 28}, "object": {"text": "negatively", "start_char": 29, "end_char": 39}},
                    {"subject": {"text": "interest rate hike", "start_char": 55, "end_char": 73}, "action": {"text": "causing",
                                                                                                             "start_char": 75, "end_char": 82}, "object": {"text": "a sharp decline", "start_char": 83, "end_char": 98}}
                ]
            },
            "output": {
                "events": [
                    {
                        "event_type": "market_reaction",
                        "trigger": {"text": "reacted", "start_char": 21, "end_char": 28},
                        "arguments": [
                            {"argument_role": "actor", "entity": {
                                "text": "Global stock markets", "type": "ORG", "start_char": 0, "end_char": 20}},
                            {"argument_role": "cause", "entity": {
                                "text": "interest rate hike", "type": "MISC", "start_char": 55, "end_char": 73}}
                        ],
                        "metadata": {"sentiment": "negative", "causality": "Interest rate hike caused market reaction."}
                    },
                    {
                        "event_type": "economic_change",
                        "trigger": {"text": "decline", "start_char": 91, "end_char": 98},
                        "arguments": [
                            {"argument_role": "cause", "entity": {
                                "text": "interest rate hike", "type": "MISC", "start_char": 55, "end_char": 73}},
                            {"argument_role": "effect", "entity": {
                                "text": "sharp decline", "type": "MISC", "start_char": 83, "end_char": 98}}
                        ],
                        "metadata": {"sentiment": "negative", "causality": "The hike directly led to the decline."}
                    }
                ],
                "extracted_entities": [
                    {"text": "Global stock markets", "type": "ORG",
                        "start_char": 0, "end_char": 20},
                    {"text": "central bank", "type": "ORG",
                        "start_char": 40, "end_char": 52},
                    {"text": "interest rate hike", "type": "MISC",
                        "start_char": 55, "end_char": 73}
                ],
                "extracted_soa_triplets": [
                    {"subject": {"text": "Global stock markets", "start_char": 0, "end_char": 20}, "action": {
                        "text": "reacted", "start_char": 21, "end_char": 28}, "object": {"text": "negatively", "start_char": 29, "end_char": 39}},
                    {"subject": {"text": "interest rate hike", "start_char": 55, "end_char": 73}, "action": {"text": "causing",
                                                                                                             "start_char": 75, "end_char": 82}, "object": {"text": "a sharp decline", "start_char": 83, "end_char": 98}}
                ],
                "original_text": "Global stock markets reacted negatively to the central bank's interest rate hike, causing a sharp decline."
            }
        },
        {
            "input": {
                "text": "The new sci-fi movie starring Anya Taylor-Joy received rave reviews from critics and fans alike.",
                "ner_entities": [
                    {"text": "Anya Taylor-Joy", "type": "PERSON",
                        "start_char": 23, "end_char": 38},
                    {"text": "critics", "type": "PERSON",
                        "start_char": 60, "end_char": 67},
                    {"text": "fans", "type": "PERSON",
                        "start_char": 72, "end_char": 76}
                ],
                "soa_triplets": [
                    {"subject": {"text": "The new sci-fi movie", "start_char": 0, "end_char": 20}, "action": {"text": "received",
                                                                                                              "start_char": 39, "end_char": 47}, "object": {"text": "rave reviews", "start_char": 48, "end_char": 60}}
                ]
            },
            "output": {
                "events": [
                    {
                        "event_type": "media_release",
                        # Implied trigger for release
                        "trigger": {"text": "movie", "start_char": 12, "end_char": 17},
                        "arguments": [
                            {"argument_role": "product", "entity": {
                                "text": "sci-fi movie", "type": "WORK_OF_ART", "start_char": 8, "end_char": 20}},
                            {"argument_role": "actor", "entity": {
                                "text": "Anya Taylor-Joy", "type": "PERSON", "start_char": 23, "end_char": 38}}
                        ],
                        "metadata": {"sentiment": "neutral", "causality": "The movie was released to the public."}
                    },
                    {
                        "event_type": "reception",
                        "trigger": {"text": "received", "start_char": 39, "end_char": 47},
                        "arguments": [
                            {"argument_role": "subject", "entity": {
                                "text": "sci-fi movie", "type": "WORK_OF_ART", "start_char": 8, "end_char": 20}},
                            {"argument_role": "reviewer", "entity": {
                                "text": "critics", "type": "PERSON", "start_char": 60, "end_char": 67}},
                            {"argument_role": "reviewer", "entity": {
                                "text": "fans", "type": "PERSON", "start_char": 72, "end_char": 76}}
                        ],
                        "metadata": {"sentiment": "positive", "causality": "The movie's quality led to positive reviews."}
                    }
                ],
                "extracted_entities": [
                    {"text": "Anya Taylor-Joy", "type": "PERSON",
                        "start_char": 23, "end_char": 38},
                    {"text": "critics", "type": "PERSON",
                        "start_char": 60, "end_char": 67},
                    {"text": "fans", "type": "PERSON",
                        "start_char": 72, "end_char": 76}
                ],
                "extracted_soa_triplets": [
                    {"subject": {"text": "The new sci-fi movie", "start_char": 0, "end_char": 20}, "action": {"text": "received",
                                                                                                              "start_char": 39, "end_char": 47}, "object": {"text": "rave reviews", "start_char": 48, "end_char": 60}}
                ],
                "original_text": "The new sci-fi movie starring Anya Taylor-Joy received rave reviews from critics and fans alike."
            }
        }
    ]

    def __new__(cls):
        """
        Ensures only one instance of the model is loaded.
        """
        if cls._instance is None:
            cls._instance = super(EventLLMModel, cls).__new__(cls)
            cls._instance._load_model()
        return cls._instance

    def _load_model(self):
        """
        Loads the pre-trained language model and tokenizer based on configurations.
        Supports GPU loading with 4-bit/8-bit quantization for memory efficiency.
        """
        settings = ConfigManager.get_settings().event_llm_service
        general_settings = ConfigManager.get_settings().general

        model_name = settings.model_name
        model_path = settings.model_path
        cache_dir = settings.model_cache_dir
        gpu_enabled = general_settings.gpu_enabled

        self.device = "cuda" if torch.cuda.is_available() and gpu_enabled else "cpu"
        logger.info(
            f"Loading Event LLM model: {model_name} to device: {self.device} from path/cache: {model_path}")

        try:
            # Initialize tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, cache_dir=cache_dir)
            logger.info("Tokenizer loaded successfully.")

            # Check for GPU and attempt quantization
            if self.device == "cuda":
                logger.info(
                    "CUDA is available. Attempting GPU model loading with quantization.")
                try:
                    logger.info(
                        "Attempting LLM model load with 4-bit quantization.")
                    quantization_config_4bit = BitsAndBytesConfig(
                        load_in_4bit=True,
                        # Recommended for better performance with 4-bit
                        bnb_4bit_compute_dtype=torch.float16
                    )
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        quantization_config=quantization_config_4bit,
                        cache_dir=cache_dir,
                        torch_dtype=torch.float16  # Ensure FP16 for 4-bit
                    )
                    logger.info("LLM model loaded with 4-bit quantization.")
                except Exception as e_4bit:
                    logger.warning(
                        f"Could not load LLM with 4-bit quantization ({e_4bit}). Falling back to 8-bit.")
                    try:
                        logger.info(
                            "Attempting LLM model load with 8-bit quantization.")
                        quantization_config_8bit = BitsAndBytesConfig(
                            load_in_8bit=True)
                        self.model = AutoModelForCausalLM.from_pretrained(
                            model_name,
                            quantization_config=quantization_config_8bit,
                            cache_dir=cache_dir
                        )
                        logger.info(
                            "LLM model loaded with 8-bit quantization.")
                    except Exception as e_8bit:
                        logger.warning(
                            f"Could not load LLM with 8-bit quantization ({e_8bit}). Falling back to full precision (FP16 if possible).")
                        logger.info(
                            "Attempting LLM model load in full precision (FP16 if CUDA, else FP32).")
                        self.model = AutoModelForCausalLM.from_pretrained(
                            model_name,
                            cache_dir=cache_dir,
                            # Use FP16 for CUDA, default for CPU
                            torch_dtype=torch.float16 if self.device == 'cuda' else None
                        ).to(self.device)
                        logger.info(
                            f"LLM model loaded in full precision on {self.device}.")
            else:
                # Load model without quantization for CPU or if GPU is disabled/unavailable
                logger.info(
                    "CUDA not available or disabled. Loading LLM model on CPU in full precision.")
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name, cache_dir=cache_dir).to(self.device)

            self.model.eval()  # Set model to evaluation mode for inference
            logger.info("Event LLM model loaded successfully.")
        except Exception as e:
            logger.error(
                f"Failed to load Event LLM model {model_name}: {e}", exc_info=True)
            raise RuntimeError(
                f"Could not load Event LLM model {model_name}") from e

    def _create_prompt(self, data: EventLLMInput, attempt_num: int = 1) -> str:
        """Helper function to construct the LLM prompt, with optional retry message and diverse examples."""
        stricter_instructions = [
            "",  # Base prompt for the first attempt
            "\n\n--- PREVIOUS ATTEMPT FAILED ---\n"
            "Your last response was not valid JSON or was not correctly formatted. You MUST try again.\n"
            "Carefully follow the instructions: Your ENTIRE response MUST be a single, complete, and valid JSON object, *enclosed within ```json and ``` markdown fences*. "
            "DO NOT include any text before or after the JSON, and DO NOT wrap it in additional markdown code fences or conversational phrases.\n"
            "--- END OF WARNING ---\n\n",
            "\n\n--- SECOND FAILED ATTEMPT ---\n"
            "URGENT: Your last two attempts failed to produce valid JSON or failed to enclose it correctly. You MUST provide a JSON object that is parsable according to the schema, with no additional text, and *EXACTLY within ```json and ``` fences*. THIS IS YOUR FINAL ATTEMPT. ENSURE VALID JSON.\n"
            "--- END OF WARNING ---\n\n"
        ]
        retry_message = stricter_instructions[min(
            attempt_num - 1, len(stricter_instructions) - 1)]

        # Build the few-shot examples part of the prompt
        examples_prompt_section = ""
        for i, example in enumerate(self._DIVERSE_EXAMPLES):
            example_input = example["input"]
            example_output_json_str = json.dumps(example["output"], indent=2)

            examples_prompt_section += f"""
--- EXAMPLE {i+1} ---
Input Text: {example_input["text"].strip()}
Named Entities:
{chr(10).join([f"- {e['text'].strip()} ({e['type'].strip()})" for e in example_input["ner_entities"]]).strip()}
Subject-Object-Action Triplets:
{chr(10).join([f"- ({t['subject']['text'].strip()}, {t['action']['text'].strip()}, {t['object']['text'].strip() if t['object'] else 'N/A'})" for t in example_input["soa_triplets"]]).strip()}

JSON Output Example (strictly follow this structure and include the fences):
```json
{example_output_json_str}
```
"""
        # Strip to remove initial/final newlines/spaces from the entire examples section
        examples_prompt_section = examples_prompt_section.strip()

        # Construct the main prompt with instructions and examples
        base_prompt = f"""
Extract events and entities from the following text, using the provided Named Entities and Subject-Object-Action triplets as context.
Synthesize additional event-related metadata (e.g., time, location, sentiment, causality).

YOUR ENTIRE RESPONSE MUST BE A SINGLE, VALID JSON OBJECT.
IT MUST STRICTLY ADHERE TO THE EventLLMGenerateResponse JSON SCHEMA.
DO NOT INCLUDE ANY TEXT BEFORE OR AFTER THE JSON OBJECT.
DO NOT INCLUDE ANY EXPLANATIONS OR CONVERSATIONAL PHRASES.
YOUR JSON OUTPUT *MUST* BE ENCLOSED WITHIN ```json AND ``` MARKDOWN FENCES.

{examples_prompt_section}

--- YOUR TASK ---
Input Text: {data.text.strip()}
Named Entities:
{chr(10).join([f"- {e.text.strip()} ({e.type.strip()})" for e in data.ner_entities]).strip()}
Subject-Object-Action Triplets:
{chr(10).join([f"- ({t.subject.text.strip()}, {t.action.text.strip()}, {t.object.text.strip() if t.object else 'N/A'})" for t in data.soa_triplets]).strip()}

JSON Output (strictly follow the examples and schema, and include the fences):
```json
""".strip()  # Strip the entire base_prompt to remove initial/final newlines/spaces

        return retry_message.strip() + base_prompt

    def _parse_llm_output(self, generated_text: str) -> EventLLMGenerateResponse:
        """
        Extracts and validates the JSON object from the LLM's raw output.
        This function is made more robust to handle cases where the LLM might echo the prompt
        or include other conversational text. It will search the entire output for JSON
        and prioritize the last valid JSON object found.
        """
        logger.debug("Starting robust JSON extraction from LLM output.")
        logger.debug(
            f"Raw text received by _parse_llm_output (first 2000 chars): \n{generated_text[:2000]}...")

        extracted_response = None

        # Updated regex to explicitly look for JSON within ```json ... ``` fences
        # This makes the parsing more robust against prompt echoing.
        json_pattern = re.compile(r'```json(.*?)```', re.DOTALL)

        # Find all potential JSON matches in the *entire* generated text.
        all_json_candidates_raw = json_pattern.findall(generated_text)
        logger.debug(
            f"Found {len(all_json_candidates_raw)} potential JSON candidates (within ```json fences).")

        # Fallback if no fenced JSON is found, try to find any JSON object
        if not all_json_candidates_raw:
            logger.warning(
                "No JSON found within ```json fences. Attempting to find any top-level JSON object.")
            # Fallback regex to find any string that starts with '{' and ends with '}'
            json_pattern_fallback = re.compile(r'(\{.*?})', re.DOTALL)
            all_json_candidates_raw = json_pattern_fallback.findall(
                generated_text)
            logger.debug(
                f"Found {len(all_json_candidates_raw)} potential JSON candidates (fallback).")

        # Iterate from the last match to the first, as the actual LLM output is most likely the last valid JSON.
        for i, json_str_candidate in enumerate(reversed(all_json_candidates_raw)):
            candidate_index = len(all_json_candidates_raw) - 1 - i
            logger.debug(
                f"Attempting to parse candidate JSON string #{candidate_index} (first 500 chars): \n{json_str_candidate[:500]}..."
            )

            try:
                # .strip() to remove leading/trailing whitespace
                parsed_dict = json.loads(json_str_candidate.strip())
                logger.debug(
                    f"Candidate JSON string #{candidate_index} successfully parsed into dictionary.")

                # If JSON parsing is successful, attempt Pydantic validation
                response_data = EventLLMGenerateResponse.parse_obj(
                    parsed_dict)
                extracted_response = response_data
                logger.info(
                    f"Successfully extracted and validated a JSON response from candidate #{candidate_index}."
                )
                break  # Found a valid response, stop searching
            except json.JSONDecodeError as e:
                logger.debug(
                    f"Candidate JSON string #{candidate_index} parsing failed: {e}. Trying next candidate."
                )
            except ValidationError as e:
                logger.debug(
                    f"Candidate JSON string #{candidate_index} failed Pydantic validation: {e}. Trying next candidate."
                )
            except Exception as e:
                logger.debug(
                    f"Candidate JSON string #{candidate_index} failed with unexpected error: {e}. Trying next candidate."
                )

        # If no valid JSON was extracted after trying all candidates
        if extracted_response is None:
            logger.error(
                f"LLM output did not contain a parsable and valid JSON structure for text: '{data.text[:100]}...'. "
                f"No valid JSON found after checking {len(all_json_candidates_raw)} candidates. "
                f"Full raw output that caused failure (first 2000 chars): {generated_text[:2000]}..."
            )
            raise ValueError(
                "LLM did not generate a recognizable and valid JSON output."
            )

        return extracted_response

    @retry(
        stop=stop_after_attempt(ConfigManager.get_settings(
        ).event_llm_service.generation_max_retries + 1),
        wait=wait_exponential(
            multiplier=1,
            min=ConfigManager.get_settings().event_llm_service.generation_retry_delay_seconds,
            max=ConfigManager.get_settings().event_llm_service.generation_retry_delay_seconds * 5
        ),
        retry=(retry_if_exception_type(
            (json.JSONDecodeError, ValidationError, ValueError, torch.cuda.OutOfMemoryError, RuntimeError))),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True
    )
    def _attempt_generation_and_parsing(self, data: EventLLMInput, retry_state: Optional[RetryCallState] = None) -> EventLLMGenerateResponse:
        """
        A single attempt to generate and parse. Decorated by tenacity.
        'retry_state' is automatically provided by tenacity during retries, and is None for the first attempt.
        """
        attempt_num = retry_state.attempt_number if retry_state else 1
        logger.debug(f"Starting LLM generation attempt #{attempt_num}.")

        prompt = self._create_prompt(data, attempt_num=attempt_num)
        logger.debug(f"LLM Input Prompt (Attempt #{attempt_num}): \n{prompt}")

        settings = ConfigManager.get_settings().event_llm_service

        input_ids = self.tokenizer(
            prompt, return_tensors="pt").input_ids.to(self.device)
        logger.debug(
            f"Input IDs shape: {input_ids.shape}, device: {input_ids.device}")

        import time
        start_inference_time = time.time()

        if self.device == 'cuda':
            torch.cuda.empty_cache()
            logger.debug(
                f"CUDA Memory Summary BEFORE inference (Attempt #{attempt_num}):\n{torch.cuda.memory_summary()}")

        try:
            with torch.no_grad():
                output_tokens = self.model.generate(
                    input_ids,
                    max_new_tokens=settings.max_new_tokens,
                    temperature=settings.temperature,
                    top_p=settings.top_p,
                    num_beams=1,
                    do_sample=True if settings.temperature > 0.0 else False,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            end_inference_time = time.time()
            logger.info(
                f"LLM inference completed in {end_inference_time - start_inference_time:.2f} seconds.")

            if self.device == 'cuda':
                logger.debug(
                    f"CUDA Memory Summary AFTER inference (Attempt #{attempt_num}):\n{torch.cuda.memory_summary()}")

            logger.debug(f"Output Tokens shape: {output_tokens.shape}")
            logger.debug(
                f"Output Tokens snippet (first 50): {output_tokens[0][:50].tolist()}")

        except Exception as e:
            end_inference_time = time.time()
            logger.error(
                f"LLM inference failed during model.generate after {end_inference_time - start_inference_time:.2f} seconds: {e}",
                exc_info=True
            )
            if self.device == 'cuda':
                logger.error(
                    f"CUDA Memory Summary ON ERROR (Attempt #{attempt_num}):\n{torch.cuda.memory_summary()}")
            raise

        generated_text = self.tokenizer.decode(
            output_tokens[0], skip_special_tokens=True)
        logger.debug(
            f"LLM Raw Output (Attempt #{attempt_num}, first 2000 chars): \n{generated_text[:2000]}..."
        )
        logger.debug(
            f"Length of raw generated text: {len(generated_text)} characters.")

        return self._parse_llm_output(generated_text)

    def generate_events(self, data: EventLLMInput) -> EventLLMGenerateResponse:
        """
        Orchestrates the generation of structured event information using the LLM.
        This method invokes the tenacity-driven retry mechanism through the decorated internal method.
        """
        if not hasattr(self.model, 'generate'):
            raise RuntimeError(
                "LLM model not loaded or not a generative model.")

        try:
            return self._attempt_generation_and_parsing(data)
        except Exception as e:
            settings = ConfigManager.get_settings().event_llm_service
            logger.error(
                f"LLM output did not produce valid JSON after {settings.generation_max_retries + 1} attempts. Final error: {e}",
                exc_info=True
            )
            raise ValueError(
                "LLM did not generate a recognizable and valid JSON output after multiple attempts.")

# src/core/event_llm_logic.py
# File path: src/core/event_llm_logic.py
