# src/core/event_llm_logic.py
# File path: src/core/event_llm_logic.py

import logging
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, BitsAndBytesConfig
from typing import List, Dict, Any, Optional
from src.schemas.data_models import EventLLMInput, EventLLMGenerateResponse, Event, EventArgument, ArgumentEntity, EventMetadata, TextSpan, Entity, SOATriplet
from src.utils.config_manager import ConfigManager
import json  # <--- Added: Import json for parsing/dumping

logger = logging.getLogger("event_llm_service")


class EventLLMModel:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(EventLLMModel, cls).__new__(cls)
            cls._instance._load_model()
        return cls._instance

    def _load_model(self):
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
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, cache_dir=cache_dir)

            if self.device == "cuda" and hasattr(torch.cuda, 'is_available') and torch.cuda.is_available():
                # Attempt to load in 8-bit for memory efficiency on GPU
                try:
                    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
                    self.model = AutoModelForSeq2SeqLM.from_pretrained(
                        model_name,
                        quantization_config=quantization_config,
                        cache_dir=cache_dir
                    )
                    logger.info("LLM model loaded with 8-bit quantization.")
                except Exception as e:
                    logger.warning(
                        f"Could not load LLM with 8-bit quantization ({e}). Falling back to full precision.")
                    self.model = AutoModelForSeq2SeqLM.from_pretrained(
                        model_name, cache_dir=cache_dir).to(self.device)
            else:
                self.model = AutoModelForSeq2SeqLM.from_pretrained(
                    model_name, cache_dir=cache_dir).to(self.device)

            self.model.eval()  # Set model to evaluation mode
            logger.info("Event LLM model loaded successfully.")
        except Exception as e:
            logger.error(
                f"Failed to load Event LLM model {model_name}: {e}", exc_info=True)
            raise RuntimeError(
                f"Could not load Event LLM model {model_name}") from e

    def generate_events(self, data: EventLLMInput) -> EventLLMGenerateResponse:
        """
        Generates structured event and entity information using the LLM.
        The LLM is instruction-tuned to output JSON directly.
        """
        if not hasattr(self.model, 'generate'):
            raise RuntimeError(
                "LLM model not loaded or not a generative model.")

        settings = ConfigManager.get_settings().event_llm_service

        # Define an example of the expected JSON output structure with dummy data.
        # This helps guide the LLM to generate actual JSON instances, not just schema definitions.
        example_json_output = {
            "events": [
                {
                    "event_type": "ExampleEvent",
                    "trigger": {"text": "triggered", "start_char": 0, "end_char": 9},
                    "arguments": [
                        {"argument_role": "Agent", "entity": {"text": "Person A",
                                                              "type": "PERSON", "start_char": 10, "end_char": 18}}
                    ],
                    "metadata": {"sentiment": "neutral", "causality": "direct"}
                }
            ],
            "extracted_entities": [
                {"text": "Person A", "type": "PERSON",
                    "start_char": 10, "end_char": 18}
            ],
            "extracted_soa_triplets": [
                {"subject": {"text": "Person A", "start_char": 10, "end_char": 18}, "action": {
                    "text": "triggered", "start_char": 0, "end_char": 9}, "object": None}
            ],
            "original_text": "Example text for event extraction."
        }

        # Construct a prompt for the LLM based on the objective and input data
        # This prompt is crucial for instruction-tuning the LLM to output specific JSON.
        prompt = f"""
        Extract events and entities from the following text, using the provided Named Entities and Subject-Object-Action triplets as context.
        Synthesize additional event-related metadata (e.g., time, location, sentiment, causality).
        Crucially, generate the final output strictly adhering to the StandardizedEventSchema JSON schema.
        Your output must be a valid JSON object.

        Input Text: {data.text}

        Named Entities:
        {chr(10).join([f"- {e.text} ({e.type})" for e in data.ner_entities])}

        Subject-Object-Action Triplets:
        {chr(10).join([f"- ({t.subject.text}, {t.action.text}, {t.object.text if t.object else 'N/A'})" for t in data.soa_triplets])}

        Output should be a JSON object, for example:
        {json.dumps(example_json_output, indent=2)}
        """  # <--- Key change: Including a concrete JSON example

        input_ids = self.tokenizer(
            prompt, return_tensors="pt").input_ids.to(self.device)

        with torch.no_grad():
            output_tokens = self.model.generate(
                input_ids,
                max_new_tokens=settings.max_new_tokens,
                temperature=settings.temperature,
                top_p=settings.top_p,
                num_beams=1,  # For greedy decoding or simple beam search
                do_sample=True if settings.temperature > 0.0 else False,  # Only sample if temp > 0
                pad_token_id=self.tokenizer.eos_token_id  # Important for some models
            )

        generated_text = self.tokenizer.decode(
            output_tokens[0], skip_special_tokens=True)
        logger.debug(f"LLM Raw Output: {generated_text[:500]}...")

        # Post-process the generated text to extract the JSON part
        # This is crucial because LLMs might sometimes generate extra text
        # before or after the JSON. A robust parser might be needed here.
        # Try to find the JSON object.
        json_start = generated_text.find('{')
        json_end = generated_text.rfind('}')

        if json_start != -1 and json_end != -1 and json_end > json_start:
            json_str = generated_text[json_start: json_end + 1]
            try:
                # Attempt to parse and validate with Pydantic
                response_data = EventLLMGenerateResponse.parse_raw(json_str)
                logger.info(
                    f"Successfully generated events for text (len: {len(data.text)}).")
                return response_data
            except Exception as e:
                logger.error(
                    f"Failed to parse LLM generated JSON for text: '{data.text[:100]}...'. Error: {e}. Raw JSON: {json_str[:500]}", exc_info=True)
                # Fallback: Create a partial response or raise error if JSON parsing fails
                raise ValueError(f"LLM generated malformed JSON: {e}")
        else:
            logger.error(
                f"LLM output did not contain a valid JSON structure (missing {{ or }} for main object) for text: '{data.text[:100]}...'. Raw output: {generated_text[:500]}")
            raise ValueError("LLM did not generate a valid JSON output.")

# src/core/event_llm_logic.py
# File path: src/core/event_llm_logic.py
