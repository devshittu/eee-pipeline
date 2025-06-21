# src/core/event_llm_logic.py
# File path: src/core/event_llm_logic.py

import logging
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from typing import List, Dict, Any, Optional
from src.schemas.data_models import EventLLMInput, EventLLMGenerateResponse, Event, EventArgument, ArgumentEntity, EventMetadata, TextSpan, Entity, SOATriplet
from src.utils.config_manager import ConfigManager
import json  # Import json for parsing/dumping
import re    # Ensure re is imported for regex operations

logger = logging.getLogger("event_llm_service")


class EventLLMModel:
    """
    Singleton class to load and manage the Event LLM model.
    Ensures the model is loaded only once.
    """
    _instance = None

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
        Supports GPU loading with 8-bit quantization for memory efficiency.
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

            # Check for GPU and attempt 8-bit quantization
            if self.device == "cuda" and hasattr(torch.cuda, 'is_available') and torch.cuda.is_available():
                try:
                    # Configure 8-bit quantization for memory optimization
                    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        quantization_config=quantization_config,
                        cache_dir=cache_dir
                    )
                    logger.info("LLM model loaded with 8-bit quantization.")
                except Exception as e:
                    # Fallback to full precision if 8-bit loading fails
                    logger.warning(
                        f"Could not load LLM with 8-bit quantization ({e}). Falling back to full precision.")
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_name, cache_dir=cache_dir).to(self.device)
            else:
                # Load model without quantization for CPU or if GPU is disabled/unavailable
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name, cache_dir=cache_dir).to(self.device)

            self.model.eval()  # Set model to evaluation mode for inference
            logger.info("Event LLM model loaded successfully.")
        except Exception as e:
            logger.error(
                f"Failed to load Event LLM model {model_name}: {e}", exc_info=True)
            raise RuntimeError(
                f"Could not load Event LLM model {model_name}") from e

    def generate_events(self, data: EventLLMInput) -> EventLLMGenerateResponse:
        """
        Generates structured event and entity information using the LLM.
        The LLM is instruction-tuned to output JSON directly based on the provided prompt.
        It includes robust JSON extraction and parsing to handle potential LLM output quirks.
        """
        if not hasattr(self.model, 'generate'):
            raise RuntimeError(
                "LLM model not loaded or not a generative model.")

        settings = ConfigManager.get_settings().event_llm_service

        # Define an example of the expected JSON output structure with dummy data.
        # This serves as a strong instruction to the LLM for the desired output format.
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

        # Construct a detailed prompt for the LLM.
        # Emphasizes strict adherence to JSON format and schema.
        prompt = f"""
        Extract events and entities from the following text, using the provided Named Entities and Subject-Object-Action triplets as context.
        Synthesize additional event-related metadata (e.g., time, location, sentiment, causality).

        YOUR ENTIRE RESPONSE MUST BE A SINGLE, VALID JSON OBJECT.
        IT MUST STRICTLY ADHERE TO THE EventLLMGenerateResponse JSON SCHEMA.
        DO NOT INCLUDE ANY TEXT BEFORE OR AFTER THE JSON OBJECT.
        DO NOT INCLUDE ANY EXPLANATIONS OR CONVERSATIONAL PHRASES.
        ENSURE THERE ARE NO EXTRA CHARACTERS OUTSIDE THE JSON OR WITHIN THE JSON THAT VIOLATE THE SCHEMA.

        Input Text: {data.text}

        Named Entities:
        {chr(10).join([f"- {e.text} ({e.type})" for e in data.ner_entities])}

        Subject-Object-Action Triplets:
        {chr(10).join([f"- ({t.subject.text}, {t.action.text}, {t.object.text if t.object else 'N/A'})" for t in data.soa_triplets])}

        JSON Output Example (strictly follow this structure):
        {json.dumps(example_json_output, indent=2)}
        """

        # Log the full input prompt for debugging purposes
        logger.debug(f"LLM Input Prompt (full): \n{prompt}")

        # Tokenize the prompt and move to the appropriate device (CPU/GPU)
        input_ids = self.tokenizer(
            prompt, return_tensors="pt").input_ids.to(self.device)

        with torch.no_grad():
            # Generate tokens from the model
            output_tokens = self.model.generate(
                input_ids,
                max_new_tokens=settings.max_new_tokens,
                temperature=settings.temperature,
                top_p=settings.top_p,
                num_beams=1,  # Using greedy decoding or simple beam search
                # Sample only if temperature > 0
                do_sample=True if settings.temperature > 0.0 else False,
                # Important for some models to handle padding
                pad_token_id=self.tokenizer.eos_token_id
            )

        # Decode the generated tokens back to text
        generated_text = self.tokenizer.decode(
            output_tokens[0], skip_special_tokens=True)
        # Log the full raw output from the LLM for detailed inspection
        logger.debug(f"LLM Raw Output (full): \n{generated_text}")

        # --- START OF ROBUST JSON EXTRACTION AND PARSING LOGIC (Python `re` compatible) ---
        extracted_response = None

        # Locate the end of the JSON Output Example and start searching from there.
        # This helps in filtering out the prompt and example from the actual LLM response.
        example_json_str = json.dumps(example_json_output, indent=2)
        generated_text_after_example = generated_text

        # Find the specific example JSON string in the generated output.
        # Use rfind to find the last occurrence in case the prompt is repeated.
        example_start_marker = "JSON Output Example (strictly follow this structure):"

        # Find the start of the example block
        example_block_start_idx = generated_text.rfind(example_start_marker)

        if example_block_start_idx != -1:
            # Find the actual end of the example JSON string
            example_json_content_idx = generated_text.find(
                example_json_str, example_block_start_idx)
            if example_json_content_idx != -1:
                # Calculate the start index of the text *after* the example JSON.
                # Adding a small buffer (e.g., 2 for newline characters) to ensure we are past it.
                relevant_text_start_index = example_json_content_idx + \
                    len(example_json_str)
                generated_text_after_example = generated_text[relevant_text_start_index:].strip(
                )
                logger.debug(
                    f"Searching for JSON after example. Snippet: {generated_text_after_example[:500]}...")
            else:
                logger.warning(
                    "Could not find full example JSON content after marker. Searching from marker start.")
                generated_text_after_example = generated_text[example_block_start_idx + len(
                    example_start_marker):].strip()
        else:
            logger.warning(
                "Could not find example JSON start marker in LLM output. Searching entire output.")

        # Regex to find all occurrences of strings that start with '{' and end with '}'.
        # This pattern is non-greedy (`.*?`) to capture individual JSON objects
        # and works with `re.DOTALL` to match across newlines.
        json_pattern = re.compile(r'(\{.*?})', re.DOTALL)

        # Find all potential JSON matches in the relevant part of the generated text
        all_json_candidates_raw = json_pattern.findall(
            generated_text_after_example)

        # Iterate from the last match to the first, as the actual LLM output is most likely the last valid JSON.
        for json_str_candidate in reversed(all_json_candidates_raw):
            # The LLM's raw output doesn't seem to use ```json fences right now,
            # but if it does in the future, uncomment this to strip them.
            # if json_str_candidate.startswith('```json') and json_str_candidate.endswith('```'):
            #     json_str_candidate = json_str_candidate[len('```json'):-len('```')].strip()

            logger.debug(
                f"Attempting to parse candidate JSON string (first 500 chars): \n{json_str_candidate[:500]}...")
            try:
                parsed_dict = json.loads(json_str_candidate)
                # If JSON parsing is successful, attempt Pydantic validation
                response_data = EventLLMGenerateResponse.parse_obj(parsed_dict)
                extracted_response = response_data
                logger.info(
                    "Successfully extracted and validated a JSON response.")
                break  # Found a valid response, stop searching
            except json.JSONDecodeError as e:
                logger.debug(
                    f"Candidate JSON parsing failed: {e}. Trying next candidate.")
            except Exception as e:  # Catch Pydantic validation errors
                logger.debug(
                    f"Candidate JSON failed Pydantic validation: {e}. Trying next candidate.")

        # If no valid JSON was extracted after trying all candidates
        if extracted_response is None:
            logger.error(
                f"LLM output did not contain a parsable and valid JSON structure for text: '{data.text[:100]}...'. Raw output: {generated_text[:500]}")
            raise ValueError(
                "LLM did not generate a recognizable and valid JSON output.")

        return extracted_response
        # --- END OF ROBUST JSON EXTRACTION AND PARSING LOGIC ---

# src/core/event_llm_logic.py
# File path: src/core/event_llm_logic.py
