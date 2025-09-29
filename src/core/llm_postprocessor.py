# src/core/llm_postprocessor.py
# File path: src/core/llm_postprocessor.py

import logging
from typing import Dict, Any, Optional, List
from src.schemas.data_models import EventLLMGenerateResponse, Event, EventMetadata

logger = logging.getLogger("llm_postprocessor")


class LLMPostProcessor:
    """
    Handles post-generation cleanup, field mapping, and normalization of model output
    to ensure the final output strictly adheres to the schema.
    
    CRITICAL CHANGE: Removed aggressive imputation logic to avoid polluting downstream
    analytics with false sentiments/causality. Now, it only cleans empty strings,
    allowing truly missing data to remain 'null' as per the schema definition.
    """

    @staticmethod
    def map_and_normalize_data(data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recursively maps inconsistent field names and performs type normalization 
        before Pydantic validation.
        """
        if isinstance(data, dict):
            new_data = {}
            for k, v in data.items():
                new_k = k
                # 1. Map inconsistent names (LLM hallucination fix)
                if k == 'predicate':
                    new_k = 'action'

                # 2. Recursively process nested structures
                processed_v = LLMPostProcessor.map_and_normalize_data(v)

                # 3. Handle list vs. single entity for EventArgument flexibility (CRITICAL FIX)
                # If the value is a list of dicts/models, assume it's meant for the 'entities' field
                if k == 'entities' and isinstance(processed_v, list):
                    new_data['entities'] = processed_v
                elif k == 'entity' and processed_v is not None:
                    # If LLM returns a list under the 'entity' key, rename it to 'entities'
                    if isinstance(processed_v, list):
                        new_data['entities'] = processed_v
                    else:
                        new_data['entity'] = processed_v
                else:
                    new_data[new_k] = processed_v

            # Final check to ensure 'entities' is used if a list was improperly mapped to 'entity'
            if "entities" not in new_data and "entity" in new_data and isinstance(new_data["entity"], list):
                new_data["entities"] = new_data["entity"]
                del new_data["entity"]

            return new_data

        elif isinstance(data, list):
            return [LLMPostProcessor.map_and_normalize_data(item) for item in data]

        # 4. Handle nulls and empty strings: Prefer returning None for optional Pydantic fields
        # If a field is explicitly "" or None, return None. Pydantic handles null correctly.
        elif data == "" or data is None:
            return None

        return data

    @staticmethod
    def post_process_response(raw_json_data: Dict[str, Any]) -> EventLLMGenerateResponse:
        """
        Executes the full post-processing pipeline: mapping and validation.
        The imputation step is removed per user request.
        """
        # 1. Map and Normalize (Handles "predicate" -> "action" and ensures lists are nested correctly)
        cleaned_data = LLMPostProcessor.map_and_normalize_data(raw_json_data)

        # 2. Validate against Pydantic schema
        # This step automatically sets missing optional fields (like metadata.sentiment) to None (null in JSON).
        validated_response = EventLLMGenerateResponse(**cleaned_data)

        # NOTE: Imputation is intentionally removed here to honor the requirement
        # that fields should be truly null if the LLM cannot provide a value.

        return validated_response

# src/core/llm_postprocessor.py
# File path: src/core/llm_postprocessor.py
