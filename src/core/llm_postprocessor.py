# src/core/llm_postprocessor.py
# File path: src/core/llm_postprocessor.py

import logging
from typing import Dict, Any, Optional, List
from src.schemas.data_models import EventLLMGenerateResponse, Event, EventMetadata

logger = logging.getLogger("llm_postprocessor")

class LLMPostProcessor:
    """
    Handles post-generation cleanup, field mapping, and imputation of missing data
    to ensure the final output strictly adheres to the schema with no null or empty fields.
    This module separates concerns from the main LLM orchestration logic (DRY/SOLID).
    """

    @staticmethod
    def map_and_normalize_data(data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recursively maps inconsistent field names and performs type normalization 
        before Pydantic validation. This is decoupled from the main LLM logic.
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
        
        # 4. Handle nulls before Pydantic validation (Pydantic will treat explicit nulls as missing if optional)
        elif data is None:
            return None # Allow Pydantic to handle Optional fields
        
        return data

    @staticmethod
    def impute_missing_metadata(response: EventLLMGenerateResponse) -> EventLLMGenerateResponse:
        """
        Imputes standard, usable strings for missing or null metadata fields (sentiment, causality).
        This prevents empty or null fields in the final output, making it immediately usable.
        """
        for event in response.events:
            if not event.metadata:
                # Initialize metadata if null (impute basic structure)
                event.metadata = EventMetadata(
                    sentiment="neutral",
                    causality="The LLM did not provide specific causality data."
                )
            else:
                # Impute missing fields within existing metadata object
                if event.metadata.sentiment is None:
                    event.metadata.sentiment = "neutral"
                
                if event.metadata.causality is None or event.metadata.causality.strip() == "":
                    # Impute specific causality based on event type if possible, otherwise generic
                    default_causality = f"The LLM was unable to determine a specific causality for the '{event.event_type}' event."
                    event.metadata.causality = default_causality
        
        return response

    @staticmethod
    def post_process_response(raw_json_data: Dict[str, Any]) -> EventLLMGenerateResponse:
        """
        Executes the full post-processing pipeline: mapping, validation, and imputation.
        """
        # 1. Map and Normalize
        cleaned_data = LLMPostProcessor.map_and_normalize_data(raw_json_data)
        
        # 2. Validate against Pydantic schema
        validated_response = EventLLMGenerateResponse(**cleaned_data)
        
        # 3. Impute missing metadata (to fix nulls in final JSON)
        final_response = LLMPostProcessor.impute_missing_metadata(validated_response)
        
        return final_response
# src/core/llm_postprocessor.py
# File path: src/core/llm_postprocessor.py
