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
        
        CRITICAL: Maps LLM output keys to schema-expected keys:
        - "entities" -> "extracted_entities"
        - "soa_triplets" -> "extracted_soa_triplets"
        - "predicate" -> "action"
        """
        if isinstance(data, dict):
            new_data = {}
            for k, v in data.items():
                new_k = k

                # Map top-level response keys to schema-expected names
                if k == 'entities':
                    new_k = 'extracted_entities'
                elif k == 'soa_triplets':
                    new_k = 'extracted_soa_triplets'
                # Map nested predicate to action
                elif k == 'predicate':
                    new_k = 'action'

                # Recursively process nested structures
                processed_v = LLMPostProcessor.map_and_normalize_data(v)

                # Handle list vs. single entity for EventArgument flexibility
                if k == 'entities' and isinstance(processed_v, list) and new_k == 'entities':
                    # This is within an EventArgument, not the top level
                    new_data['entities'] = processed_v
                elif k == 'entity' and processed_v is not None:
                    if isinstance(processed_v, list):
                        new_data['entities'] = processed_v
                    else:
                        new_data['entity'] = processed_v
                else:
                    new_data[new_k] = processed_v

            # Final check for entity/entities at argument level
            if "entities" not in new_data and "entity" in new_data and isinstance(new_data["entity"], list):
                new_data["entities"] = new_data["entity"]
                del new_data["entity"]

            return new_data

        elif isinstance(data, list):
            return [LLMPostProcessor.map_and_normalize_data(item) for item in data]

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
