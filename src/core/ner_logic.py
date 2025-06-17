# src/core/ner_logic.py
# File path: src/core/ner_logic.py

import logging
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from typing import List
from src.schemas.data_models import Entity
from src.utils.config_manager import ConfigManager

logger = logging.getLogger("ner_service")


class NERModel:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(NERModel, cls).__new__(cls)
            cls._instance._load_model()
        return cls._instance

    def _load_model(self):
        settings = ConfigManager.get_settings().ner_service
        model_name = settings.model_name
        cache_dir = settings.model_cache_dir
        logger.info(f"Loading NER model: {model_name} from cache: {cache_dir}")
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_name, cache_dir=cache_dir)
            model = AutoModelForTokenClassification.from_pretrained(
                model_name, cache_dir=cache_dir)
            self.nlp_pipeline = pipeline(
                "ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple")
            logger.info("NER model loaded successfully.")
        except Exception as e:
            logger.error(
                f"Failed to load NER model {model_name}: {e}", exc_info=True)
            raise RuntimeError(f"Could not load NER model {model_name}") from e

    def predict(self, text: str) -> List[Entity]:
        """
        Performs Named Entity Recognition on the input text.
        """
        if not self.nlp_pipeline:
            raise RuntimeError("NER model not loaded.")

        try:
            ner_results = self.nlp_pipeline(text)
            entities = []
            for res in ner_results:
                # Ensure the entity type (group_type) matches the expected 'type' field
                entities.append(
                    Entity(
                        text=res['word'],
                        type=res['entity_group'],
                        start_char=res['start'],
                        end_char=res['end']
                    )
                )
            logger.debug(
                f"Processed text for NER: '{text[:50]}...' found {len(entities)} entities.")
            return entities
        except Exception as e:
            logger.error(
                f"Error during NER prediction for text: '{text[:100]}...': {e}", exc_info=True)
            raise
