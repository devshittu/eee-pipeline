# src/core/dp_logic.py
# File path: src/core/dp_logic.py

import logging
import spacy
from typing import List, Dict, Any
from src.schemas.data_models import SOATriplet, TextSpan
from src.utils.config_manager import ConfigManager

logger = logging.getLogger("dp_service")


class DPModel:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DPModel, cls).__new__(cls)
            cls._instance._load_model()
        return cls._instance

    def _load_model(self):
        settings = ConfigManager.get_settings().dp_service
        model_name = settings.model_name
        cache_dir = settings.model_cache_dir

        logger.info(f"Loading Dependency Parsing model: {model_name}")
        try:
            # Load the model by name, relying on spaCy's package resolution
            # SPACY_DATA environment variable ensures cache_dir is checked
            # Disable NER to save memory
            self.nlp = spacy.load(model_name, disable=["ner"])
            logger.info("Dependency Parsing model loaded successfully.")
        except Exception as e:
            logger.error(
                f"Failed to load Dependency Parsing model {model_name}: {e}", exc_info=True)
            raise RuntimeError(f"Could not load DP model {model_name}") from e

    def extract_soa(self, text: str) -> List[SOATriplet]:
        """
        Performs dependency parsing and extracts Subject-Action-Object triplets.
        A basic rule-based approach is used here. For more complex scenarios,
        advanced linguistic rule sets or ML models specifically for SOA extraction
        might be integrated.
        """
        if not hasattr(self, 'nlp') or self.nlp is None:
            raise RuntimeError("DP model not loaded.")

        doc = self.nlp(text)
        soa_triplets = []

        for sent in doc.sents:
            root = sent.root
            # A simple heuristic for S-A-O:
            # Subject: Noun chunk attached as nsubj or agent to the root verb
            # Action: The root verb itself (or its head if it's a participle)
            # Object: Noun chunk attached as dobj or pobj (prepositional object)

            subject = None
            action = None
            obj = None

            # Find action (root verb)
            if root.pos_ == "VERB":
                action = root
            elif root.head.pos_ == "VERB":  # e.g., for participles
                action = root.head

            if action:
                # Find subject
                subjects = [child for child in action.children if child.dep_ in (
                    "nsubj", "nsubjpass", "agent")]
                if subjects:
                    subject = subjects[0]  # Take the first subject found

                # Find object
                objects = [
                    child for child in action.children if child.dep_ in ("dobj", "pobj")]
                if objects:
                    obj = objects[0]  # Take the first object found
                elif action.dep_ == "acomp" and action.head.pos_ == "VERB":  # For predicate adjectives like "is good"
                    obj = action

                if subject and action:
                    soa_triplets.append(
                        SOATriplet(
                            subject=TextSpan(
                                text=subject.text, start_char=subject.idx, end_char=subject.idx + len(subject.text)),
                            action=TextSpan(
                                text=action.text, start_char=action.idx, end_char=action.idx + len(action.text)),
                            object=TextSpan(text=obj.text, start_char=obj.idx,
                                            end_char=obj.idx + len(obj.text)) if obj else None
                        )
                    )
        logger.debug(
            f"Processed text for DP: '{text[:50]}...' found {len(soa_triplets)} S-O-A triplets.")
        return soa_triplets

# # src/core/dp_logic.py
# # File path: src/core/dp_logic.py
