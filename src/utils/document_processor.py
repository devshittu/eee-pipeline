# src/utils/document_processor.py
# File path: src/utils/document_processor.py

"""
Utility module for extracting and mapping fields from diverse upstream document schemas.
Handles text extraction, context metadata aggregation, and field validation.
"""

import logging
from datetime import datetime
from dateutil import parser
from typing import Dict, Any, Optional, List
from src.utils.config_manager import ConfigManager, DocumentFieldMappingSettings
from src.schemas.data_models import EventLLMGenerateResponse
logger = logging.getLogger("document_processor")


class DocumentProcessor:
    """
    Processes documents with flexible schemas based on configured field mappings.
    Extracts main text, context metadata, and preserves pass-through fields.
    """

    def __init__(self, config: Optional[DocumentFieldMappingSettings] = None):
        """
        Initialize processor with field mapping configuration.

        Args:
            config: Field mapping settings. If None, loads from ConfigManager.
        """
        if config is None:
            settings = ConfigManager.get_settings()
            self.config = settings.document_field_mapping
        else:
            self.config = config

        logger.info(
            f"DocumentProcessor initialized with text_field='{self.config.text_field}'")

    def extract_text(self, document: Dict[str, Any]) -> str:
        """
        Extract main text from document using configured field mapping with fallback chain.

        Args:
            document: Source document dictionary

        Returns:
            Extracted text string

        Raises:
            ValueError: If no text field found or all fields are empty
        """
        # Try primary text field
        primary_field = self.config.text_field
        if primary_field in document and document[primary_field]:
            text = document[primary_field]
            if isinstance(text, str) and text.strip():
                logger.debug(
                    f"Extracted text from primary field '{primary_field}' (length: {len(text)})")
                return text.strip()

        # Try fallback fields
        for fallback_field in self.config.text_field_fallbacks:
            if fallback_field in document and document[fallback_field]:
                text = document[fallback_field]
                if isinstance(text, str) and text.strip():
                    logger.info(
                        f"Extracted text from fallback field '{fallback_field}' (length: {len(text)})")
                    return text.strip()

        # No valid text found
        available_fields = list(document.keys())
        raise ValueError(
            f"No valid text found in document. "
            f"Checked fields: [{primary_field}] + fallbacks {self.config.text_field_fallbacks}. "
            f"Available fields: {available_fields}"
        )

    def extract_context_metadata(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract context fields specified in configuration.

        Args:
            document: Source document dictionary

        Returns:
            Dictionary of context metadata (only non-null values)
        """
        context = {}

        for field_name in self.config.context_fields:
            if field_name in document:
                value = document[field_name]
                # Only include non-null, non-empty values
                if value is not None:
                    if isinstance(value, str) and not value.strip():
                        continue  # Skip empty strings
                    context[field_name] = value

        logger.debug(
            f"Extracted {len(context)} context fields: {list(context.keys())}")
        return context

    def _normalize_entity_text(self, text: str) -> str:
        """
        Normalize entity text for better deduplication.
        Handles partial matches like "Robinson" vs "Tyler Robinson".

        Args:
            text: Entity text to normalize

        Returns:
            Normalized text (lowercase, stripped)
        """
        return text.lower().strip()

    def _is_entity_subset(self, short_text: str, long_text: str) -> bool:
        """
        Check if short_text is a substring/subset of long_text (e.g., "Robinson" in "Tyler Robinson").

        Args:
            short_text: Shorter entity text (already normalized)
            long_text: Longer entity text (already normalized)

        Returns:
            True if short_text appears as a token in long_text
        """
        # Split into tokens and check if short is a complete token in long
        long_tokens = long_text.split()
        short_tokens = short_text.split()

        # Check if all short tokens appear consecutively in long tokens
        if len(short_tokens) == 1:
            # Single token - check if it's a complete word in long_text
            return short_text in long_tokens
        else:
            # Multi-token - check for consecutive sequence
            for i in range(len(long_tokens) - len(short_tokens) + 1):
                if long_tokens[i:i+len(short_tokens)] == short_tokens:
                    return True
        return False

    def merge_entities(
        self,
        upstream_entities: Optional[List[Dict[str, Any]]],
        ner_entities: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Merge pre-extracted entities from upstream with NER-detected entities.

        CRITICAL FIX: Improved deduplication that handles partial name matches
        (e.g., "Robinson" and "Tyler Robinson" are treated as the same entity,
        preferring the longer/more complete form).

        Args:
            upstream_entities: Entities from upstream source (if available)
            ner_entities: Entities detected by NER service

        Returns:
            Merged and deduplicated entity list
        """
        if not upstream_entities:
            return ner_entities

        # Normalize and deduplicate
        all_entities = []

        # Add upstream entities first (they have priority for offsets)
        for entity in upstream_entities:
            if isinstance(entity, dict) and "text" in entity and "type" in entity:
                all_entities.append(entity)

        # Add NER entities
        for entity in ner_entities:
            all_entities.append(entity)

        # Advanced deduplication: group by type first, then merge within type
        entities_by_type = {}
        for entity in all_entities:
            entity_type = entity.get("type", "").upper()
            if entity_type not in entities_by_type:
                entities_by_type[entity_type] = []
            entities_by_type[entity_type].append(entity)

        merged = []
        for entity_type, entities in entities_by_type.items():
            # Sort by text length (descending) - prefer longer forms
            entities_sorted = sorted(
                entities, key=lambda e: len(e["text"]), reverse=True)

            kept = []
            for entity in entities_sorted:
                norm_text = self._normalize_entity_text(entity["text"])

                # Check if this entity is a subset of any already-kept entity
                is_duplicate = False
                for kept_entity in kept:
                    kept_norm = self._normalize_entity_text(
                        kept_entity["text"])

                    # If current entity is shorter and is a subset of kept entity, skip it
                    if len(norm_text) < len(kept_norm) and self._is_entity_subset(norm_text, kept_norm):
                        is_duplicate = True
                        logger.debug(
                            f"Skipping duplicate entity '{entity['text']}' (subset of '{kept_entity['text']}')")
                        break

                    # If current entity is longer and kept entity is a subset, replace kept
                    elif len(norm_text) > len(kept_norm) and self._is_entity_subset(kept_norm, norm_text):
                        logger.debug(
                            f"Replacing '{kept_entity['text']}' with longer form '{entity['text']}'")
                        kept.remove(kept_entity)
                        break

                    # Exact match
                    elif norm_text == kept_norm:
                        is_duplicate = True
                        break

                if not is_duplicate:
                    kept.append(entity)

            merged.extend(kept)

        logger.debug(
            f"Merged entities: {len(upstream_entities or [])} upstream + {len(ner_entities)} NER = {len(merged)} total (after deduplication)")
        return merged

    def prepare_enriched_input(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare complete enriched input for orchestrator processing.

        Args:
            document: Source document with flexible schema

        Returns:
            Dictionary with extracted text, context metadata, and source document
        """
        text = self.extract_text(document)
        context_metadata = self.extract_context_metadata(document)
        document_id = self.extract_document_id(document)

        # NEW: Normalize publication date
        normalized_date = None
        for date_field in ["cleaned_publication_date", "publication_date", "date", "published_at"]:
            if date_field in document:
                normalized_date = self.normalize_date(document[date_field])
                if normalized_date:
                    break

        # Extract upstream entities if available
        upstream_entities = None
        if "entities" in document and isinstance(document["entities"], list):
            upstream_entities = document["entities"]

        return {
            "text": text,
            "context_metadata": context_metadata,
            "upstream_entities": upstream_entities,
            "source_document": document,
            "document_id": document_id,
            "normalized_date": normalized_date
        }

    def format_context_for_prompt(self, context_metadata: Dict[str, Any]) -> str:
        """
        Format context metadata into a concise string for LLM prompt injection.

        CRITICAL FIX: Reduced verbosity to preserve token budget for few-shot examples.
        Only include most critical context fields.

        Args:
            context_metadata: Dictionary of context fields

        Returns:
            Formatted string for prompt (concise format)
        """
        if not context_metadata:
            return ""

        # CRITICAL: Limit to top 3 most important fields to preserve token budget
        priority_fields = [
            ("cleaned_title", "Title"),
            ("cleaned_publication_date", "Date"),
            ("cleaned_author", "Author"),
        ]

        lines = []
        added_count = 0

        for field_key, field_label in priority_fields:
            if field_key in context_metadata and added_count < 3:
                value = context_metadata[field_key]
                if isinstance(value, str) and value.strip():
                    lines.append(f"{field_label}: {value}")
                    added_count += 1

        if lines:
            # Concise format - single line
            return "[Context: " + " | ".join(lines) + "]"
        return ""

    def extract_document_id(self, document: Dict[str, Any]) -> str:
        """
        Extract or generate a persistent document identifier.
        Tries multiple common ID fields before generating a UUID.

        Priority order:
        1. document_id (your standard)
        2. id, _id (common in databases)
        3. url, source_url (unique identifiers)
        4. Generate UUID based on text hash (deterministic fallback)
        """
        import hashlib

        # Try explicit ID fields
        for id_field in ["document_id", "id", "_id", "doc_id"]:
            if id_field in document and document[id_field]:
                return str(document[id_field])

        # Try URL as unique identifier
        for url_field in ["cleaned_source_url", "url", "source_url", "link"]:
            if url_field in document and document[url_field]:
                return hashlib.sha256(str(document[url_field]).encode()).hexdigest()[:16]

        # Fallback: Generate deterministic ID from text hash
        text = self.extract_text(document)
        return hashlib.sha256(text.encode()).hexdigest()[:16]

    def normalize_date(self, date_value: Any) -> Optional[str]:
        """
        Normalize any date format to ISO-8601 (YYYY-MM-DDTHH:MM:SSZ).

        Handles:
        - ISO strings: "2024-10-15T10:30:00Z"
        - Human dates: "October 15, 2024", "15 Oct 2024"
        - Timestamps: 1697368200
        - Relative: "yesterday", "last Friday" (requires reference date)

        Returns:
            ISO-8601 string or None if unparseable
        """
        if not date_value:
            return None

        try:
            # Handle timestamp (int or float)
            if isinstance(date_value, (int, float)):
                dt = datetime.fromtimestamp(date_value)
                return dt.isoformat() + "Z"

            # Handle string dates
            if isinstance(date_value, str):
                # Try parsing with dateutil (handles most formats)
                dt = parser.parse(date_value, fuzzy=True)
                return dt.isoformat() + "Z"

            # Handle datetime objects
            if isinstance(date_value, datetime):
                return date_value.isoformat() + "Z"

            return None

        except (ValueError, TypeError, parser.ParserError) as e:
            logger.warning(f"Failed to parse date '{date_value}': {e}")
            return None

    def extract_causality_edges(self, response: EventLLMGenerateResponse) -> List[Dict[str, Any]]:
        """
        Extract causality relationships as graph edges from event metadata.
        
        Returns:
            List of edge dictionaries: {"source": event_id, "target": event_id, "causality": str}
        """
        edges = []

        for i, event in enumerate(response.events):
            if not event.metadata or not event.metadata.causality:
                continue

            source_event_id = f"{response.document_id}:event_{i}"

            # Parse causality text to find mentions of other events
            # Simple heuristic: look for event triggers mentioned in causality
            causality_text = event.metadata.causality.lower()

            for j, other_event in enumerate(response.events):
                if i == j:
                    continue

                # Check if other event's trigger appears in this event's causality
                if other_event.trigger.text.lower() in causality_text:
                    target_event_id = f"{response.document_id}:event_{j}"
                    edges.append({
                        "source": source_event_id,
                        "target": target_event_id,
                        "causality": event.metadata.causality,
                        "edge_type": "causes"
                    })

        return edges
# src/utils/document_processor.py
# File path: src/utils/document_processor.py
