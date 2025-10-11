# src/core/event_linker.py - NEW FILE

import logging
from typing import List, Dict, Set, Tuple
from datetime import datetime, timedelta
from src.schemas.data_models import Event, Entity

logger = logging.getLogger("event_linker")


class EventLinker:
    """
    Links events across documents that refer to the same real-world occurrence.
    Uses entity overlap, temporal proximity, and event type matching.
    """

    def __init__(
        self,
        entity_overlap_threshold: float = 0.5,
        temporal_window_days: int = 7,
        min_entities_for_match: int = 2
    ):
        """
        Args:
            entity_overlap_threshold: Min % of entities that must overlap (0.0-1.0)
            temporal_window_days: Max days between events to consider them related
            min_entities_for_match: Minimum shared entities required for match
        """
        self.entity_overlap_threshold = entity_overlap_threshold
        self.temporal_window_days = temporal_window_days
        self.min_entities_for_match = min_entities_for_match

    def extract_event_entities(self, event: Event) -> Set[Tuple[str, str]]:
        """
        Extract all entities from an event as (normalized_text, type) tuples.
        """
        entities = set()

        for arg in event.arguments:
            if arg.entity:
                entities.add(
                    (arg.entity.text.lower().strip(), arg.entity.type))
            if arg.entities:
                for entity in arg.entities:
                    entities.add((entity.text.lower().strip(), entity.type))

        return entities

    def calculate_entity_overlap(
        self,
        entities1: Set[Tuple[str, str]],
        entities2: Set[Tuple[str, str]]
    ) -> float:
        """
        Calculate Jaccard similarity between two entity sets.
        Returns: overlap score (0.0 to 1.0)
        """
        if not entities1 or not entities2:
            return 0.0

        intersection = entities1.intersection(entities2)
        union = entities1.union(entities2)

        return len(intersection) / len(union) if union else 0.0

    def calculate_temporal_proximity(
        self,
        date1: Optional[str],
        date2: Optional[str]
    ) -> bool:
        """
        Check if two dates are within temporal window.
        Returns: True if dates are close enough, False otherwise
        """
        if not date1 or not date2:
            return True  # Can't rule out based on missing dates

        try:
            dt1 = datetime.fromisoformat(date1.replace("Z", "+00:00"))
            dt2 = datetime.fromisoformat(date2.replace("Z", "+00:00"))

            delta = abs((dt1 - dt2).days)
            return delta <= self.temporal_window_days

        except (ValueError, AttributeError):
            logger.warning(
                f"Failed to parse dates for proximity: {date1}, {date2}")
            return True  # Can't rule out

    def are_events_linked(
        self,
        event1: Event,
        event1_date: Optional[str],
        event2: Event,
        event2_date: Optional[str]
    ) -> bool:
        """
        Determine if two events refer to the same real-world occurrence.
        
        Matching criteria:
        1. Same event type (exact match)
        2. High entity overlap (>= threshold)
        3. Temporal proximity (within window)
        """
        # Check event type match
        if event1.event_type != event2.event_type:
            return False

        # Check temporal proximity
        if not self.calculate_temporal_proximity(event1_date, event2_date):
            return False

        # Extract and compare entities
        entities1 = self.extract_event_entities(event1)
        entities2 = self.extract_event_entities(event2)

        # Require minimum entities for meaningful comparison
        if len(entities1) < self.min_entities_for_match or len(entities2) < self.min_entities_for_match:
            return False

        # Calculate overlap
        overlap_score = self.calculate_entity_overlap(entities1, entities2)

        is_match = overlap_score >= self.entity_overlap_threshold

        if is_match:
            logger.debug(
                f"Linked events: {event1.event_type} "
                f"(overlap: {overlap_score:.2f}, entities: {len(entities1.intersection(entities2))})"
            )

        return is_match

    def link_events_in_batch(
        self,
        events_with_metadata: List[Dict]
    ) -> Dict[str, List[str]]:
        """
        Link events across multiple documents in a batch.
        
        Args:
            events_with_metadata: List of dicts with keys:
                - event: Event object
                - document_id: str
                - normalized_date: Optional[str]
                - event_id: str (unique identifier for this event)
        
        Returns:
            Dict mapping canonical_event_id -> [related_event_ids]
        """
        linked_groups: Dict[str, List[str]] = {}
        processed: Set[str] = set()

        for i, item1 in enumerate(events_with_metadata):
            event_id1 = item1["event_id"]

            if event_id1 in processed:
                continue

            # Start new group with this event as canonical
            current_group = [event_id1]

            # Find all related events
            for j, item2 in enumerate(events_with_metadata):
                if i == j:
                    continue

                event_id2 = item2["event_id"]

                if event_id2 in processed:
                    continue

                # Check if events are linked
                if self.are_events_linked(
                    item1["event"],
                    item1.get("normalized_date"),
                    item2["event"],
                    item2.get("normalized_date")
                ):
                    current_group.append(event_id2)
                    processed.add(event_id2)

            if len(current_group) > 1:  # Only store if there are linked events
                linked_groups[event_id1] = current_group
                processed.add(event_id1)

        logger.info(
            f"Linked {len(linked_groups)} event groups from {len(events_with_metadata)} total events")
        return linked_groups
