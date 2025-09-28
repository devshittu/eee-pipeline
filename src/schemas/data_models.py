# src/schemas/data_models.py
# File path: src/schemas/data_models.py

from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field, HttpUrl
import uuid

# --- Common Models ---


class TextSpan(BaseModel):
    """Represents a span of text within a larger string."""
    text: str
    start_char: int
    end_char: int

# --- NER Service Models ---


class Entity(BaseModel):
    """Represents a named entity recognized in text."""
    text: str = Field(..., description="The detected entity text.")
    type: str = Field(...,
                      description="The type of the entity (e.g., PER, LOC, ORG).")
    start_char: int = Field(
        ..., description="Starting character index of the entity in the original text.")
    end_char: int = Field(
        ..., description="Ending character index of the entity in the original text.")


class NERPredictRequest(BaseModel):
    """Request model for the NER service's predict endpoint."""
    text: str = Field(...,
                      description="The input text for Named Entity Recognition.")


class NERPredictResponse(BaseModel):
    """Response model for the NER service's predict endpoint."""
    entities: List[Entity] = Field(
        default_factory=list, description="A list of recognized named entities.")
    text: str = Field(..., description="The original input text.")

# --- DP Service Models ---


class SOATriplet(BaseModel):
    """Represents a Subject-Object-Action triplet."""
    subject: TextSpan = Field(..., description="The subject of the action.")
    action: TextSpan = Field(..., description="The action/verb performed.")
    object: Optional[TextSpan] = Field(
        None, description="The object of the action, if any.")


class DPExtractSOARequest(BaseModel):
    """Request model for the DP service's extract-soa endpoint."""
    text: str = Field(...,
                      description="The input text for Dependency Parsing.")


class DPExtractSOAResponse(BaseModel):
    """Response model for the DP service's extract-soa endpoint."""
    soa_triplets: List[SOATriplet] = Field(
        default_factory=list, description="A list of extracted Subject-Object-Action triplets.")
    text: str = Field(..., description="The original input text.")

# --- Event LLM Service Models ---


class EventLLMInput(BaseModel):
    """
    Structured input for the Event LLM service, combining raw text, NER, and SOA results.
    """
    text: str = Field(..., description="The original raw text.")
    ner_entities: List[Entity] = Field(
        default_factory=list, description="List of named entities from NER service.")
    soa_triplets: List[SOATriplet] = Field(
        default_factory=list, description="List of S-O-A triplets from Dependency Parsing service.")


class ArgumentEntity(BaseModel):
    """Represents an entity linked to an event argument."""
    text: str = Field(..., description="The text of the argument entity.")
    type: str = Field(...,
                      description="The type of the argument entity (e.g., PER, LOC, ORG).")
    start_char: int = Field(..., description="Starting character index.")
    end_char: int = Field(..., description="Ending character index.")
    # Add other relevant entity fields as needed


class EventArgument(BaseModel):
    """
    Represents an argument of an extracted event.
    
    CRITICAL FIX: This model is re-engineered to handle cases where an argument role
    (e.g., 'recipients') is filled by a list of entities rather than a single one.
    This resolves the Pydantic ValidationError.
    """
    argument_role: str = Field(
        ..., description="The role of the argument (e.g., Agent, Patient, Time, Location, Recipients).")
    # Use Optional and Union to accept either a single entity or a list of entities.
    entity: Optional[ArgumentEntity] = Field(
        None, description="The single entity filling this argument role, if applicable.")
    entities: Optional[List[ArgumentEntity]] = Field(
        None, description="A list of entities filling this argument role, if applicable.")


class EventMetadata(BaseModel):
    """Additional metadata for an extracted event."""
    sentiment: Optional[str] = Field(
        None, description="Sentiment associated with the event (e.g., positive, negative, neutral).")
    causality: Optional[str] = Field(
        None, description="Causality information related to the event.")
    # Add other metadata fields as needed (e.g., time, location if not arguments)


class Event(BaseModel):
    """Represents a single extracted event."""
    event_type: str = Field(
        ..., description="The type of the event (e.g., Product_Launch, Acquisition).")
    trigger: TextSpan = Field(...,
                              description="The text span that triggers the event.")
    arguments: List[EventArgument] = Field(
        default_factory=list, description="A list of arguments associated with the event.")
    metadata: Optional[EventMetadata] = Field(
        None, description="Additional metadata for the event.")


class EventLLMGenerateResponse(BaseModel):
    """Response model for the Event LLM service's generate-events endpoint."""
    events: List[Event] = Field(
        default_factory=list, description="A list of extracted events.")
    extracted_entities: List[Entity] = Field(
        default_factory=list, description="All extracted entities (potentially enriched by LLM).")
    extracted_soa_triplets: List[SOATriplet] = Field(
        default_factory=list, description="All extracted S-O-A triplets.")
    original_text: str = Field("", description="The original input text.")
    # FIX: Add job_id field as it is used and logged by the API layer.
    job_id: str = Field(default_factory=lambda: str(
        uuid.uuid4()), description="Unique ID for the generation job/trace.")


# --- Orchestrator Service Models ---


class ProcessTextRequest(BaseModel):
    """Request model for processing a single text input."""
    text: str = Field(..., description="The single text string to process.")


class ProcessTextResponse(EventLLMGenerateResponse):
    """Response model for processing a single text, inheriting from LLM response."""
    # This directly uses the structure generated by the LLM service as the final output.
    pass


class ProcessBatchRequest(BaseModel):
    """Request model for processing a list of text inputs (batch)."""
    texts: List[str] = Field(...,
                             description="A list of text strings to process in batch.")
    job_id: str = Field(default_factory=lambda: str(
        uuid.uuid4()), description="Unique ID for the batch job.")


class BatchJobStatusResponse(BaseModel):
    """Response model for checking the status of a batch job."""
    job_id: str = Field(..., description="The unique ID of the batch job.")
    status: str = Field(
        ..., description="Current status of the job (e.g., PENDING, STARTED, SUCCESS, FAILURE).")
    results: Optional[List[EventLLMGenerateResponse]] = Field(
        None, description="List of results if job is SUCCESS.")
    error: Optional[str] = Field(
        None, description="Error message if job is FAILURE.")
    progress: Optional[float] = Field(
        None, description="Progress of the job as a percentage (0.0 to 1.0).")
    total_items: Optional[int] = Field(
        None, description="Total number of items in the batch.")
    processed_items: Optional[int] = Field(
        None, description="Number of items processed so far.")

# --- Celery Task Models (for internal Celery payloads) ---


class CeleryBatchProcessTaskPayload(BaseModel):
    """
    Payload for a Celery task that processes a chunk of a batch.
    Includes an ID to track the original position in the larger batch.
    """
    task_id: str = Field(default_factory=lambda: str(
        uuid.uuid4()), description="Unique ID for this specific Celery sub-task.")
    job_id: str = Field(..., description="The overarching batch job ID.")
    text_data: List[Dict[str, Any]] = Field(
        ..., description="List of dictionaries, each with 'id' and 'text'.")


class CelerySingleDocumentProcessPayload(BaseModel):
    """
    Payload for a Celery task that processes a single document.
    """
    document_id: str = Field(
        ..., description="Unique ID for the document (e.g., from original batch or file).")
    text: str = Field(..., description="The single text string to process.")
    job_id: Optional[str] = Field(
        None, description="The overarching batch job ID, if part of one.")


class CeleryTaskResult(BaseModel):
    """
    Structure for results returned by a single Celery task.
    """
    task_id: str
    job_id: str
    processed_data: List[EventLLMGenerateResponse]
    success: bool
    error: Optional[str] = None


# src/schemas/data_models.py
# File path: src/schemas/data_models.py
