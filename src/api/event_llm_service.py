# src/api/event_llm_service.py
# File path: src/api/event_llm_service.py

import logging
from fastapi import FastAPI, HTTPException, status
from pydantic import ValidationError
from src.schemas.data_models import EventLLMInput, EventLLMGenerateResponse
from src.core.event_llm_logic import EventLLMModel
from src.utils.config_manager import ConfigManager
from src.utils.logger import setup_logging
import uuid

# Load settings and configure logging
settings = ConfigManager.get_settings()
setup_logging()
logger = logging.getLogger("event_llm_service")

# Initialize FastAPI app
app = FastAPI(
    title="Event & Schema Generation Service",
    description="Microservice for event and schema generation using a fine-tuned LLM.",
    version="1.0.0"
)

# Initialize the Event LLM model (singleton pattern)
llm_model = None
try:
    logger.info("Initializing Event LLM model...")
    llm_model = EventLLMModel()
    logger.info("Event LLM model initialized successfully.")
except Exception as e:
    logger.critical(
        f"Failed to initialize Event LLM model at startup: {e}", exc_info=True)
    raise RuntimeError(f"Could not initialize Event LLM model: {e}")


@app.on_event("startup")
async def startup_event():
    """Startup event to ensure model is loaded."""
    logger.info("Event LLM Service starting up...")
    # Ensure llm_model and its internal 'model' attribute are initialized
    if not hasattr(llm_model, 'model') or llm_model.model is None:
        logger.error("Event LLM model was not initialized during startup.")
        raise RuntimeError("Event LLM model is not ready.")
    logger.info("Event LLM Service startup complete.")


@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    """
    Health check endpoint.
    Returns 200 OK if the service is running and the model is loaded.
    """
    if hasattr(llm_model, 'model') and llm_model.model is not None:
        return {"status": "ok", "model_loaded": True}
    logger.error("Health check failed: Event LLM model not loaded.")
    raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                        detail="Event LLM model not loaded.")


@app.post("/generate-events", response_model=EventLLMGenerateResponse)
async def generate_events(request: EventLLMInput):
    """
    Accepts structured input (raw text, NER entities, SOA triplets) and returns the final
    standardized JSON output of events, entities, and metadata.
    This endpoint now correctly calls the two-pass pipeline for long articles.
    """
    trace_id = str(uuid.uuid4())
    logger.info("Received request for event generation.", extra={
                "trace_id": trace_id, "text_length": len(request.text)})
    try:
        if not hasattr(llm_model, 'model') or llm_model.model is None:
            logger.error(
                "LLM generation requested but model is not loaded.", extra={"trace_id": trace_id})
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                                detail="Event LLM model is not loaded.")

        # Corrected method call to use the new two-pass pipeline.
        # process_article_in_chunks only takes text and ner_entities from the request.
        # This is a key change from the orchestrator's perspective (which sends SOA triplets)
        # but the LLM logic internally handles the process and synthesizes the main events.
        response = llm_model.process_article_in_chunks(
            article_text=request.text, ner_entities=request.ner_entities)

        logger.info("Successfully generated events.", extra={
            "trace_id": trace_id, "job_id": response.job_id})
        return response
    except ValidationError as e:
        logger.warning(
            f"Invalid request payload for /generate-events: {e.errors()}", exc_info=True, extra={"trace_id": trace_id})
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail=f"Invalid input: {e.errors()}")
    except ValueError as e:
        logger.error(
            f"LLM generation/parsing error: {e}", exc_info=True, extra={"trace_id": trace_id})
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"LLM generation/parsing error: {e}")
    except HTTPException:
        # Re-raise HTTPException directly as it's already a well-formed HTTP error
        raise
    except Exception as e:
        logger.error(
            f"Internal server error during event generation: {e}", exc_info=True, extra={"trace_id": trace_id})
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail="Internal server error during event generation.")


# src/api/event_llm_service.py
# File path: src/api/event_llm_service.py

