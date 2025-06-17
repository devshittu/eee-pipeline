# src/api/event_llm_service.py
# File path: src/api/event_llm_service.py

import logging
from fastapi import FastAPI, HTTPException, status
from pydantic import ValidationError
from src.schemas.data_models import EventLLMInput, EventLLMGenerateResponse
from src.core.event_llm_logic import EventLLMModel
from src.utils.config_manager import ConfigManager
from src.utils.logger import setup_logging

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
    Accepts structured input (raw text, NER entities, S-O-A triplets)
    and returns the final standardized JSON output of events, entities, and metadata.
    This endpoint is optimized for generative inference.
    """
    try:
        # Corrected typo: changed 'llm\u0cae' to 'llm_model'
        if not hasattr(llm_model, 'model') or llm_model.model is None:
            logger.error("LLM generation requested but model is not loaded.")
            raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                                detail="Event LLM model is not loaded.")

        response = llm_model.generate_events(request)
        logger.info(
            f"Successfully generated events for text (len: {len(request.text)}).")
        return response
    except ValidationError as e:
        logger.warning(
            f"Invalid request payload for /generate-events: {e.errors()}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail=f"Invalid input: {e.errors()}")
    except ValueError as e:
        logger.error(f"LLM generation/parsing error: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"LLM generation/parsing error: {e}")
    except HTTPException:
        # Re-raise HTTPException directly as it's already a well-formed HTTP error
        raise
    except Exception as e:
        logger.error(
            f"Internal server error during event generation: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail="Internal server error during event generation.")

# # src/api/event_llm_service.py
# # File path: src/api/event_llm_service.py
