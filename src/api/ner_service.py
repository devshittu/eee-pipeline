# src/api/ner_service.py
# File path: src/api/ner_service.py

import logging
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from pydantic import ValidationError
from src.schemas.data_models import NERPredictRequest, NERPredictResponse, Entity
from src.core.ner_logic import NERModel
from src.utils.config_manager import ConfigManager
from src.utils.logger import setup_logging

# Load settings and configure logging
settings = ConfigManager.get_settings()
setup_logging()  # Setup logging before FastAPI app creation
logger = logging.getLogger("ner_service")

# Initialize FastAPI app
app = FastAPI(
    title="NER Service",
    description="Microservice for Named Entity Recognition using Hugging Face models.",
    version="1.0.0"
)

# Initialize the NER model (singleton pattern)
try:
    ner_model = NERModel()
except Exception as e:
    logger.critical(f"Failed to initialize NER model at startup: {e}")
    # Consider more robust startup, e.g., FastAPI's lifespan events
    # For now, if model fails to load, app won't start correctly.


@app.on_event("startup")
async def startup_event():
    """Startup event to ensure model is loaded."""
    logger.info("NER Service starting up...")
    # Model is initialized during app creation, but this can be a good place
    # for additional checks or async model loading if needed.
    if not hasattr(ner_model, 'nlp_pipeline'):
        logger.error("NER model pipeline was not initialized during startup.")
        raise RuntimeError("NER model pipeline is not ready.")
    logger.info("NER Service startup complete.")


@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    """
    Health check endpoint.
    Returns 200 OK if the service is running and the model is loaded.
    """
    if hasattr(ner_model, 'nlp_pipeline') and ner_model.nlp_pipeline is not None:
        return {"status": "ok", "model_loaded": True}
    logger.error("Health check failed: NER model not loaded.")
    raise HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="NER model not loaded.")


@app.post("/predict", response_model=NERPredictResponse)
async def predict_entities(request: NERPredictRequest):
    """
    Accepts text input and returns extracted entities with types, text, and character spans.
    """
    try:
        if not hasattr(ner_model, 'nlp_pipeline') or ner_model.nlp_pipeline is None:
            logger.error("NER prediction requested but model is not loaded.")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="NER model is not loaded.")

        entities = ner_model.predict(request.text)
        logger.info(
            f"Successfully processed NER for text (len: {len(request.text)}), found {len(entities)} entities.")
        return NERPredictResponse(entities=entities, text=request.text)
    except ValidationError as e:
        logger.warning(
            f"Invalid request payload for /predict: {e.errors()}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail=f"Invalid input: {e.errors()}")
    except HTTPException:  # Re-raise FastAPI HTTPExceptions
        raise
    except Exception as e:
        logger.error(
            f"Internal server error during NER prediction: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail="Internal server error during NER prediction.")
