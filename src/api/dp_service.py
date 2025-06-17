# src/api/dp_service.py
# File path: src/api/dp_service.py

import logging
from fastapi import FastAPI, HTTPException, status
from pydantic import ValidationError
from src.schemas.data_models import DPExtractSOARequest, DPExtractSOAResponse, SOATriplet
from src.core.dp_logic import DPModel
from src.utils.config_manager import ConfigManager
from src.utils.logger import setup_logging

# Load settings and configure logging
settings = ConfigManager.get_settings()
setup_logging()
logger = logging.getLogger("dp_service")

# Initialize FastAPI app
app = FastAPI(
    title="DP Service",
    description="Microservice for Dependency Parsing and S-O-A triplet extraction.",
    version="1.0.0"
)

# Initialize the DP model (singleton pattern)
dp_model = None
try:
    logger.info("Initializing DP model...")
    dp_model = DPModel()
    logger.info("DP model initialized successfully.")
except Exception as e:
    logger.critical(
        f"Failed to initialize DP model at startup: {e}", exc_info=True)
    raise RuntimeError(f"Could not initialize DP model: {e}")


@app.on_event("startup")
async def startup_event():
    """Startup event to ensure model is loaded."""
    logger.info("DP Service starting up...")
    if not hasattr(dp_model, 'nlp') or dp_model.nlp is None:
        logger.error("DP model was not initialized during startup.")
        raise RuntimeError("DP model is not ready.")
    logger.info("DP Service startup complete.")


@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check():
    """
    Health check endpoint.
    Returns 200 OK if the service is running and the model is loaded.
    """
    if hasattr(dp_model, 'nlp') and dp_model.nlp is not None:
        return {"status": "ok", "model_loaded": True}
    logger.error("Health check failed: DP model not loaded.")
    raise HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="DP model not loaded.")


@app.post("/extract-soa", response_model=DPExtractSOAResponse)
async def extract_soa_triplets(request: DPExtractSOARequest):
    """
    Accepts text input and returns a list of identified S-O-A triplets.
    """
    try:
        if not hasattr(dp_model, 'nlp') or dp_model.nlp is None:
            logger.error("DP extraction requested but model is not loaded.")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail="DP model is not loaded.")

        soa_triplets = dp_model.extract_soa(request.text)
        logger.info(
            f"Successfully processed DP for text (len: {len(request.text)}), found {len(soa_triplets)} S-O-A triplets.")
        return DPExtractSOAResponse(soa_triplets=soa_triplets, text=request.text)
    except ValidationError as e:
        logger.warning(
            f"Invalid request payload for /extract-soa: {e.errors()}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST,
                            detail=f"Invalid input: {e.errors()}")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            f"Internal server error during S-O-A extraction: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail="Internal server error during S-O-A extraction.")

# # src/api/dp_service.py
# # File path: src/api/dp_service.py
