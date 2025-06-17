#!/bin/bash
set -e

# Activate the virtual environment
source /opt/venv/bin/activate

# Create log directory if it doesn't exist
mkdir -p /app/logs

# Load configuration and ensure logging is set up
/opt/venv/bin/python -c "from src.utils.logger import setup_logging; setup_logging(); from src.utils.config_manager import ConfigManager; ConfigManager.get_settings()"

# Get model name from config
MODEL_NAME=$(/opt/venv/bin/python -c "from src.utils.config_manager import ConfigManager; print(ConfigManager.get_settings().ner_service.model_name)")
MODEL_CACHE_DIR=$(/opt/venv/bin/python -c "from src.utils.config_manager import ConfigManager; print(ConfigManager.get_settings().ner_service.model_cache_dir)")

echo "Attempting to pre-download NER model: ${MODEL_NAME} to ${MODEL_CACHE_DIR}"

# Pre-download the Hugging Face model
# This ensures the model is present before the Uvicorn server starts
/opt/venv/bin/python -c """
from transformers import AutoTokenizer, AutoModelForTokenClassification
try:
    _ = AutoTokenizer.from_pretrained('${MODEL_NAME}', cache_dir='${MODEL_CACHE_DIR}')
    _ = AutoModelForTokenClassification.from_pretrained('${MODEL_NAME}', cache_dir='${MODEL_CACHE_DIR}')
    print('NER model pre-downloaded successfully.')
except Exception as e:
    print(f'Error pre-downloading NER model: {e}')
    exit(1)
"""

echo "Starting NER service..."
# Execute the main command passed to the script
exec "$@"


# entrypoint_ner.sh
# File path: entrypoint_ner.sh

