#!/bin/bash
set -e

# Activate the virtual environment
source /opt/venv/bin/activate

# Create log directory if it doesn't exist
mkdir -p /app/logs

# Load configuration and ensure logging is set up
python -c "from src.utils.logger import setup_logging; setup_logging(); from src.utils.config_manager import ConfigManager; ConfigManager.get_settings()"

# Get model name from config
MODEL_NAME=$(python -c "from src.utils.config_manager import ConfigManager; print(ConfigManager.get_settings().dp_service.model_name)")

echo "Installing Dependency Parsing model: ${MODEL_NAME}"

# Install the model as a package if not already present
if ! python -c "import spacy; spacy.load('${MODEL_NAME}')" 2>/dev/null; then
    echo "Model not found or not loadable. Downloading ${MODEL_NAME}..."
    if ! python -m spacy download "${MODEL_NAME}"; then
        echo "Error: Failed to download and install spaCy model ${MODEL_NAME}."
        exit 1
    fi
fi

# Validate the model installation
if ! python -c "import spacy; spacy.load('${MODEL_NAME}')" 2>/dev/null; then
    echo "Error: Failed to load spaCy model ${MODEL_NAME} after installation."
    exit 1
fi

echo "Dependency Parsing model installed and validated successfully."

echo "Starting DP service..."
exec "$@"


# entrypoint_dp.sh
# File path: entrypoint_dp.sh

