#!/bin/bash
set -e

# Activate the virtual environment
source /opt/venv/bin/activate

# Create log directory if it doesn't exist
mkdir -p /app/logs

# Load configuration and ensure logging is set up
python -c "from src.utils.logger import setup_logging; setup_logging(); from src.utils.config_manager import ConfigManager; ConfigManager.get_settings()"

# Get model name and path from config
MODEL_NAME=$(python -c "from src.utils.config_manager import ConfigManager; print(ConfigManager.get_settings().event_llm_service.model_name)")
MODEL_PATH=$(python -c "from src.utils.config_manager import ConfigManager; print(ConfigManager.get_settings().event_llm_service.model_path)")
GPU_ENABLED=$(python -c "from src.utils.config_manager import ConfigManager; print(ConfigManager.get_settings().general.gpu_enabled)")

echo "Attempting to pre-download/load Event LLM model: ${MODEL_NAME} from path/cache: ${MODEL_PATH}"

# Pre-download the Hugging Face LLM model
python -c "
import os
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, BitsAndBytesConfig
model_path = os.getenv('LLM_MODEL_PATH', '${MODEL_PATH}')
model_name = os.getenv('LLM_MODEL_NAME', '${MODEL_NAME}')
gpu_enabled = os.getenv('GPU_ENABLED', 'False').lower() == 'true'
device = 'cuda' if gpu_enabled and torch.cuda.is_available() else 'cpu'
print(f'Loading model {model_name} to device {device} from {model_path}...')
try:
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=model_path)
    if device == 'cuda':
        try:
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name,
                quantization_config=quantization_config,
                cache_dir=model_path
            )
            print('LLM model loaded with 8-bit quantization on cuda.')
        except Exception as e:
            print(f'8-bit quantization failed: {e}. Falling back to full precision.')
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name, cache_dir=model_path).to(device)
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, cache_dir=model_path).to(device)
    print(f'LLM model pre-downloaded/loaded successfully on {device}.')
except Exception as e:
    print(f'Error pre-downloading/loading LLM model: {e}')
    exit(1)
"

echo "Starting Event LLM service..."
exec "$@"

# entrypoint_event_llm.sh
# File path: entrypoint_event_llm.sh
