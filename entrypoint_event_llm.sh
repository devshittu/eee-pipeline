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

# # --- DEBUGGING LINES ADDED HERE ---
# echo "--- DEBUG: Checking HUGGINGFACE_TOKEN environment variable (from shell) ---"
# echo "HUGGINGFACE_TOKEN value: \"${HUGGINGFACE_TOKEN}\"" # Check HUGGINGFACE_TOKEN directly
# echo "------------------------------------------------------------------"
# # --- END DEBUGGING LINE ---

# Pre-download the Hugging Face LLM model
python -c "
import os
import torch
# *** IMPORTANT CHANGE HERE ***
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig # Changed from AutoModelForSeq2SeqLM to AutoModelForCausalLM
from huggingface_hub import login, HfFolder

model_path = os.getenv('LLM_MODEL_PATH', '${MODEL_PATH}')
model_name = os.getenv('LLM_MODEL_NAME', '${MODEL_NAME}')
gpu_enabled = os.getenv('GPU_ENABLED', 'False').lower() == 'true'
device = 'cuda' if gpu_enabled and torch.cuda.is_available() else 'cpu'

# # --- DEBUGGING: Print all environment variables for a full check ---
# print('--- DEBUG: All environment variables in Python ---')
# for key, value in os.environ.items():
#     print(f'{key}={value}')
# print('----------------------------------------------------')
# # --- END DEBUGGING ---

# Authenticate Hugging Face Hub
hf_token = os.getenv('HUGGINGFACE_TOKEN')
if not hf_token:
    hf_token = os.getenv('HF_TOKEN')

if hf_token:
    print('Hugging Face Token environment variable is present in Python script.')
    try:
        login(token=hf_token)
        print('Successfully logged in to Hugging Face Hub within Python script.')
    except Exception as e:
        print(f'Error during Hugging Face Hub login in Python script: {e}')
else:
    print('Hugging Face Token environment variable is NOT present in Python script or is empty.')

print(f'Loading model {model_name} to device {device} from {model_path}...')
try:
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=model_path)
    if device == 'cuda':
        try:
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            # *** IMPORTANT CHANGE HERE ***
            model = AutoModelForCausalLM.from_pretrained( # Changed from AutoModelForSeq2SeqLM to AutoModelForCausalLM
                model_name,
                quantization_config=quantization_config,
                cache_dir=model_path
            )
            print('LLM model loaded with 8-bit quantization on cuda.')
        except Exception as e:
            print(f'8-bit quantization failed: {e}. Falling back to full precision.')
            # *** IMPORTANT CHANGE HERE ***
            model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=model_path).to(device) # Changed from AutoModelForSeq2SeqLM to AutoModelForCausalLM
    else:
        # *** IMPORTANT CHANGE HERE ***
        model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=model_path).to(device) # Changed from AutoModelForSeq2SeqLM to AutoModelForCausalLM
    print(f'LLM model pre-downloaded/loaded successfully on {device}.')
except Exception as e:
    print(f'Error pre-downloading/loading LLM model: {e}')
    exit(1)
"

echo "Starting Event LLM service..."
exec "$@"

# entrypoint_event_llm.sh
# File path: entrypoint_event_llm.sh
