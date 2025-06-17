#!/bin/bash
set -e

# Activate the virtual environment
source /opt/venv/bin/activate

# Create log directory if it doesn't exist
mkdir -p /app/logs

# Set service name for logging
export SERVICE_NAME="celery_worker"

export PYTHONPATH="/app:$PYTHONPATH"

# Load configuration and ensure logging is set up
# This ensures config and logging are ready before Celery starts
python -c "from src.utils.logger import setup_logging; setup_logging(); from src.utils.config_manager import ConfigManager; ConfigManager.get_settings()"

echo "Starting Celery worker..."

# Execute the main command passed to the script (which will be `celery -A ... worker ...`)
exec "$@"

# entrypoint_celery_worker.sh
# File path: entrypoint_celery_worker.sh
