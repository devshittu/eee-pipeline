#!/bin/bash
set -e

# Activate the virtual environment
source /opt/venv/bin/activate

# Create log and data directories if they don't exist
mkdir -p /app/logs /app/data

# Set service name for logging
export SERVICE_NAME="orchestrator_service"

# Load configuration and ensure logging is set up
python -c "from src.utils.logger import setup_logging; setup_logging(); from src.utils.config_manager import ConfigManager; ConfigManager.get_settings()"

echo "Starting Orchestrator service..."
# Execute the main command passed to the script
exec "$@"

# entrypoint_orchestrator.sh
# File path: entrypoint_orchestrator.sh
