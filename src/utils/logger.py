# src/utils/logger.py
# File path: src/utils/logger.py

import logging
import logging.config
from pythonjsonlogger.jsonlogger import JsonFormatter
import os
import yaml


class CustomJsonFormatter(JsonFormatter):
    def add_fields(self, log_record, message_dict):
        super(CustomJsonFormatter, self).add_fields(log_record, message_dict)
        # Add custom fields here if needed
        # For example, to add `service_name` from environment variable
        # service_name = os.getenv("SERVICE_NAME", "unknown_service")
        # log_record['service_name'] = service_name


def setup_logging(config_path: str = "./config/settings.yaml"):
    """
    Sets up structured logging based on the configuration file.
    """
    if not os.path.exists(config_path):
        logging.warning(
            f"Logging configuration file not found at {config_path}. Using default console logging.")
        logging.basicConfig(
            level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        return

    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        log_config = config.get("logging")
        if log_config:
            # Ensure log directories exist before configuring handlers
            for handler_name, handler_config in log_config.get("handlers", {}).items():
                if 'filename' in handler_config:
                    log_dir = os.path.dirname(handler_config['filename'])
                    if log_dir and not os.path.exists(log_dir):
                        os.makedirs(log_dir, exist_ok=True)
            logging.config.dictConfig(log_config)
            logging.info("Logging configured successfully.")
        else:
            logging.warning(
                "No 'logging' section found in settings.yaml. Using default console logging.")
            logging.basicConfig(
                level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    except Exception as e:
        logging.error(
            f"Error setting up logging from {config_path}: {e}", exc_info=True)
        logging.basicConfig(
            # Fallback
            level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
