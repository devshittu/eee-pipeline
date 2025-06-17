# src/utils/common_utils.py

import uuid
import os
import sys
from typing import Any, Dict
from src.utils.logger import logger
from src.schemas.data_models import ErrorResponse


def generate_trace_id() -> str:
    """Generates a unique trace ID for request tracking."""
    return str(uuid.uuid4())


def handle_exception(e: Exception, service_name: str, trace_id: Optional[str] = None) -> ErrorResponse:
    """
    Centralized exception handler that logs the error and returns a standardized ErrorResponse.
    """
    error_code = "INTERNAL_SERVER_ERROR"
    if isinstance(e, ValueError):
        error_code = "BAD_REQUEST"
        status_code = 400
    elif isinstance(e, FileNotFoundError):
        error_code = "NOT_FOUND"
        status_code = 404
    else:
        status_code = 500

    if not trace_id:
        trace_id = generate_trace_id()

    logger.error(f"[{service_name}] An error occurred: {e}",
                 exc_info=True,
                 extra_data={"trace_id": trace_id, "error_code": error_code})

    return ErrorResponse(
        detail=f"An unexpected error occurred: {e}",
        error_code=error_code,
        trace_id=trace_id
    ), status_code


def get_service_url(service_name: str, config: Any) -> str:
    """
    Constructs the internal URL for a microservice based on Docker Compose service names
    and configured ports.
    """
    host = service_name.replace('-', '_')  # Docker Compose service name
    port = getattr(config.service_ports, host)
    return f"http://{service_name}:{port}"


def ensure_dirs_exist(paths: List[str]):
    """Ensures that a list of directories exist."""
    for path in paths:
        os.makedirs(path, exist_ok=True)
