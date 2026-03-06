"""MindMesh AI — Logging Configuration."""

import logging
import sys

from .config import settings


def setup_logging() -> logging.Logger:
    """Configure and return the application logger."""
    _logger = logging.getLogger("mindmesh")
    _logger.setLevel(getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO))

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO))

    formatter = logging.Formatter(settings.LOG_FORMAT)
    console_handler.setFormatter(formatter)

    if not _logger.handlers:
        _logger.addHandler(console_handler)

    return _logger


logger = setup_logging()
