"""Structured logging configuration using structlog."""

import logging
from typing import Any

import structlog


def configure_logging(log_level: str = "INFO") -> None:
    """
    Configure structlog with consistent processors and formatters.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    # Convert string log level to logging constant
    level = getattr(logging, log_level.upper(), logging.INFO)

    # Configure standard library logging
    logging.basicConfig(
        level=level,
        format="%(message)s",
    )

    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            # Add context like session_id, doc_id if present in kwargs
            structlog.processors.KeyValueRenderer(
                key_order=["timestamp", "level", "event"],
                sort_keys=False,
            ),
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str = __name__) -> Any:
    """Get a structlog logger instance.

    Args:
        name: Logger name (typically __name__)

    Returns:
        structlog logger instance
    """
    return structlog.get_logger(name)
