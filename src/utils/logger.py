"""
Logging utility module for video-summarizer-cli.

Provides centralized logging configuration using Rich for pretty console output.
"""

import logging
from logging import Logger

from rich.logging import RichHandler


def setup_logging(level: str = "INFO", fmt: str = "%(asctime)s [%(levelname)s] %(name)s: %(message)s") -> None:
    """
    Configure the root logger with RichHandler for pretty console output.

    This function should be called once at application startup to set up
    global logging configuration. Individual modules should use get_logger()
    to obtain logger instances.

    Args:
        level: Logging level as string (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        fmt: Log message format string.

    Raises:
        ValueError: If level is not a valid logging level.
    """
    numeric_level = getattr(logging, level.upper(), None)

    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid logging level: {level}")

    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    root_logger.handlers.clear()

    handler = RichHandler(
        show_time=True,
        show_path=False,
        rich_tracebacks=True,
        tracebacks_show_locals=True,
    )
    handler.setFormatter(logging.Formatter(fmt, datefmt="%Y-%m-%d %H:%M:%S"))

    root_logger.addHandler(handler)


def get_logger(name: str) -> Logger:
    """
    Get a named logger instance.

    This function retrieves an existing logger or creates a new one.
    It does not configure handlers — use setup_logging() once at startup.

    Args:
        name: Logger name, typically __name__ of the calling module.

    Returns:
        Logger instance with the specified name.

    Example:
        >>> from src.utils.logger import setup_logging, get_logger
        >>> setup_logging(level="DEBUG")
        >>> logger = get_logger(__name__)
        >>> logger.info("Application started")
    """
    return logging.getLogger(name)
