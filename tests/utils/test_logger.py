"""
Tests for src.utils.logger module.
"""

import logging

from src.utils.logger import get_logger, setup_logging


def test_setup_logging_sets_level():
    setup_logging(level="DEBUG")

    root_logger = logging.getLogger()
    assert root_logger.level == logging.DEBUG


def test_setup_logging_default_level():
    setup_logging()

    root_logger = logging.getLogger()
    assert root_logger.level == logging.INFO


def test_get_logger_returns_named_logger():
    logger = get_logger("test_module")

    assert logger.name == "test_module"
    assert isinstance(logger, logging.Logger)


def test_get_logger_returns_same_instance():
    logger1 = get_logger("duplicate_test")
    logger2 = get_logger("duplicate_test")

    assert logger1 is logger2
