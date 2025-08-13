"""Unit tests for RAGToolBox logging setup."""
# pylint: disable=redefined-outer-name

import logging
from pathlib import Path
import pytest
from RAGToolBox.logging import setup_logging, LoggingConfig


def _clear_root_handlers():
    """Remove all handlers from the root logger."""
    root = logging.getLogger()
    for h in root.handlers[:]:
        root.removeHandler(h)


@pytest.fixture(autouse=True)
def _reset_logging():
    """Ensure a clean logging config before/after each test."""
    _clear_root_handlers()
    yield
    _clear_root_handlers()
    logging.getLogger().setLevel(logging.WARNING)

# =====================
# UNIT TESTS
# =====================

def test_setup_logging_with_no_config():
    """Test setup_loggin function can load a default when no LoggingConfig is supplied"""
    setup_logging()
    root = logging.getLogger()
    handlers = root.handlers

    assert root.level == logging.DEBUG
    assert len(handlers) == 1
    assert isinstance(handlers[0], logging.StreamHandler)
    assert handlers[0].level == logging.INFO

def test_setup_logging_console_only():
    """Test console handler exists with the configured level; no file handler by default."""
    setup_logging(LoggingConfig(console_level="WARNING", log_file=None, force=True))

    root = logging.getLogger()
    handlers = root.handlers

    # Exactly one handler and it's a StreamHandler
    assert len(handlers) == 1
    assert isinstance(handlers[0], logging.StreamHandler)
    assert handlers[0].level == logging.WARNING

    # Root level is permissive so handlers do the filtering
    assert root.level == logging.DEBUG

def test_setup_logging_adds_file_handler(tmp_path: Path):
    """Test when log_file is set, a RotatingFileHandler is added and receives DEBUG logs."""
    log_path = tmp_path / "app.log"
    setup_logging(LoggingConfig(console_level="ERROR", log_file=str(log_path),
                                file_level="DEBUG", force=True))

    root = logging.getLogger()
    # Should have two handlers: console + file
    assert len(root.handlers) == 2
    kinds = {type(h) for h in root.handlers}
    assert logging.StreamHandler in kinds
    assert logging.handlers.RotatingFileHandler in kinds

    # Write a DEBUG log and verify it lands in the file
    logger = logging.getLogger("RAGToolBox.test")
    logger.debug("hello-debug-to-file")
    # Force handlers to flush
    for h in root.handlers:
        h.flush()

    content = log_path.read_text()
    assert "hello-debug-to-file" in content

def test_setup_logging_force_replaces_handlers():
    """Test force=True should replace existing handlers rather than stacking duplicates."""
    # First config: INFO console
    setup_logging(LoggingConfig(console_level="INFO", log_file=None, force=True))
    root = logging.getLogger()
    assert len(root.handlers) == 1
    first_handler_id = id(root.handlers[0])
    assert root.handlers[0].level == logging.INFO

    # Second config: ERROR console, with force=True
    setup_logging(LoggingConfig(console_level="ERROR", log_file=None, force=True))
    root = logging.getLogger()
    assert len(root.handlers) == 1
    # New handler instance (replaced)
    assert id(root.handlers[0]) != first_handler_id
    assert root.handlers[0].level == logging.ERROR

def test_setup_logging_no_force_appends_handlers():
    """Test force=False should keep existing handlers and add new ones."""
    # Start from a clean single console handler
    setup_logging(LoggingConfig(console_level="INFO", log_file=None, force=True))
    root = logging.getLogger()
    assert len(root.handlers) == 1

    # Add another console handler without forcing
    setup_logging(LoggingConfig(console_level="DEBUG", log_file=None, force=False))
    root = logging.getLogger()
    assert len(root.handlers) == 2
    # Ensure at least one handler has DEBUG level (the newly added one)
    assert any(h.level == logging.DEBUG for h in root.handlers)
