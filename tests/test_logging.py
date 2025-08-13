"""Unit tests for RAGToolBox logging setup."""
# pylint: disable=redefined-outer-name

import logging
import argparse
from pathlib import Path
from pydantic.v1.errors import NoneIsAllowedError
import pytest
from RAGToolBox.logging import RAGTBLogger, LoggingConfig


def _clear_root_handlers() -> None:
    """Remove all handlers from the root logger."""
    root = logging.getLogger()
    for h in root.handlers[:]:
        root.removeHandler(h)


@pytest.fixture(autouse=True)
def _reset_logging() -> None:
    """Ensure a clean logging config before/after each test."""
    _clear_root_handlers()
    yield
    _clear_root_handlers()
    logging.getLogger().setLevel(logging.WARNING)

# =====================
# UNIT TESTS
# =====================

def test_setup_logging_with_no_config() -> None:
    """Test setup_loggin function can load a default when no LoggingConfig is supplied"""
    RAGTBLogger.setup_logging()
    root = logging.getLogger()
    handlers = root.handlers

    assert root.level == logging.DEBUG
    assert len(handlers) == 1
    assert isinstance(handlers[0], logging.StreamHandler)
    assert handlers[0].level == logging.INFO

def test_setup_logging_console_only() -> None:
    """Test console handler exists with the configured level; no file handler by default."""
    RAGTBLogger.setup_logging(LoggingConfig(console_level="WARNING", log_file=None, force=True))

    root = logging.getLogger()
    handlers = root.handlers

    # Exactly one handler and it's a StreamHandler
    assert len(handlers) == 1
    assert isinstance(handlers[0], logging.StreamHandler)
    assert handlers[0].level == logging.WARNING

    # Root level is permissive so handlers do the filtering
    assert root.level == logging.DEBUG

def test_setup_logging_adds_file_handler(tmp_path: Path) -> None:
    """Test when log_file is set, a RotatingFileHandler is added and receives DEBUG logs."""
    log_path = tmp_path / "app.log"
    RAGTBLogger.setup_logging(LoggingConfig(console_level="ERROR", log_file=str(log_path),
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

def test_setup_logging_force_replaces_handlers() -> None:
    """Test force=True should replace existing handlers rather than stacking duplicates."""
    # First config: INFO console
    RAGTBLogger.setup_logging(LoggingConfig(console_level="INFO", log_file=None, force=True))
    root = logging.getLogger()
    assert len(root.handlers) == 1
    first_handler_id = id(root.handlers[0])
    assert root.handlers[0].level == logging.INFO

    # Second config: ERROR console, with force=True
    RAGTBLogger.setup_logging(LoggingConfig(console_level="ERROR", log_file=None, force=True))
    root = logging.getLogger()
    assert len(root.handlers) == 1
    # New handler instance (replaced)
    assert id(root.handlers[0]) != first_handler_id
    assert root.handlers[0].level == logging.ERROR

def test_setup_logging_no_force_appends_handlers() -> None:
    """Test force=False should keep existing handlers and add new ones."""
    # Start from a clean single console handler
    RAGTBLogger.setup_logging(LoggingConfig(console_level="INFO", log_file=None, force=True))
    root = logging.getLogger()
    assert len(root.handlers) == 1

    # Add another console handler without forcing
    RAGTBLogger.setup_logging(LoggingConfig(console_level="DEBUG", log_file=None, force=False))
    root = logging.getLogger()
    assert len(root.handlers) == 2
    # Ensure at least one handler has DEBUG level (the newly added one)
    assert any(h.level == logging.DEBUG for h in root.handlers)


def test_add_logging_args_defaults_and_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test add_logging_args attaches flags and respects defaults & CLI overrides."""
    # Ensure env vars do not affect defaults for this test
    monkeypatch.delenv("RAGTB_LOG_LEVEL", raising=False)
    monkeypatch.delenv("RAGTB_LOG_FILE", raising=False)
    monkeypatch.delenv("RAGTB_LOG_FILE_LEVEL", raising=False)

    parser = argparse.ArgumentParser()
    RAGTBLogger.add_logging_args(parser)

    # Parse with no flags -> defaults
    args = parser.parse_args([])
    assert hasattr(args, "log_level")
    assert hasattr(args, "log_file")
    assert hasattr(args, "log_file_level")
    assert args.log_level == "INFO"
    assert args.log_file is None
    assert args.log_file_level == "DEBUG"

    # Parse with overrides
    args2 = parser.parse_args([
        "--log-level", "ERROR",
        "--log-file", "/tmp/ragtb_test.log",
        "--log-file-level", "WARNING",
    ])
    assert args2.log_level == "ERROR"
    assert args2.log_file == "/tmp/ragtb_test.log"
    assert args2.log_file_level == "WARNING"


def test_configure_logging_from_args_with_file(tmp_path: Path) -> None:
    """
    configure_logging_from_args should install console + rotating file handler
    with the specified levels, and logs should land in the file.
    """
    log_file = tmp_path / "app.log"
    args = argparse.Namespace(
        log_level="ERROR",
        log_file=str(log_file),
        log_file_level="DEBUG",
    )

    # Configure logging from args
    RAGTBLogger.configure_logging_from_args(args)

    # Validate handlers
    root = logging.getLogger()
    kinds = {type(h) for h in root.handlers}
    assert logging.StreamHandler in kinds
    assert logging.handlers.RotatingFileHandler in kinds

    # Console handler level should be ERROR
    console_levels = [h.level for h in root.handlers if isinstance(h, logging.StreamHandler)]
    assert any(level == logging.ERROR for level in console_levels)

    # File handler level should be DEBUG
    file_levels = [
        h.level for h in root.handlers if isinstance(h, logging.handlers.RotatingFileHandler)
        ]
    assert any(level == logging.DEBUG for level in file_levels)

    # Emit a DEBUG log and ensure it is written to the file
    logger = logging.getLogger("RAGToolBox.test")
    logger.debug("debug-to-file")
    for h in root.handlers:
        try:
            h.flush()
        except Exception: # pylint: disable=broad-exception-caught
            pass

    assert log_file.exists()
    content = log_file.read_text()
    assert "debug-to-file" in content
