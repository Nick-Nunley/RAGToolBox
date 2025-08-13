"""
RAGToolBox Logging module.

Provides a function for setting up loggers through module-specific
CLI-entry points.
"""

from __future__ import annotations
import logging
import logging.handlers
import sys
from typing import Optional, Literal
from dataclasses import dataclass


_FMT = "%(asctime)s %(levelname)s %(name)s:%(lineno)d %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"

@dataclass(frozen=True)
class LoggingConfig:
    """
    Holds all the optional config settings for Logging.
    """
    console_level: Literal["CRITICAL","ERROR","WARNING","INFO","DEBUG","NOTSET"] = "INFO"
    log_file: Optional[str] = None
    file_level: Literal["CRITICAL","ERROR","WARNING","INFO","DEBUG","NOTSET"] = "DEBUG"
    rotate_max_bytes: int = 5_000_000
    rotate_backups: int = 3
    force: bool = True

def setup_logging(config: Optional[LoggingConfig] = None) -> None:
    """
    Configure logging with separate console and optional file handlers.

    - Console: shown by default at 'console_level' (no file created).
    - File: only added if 'log_file' is provided; captures 'file_level' (usually DEBUG).

    'force=True' clears existing handlers so repeated calls don't duplicate output.
    """
    if config is None:
        config = LoggingConfig()
    root = logging.getLogger()
    if config.force:
        for h in root.handlers[:]:
            root.removeHandler(h)

    root.setLevel(logging.DEBUG)

    # Console handler (stderr)
    ch = logging.StreamHandler(stream=sys.stderr)
    ch.setLevel(getattr(logging, config.console_level.upper(), logging.INFO))
    ch.setFormatter(logging.Formatter(_FMT, _DATEFMT))
    root.addHandler(ch)

    # Optional rotating file handler
    if config.log_file:
        fh = logging.handlers.RotatingFileHandler(
            config.log_file, maxBytes=config.rotate_max_bytes, backupCount=config.rotate_backups
        )
        fh.setLevel(getattr(logging, config.file_level.upper(), logging.DEBUG))
        fh.setFormatter(logging.Formatter(_FMT, _DATEFMT))
        root.addHandler(fh)
