"""
RAGToolBox Logging module.

Provides a function for setting up loggers through module-specific
CLI-entry points.
"""

from __future__ import annotations
import logging
import logging.handlers
import os
import argparse
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

class RAGTBLogger:
    """Class wrapping up logging components specific to RAGToolBox"""

    @staticmethod
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

    @staticmethod
    def add_logging_args(parser: argparse.ArgumentParser) -> None:
        """Attach standard logging options to any CLI parser."""
        parser.add_argument(
            '--log-level',
            default = os.getenv('RAGTB_LOG_LEVEL', 'INFO'),
            choices = ['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG', 'NOTSET'],
            help = 'Console logging level (default: INFO)'
            )

        parser.add_argument(
            '--log-file',
            default = os.getenv('RAGTB_LOG_FILE'),
            help = 'If set, write detailed logs to this file (rotating)'
            )

        parser.add_argument(
            '--log-file-level',
            default = os.getenv('RAGTB_LOG_FILE_LEVEL', 'DEBUG'),
            choices = ['CRITICAL', 'ERROR', 'WARNING', 'INFO', 'DEBUG', 'NOTSET'],
            help = 'File log level if --log-file is provided (default: DEBUG)'
            )

    @staticmethod
    def configure_logging_from_args(args: argparse.Namespace) -> None:
        """Apply logging config based on parsed args."""
        RAGTBLogger.setup_logging(LoggingConfig(
            console_level=args.log_level,
            log_file=args.log_file,
            file_level=args.log_file_level,
            ))
