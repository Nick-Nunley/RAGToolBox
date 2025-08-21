"""
RAGToolBox logging utilities.

Provides a small, consistent logging setup for both library usage and CLI
entry points. You can either:

- Call :func:`RAGTBLogger.setup_logging` directly with a :class:`LoggingConfig`
- Or wire standard flags into any argparse parser via
  :func:`RAGTBLogger.add_logging_args` and then call
  :func:`RAGTBLogger.configure_logging_from_args`.

Console and file logging can be configured independently; file logs use a
rotating handler.
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
    Configuration for RAGToolBox logging.

    Attributes:
        console_level:
            Minimum level for messages printed to stderr. One of:
            ``"CRITICAL"``, ``"ERROR"``, ``"WARNING"``, ``"INFO"``, ``"DEBUG"``, ``"NOTSET"``.
        log_file:
            Optional path to a log file. If provided, a rotating file handler is
            configured in addition to console logging.
        file_level:
            Minimum level for messages written to the log file (if enabled).
        rotate_max_bytes:
            Maximum size (in bytes) for the rotating log file before rollover.
        rotate_backups:
            Number of backup log files to keep.
        force:
            If ``True``, existing handlers on the root logger are removed before
            configuring new handlers. This prevents duplicate logs when
            re-initializing from multiple entry points or tests.
    """
    console_level: Literal["CRITICAL","ERROR","WARNING","INFO","DEBUG","NOTSET"] = "INFO"
    log_file: Optional[str] = None
    file_level: Literal["CRITICAL","ERROR","WARNING","INFO","DEBUG","NOTSET"] = "DEBUG"
    rotate_max_bytes: int = 5_000_000
    rotate_backups: int = 3
    force: bool = True

class RAGTBLogger:
    """
    Helper for consistent logger configuration across RAGToolBox.

    Typical usage:
        >>> from RAGToolBox.logging import RAGTBLogger, LoggingConfig
        >>> RAGTBLogger.setup_logging(LoggingConfig(console_level="INFO"))

    Or integrate with a CLI:
        >>> parser = argparse.ArgumentParser()
        >>> RAGTBLogger.add_logging_args(parser)
        >>> args = parser.parse_args()
        >>> RAGTBLogger.configure_logging_from_args(args)
    """

    @staticmethod
    def setup_logging(config: Optional[LoggingConfig] = None) -> None:
        """
        Configure root logging with a console handler and optional rotating file handler.

        Behavior:
            - Console handler (stderr) uses ``config.console_level``.
            - If ``config.log_file`` is set, a rotating file handler is added with
              ``config.file_level`` and rollover controlled by
              ``config.rotate_max_bytes`` / ``config.rotate_backups``.
            - When ``config.force`` is ``True``, any existing handlers on the root
              logger are removed to avoid duplicate output.

        Args:
            config:
                The logging configuration. If omitted, a default
                :class:`LoggingConfig` is used.

        Side Effects:
            Modifies the root logger's handlers and level (root is set to DEBUG to
            allow handlers to filter).
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
        """
        Add standard logging flags to an argparse parser.

        Flags added:
            --log-level
                Console logging level. Defaults to the value of the
                ``RAGTB_LOG_LEVEL`` environment variable if set, otherwise
                ``WARNING``.
            --log-file
                Path to a rotating log file. Defaults to ``RAGTB_LOG_FILE`` env var (if set).
            --log-file-level
                File logging level. Defaults to the value of
                ``RAGTB_LOG_FILE_LEVEL`` env var if set, otherwise ``DEBUG``.

        Args:
            parser: The parser to mutate.
        """
        parser.add_argument(
            '--log-level',
            default = os.getenv('RAGTB_LOG_LEVEL', 'WARNING'),
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
        """
        Construct a :class:`LoggingConfig` from parsed argparse args and configure logging.

        Expects the parser to have been augmented by
        :func:`RAGTBLogger.add_logging_args`.

        Args:
            args: Parsed argparse namespace containing ``log_level``,
                  ``log_file``, and ``log_file_level``.
        """
        RAGTBLogger.setup_logging(LoggingConfig(
            console_level=args.log_level,
            log_file=args.log_file,
            file_level=args.log_file_level,
            ))
