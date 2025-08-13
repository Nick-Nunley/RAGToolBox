from __future__ import annotations
import logging
import logging.handlers
import sys
from typing import Optional, Literal

_FMT = "%(asctime)s %(levelname)s %(name)s:%(lineno)d %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"

def setup_logging(
    console_level: Literal["CRITICAL","ERROR","WARNING","INFO","DEBUG","NOTSET"] = "INFO",
    log_file: Optional[str] = None,
    file_level: Literal["CRITICAL","ERROR","WARNING","INFO","DEBUG","NOTSET"] = "DEBUG",
    rotate_max_bytes: int = 5_000_000,
    rotate_backups: int = 3,
    force: bool = True,
    ) -> None:
    """
    Configure logging with separate console and optional file handlers.

    - Console: shown by default at 'console_level' (no file created).
    - File: only added if 'log_file' is provided; captures 'file_level' (usually DEBUG).

    'force=True' clears existing handlers so repeated calls don't duplicate output.
    """
    root = logging.getLogger()
    if force:
        for h in root.handlers[:]:
            root.removeHandler(h)

    root.setLevel(logging.DEBUG)

    # Console handler (stderr)
    ch = logging.StreamHandler(stream=sys.stderr)
    ch.setLevel(getattr(logging, console_level.upper(), logging.INFO))
    ch.setFormatter(logging.Formatter(_FMT, _DATEFMT))
    root.addHandler(ch)

    # Optional rotating file handler
    if log_file:
        fh = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=rotate_max_bytes, backupCount=rotate_backups
        )
        fh.setLevel(getattr(logging, file_level.upper(), logging.DEBUG))
        fh.setFormatter(logging.Formatter(_FMT, _DATEFMT))
        root.addHandler(fh)
