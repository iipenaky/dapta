"""
utils/logger.py
---------------
Structured, colour-coded logger used across all DAPTA modules.
Uses Python's built-in logging + rich for pretty console output.
"""

from __future__ import annotations
import logging
import sys
from pathlib import Path
from typing import Optional

try:
    from rich.logging import RichHandler
    _RICH_AVAILABLE = True
except ImportError:
    _RICH_AVAILABLE = False


def get_logger(
    name: str,
    level: int = logging.INFO,
    log_file: Optional[str | Path] = None,
) -> logging.Logger:
    """
    Return a named logger with console (rich) and optional file handler.

    Parameters
    ----------
    name      : Module name, e.g. ``"dapta.dae.metrics"``
    level     : Logging level (default INFO)
    log_file  : If provided, also write to this file

    Returns
    -------
    logging.Logger
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers on re-import
    if logger.handlers:
        return logger

    logger.setLevel(level)

    # Console handler
    if _RICH_AVAILABLE:
        console_handler = RichHandler(
            rich_tracebacks=True,
            show_path=True,
            markup=True,
        )
    else:
        console_handler = logging.StreamHandler(sys.stdout)
        fmt = logging.Formatter(
            "[%(asctime)s] %(levelname)s %(name)s — %(message)s",
            datefmt="%H:%M:%S",
        )
        console_handler.setFormatter(fmt)

    console_handler.setLevel(level)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_file is not None:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_fmt = logging.Formatter(
            "[%(asctime)s] %(levelname)s %(name)s — %(message)s"
        )
        file_handler.setFormatter(file_fmt)
        logger.addHandler(file_handler)

    return logger
