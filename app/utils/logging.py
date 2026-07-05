"""Logging configuration."""

from __future__ import annotations

import logging
import sys

from app.config import LOG_LEVEL


def setup_logging(name: str = "sales_assistant") -> logging.Logger:
    """Configure and return a logger instance."""
    # Fix Persian log output on Windows console
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
            sys.stderr.reconfigure(encoding="utf-8")
        except Exception:
            pass

    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        )
        logger.addHandler(handler)
        logger.setLevel(getattr(logging, LOG_LEVEL.upper(), logging.INFO))
    return logger


logger = setup_logging()
