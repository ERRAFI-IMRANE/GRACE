"""Minimal, consistent logging utility for training and evaluation scripts."""

import logging
import sys


def get_logger(name: str = "grace", level: int = logging.INFO) -> logging.Logger:
    """
    Create (or retrieve) a configured logger with consistent formatting.

    Args:
        name: logger name, typically the script or module name.
        level: logging level (default: logging.INFO).
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(level)
    return logger
