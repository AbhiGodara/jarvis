"""
core/logging_config.py — Structured logging setup for JARVIS.

Call setup_logging() once in main.py before anything else.
All other modules just call logging.getLogger(__name__).
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path


def setup_logging(level: str = "INFO", log_to_file: bool = False, log_file: str = "jarvis.log") -> None:
    """Configure root logger with console (and optionally file) handlers."""
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    fmt = "%(asctime)s  %(levelname)-8s  %(name)-30s  %(message)s"
    datefmt = "%H:%M:%S"

    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]

    if log_to_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(level=numeric_level, format=fmt, datefmt=datefmt, handlers=handlers, force=True)

    # Silence noisy third-party loggers
    for noisy in [
        "google", "urllib3", "matplotlib", "openwakeword",
        "speech_recognition", "httpcore", "httpx",
        "chromadb", "sentence_transformers", "transformers",
    ]:
        logging.getLogger(noisy).setLevel(logging.WARNING)
