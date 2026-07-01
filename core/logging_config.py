"""
core/logging_config.py — Structured logging setup for JARVIS.

Call setup_logging() once in main.py before anything else.
All other modules just call logging.getLogger(__name__).
"""
from __future__ import annotations

import io
import logging
import sys
from pathlib import Path
from typing import TextIO


def _force_utf8(stream: TextIO | None) -> TextIO | None:
    """Make a text stream tolerate non-Latin-1 characters.

    Windows consoles default to cp1252, which cannot encode characters that
    appear routinely in JARVIS output — the '→' in routing logs, '₹' in stock
    prices, the '✅' in the ingest tool. Writing them raises
    UnicodeEncodeError, which the logging module swallows into noisy
    '--- Logging error ---' tracebacks (and breaks plain print()).

    Reconfiguring to UTF-8 with errors='replace' fixes every such site at once
    and can never crash: an unrenderable glyph degrades to '?' instead.
    """
    if stream is None:
        return stream
    try:
        stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    except (AttributeError, ValueError, io.UnsupportedOperation):
        # Not a reconfigurable TextIOWrapper (e.g. already wrapped, or a pytest
        # capture object). Leaving it as-is is safe — worst case is the original
        # behaviour, not a regression.
        pass
    return stream


def setup_logging(level: str = "INFO", log_to_file: bool = False, log_file: str = "jarvis.log") -> None:
    """Configure root logger with console (and optionally file) handlers."""
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    # Do this first so both logging and bare print() stop choking on Unicode.
    _force_utf8(sys.stdout)
    _force_utf8(sys.stderr)

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
