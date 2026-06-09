"""
core/config.py — Centralised configuration management for JARVIS.

Loads config.yaml once, validates required keys, and exposes a typed
Config dataclass.  All modules import from here instead of re-opening
config.yaml individually.
"""
from __future__ import annotations

import os
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

_ROOT = Path(__file__).parent.parent          # project root
_CONFIG_PATH = _ROOT / "config.yaml"


@dataclass
class Config:
    # Wake word
    wake_word: str = "hey_jarvis"
    mic_device_index: int | None = None

    # STT
    stt_backend: str = "openai"
    stt_openai_model: str = "whisper-1"
    stt_timeout: int = 8
    stt_phrase_limit: int = 15
    stt_energy_threshold: int = 300
    stt_language: str = "en-US"

    # TTS
    tts_engine: str = "openai"
    tts_openai_voice: str = "onyx"
    tts_openai_model: str = "tts-1"
    tts_pyttsx3_rate: int = 165

    # LLM
    llm_model: str = "gemini-2.5-flash"
    llm_max_history_turns: int = 20
    llm_max_input_chars: int = 500

    # Weather
    weather_default_city: str = "London"
    weather_units: str = "metric"

    # News
    news_country: str = "in"
    news_headline_count: int = 3

    # Notes
    notes_file: str = "notes.txt"

    # Stocks
    stocks_currency: str = "USD"

    # Logging
    log_level: str = "INFO"
    log_to_file: bool = False
    log_file_path: str = "jarvis.log"

    # Apps dict
    apps: dict[str, str] = field(default_factory=dict)

    # Email
    email_sender: str = ""
    email_contacts: dict[str, str] = field(default_factory=dict)

    # Calendar
    calendar_timezone: str = "UTC"

    # RAG
    rag_enabled: bool = True
    rag_chunk_size: int = 512
    rag_chunk_overlap: int = 64
    rag_top_k: int = 4
    rag_docs_dir: str = "docs"

    # MCP
    mcp_enabled: bool = True
    mcp_servers: list[dict[str, Any]] = field(default_factory=list)

    # Agent
    agent_max_steps: int = 5
    agent_think_aloud: bool = False       # log chain-of-thought to console

    # Evaluation
    eval_enabled: bool = True
    eval_db_path: str = "evaluation/eval_log.db"

    # API keys (loaded from .env)
    gemini_api_key: str = field(default_factory=lambda: os.getenv("GEMINI_API_KEY", ""))
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    openweather_api_key: str = field(default_factory=lambda: os.getenv("OPENWEATHER_API_KEY", ""))
    news_api_key: str = field(default_factory=lambda: os.getenv("NEWS_API_KEY", ""))
    smtp_sender_email: str = field(default_factory=lambda: os.getenv("SMTP_SENDER_EMAIL", ""))
    smtp_sender_password: str = field(default_factory=lambda: os.getenv("SMTP_SENDER_PASSWORD", ""))


def load_config(path: Path = _CONFIG_PATH) -> Config:
    """Load and validate config.yaml, returning a populated Config instance."""
    raw: dict[str, Any] = {}
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
    else:
        logger.warning(f"config.yaml not found at {path}. Using all defaults.")

    valid_fields = Config.__dataclass_fields__
    filtered = {k: v for k, v in raw.items() if k in valid_fields}
    cfg = Config(**filtered)

    if not cfg.gemini_api_key:
        logger.error("GEMINI_API_KEY not set. LLM will not work.")
    if not cfg.openai_api_key:
        logger.warning("OPENAI_API_KEY not set. STT/TTS will fall back to offline engines.")

    return cfg


# Singleton — import get_config() everywhere
_cfg: Config | None = None


def get_config() -> Config:
    global _cfg
    if _cfg is None:
        _cfg = load_config()
    return _cfg
