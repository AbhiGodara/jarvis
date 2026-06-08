"""
Text-to-Speech module for JARVIS.

Primary engine: OpenAI TTS API
  - Voice: "onyx" (deep, masculine, perfect for JARVIS)
  - Model: "tts-1" (fastest) or "tts-1-hd" (higher quality)
  - Streams MP3 bytes directly to pygame for playback
  - No local model files, no C drive installs

Fallback: pyttsx3 (offline, used if OpenAI API is unreachable)
"""
import io
import os
import threading
import logging
import tempfile
import yaml
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# Load config
try:
    with open("config.yaml", "r") as f:
        _config = yaml.safe_load(f)
except Exception:
    _config = {}

TTS_ENGINE = _config.get("tts_engine", "openai")
OPENAI_VOICE = _config.get("tts_openai_voice", "onyx")
OPENAI_TTS_MODEL = _config.get("tts_openai_model", "tts-1")
PYTTSX3_RATE = _config.get("tts_pyttsx3_rate", 165)

# Init pygame mixer
import pygame
pygame.mixer.init()
_tts_lock = threading.Lock()

# Init offline fallback engine
import pyttsx3
_offline_engine = pyttsx3.init()
_offline_engine.setProperty("rate", PYTTSX3_RATE)
_offline_engine.setProperty("volume", 1.0)

# Mute pygame intro message
import os as _os
_os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"


def stop() -> None:
    """Stop any currently playing speech immediately."""
    try:
        if pygame.mixer.get_init() and pygame.mixer.music.get_busy():
            pygame.mixer.music.stop()
            pygame.mixer.music.unload()
    except Exception:
        pass
    try:
        _offline_engine.stop()
    except Exception:
        pass


def speak(text: str, **kwargs) -> None:
    """
    Convert text to speech and play it.

    Uses OpenAI TTS as the primary engine (best quality, natural voice).
    Falls back to pyttsx3 offline if OpenAI is unavailable.

    Args:
        text: The text string to speak aloud
    """
    if not text or not text.strip():
        return

    with _tts_lock:
        if TTS_ENGINE == "openai":
            try:
                _speak_openai(text)
                return
            except Exception as e:
                logger.warning(f"OpenAI TTS failed: {e}. Falling back to pyttsx3.")

        _speak_pyttsx3(text)


def _speak_openai(text: str) -> None:
    """Synthesise speech using OpenAI TTS API and play via pygame."""
    from openai import OpenAI

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set in .env file.")

    client = OpenAI(api_key=api_key)

    logger.debug(f"OpenAI TTS: synthesizing {len(text)} chars as '{OPENAI_VOICE}'...")

    # Stream response to a temp file
    tmp_path = tempfile.mktemp(suffix=".mp3")
    try:
        response = client.audio.speech.create(
            model=OPENAI_TTS_MODEL,
            voice=OPENAI_VOICE,
            input=text,
            response_format="mp3"
        )
        response.stream_to_file(tmp_path)

        pygame.mixer.music.load(tmp_path)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.wait(50)

    finally:
        pygame.mixer.music.stop()
        try:
            pygame.mixer.music.unload()
        except Exception:
            pass
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except OSError:
            pass


def _speak_pyttsx3(text: str) -> None:
    """Synthesise and play speech using the offline pyttsx3 engine."""
    _offline_engine.say(text)
    _offline_engine.runAndWait()
