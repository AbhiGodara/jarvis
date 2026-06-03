import os
import tempfile
import threading
import logging

import pygame
import pyttsx3
from gtts import gTTS
import requests.exceptions

logger = logging.getLogger(__name__)

pygame.mixer.init()
_tts_lock = threading.Lock()

_offline_engine = pyttsx3.init()
_offline_engine.setProperty("rate", 165)
_offline_engine.setProperty("volume", 1.0)


def stop() -> None:
    """Stop any ongoing speech playback."""
    if pygame.mixer.get_init() and pygame.mixer.music.get_busy():
        pygame.mixer.music.stop()
    try:
        _offline_engine.stop()
    except Exception:
        pass


def speak(text: str, accent: str = "co.uk", engine: str = "gtts") -> None:
    """
    Convert text to speech and play it through the default audio output.

    Uses gTTS (Google) as the primary engine with automatic fallback to
    pyttsx3 (offline) if Google is unreachable. Thread-safe.

    Args:
        text: The text string to speak aloud
        accent: gTTS TLD accent code ("co.uk", "com", "com.au")
        engine: Primary engine to attempt. "gtts" or "pyttsx3"
    """
    if not text or not text.strip():
        return

    with _tts_lock:
        if engine == "gtts":
            try:
                _speak_gtts(text, accent)
                return
            except (requests.exceptions.ConnectionError,
                    requests.exceptions.Timeout,
                    Exception) as e:
                logger.warning(f"gTTS failed: {e}. Falling back to pyttsx3.")

        _speak_pyttsx3(text)


def _speak_gtts(text: str, accent: str) -> None:
    """
    Internal: synthesise speech using Google TTS and play via pygame.
    Raises on any network failure so the caller can fall back.
    """
    tts = gTTS(text=text, lang="en", tld=accent)
    tmp_path = tempfile.mktemp(suffix=".mp3")

    try:
        tts.save(tmp_path)
        pygame.mixer.music.load(tmp_path)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():
            pygame.time.wait(50)
    finally:
        pygame.mixer.music.stop()
        pygame.mixer.music.unload()
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except OSError:
            pass


def _speak_pyttsx3(text: str) -> None:
    """Internal: synthesise and play speech using the offline pyttsx3 engine."""
    _offline_engine.say(text)
    _offline_engine.runAndWait()
