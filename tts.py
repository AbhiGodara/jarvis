"""
Text-to-Speech module for JARVIS.

Primary engine: OpenAI TTS API (PCM streaming)
  - Voice: "onyx" (deep, masculine, perfect for JARVIS)
  - Model: "tts-1" (fastest) or "tts-1-hd" (higher quality)
  - Downloads PCM via HTTP streaming, plays via sounddevice.
  - Faster than the old buffered-MP3 approach: no disk write, no MP3 decode.

Fallback: pyttsx3 (offline, used if OpenAI API is unreachable)
"""
import os
import threading
import logging
import numpy as np
from dotenv import load_dotenv

# Phrases pre-downloaded at startup so they play without API latency.
_audio_cache: dict[str, np.ndarray] = {}

import sounddevice as sd

from core.config import get_config

load_dotenv()
logger = logging.getLogger(__name__)

_cfg = get_config()
TTS_ENGINE = _cfg.tts_engine
OPENAI_VOICE = _cfg.tts_openai_voice
OPENAI_TTS_MODEL = _cfg.tts_openai_model
PYTTSX3_RATE = _cfg.tts_pyttsx3_rate
OPENAI_TTS_TIMEOUT = _cfg.tts_openai_timeout

# OpenAI PCM output: 24 kHz, 16-bit signed, mono
_TTS_SAMPLE_RATE = 24000

# Set by stop() to abort an in-progress streaming download.
# stop() also calls sd.stop() to interrupt playback.
_stop_requested = threading.Event()

# Serialise speak() calls so concurrent threads don't clash on the audio device.
_tts_lock = threading.Lock()

# Reuse one OpenAI client across calls (avoids per-call connection pool setup).
_openai_client = None


def _get_openai_client():
    global _openai_client
    if _openai_client is None:
        from openai import OpenAI
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set in .env file.")
        _openai_client = OpenAI(api_key=api_key)
    return _openai_client


# Init offline fallback engine at import time
import pyttsx3
_offline_engine = pyttsx3.init()
_offline_engine.setProperty("rate", PYTTSX3_RATE)
_offline_engine.setProperty("volume", 1.0)


def preload(text: str) -> None:
    """Pre-download TTS audio for a phrase and cache it in memory.

    Call this in a background thread at startup for frequently-spoken phrases
    (e.g. "Yes, sir?") so the first real speak() call plays from the cache
    instead of hitting the API.
    """
    if text in _audio_cache:
        return
    try:
        client = _get_openai_client()
        chunks: list[bytes] = []
        with client.audio.speech.with_streaming_response.create(
            model=OPENAI_TTS_MODEL,
            voice=OPENAI_VOICE,
            input=text,
            response_format="pcm",
            timeout=OPENAI_TTS_TIMEOUT,
        ) as response:
            for chunk in response.iter_bytes(chunk_size=4096):
                if chunk:
                    chunks.append(chunk)
        if chunks:
            _audio_cache[text] = np.frombuffer(b"".join(chunks), dtype=np.int16)
            logger.info(f"TTS preloaded: '{text}' ({len(chunks)} chunks)")
    except Exception as e:
        logger.debug(f"TTS preload failed for '{text}': {e}")


def stop() -> None:
    """Interrupt any currently playing speech immediately."""
    _stop_requested.set()
    try:
        sd.stop()   # stops the active sd.play() stream; safe if nothing is playing
    except Exception:
        pass
    try:
        _offline_engine.stop()
    except Exception:
        pass


def speak(text: str, **kwargs) -> None:
    """
    Convert text to speech and play it (blocks until finished or stopped).

    Primary: OpenAI TTS PCM streaming via sounddevice.
    Fallback: pyttsx3 offline engine.
    """
    if not text or not text.strip():
        return

    with _tts_lock:
        _stop_requested.clear()
        if TTS_ENGINE == "openai":
            try:
                _speak_openai(text)
                return
            except Exception as e:
                logger.warning(f"OpenAI TTS failed: {e}. Falling back to pyttsx3.")

        _speak_pyttsx3(text)


def _speak_openai(text: str) -> None:
    """
    Download PCM from OpenAI TTS and play via sounddevice.

    Uses response_format='pcm' (raw 24kHz int16) which avoids MP3 encode/
    decode overhead. We collect all chunks (fast network download) then play
    via sd.play()+sd.wait(), which is the simplest correct way to ensure
    the full audio plays before we return.
    """
    client = _get_openai_client()
    logger.debug(f"OpenAI TTS: {len(text)} chars, voice={OPENAI_VOICE}")

    # Serve from cache if this phrase was preloaded
    cached = _audio_cache.get(text)
    if cached is not None:
        logger.debug(f"TTS cache hit: '{text}'")
        audio = cached
    else:
        pcm_chunks: list[bytes] = []
        with client.audio.speech.with_streaming_response.create(
            model=OPENAI_TTS_MODEL,
            voice=OPENAI_VOICE,
            input=text,
            response_format="pcm",
            timeout=OPENAI_TTS_TIMEOUT,
        ) as response:
            for chunk in response.iter_bytes(chunk_size=4096):
                if _stop_requested.is_set():
                    return
                if chunk:
                    pcm_chunks.append(chunk)

        if not pcm_chunks or _stop_requested.is_set():
            return

        audio = np.frombuffer(b"".join(pcm_chunks), dtype=np.int16)
    sd.play(audio, samplerate=_TTS_SAMPLE_RATE)
    # sd.wait() blocks until playback finishes.
    # stop() calls sd.stop() which causes sd.wait() to return immediately.
    sd.wait()


def speak_offline(text: str) -> None:
    """Speak using the offline pyttsx3 engine — instant, no network round-trip.

    Use for startup announcements where response time matters more than voice
    quality. OpenAI TTS takes 2–5 s (including retries); pyttsx3 is < 100 ms.
    """
    if not text or not text.strip():
        return
    with _tts_lock:
        _speak_pyttsx3(text)


def _speak_pyttsx3(text: str) -> None:
    """Synthesise and play speech using the offline pyttsx3 engine."""
    _offline_engine.say(text)
    _offline_engine.runAndWait()
