"""
Speech-to-Text module for JARVIS.

Backend: OpenAI Whisper API (whisper-1)
  - No local model download
  - Accurate across accents (including Indian English)
  - ~300-500ms transcription latency

There is no offline STT fallback: on API failure listen() returns None
and main.py tells the user it didn't catch that.
"""
import io
import wave
import logging
import numpy as np
import sounddevice as sd
import os
from dotenv import load_dotenv

from core.config import get_config

load_dotenv()
logger = logging.getLogger(__name__)

_cfg = get_config()
MIC_INDEX = _cfg.mic_device_index
STT_BACKEND = _cfg.stt_backend
OPENAI_MODEL = _cfg.stt_openai_model
SAMPLE_RATE = 16000  # Whisper expects 16kHz mono

# Reuse one OpenAI client across calls — constructing a new client (and
# connection pool) per transcription added latency to every interaction.
_openai_client = None


def _get_openai_client():
    global _openai_client
    if _openai_client is None:
        from openai import OpenAI
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return None
        _openai_client = OpenAI(api_key=api_key)
    return _openai_client


def _record_audio(timeout: int = 8, phrase_limit: int = 15) -> np.ndarray | None:
    """
    Record audio from the microphone with voice activity detection.

    Waits up to `timeout` seconds for speech to begin.
    Records until silence is detected or `phrase_limit` seconds elapse.

    Returns:
        numpy array of int16 audio samples, or None if nothing was detected.
    """
    chunk_duration = 0.1   # 100ms chunks
    chunk_size = int(SAMPLE_RATE * chunk_duration)
    silence_threshold = 0.015   # RMS threshold below which audio is "silent"
    silence_after_speech = 1.2  # Stop after 1.2s of silence — long enough not
                                # to clip mid-sentence pauses
    # Require at least this much *voiced* audio, otherwise it's a cough, a room
    # echo of JARVIS's own voice, or a keyboard click — not a command. Sending
    # such clips to Whisper is the main source of hallucinated transcripts
    # ("Thank you.", "For more information visit www...").
    min_voiced_chunks = 4       # 0.4 s of actual speech

    frames = []
    started_speaking = False
    voiced_chunks = 0
    silent_chunks = 0
    elapsed = 0.0
    silent_chunks_needed = int(silence_after_speech / chunk_duration)

    logger.info("Listening for speech...")

    try:
        with sd.InputStream(
            device=MIC_INDEX,
            samplerate=SAMPLE_RATE,
            channels=1,
            dtype="int16",
            blocksize=chunk_size
        ) as stream:
            while elapsed < (timeout + phrase_limit):
                chunk, _ = stream.read(chunk_size)
                audio_data = chunk.flatten().astype(np.float32) / 32768.0
                rms = float(np.sqrt(np.mean(audio_data ** 2)))
                elapsed += chunk_duration

                if rms > silence_threshold:
                    # Speech detected
                    if not started_speaking:
                        logger.info("Speech detected, recording...")
                        started_speaking = True
                    voiced_chunks += 1
                    silent_chunks = 0
                    frames.append(chunk.copy())
                else:
                    if started_speaking:
                        # Accumulate silence frames too (for natural trailing)
                        frames.append(chunk.copy())
                        silent_chunks += 1
                        if silent_chunks >= silent_chunks_needed:
                            logger.info("Silence detected, stopping recording.")
                            break
                    else:
                        # Still waiting for speech to begin
                        if elapsed > timeout:
                            logger.debug("Timeout: no speech detected.")
                            return None

    except Exception as e:
        logger.error(f"Microphone recording failed: {e}")
        return None

    if not frames or voiced_chunks < min_voiced_chunks:
        logger.info(
            f"Discarding clip: only {voiced_chunks} voiced chunks "
            f"(< {min_voiced_chunks}) — treating as noise/echo, not speech."
        )
        return None

    return np.concatenate(frames, axis=0)


def _numpy_to_wav_bytes(audio: np.ndarray) -> bytes:
    """Convert a numpy int16 array to raw WAV bytes in memory."""
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 2 bytes = int16
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio.tobytes())
    buf.seek(0)
    return buf.read()


# Phrases Whisper invents on silence, breath, or echo of JARVIS's own voice —
# they come from its YouTube-caption training data and are never real commands.
_HALLUCINATION_EXACT = {
    "you", "you.", ".", "!", "?", "so", "so.", "um", "uh", "uhh", "hmm", "mm",
}
_HALLUCINATION_SUBSTRINGS = (
    "for more information",
    "visit www",
    "thanks for watching",
    "thank you for watching",
    "please subscribe",
    "like and subscribe",
    "subscribe to",
    "see you in the next video",
    "see you next time",
    "amara.org",
    "subtitles by",
    "subtitled by",
)


def _is_hallucination(text: str) -> bool:
    """True if a transcript is a known Whisper hallucination, not a real command."""
    if not text:
        return False
    norm = text.strip().lower()
    if norm in _HALLUCINATION_EXACT:
        return True
    return any(s in norm for s in _HALLUCINATION_SUBSTRINGS)


def _transcribe_openai(audio: np.ndarray) -> str | None:
    """Send audio to OpenAI Whisper API and return transcript."""
    try:
        client = _get_openai_client()
        if client is None:
            logger.error("OPENAI_API_KEY not set. Add it to your .env file.")
            return None

        wav_bytes = _numpy_to_wav_bytes(audio)

        audio_file = io.BytesIO(wav_bytes)
        audio_file.name = "audio.wav"  # OpenAI needs a filename with extension

        logger.debug(f"Sending {len(wav_bytes) // 1024}KB audio to Whisper API...")
        transcript = client.audio.transcriptions.create(
            model=OPENAI_MODEL,
            file=audio_file,
            language="en",
            # "json" is supported by every STT model (whisper-1 AND the newer
            # gpt-4o-transcribe / gpt-4o-mini-transcribe, which reject "text").
            # The response handling below reads .text off the JSON object.
            response_format="json",
            # temperature=0 makes transcription deterministic and far less prone
            # to inventing text on unclear audio (a documented cause of the
            # "Thank you." / "visit www..." hallucinations).
            temperature=0,
            # A SHORT, focused prompt biases spelling of hard words (Jarvis,
            # RAG, notes.txt) without the long-prompt "keep talking" effect
            # that itself induces hallucinations.
            prompt=(
                "Voice command to JARVIS, an AI assistant. Indian English. "
                "Terms: JARVIS, RAG, MCP, notes.txt, YouTube, Nifty, Sensex, "
                "Bikaner, Jaipur, Delhi, TCS, Reliance, Infosys."
            )
        )
        text = transcript.strip() if isinstance(transcript, str) else transcript.text.strip()

        if _is_hallucination(text):
            logger.info(f"Ignoring likely Whisper hallucination: '{text}'")
            return ""

        logger.info(f"Heard: '{text}'")
        if text:
            return text.lower()
        # Whisper returned an empty transcript: audio was detected but was
        # noise, a cough, or a very short sound.  Return "" (not None) so
        # the caller can distinguish "noise heard" from "silence timeout".
        return ""

    except Exception as e:
        logger.error(f"OpenAI Whisper transcription failed: {e}")
        return None


def listen(timeout: int = 8, phrase_limit: int = 15, **kwargs) -> str | None:
    """
    Record audio from the microphone and transcribe it.

    Args:
        timeout: Seconds to wait for speech to begin
        phrase_limit: Maximum seconds of speech to capture

    Returns:
        Lowercase transcript string, or None on failure/silence
    """
    audio = _record_audio(timeout=timeout, phrase_limit=phrase_limit)
    if audio is None:
        return None

    if STT_BACKEND == "openai":
        return _transcribe_openai(audio)

    # Fallback: log warning and return None
    logger.warning(f"Unknown STT backend '{STT_BACKEND}'. Returning None.")
    return None
