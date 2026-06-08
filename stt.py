"""
Speech-to-Text module for JARVIS.

Primary backend: OpenAI Whisper API (whisper-1)
  - No local model download
  - Accurate across accents (including Indian English)
  - ~300-500ms transcription latency
  - Falls back to pyttsx3-style silence if API is unavailable

Fallback: pyttsx3 offline voice recognition is not available, so
          on API failure we return None and main.py handles the retry.
"""
import io
import wave
import logging
import numpy as np
import sounddevice as sd
import yaml
import os
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# Load config
try:
    with open("config.yaml", "r") as f:
        _config = yaml.safe_load(f)
except Exception:
    _config = {}

MIC_INDEX = _config.get("mic_device_index", None)
STT_BACKEND = _config.get("stt_backend", "openai")
OPENAI_MODEL = _config.get("stt_openai_model", "whisper-1")
SAMPLE_RATE = 16000  # Whisper expects 16kHz mono


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
    silence_threshold = 0.01    # RMS threshold below which audio is "silent"
    silence_after_speech = 1.5  # Stop after 1.5s of silence following speech
    
    frames = []
    started_speaking = False
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

    if not frames:
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


def _transcribe_openai(audio: np.ndarray) -> str | None:
    """Send audio to OpenAI Whisper API and return transcript."""
    try:
        from openai import OpenAI
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            logger.error("OPENAI_API_KEY not set. Add it to your .env file.")
            return None

        client = OpenAI(api_key=api_key)
        wav_bytes = _numpy_to_wav_bytes(audio)

        audio_file = io.BytesIO(wav_bytes)
        audio_file.name = "audio.wav"  # OpenAI needs a filename with extension

        logger.debug(f"Sending {len(wav_bytes) // 1024}KB audio to Whisper API...")
        transcript = client.audio.transcriptions.create(
            model=OPENAI_MODEL,
            file=audio_file,
            language="en",
            response_format="text"
        )
        text = transcript.strip() if isinstance(transcript, str) else transcript.text.strip()
        logger.info(f"Heard: '{text}'")
        return text.lower() if text else None

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
