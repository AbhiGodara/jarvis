"""
Wake word detection module for JARVIS.

Uses OpenWakeWord to detect "Hey JARVIS" from the microphone.

IMPORTANT CHANGE: The model is now loaded ONCE at startup via `load_model()`
and passed to `wait_for_wake_word()` on every call.  This eliminates the
"Loading wake word model..." log after every command activation.
"""
import logging
import numpy as np
import sounddevice as sd
import queue

from core.config import get_config

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
CHUNK_DURATION = 0.08
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION)
ACTIVATION_THRESHOLD = 0.20

MIC_INDEX = get_config().mic_device_index


def load_model(model_name: str = "hey_jarvis"):
    """
    Load the OpenWakeWord model once at startup.

    Call this from main.py at program start.  Pass the returned model
    object to every call of wait_for_wake_word().

    Returns:
        Loaded openwakeword.model.Model instance
    """
    from openwakeword.model import Model
    logger.info(f"Loading wake word model: {model_name} (once at startup)")
    oww = Model(
        wakeword_models=[model_name],
        inference_framework="onnx"
    )
    logger.info("Wake word model loaded successfully.")
    return oww


def wait_for_wake_word(oww=None, model_name: str = "hey_jarvis") -> None:
    """
    Block until the wake word is detected in the microphone stream.

    Args:
        oww: Pre-loaded OpenWakeWord model (from load_model()). If None,
             the model is loaded on the spot (slower, backward-compatible).
        model_name: The wake word name to detect.
    """
    if oww is None:
        oww = load_model(model_name)

    # Clear the model's internal audio buffer from the previous activation —
    # stale frames can re-trigger the wake word immediately.
    try:
        oww.reset()
    except Exception:
        pass

    q: queue.Queue = queue.Queue()

    def audio_callback(indata, frames, time_info, status):
        if status:
            logger.warning(f"Microphone status: {status}")
        q.put(indata.copy())

    logger.info(f"Idle — waiting for wake word '{model_name}'...")

    with sd.InputStream(
        device=MIC_INDEX,
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="int16",
        blocksize=CHUNK_SIZE,
        callback=audio_callback
    ):
        while True:
            audio_chunk = q.get()
            audio_data = audio_chunk.flatten()

            predictions = oww.predict(audio_data)
            score = predictions.get(model_name, 0.0)

            if score >= ACTIVATION_THRESHOLD:
                logger.info(f"Wake word detected! Confidence={score:.3f}")
                return