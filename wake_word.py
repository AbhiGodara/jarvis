import logging
import numpy as np
import sounddevice as sd
from openwakeword.model import Model

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
CHUNK_DURATION = 0.08
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_DURATION)
ACTIVATION_THRESHOLD = 0.45

import yaml
try:
    with open("config.yaml", "r") as f:
        _config = yaml.safe_load(f)
    MIC_INDEX = _config.get("mic_device_index", None)
except Exception:
    MIC_INDEX = None

import queue

def wait_for_wake_word(model_name: str = "hey_jarvis") -> None:
    """
    Block execution until the specified wake word is detected via microphone.

    Runs openwakeword inference on 80ms audio chunks. Uses the ONNX runtime
    for fast, local, offline detection. Returns only when the wake word fires.

    Args:
        model_name: openwakeword model name. The model is auto-downloaded on first use.
    """
    oww = Model(wakeword_models=[model_name], inference_framework="onnx")
    logger.info(f"Idle. Waiting for wake word: '{model_name}'...")

    q = queue.Queue()
    
    def audio_callback(indata, frames, time, status):
        """This is called for each audio block by sounddevice."""
        if status:
            pass  # Ignore status warnings like underflow for now
        q.put(indata.copy())

    with sd.InputStream(
        device=MIC_INDEX,
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="int16",
        blocksize=CHUNK_SIZE,
        callback=audio_callback
    ):
        frames = 0
        while True:
            audio_chunk = q.get()
            audio_float = audio_chunk.flatten().astype(np.float32) / 32768.0
            
            frames += 1
            if frames % 50 == 0:  # Every ~4 seconds
                max_vol = np.max(np.abs(audio_float))
                if max_vol < 0.005:
                    logger.warning("Microphone volume is extremely low. Is your mic muted or disconnected?")
                
            predictions = oww.predict(audio_float)
            score = predictions.get(model_name, 0)

            if score >= ACTIVATION_THRESHOLD:
                logger.info(f"Wake word detected. Confidence: {score:.2f}")
                return
