import speech_recognition as sr
import logging
import yaml

try:
    with open("config.yaml", "r") as f:
        _config = yaml.safe_load(f)
    MIC_INDEX = _config.get("mic_device_index", None)
except Exception:
    MIC_INDEX = None

logger = logging.getLogger(__name__)

recognizer = sr.Recognizer()
recognizer.pause_threshold = 1.0
recognizer.dynamic_energy_threshold = True


def listen(language: str = "en-US", timeout: int = 8, phrase_limit: int = 15) -> str | None:
    """
    Listen from the default microphone and return transcribed text.

    Args:
        language: BCP-47 language code for recognition (e.g. "en-US", "hi-IN")
        timeout: Seconds to wait for speech to begin before giving up
        phrase_limit: Maximum seconds of audio to capture

    Returns:
        Lowercase transcribed string, or None if nothing was understood or an error occurred
    """
    with sr.Microphone(device_index=MIC_INDEX) as source:
        logger.debug("Adjusting for ambient noise...")
        recognizer.adjust_for_ambient_noise(source, duration=0.5)
        logger.info("Listening...")

        try:
            audio = recognizer.listen(source, timeout=timeout, phrase_time_limit=phrase_limit)
        except sr.WaitTimeoutError:
            logger.debug("No speech detected within timeout window.")
            return None

    try:
        text = recognizer.recognize_google(audio, language=language)
        logger.info(f"Heard: '{text}'")
        return text.lower().strip()
    except sr.UnknownValueError:
        logger.debug("Audio captured but could not be understood.")
        return None
    except sr.RequestError as e:
        logger.error(f"Google STT service error: {e}")
        return None
