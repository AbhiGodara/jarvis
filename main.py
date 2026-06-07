import atexit
import logging
import yaml
from dotenv import load_dotenv

from wake_word import wait_for_wake_word
from stt import listen
import threading
from tts import speak, stop as stop_tts
from brain import Brain
from llm import LLMClient

load_dotenv()

# ── Load configuration ────────────────────────────────────────────────────────
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

# ── Configure logging ─────────────────────────────────────────────────────────
log_level = getattr(logging, config.get("log_level", "INFO").upper(), logging.INFO)
handlers = [logging.StreamHandler()]

if config.get("log_to_file"):
    handlers.append(logging.FileHandler(config.get("log_file_path", "jarvis.log")))

logging.basicConfig(
    level=log_level,
    format="%(asctime)s  %(levelname)-8s  %(name)-20s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=handlers,
)

# Mute noisy third-party loggers
for noisy_logger in ["google", "urllib3", "matplotlib", "openwakeword", "speech_recognition"]:
    logging.getLogger(noisy_logger).setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
STOP_KEYWORDS = {"stop", "exit", "quit", "goodbye", "shut down", "power off", "bye"}
WAKE_WORD = config.get("wake_word", "hey_jarvis")
TTS_ENGINE = config.get("tts_engine", "gtts")
TTS_ACCENT = config.get("tts_accent", "co.uk")
STT_LANGUAGE = config.get("stt_language", "en-US")
STT_TIMEOUT = int(config.get("stt_timeout", 8))
STT_PHRASE_LIMIT = int(config.get("stt_phrase_limit", 15))


def _speak(text: str) -> None:
    """Convenience wrapper that passes engine and accent from config."""
    speak(text, accent=TTS_ACCENT, engine=TTS_ENGINE)


def _speak_async(text: str) -> None:
    """Speak asynchronously so the main loop can listen for interruptions."""
    threading.Thread(target=_speak, args=(text,), daemon=True).start()


def _strip_wake_word(text: str) -> str:
    """
    Remove accidental wake word transcription from the start of a command.

    STT sometimes captures the wake word itself. This cleans it out so
    commands like 'hey jarvis play music' become 'play music'.
    """
    for prefix in ["hey jarvis", "jarvis"]:
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
    return text


def _on_shutdown() -> None:
    """Called by atexit when the program exits. Speaks a farewell message."""
    logger.info("JARVIS shutting down.")
    _speak("Going offline. Goodbye.")


atexit.register(_on_shutdown)


def main() -> None:
    """Main entry point. Initialises JARVIS and runs the voice assistant loop."""
    logger.info("=" * 60)
    logger.info("  JARVIS initialising...")
    logger.info("=" * 60)

    # Initialise core modules
    llm = LLMClient(
        model_name=config.get("llm_model", "gemini-2.5-flash"),
        max_history_turns=int(config.get("llm_max_history_turns", 20)),
    )
    brain = Brain(llm_client=llm)

    logger.info("JARVIS is ready.")
    _speak("JARVIS online. Ready when you are.")

    # ── Main loop ─────────────────────────────────────────────────────────────
    while True:
        try:
            # Phase 1: Wait for wake word
            wait_for_wake_word(model_name=WAKE_WORD)
            
            # Interruption: Stop any ongoing TTS immediately
            stop_tts()

            # Phase 2: Audio acknowledgment
            _speak("Yes?")

            # Phase 3: Listen for command
            command = listen(
                language=STT_LANGUAGE,
                timeout=STT_TIMEOUT,
                phrase_limit=STT_PHRASE_LIMIT,
            )

            if not command:
                _speak("I didn't catch that.")
                continue

            command = _strip_wake_word(command)
            logger.info(f"Command: '{command}'")

            if not command:
                continue

            # Phase 4: Check for stop keywords
            if any(kw in command for kw in STOP_KEYWORDS):
                logger.info("Stop keyword detected. Exiting loop.")
                break

            # Phase 5: Process and respond
            response = brain.process(command)
            logger.info(f"Response: '{response[:80]}{'...' if len(response) > 80 else ''}'")
            _speak_async(response)

        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received.")
            break

        except Exception as e:
            logger.exception(f"Unexpected error in main loop: {e}")
            _speak("Something went wrong. I'll keep listening.")
            # Do NOT break — continue the loop to stay alive


if __name__ == "__main__":
    main()
