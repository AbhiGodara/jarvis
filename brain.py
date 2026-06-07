import logging
import yaml
from commands.registry import match
from llm import LLMClient

# Import all command modules so their @command decorators register them
import commands.time_cmd        # noqa: F401
import commands.youtube_cmd     # noqa: F401
import commands.wikipedia_cmd   # noqa: F401
import commands.weather_cmd     # noqa: F401
import commands.reminder_cmd    # noqa: F401
import commands.math_cmd        # noqa: F401
import commands.news_cmd        # noqa: F401
import commands.notes_cmd       # noqa: F401
import commands.apps_cmd        # noqa: F401
import commands.stocks_cmd      # noqa: F401
import commands.translate_cmd   # noqa: F401

logger = logging.getLogger(__name__)

with open("config.yaml", "r") as f:
    _config = yaml.safe_load(f)

_MAX_INPUT = _config.get("llm_max_input_chars", 500)


class Brain:
    """
    The central command router for JARVIS.

    Routes each transcribed command to either a local command handler
    (from the commands/ plugin folder) or the LLM fallback.
    """

    def __init__(self, llm_client: LLMClient):
        """
        Args:
            llm_client: An initialised LLMClient instance for conversational fallback
        """
        self.llm = llm_client
        logger.info("Brain initialised with all command plugins loaded.")

    def process(self, command: str) -> str:
        """
        Process a voice command and return a spoken response.

        Args:
            command: Lowercased, stripped transcribed text from the user

        Returns:
            A response string suitable for TTS playback
        """
        if not command or not command.strip():
            return "I'm listening. What can I do for you?"

        # Guard against unexpectedly long inputs before LLM call
        if len(command) > _MAX_INPUT:
            logger.warning(f"Command truncated from {len(command)} to {_MAX_INPUT} characters.")
            command = command[:_MAX_INPUT]

        # Try local commands first (fast, no API cost)
        local_response = match(command)
        if local_response is not None:
            logger.info("Handled by local command plugin.")
            return local_response

        # Fall through to LLM
        logger.info("No local match. Routing to LLM.")
        return self.llm.ask(command)
