import logging
import wikipedia
from commands.registry import command

logger = logging.getLogger(__name__)


@command(keywords=["wikipedia", "wiki", "who is", "tell me about"])
def search_wikipedia(text: str) -> str:
    """Fetch a brief Wikipedia summary and return it as a spoken response."""
    for trigger in ["wikipedia", "wiki", "who is", "tell me about"]:
        text = text.replace(trigger, "").strip()

    if not text:
        return "What topic should I look up?"

    try:
        summary = wikipedia.summary(text, sentences=2)
        return summary

    except wikipedia.DisambiguationError as e:
        logger.debug(f"Disambiguation for '{text}': {e.options[:3]}")
        try:
            return wikipedia.summary(e.options[0], sentences=2)
        except Exception:
            options = ", ".join(e.options[:3])
            return f"There are multiple results. Did you mean {options}?"

    except wikipedia.PageError:
        return f"I couldn't find a Wikipedia page for {text}."

    except Exception as e:
        logger.error(f"Wikipedia lookup failed: {e}")
        return "The Wikipedia lookup failed. Please try again."
