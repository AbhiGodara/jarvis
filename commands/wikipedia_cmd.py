import logging
import re
import wikipedia
from commands.registry import command

logger = logging.getLogger(__name__)

# Queries containing these words are time-sensitive — Wikipedia's static content
# will give wrong/outdated answers. Let the LLM handle them instead.
_LLM_ONLY_PHRASES = [
    "current", "latest", "now", "today", "right now",
    "presently", "at the moment", "recently", "new",
]

# Phrases to strip from the search query. Word-boundary regex: plain
# substring replace mangled words ("wikileaks" → "leaks").
_STRIP_RE = re.compile(r"\b(wikipedia|wiki|who is|tell me about|what is|search for)\b")


def _should_use_llm(text: str) -> bool:
    """Return True if this query should bypass Wikipedia and go to LLM."""
    lower = text.lower()
    return any(phrase in lower for phrase in _LLM_ONLY_PHRASES)


# Examples are deliberately wikipedia-flavored lookups, NOT generic "who is X"
# phrasings — plain knowledge questions ("who is virat kohli") must stay on
# the direct-LLM tier, and generic examples here would semantically veto them.
@command(
    keywords=["wikipedia", "wiki", "who is", "tell me about"],
    description="Search Wikipedia for information about a topic or person",
    examples=[
        "look up nikola tesla on wikipedia",
        "search wikipedia for the eiffel tower",
        "give me the wikipedia summary of the taj mahal",
        "wiki the french revolution",
        "find the wikipedia page about black holes",
    ],
)
def search_wikipedia(text: str) -> str:
    """Search Wikipedia for a topic and return a brief spoken summary."""
    # Let LLM handle time-sensitive questions ("current PM", "latest news", etc.)
    if _should_use_llm(text):
        logger.info(f"Wikipedia: deferring time-sensitive query to LLM: '{text}'")
        return None  # None tells the registry to fall through to LLM

    # Strip trigger phrases from search query (whole words only)
    query = _STRIP_RE.sub(" ", text)

    # Clean up extra whitespace and punctuation
    query = " ".join(query.split()).strip("?!.,;: ")

    if not query:
        return "What topic should I look up on Wikipedia?"

    logger.info(f"Wikipedia search: '{query}'")

    try:
        summary = wikipedia.summary(query, sentences=2, auto_suggest=True)
        return summary

    except wikipedia.DisambiguationError as e:
        logger.debug(f"Disambiguation for '{query}': {e.options[:3]}")
        try:
            return wikipedia.summary(e.options[0], sentences=2)
        except Exception:
            options = ", ".join(e.options[:3])
            return f"There are multiple results for that. Did you mean: {options}?"

    except wikipedia.PageError:
        # Don't return an error — return None so LLM gets a chance
        logger.info(f"No Wikipedia page for '{query}'. Deferring to LLM.")
        return None

    except Exception as e:
        logger.error(f"Wikipedia lookup failed: {e}")
        return None  # Let LLM handle it
