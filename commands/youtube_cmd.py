import re
import webbrowser
import urllib.parse
from commands.registry import command

# Word-boundary stripping: plain str.replace mangled queries
# ("play coldplay" → "cold") because 'play' matched inside words.
# Longest alternatives first so "search for" wins over "search".
_TRIGGER_RE = re.compile(
    r"\b(i want to listen to|search for|on youtube|put on|go to|open|play|search|youtube)\b"
)
_LEADING_FILLER_RE = re.compile(r"^(and|for|me|some)\s+")


def _extract_query(text: str) -> str:
    """Remove trigger phrases (whole words only) to isolate the search query."""
    query = " ".join(_TRIGGER_RE.sub(" ", text).split())
    query = _LEADING_FILLER_RE.sub("", query)
    return query.strip(" .,!?'\"")


# priority=5: "open youtube and play believer" must come here, not to
# open_app — 'youtube'/'play' are more specific than the generic 'open '.
@command(keywords=["play ", "youtube", "put on", "i want to listen to"], priority=5)
def play_youtube(text: str) -> str:
    """Open or search YouTube based on the user's command."""
    query_text = _extract_query(text)

    if not query_text:
        # "open youtube" with nothing to search → just open the site
        webbrowser.open("https://www.youtube.com")
        return "Opening YouTube."

    query = urllib.parse.quote_plus(query_text)
    url = f"https://www.youtube.com/results?search_query={query}"
    webbrowser.open(url)
    return f"Opening YouTube for {query_text}."
