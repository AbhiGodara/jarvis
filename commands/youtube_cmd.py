import webbrowser
import urllib.parse
from commands.registry import command


@command(keywords=["play ", "youtube", "put on", "i want to listen to"])
def play_youtube(text: str) -> str:
    """Open YouTube in the browser with a search query extracted from the user's command."""
    # Remove trigger phrases to isolate the search query
    triggers = ["i want to listen to", "put on", "play", "on youtube", "youtube"]
    for trigger in triggers:
        text = text.replace(trigger, " ")
    
    text = " ".join(text.split())

    if not text:
        return "What would you like me to play?"

    query = urllib.parse.quote_plus(text)
    url = f"https://www.youtube.com/results?search_query={query}"
    webbrowser.open(url)
    return f"Opening YouTube for {text}."
