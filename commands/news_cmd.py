import os
import re
import logging
import requests
import yaml
from dotenv import load_dotenv
from commands.registry import command

load_dotenv()
logger = logging.getLogger(__name__)

with open("config.yaml", "r") as f:
    _config = yaml.safe_load(f)

_API_KEY = os.getenv("NEWS_API_KEY")
_COUNTRY = _config.get("news_country", "us")
_COUNT = min(int(_config.get("news_headline_count", 3)), 5)
_ORDINALS = ["One", "Two", "Three", "Four", "Five"]


@command(keywords=["news", "headlines", "what's happening", "what happened today", "top stories"])
def get_news(text: str) -> str:
    """Fetch and read the top news headlines aloud."""
    if not _API_KEY:
        return "The news feature is not configured. Please add your NewsAPI key to the .env file."

    try:
        response = requests.get(
            "https://newsapi.org/v2/top-headlines",
            params={"country": _COUNTRY, "apiKey": _API_KEY, "pageSize": _COUNT},
            timeout=5
        )
        response.raise_for_status()
        articles = response.json().get("articles", [])

        if not articles:
            return "I couldn't find any headlines right now."

        headlines = []
        for article in articles[:_COUNT]:
            title = article.get("title", "")
            # Remove source attribution like "- BBC News" from the end
            title = re.sub(r"\s*-\s*[^-]+$", "", title).strip()
            if title:
                headlines.append(title)

        if not headlines:
            return "I found articles but couldn't extract any titles."

        intro = "Here are today's top headlines."
        items = ". ".join(
            f"{_ORDINALS[i]}: {h}" for i, h in enumerate(headlines)
        )
        return f"{intro} {items}."

    except requests.exceptions.ConnectionError:
        return "I can't reach the news service. Check your internet connection."
    except requests.exceptions.Timeout:
        return "The news request timed out."
    except Exception as e:
        logger.error(f"News fetch failed: {e}")
        return "Something went wrong fetching the news."
