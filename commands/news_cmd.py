import re
import logging
import requests
import xml.etree.ElementTree as ET

from commands.registry import command
from core.config import get_config

logger = logging.getLogger(__name__)

_cfg = get_config()
_API_KEY = _cfg.news_api_key
_COUNTRY = _cfg.news_country
_COUNT = min(int(_cfg.news_headline_count), 5)
_ORDINALS = ["One", "Two", "Three", "Four", "Five"]

# RSS feeds — no API key required.  Used when NewsAPI is missing/empty.
# Multiple fallbacks per category so one dead URL doesn't break news entirely.
_RSS_FEEDS: dict[str, list[str]] = {
    "default": [
        "https://feeds.feedburner.com/ndtvnews-top-stories",
        "https://timesofindia.indiatimes.com/rssfeeds/1221656.cms",
        "https://feeds.bbci.co.uk/news/world/asia/india/rss.xml",
    ],
    "cricket": [
        "https://feeds.feedburner.com/ndtvnews-sports",
        "https://www.espncricinfo.com/rss/content/story/feeds/6.xml",
    ],
    "sports": [
        "https://feeds.feedburner.com/ndtvnews-sports",
        "https://timesofindia.indiatimes.com/rssfeeds/4719161.cms",
    ],
    "business": [
        "https://feeds.feedburner.com/ndtvnews-business-profit-news",
        "https://economictimes.indiatimes.com/rssfeedstopstories.cms",
    ],
    "tech": [
        "https://feeds.feedburner.com/ndtvnews-tech-gadgets",
        "https://timesofindia.indiatimes.com/rssfeeds/66949542.cms",
    ],
}

_TOPIC_KEYWORDS: dict[str, tuple[str, ...]] = {
    "cricket": ("cricket", "ipl", "t20", "test match", "bcci", "virat", "rohit"),
    "sports":  ("sports", "football", "tennis", "badminton", "hockey", "kabaddi"),
    "business": ("business", "economy", "market", "finance", "gdp", "sensex", "nifty"),
    "tech":    ("tech", "technology", "ai", "smartphone", "startup", "iphone"),
}


def _detect_topic(text: str) -> str:
    lower = text.lower()
    for topic, keywords in _TOPIC_KEYWORDS.items():
        if any(k in lower for k in keywords):
            return topic
    return "default"


def _parse_rss(xml_bytes: bytes, limit: int) -> list[str]:
    """Extract article titles from RSS 2.0 XML bytes."""
    titles: list[str] = []
    try:
        root = ET.fromstring(xml_bytes)
        for item in root.findall(".//item"):
            el = item.find("title")
            if el is not None and el.text:
                title = re.sub(r"\s*[-|]\s*[^-|]+$", "", el.text).strip()
                if title and len(title) > 10:
                    titles.append(title)
                    if len(titles) >= limit:
                        break
    except ET.ParseError as exc:
        logger.debug(f"RSS parse error: {exc}")
    return titles


def _fetch_rss(topic: str, limit: int) -> list[str]:
    """Try each RSS URL for the topic and return the first successful result."""
    urls = list(_RSS_FEEDS.get(topic, []))
    # Always append default feeds as fallback
    if topic != "default":
        urls += _RSS_FEEDS["default"]

    for url in urls:
        try:
            resp = requests.get(
                url, timeout=6,
                headers={"User-Agent": "Mozilla/5.0 JARVIS/1.0"},
            )
            if resp.status_code == 200:
                titles = _parse_rss(resp.content, limit)
                if titles:
                    logger.info(f"RSS news: got {len(titles)} headlines from {url}")
                    return titles
        except Exception as exc:
            logger.debug(f"RSS {url} failed: {exc}")
    return []


@command(keywords=["news", "headlines", "what's happening", "what happened today", "top stories"])
def get_news(text: str) -> str:
    """Fetch and read the top news headlines aloud."""
    topic = _detect_topic(text)
    headlines: list[str] = []

    # 1. Try NewsAPI first (faster, structured data)
    if _API_KEY:
        try:
            resp = requests.get(
                "https://newsapi.org/v2/top-headlines",
                params={"country": _COUNTRY, "apiKey": _API_KEY, "pageSize": _COUNT},
                timeout=5,
            )
            resp.raise_for_status()
            for article in resp.json().get("articles", [])[:_COUNT]:
                title = re.sub(r"\s*-\s*[^-]+$", "", article.get("title", "")).strip()
                if title:
                    headlines.append(title)
        except Exception as exc:
            logger.warning(f"NewsAPI failed: {exc}. Falling back to RSS.")

    # 2. RSS fallback — no API key, works for India
    if not headlines:
        headlines = _fetch_rss(topic, _COUNT)

    if not headlines:
        return "I'm having trouble reaching the news services right now, sir. Check your internet connection."

    topic_phrase = f" on {topic}" if topic != "default" else ""
    intro = f"Here are the latest headlines{topic_phrase}."
    items = ". ".join(f"{_ORDINALS[i]}: {h}" for i, h in enumerate(headlines[:_COUNT]))
    return f"{intro} {items}."
