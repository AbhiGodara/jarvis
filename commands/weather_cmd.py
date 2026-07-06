import re
import logging
import requests
from commands.registry import command
from core.config import get_config

logger = logging.getLogger(__name__)
cfg = get_config()

_API_KEY = cfg.openweather_api_key
_DEFAULT_CITY = cfg.weather_default_city
_UNITS = cfg.weather_units
_UNIT_SYMBOL = "°C" if _UNITS == "metric" else "°F"
_BASE_URL = "https://api.openweathermap.org/data/2.5/weather"

# Ordered by specificity — longest triggers first so they match before shorter ones
# ("weather of Bikaner" silently fell back to the default city before the
# of/at variants were added — seen in runtime logs.)
_CITY_TRIGGERS = [
    "temperature in",
    "temperature of",
    "temperature at",
    "weather in",
    "weather of",
    "weather at",
    "forecast for",
    "forecast in",
    "forecast of",
    "how hot in",
    "how cold in",
    "raining in",
    "weather for",
]


def _clean_city(city: str) -> str:
    """Strip trailing punctuation and filler words from an extracted city."""
    city = re.sub(r"[?!.,;:]+$", "", city).strip()
    city = re.sub(
        r"\b(please|now|today|tonight|tomorrow|currently|right now|outside|at the moment)\b",
        "", city, flags=re.IGNORECASE,
    ).strip()
    return re.sub(r"^the\s+", "", city).strip()


def _extract_city(text: str) -> str:
    """
    Extract a city name from the command text.

    Tries longest triggers first to avoid over-greedy matches.
    Strips trailing punctuation and filler words.
    Falls back to the default city from config.
    """
    for trigger in _CITY_TRIGGERS:
        if trigger in text:
            city = _clean_city(text.split(trigger, 1)[-1])
            if city:
                logger.debug(f"Extracted city: '{city}' (trigger: '{trigger}')")
                return city

    # Generic fallback: "how hot is it in jaipur", "what's the weather like in
    # mumbai today" — phrasings where no "<keyword> in" trigger is adjacent.
    # Take whatever follows the last in/at/for as the city candidate.
    m = re.search(r"\b(?:in|at|for)\s+([a-z][a-z .'-]*)$", text.strip())
    if m:
        city = _clean_city(m.group(1))
        if city:
            logger.debug(f"Extracted city (fallback): '{city}'")
            return city

    return _DEFAULT_CITY


# priority=5: 'weather'/'temperature' are specific signals — must beat
# generic keyword owners like math's "what is ".
@command(
    keywords=["weather", "temperature", "forecast", "how hot", "how cold", "is it raining", "is it sunny"],
    description="Get current weather conditions for a city",
    priority=5,
    examples=[
        "what's it like outside",
        "do i need an umbrella today",
        "is it going to rain today",
        "how hot is it in delhi",
        "will i need a jacket this evening",
        "what's the temperature right now",
    ],
)
def get_weather(text: str) -> str:
    """Fetch current weather conditions for a city from OpenWeatherMap."""
    if not _API_KEY:
        return (
            "The weather feature is not set up. "
            "Please add your OPENWEATHER_API_KEY to the .env file."
        )

    city = _extract_city(text)
    logger.info(f"Weather lookup for city: '{city}'")

    try:
        response = requests.get(
            _BASE_URL,
            params={"q": city, "appid": _API_KEY, "units": _UNITS},
            timeout=5
        )
        response.raise_for_status()
        data = response.json()

        description = data["weather"][0]["description"]
        temp = round(data["main"]["temp"])
        feels_like = round(data["main"]["feels_like"])
        humidity = data["main"]["humidity"]
        city_name = data["name"]
        country = data["sys"].get("country", "")

        return (
            f"In {city_name}, {country}, it's currently {description} "
            f"with a temperature of {temp}{_UNIT_SYMBOL}, "
            f"feeling like {feels_like}{_UNIT_SYMBOL}, "
            f"and humidity at {humidity}%."
        )

    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            return (
                f"I couldn't find weather data for {city}. "
                f"Please check the city name and try again."
            )
        elif e.response.status_code == 401:
            return "The weather API key is invalid. Please check your .env file."
        else:
            logger.error(f"Weather API HTTP error: {e}")
            return "The weather service returned an error. Please try again."

    except requests.exceptions.ConnectionError:
        return "I can't reach the weather service. Check your internet connection."

    except requests.exceptions.Timeout:
        return "The weather request timed out. Please try again."

    except Exception as e:
        logger.error(f"Weather lookup failed: {e}")
        return "Something went wrong with the weather lookup."
