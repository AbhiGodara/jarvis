import os
import logging
import requests
import yaml
from dotenv import load_dotenv
from commands.registry import command

load_dotenv()
logger = logging.getLogger(__name__)

with open("config.yaml", "r") as f:
    _config = yaml.safe_load(f)

_API_KEY = os.getenv("OPENWEATHER_API_KEY")
_DEFAULT_CITY = _config.get("weather_default_city", "London")
_UNITS = _config.get("weather_units", "metric")
_UNIT_SYMBOL = "°C" if _UNITS == "metric" else "°F"
_BASE_URL = "https://api.openweathermap.org/data/2.5/weather"


def _extract_city(text: str) -> str:
    """Try to extract a city name from the command text. Falls back to default."""
    for trigger in ["weather in", "temperature in", "forecast for", "weather"]:
        if trigger in text:
            city = text.split(trigger, 1)[-1].strip()
            if city:
                return city
    return _DEFAULT_CITY


@command(keywords=["weather", "temperature", "forecast", "how hot", "how cold", "is it raining", "is it sunny"])
def get_weather(text: str) -> str:
    """Fetch current weather conditions for a city from OpenWeatherMap."""
    if not _API_KEY:
        return "The weather feature is not set up. Please add your OpenWeatherMap API key to the .env file."

    city = _extract_city(text)

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
        city_name = data["name"]

        return (
            f"In {city_name}, it's currently {description} "
            f"with a temperature of {temp}{_UNIT_SYMBOL}, "
            f"feeling like {feels_like}{_UNIT_SYMBOL}."
        )

    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            return f"I couldn't find weather data for {city}. Check the city name."
        elif e.response.status_code == 401:
            return "The weather API key is invalid. Please check your .env file."
        else:
            logger.error(f"Weather API HTTP error: {e}")
            return "The weather service returned an error. Please try again."

    except requests.exceptions.ConnectionError:
        return "I can't reach the weather service right now. Check your internet connection."

    except requests.exceptions.Timeout:
        return "The weather request timed out. Please try again."

    except Exception as e:
        logger.error(f"Weather lookup failed unexpectedly: {e}")
        return "Something went wrong with the weather lookup."
