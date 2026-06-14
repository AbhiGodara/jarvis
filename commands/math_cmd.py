import re
import logging
from simpleeval import simple_eval, InvalidExpression
from commands.registry import command

logger = logging.getLogger(__name__)

_UNIT_CONVERSIONS = [
    (r"(\d+\.?\d*)\s*km to miles",   lambda n: (n * 0.621371, f"{n} km is {round(n * 0.621371, 2)} miles")),
    (r"(\d+\.?\d*)\s*miles to km",   lambda n: (n * 1.60934,  f"{n} miles is {round(n * 1.60934, 2)} km")),
    (r"(\d+\.?\d*)\s*kg to pounds",  lambda n: (n * 2.20462,  f"{n} kg is {round(n * 2.20462, 2)} pounds")),
    (r"(\d+\.?\d*)\s*pounds to kg",  lambda n: (n * 0.453592, f"{n} pounds is {round(n * 0.453592, 2)} kg")),
    (r"(\d+\.?\d*)\s*(celsius|c) to (fahrenheit|f)", lambda n: (None, f"{n}°C is {round((n * 9/5) + 32, 1)}°F")),
    (r"(\d+\.?\d*)\s*(fahrenheit|f) to (celsius|c)", lambda n: (None, f"{n}°F is {round((n - 32) * 5/9, 1)}°C")),
]


def _try_unit_conversion(text: str) -> str | None:
    """Try to match unit conversion patterns. Returns result string or None."""
    for pattern, formatter in _UNIT_CONVERSIONS:
        match = re.search(pattern, text)
        if match:
            value = float(match.group(1))
            _, result_str = formatter(value)
            return result_str
    return None


def _try_percentage(text: str) -> str | None:
    """Handle 'X percent of Y' patterns."""
    match = re.search(r"(\d+\.?\d*)\s*percent\s*of\s*(\d+\.?\d*)", text)
    if match:
        pct = float(match.group(1))
        total = float(match.group(2))
        result = round(pct / 100 * total, 2)
        return f"{pct}% of {total} is {result}."
    return None


# Phrases that look like math triggers but are actually knowledge/weather queries
_NON_MATH_PHRASES = [
    "temperature", "weather", "forecast", "who is", "what is the pm",
    "what is the president", "what is the capital", "who was", "who are",
    "current", "prime minister", "minister", "president", "governor",
]


@command(
    keywords=["calculate", "what is ", "how much is", "convert ", "times", "divided by", "percent of"],
    description="Evaluate math expressions, unit conversions, and calculations"
)
def calculate(text: str) -> str:
    """Evaluate maths expressions and perform unit conversions."""
    # Bail out immediately for knowledge/weather queries — let LLM or weather handle them
    lower = text.lower()
    if any(phrase in lower for phrase in _NON_MATH_PHRASES):
        return None

    # Try unit conversion first
    conversion = _try_unit_conversion(text)
    if conversion:
        return conversion + "."

    # Try percentage
    percentage = _try_percentage(text)
    if percentage:
        return percentage

    # Try maths expression
    for trigger in ["calculate", "what is", "how much is", "compute"]:
        text = text.replace(trigger, "").strip()

    # Translate spoken words to operators
    replacements = {
        " plus ": " + ", " minus ": " - ", " times ": " * ",
        " multiplied by ": " * ", " divided by ": " / ",
        " squared": " ** 2", " to the power of ": " ** ",
    }
    for spoken, symbol in replacements.items():
        text = text.replace(spoken, symbol)

    try:
        result = simple_eval(text)
        if isinstance(result, float) and result == int(result):
            result = int(result)
        return f"The answer is {result}."
    except (InvalidExpression, Exception) as e:
        logger.debug(f"Math eval failed for '{text}': {e}")
        return None

