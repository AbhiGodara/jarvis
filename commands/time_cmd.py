import re
from datetime import datetime
from commands.registry import command


@command(keywords=["time", "what time", "date", "what day", "what's today", "today's date"])
def get_time_date(text: str) -> str | None:
    """Return the current time or date depending on what the user asked."""
    # "time in New York" / "timezone" questions need real timezone math —
    # decline so the planner/LLM handles them instead of answering local time.
    if (
        re.search(r"\btime (is it )?(in|at|for)\b", text)
        or "timezone" in text
        or "time zone" in text
    ):
        return None

    now = datetime.now()

    if any(word in text for word in ["date", "day", "today"]):
        formatted = now.strftime("%A, %B %d, %Y")
        return f"Today is {formatted}."

    formatted = now.strftime("%I:%M %p")
    return f"The current time is {formatted}."
