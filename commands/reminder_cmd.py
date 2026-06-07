import re
import atexit
import logging
import threading
from commands.registry import command

logger = logging.getLogger(__name__)

_active_timers: list[threading.Timer] = []


def _cancel_all_timers():
    """Cancel all pending reminder timers. Called at shutdown."""
    for timer in _active_timers:
        timer.cancel()
    logger.info(f"Cancelled {len(_active_timers)} pending reminder(s).")


atexit.register(_cancel_all_timers)


def _parse_duration(text: str) -> int | None:
    """
    Extract a duration in seconds from text like 'in 5 minutes' or 'in 2 hours'.

    Returns seconds as an integer, or None if no duration was found.
    """
    patterns = [
        (r"in (\d+) second", 1),
        (r"in (\d+) minute", 60),
        (r"in (\d+) hour", 3600),
        (r"in a minute", None),       # Special case: 1 minute
        (r"in an hour", None),         # Special case: 1 hour
    ]
    if "in a minute" in text:
        return 60
    if "in an hour" in text:
        return 3600

    for pattern, multiplier in patterns:
        if multiplier is None:
            continue
        match = re.search(pattern, text)
        if match:
            return int(match.group(1)) * multiplier

    return None


def _parse_message(text: str) -> str:
    """Extract the reminder message from text like 'remind me in 5 minutes to check the oven'."""
    for marker in [" to ", " about "]:
        if marker in text:
            return text.split(marker, 1)[-1].strip()
    return "your reminder"


def _fire_reminder(message: str):
    """Called by the timer thread when a reminder is due. Speaks the reminder aloud."""
    from tts import speak  # Import here to avoid circular imports
    logger.info(f"Reminder fired: '{message}'")
    speak(f"Reminder: {message}")


@command(keywords=["remind me", "set a reminder", "reminder"])
def set_reminder(text: str) -> str:
    """Parse a reminder command and schedule a voice alert using a background timer."""
    duration = _parse_duration(text)
    if duration is None:
        return "I couldn't work out the time for that reminder. Try saying 'remind me in 5 minutes to check the oven'."

    message = _parse_message(text)

    timer = threading.Timer(duration, _fire_reminder, args=[message])
    timer.daemon = True
    timer.start()
    _active_timers.append(timer)

    minutes = duration // 60
    seconds = duration % 60
    if minutes > 0:
        time_str = f"{minutes} minute{'s' if minutes != 1 else ''}"
    else:
        time_str = f"{seconds} second{'s' if seconds != 1 else ''}"

    return f"Done. I'll remind you to {message} in {time_str}."
