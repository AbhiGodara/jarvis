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
    Extract a duration in seconds from text like 'in 5 minutes', 'for 10
    minutes', or 'after 2 hours'.

    Returns seconds as an integer, or None if no duration was found.
    """
    if re.search(r"\bin (a|one) minute\b", text):
        return 60
    if re.search(r"\bin (an|one) hour\b", text):
        return 3600
    if re.search(r"\bin half an hour\b", text):
        return 1800

    # "in 5 minutes" / "for 10 minutes" / "after 2 hours" — all common
    # phrasings; the old parser only accepted "in".
    patterns = [
        (r"(?:in|for|after) (\d+) second", 1),
        (r"(?:in|for|after) (\d+) minute", 60),
        (r"(?:in|for|after) (\d+) hour", 3600),
    ]
    for pattern, multiplier in patterns:
        match = re.search(pattern, text)
        if match:
            return int(match.group(1)) * multiplier

    return None


def _parse_message(text: str) -> str | None:
    """Extract the reminder message from text like 'remind me in 5 minutes to check the oven'."""
    for marker in [" to ", " about "]:
        if marker in text:
            message = text.split(marker, 1)[-1].strip()
            # "remind me to call mom in 10 minutes" — the duration trails the
            # message; it belongs to the schedule, not the spoken reminder.
            message = re.sub(
                r"\s*\b(?:in|for|after)\s+(?:\d+|a|an|one|half an)\s*(?:seconds?|minutes?|hours?)\b.*$",
                "", message,
            ).strip(" .,!?")
            if message:
                return message
    return None


def _fire_reminder(message: str):
    """Called by the timer thread when a reminder is due. Speaks the reminder aloud."""
    from tts import speak  # Import here to avoid circular imports
    logger.info(f"Reminder fired: '{message}'")
    speak(f"Reminder: {message}")


@command(
    keywords=["remind me", "set a reminder", "reminder"],
    examples=[
        "remind me in 10 minutes to check the oven",
        "set a reminder for my meeting",
        "ping me in 5 minutes",
        "don't let me forget to call mom",
        "give me a nudge in half an hour",
    ],
)
def set_reminder(text: str) -> str:
    """Parse a reminder command and schedule a voice alert using a background timer."""
    duration = _parse_duration(text)
    if duration is None:
        return "I couldn't work out the time for that reminder. Try saying 'remind me in 5 minutes to check the oven'."

    message = _parse_message(text)

    timer = threading.Timer(duration, _fire_reminder, args=[message or "your reminder"])
    timer.daemon = True
    timer.start()
    _active_timers.append(timer)

    minutes = duration // 60
    seconds = duration % 60
    if minutes > 0:
        time_str = f"{minutes} minute{'s' if minutes != 1 else ''}"
    else:
        time_str = f"{seconds} second{'s' if seconds != 1 else ''}"

    # No parsed message → don't say "remind you to your reminder".
    if message:
        return f"Done. I'll remind you to {message} in {time_str}."
    return f"Done. I'll remind you in {time_str}."
