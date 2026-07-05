import os
import logging
from datetime import datetime
from commands.registry import command
from core.config import get_config

logger = logging.getLogger(__name__)

_NOTES_FILE = get_config().notes_file
_SAVE_TRIGGERS = ["take a note", "make a note", "note that", "save a note", "add a note"]
_READ_TRIGGERS = ["read my notes", "what are my notes", "show my notes", "list my notes"]


@command(keywords=["take a note", "make a note", "note that", "save a note", "add a note",
                   "read my notes", "what are my notes", "show my notes", "list my notes"])
def manage_notes(text: str) -> str:
    """Save a new voice note or read existing notes back aloud."""

    # Detect intent: read or save
    if any(trigger in text for trigger in _READ_TRIGGERS):
        return _read_notes()

    for trigger in _SAVE_TRIGGERS:
        if trigger in text:
            content = text.split(trigger, 1)[-1].strip()
            return _save_note(content)

    return "Did you want to save a note or hear your existing notes?"


def _save_note(content: str) -> str:
    """Append a timestamped note to the notes file."""
    if not content:
        return "What would you like to note down?"

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    entry = f"[{timestamp}] {content}\n"

    try:
        with open(_NOTES_FILE, "a", encoding="utf-8") as f:
            f.write(entry)
        logger.info(f"Note saved: '{content}'")
        return "Note saved."
    except IOError as e:
        logger.error(f"Failed to save note: {e}")
        return "I couldn't save that note. There might be a file permission issue."


def _read_notes() -> str:
    """Read the most recent notes from the notes file."""
    if not os.path.exists(_NOTES_FILE):
        return "You have no saved notes."

    try:
        with open(_NOTES_FILE, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]

        if not lines:
            return "Your notes file is empty."

        recent = lines[-3:]  # Read only the last 3 notes to keep it concise
        count = len(lines)
        intro = f"You have {count} note{'s' if count != 1 else ''} total. Here are the most recent."
        notes_text = ". ".join(recent)
        return f"{intro} {notes_text}"

    except IOError as e:
        logger.error(f"Failed to read notes: {e}")
        return "I couldn't read your notes."
