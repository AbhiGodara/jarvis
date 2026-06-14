import re
import logging
import subprocess
import platform
from commands.registry import command
from core.config import get_config

logger = logging.getLogger(__name__)

_APPS: dict[str, str] = get_config().apps
_SYSTEM = platform.system()  # "Windows", "Linux", "Darwin"

# Strip trigger verbs as whole words only — plain str.replace mangled app
# names ("launch runescape" → "launch escape").
_TRIGGER_RE = re.compile(r"\b(open|launch|start|run)\b")


def _extract_app_name(text: str) -> str:
    return " ".join(_TRIGGER_RE.sub(" ", text).split())


@command(keywords=["open ", "launch ", "start ", "run "])
def open_app(text: str) -> str:
    """Open an application or file by voice command."""
    text = _extract_app_name(text)

    if not text:
        return "Which app would you like me to open?"

    # Try to match against the apps dictionary
    app_cmd = None
    for name, cmd in _APPS.items():
        if name in text or text in name:
            app_cmd = cmd
            break

    if not app_cmd:
        return f"I don't know how to open {text}. You can add it to the apps section in config.yaml."

    try:
        if _SYSTEM == "Windows":
            subprocess.Popen(app_cmd, shell=True)
        elif _SYSTEM == "Darwin":  # macOS
            subprocess.Popen(["open", "-a", app_cmd])
        else:  # Linux
            subprocess.Popen([app_cmd])

        return f"Opening {text}."

    except FileNotFoundError:
        return f"I couldn't find {text} on your system. Make sure it's installed."
    except Exception as e:
        logger.error(f"Failed to open app '{app_cmd}': {e}")
        return f"I ran into a problem opening {text}."
