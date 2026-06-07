import os
import logging
import subprocess
import platform
import yaml
from commands.registry import command

logger = logging.getLogger(__name__)

with open("config.yaml", "r") as f:
    _config = yaml.safe_load(f)

_APPS: dict[str, str] = _config.get("apps", {})
_SYSTEM = platform.system()  # "Windows", "Linux", "Darwin"


@command(keywords=["open ", "launch ", "start ", "run "])
def open_app(text: str) -> str:
    """Open an application or file by voice command."""
    for trigger in ["open", "launch", "start", "run"]:
        text = text.replace(trigger, "").strip()

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
