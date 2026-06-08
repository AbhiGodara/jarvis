import os
import time
import logging
import platform
from datetime import datetime
import pyautogui
from commands.registry import command

logger = logging.getLogger(__name__)


@command(keywords=["screenshot", "capture the screen", "capture screen", "switch window", "switch the window", "hide files", "make files visible", "show files"])
def os_utilities(text: str) -> str:
    """Handle desktop utilities like screenshot, window switching, and file visibility."""
    lower_text = text.lower()

    # --- 1. Screenshot ---
    if any(kw in lower_text for kw in ["screenshot", "capture the screen", "capture screen"]):
        try:
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            filename = f"screenshot_{timestamp}.png"
            
            # Take screenshot and save it
            screenshot = pyautogui.screenshot()
            screenshot.save(filename)
            
            abs_path = os.path.abspath(filename)
            logger.info(f"Screenshot saved to {abs_path}")
            
            # Optional: Open the screenshot to show the user
            try:
                if platform.system() == "Windows":
                    os.startfile(abs_path)
                elif platform.system() == "Darwin":
                    os.system(f"open {abs_path}")
                else:
                    os.system(f"xdg-open {abs_path}")
            except Exception as e:
                logger.warning(f"Could not open screenshot: {e}")

            return f"I have taken a screenshot and saved it as {filename}."
        except Exception as e:
            logger.error(f"Screenshot failed: {e}")
            return "I failed to take a screenshot."

    # --- 2. Switch Window ---
    elif "switch" in lower_text and "window" in lower_text:
        try:
            logger.info("Switching windows via Alt+Tab")
            pyautogui.keyDown("alt")
            pyautogui.press("tab")
            time.sleep(0.5)
            pyautogui.keyUp("alt")
            return "Switching the window."
        except Exception as e:
            logger.error(f"Window switch failed: {e}")
            return "I was unable to switch windows."

    # --- 3. Hide / Show Files in Current Folder (Windows only) ---
    elif any(kw in lower_text for kw in ["hide files", "hide this folder"]):
        if platform.system() != "Windows":
            return "Hiding files via attrib is only supported on Windows."
        try:
            # Hide the folder files (excluding python scripts and git metadata is complex,
            # so we just run standard attrib in the folder as JARVIS-master did, but logs a warning)
            logger.info("Hiding files in current directory...")
            # We target files in the current folder, not the whole venv or git directories to avoid breaking things!
            # Let's run it just for specific files in CWD or generally in CWD without subdirectories (/s /d might hide venv which is bad)
            # JARVIS-master ran: os.system("attrib +h /s /d")
            # Let's just do "attrib +h" for standard text/log/config files to be safe, or just hide/unhide notes.txt and logs.
            # To match JARVIS-master behavior but make it safer, we only hide notes.txt and logs:
            os.system("attrib +h notes.txt")
            if os.path.exists("jarvis.log"):
                os.system("attrib +h jarvis.log")
            return "I have hidden your sensitive notes and logs."
        except Exception as e:
            logger.error(f"Hiding files failed: {e}")
            return "I could not hide the files."

    elif any(kw in lower_text for kw in ["visible", "show files"]):
        if platform.system() != "Windows":
            return "Showing files via attrib is only supported on Windows."
        try:
            logger.info("Making files visible...")
            os.system("attrib -h notes.txt")
            if os.path.exists("jarvis.log"):
                os.system("attrib -h jarvis.log")
            return "I have made your files visible."
        except Exception as e:
            logger.error(f"Unhiding files failed: {e}")
            return "I could not make the files visible."

    return "Command not recognized in OS Utilities."
