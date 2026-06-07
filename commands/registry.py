"""
Command registry for JARVIS.

Usage:
    from commands.registry import command

    @command(keywords=["weather", "temperature"])
    def my_handler(text: str) -> str:
        return "It is sunny."

The handler receives the full lowercased command string and returns a response string.
"""

import logging

logger = logging.getLogger(__name__)

_registry: list[tuple[list[str], callable]] = []


def command(keywords: list[str]):
    """
    Decorator to register a voice command handler.

    Args:
        keywords: List of strings. If any of these appear in the user's command,
                  this handler will be called.

    Example:
        @command(keywords=["remind me", "set a reminder"])
        def reminder(text: str) -> str:
            ...
    """
    def decorator(fn):
        _registry.append((keywords, fn))
        logger.debug(f"Registered command '{fn.__name__}' with keywords: {keywords}")
        return fn
    return decorator


def match(text: str) -> str | None:
    """
    Find and execute the first command whose keywords appear in the input text.

    Args:
        text: The user's lowercased command string

    Returns:
        The handler's response string, or None if no command matched
    """
    lower = text.lower()
    for keywords, handler in _registry:
        if any(kw in lower for kw in keywords):
            logger.info(f"Command matched by keywords {keywords} → handler '{handler.__name__}'")
            return handler(lower)
    return None
