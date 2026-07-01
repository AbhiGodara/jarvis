"""
commands/registry.py — Command plugin registry for JARVIS.

Changes from v1:
  - Auto-discovers all *_cmd.py modules in the commands/ package
  - Exports a Gemini-compatible tool schema so the Planner can use
    function calling rather than keyword scanning
  - Still supports the original @command(keywords=[...]) decorator
    for backward compatibility with all existing command modules
"""
from __future__ import annotations

import importlib
import logging
import pkgutil
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

logger = logging.getLogger(__name__)


@dataclass
class _Command:
    """A registered voice command."""
    keywords: list[str]
    patterns: list[re.Pattern]
    handler: Callable
    description: str
    priority: int = 0       # higher wins when several commands match
    order: int = 0          # registration order (stable tie-break)


_registry: list[_Command] = []

# Tool schema for Gemini function calling
_tool_schemas: list[dict] = []


def _compile_keyword(kw: str) -> re.Pattern:
    """
    Compile a trigger keyword into a word-boundary pattern.

    Plain substring matching caused misroutes: 'mail' fired on 'email',
    'time' on 'times'. A trailing space in a keyword (e.g. "open ") means
    "this word followed by more text".
    """
    if kw.endswith(" "):
        return re.compile(r"(?<!\w)" + re.escape(kw.strip()) + r"\s+\S")
    return re.compile(r"(?<!\w)" + re.escape(kw) + r"(?!\w)")


def command(keywords: list[str], description: str = "", priority: int = 0):
    """
    Decorator to register a voice command handler.

    Args:
        keywords:    Trigger phrases (used for fast keyword matching)
        description: Human-readable description (used in LLM tool schema)
        priority:    Match priority. When multiple commands match a query, the
                     highest priority wins (ties broken by registration order).
                     Use a positive priority for commands whose keywords are
                     specific ("youtube", "temperature in") so they beat
                     generic verbs like "open " or "what is ".

    Example:
        @command(keywords=["weather", "temperature"], description="Get current weather")
        def get_weather(text: str) -> str:
            ...
    """
    def decorator(fn: Callable) -> Callable:
        desc = description or (fn.__doc__ or "").strip().split("\n")[0]
        patterns = [_compile_keyword(kw.lower()) for kw in keywords]
        _registry.append(_Command(
            keywords=keywords,
            patterns=patterns,
            handler=fn,
            description=desc,
            priority=priority,
            order=len(_registry),
        ))
        _tool_schemas.append({
            "name": fn.__name__,
            "description": desc,
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The full user command text"
                    }
                },
                "required": ["text"]
            }
        })
        logger.debug(f"Registered command '{fn.__name__}' with keywords: {keywords}")
        return fn
    return decorator


def match(text: str) -> str | None:
    """
    Keyword-based matching (fast path, no LLM cost).

    Returns the handler's response, or None if nothing matched.
    """
    lower = text.lower()
    for cmd in _scan_order():
        if any(p.search(lower) for p in cmd.patterns):
            logger.info(f"Keyword match → '{cmd.handler.__name__}' (priority {cmd.priority})")
            result = cmd.handler(lower)
            if result is not None:
                return result
    return None


def _scan_order() -> list[_Command]:
    """Commands sorted by priority (highest first), then registration order."""
    return sorted(_registry, key=lambda c: (-c.priority, c.order))


def find_matching_tools(text: str) -> list[str]:
    """Return handler names whose keywords match, in scan order (tests/debugging)."""
    lower = text.lower()
    return [
        cmd.handler.__name__
        for cmd in _scan_order()
        if any(p.search(lower) for p in cmd.patterns)
    ]


def dispatch(tool_name: str, text: str) -> str | None:
    """
    Direct dispatch by function name (used by Planner Agent tool-calling).

    Returns the handler's response, or None if tool_name not found.
    """
    for cmd in _registry:
        if cmd.handler.__name__ == tool_name:
            logger.info(f"Tool dispatch → '{tool_name}'")
            return cmd.handler(text)
    logger.warning(f"dispatch(): unknown tool '{tool_name}'")
    return None


def has_tool(tool_name: str) -> bool:
    """True if a local command with this name is registered.

    The planner checks this before dispatching so MCP tool names are never
    probed against the local registry (which logged spurious
    "unknown tool" warnings on every MCP call).
    """
    return any(cmd.handler.__name__ == tool_name for cmd in _registry)


def get_tool_schemas() -> list[dict]:
    """Return Gemini-compatible function declarations for all registered tools."""
    return list(_tool_schemas)


def list_tools() -> list[str]:
    """Return list of all registered tool names."""
    return [cmd.handler.__name__ for cmd in _registry]


def auto_discover(package_path: Path | None = None) -> None:
    """
    Auto-import all *_cmd.py modules inside the commands package.
    This eliminates the need to manually list imports in brain.py or main.py.
    """
    if package_path is None:
        package_path = Path(__file__).parent

    for module_info in pkgutil.iter_modules([str(package_path)]):
        if module_info.name.endswith("_cmd"):
            full_name = f"commands.{module_info.name}"
            try:
                importlib.import_module(full_name)
                logger.debug(f"Auto-imported command module: {full_name}")
            except Exception as e:
                logger.warning(f"Failed to import {full_name}: {e}")
