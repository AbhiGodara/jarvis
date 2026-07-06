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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

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
    # Natural paraphrases for the semantic router (Mk-III Phase 3) — queries
    # that SHOULD reach this command even though no keyword matches.
    examples: list[str] = field(default_factory=list)


_registry: list[_Command] = []

# Tool schema for Gemini function calling
_tool_schemas: list[dict] = []

# Semantic example index: (normalized embedding matrix [N, d], owner per row).
# Built lazily on the first semantic_best() call once the embedder is up.
_semantic_index: tuple[np.ndarray, list[_Command]] | None = None


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


def command(
    keywords: list[str],
    description: str = "",
    priority: int = 0,
    examples: list[str] | None = None,
):
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
        examples:    4-8 natural paraphrases for the semantic router (Mk-III
                     Phase 3) — how users actually phrase this request when no
                     keyword matches ("do i need an umbrella" → weather).

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
            examples=[e.lower() for e in (examples or [])],
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


def _ensure_semantic_index() -> tuple[np.ndarray, list[_Command]] | None:
    """Build (once) the example-embedding matrix. None while the embedder is
    still loading — the next query simply retries."""
    global _semantic_index
    if _semantic_index is not None:
        return _semantic_index

    texts: list[str] = []
    owners: list[_Command] = []
    for cmd in _registry:
        for example in cmd.examples:
            texts.append(example)
            owners.append(cmd)
    if not texts:
        return None

    from core import embedder
    matrix = embedder.encode_if_ready(texts)
    if matrix is None:
        return None

    _semantic_index = (matrix, owners)
    logger.info(
        f"Semantic router index built: {len(texts)} examples across "
        f"{len({c.handler.__name__ for c in owners})} commands."
    )
    return _semantic_index


def semantic_best(query: str) -> tuple[_Command, float] | None:
    """Best-scoring command by example similarity (Mk-III Phase 3).

    Returns (command, cosine_score) regardless of threshold, or None when the
    semantic index isn't available (embedder still loading, no examples).
    Embeddings are L2-normalized, so cosine similarity is a dot product.
    """
    index = _ensure_semantic_index()
    if index is None:
        return None
    from core import embedder
    query_vec = embedder.encode_if_ready([query.lower()])
    if query_vec is None:
        return None
    matrix, owners = index
    scores = matrix @ query_vec[0]
    best = int(np.argmax(scores))
    return owners[best], float(scores[best])


def semantic_match(query: str, threshold: float = 0.62) -> tuple[_Command, float] | None:
    """semantic_best() gated by the routing threshold."""
    best = semantic_best(query)
    if best is not None and best[1] >= threshold:
        return best
    return None


def reset_semantic_index() -> None:
    """Drop the cached example matrix (tests that register extra commands)."""
    global _semantic_index
    _semantic_index = None


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
