"""
agents/memory_manager.py — Two-tier memory system for JARVIS.

Short-term (STM):
  - In-memory ring buffer of recent conversation turns (per session)
  - Cleared on each JARVIS restart
  - Used for within-session conversational context

Long-term (LTM):
  - Persistent JSON key-value store in memory.json
  - Stores user facts: name, city, preferences
  - Survives restarts

Context injection into LLM:
  get_context() combines both tiers into a single system-prompt string.

Backward compatibility:
  Module-level remember(), forget(), get_context(), list_all() functions
  keep existing command modules working without changes.
"""
from __future__ import annotations

import json
import logging
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque

logger = logging.getLogger(__name__)

MEMORY_FILE = "memory.json"
STM_MAX_TURNS = 10   # Keep last 10 turns in short-term memory


@dataclass
class Turn:
    """A single conversation turn stored in short-term memory."""
    role: str           # "user" | "assistant"
    text: str
    timestamp: float = 0.0

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = time.time()


class MemoryManager:
    """
    Two-tier memory: short-term (in-memory ring buffer) + long-term (JSON).
    """

    def __init__(self, ltm_path: str = MEMORY_FILE):
        self._ltm_path = Path(ltm_path)
        self._stm: Deque[Turn] = deque(maxlen=STM_MAX_TURNS)

    # ── Short-term memory ─────────────────────────────────────────────────

    def add_turn(self, role: str, text: str) -> None:
        """Record a conversation turn in short-term memory."""
        self._stm.append(Turn(role=role, text=text))

    def get_recent_turns(self, n: int = 5) -> list[Turn]:
        """Return the n most recent conversation turns."""
        turns = list(self._stm)
        return turns[-n:] if len(turns) > n else turns

    def clear_stm(self) -> None:
        """Clear short-term memory (e.g., on session restart)."""
        self._stm.clear()

    # ── Long-term memory ──────────────────────────────────────────────────

    def save_fact(self, key: str, value: str) -> None:
        """Persist a user fact to long-term memory."""
        facts = self._load_ltm()
        facts[key.lower().strip()] = value.strip()
        self._save_ltm(facts)
        logger.info(f"Memory saved: {key} = {value}")

    def get_fact(self, key: str) -> str | None:
        """Retrieve a specific fact from long-term memory."""
        return self._load_ltm().get(key.lower().strip())

    def forget(self, key: str) -> None:
        """Remove a fact from long-term memory."""
        facts = self._load_ltm()
        if key.lower() in facts:
            del facts[key.lower()]
            self._save_ltm(facts)

    def all_facts(self) -> dict[str, str]:
        return self._load_ltm()

    # ── Context for LLM injection ─────────────────────────────────────────

    def get_context(self, max_facts: int = 10, include_turns: bool = True) -> str:
        """
        Build a combined context string for LLM system prompt injection.
        Combines long-term facts + recent short-term turns.

        Pass include_turns=False when the LLM call already carries the
        conversation history (the planner does) — otherwise recent turns
        get sent to the model twice.
        """
        parts: list[str] = []

        facts = list(self._load_ltm().items())[:max_facts]
        if facts:
            fact_str = ". ".join(f"User's {k}: {v}" for k, v in facts)
            parts.append(f"[User Facts] {fact_str}.")

        if include_turns:
            recent = self.get_recent_turns(n=4)
            if recent:
                turn_strs = [f"{t.role}: {t.text}" for t in recent]
                parts.append("[Recent conversation]\n" + "\n".join(turn_strs))

        return "\n\n".join(parts)

    # ── Private I/O ───────────────────────────────────────────────────────

    def _load_ltm(self) -> dict[str, str]:
        if not self._ltm_path.exists():
            return {}
        try:
            data = json.loads(self._ltm_path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except Exception as e:
            logger.warning(f"Could not load {self._ltm_path}: {e}")
            return {}

    def _save_ltm(self, data: dict) -> None:
        try:
            self._ltm_path.write_text(
                json.dumps(data, indent=2, ensure_ascii=False),
                encoding="utf-8"
            )
        except Exception as e:
            logger.warning(f"Could not save {self._ltm_path}: {e}")


# ── Module-level backward-compatible API ─────────────────────────────────────
# These match the old memory.py interface so existing command modules
# (e.g. commands that call memory.remember()) keep working unchanged.

_default_manager = MemoryManager()


def remember(key: str, value: str) -> None:
    _default_manager.save_fact(key, value)


def forget(key: str) -> None:
    _default_manager.forget(key)


def get_context(max_facts: int = 10) -> str:
    return _default_manager.get_context(max_facts)


def list_all() -> dict:
    return _default_manager.all_facts()
