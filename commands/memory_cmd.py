"""
commands/memory_cmd.py — Explicit long-term memory commands (Mk-III Phase 7).

Deterministic, zero-LLM read/write over the persistent fact store:

  "remember that my name is Tony"   -> saves name = Tony
  "remember I live in Boston"       -> saves location = Boston
  "forget my name"                  -> removes the name fact
  "what do you remember about me"   -> reads saved facts back aloud

These run on the keyword fast path, so they never spend an API call — the
counterpart to the heuristic-gated background extractor in the planner, which
handles facts stated in passing. Saved facts are injected into later prompts by
llm.py, so JARVIS can use them in ordinary answers and across restarts.
"""
from __future__ import annotations

import re

from commands.registry import command
from agents.memory_manager import _default_manager as memory

_SAVE_TRIGGERS = ["remember that", "remember my", "remember this",
                  "please remember", "keep in mind"]
_FORGET_TRIGGERS = ["forget my", "forget that", "forget about", "forget the",
                    "forget what"]
_READ_TRIGGERS = ["what do you remember", "what do you know about me",
                  "what have you remembered", "what do you remember about me"]


@command(
    keywords=_SAVE_TRIGGERS + _FORGET_TRIGGERS + _READ_TRIGGERS,
    description="Save, recall, or forget durable facts about the user",
    examples=[
        "remember that my name is tony",
        "please remember i live in boston",
        "keep in mind that my favourite colour is blue",
        "what do you know about me",
        "what do you remember about me",
        "forget my address",
    ],
)
def manage_memory(text: str) -> str | None:
    """Route an explicit memory utterance to read/forget/save (no LLM)."""
    t = text.strip()

    if any(trigger in t for trigger in _READ_TRIGGERS):
        return _read()

    for trigger in _FORGET_TRIGGERS:
        if trigger in t:
            return _forget(t.split(trigger, 1)[-1])

    for trigger in _SAVE_TRIGGERS:
        if trigger in t:
            return _save(t.split(trigger, 1)[-1])

    # Matched a keyword by word boundary but no sub-intent fit — decline so
    # the planner's next tier can try.
    return None


def _save(content: str) -> str:
    content = content.strip().strip(".!?,").strip()
    if not content:
        return "What would you like me to remember, sir?"

    key, value = _extract_kv(content)
    if key and value:
        memory.save_fact(key, value)
        return f"Noted, sir. I'll remember that your {key.replace('_', ' ')} is {value}."

    # No clean key/value — store the phrase verbatim so it can still be recalled.
    memory.save_fact(_freeform_key(content), content)
    return "I'll remember that, sir."


def _forget(rest: str) -> str:
    rest = rest.strip().strip(".!?,").strip()
    key, _ = _extract_kv(rest)
    if not key:
        key = _normalize_key(rest)
    if not key:
        return "What would you like me to forget, sir?"
    if memory.get_fact(key) is None:
        return f"I don't have anything saved for your {key.replace('_', ' ')}, sir."
    memory.forget(key)
    return f"Done, sir. I've forgotten your {key.replace('_', ' ')}."


def _read() -> str:
    facts = memory.all_facts()
    if not facts:
        return "I don't have anything saved about you yet, sir."
    parts: list[str] = []
    for key, value in list(facts.items())[:8]:
        if key.startswith("note_"):
            parts.append(str(value))
        else:
            parts.append(f"your {key.replace('_', ' ')} is {value}")
    return "Here's what I remember, sir. " + ". ".join(parts) + "."


# ── Parsing helpers ──────────────────────────────────────────────────────────

def _extract_kv(content: str) -> tuple[str | None, str | None]:
    """Pull a (key, value) fact out of a spoken statement, or (None, None)."""
    c = content.strip().lower()

    m = re.match(r"call me (.+)", c)
    if m:
        return ("name", m.group(1).strip())

    m = re.match(r"i live (?:in|at) (.+)", c)
    if m:
        return ("location", m.group(1).strip())

    m = re.match(r"i work (?:at|for) (.+)", c)
    if m:
        return ("workplace", m.group(1).strip())

    if " is " in c:
        key, _, value = c.partition(" is ")
        return (_normalize_key(key), value.strip())

    return (None, None)


def _normalize_key(key: str) -> str:
    """'that my favourite colour' -> 'favourite_colour'; strips leading fillers.

    The trailing '+' lets a run of leading filler words fall away at once, so a
    key left over from 'keep in mind that my ...' ('that my name') collapses to
    just 'name'.
    """
    key = key.strip().lower()
    key = re.sub(r"^((?:that|my|the|a|an)\s+)+", "", key)
    key = re.sub(r"[^a-z0-9]+", "_", key).strip("_")
    return key[:40]


def _freeform_key(content: str) -> str:
    """Stable 'note_...' slug so a verbatim memory can be listed back."""
    slug = re.sub(r"[^a-z0-9]+", "_", content.lower()).strip("_")
    return ("note_" + slug)[:48]
