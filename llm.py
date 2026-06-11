"""
llm.py — LLM client for JARVIS.

This is the application-facing LLM interface. It owns conversation history,
the persona/system prompt, and the tool-call marker protocol — and delegates
actual generation to a ProviderManager (llm_providers/), which routes across
OpenAI and Gemini with automatic failover. No business logic in this file or
above it knows which provider answered.

Provider selection (see llm_providers/manager.py):
  ENVIRONMENT=production   → OpenAI → Gemini key 1 → Gemini key 2
  ENVIRONMENT=development  → Gemini key 1 → Gemini key 2
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from llm_providers import AllProvidersFailedError, ProviderManager

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).parent

# Always resolve against the project root so reads and writes hit the same
# file regardless of the current working directory.
HISTORY_FILE = _PROJECT_ROOT / "history.json"
PERSONA_FILE = _PROJECT_ROOT / "persona.txt"


def _load_persona() -> str:
    """Load the JARVIS personality prompt from persona.txt."""
    if PERSONA_FILE.exists():
        try:
            return PERSONA_FILE.read_text(encoding="utf-8").strip()
        except Exception:
            pass
    return (
        "You are JARVIS, a highly intelligent and concise voice assistant. "
        "Keep all responses brief and spoken-English friendly — no markdown, "
        "no bullet points, just natural conversational sentences."
    )


def _load_history(max_turns: int) -> list:
    """Load conversation history from disk, capped at max_turns pairs."""
    if HISTORY_FILE.exists():
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                history = json.load(f)
            cap = max_turns * 2
            return history[-cap:] if len(history) > cap else history
        except Exception as e:
            logger.warning(f"Could not load history: {e}")
    return []


def _save_history(history: list) -> None:
    """Persist current chat history to disk."""
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)
    except Exception as e:
        logger.warning(f"Failed to save history: {e}")


class LLMClient:
    """
    Conversation-level LLM interface.

    Supports:
      - Standard conversation with persistent history
      - Native function calling via tool_schemas (for PlannerAgent)
      - RAG-grounded answers (skip_history=True)
      - Token usage tracking (for EvalLogger)
      - Multi-provider failover (transparent to callers)
    """

    def __init__(
        self,
        model_name: str,
        max_history_turns: int = 20,
        provider_manager: ProviderManager | None = None,
    ):
        self.model_name = model_name
        self.max_turns = max_history_turns
        self.persona = _load_persona()

        if provider_manager is None:
            from core.config import get_config
            from llm_providers import build_manager_from_config
            provider_manager = build_manager_from_config(get_config())
        self.providers = provider_manager

        self._history: list[dict] = _load_history(max_history_turns)
        self._last_input_tokens = 0
        self._last_output_tokens = 0
        self._last_error: str | None = None

        logger.info(
            f"LLMClient ready. Providers: {' → '.join(self.providers.provider_names)}. "
            f"History: {len(self._history)} messages loaded."
        )

    @property
    def last_input_tokens(self) -> int:
        return self._last_input_tokens

    @property
    def last_output_tokens(self) -> int:
        return self._last_output_tokens

    @property
    def last_provider(self) -> str | None:
        """Name of the provider that served the most recent successful call."""
        return self.providers.last_provider

    @property
    def last_error(self) -> str | None:
        """Error from the most recent ask() call, or None if it succeeded.

        ask() returns a user-friendly fallback string on failure so the voice
        loop always has something to speak — callers that need to know whether
        the answer is real (planner, eval logging) must check this.
        """
        return self._last_error

    def ask(
        self,
        prompt: str,
        memory_context: str = "",
        skip_history: bool = False,
        tool_schemas: list[dict] | None = None,
        max_history_messages: int | None = None,
    ) -> str:
        """
        Send a prompt through the provider chain and return the response text.

        Args:
            prompt:         The user's query
            memory_context: Optional context to inject into system prompt (memory or RAG)
            skip_history:   If True, don't read/write conversation history (use for RAG)
            tool_schemas:   Optional function declarations for tool calling
            max_history_messages: Cap on history messages SENT with this request
                            (history is still recorded in full). Planning calls
                            don't need 40 messages of context — trimming them
                            measurably cuts tokens and latency.

        Returns:
            Response text, or a __TOOL_CALL__ marker string if function calling triggered.
        """
        if not prompt or not prompt.strip():
            return "I didn't catch that."

        system_text = self.persona
        if memory_context:
            system_text += f"\n\n{memory_context}"

        messages: list[dict] = []
        if not skip_history:
            history = self._history
            if max_history_messages is not None:
                history = history[-max_history_messages:]
            for msg in history:
                parts = msg.get("parts", [])
                if parts and isinstance(parts[0], str) and parts[0]:
                    role = "assistant" if msg.get("role") == "model" else "user"
                    messages.append({"role": role, "content": parts[0]})
        messages.append({"role": "user", "content": prompt})

        self._last_error = None
        try:
            response = self.providers.generate(
                system=system_text,
                messages=messages,
                tools=tool_schemas,
                temperature=0.7,
            )
        except AllProvidersFailedError as e:
            logger.error(f"All LLM providers failed: {e}")
            self._last_error = str(e)
            if e.all_quota:
                return (
                    "I've hit the usage limits on all my language model providers. "
                    "Local commands like time, weather, and notes still work."
                )
            return "I'm having trouble connecting right now. Please try again in a moment."

        self._last_input_tokens = response.input_tokens
        self._last_output_tokens = response.output_tokens

        if response.is_tool_call:
            return f"__TOOL_CALL__:{response.tool_name}:{json.dumps(response.tool_args)}"

        answer = response.text or "I'm not sure how to respond to that."

        if not skip_history:
            self.record_turn(prompt, answer)

        return answer

    def record_turn(self, user_text: str, assistant_text: str) -> None:
        """
        Append a user/assistant exchange to conversation history without an API call.

        Used by the planner for tool-assisted answers: the internal synthesis
        prompt must not be stored as if the user said it, but the actual
        exchange should still be remembered.
        """
        self._history.append({"role": "user", "parts": [user_text]})
        self._history.append({"role": "model", "parts": [assistant_text]})
        cap = self.max_turns * 2
        if len(self._history) > cap:
            self._history = self._history[-cap:]
        _save_history(self._history)

    def plan(
        self,
        prompt: str,
        tool_schemas: list[dict],
        max_history_messages: int | None = 8,
    ) -> dict | str:
        """
        Ask the LLM to choose a tool OR answer directly.

        Returns either:
          {"tool": "tool_name", "args": {...}}   — tool call decision
          str                                    — direct LLM answer
        """
        response = self.ask(
            prompt,
            tool_schemas=tool_schemas,
            max_history_messages=max_history_messages,
        )

        if response.startswith("__TOOL_CALL__:"):
            parts = response.split(":", 2)
            if len(parts) == 3:
                try:
                    return {"tool": parts[1], "args": json.loads(parts[2])}
                except json.JSONDecodeError:
                    pass

        return response
