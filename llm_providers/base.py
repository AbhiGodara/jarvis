"""
llm_providers/base.py — Provider-neutral LLM interface for JARVIS.

Business logic (planner, RAG, synthesis) talks to LLMClient, which talks to
a ProviderManager, which talks to LLMProvider implementations. Nothing above
this layer knows whether the answer came from Gemini, OpenAI, or anything
added later (Claude, Groq, local models): implement LLMProvider, register it
in the manager, done.

Message format (provider-neutral):
    [{"role": "user" | "assistant", "content": "..."}, ...]

Tool schema format (provider-neutral, what the command registry produces):
    {"name": ..., "description": ..., "parameters": {JSON schema}}
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass, field


# Substrings that indicate a transient failure worth retrying / failing over.
_TRANSIENT_MARKERS = (
    "503", "unavailable", "429", "resource_exhausted", "rate limit",
    "deadline", "timeout", "timed out", "overloaded", "high demand",
    "connection", "network",
)

# Daily/billing quota exhaustion — failover helps, retrying the same key does not.
_QUOTA_MARKERS = ("quota", "insufficient_quota", "billing")


def classify_error(exc: Exception) -> tuple[bool, bool]:
    """Return (transient, quota) flags for a raw provider exception."""
    msg = str(exc).lower()
    quota = any(m in msg for m in _QUOTA_MARKERS)
    transient = quota or any(m in msg for m in _TRANSIENT_MARKERS)
    return transient, quota


class ProviderError(Exception):
    """Normalized provider failure.

    transient: retrying (another provider, or the same one later) may succeed
    quota:     this key is exhausted — cool it down for a long time
    """

    def __init__(self, message: str, *, transient: bool = False, quota: bool = False):
        super().__init__(message)
        self.transient = transient
        self.quota = quota

    @classmethod
    def from_exception(cls, exc: Exception) -> "ProviderError":
        transient, quota = classify_error(exc)
        return cls(str(exc), transient=transient, quota=quota)


class AllProvidersFailedError(Exception):
    """Every configured provider failed for one request."""

    def __init__(self, errors: dict[str, ProviderError]):
        self.errors = errors
        summary = "; ".join(f"{name}: {str(e)[:120]}" for name, e in errors.items())
        super().__init__(f"All LLM providers failed — {summary}")

    @property
    def all_quota(self) -> bool:
        return bool(self.errors) and all(e.quota for e in self.errors.values())


class StreamInterruptedError(Exception):
    """A streamed answer failed AFTER emitting text (Mk-III Phase 1).

    Failover is impossible at this point — switching providers mid-answer
    would audibly change register and content — so the manager stops the
    stream and hands the caller what was already said.
    """

    def __init__(self, partial_text: str, cause: ProviderError):
        self.partial_text = partial_text
        self.cause = cause
        super().__init__(
            f"Stream interrupted after {len(partial_text)} chars: {cause}"
        )


@dataclass
class ProviderResponse:
    """Normalized result of one generation call."""
    text: str | None = None
    tool_name: str | None = None
    tool_args: dict = field(default_factory=dict)
    input_tokens: int = 0
    output_tokens: int = 0
    provider: str = ""

    @property
    def is_tool_call(self) -> bool:
        return self.tool_name is not None


class LLMProvider(ABC):
    """One LLM backend. Implementations must be stateless per call and raise
    ProviderError (use ProviderError.from_exception) on any failure."""

    name: str = "provider"

    @abstractmethod
    def generate(
        self,
        system: str,
        messages: list[dict],
        tools: list[dict] | None = None,
        temperature: float = 0.7,
    ) -> ProviderResponse:
        """Run one generation. Returns text OR a tool call."""
        raise NotImplementedError

    def generate_stream(
        self,
        system: str,
        messages: list[dict],
        temperature: float = 0.7,
    ) -> Iterator[str]:
        """Yield the answer as text deltas (Mk-III Phase 1).

        Text-only: tool-calling requests keep using generate() — partial tool
        calls are never streamed. This default runs the non-streaming
        generate() and yields the full text once, so a provider works
        unmodified before it grows a native streaming path.
        """
        response = self.generate(system, messages, tools=None, temperature=temperature)
        if response.text:
            yield response.text
