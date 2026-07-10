"""
llm_providers/openai_provider.py — OpenAI backend (chat.completions).
"""
from __future__ import annotations

import json
import logging

from .base import LLMProvider, ProviderError, ProviderResponse

logger = logging.getLogger(__name__)

_REQUEST_TIMEOUT_S = 20


class OpenAIProvider(LLMProvider):
    def __init__(self, api_key: str, model: str = "gpt-4o-mini", name: str = "openai"):
        if not api_key:
            raise ValueError(f"OpenAIProvider '{name}' constructed without an API key")
        from openai import OpenAI

        self.name = name
        self.model = model
        self.client = OpenAI(api_key=api_key, timeout=_REQUEST_TIMEOUT_S, max_retries=0)

    @staticmethod
    def _build_chat_messages(system: str, messages: list[dict]) -> list[dict]:
        """Translate the provider-neutral message list.

        Mk-III Phase 4 adds two structured shapes for the tool-result loop:
          {"role": "assistant", "tool_call": {"name", "args", "id"}}
          {"role": "tool", "name", "content", "id"}
        mapped to OpenAI's assistant.tool_calls + role:"tool" messages with a
        matching tool_call_id.
        """
        chat_messages = [{"role": "system", "content": system}]
        for m in messages:
            tool_call = m.get("tool_call")
            if m["role"] == "assistant" and tool_call:
                chat_messages.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{
                        "id": tool_call.get("id") or "call_0",
                        "type": "function",
                        "function": {
                            "name": tool_call["name"],
                            "arguments": json.dumps(tool_call.get("args", {})),
                        },
                    }],
                })
            elif m["role"] == "tool":
                chat_messages.append({
                    "role": "tool",
                    "tool_call_id": m.get("id") or "call_0",
                    "content": m.get("content", ""),
                })
            elif m.get("content"):
                chat_messages.append({"role": m["role"], "content": m["content"]})
        return chat_messages

    def generate(
        self,
        system: str,
        messages: list[dict],
        tools: list[dict] | None = None,
        temperature: float = 0.7,
    ) -> ProviderResponse:
        chat_messages = self._build_chat_messages(system, messages)

        kwargs: dict = {
            "model": self.model,
            "messages": chat_messages,
            "temperature": temperature,
        }
        if tools:
            kwargs["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": tool["name"],
                        "description": tool.get("description", ""),
                        "parameters": tool.get("parameters")
                        or {"type": "object", "properties": {}},
                    },
                }
                for tool in tools
            ]
            kwargs["tool_choice"] = "auto"

        try:
            response = self.client.chat.completions.create(**kwargs)
        except Exception as e:
            raise ProviderError.from_exception(e) from e

        result = ProviderResponse(provider=self.name)

        usage = getattr(response, "usage", None)
        if usage:
            result.input_tokens = getattr(usage, "prompt_tokens", 0) or 0
            result.output_tokens = getattr(usage, "completion_tokens", 0) or 0

        message = response.choices[0].message
        if message.tool_calls:
            call = message.tool_calls[0]
            result.tool_name = call.function.name
            try:
                result.tool_args = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError:
                result.tool_args = {}
            return result

        result.text = message.content.strip() if message.content else None
        return result

    def generate_stream(self, system, messages, temperature=0.7):
        """Yield content deltas via chat.completions streaming (text-only)."""
        chat_messages = self._build_chat_messages(system, messages)
        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=chat_messages,
                temperature=temperature,
                stream=True,
            )
            for chunk in stream:
                if not chunk.choices:   # e.g. a trailing usage-only chunk
                    continue
                delta = chunk.choices[0].delta
                if delta and delta.content:
                    yield delta.content
        except Exception as e:
            # Raised mid-iteration too — the manager decides whether this is
            # a pre-first-token failover or a StreamInterruptedError.
            raise ProviderError.from_exception(e) from e
