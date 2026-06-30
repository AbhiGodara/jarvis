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

    def generate(
        self,
        system: str,
        messages: list[dict],
        tools: list[dict] | None = None,
        temperature: float = 0.7,
    ) -> ProviderResponse:
        chat_messages = [{"role": "system", "content": system}]
        chat_messages += [
            {"role": m["role"], "content": m["content"]}
            for m in messages
            if m.get("content")
        ]

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
