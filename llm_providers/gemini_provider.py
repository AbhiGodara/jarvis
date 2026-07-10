"""
llm_providers/gemini_provider.py — Google Gemini backend (google-genai SDK).
"""
from __future__ import annotations

import logging

from .base import LLMProvider, ProviderError, ProviderResponse

logger = logging.getLogger(__name__)


class GeminiProvider(LLMProvider):
    def __init__(self, api_key: str, model: str = "gemini-2.5-flash", name: str = "gemini"):
        if not api_key:
            raise ValueError(f"GeminiProvider '{name}' constructed without an API key")
        from google import genai
        from google.genai import types

        self._types = types
        self.name = name
        self.model = model
        self.client = genai.Client(api_key=api_key)

    def _build_contents(self, messages: list[dict]) -> list:
        """Translate the provider-neutral message list.

        Mk-III Phase 4 adds two structured shapes for the tool-result loop:
          {"role": "assistant", "tool_call": {"name", "args", "id"}}
          {"role": "tool", "name", "content", "id"}
        mapped to Gemini FunctionCall / FunctionResponse parts. The Gemini API
        only accepts user|model roles: function calls ride on "model",
        function responses on "user".
        """
        t = self._types
        contents = []
        for m in messages:
            tool_call = m.get("tool_call")
            if m["role"] == "assistant" and tool_call:
                contents.append(t.Content(
                    role="model",
                    parts=[t.Part(function_call=t.FunctionCall(
                        name=tool_call["name"],
                        args=tool_call.get("args", {}),
                    ))],
                ))
            elif m["role"] == "tool":
                contents.append(t.Content(
                    role="user",
                    parts=[t.Part(function_response=t.FunctionResponse(
                        name=m.get("name", ""),
                        response={"result": m.get("content", "")},
                    ))],
                ))
            elif m.get("content"):
                contents.append(t.Content(
                    role="model" if m["role"] == "assistant" else "user",
                    parts=[t.Part(text=m["content"])],
                ))
        return contents

    def _base_config_kwargs(self, system: str, temperature: float) -> dict:
        t = self._types
        config_kwargs = {"system_instruction": system, "temperature": temperature}

        # Gemini 2.5 models "think" before answering by default (~2.5s/call
        # measured) — latency matters more than deep reasoning for a voice
        # assistant, so disable it where supported.
        if "2.5" in self.model:
            try:
                config_kwargs["thinking_config"] = t.ThinkingConfig(thinking_budget=0)
            except Exception:
                pass
        return config_kwargs

    def generate(
        self,
        system: str,
        messages: list[dict],
        tools: list[dict] | None = None,
        temperature: float = 0.7,
    ) -> ProviderResponse:
        t = self._types

        contents = self._build_contents(messages)
        config_kwargs = self._base_config_kwargs(system, temperature)

        if tools:
            declarations = [
                t.FunctionDeclaration(
                    name=tool["name"],
                    description=tool.get("description", ""),
                    parameters=tool.get("parameters", {}),
                )
                for tool in tools
            ]
            config_kwargs["tools"] = [t.Tool(function_declarations=declarations)]
            config_kwargs["tool_config"] = t.ToolConfig(
                function_calling_config=t.FunctionCallingConfig(mode="AUTO")
            )

        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=contents,
                config=t.GenerateContentConfig(**config_kwargs),
            )
        except Exception as e:
            raise ProviderError.from_exception(e) from e

        result = ProviderResponse(provider=self.name)

        usage = getattr(response, "usage_metadata", None)
        if usage:
            result.input_tokens = getattr(usage, "prompt_token_count", 0) or 0
            result.output_tokens = getattr(usage, "candidates_token_count", 0) or 0

        if response.candidates:
            candidate = response.candidates[0]
            if candidate.content and candidate.content.parts:
                for part in candidate.content.parts:
                    if getattr(part, "function_call", None):
                        fc = part.function_call
                        result.tool_name = fc.name
                        result.tool_args = dict(fc.args)
                        return result

        result.text = response.text.strip() if response.text else None
        return result

    def generate_stream(self, system, messages, temperature=0.7):
        """Yield text deltas via generate_content_stream (text-only)."""
        t = self._types
        contents = self._build_contents(messages)
        config_kwargs = self._base_config_kwargs(system, temperature)
        try:
            stream = self.client.models.generate_content_stream(
                model=self.model,
                contents=contents,
                config=t.GenerateContentConfig(**config_kwargs),
            )
            for chunk in stream:
                text = getattr(chunk, "text", None)
                if text:
                    yield text
        except Exception as e:
            # Raised mid-iteration too — the manager decides whether this is
            # a pre-first-token failover or a StreamInterruptedError.
            raise ProviderError.from_exception(e) from e
