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

    def generate(
        self,
        system: str,
        messages: list[dict],
        tools: list[dict] | None = None,
        temperature: float = 0.7,
    ) -> ProviderResponse:
        t = self._types

        contents = [
            t.Content(
                role="model" if m["role"] == "assistant" else "user",
                parts=[t.Part(text=m["content"])],
            )
            for m in messages
            if m.get("content")
        ]

        config_kwargs = {"system_instruction": system, "temperature": temperature}

        # Gemini 2.5 models "think" before answering by default (~2.5s/call
        # measured) — latency matters more than deep reasoning for a voice
        # assistant, so disable it where supported.
        if "2.5" in self.model:
            try:
                config_kwargs["thinking_config"] = t.ThinkingConfig(thinking_budget=0)
            except Exception:
                pass

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
