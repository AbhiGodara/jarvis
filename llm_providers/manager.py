"""
llm_providers/manager.py — Priority-ordered provider routing with failover.

Behavior:
  - Providers are tried in priority order; the first healthy one answers.
  - On failure the provider goes into cooldown and the next one is tried —
    the caller never notices a switch.
  - Cooldowns: short for transient errors (the provider usually recovers),
    long for quota exhaustion (the key is dead for a while).
  - Recovery: when a cooldown lapses, the provider silently regains its
    priority slot (production: OpenAI is preferred again automatically).
  - If every provider is cooling down, they are force-tried anyway — a
    possibly-recovered provider beats a guaranteed failure.

Provider selection by environment:
  production  → OpenAI, then Gemini key 1, then Gemini key 2
  development → Gemini key 1, then Gemini key 2  (preserves OpenAI credits)
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from .base import AllProvidersFailedError, LLMProvider, ProviderError, ProviderResponse

logger = logging.getLogger(__name__)

_TRANSIENT_COOLDOWN_S = 30
_QUOTA_COOLDOWN_S = 15 * 60


@dataclass
class _ProviderState:
    provider: LLMProvider
    cooldown_until: float = 0.0
    failures: int = 0


class ProviderManager:
    def __init__(
        self,
        providers: list[LLMProvider],
        transient_cooldown_s: float = _TRANSIENT_COOLDOWN_S,
        quota_cooldown_s: float = _QUOTA_COOLDOWN_S,
        retry_pause_s: float = 0.5,
    ):
        if not providers:
            raise ValueError("ProviderManager needs at least one provider")
        self._states = [_ProviderState(p) for p in providers]
        self._transient_cooldown_s = transient_cooldown_s
        self._quota_cooldown_s = quota_cooldown_s
        self._retry_pause_s = retry_pause_s
        self.last_provider: str | None = None
        logger.info(
            "ProviderManager ready. Priority: "
            + " → ".join(p.name for p in providers)
        )

    @property
    def provider_names(self) -> list[str]:
        return [s.provider.name for s in self._states]

    def generate(
        self,
        system: str,
        messages: list[dict],
        tools: list[dict] | None = None,
        temperature: float = 0.7,
    ) -> ProviderResponse:
        errors: dict[str, ProviderError] = {}
        request = dict(system=system, messages=messages, tools=tools, temperature=temperature)

        # Pass 1: healthy providers in priority order.
        now = time.time()
        response = self._try_providers(
            lambda s: s.cooldown_until <= now, errors, request
        )
        if response:
            return response

        # Pass 2: providers we skipped because they were cooling down —
        # a possibly-recovered provider beats a guaranteed failure.
        response = self._try_providers(
            lambda s: s.provider.name not in errors, errors, request
        )
        if response:
            return response

        # Pass 3: one paced retry for transient failures (keeps resilience
        # for single-provider setups; quota failures are excluded — the key
        # won't recover in half a second).
        retryable = {n for n, e in errors.items() if e.transient and not e.quota}
        if retryable:
            time.sleep(self._retry_pause_s)
            response = self._try_providers(
                lambda s: s.provider.name in retryable, errors, request
            )
            if response:
                return response

        raise AllProvidersFailedError(errors)

    def _try_providers(
        self,
        should_try,
        errors: dict[str, ProviderError],
        request: dict,
    ) -> ProviderResponse | None:
        """Attempt providers (in priority order) that pass the filter."""
        for state in self._states:
            name = state.provider.name
            if not should_try(state):
                continue
            try:
                response = state.provider.generate(**request)
                state.failures = 0
                state.cooldown_until = 0.0
                errors.pop(name, None)
                if self.last_provider != name:
                    logger.info(f"LLM provider in use: {name}")
                self.last_provider = name
                return response
            except ProviderError as e:
                errors[name] = e
                self._apply_cooldown(state, e)
                logger.warning(
                    f"Provider '{name}' failed "
                    f"({'quota' if e.quota else 'transient' if e.transient else 'hard'}): "
                    f"{str(e)[:140]} — failing over."
                )
        return None

    def _apply_cooldown(self, state: _ProviderState, error: ProviderError) -> None:
        state.failures += 1
        if error.quota:
            cooldown = self._quota_cooldown_s
        elif error.transient:
            # Back off harder on repeated failures, capped at the quota window
            cooldown = min(
                self._transient_cooldown_s * (2 ** (state.failures - 1)),
                self._quota_cooldown_s,
            )
        else:
            cooldown = self._transient_cooldown_s
        state.cooldown_until = time.time() + cooldown


def build_manager_from_config(cfg) -> ProviderManager:
    """
    Assemble the provider chain from typed config.

    cfg needs: environment, openai_api_key, llm_openai_model,
               gemini_api_key1, gemini_api_key2, gemini_api_key (legacy), llm_model
    """
    from .gemini_provider import GeminiProvider
    from .openai_provider import OpenAIProvider

    env = (cfg.environment or "development").strip().lower()
    gemini1 = cfg.gemini_api_key1 or cfg.gemini_api_key   # legacy single-key fallback
    gemini2 = cfg.gemini_api_key2

    providers: list[LLMProvider] = []

    if env == "production" and cfg.openai_api_key:
        providers.append(
            OpenAIProvider(api_key=cfg.openai_api_key, model=cfg.llm_openai_model, name="openai")
        )
    elif env == "production":
        logger.warning("ENVIRONMENT=production but OPENAI_API_KEY is missing — Gemini only.")

    if gemini1:
        providers.append(GeminiProvider(api_key=gemini1, model=cfg.llm_model, name="gemini-1"))
    if gemini2 and gemini2 != gemini1:
        providers.append(GeminiProvider(api_key=gemini2, model=cfg.llm_model, name="gemini-2"))

    if not providers:
        raise ValueError(
            "No LLM provider keys configured. Set GEMINI_API_KEY1 (or GEMINI_API_KEY), "
            "GEMINI_API_KEY2, and/or OPENAI_API_KEY in .env"
        )

    logger.info(f"LLM environment: {env}")
    return ProviderManager(providers)
