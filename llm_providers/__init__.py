"""llm_providers — provider-neutral LLM backends for JARVIS.

Add a new provider by subclassing LLMProvider and appending it to the chain
in manager.build_manager_from_config(). No application code changes needed.
"""
from .base import (
    AllProvidersFailedError,
    LLMProvider,
    ProviderError,
    ProviderResponse,
    StreamInterruptedError,
)
from .manager import ProviderManager, build_manager_from_config

__all__ = [
    "AllProvidersFailedError",
    "LLMProvider",
    "ProviderError",
    "ProviderResponse",
    "StreamInterruptedError",
    "ProviderManager",
    "build_manager_from_config",
]
