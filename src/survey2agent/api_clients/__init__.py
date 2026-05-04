"""API clients.

Public surface:
    - CompletionRequest, CompletionResult, ModelConfig
    - BatchHandle, BatchResultItem, BatchStatus
    - CacheStore, CacheMissError, compute_cache_key, CACHE_KEY_VERSION
    - SyncLLMClient, BatchLLMClient
    - OpenAIBatchClient, GeminiBatchClient, OpenRouterSyncClient
    - load_model_config, load_all_model_configs
"""

from __future__ import annotations

from typing import Any

from .base import BatchLLMClient, SyncLLMClient
from .cache import CACHE_KEY_VERSION, CacheMissError, CacheStore, compute_cache_key
from .config_loader import load_all_model_configs, load_model_config
from .types import (
    BatchHandle,
    BatchResultItem,
    BatchStatus,
    CompletionRequest,
    CompletionResult,
    ModelConfig,
)

__all__ = [
    "BatchHandle",
    "BatchLLMClient",
    "BatchResultItem",
    "BatchStatus",
    "CACHE_KEY_VERSION",
    "CacheMissError",
    "CacheStore",
    "CompletionRequest",
    "CompletionResult",
    "GeminiBatchClient",
    "ModelConfig",
    "OpenAIBatchClient",
    "OpenRouterSyncClient",
    "SyncLLMClient",
    "compute_cache_key",
    "load_all_model_configs",
    "load_model_config",
]


# Provider client classes are imported lazily so that pulling in the public
# data types and abstract bases does not require httpx / openai / google-genai
# to be installed in the current environment.
_LAZY = {
    "OpenRouterSyncClient": ("openrouter_sync", "OpenRouterSyncClient"),
    "OpenAIBatchClient": ("openai_batch", "OpenAIBatchClient"),
    "GeminiBatchClient": ("gemini_batch", "GeminiBatchClient"),
}


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        module_name, attr = _LAZY[name]
        from importlib import import_module

        module = import_module(f".{module_name}", __name__)
        value = getattr(module, attr)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
