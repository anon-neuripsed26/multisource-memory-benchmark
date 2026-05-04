"""Abstract base classes for LLM API clients.

Two ABCs:
    - `SyncLLMClient`: per-call cache-aware completion. Used by OpenRouter
      (qwen3, deepseek-v3.2) which has no batch API.
    - `BatchLLMClient`: submit / poll / fetch lifecycle. Used by OpenAI
      (gpt-5.4 via Batch API) and Gemini (3.1-pro-preview via batches).

Both share a single `ModelConfig` per instance and apply default params from
that config to unset request fields before any network call.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import replace
from typing import Any, ClassVar

from .cache import CacheMissError, CacheStore, compute_cache_key
from .types import (
    BatchHandle,
    BatchResultItem,
    BatchStatus,
    CompletionRequest,
    CompletionResult,
    ModelConfig,
)


class _BoundClient(ABC):
    """Common scaffolding shared by sync and batch clients."""

    provider: ClassVar[str] = ""

    # Provider-allowed `CompletionRequest` fields. Setting any field outside
    # this set raises ValueError at validation time.
    allowed_request_fields: ClassVar[frozenset[str]] = frozenset()

    def __init__(self, model_config: ModelConfig, cache_store: CacheStore) -> None:
        if not self.provider:
            raise TypeError(f"{type(self).__name__} must set class attr `provider`")
        if model_config.provider != self.provider:
            raise ValueError(
                f"ModelConfig.provider={model_config.provider!r} does not match "
                f"client provider={self.provider!r}"
            )
        self.model_config = model_config
        self.cache_store = cache_store

    # ------------------------------------------------------------------
    # Internals shared by both client kinds
    # ------------------------------------------------------------------

    def _apply_defaults(self, request: CompletionRequest) -> CompletionRequest:
        """Fill unset (`None`) request fields from `ModelConfig.default_params`."""
        defaults = self.model_config.default_params
        if not defaults:
            return request
        updates: dict[str, Any] = {}
        for field_name, default_value in defaults.items():
            if not hasattr(request, field_name):
                continue
            current = getattr(request, field_name)
            if current is None:
                updates[field_name] = default_value
        if not updates:
            return request
        return replace(request, **updates)

    def _validate_request(self, request: CompletionRequest) -> None:
        """Reject provider-incompatible fields."""
        for field_name in (
            "user_prompt", "system_prompt", "temperature",
            "reasoning_effort", "thinking_level", "response_schema",
            "top_p", "seed",
        ):
            value = getattr(request, field_name)
            if value is None:
                continue
            if field_name not in self.allowed_request_fields:
                raise ValueError(
                    f"{type(self).__name__} (provider={self.provider}) does not accept "
                    f"request field {field_name!r} (got value {value!r}). "
                    f"Allowed fields: {sorted(self.allowed_request_fields)}"
                )


# ---------------------------------------------------------------------------
# Sync
# ---------------------------------------------------------------------------

class SyncLLMClient(_BoundClient):
    """Per-call cache-aware completion.

    Orchestration:
        1. Apply default params from `ModelConfig`.
        2. Validate provider-allowed fields.
        3. Compute SHA256 cache key.
        4. Try cache; if hit, return with `cache_hit=True`.
        5. If miss and `allow_api_call=False`, raise `CacheMissError`.
        6. If allowed, call `_raw_complete()` and persist the response text.
    """

    def complete(
        self,
        request: CompletionRequest,
        *,
        allow_api_call: bool = False,
    ) -> CompletionResult:
        request = self._apply_defaults(request)
        self._validate_request(request)
        key = compute_cache_key(request, self.model_config.api_model_id)
        cached = self.cache_store.get(key, self.provider, self.model_config.api_model_id)
        if cached is not None:
            cached.cache_hit = True
            return cached
        if not allow_api_call:
            raise CacheMissError(
                key=key,
                provider=self.provider,
                model_id=self.model_config.api_model_id,
            )
        text = self._raw_complete(request)
        if not isinstance(text, str):
            raise TypeError(
                f"{type(self).__name__}._raw_complete() must return str, "
                f"got {type(text).__name__}"
            )
        result = CompletionResult(
            text=text,
            finish_reason=None,
            model_id=self.model_config.api_model_id,
            provider=self.provider,
            cache_hit=False,
        )
        self.cache_store.put(key, result)
        return result

    @abstractmethod
    def _raw_complete(self, request: CompletionRequest) -> str:
        """Live API call. Returns the response text."""


# ---------------------------------------------------------------------------
# Batch
# ---------------------------------------------------------------------------

class BatchLLMClient(_BoundClient):
    """Submit a list of requests as a single provider batch job."""

    @abstractmethod
    def submit_batch(self, requests: list[CompletionRequest]) -> BatchHandle:
        """Upload `requests` as a provider-side batch and return a handle."""

    @abstractmethod
    def poll_status(self, handle: BatchHandle) -> BatchStatus:
        """Return the current status of a previously submitted batch."""

    @abstractmethod
    def fetch_results(self, handle: BatchHandle) -> list[BatchResultItem]:
        """Download and parse the per-item results of a completed batch."""

    def run_batch_blocking(
        self,
        requests: list[CompletionRequest],
        *,
        poll_every_s: int = 60,
    ) -> list[BatchResultItem]:
        """Submit → poll → fetch in one call. Blocks until terminal state."""
        if not requests:
            return []
        for req in requests:
            self._validate_request(self._apply_defaults(req))
        applied = [self._apply_defaults(req) for req in requests]
        handle = self.submit_batch(applied)
        while True:
            status = self.poll_status(handle)
            if status in (
                BatchStatus.COMPLETED,
                BatchStatus.FAILED,
                BatchStatus.EXPIRED,
                BatchStatus.CANCELLED,
            ):
                break
            time.sleep(poll_every_s)
        return self.fetch_results(handle)
