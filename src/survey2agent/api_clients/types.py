"""Typed dataclasses for the API client layer.

All API-layer interactions go through these objects. No untyped dicts cross
public boundaries. Cache keys are computed from `CompletionRequest` fields
plus `model_id` (provided by the bound client) plus `CACHE_KEY_VERSION`
(defined in `cache.py`).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Mapping


# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CompletionRequest:
    """A single completion request.

    Every field that can affect the model's output appears here explicitly.
    There is intentionally no `extra_body` / `**kwargs` escape hatch: any new
    parameter MUST be added as a typed field and accompanied by a cache-key
    version bump (see `cache.CACHE_KEY_VERSION`). This keeps cache keys
    correct by construction.

    Provider-specific fields:
        - `reasoning_effort` is for OpenAI reasoning models (e.g., gpt-5.4).
        - `thinking_level` is for Gemini thinking models (e.g., 3.1-pro).
        - `response_schema` is a Pydantic model class for structured output;
          required by the Gemini batch client, accepted by the OpenAI batch
          client, ignored on the wire by the OpenRouter sync client.
        - `custom_id` is a per-item correlation token used by batch clients to
          pair input requests with returned results.
    Sending a provider-incompatible field raises in the provider client.
    """

    user_prompt: str
    system_prompt: str = ""
    temperature: float | None = None
    reasoning_effort: str | None = None
    thinking_level: str | None = None
    response_schema: Any = None
    top_p: float | None = None
    seed: int | None = None
    custom_id: str | None = None

    def to_cache_payload(self) -> dict[str, Any]:
        """Canonical, deterministic dict for cache key computation.

        Field order is fixed by this method; it does NOT depend on dataclass
        field declaration order at the call site. JSON serialization in
        `cache.compute_cache_key` adds `sort_keys=True` as a second guarantee.

        `response_schema` is reduced to its module-qualified class name to keep
        it JSON-serializable; `custom_id` is excluded because it is purely a
        correlation token that does not influence model output.
        """
        schema = self.response_schema
        if schema is None:
            schema_repr: str | None = None
        elif isinstance(schema, type):
            schema_repr = f"{schema.__module__}.{schema.__qualname__}"
        else:
            schema_repr = repr(schema)
        return {
            "user_prompt": self.user_prompt,
            "system_prompt": self.system_prompt,
            "temperature": self.temperature,
            "reasoning_effort": self.reasoning_effort,
            "thinking_level": self.thinking_level,
            "response_schema": schema_repr,
            "top_p": self.top_p,
            "seed": self.seed,
        }


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------

@dataclass
class CompletionResult:
    """Result of a completion call.

    `cache_hit` is set by the orchestrator (`SyncLLMClient.complete()` or the
    batch client when reading a cached batch item), not by provider parsing
    code.
    """

    text: str
    finish_reason: str | None
    model_id: str
    provider: str
    cache_hit: bool


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ModelConfig:
    """Single entry from `configs/models.yaml`, after parsing."""

    paper_alias: str
    provider: str
    api_model_id: str
    api_endpoint: str | None
    # Use Mapping (not dict) to allow MappingProxyType wrapping by the loader.
    # `_apply_defaults()` only reads via .items(), so any read-only mapping is OK.
    default_params: Mapping[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Batch
# ---------------------------------------------------------------------------

class BatchStatus(str, Enum):
    """Provider-agnostic batch job lifecycle status."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    EXPIRED = "expired"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class BatchHandle:
    """Opaque handle returned by `BatchLLMClient.submit_batch()`.

    Serializable via `to_json()` / `from_json()` so a long-running batch can
    be resumed by a separate process.
    """

    provider: str
    batch_id: str
    model_id: str
    submitted_at: str | None = None
    provider_metadata: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, ensure_ascii=False)

    @classmethod
    def from_json(cls, payload: str) -> "BatchHandle":
        data = json.loads(payload)
        return cls(
            provider=data["provider"],
            batch_id=data["batch_id"],
            model_id=data["model_id"],
            submitted_at=data.get("submitted_at"),
            provider_metadata=dict(data.get("provider_metadata", {})),
        )


@dataclass(frozen=True)
class BatchResultItem:
    """One item in a batch result list.

    Exactly one of (`text`, `error_message`) is populated. A failed item has
    `text=None`, `finish_reason=None`, and a non-empty `error_message`.
    """

    custom_id: str
    text: str | None
    finish_reason: str | None
    error_message: str | None
