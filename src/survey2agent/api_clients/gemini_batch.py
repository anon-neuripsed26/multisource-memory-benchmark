"""Gemini Batch API client (gemini-3.1-pro-preview with thinking_level=high).

Lifecycle:
    1. `submit_batch(requests)` — build inline requests and call
       `client.batches.create()` from the `google.genai` SDK.
    2. `poll_status(handle)` — query batch state, mapped to `BatchStatus`.
    3. `fetch_results(handle)` — download per-item responses, parse each
       into a `BatchResultItem`.

Per-item failures (parse error, content filter, etc.) surface as a
`BatchResultItem` with `text=None` and a non-empty `error_message`.

Important: in the Gemini batch API, `response_schema` MUST be a Pydantic
model class. Raw JSON Schema dicts are silently dropped server-side, which
yields unstructured text output that fails downstream parsing. We assert
the type at submit time.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, ClassVar

from pydantic import BaseModel

from .base import BatchLLMClient
from .types import (
    BatchHandle,
    BatchResultItem,
    BatchStatus,
    CompletionRequest,
)


_STATUS_MAP: dict[str, BatchStatus] = {
    "JOB_STATE_PENDING": BatchStatus.PENDING,
    "JOB_STATE_QUEUED": BatchStatus.PENDING,
    "JOB_STATE_RUNNING": BatchStatus.IN_PROGRESS,
    "JOB_STATE_SUCCEEDED": BatchStatus.COMPLETED,
    "JOB_STATE_FAILED": BatchStatus.FAILED,
    "JOB_STATE_CANCELLED": BatchStatus.CANCELLED,
    "JOB_STATE_EXPIRED": BatchStatus.EXPIRED,
}


class GeminiBatchClient(BatchLLMClient):
    provider: ClassVar[str] = "google"
    allowed_request_fields: ClassVar[frozenset[str]] = frozenset(
        {
            "user_prompt",
            "system_prompt",
            "temperature",
            "thinking_level",
            "response_schema",
            "seed",
            "custom_id",
        }
    )

    def _client(self) -> Any:
        from google import genai  # type: ignore[import-not-found]

        return genai.Client()

    # ------------------------------------------------------------------
    # Submit
    # ------------------------------------------------------------------

    def submit_batch(self, requests: list[CompletionRequest]) -> BatchHandle:
        if not requests:
            raise ValueError("cannot submit an empty batch")
        applied = [self._apply_defaults(r) for r in requests]
        for r in applied:
            self._validate_request(r)
            if not r.custom_id:
                raise ValueError(
                    "GeminiBatchClient requires CompletionRequest.custom_id "
                    "for batch correlation"
                )

        seen: set[str] = set()
        inline_requests: list[dict[str, Any]] = []
        for req in applied:
            if req.custom_id in seen:
                raise ValueError(f"duplicate custom_id in batch: {req.custom_id!r}")
            seen.add(req.custom_id)
            inline_requests.append(self._build_inline_request(req))

        client = self._client()
        job = client.batches.create(
            model=self.model_config.api_model_id,
            src=inline_requests,
        )
        job_name = getattr(job, "name", None) or getattr(job, "id", None)
        if not job_name:
            raise RuntimeError("Gemini batches.create returned no job identifier")
        return BatchHandle(
            provider=self.provider,
            batch_id=str(job_name),
            model_id=self.model_config.api_model_id,
            submitted_at=datetime.now(timezone.utc).isoformat(),
        )

    def _build_inline_request(self, request: CompletionRequest) -> dict[str, Any]:
        config: dict[str, Any] = {}
        if request.thinking_level is not None:
            config["thinking_config"] = {"thinking_level": request.thinking_level}
        if request.temperature is not None:
            config["temperature"] = request.temperature
        if request.seed is not None:
            config["seed"] = request.seed
        if request.response_schema is not None:
            schema = request.response_schema
            assert isinstance(schema, type) and issubclass(schema, BaseModel), (
                "GeminiBatchClient.response_schema must be a Pydantic BaseModel "
                "subclass; raw dict schemas are silently dropped by the batch API"
            )
            config["response_mime_type"] = "application/json"
            config["response_schema"] = schema

        if request.system_prompt:
            user_text = f"{request.system_prompt}\n\n{request.user_prompt}"
        else:
            user_text = request.user_prompt

        return {
            "contents": [{"role": "user", "parts": [{"text": user_text}]}],
            "metadata": {"custom_id": request.custom_id or ""},
            "config": config,
        }

    # ------------------------------------------------------------------
    # Poll
    # ------------------------------------------------------------------

    def poll_status(self, handle: BatchHandle) -> BatchStatus:
        client = self._client()
        job = client.batches.get(name=handle.batch_id)
        state = getattr(job, "state", None) or getattr(job, "status", None) or ""
        state_str = state.name if hasattr(state, "name") else str(state)
        return _STATUS_MAP.get(state_str, BatchStatus.IN_PROGRESS)

    # ------------------------------------------------------------------
    # Fetch
    # ------------------------------------------------------------------

    def fetch_results(self, handle: BatchHandle) -> list[BatchResultItem]:
        client = self._client()
        job = client.batches.get(name=handle.batch_id)
        dest = getattr(job, "dest", None)
        inlined = getattr(dest, "inlined_responses", None) if dest else None
        items: list[BatchResultItem] = []
        if not inlined:
            return items
        for entry in inlined:
            metadata = getattr(entry, "metadata", None) or {}
            custom_id = str(metadata.get("custom_id") or getattr(entry, "key", "") or "")
            error = getattr(entry, "error", None)
            if error is not None:
                items.append(BatchResultItem(
                    custom_id=custom_id,
                    text=None,
                    finish_reason=None,
                    error_message=str(error),
                ))
                continue
            response = getattr(entry, "response", None)
            text = self._extract_text(response)
            finish_reason = self._extract_finish_reason(response)
            if text is None:
                items.append(BatchResultItem(
                    custom_id=custom_id,
                    text=None,
                    finish_reason=finish_reason,
                    error_message="response has no text candidate",
                ))
            else:
                items.append(BatchResultItem(
                    custom_id=custom_id,
                    text=text,
                    finish_reason=finish_reason,
                    error_message=None,
                ))
        return items

    @staticmethod
    def _extract_text(response: Any) -> str | None:
        if response is None:
            return None
        text_attr = getattr(response, "text", None)
        if isinstance(text_attr, str) and text_attr:
            return text_attr
        candidates = getattr(response, "candidates", None) or []
        for cand in candidates:
            content = getattr(cand, "content", None)
            parts = getattr(content, "parts", None) or []
            chunks = [getattr(p, "text", "") for p in parts]
            joined = "".join(c for c in chunks if isinstance(c, str))
            if joined:
                return joined
        return None

    @staticmethod
    def _extract_finish_reason(response: Any) -> str | None:
        if response is None:
            return None
        candidates = getattr(response, "candidates", None) or []
        if not candidates:
            return None
        reason = getattr(candidates[0], "finish_reason", None)
        if reason is None:
            return None
        return reason.name if hasattr(reason, "name") else str(reason)
