"""OpenAI Batch API client (gpt-5.4 reasoning model).

Lifecycle:
    1. `submit_batch(requests)` — write a JSONL of chat-completions requests,
       upload via the Files API, create a batch via `client.batches.create()`,
       and return a `BatchHandle` with the batch id.
    2. `poll_status(handle)` — query batch status, mapped to `BatchStatus`.
    3. `fetch_results(handle)` — download the output (and error) files,
       parse one `BatchResultItem` per `custom_id`.

Per-item failures (parse error, content filter, etc.) surface as a
`BatchResultItem` with `text=None` and a non-empty `error_message`. The
client never raises for partial failures.
"""

from __future__ import annotations

import io
import json
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
    "validating": BatchStatus.PENDING,
    "in_progress": BatchStatus.IN_PROGRESS,
    "finalizing": BatchStatus.IN_PROGRESS,
    "completed": BatchStatus.COMPLETED,
    "failed": BatchStatus.FAILED,
    "expired": BatchStatus.EXPIRED,
    "cancelled": BatchStatus.CANCELLED,
    "cancelling": BatchStatus.CANCELLED,
}


class OpenAIBatchClient(BatchLLMClient):
    provider: ClassVar[str] = "openai"
    allowed_request_fields: ClassVar[frozenset[str]] = frozenset(
        {
            "user_prompt",
            "system_prompt",
            "reasoning_effort",
            "response_schema",
            "custom_id",
        }
    )

    def _client(self) -> Any:
        from openai import OpenAI  # type: ignore[import-not-found]

        kwargs: dict[str, Any] = {}
        base_url = self._normalise_base_url(self.model_config.api_endpoint)
        if base_url:
            kwargs["base_url"] = base_url
        return OpenAI(**kwargs)

    @staticmethod
    def _normalise_base_url(api_endpoint: str | None) -> str | None:
        """Convert a chat-completions endpoint into an OpenAI SDK API root.

        ``configs/models.yaml`` records the exact chat endpoint used in paper
        tables, but the OpenAI SDK client needs the API root because Files and
        Batches live beside ``/chat/completions``.
        """

        if not api_endpoint:
            return None
        endpoint = api_endpoint.rstrip("/")
        suffix = "/chat/completions"
        if endpoint.endswith(suffix):
            endpoint = endpoint[: -len(suffix)]
        return endpoint or None

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
                    "OpenAIBatchClient requires CompletionRequest.custom_id "
                    "for batch correlation"
                )

        seen: set[str] = set()
        lines: list[str] = []
        for req in applied:
            if req.custom_id in seen:
                raise ValueError(f"duplicate custom_id in batch: {req.custom_id!r}")
            seen.add(req.custom_id)
            lines.append(json.dumps(self._build_jsonl_line(req), ensure_ascii=False))
        payload_bytes = ("\n".join(lines) + "\n").encode("utf-8")

        client = self._client()
        upload = client.files.create(
            file=("batch_input.jsonl", io.BytesIO(payload_bytes)),
            purpose="batch",
        )
        batch = client.batches.create(
            input_file_id=upload.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
        )
        return BatchHandle(
            provider=self.provider,
            batch_id=batch.id,
            model_id=self.model_config.api_model_id,
            submitted_at=datetime.now(timezone.utc).isoformat(),
            provider_metadata={"input_file_id": upload.id},
        )

    def _build_jsonl_line(self, request: CompletionRequest) -> dict[str, Any]:
        messages: list[dict[str, str]] = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        messages.append({"role": "user", "content": request.user_prompt})

        body: dict[str, Any] = {
            "model": self.model_config.api_model_id,
            "messages": messages,
        }
        if request.reasoning_effort is not None:
            body["reasoning_effort"] = request.reasoning_effort
        if request.response_schema is not None:
            schema = request.response_schema
            if not (isinstance(schema, type) and issubclass(schema, BaseModel)):
                raise TypeError(
                    "OpenAIBatchClient.response_schema must be a Pydantic BaseModel "
                    f"subclass, got {type(schema).__name__}"
                )
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": schema.__name__,
                    "schema": schema.model_json_schema(),
                    "strict": True,
                },
            }
        return {
            "custom_id": request.custom_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": body,
        }

    # ------------------------------------------------------------------
    # Poll
    # ------------------------------------------------------------------

    def poll_status(self, handle: BatchHandle) -> BatchStatus:
        client = self._client()
        batch = client.batches.retrieve(handle.batch_id)
        status_str = getattr(batch, "status", None) or ""
        return _STATUS_MAP.get(status_str, BatchStatus.IN_PROGRESS)

    # ------------------------------------------------------------------
    # Fetch
    # ------------------------------------------------------------------

    def fetch_results(self, handle: BatchHandle) -> list[BatchResultItem]:
        client = self._client()
        batch = client.batches.retrieve(handle.batch_id)
        items_by_id: dict[str, BatchResultItem] = {}

        output_file_id = getattr(batch, "output_file_id", None)
        if output_file_id:
            for raw_line in self._read_file_lines(client, output_file_id):
                item = self._parse_output_line(raw_line)
                if item is not None:
                    items_by_id[item.custom_id] = item

        error_file_id = getattr(batch, "error_file_id", None)
        if error_file_id:
            for raw_line in self._read_file_lines(client, error_file_id):
                item = self._parse_error_line(raw_line)
                if item is not None and item.custom_id not in items_by_id:
                    items_by_id[item.custom_id] = item

        return list(items_by_id.values())

    @staticmethod
    def _read_file_lines(client: Any, file_id: str) -> list[str]:
        response = client.files.content(file_id)
        text = response.text if hasattr(response, "text") else response.read().decode("utf-8")
        return [line for line in text.splitlines() if line.strip()]

    @staticmethod
    def _parse_output_line(raw: str) -> BatchResultItem | None:
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            return BatchResultItem(
                custom_id="", text=None, finish_reason=None,
                error_message=f"jsonl parse error: {exc}",
            )
        custom_id = str(payload.get("custom_id") or "")
        error = payload.get("error")
        if error:
            return BatchResultItem(
                custom_id=custom_id, text=None, finish_reason=None,
                error_message=str(error),
            )
        response_obj = payload.get("response") or {}
        body = response_obj.get("body") or {}
        choices = body.get("choices") or []
        if not choices:
            return BatchResultItem(
                custom_id=custom_id, text=None, finish_reason=None,
                error_message="response body has no choices",
            )
        first = choices[0]
        message = first.get("message") or {}
        text = message.get("content")
        finish_reason = first.get("finish_reason")
        if text is None:
            return BatchResultItem(
                custom_id=custom_id, text=None, finish_reason=finish_reason,
                error_message="response choice has no message.content",
            )
        return BatchResultItem(
            custom_id=custom_id,
            text=str(text),
            finish_reason=str(finish_reason) if finish_reason is not None else None,
            error_message=None,
        )

    @staticmethod
    def _parse_error_line(raw: str) -> BatchResultItem | None:
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            return BatchResultItem(
                custom_id="", text=None, finish_reason=None,
                error_message=f"jsonl parse error: {exc}",
            )
        custom_id = str(payload.get("custom_id") or "")
        error = payload.get("error") or payload.get("response") or payload
        return BatchResultItem(
            custom_id=custom_id,
            text=None,
            finish_reason=None,
            error_message=json.dumps(error, ensure_ascii=False),
        )
