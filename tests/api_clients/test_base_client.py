"""Tests for `SyncLLMClient` and `BatchLLMClient` orchestration."""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

import pytest

from survey2agent.api_clients import (
    BatchHandle,
    BatchLLMClient,
    BatchResultItem,
    BatchStatus,
    CacheMissError,
    CacheStore,
    CompletionRequest,
    ModelConfig,
    SyncLLMClient,
)


def _mc(**overrides: Any) -> ModelConfig:
    base: dict[str, Any] = dict(
        paper_alias="Mock-1",
        provider="mock",
        api_model_id="mock-1",
        api_endpoint=None,
        default_params={},
    )
    base.update(overrides)
    return ModelConfig(**base)


# ---------------------------------------------------------------------------
# Sync client
# ---------------------------------------------------------------------------

class _MockSync(SyncLLMClient):
    provider: ClassVar[str] = "mock"
    allowed_request_fields: ClassVar[frozenset[str]] = frozenset(
        {"user_prompt", "system_prompt", "temperature",
         "reasoning_effort", "top_p", "seed"}
    )

    def __init__(self, model_config: ModelConfig, cache_store: CacheStore) -> None:
        super().__init__(model_config, cache_store)
        self.api_calls = 0

    def _raw_complete(self, request: CompletionRequest) -> str:
        self.api_calls += 1
        return f"echo:{request.user_prompt}"


def test_sync_cache_miss_without_allow_raises(tmp_path: Path) -> None:
    client = _MockSync(_mc(), CacheStore(tmp_path))
    with pytest.raises(CacheMissError):
        client.complete(CompletionRequest(user_prompt="hi"))
    assert client.api_calls == 0


def test_sync_cache_miss_with_allow_calls_api_once(tmp_path: Path) -> None:
    client = _MockSync(_mc(), CacheStore(tmp_path))
    result = client.complete(CompletionRequest(user_prompt="hi"), allow_api_call=True)
    assert client.api_calls == 1
    assert result.text == "echo:hi"
    assert result.cache_hit is False
    assert result.provider == "mock"
    assert result.model_id == "mock-1"


def test_sync_second_call_hits_cache(tmp_path: Path) -> None:
    client = _MockSync(_mc(), CacheStore(tmp_path))
    client.complete(CompletionRequest(user_prompt="x"), allow_api_call=True)
    result2 = client.complete(CompletionRequest(user_prompt="x"))
    assert client.api_calls == 1
    assert result2.cache_hit is True
    assert result2.text == "echo:x"


def test_sync_default_params_fill_unset_fields(tmp_path: Path) -> None:
    cfg = _mc(default_params={"reasoning_effort": "xhigh", "temperature": 0.7})
    client = _MockSync(cfg, CacheStore(tmp_path))
    req = CompletionRequest(user_prompt="x")  # both unset
    client.complete(req, allow_api_call=True)
    # Re-issuing the unset-field request should hit the cache (defaults applied).
    client.complete(req)
    assert client.api_calls == 1

    req2 = CompletionRequest(user_prompt="x", reasoning_effort="low")
    with pytest.raises(CacheMissError):
        client.complete(req2)


def test_sync_disallowed_field_rejected(tmp_path: Path) -> None:
    client = _MockSync(_mc(), CacheStore(tmp_path))
    with pytest.raises(ValueError, match="thinking_level"):
        client.complete(
            CompletionRequest(user_prompt="x", thinking_level="high"),
            allow_api_call=True,
        )


def test_sync_wrong_provider_raises(tmp_path: Path) -> None:
    cfg = _mc(provider="openai")
    with pytest.raises(ValueError, match="provider"):
        _MockSync(cfg, CacheStore(tmp_path))


# ---------------------------------------------------------------------------
# Batch client
# ---------------------------------------------------------------------------

class _MockBatch(BatchLLMClient):
    provider: ClassVar[str] = "mock"
    allowed_request_fields: ClassVar[frozenset[str]] = frozenset(
        {"user_prompt", "system_prompt", "temperature", "custom_id"}
    )

    def __init__(self, model_config: ModelConfig, cache_store: CacheStore) -> None:
        super().__init__(model_config, cache_store)
        self.submitted: list[CompletionRequest] = []
        self.poll_count = 0

    def submit_batch(self, requests: list[CompletionRequest]) -> BatchHandle:
        self.submitted = list(requests)
        return BatchHandle(
            provider=self.provider,
            batch_id="batch-mock-1",
            model_id=self.model_config.api_model_id,
        )

    def poll_status(self, handle: BatchHandle) -> BatchStatus:
        self.poll_count += 1
        return BatchStatus.COMPLETED

    def fetch_results(self, handle: BatchHandle) -> list[BatchResultItem]:
        return [
            BatchResultItem(
                custom_id=req.custom_id or "",
                text=f"echo:{req.user_prompt}",
                finish_reason="stop",
                error_message=None,
            )
            for req in self.submitted
        ]


def test_batch_run_blocking_returns_per_item(tmp_path: Path) -> None:
    client = _MockBatch(_mc(), CacheStore(tmp_path))
    requests = [
        CompletionRequest(user_prompt="a", custom_id="A"),
        CompletionRequest(user_prompt="b", custom_id="B"),
    ]
    items = client.run_batch_blocking(requests, poll_every_s=0)
    assert [item.custom_id for item in items] == ["A", "B"]
    assert items[0].text == "echo:a"
    assert items[0].error_message is None
    assert client.poll_count == 1


def test_batch_empty_returns_empty(tmp_path: Path) -> None:
    client = _MockBatch(_mc(), CacheStore(tmp_path))
    assert client.run_batch_blocking([]) == []


def test_batch_disallowed_field_rejected(tmp_path: Path) -> None:
    client = _MockBatch(_mc(), CacheStore(tmp_path))
    with pytest.raises(ValueError, match="reasoning_effort"):
        client.run_batch_blocking(
            [CompletionRequest(user_prompt="x", custom_id="A", reasoning_effort="high")],
            poll_every_s=0,
        )


def test_batch_handle_round_trip() -> None:
    handle = BatchHandle(
        provider="openai",
        batch_id="batch_abc",
        model_id="gpt-5.4",
        submitted_at="2026-04-23T00:00:00+00:00",
        provider_metadata={"input_file_id": "file_xyz"},
    )
    restored = BatchHandle.from_json(handle.to_json())
    assert restored == handle
