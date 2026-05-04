"""Tests for `api_clients.cache.CacheStore` round-trip."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from survey2agent.api_clients import (
    CacheStore,
    CompletionRequest,
    CompletionResult,
    compute_cache_key,
)
from survey2agent.api_clients.cache import CACHE_FILE_VERSION, CACHE_KEY_VERSION


def _result(model_id: str = "gpt-5.4", provider: str = "openai") -> CompletionResult:
    return CompletionResult(
        text="hello there",
        finish_reason=None,
        model_id=model_id,
        provider=provider,
        cache_hit=False,
    )


def test_cache_key_version_is_v2() -> None:
    assert CACHE_KEY_VERSION == "v2"
    assert CACHE_FILE_VERSION == "v2"


def test_round_trip(tmp_path: Path) -> None:
    store = CacheStore(tmp_path)
    req = CompletionRequest(user_prompt="hi")
    key = compute_cache_key(req, "gpt-5.4")

    assert store.get(key, "openai", "gpt-5.4") is None

    result = _result()
    store.put(key, result)

    fetched = store.get(key, "openai", "gpt-5.4")
    assert fetched is not None
    assert fetched.text == result.text
    assert fetched.cache_hit is True
    assert fetched.model_id == "gpt-5.4"
    assert fetched.provider == "openai"

    on_disk = json.loads(
        (tmp_path / "openai" / "gpt-5.4" / f"{key}.json").read_text(encoding="utf-8")
    )
    assert on_disk == {
        "cache_file_version": "v2",
        "model": "gpt-5.4",
        "response": "hello there",
    }


def test_record_keys_are_exactly_three(tmp_path: Path) -> None:
    """The on-disk record must contain exactly {cache_file_version, model, response}."""
    store = CacheStore(tmp_path)
    req = CompletionRequest(user_prompt="hi")
    key = compute_cache_key(req, "gpt-5.4")
    store.put(key, _result())
    on_disk = json.loads(
        (tmp_path / "openai" / "gpt-5.4" / f"{key}.json").read_text(encoding="utf-8")
    )
    assert set(on_disk.keys()) == {"cache_file_version", "model", "response"}


def test_path_sanitizes_slash_in_model_id(tmp_path: Path) -> None:
    store = CacheStore(tmp_path)
    req = CompletionRequest(user_prompt="hi")
    model_id = "deepseek/deepseek-v3.2"
    key = compute_cache_key(req, model_id)
    result = _result(model_id=model_id, provider="openrouter")
    store.put(key, result)

    assert (tmp_path / "openrouter" / "deepseek__deepseek-v3.2" / f"{key}.json").exists()
    fetched = store.get(key, "openrouter", model_id)
    assert fetched is not None
    assert fetched.model_id == model_id


def test_atomic_write_overwrite(tmp_path: Path) -> None:
    store = CacheStore(tmp_path)
    req = CompletionRequest(user_prompt="hi")
    key = compute_cache_key(req, "gpt-5.4")
    store.put(key, _result())
    store.put(key, _result())  # idempotent
    fetched = store.get(key, "openai", "gpt-5.4")
    assert fetched is not None


def test_unknown_schema_version_raises(tmp_path: Path) -> None:
    key = compute_cache_key(CompletionRequest(user_prompt="hi"), "gpt-5.4")
    path = tmp_path / "openai" / "gpt-5.4" / f"{key}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"cache_file_version": "v9", "model": "gpt-5.4", "response": "x"}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="cache_file_version"):
        CacheStore(tmp_path).get(key, "openai", "gpt-5.4")


def test_canonical_request_payload_yields_stable_key() -> None:
    req_a = CompletionRequest(user_prompt="hi", system_prompt="be brief", temperature=0.0)
    req_b = CompletionRequest(user_prompt="hi", system_prompt="be brief", temperature=0.0)
    assert compute_cache_key(req_a, "gpt-5.4") == compute_cache_key(req_b, "gpt-5.4")
