"""Tests for the Gemini Batch API client helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

try:
    from google.genai import types
except ImportError:  # pragma: no cover - optional API extra
    types = None

from survey2agent.api_clients import CacheStore, CompletionRequest, ModelConfig
from survey2agent.api_clients.gemini_batch import GeminiBatchClient


def _client(tmp_path: Path) -> GeminiBatchClient:
    return GeminiBatchClient(
        ModelConfig(
            paper_alias="Gemini",
            provider="google",
            api_model_id="gemini-test",
            api_endpoint=None,
            default_params={},
        ),
        CacheStore(tmp_path),
    )


@pytest.mark.skipif(types is None, reason="Gemini Batch API tests require google-genai.")
def test_gemini_inline_request_matches_current_sdk_schema(tmp_path: Path) -> None:
    req = CompletionRequest(
        user_prompt="answer this",
        system_prompt="be concise",
        temperature=0.0,
        custom_id="persona__A1",
    )
    payload = _client(tmp_path)._build_inline_request(req)

    parsed = types.InlinedRequest(**payload)
    assert parsed.metadata == {"custom_id": "persona__A1"}
    assert parsed.config is not None
    assert parsed.config.temperature == 0.0
