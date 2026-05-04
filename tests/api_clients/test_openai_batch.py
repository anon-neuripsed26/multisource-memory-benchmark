"""Tests for the OpenAI Batch API client helpers."""

from __future__ import annotations

from survey2agent.api_clients.openai_batch import OpenAIBatchClient


def test_openai_chat_endpoint_is_normalised_to_api_root() -> None:
    assert (
        OpenAIBatchClient._normalise_base_url(
            "https://api.openai.com/v1/chat/completions"
        )
        == "https://api.openai.com/v1"
    )


def test_openai_base_url_without_chat_suffix_is_preserved() -> None:
    assert (
        OpenAIBatchClient._normalise_base_url("https://example.test/openai/v1")
        == "https://example.test/openai/v1"
    )


def test_openai_missing_endpoint_uses_sdk_default() -> None:
    assert OpenAIBatchClient._normalise_base_url(None) is None
