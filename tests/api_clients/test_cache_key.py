"""Tests for `api_clients.cache.compute_cache_key`."""

from __future__ import annotations

from survey2agent.api_clients import CompletionRequest, compute_cache_key
from survey2agent.api_clients.cache import CACHE_KEY_VERSION


def test_key_is_deterministic() -> None:
    req = CompletionRequest(user_prompt="hello", system_prompt="be brief", temperature=0.0)
    k1 = compute_cache_key(req, "gpt-5.4")
    k2 = compute_cache_key(req, "gpt-5.4")
    assert k1 == k2
    assert len(k1) == 64
    assert all(c in "0123456789abcdef" for c in k1)


def test_key_differs_on_prompt() -> None:
    a = CompletionRequest(user_prompt="hello")
    b = CompletionRequest(user_prompt="hello!")
    assert compute_cache_key(a, "m") != compute_cache_key(b, "m")


def test_key_differs_on_model_id() -> None:
    req = CompletionRequest(user_prompt="hello")
    assert compute_cache_key(req, "model-a") != compute_cache_key(req, "model-b")


def test_key_differs_on_provider_specific_field() -> None:
    a = CompletionRequest(user_prompt="x", reasoning_effort="high")
    b = CompletionRequest(user_prompt="x", reasoning_effort="xhigh")
    assert compute_cache_key(a, "m") != compute_cache_key(b, "m")


def test_key_differs_on_thinking_level() -> None:
    a = CompletionRequest(user_prompt="x", thinking_level="low")
    b = CompletionRequest(user_prompt="x", thinking_level="high")
    assert compute_cache_key(a, "m") != compute_cache_key(b, "m")


def test_key_independent_of_custom_id() -> None:
    """custom_id is a correlation token; it does not influence model output."""
    a = CompletionRequest(user_prompt="x", custom_id="alpha")
    b = CompletionRequest(user_prompt="x", custom_id="beta")
    assert compute_cache_key(a, "m") == compute_cache_key(b, "m")


def test_key_version_constant() -> None:
    assert CACHE_KEY_VERSION == "v2"
