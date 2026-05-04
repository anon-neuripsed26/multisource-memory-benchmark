"""Tests for `Prediction.raw_answer`, `RawLLMOutput`, and `normalize_to_prediction`."""

from __future__ import annotations

import pytest

from survey2agent.methods import (
    SKIP_SENTINEL,
    Prediction,
    RawLLMOutput,
    normalize_to_prediction,
)


# ── Prediction.raw_answer invariants ───────────────────────────────────────


def test_prediction_default_raw_answer_is_none() -> None:
    p = Prediction(answer="X", would_skip=False)
    assert p.raw_answer is None


def test_prediction_skip_default_raw_answer_is_none() -> None:
    p = Prediction(answer=SKIP_SENTINEL, would_skip=True)
    assert p.raw_answer is None


def test_prediction_consistent_raw_answer_ok() -> None:
    p = Prediction(answer="X", would_skip=False, raw_answer="X")
    assert p.raw_answer == "X"


def test_prediction_skip_with_underlying_label_ok() -> None:
    p = Prediction(answer=SKIP_SENTINEL, would_skip=True, raw_answer="X")
    assert p.answer == SKIP_SENTINEL
    assert p.would_skip is True
    assert p.raw_answer == "X"


def test_prediction_inconsistent_raw_answer_raises() -> None:
    with pytest.raises(ValueError, match="raw_answer must equal answer"):
        Prediction(answer="X", would_skip=False, raw_answer="Y")


def test_prediction_raw_answer_skip_sentinel_raises() -> None:
    with pytest.raises(ValueError, match="raw_answer must not be SKIP_SENTINEL"):
        Prediction(answer="X", would_skip=False, raw_answer=SKIP_SENTINEL)


def test_prediction_skip_with_skip_raw_raises() -> None:
    with pytest.raises(ValueError, match="raw_answer must not be SKIP_SENTINEL"):
        Prediction(answer=SKIP_SENTINEL, would_skip=True, raw_answer=SKIP_SENTINEL)


# ── RawLLMOutput ───────────────────────────────────────────────────────────


def test_raw_llm_output_round_trip() -> None:
    r = RawLLMOutput(answer="X", would_skip=False)
    assert r.answer == "X"
    assert r.would_skip is False
    r2 = RawLLMOutput(answer="Y", would_skip=True)
    assert r2.answer == "Y"
    assert r2.would_skip is True


# ── normalize_to_prediction ────────────────────────────────────────────────


def test_normalize_skip_preserves_raw_label() -> None:
    p = normalize_to_prediction(RawLLMOutput("X", would_skip=True), "A1")
    assert p.answer == SKIP_SENTINEL
    assert p.would_skip is True
    assert p.raw_answer == "X"


def test_normalize_non_skip_sets_raw_equal_to_answer() -> None:
    p = normalize_to_prediction(RawLLMOutput("X", would_skip=False), "A1")
    assert p.answer == "X"
    assert p.would_skip is False
    assert p.raw_answer == "X"
