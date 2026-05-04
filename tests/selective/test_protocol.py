"""Tests for selective.protocol."""

from __future__ import annotations

import json

import pytest

from survey2agent.methods.base import SKIP_SENTINEL, Prediction
from survey2agent.selective.protocol import (
    EvaluationOutcome,
    classify_prediction,
    reduce_outcomes,
)


def test_classify_true_answer():
    pred = Prediction(answer="X", would_skip=False)
    assert classify_prediction(pred, "X") == EvaluationOutcome.TRUE_ANSWER


def test_classify_false_answer():
    pred = Prediction(answer="X", would_skip=False)
    assert classify_prediction(pred, "Y") == EvaluationOutcome.FALSE_ANSWER


def test_classify_skip():
    pred = Prediction(answer=SKIP_SENTINEL, would_skip=True)
    assert classify_prediction(pred, "X") == EvaluationOutcome.SKIP


def test_classify_skip_with_raw_answer_still_skip():
    pred = Prediction(answer=SKIP_SENTINEL, would_skip=True, raw_answer="X")
    assert classify_prediction(pred, "X") == EvaluationOutcome.SKIP


def test_classify_invalid_label_raises():
    pred = Prediction(answer="X", would_skip=False)
    with pytest.raises(ValueError):
        classify_prediction(pred, SKIP_SENTINEL)


def test_reduce_outcomes_counts():
    outcomes = [
        EvaluationOutcome.TRUE_ANSWER,
        EvaluationOutcome.TRUE_ANSWER,
        EvaluationOutcome.FALSE_ANSWER,
        EvaluationOutcome.SKIP,
        EvaluationOutcome.SKIP,
    ]
    assert reduce_outcomes(outcomes) == {
        "true_answer": 2,
        "false_answer": 1,
        "skip": 2,
    }


def test_reduce_outcomes_empty_returns_zeros():
    assert reduce_outcomes([]) == {"true_answer": 0, "false_answer": 0, "skip": 0}


def test_evaluation_outcome_is_str_enum_json_serializable():
    assert isinstance(EvaluationOutcome.TRUE_ANSWER, str)
    assert EvaluationOutcome.TRUE_ANSWER == "true_answer"
    payload = {"outcome": EvaluationOutcome.SKIP}
    assert json.dumps(payload) == '{"outcome": "skip"}'
