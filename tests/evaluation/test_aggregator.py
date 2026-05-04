"""Tests for evaluation.aggregator (macro default + micro siblings)."""

from __future__ import annotations

import pytest

from survey2agent.evaluation.aggregator import aggregate_metrics
from survey2agent.evaluation.runner import EvaluationResult
from survey2agent.methods.base import Prediction
from survey2agent.selective.protocol import EvaluationOutcome


def _mk(
    answer: str,
    label: str,
    *,
    skip: bool = False,
    raw: str | None = None,
    qid: str = "A1",
    persona: str = "p",
) -> EvaluationResult:
    if skip:
        pred = Prediction(answer="SKIP", would_skip=True, raw_answer=raw)
        outcome = EvaluationOutcome.SKIP
    else:
        pred = Prediction(answer=answer, would_skip=False, raw_answer=raw)
        outcome = (
            EvaluationOutcome.TRUE_ANSWER
            if answer == label
            else EvaluationOutcome.FALSE_ANSWER
        )
    return EvaluationResult(
        method_name="dummy",
        persona_id=persona,
        qid=qid,
        prediction=pred,
        label=label,
        outcome=outcome,
        reasoning_type="_test_type",
        topic="_test_topic",
        difficulty_class="_test_diff",
    )


def test_aggregate_three_correct_one_wrong_single_qid():
    # Single qid → macro == micro for sel/forced
    results = [
        _mk("X", "X"),
        _mk("X", "X"),
        _mk("X", "X"),
        _mk("X", "Y"),
    ]
    m = aggregate_metrics(results)
    assert m["n"] == 4
    assert m["coverage"] == 1.0
    assert m["selective_accuracy"] == 0.75
    assert m["selective_accuracy_micro"] == 0.75
    assert m["forced_accuracy"] == 0.75
    assert m["forced_accuracy_micro"] == 0.75
    assert m["n_correct"] == 3
    assert m["n_wrong"] == 1
    assert m["n_skip"] == 0
    # F0.5 macro: P=0.75, cov=1.0, forced=0.75 → R = 0.75*1/0.75 = 1.0
    # → F0.5 = 1.25*0.75*1 / (0.25*0.75 + 1) ≈ 0.7895
    assert m["f_beta"] == pytest.approx(0.7894736842105263, abs=1e-6)
    # F0.5 micro: TP=3, FP=1, FN=0 → P=0.75, R=1.0 → same value
    assert m["f_beta_micro"] == pytest.approx(0.7894736842105263, abs=1e-6)


def test_aggregate_macro_differs_from_micro_across_qids():
    """Two qids with different sample sizes → macro and micro diverge."""
    # Q1: 4 records, all correct (acc 1.0)
    # Q2: 2 records, 0 correct (acc 0.0)
    # micro acc = 4/6 = 0.667; macro acc = mean(1.0, 0.0) = 0.5
    results = (
        [_mk("X", "X", qid="Q1") for _ in range(4)]
        + [_mk("X", "Y", qid="Q2") for _ in range(2)]
    )
    m = aggregate_metrics(results)
    assert m["selective_accuracy_micro"] == pytest.approx(4 / 6)
    assert m["selective_accuracy"] == pytest.approx(0.5)
    assert m["forced_accuracy_micro"] == pytest.approx(4 / 6)
    assert m["forced_accuracy"] == pytest.approx(0.5)


def test_aggregate_macro_drops_all_skip_qid():
    """A qid where every record is SKIP must be excluded from macro sel."""
    # Q1: 2 answered, 2 correct (acc 1.0)
    # Q2: 2 SKIP → dropped
    results = [
        _mk("X", "X", qid="Q1"),
        _mk("X", "X", qid="Q1"),
        _mk("", "Y", skip=True, qid="Q2"),
        _mk("", "Y", skip=True, qid="Q2"),
    ]
    m = aggregate_metrics(results)
    assert m["selective_accuracy"] == pytest.approx(1.0)  # only Q1 counted
    # micro: 2 answered, 2 correct → 1.0 (same here by coincidence)
    assert m["selective_accuracy_micro"] == pytest.approx(1.0)
    assert m["coverage"] == 0.5


def test_aggregate_one_correct_three_skip():
    results = [
        _mk("X", "X"),
        _mk("", "X", skip=True, raw="X"),
        _mk("", "Y", skip=True, raw="Y"),
        _mk("", "Z", skip=True, raw="Z"),
    ]
    m = aggregate_metrics(results)
    assert m["n"] == 4
    assert m["coverage"] == 0.25
    assert m["selective_accuracy"] == 1.0
    # forced uses raw_answer when present: all four match.
    assert m["forced_accuracy"] == 1.0
    assert m["forced_accuracy_micro"] == 1.0
    assert m["n_correct"] == 1
    assert m["n_wrong"] == 0
    assert m["n_skip"] == 3


def test_aggregate_all_skip():
    results = [_mk("", "X", skip=True) for _ in range(5)]
    m = aggregate_metrics(results)
    assert m["n"] == 5
    assert m["coverage"] == 0.0
    assert m["selective_accuracy"] == 0.0
    assert m["selective_accuracy_micro"] == 0.0
    assert m["n_skip"] == 5
    assert m["f_beta"] == 0.0
    assert m["f_beta_micro"] == 0.0


def test_aggregate_empty():
    m = aggregate_metrics([])
    expected_keys = {
        "n", "coverage",
        "selective_accuracy", "selective_accuracy_micro",
        "forced_accuracy", "forced_accuracy_micro",
        "f_beta", "f_beta_micro",
        "n_correct", "n_wrong", "n_skip",
    }
    assert set(m.keys()) == expected_keys
    for k, v in m.items():
        assert v == 0 or v == 0.0


def test_aggregate_metrics_macro_default_keys_present():
    """Output dict must carry both macro and micro siblings."""
    results = [_mk("X", "X")]
    m = aggregate_metrics(results)
    for k in (
        "selective_accuracy", "selective_accuracy_micro",
        "forced_accuracy",   "forced_accuracy_micro",
        "f_beta",            "f_beta_micro",
        "coverage",
    ):
        assert k in m, f"missing key {k!r}"
