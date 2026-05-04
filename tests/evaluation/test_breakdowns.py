"""Tests for evaluation.breakdowns."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.needs_data

from survey2agent.evaluation.aggregator import aggregate_metrics
from survey2agent.evaluation.breakdowns import (
    breakdown_by,
    breakdown_by_qid_difficulty,
)
from survey2agent.evaluation.data_loaders import (
    build_training_records,
    load_atoms_for_seed,
    load_ground_truths,
    load_splits,
)
from survey2agent.evaluation.runner import EvaluationResult, run_method
from survey2agent.extraction.atoms import EXPECTED_QUESTION_IDS
from survey2agent.methods.base import Prediction
from survey2agent.methods.oracle import OracleExtraction
from survey2agent.selective.protocol import EvaluationOutcome


# ── helpers ─────────────────────────────────────────────────────────────────


def _mk(
    answer: str,
    label: str,
    *,
    skip: bool = False,
    qid: str = "A1",
    persona: str = "p",
    reasoning_type: str = "_test_type",
    topic: str = "_test_topic",
    difficulty_class: str = "_test_diff",
) -> EvaluationResult:
    if skip:
        pred = Prediction(answer="SKIP", would_skip=True, raw_answer=None)
        outcome = EvaluationOutcome.SKIP
    else:
        pred = Prediction(answer=answer, would_skip=False, raw_answer=None)
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
        reasoning_type=reasoning_type,
        topic=topic,
        difficulty_class=difficulty_class,
    )


# ── Item 4 tests ────────────────────────────────────────────────────────────


def test_breakdown_by_reasoning_type_groups_correctly():
    results = [
        _mk("X", "X", reasoning_type="arbitration", qid="A1"),
        _mk("X", "Y", reasoning_type="arbitration", qid="A1"),
        _mk("X", "X", reasoning_type="identity", qid="B1"),
    ]
    out = breakdown_by(results, key="reasoning_type")
    assert set(out) == {"arbitration", "identity"}
    assert out["arbitration"]["n"] == 2
    assert out["identity"]["n"] == 1


def test_breakdown_by_topic_groups_correctly():
    results = [
        _mk("X", "X", topic="sleep", qid="A1"),
        _mk("X", "X", topic="sleep", qid="A1"),
        _mk("X", "Y", topic="diet", qid="A2"),
    ]
    out = breakdown_by(results, key="topic")
    assert set(out) == {"sleep", "diet"}
    assert out["sleep"]["n"] == 2
    assert out["diet"]["n"] == 1
    assert out["sleep"]["forced_accuracy"] == pytest.approx(1.0)
    assert out["diet"]["forced_accuracy"] == pytest.approx(0.0)


def test_breakdown_by_difficulty_class_groups_correctly():
    results = [
        _mk("X", "X", difficulty_class="stable", qid="A1"),
        _mk("X", "Y", difficulty_class="temporal_shift", qid="A1"),
        _mk("X", "X", difficulty_class="stated_vs_revealed", qid="A1"),
    ]
    out = breakdown_by(results, key="difficulty_class")
    assert set(out) == {"stable", "temporal_shift", "stated_vs_revealed"}
    for k in out:
        assert out[k]["n"] == 1


def test_breakdown_by_invalid_key_raises():
    with pytest.raises(ValueError, match="key must be one of"):
        breakdown_by([_mk("X", "X")], key="bogus")  # type: ignore[arg-type]


def test_breakdown_by_empty_results_returns_empty_dict():
    assert breakdown_by([], key="reasoning_type") == {}
    assert breakdown_by_qid_difficulty([]) == {}


def test_breakdown_by_qid_difficulty_cross_product():
    results = [
        _mk("X", "X", qid="A1", difficulty_class="stable"),
        _mk("X", "X", qid="A1", difficulty_class="temporal_shift"),
        _mk("X", "X", qid="A2", difficulty_class="stable"),
        _mk("X", "X", qid="A2", difficulty_class="temporal_shift"),
    ]
    out = breakdown_by_qid_difficulty(results)
    assert set(out) == {"A1", "A2"}
    assert set(out["A1"]) == {"stable", "temporal_shift"}
    assert set(out["A2"]) == {"stable", "temporal_shift"}
    for qid in ("A1", "A2"):
        for d in ("stable", "temporal_shift"):
            assert out[qid][d]["n"] == 1


def test_breakdown_by_qid_difficulty_drops_empty_cells():
    # Only 2 of 4 (qid, difficulty) cells populated.
    results = [
        _mk("X", "X", qid="A1", difficulty_class="stable"),
        _mk("X", "X", qid="A2", difficulty_class="temporal_shift"),
    ]
    out = breakdown_by_qid_difficulty(results)
    assert set(out) == {"A1", "A2"}
    assert set(out["A1"]) == {"stable"}
    assert set(out["A2"]) == {"temporal_shift"}


def test_breakdown_metrics_match_subset_aggregate():
    # 3 reasoning types × 10 results each. Verify per-group metrics equal
    # an independent aggregate_metrics call on the filtered subset.
    results: list[EvaluationResult] = []
    for i in range(10):
        results.append(_mk("X", "X" if i % 2 == 0 else "Y",
                           reasoning_type="arbitration",
                           qid="A1" if i < 5 else "A2"))
    for i in range(10):
        results.append(_mk("X", "X",
                           reasoning_type="identity",
                           qid="B1" if i < 5 else "B2"))
    for i in range(10):
        results.append(_mk("", "Y", skip=True,
                           reasoning_type="control",
                           qid="Ctrl1" if i < 5 else "Ctrl2"))

    out = breakdown_by(results, key="reasoning_type")
    for rt in ("arbitration", "identity", "control"):
        subset = [r for r in results if r.reasoning_type == rt]
        expected = aggregate_metrics(subset)
        assert out[rt] == expected


def test_breakdown_macro_uses_per_qid_macro_within_group():
    # Single reasoning_type "arbitration" with two qids of unequal size:
    # A1 has 5 results all correct (sel=1.0), A2 has 10 results all wrong
    # (sel=0.0). Macro within group = mean(1.0, 0.0) = 0.5.
    # Micro within group = 5/15 ≈ 0.333.
    results = (
        [_mk("X", "X", qid="A1", reasoning_type="arbitration") for _ in range(5)]
        + [_mk("X", "Y", qid="A2", reasoning_type="arbitration") for _ in range(10)]
    )
    out = breakdown_by(results, key="reasoning_type")
    assert out["arbitration"]["selective_accuracy"] == pytest.approx(0.5)
    assert out["arbitration"]["selective_accuracy_micro"] == pytest.approx(5 / 15)
    assert out["arbitration"]["forced_accuracy"] == pytest.approx(0.5)
    assert out["arbitration"]["forced_accuracy_micro"] == pytest.approx(5 / 15)


def test_aggregate_metrics_with_breakdown_arg():
    results = [
        _mk("X", "X", reasoning_type="arbitration", qid="A1"),
        _mk("X", "Y", reasoning_type="identity", qid="B1"),
    ]
    m = aggregate_metrics(results, breakdown="reasoning_type")
    # Top-level keys preserved.
    assert m["n"] == 2
    assert "selective_accuracy" in m
    assert "f_beta" in m
    # New breakdown key.
    assert "breakdown" in m
    assert set(m["breakdown"]) == {"arbitration", "identity"}
    assert m["breakdown"]["arbitration"]["n"] == 1


def test_aggregate_metrics_with_breakdown_qid_difficulty():
    results = [
        _mk("X", "X", qid="A1", difficulty_class="stable"),
        _mk("X", "X", qid="A1", difficulty_class="temporal_shift"),
        _mk("X", "X", qid="A2", difficulty_class="stable"),
    ]
    m = aggregate_metrics(results, breakdown="qid_difficulty")
    assert "breakdown" in m
    assert set(m["breakdown"]) == {"A1", "A2"}
    assert set(m["breakdown"]["A1"]) == {"stable", "temporal_shift"}
    assert set(m["breakdown"]["A2"]) == {"stable"}


def test_aggregate_metrics_breakdown_none_omits_breakdown_key():
    results = [_mk("X", "X")]
    m = aggregate_metrics(results)
    assert "breakdown" not in m


def test_aggregate_metrics_breakdown_empty_results():
    m = aggregate_metrics([], breakdown="reasoning_type")
    assert m["n"] == 0
    assert m["breakdown"] == {}


# ── Item 4 / Test #12: end-to-end real-data spot check ──────────────────────


SEED = "s20260321"
QIDS = EXPECTED_QUESTION_IDS


@pytest.fixture(scope="module")
def oracle_results():
    splits = load_splits()
    atoms = load_atoms_for_seed(SEED, mode="oracle")
    gts = load_ground_truths(SEED)
    test = build_training_records(atoms, gts, splits["test"], qids=list(QIDS))
    return run_method(OracleExtraction(), test)


def test_breakdown_by_difficulty_class_on_real_oracle_data(oracle_results):
    out = breakdown_by(oracle_results, key="difficulty_class")
    # All three difficulty classes present with non-zero counts.
    assert set(out) == {"stable", "temporal_shift", "stated_vs_revealed"}
    for d in ("stable", "temporal_shift", "stated_vs_revealed"):
        assert out[d]["n"] > 0
    # Print spot-check.
    print()
    print(f"[per-difficulty spot-check] oracle / {SEED} / test split")
    for d in sorted(out):
        m = out[d]
        print(
            f"  {d:22s}  n={m['n']:4d}  "
            f"sel_acc={m['selective_accuracy']:.4f}  "
            f"forced={m['forced_accuracy']:.4f}  "
            f"cov={m['coverage']:.4f}"
        )
