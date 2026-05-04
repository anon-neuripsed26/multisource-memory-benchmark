"""Tests for multi_seed per-axis pooled bootstrap CI."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.needs_data

from survey2agent.evaluation import (
    CANONICAL_SEEDS,
    aggregate_pooled_with_breakdown_ci,
    aggregate_pooled_with_qid_difficulty_ci,
    evaluate_method_across_seeds,
)
from survey2agent.evaluation.runner import EvaluationResult
from survey2agent.methods.base import Prediction
from survey2agent.methods.majority_class import MajorityClass
from survey2agent.selective.protocol import EvaluationOutcome


# ── Helpers ──────────────────────────────────────────────────────────────


_CI_KEYS = {
    "point",
    "mean",
    "std",
    "ci_low",
    "ci_high",
    "n_bootstrap",
    "cluster_by",
    "n_seeds",
    "n_personas_pooled",
}


def _r(
    persona: str,
    *,
    qid: str = "A1",
    label: str = "x",
    pred: str = "x",
    reasoning_type: str = "arbitration",
    topic: str = "sleep",
    difficulty_class: str = "stable",
) -> EvaluationResult:
    p = Prediction(answer=pred, would_skip=False)
    outcome = (
        EvaluationOutcome.TRUE_ANSWER
        if pred == label
        else EvaluationOutcome.FALSE_ANSWER
    )
    return EvaluationResult(
        method_name="Mock",
        persona_id=persona,
        qid=qid,
        prediction=p,
        label=label,
        outcome=outcome,
        reasoning_type=reasoning_type,
        topic=topic,
        difficulty_class=difficulty_class,
    )


def _two_seed_two_difficulty() -> dict[str, list[EvaluationResult]]:
    """Mock: 2 seeds × 4 personas × 2 difficulty classes (stable, temporal_shift)."""
    out: dict[str, list[EvaluationResult]] = {}
    for s_idx in range(2):
        seed = CANONICAL_SEEDS[s_idx]
        recs: list[EvaluationResult] = []
        for p_idx in range(4):
            diff = "stable" if p_idx % 2 == 0 else "temporal_shift"
            label = "x" if p_idx % 2 == 0 else "y"
            recs.append(
                _r(
                    f"p_{p_idx:03d}",
                    label=label,
                    pred="x",
                    difficulty_class=diff,
                )
            )
        out[seed] = recs
    return out


# ── Item 1: aggregate_pooled_with_breakdown_ci ───────────────────────────


def test_aggregate_pooled_with_breakdown_ci_returns_per_cell_dict():
    out = aggregate_pooled_with_breakdown_ci(
        _two_seed_two_difficulty(),
        breakdown_key="difficulty_class",
        n_bootstrap=20,
    )
    assert set(out.keys()) == {"stable", "temporal_shift"}


def test_aggregate_pooled_with_breakdown_ci_each_cell_has_ci_keys():
    out = aggregate_pooled_with_breakdown_ci(
        _two_seed_two_difficulty(),
        breakdown_key="difficulty_class",
        n_bootstrap=20,
    )
    for cell, ci in out.items():
        assert _CI_KEYS.issubset(ci.keys()), f"cell {cell} missing keys"


def test_aggregate_pooled_with_breakdown_ci_invalid_key_raises():
    with pytest.raises(ValueError, match="breakdown_key"):
        aggregate_pooled_with_breakdown_ci(
            _two_seed_two_difficulty(),
            breakdown_key="bogus",  # type: ignore[arg-type]
            n_bootstrap=10,
        )


def test_aggregate_pooled_with_breakdown_ci_empty_cells_omitted():
    """Mock has only 2 of 3 difficulty classes — output has exactly 2 keys."""
    out = aggregate_pooled_with_breakdown_ci(
        _two_seed_two_difficulty(),
        breakdown_key="difficulty_class",
        n_bootstrap=10,
    )
    assert "stated_vs_revealed" not in out
    assert len(out) == 2


def test_aggregate_pooled_with_breakdown_ci_persona_count_correct():
    """Each cell has 2 personas/seed × 2 seeds = 4 unique seed-prefixed personas."""
    out = aggregate_pooled_with_breakdown_ci(
        _two_seed_two_difficulty(),
        breakdown_key="difficulty_class",
        n_bootstrap=10,
    )
    for cell, ci in out.items():
        assert ci["n_personas_pooled"] == 4, f"{cell}: {ci['n_personas_pooled']}"
        assert ci["n_seeds"] == 2


def test_aggregate_pooled_with_breakdown_ci_persona_cluster_default():
    out = aggregate_pooled_with_breakdown_ci(
        _two_seed_two_difficulty(),
        breakdown_key="difficulty_class",
        n_bootstrap=10,
    )
    for ci in out.values():
        assert ci["cluster_by"] == "persona"


def test_aggregate_pooled_with_breakdown_ci_n_bootstrap_default_2000():
    import inspect

    sig = inspect.signature(aggregate_pooled_with_breakdown_ci)
    assert sig.parameters["n_bootstrap"].default == 2000


# ── Item 2: aggregate_pooled_with_qid_difficulty_ci ──────────────────────


def test_qid_difficulty_ci_returns_nested_dict():
    """Mock with 2 qids × 2 difficulty classes."""
    seed_results: dict[str, list[EvaluationResult]] = {}
    for s_idx in range(2):
        seed = CANONICAL_SEEDS[s_idx]
        recs: list[EvaluationResult] = []
        for p_idx in range(4):
            diff = "stable" if p_idx % 2 == 0 else "temporal_shift"
            for qid in ("A1", "A2"):
                recs.append(
                    _r(f"p_{p_idx:03d}", qid=qid, difficulty_class=diff)
                )
        seed_results[seed] = recs

    out = aggregate_pooled_with_qid_difficulty_ci(seed_results, n_bootstrap=10)
    assert set(out.keys()) == {"A1", "A2"}
    for qid, inner in out.items():
        assert set(inner.keys()) == {"stable", "temporal_shift"}
        for diff, ci in inner.items():
            assert _CI_KEYS.issubset(ci.keys())
            assert ci["cluster_by"] == "persona"


def test_qid_difficulty_ci_n_bootstrap_default_2000():
    import inspect

    sig = inspect.signature(aggregate_pooled_with_qid_difficulty_ci)
    assert sig.parameters["n_bootstrap"].default == 2000


# ── Item 5.7: end-to-end real-data ───────────────────────────────────────


@pytest.mark.slow
def test_majority_class_breakdown_ci_real_data():
    """MajorityClass on 2 seeds, oracle, test split, breakdown by difficulty_class."""
    seed_results = evaluate_method_across_seeds(
        MajorityClass,
        seeds=("s20260321", "s20260322"),
        atoms_mode="oracle",
        split="test",
    )
    out = aggregate_pooled_with_breakdown_ci(
        seed_results,
        breakdown_key="difficulty_class",
        n_bootstrap=200,
    )
    assert set(out.keys()) == {"stable", "temporal_shift", "stated_vs_revealed"}
    for cell, ci in out.items():
        assert ci["n_seeds"] == 2
        # Test split has 120 personas/seed; difficulty is split-balanced
        # (40/class/seed) → 80 unique seed-prefixed personas per cell.
        assert ci["n_personas_pooled"] == 80, f"{cell}: {ci['n_personas_pooled']}"
        assert 0.0 <= ci["point"] <= 1.0
        assert ci["ci_low"] <= ci["point"] <= ci["ci_high"], (
            f"{cell}: point={ci['point']} not in [{ci['ci_low']}, {ci['ci_high']}]"
        )
        assert ci["cluster_by"] == "persona"
