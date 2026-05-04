"""Tests for evaluation.multi_seed."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.needs_data

from survey2agent.evaluation import (
    CANONICAL_SEEDS,
    aggregate_per_seed_then_average,
    aggregate_per_seed_then_average_with_breakdown,
    aggregate_pooled_with_ci,
    evaluate_method_across_seeds,
    pool_results_across_seeds,
)
from survey2agent.evaluation.runner import EvaluationResult
from survey2agent.methods.base import Prediction
from survey2agent.methods.majority_class import MajorityClass
from survey2agent.methods.random_baseline import Random
from survey2agent.selective.protocol import EvaluationOutcome


# ── Helpers ──────────────────────────────────────────────────────────────


def _mock_result(persona: str, qid: str = "A1", label: str = "x") -> EvaluationResult:
    pred = Prediction(answer=label, would_skip=False)
    return EvaluationResult(
        method_name="MockMethod",
        persona_id=persona,
        qid=qid,
        prediction=pred,
        label=label,
        outcome=EvaluationOutcome.TRUE_ANSWER,
        reasoning_type="arbitration",
        topic="sleep",
        difficulty_class="stable",
    )


# ── Pure-function tests (no I/O) ─────────────────────────────────────────


def test_canonical_seeds_constant_has_4_entries():
    assert isinstance(CANONICAL_SEEDS, tuple)
    assert len(CANONICAL_SEEDS) == 4
    assert CANONICAL_SEEDS == (
        "s20260321",
        "s20260322",
        "s20260323",
        "s20260324",
    )


def test_pool_results_across_seeds_prefixes_persona_id():
    seed_results = {
        "s20260321": [_mock_result("p_001")],
        "s20260322": [_mock_result("p_001")],
    }
    pooled = pool_results_across_seeds(seed_results)
    pids = {r.persona_id for r in pooled}
    assert pids == {"s20260321_p_001", "s20260322_p_001"}


def test_pool_results_preserves_other_fields():
    src = _mock_result("p_042", qid="C2", label="yes")
    pooled = pool_results_across_seeds({"s20260323": [src]})
    assert len(pooled) == 1
    r = pooled[0]
    assert r.persona_id == "s20260323_p_042"
    assert r.qid == "C2"
    assert r.label == "yes"
    assert r.prediction is src.prediction
    assert r.outcome is EvaluationOutcome.TRUE_ANSWER
    assert r.reasoning_type == "arbitration"
    assert r.topic == "sleep"
    assert r.difficulty_class == "stable"
    assert r.method_name == "MockMethod"


def test_pool_results_concatenates_lengths():
    a = [_mock_result(f"p_{i:03d}") for i in range(5)]
    b = [_mock_result(f"p_{i:03d}") for i in range(7)]
    pooled = pool_results_across_seeds({"s20260321": a, "s20260322": b})
    assert len(pooled) == 12


def test_pool_results_empty_input():
    assert pool_results_across_seeds({}) == []


# ── evaluate_method_across_seeds (real-data, oracle mode) ────────────────


def test_evaluate_method_across_seeds_runs_per_seed_independently():
    out = evaluate_method_across_seeds(
        lambda: Random(seed=42),
        seeds=("s20260321", "s20260322"),
        atoms_mode="oracle",
        split="test",
    )
    assert set(out.keys()) == {"s20260321", "s20260322"}
    # 120 personas test-split × 18 qids = 2160 per seed.
    for seed, results in out.items():
        assert len(results) == 2160, f"seed {seed}: got {len(results)}"
        assert all(isinstance(r, EvaluationResult) for r in results)


def test_evaluate_method_across_seeds_invalid_seed_raises():
    with pytest.raises(ValueError, match="unknown seed"):
        evaluate_method_across_seeds(
            lambda: Random(seed=0),
            seeds=("s99999999",),
            atoms_mode="oracle",
        )


def test_evaluate_method_across_seeds_empty_seeds_raises():
    with pytest.raises(ValueError, match="non-empty"):
        evaluate_method_across_seeds(
            lambda: Random(seed=0),
            seeds=(),
            atoms_mode="oracle",
        )


# ── aggregate_pooled_with_ci (mock data) ─────────────────────────────────


def _mock_seed_results(n_seeds: int, n_personas: int) -> dict[str, list[EvaluationResult]]:
    out: dict[str, list[EvaluationResult]] = {}
    for s_idx in range(n_seeds):
        seed = CANONICAL_SEEDS[s_idx]
        # Mix of correct / incorrect to give a non-trivial f_beta.
        recs = []
        for p_idx in range(n_personas):
            pid = f"p_{p_idx:03d}"
            label = "x" if p_idx % 2 == 0 else "y"
            pred_value = "x"  # always "x" → ~50% correct
            pred = Prediction(answer=pred_value, would_skip=False)
            outcome = (
                EvaluationOutcome.TRUE_ANSWER
                if pred_value == label
                else EvaluationOutcome.FALSE_ANSWER
            )
            recs.append(
                EvaluationResult(
                    method_name="Mock",
                    persona_id=pid,
                    qid="A1",
                    prediction=pred,
                    label=label,
                    outcome=outcome,
                    reasoning_type="arbitration",
                    topic="sleep",
                    difficulty_class="stable",
                )
            )
        out[seed] = recs
    return out


def test_aggregate_pooled_with_ci_returns_expected_keys():
    seed_results = _mock_seed_results(n_seeds=2, n_personas=8)
    out = aggregate_pooled_with_ci(seed_results, n_bootstrap=50)
    expected = {
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
    assert expected.issubset(out.keys())


def test_aggregate_pooled_ci_uses_persona_cluster():
    seed_results = _mock_seed_results(n_seeds=2, n_personas=4)
    out = aggregate_pooled_with_ci(seed_results, n_bootstrap=20)
    assert out["cluster_by"] == "persona"


def test_aggregate_pooled_personas_count_correct():
    seed_results = _mock_seed_results(n_seeds=2, n_personas=3)
    out = aggregate_pooled_with_ci(seed_results, n_bootstrap=20)
    assert out["n_seeds"] == 2
    assert out["n_personas_pooled"] == 6  # 3 unique personas × 2 seeds (prefixed)


def test_aggregate_pooled_n_bootstrap_default_2000():
    """Default matches the paper's B = 2,000."""
    import inspect

    sig = inspect.signature(aggregate_pooled_with_ci)
    assert sig.parameters["n_bootstrap"].default == 2000


# ── End-to-end real-data smoke test ──────────────────────────────────────


@pytest.mark.slow
def test_majority_class_pooled_2_seeds_real_data():
    """End-to-end: MajorityClass on 2 seeds, oracle mode, test split."""
    seed_results = evaluate_method_across_seeds(
        MajorityClass,
        seeds=("s20260321", "s20260322"),
        atoms_mode="oracle",
        split="test",
    )
    out = aggregate_pooled_with_ci(seed_results, n_bootstrap=200)
    assert out["n_seeds"] == 2
    # 120 test personas per seed × 2 seeds = 240 unique seed-prefixed personas.
    assert out["n_personas_pooled"] == 240
    assert 0.0 <= out["point"] <= 1.0
    assert out["ci_low"] <= out["point"] <= out["ci_high"]
    assert out["cluster_by"] == "persona"


# ── aggregate_per_seed_then_average ─────────────────────────────


def _seed_results_with_target_sel_acc(target_correct_frac: float, n: int = 10, seed_name: str = "s20260321") -> list[EvaluationResult]:
    """Build a list of EvaluationResults with a controlled correct fraction.

    Uses a single qid so selective_accuracy_macro == correct_frac (ignoring
    SKIPs, which we don't emit). All predictions are non-skip.
    """
    n_correct = int(round(target_correct_frac * n))
    out: list[EvaluationResult] = []
    for i in range(n):
        is_correct = i < n_correct
        label = "x"
        pred_value = "x" if is_correct else "y"
        pred = Prediction(answer=pred_value, would_skip=False)
        outcome = (
            EvaluationOutcome.TRUE_ANSWER
            if is_correct
            else EvaluationOutcome.FALSE_ANSWER
        )
        out.append(
            EvaluationResult(
                method_name="Mock",
                persona_id=f"{seed_name}_p_{i:03d}",
                qid="A1",
                prediction=pred,
                label=label,
                outcome=outcome,
                reasoning_type="arbitration",
                topic="sleep",
                difficulty_class="stable",
            )
        )
    return out


def test_aggregate_per_seed_then_average_returns_arithmetic_mean():
    """3 seeds with sel_acc 0.5/0.7/0.9 → mean 0.7 (paper 2x2 Factorial Decomposition reduction)."""
    seed_results = {
        "s20260321": _seed_results_with_target_sel_acc(0.5, n=10),
        "s20260322": _seed_results_with_target_sel_acc(0.7, n=10),
        "s20260323": _seed_results_with_target_sel_acc(0.9, n=10),
    }
    result = aggregate_per_seed_then_average(
        seed_results, metric="selective_accuracy"
    )
    assert result == pytest.approx(0.7, abs=1e-9)


def test_aggregate_per_seed_then_average_metric_dispatch():
    """Helper accepts every standard aggregate_metrics key."""
    seed_results = {
        "s20260321": _seed_results_with_target_sel_acc(0.5, n=10),
        "s20260322": _seed_results_with_target_sel_acc(0.7, n=10),
    }
    for metric in (
        "selective_accuracy",
        "coverage",
        "forced_accuracy",
        "f_beta",
        "selective_accuracy_micro",
        "f_beta_micro",
    ):
        value = aggregate_per_seed_then_average(seed_results, metric=metric)
        assert isinstance(value, float)
        assert 0.0 <= value <= 1.0


def test_aggregate_per_seed_then_average_empty_raises():
    with pytest.raises(ValueError, match="non-empty"):
        aggregate_per_seed_then_average({}, metric="selective_accuracy")


def test_aggregate_per_seed_then_average_invalid_metric_raises():
    seed_results = {
        "s20260321": _seed_results_with_target_sel_acc(0.5, n=4),
    }
    with pytest.raises(KeyError, match="bogus_metric"):
        aggregate_per_seed_then_average(seed_results, metric="bogus_metric")


def test_aggregate_per_seed_then_average_single_seed_returns_seed_value():
    seed_results = {
        "s20260321": _seed_results_with_target_sel_acc(0.6, n=10),
    }
    result = aggregate_per_seed_then_average(
        seed_results, metric="selective_accuracy"
    )
    assert result == pytest.approx(0.6, abs=1e-9)


def test_aggregate_per_seed_then_average_empty_seed_results_raises():
    seed_results = {
        "s20260321": _seed_results_with_target_sel_acc(0.5, n=4),
        "s20260322": [],
    }
    with pytest.raises(ValueError, match="no results"):
        aggregate_per_seed_then_average(seed_results, metric="selective_accuracy")


# ── aggregate_per_seed_then_average_with_breakdown ──────────────────


def _mk_eval(
    persona: str, qid: str, label: str, pred_value: str, *,
    reasoning_type: str = "arbitration", difficulty_class: str = "stable",
    topic: str = "sleep",
) -> EvaluationResult:
    pred = Prediction(answer=pred_value, would_skip=False)
    outcome = (
        EvaluationOutcome.TRUE_ANSWER if pred_value == label
        else EvaluationOutcome.FALSE_ANSWER
    )
    return EvaluationResult(
        method_name="Mock",
        persona_id=persona, qid=qid, prediction=pred, label=label,
        outcome=outcome, reasoning_type=reasoning_type, topic=topic,
        difficulty_class=difficulty_class,
    )


def test_breakdown_helper_single_key_groups_by_attribute():
    """Single-axis breakdown groups by reasoning_type and averages across seeds."""
    # Seed 1: arbitration 1/2 correct = 0.5; identity 2/2 = 1.0
    # Seed 2: arbitration 2/2 = 1.0; identity 1/2 = 0.5
    # Means: arb (0.5+1.0)/2 = 0.75; id (1.0+0.5)/2 = 0.75
    s1 = [
        _mk_eval("p1", "A1", "x", "x"),
        _mk_eval("p2", "A1", "x", "y"),
        _mk_eval("p1", "B2", "x", "x", reasoning_type="identity"),
        _mk_eval("p2", "B2", "x", "x", reasoning_type="identity"),
    ]
    s2 = [
        _mk_eval("p1", "A1", "x", "x"),
        _mk_eval("p2", "A1", "x", "x"),
        _mk_eval("p1", "B2", "x", "x", reasoning_type="identity"),
        _mk_eval("p2", "B2", "x", "y", reasoning_type="identity"),
    ]
    out = aggregate_per_seed_then_average_with_breakdown(
        {"s20260321": s1, "s20260322": s2},
        breakdown_keys="reasoning_type", metric="forced_accuracy",
    )
    assert ("arbitration",) in out
    assert ("identity",) in out
    assert out[("arbitration",)]["point"] == pytest.approx(0.75)
    assert out[("identity",)]["point"] == pytest.approx(0.75)
    assert out[("arbitration",)]["n_seeds_with_cell"] == 2
    assert out[("arbitration",)]["per_seed_values"] == [pytest.approx(0.5), pytest.approx(1.0)]


def test_breakdown_helper_multi_key_groups_by_tuple():
    """(reasoning_type, difficulty_class) cross-product reduces per cell."""
    # Two cells: (arb, stable) seed1=1.0/seed2=0.0 -> 0.5
    #            (arb, svr)    seed1=0.0/seed2=1.0 -> 0.5
    s1 = [
        _mk_eval("p1", "A1", "x", "x", difficulty_class="stable"),
        _mk_eval("p2", "A1", "x", "y", difficulty_class="stated_vs_revealed"),
    ]
    s2 = [
        _mk_eval("p1", "A1", "x", "y", difficulty_class="stable"),
        _mk_eval("p2", "A1", "x", "x", difficulty_class="stated_vs_revealed"),
    ]
    out = aggregate_per_seed_then_average_with_breakdown(
        {"s20260321": s1, "s20260322": s2},
        breakdown_keys=("reasoning_type", "difficulty_class"),
        metric="forced_accuracy",
    )
    assert out[("arbitration", "stable")]["point"] == pytest.approx(0.5)
    assert out[("arbitration", "stated_vs_revealed")]["point"] == pytest.approx(0.5)


def test_breakdown_helper_omits_seed_when_cell_absent():
    """If a cell has no records in a seed, it averages over only the seeds that do."""
    s1 = [_mk_eval("p1", "A1", "x", "x", reasoning_type="arbitration")]
    s2 = [_mk_eval("p2", "B2", "x", "y", reasoning_type="identity")]
    out = aggregate_per_seed_then_average_with_breakdown(
        {"s20260321": s1, "s20260322": s2},
        breakdown_keys="reasoning_type", metric="forced_accuracy",
    )
    # Each cell appears in only one seed
    assert out[("arbitration",)]["point"] == pytest.approx(1.0)
    assert out[("arbitration",)]["n_seeds_with_cell"] == 1
    assert out[("identity",)]["point"] == pytest.approx(0.0)
    assert out[("identity",)]["n_seeds_with_cell"] == 1


def test_breakdown_helper_invalid_key_raises():
    s1 = [_mk_eval("p1", "A1", "x", "x")]
    with pytest.raises(ValueError, match="not in"):
        aggregate_per_seed_then_average_with_breakdown(
            {"s20260321": s1}, breakdown_keys="bogus", metric="forced_accuracy",
        )


def test_breakdown_helper_empty_seed_results_raises():
    with pytest.raises(ValueError, match="non-empty"):
        aggregate_per_seed_then_average_with_breakdown(
            {}, breakdown_keys="reasoning_type", metric="forced_accuracy",
        )


def test_breakdown_helper_supports_qid_axis():
    """qid is in the allowed axis set."""
    s1 = [
        _mk_eval("p1", "A1", "x", "x"),
        _mk_eval("p1", "A2", "x", "y"),
    ]
    out = aggregate_per_seed_then_average_with_breakdown(
        {"s20260321": s1}, breakdown_keys="qid", metric="forced_accuracy",
    )
    assert ("A1",) in out
    assert ("A2",) in out
