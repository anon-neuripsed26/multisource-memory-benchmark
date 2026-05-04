"""Tests for evaluation.bootstrap (persona-cluster default)."""

from __future__ import annotations

from survey2agent.evaluation.bootstrap import bootstrap_ci
from survey2agent.evaluation.runner import EvaluationResult
from survey2agent.methods.base import Prediction
from survey2agent.selective.protocol import EvaluationOutcome


def _mk(correct: bool, *, persona: str = "p", qid: str = "A1") -> EvaluationResult:
    label = "X"
    answer = "X" if correct else "Y"
    pred = Prediction(answer=answer, would_skip=False)
    outcome = (
        EvaluationOutcome.TRUE_ANSWER if correct else EvaluationOutcome.FALSE_ANSWER
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


def _mk_persona(persona: str, n_correct: int, n_wrong: int) -> list[EvaluationResult]:
    """Build a persona's record block: ``n_correct`` correct + ``n_wrong`` wrong,
    using distinct qids so macro reduction is well-defined."""
    out: list[EvaluationResult] = []
    qid_idx = 0
    for _ in range(n_correct):
        out.append(_mk(True, persona=persona, qid=f"Q{qid_idx}"))
        qid_idx += 1
    for _ in range(n_wrong):
        out.append(_mk(False, persona=persona, qid=f"Q{qid_idx}"))
        qid_idx += 1
    return out


# ---- iid path (legacy escape hatch) ----


def test_bootstrap_iid_mean_close_to_point():
    results = [_mk(True) for _ in range(70)] + [_mk(False) for _ in range(30)]
    out = bootstrap_ci(
        results,
        metric="f_beta",
        n_bootstrap=300,
        seed=123,
        cluster_by="iid",
    )
    assert out["n_bootstrap"] == 300
    assert out["cluster_by"] == "iid"
    assert abs(out["mean"] - out["point"]) < 0.05
    assert out["ci_low"] <= out["point"] <= out["ci_high"]
    assert out["std"] > 0.0


def test_bootstrap_iid_determinism():
    results = [_mk(True) for _ in range(50)] + [_mk(False) for _ in range(50)]
    a = bootstrap_ci(
        results, metric="selective_accuracy", n_bootstrap=200,
        seed=7, cluster_by="iid",
    )
    b = bootstrap_ci(
        results, metric="selective_accuracy", n_bootstrap=200,
        seed=7, cluster_by="iid",
    )
    assert a == b


def test_bootstrap_iid_metric_fn_constant():
    """IID resample of N records returns N records every replicate."""
    results = [_mk(True) for _ in range(100)]
    out = bootstrap_ci(
        results,
        metric_fn=lambda rs: float(len(rs)),
        n_bootstrap=50,
        seed=11,
        cluster_by="iid",
    )
    assert out["point"] == 100.0
    assert out["mean"] == 100.0
    assert out["std"] == 0.0
    assert out["ci_low"] == 100.0
    assert out["ci_high"] == 100.0


def test_bootstrap_iid_different_seeds_give_different_bounds():
    results = [_mk(True) for _ in range(60)] + [_mk(False) for _ in range(40)]
    a = bootstrap_ci(
        results, metric="selective_accuracy", n_bootstrap=200,
        seed=1, cluster_by="iid",
    )
    b = bootstrap_ci(
        results, metric="selective_accuracy", n_bootstrap=200,
        seed=2, cluster_by="iid",
    )
    assert abs(a["mean"] - 0.6) < 0.05
    assert abs(b["mean"] - 0.6) < 0.05
    assert (a["ci_low"], a["ci_high"]) != (b["ci_low"], b["ci_high"])


# ---- empty / error paths ----


def test_bootstrap_empty_returns_zeros():
    out = bootstrap_ci([], metric="f_beta", n_bootstrap=100, seed=1)
    assert out["point"] == 0.0
    assert out["mean"] == 0.0
    assert out["std"] == 0.0
    assert out["ci_low"] == 0.0
    assert out["ci_high"] == 0.0
    assert out["n_bootstrap"] == 0


def test_bootstrap_invalid_cluster_by_raises():
    import pytest
    results = [_mk(True), _mk(False)]
    with pytest.raises(ValueError, match="cluster_by"):
        bootstrap_ci(results, n_bootstrap=10, cluster_by="bogus")


# ---- persona-cluster path (default; paper convention) ----


def test_bootstrap_persona_cluster_default():
    """Default ``cluster_by="persona"``; resamples N personas, not N rows."""
    # 10 personas × 6 records each = 60 rows.
    # Mix of accuracies so cluster bootstrap has nonzero variance.
    results: list[EvaluationResult] = []
    for i in range(10):
        # Each persona has 4 correct + 2 wrong → per-persona acc = 0.667
        # but persona variance comes from differing qids.
        results.extend(_mk_persona(f"p{i}", n_correct=4, n_wrong=2))

    out = bootstrap_ci(
        results, metric="selective_accuracy", n_bootstrap=200, seed=42,
    )
    assert out["cluster_by"] == "persona"
    assert out["n_bootstrap"] == 200
    # All personas identical → cluster resample variance should be ~0
    # (float epsilon from per-replicate division ordering).
    assert out["std"] < 1e-10
    assert abs(out["ci_low"] - out["point"]) < 1e-10
    assert abs(out["ci_high"] - out["point"]) < 1e-10


def test_bootstrap_persona_cluster_resamples_personas_not_rows():
    """Verify cluster sample size equals N_personas times persona block size,
    not N_rows resampled IID."""
    # 5 personas of size 4 = 20 rows
    results: list[EvaluationResult] = []
    for i in range(5):
        results.extend(_mk_persona(f"p{i}", n_correct=2, n_wrong=2))

    # metric_fn returns count of records in the replicate
    out = bootstrap_ci(
        results,
        metric_fn=lambda rs: float(len(rs)),
        n_bootstrap=50,
        seed=1,
    )
    # Cluster bootstrap: pick 5 personas with replacement, each contributes
    # 4 records → every replicate has exactly 5*4 = 20 records.
    assert out["mean"] == 20.0
    assert out["std"] == 0.0


def test_bootstrap_persona_cluster_variance_with_heterogeneous_personas():
    """Persona-cluster bootstrap must show variance when personas differ."""
    # Heterogeneous: half all-correct, half all-wrong personas
    results: list[EvaluationResult] = []
    for i in range(5):
        results.extend(_mk_persona(f"good{i}", n_correct=4, n_wrong=0))
    for i in range(5):
        results.extend(_mk_persona(f"bad{i}", n_correct=0, n_wrong=4))

    out = bootstrap_ci(
        results, metric="selective_accuracy", n_bootstrap=300, seed=42,
    )
    assert out["cluster_by"] == "persona"
    # With 10 personas of bimodal accuracy, resampling should see variance.
    assert out["std"] > 0.0
    assert out["ci_low"] < out["point"] or out["ci_high"] > out["point"]


def test_bootstrap_persona_cluster_determinism():
    results: list[EvaluationResult] = []
    for i in range(8):
        results.extend(_mk_persona(f"p{i}", n_correct=3, n_wrong=1))
    a = bootstrap_ci(results, metric="f_beta", n_bootstrap=100, seed=7)
    b = bootstrap_ci(results, metric="f_beta", n_bootstrap=100, seed=7)
    assert a == b
