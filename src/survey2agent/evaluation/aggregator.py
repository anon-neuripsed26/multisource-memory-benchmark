"""Headline-metric aggregation over ``EvaluationResult`` records.

Default reduction is **macro-over-qid** to match the paper's reporting
convention (v1.0 reference). Micro siblings remain available as
``*_micro`` keys for transparency.

Returned keys:
    n, n_correct, n_wrong, n_skip,
    coverage,                 # micro by definition (Σanswered/Σtotal)
    selective_accuracy,       # macro
    selective_accuracy_micro, # micro sibling
    forced_accuracy,          # macro (legacy ``overall_macro``)
    forced_accuracy_micro,    # micro sibling
    f_beta,                   # macro-derived (paper Table 5)
    f_beta_micro,             # micro sibling
"""

from __future__ import annotations

from typing import Any, Literal, Sequence

from survey2agent.selective.metrics import (
    coverage,
    f_beta_selective_macro,
    f_beta_selective_micro,
    forced_accuracy_macro,
    forced_accuracy_micro,
    selective_accuracy_macro,
    selective_accuracy_micro,
)
from survey2agent.selective.protocol import reduce_outcomes

from .runner import EvaluationResult

__all__ = ["aggregate_metrics"]


def _empty_metrics() -> dict[str, float]:
    return {
        "n": 0,
        "coverage": 0.0,
        "selective_accuracy": 0.0,
        "selective_accuracy_micro": 0.0,
        "forced_accuracy": 0.0,
        "forced_accuracy_micro": 0.0,
        "f_beta": 0.0,
        "f_beta_micro": 0.0,
        "n_correct": 0,
        "n_wrong": 0,
        "n_skip": 0,
    }


def aggregate_metrics(
    results: Sequence[EvaluationResult],
    *,
    beta: float = 0.5,
    breakdown: Literal[
        "reasoning_type", "topic", "difficulty_class", "qid_difficulty"
    ] | None = None,
) -> dict[str, Any]:
    """Compute headline metrics on a list of :class:`EvaluationResult`.

    Defaults to **macro-over-qid** for ``selective_accuracy``,
    ``forced_accuracy``, and ``f_beta``; ``*_micro`` siblings expose the
    legacy flat-list reductions. ``coverage`` is always micro.

    Empty input → all zeros (no exception, no NaN).

    If ``breakdown`` is provided, the output additionally carries a
    ``"breakdown"`` key with the per-group metrics dict produced by
    :mod:`.breakdowns` (``breakdown_by`` for the three single-axis keys,
    ``breakdown_by_qid_difficulty`` for ``"qid_difficulty"``). Top-level
    keys are unchanged for backward compatibility.
    """
    if len(results) == 0:
        out: dict[str, Any] = dict(_empty_metrics())
        if breakdown is not None:
            out["breakdown"] = {}
        return out

    preds = [r.prediction for r in results]
    labels = [r.label for r in results]
    qids = [r.qid for r in results]
    counts = reduce_outcomes([r.outcome for r in results])

    cov = coverage(preds)
    sel_macro = selective_accuracy_macro(preds, labels, qids)
    sel_micro = selective_accuracy_micro(preds, labels)
    forced_macro = forced_accuracy_macro(preds, labels, qids)
    forced_micro = forced_accuracy_micro(preds, labels)
    fb_macro = f_beta_selective_macro(sel_macro, cov, forced_macro, beta=beta)
    fb_micro = f_beta_selective_micro(preds, labels, beta=beta)

    out = {
        "n": len(results),
        "coverage": cov,
        "selective_accuracy": sel_macro,
        "selective_accuracy_micro": sel_micro,
        "forced_accuracy": forced_macro,
        "forced_accuracy_micro": forced_micro,
        "f_beta": fb_macro,
        "f_beta_micro": fb_micro,
        "n_correct": counts["true_answer"],
        "n_wrong": counts["false_answer"],
        "n_skip": counts["skip"],
    }

    if breakdown is not None:
        # Lazy local import: breakdowns imports aggregate_metrics.
        from .breakdowns import breakdown_by, breakdown_by_qid_difficulty

        if breakdown == "qid_difficulty":
            out["breakdown"] = breakdown_by_qid_difficulty(results, beta=beta)
        else:
            out["breakdown"] = breakdown_by(results, key=breakdown, beta=beta)

    return out
