"""Per-axis breakdown utilities over ``EvaluationResult`` records.

Group results by ``reasoning_type`` / ``topic`` / ``difficulty_class`` (or
the ``(qid, difficulty_class)`` cross-product) and delegate per-group
metrics to :func:`aggregate_metrics` so the output shape mirrors the main
aggregator (macro-over-qid + micro siblings).

Cells with zero results are omitted (defaultdict only accumulates on
append). Groups are returned in sorted-key order for deterministic output.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Literal, Sequence

from .aggregator import aggregate_metrics
from .runner import EvaluationResult

__all__ = ["breakdown_by", "breakdown_by_qid_difficulty"]


_ALLOWED_KEYS = ("reasoning_type", "topic", "difficulty_class")


def breakdown_by(
    results: Sequence[EvaluationResult],
    *,
    key: Literal["reasoning_type", "topic", "difficulty_class"],
    beta: float = 0.5,
) -> dict[str, dict[str, float]]:
    """Group ``results`` by ``key`` then call :func:`aggregate_metrics` per group.

    Parameters
    ----------
    results:
        Sequence of :class:`EvaluationResult`.
    key:
        One of ``"reasoning_type"``, ``"topic"``, ``"difficulty_class"``.
    beta:
        F-beta beta value forwarded to :func:`aggregate_metrics`.

    Returns
    -------
    ``{group_value: metrics_dict}``. Groups with no results are omitted.
    Output is keyed in sorted order for deterministic iteration.
    """
    if key not in _ALLOWED_KEYS:
        raise ValueError(
            f"key must be one of {_ALLOWED_KEYS!r}, got {key!r}"
        )

    groups: dict[str, list[EvaluationResult]] = defaultdict(list)
    for r in results:
        groups[getattr(r, key)].append(r)

    return {
        g: aggregate_metrics(groups[g], beta=beta)
        for g in sorted(groups)
    }


def breakdown_by_qid_difficulty(
    results: Sequence[EvaluationResult],
    *,
    beta: float = 0.5,
) -> dict[str, dict[str, dict[str, float]]]:
    """Cross-product breakdown by ``(qid, difficulty_class)``.

    Returns ``{qid: {difficulty_class: metrics_dict}}``. Cells with zero
    results are omitted. Both outer and inner dicts iterate in sorted order.
    """
    groups: dict[tuple[str, str], list[EvaluationResult]] = defaultdict(list)
    for r in results:
        groups[(r.qid, r.difficulty_class)].append(r)

    out: dict[str, dict[str, dict[str, float]]] = {}
    for qid, diff in sorted(groups):
        out.setdefault(qid, {})[diff] = aggregate_metrics(
            groups[(qid, diff)], beta=beta
        )
    return out
