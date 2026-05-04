"""Generic multi-axis threshold grid search.

Optimizes selective F_β over an arbitrary product of threshold axes. Each
candidate axis assignment yields a per-example "would skip" boolean mask via
a user-supplied predicate; F_β is then computed on the resulting selective
outcome counts. The optimizer is method-agnostic; ABFSelective.calibrate()
uses an inlined equivalent search and is not refactored to call this
function (byte-equivalence lock with the published reference run).
"""

from __future__ import annotations

import itertools
from typing import Any, Callable, Mapping, Sequence

import numpy as np


def _f_beta_from_counts(tp: float, fp: float, fn: float, beta: float) -> float:
    if (tp + fp) == 0.0 or (tp + fn) == 0.0:
        return 0.0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    beta2 = beta * beta
    denom = beta2 * precision + recall
    if denom == 0.0:
        return 0.0
    return (1.0 + beta2) * precision * recall / denom


def grid_search_f_beta(
    scores: Mapping[str, np.ndarray],
    labels: np.ndarray,
    *,
    axes: Mapping[str, Sequence[float]],
    skip_predicate: Callable[..., np.ndarray],
    beta: float = 0.5,
) -> tuple[dict[str, float], float]:
    """Find argmax F_β over a multi-dim threshold grid.

    Args:
        scores: Mapping from score-name to a length-``N`` array. Must contain
            ``"predicted_label"`` (per-example predicted label, same dtype as
            ``labels``); other keys are passed through to ``skip_predicate``.
        labels: Ground-truth label array of shape ``(N,)``.
        axes: Mapping ``{axis_name: candidate_values}``. Axis order is the
            iteration order of the mapping; ties are broken first-encountered.
        skip_predicate: Callable invoked as
            ``skip_predicate(**axis_assignment, scores=scores)`` returning a
            boolean array of shape ``(N,)``: ``True`` means "would skip".
        beta: F_β weight; default ``0.5`` favors precision.

    Returns:
        A tuple ``(best_axes_assignment, best_f_score)`` where the first
        element maps each axis name to the chosen scalar threshold.

    Raises:
        ValueError: if ``axes`` is empty, any axis has no candidates,
            ``scores`` is empty, ``"predicted_label"`` is missing, or array
            lengths disagree.
    """
    if not axes:
        raise ValueError("axes must contain at least one threshold axis")
    if not scores:
        raise ValueError("scores must contain at least one named array")
    if "predicted_label" not in scores:
        raise ValueError("scores must include a 'predicted_label' array")
    n = len(labels)
    for name, arr in scores.items():
        if len(arr) != n:
            raise ValueError(
                f"scores[{name!r}] length {len(arr)} != labels length {n}"
            )
    for name, candidates in axes.items():
        if len(candidates) == 0:
            raise ValueError(f"axis {name!r} has no candidate values")

    predicted = np.asarray(scores["predicted_label"])
    labels_arr = np.asarray(labels)
    correct_mask = predicted == labels_arr

    axis_names = list(axes.keys())
    candidate_lists = [list(axes[name]) for name in axis_names]

    best_score: float = -1.0
    best_assignment: dict[str, float] = {name: candidate_lists[i][0] for i, name in enumerate(axis_names)}

    for combo in itertools.product(*candidate_lists):
        assignment = dict(zip(axis_names, combo))
        skip_mask = np.asarray(skip_predicate(scores=scores, **assignment), dtype=bool)
        if skip_mask.shape != (n,):
            raise ValueError(
                f"skip_predicate returned shape {skip_mask.shape}, expected ({n},)"
            )
        answered = ~skip_mask
        tp = float(np.sum(answered & correct_mask))
        fp = float(np.sum(answered & ~correct_mask))
        fn = float(np.sum(skip_mask))
        score = _f_beta_from_counts(tp, fp, fn, beta)
        if score > best_score:
            best_score = score
            best_assignment = {name: float(val) for name, val in assignment.items()}

    if best_score < 0.0:
        best_score = 0.0
    return best_assignment, best_score
