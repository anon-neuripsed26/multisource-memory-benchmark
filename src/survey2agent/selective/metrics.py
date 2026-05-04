"""Selective evaluation metrics (macro-over-qid + micro siblings).

Pure functions over `Sequence[Prediction]` and `Sequence[str]` ground-truth
labels. SKIP semantics follow the `Prediction` dataclass invariant in
`methods.base`.

This module exposes both **macro** (default; mean over per-qid scores,
matching the paper's reporting convention) and **micro** (flat over all
records) reductions. Public names ``selective_accuracy`` /
``forced_accuracy`` / ``f_beta_selective`` dispatch to macro when ``qids``
is supplied; otherwise they emit a ``DeprecationWarning`` and fall back to
the micro variant (preserving legacy callers).

Macro reductions match v1.0 reference exactly:
- ``selective_accuracy_macro``: mean over qids of
  ``answered_correct/answered``, dropping qids with ``answered == 0``.
- ``forced_accuracy_macro``: mean over qids of forced per-qid accuracy
  (the legacy ``overall_macro`` field; SKIP without ``raw_answer`` counts
  as wrong per ``forced_accuracy`` semantics).
- ``f_beta_selective_macro``: pure scalar ``F_β`` from
  ``(P=sel_acc_macro, R=sel_acc_macro * coverage / forced_acc_macro)``.

``coverage`` remains micro by definition (``Σanswered/Σtotal``).
``risk_coverage_curve`` and ``aurc`` are unchanged.
"""

from __future__ import annotations

import warnings
from collections import defaultdict
from typing import Sequence

import numpy as np

from survey2agent.methods.base import Prediction


def _check_lengths(predictions: Sequence[Prediction], labels: Sequence[str]) -> None:
    if len(predictions) != len(labels):
        raise ValueError(
            f"predictions and labels length mismatch: {len(predictions)} vs {len(labels)}"
        )


def _check_qids(predictions: Sequence[Prediction], qids: Sequence[str]) -> None:
    if len(predictions) != len(qids):
        raise ValueError(
            f"predictions and qids length mismatch: {len(predictions)} vs {len(qids)}"
        )


def coverage(predictions: Sequence[Prediction]) -> float:
    """Fraction of predictions that did not abstain. Range ``[0, 1]``.

    Always micro (``Σanswered/Σtotal``); macro coverage is not a reported
    metric.
    """
    if len(predictions) == 0:
        raise ValueError("coverage requires at least one prediction")
    return 1.0 - sum(1 for p in predictions if p.would_skip) / len(predictions)


# ---- micro reductions (legacy) ----


def selective_accuracy_micro(
    predictions: Sequence[Prediction],
    labels: Sequence[str],
) -> float:
    """Micro selective accuracy: ``correct / answered`` over the flat list.

    Returns ``0.0`` if all predictions abstain.
    """
    if len(predictions) == 0:
        raise ValueError("selective_accuracy_micro requires at least one prediction")
    _check_lengths(predictions, labels)
    answered = [(p, lab) for p, lab in zip(predictions, labels) if not p.would_skip]
    if not answered:
        return 0.0
    correct = sum(1 for p, lab in answered if p.answer == lab)
    return correct / len(answered)


def forced_accuracy_micro(
    predictions: Sequence[Prediction],
    labels: Sequence[str],
) -> float:
    """Micro forced accuracy.

    Uses ``raw_answer`` when available (Selective methods scored in forced
    mode), else ``answer``. SKIP without ``raw_answer`` counts as wrong.
    """
    if len(predictions) == 0:
        raise ValueError("forced_accuracy_micro requires at least one prediction")
    _check_lengths(predictions, labels)
    correct = 0
    for p, lab in zip(predictions, labels):
        chosen = p.raw_answer if p.raw_answer is not None else p.answer
        if chosen == lab:
            correct += 1
    return correct / len(predictions)


def f_beta_selective_micro(
    predictions: Sequence[Prediction],
    labels: Sequence[str],
    *,
    beta: float = 0.5,
) -> float:
    """Cost-asymmetric selective F_β, micro reduction (default β=0.5).

    With ``TP`` = non-skip & correct, ``FP`` = non-skip & wrong,
    ``FN`` = skip (every example treated as answerable per the locked
    "skip is never GT" constraint), returns
    ``(1 + β²) · P · R / (β² · P + R)``. Returns ``0.0`` when either
    ``TP + FP == 0`` or ``TP + FN == 0``.
    """
    if len(predictions) == 0:
        raise ValueError("f_beta_selective_micro requires at least one prediction")
    _check_lengths(predictions, labels)
    tp = fp = fn = 0
    for p, lab in zip(predictions, labels):
        if p.would_skip:
            fn += 1
        elif p.answer == lab:
            tp += 1
        else:
            fp += 1
    if (tp + fp) == 0 or (tp + fn) == 0:
        return 0.0
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    beta2 = beta * beta
    denom = beta2 * precision + recall
    if denom == 0.0:
        return 0.0
    return (1.0 + beta2) * precision * recall / denom


# ---- macro reductions (paper default) ----


def _group_by_qid(
    predictions: Sequence[Prediction],
    labels: Sequence[str],
    qids: Sequence[str],
) -> dict[str, list[tuple[Prediction, str]]]:
    grouped: dict[str, list[tuple[Prediction, str]]] = defaultdict(list)
    for p, lab, q in zip(predictions, labels, qids):
        grouped[q].append((p, lab))
    return grouped


def selective_accuracy_macro(
    predictions: Sequence[Prediction],
    labels: Sequence[str],
    qids: Sequence[str],
) -> float:
    """Macro selective accuracy: mean over qids of ``correct/answered``.

    Qids with zero answered records are dropped (matches
    v1.0 reference). Returns ``0.0`` if
    no qid has any answered record.
    """
    if len(predictions) == 0:
        raise ValueError("selective_accuracy_macro requires at least one prediction")
    _check_lengths(predictions, labels)
    _check_qids(predictions, qids)
    per_q = _group_by_qid(predictions, labels, qids)
    sel_accs: list[float] = []
    for _q, items in per_q.items():
        answered = [(p, lab) for p, lab in items if not p.would_skip]
        if not answered:
            continue  # drop empty qid (paper convention)
        correct = sum(1 for p, lab in answered if p.answer == lab)
        sel_accs.append(correct / len(answered))
    if not sel_accs:
        return 0.0
    return sum(sel_accs) / len(sel_accs)


def forced_accuracy_macro(
    predictions: Sequence[Prediction],
    labels: Sequence[str],
    qids: Sequence[str],
) -> float:
    """Macro forced accuracy: mean over qids of per-qid forced accuracy.

    Matches the legacy ``overall_macro`` field. Per-qid forced accuracy
    uses the same ``raw_answer``-fallback rule as
    :func:`forced_accuracy_micro` (SKIP without ``raw_answer`` counts as
    wrong).
    """
    if len(predictions) == 0:
        raise ValueError("forced_accuracy_macro requires at least one prediction")
    _check_lengths(predictions, labels)
    _check_qids(predictions, qids)
    per_q = _group_by_qid(predictions, labels, qids)
    accs: list[float] = []
    for _q, items in per_q.items():
        if not items:
            continue
        correct = 0
        for p, lab in items:
            chosen = p.raw_answer if p.raw_answer is not None else p.answer
            if chosen == lab:
                correct += 1
        accs.append(correct / len(items))
    if not accs:
        return 0.0
    return sum(accs) / len(accs)


def f_beta_selective_macro(
    sel_acc_macro: float,
    coverage_macro: float,
    forced_acc_macro: float,
    *,
    beta: float = 0.5,
) -> float:
    """Macro-derived selective F_β from already-computed scalars.

    ``P = sel_acc_macro``;
    ``R = (sel_acc_macro * coverage) / forced_acc_macro``
    (interpretation: the fraction of forced-correct instances that are
    answered).
    ``F_β = (1 + β²) · P · R / (β² · P + R)``.

    Returns ``0.0`` when ``P == 0`` or ``R == 0`` or
    ``forced_acc_macro <= 0``. Reproduces paper Table 5 F0.5 within
    ±0.001.
    """
    if forced_acc_macro <= 0.0:
        return 0.0
    p = float(sel_acc_macro)
    r = (float(sel_acc_macro) * float(coverage_macro)) / float(forced_acc_macro)
    if p <= 0.0 or r <= 0.0:
        return 0.0
    beta2 = beta * beta
    denom = beta2 * p + r
    if denom == 0.0:
        return 0.0
    return (1.0 + beta2) * p * r / denom


# ---- public dispatchers (default = macro when qids supplied) ----


_DEPRECATION_MSG = (
    "Pass qids to use macro reduction (paper convention); "
    "falling back to micro reduction"
)


def selective_accuracy(
    predictions: Sequence[Prediction],
    labels: Sequence[str],
    qids: Sequence[str] | None = None,
) -> float:
    """Public dispatcher. Macro when ``qids`` is supplied, else micro+warn.

    Macro path matches paper Table 5 ``selective_accuracy``.
    """
    if qids is None:
        warnings.warn(_DEPRECATION_MSG, DeprecationWarning, stacklevel=2)
        return selective_accuracy_micro(predictions, labels)
    return selective_accuracy_macro(predictions, labels, qids)


def forced_accuracy(
    predictions: Sequence[Prediction],
    labels: Sequence[str],
    qids: Sequence[str] | None = None,
) -> float:
    """Public dispatcher. Macro when ``qids`` is supplied, else micro+warn.

    Macro path matches paper Table 5 ``forced_accuracy`` (the legacy
    ``overall_macro``).
    """
    if qids is None:
        warnings.warn(_DEPRECATION_MSG, DeprecationWarning, stacklevel=2)
        return forced_accuracy_micro(predictions, labels)
    return forced_accuracy_macro(predictions, labels, qids)


def f_beta_selective(
    predictions: Sequence[Prediction],
    labels: Sequence[str],
    qids: Sequence[str] | None = None,
    *,
    beta: float = 0.5,
) -> float:
    """Public dispatcher. Macro-derived F_β when ``qids`` supplied, else
    micro+warn.

    Macro path computes
    ``(sel_acc_macro, coverage, forced_acc_macro)`` and feeds them to
    :func:`f_beta_selective_macro`. Reproduces paper Table 5 F0.5.
    """
    if qids is None:
        warnings.warn(_DEPRECATION_MSG, DeprecationWarning, stacklevel=2)
        return f_beta_selective_micro(predictions, labels, beta=beta)
    sel = selective_accuracy_macro(predictions, labels, qids)
    cov = coverage(predictions)
    forced = forced_accuracy_macro(predictions, labels, qids)
    return f_beta_selective_macro(sel, cov, forced, beta=beta)


# ---- risk-coverage curve / AURC (unchanged) ----


def risk_coverage_curve(
    confidences: Sequence[float],
    correct: Sequence[bool],
) -> tuple[list[float], list[float]]:
    """Sweep confidence threshold and return ``(coverages, risks)``.

    Examples are sorted by descending confidence. For each ``k = 0..N``,
    the top ``k`` are "accepted"; coverage is ``k/N`` and risk is the
    error rate among the accepted (``0.0`` at ``k=0`` by convention).
    Output lists have length ``N + 1`` and start with the
    ``coverage=0, risk=0`` anchor.
    """
    if len(confidences) != len(correct):
        raise ValueError(
            f"confidences and correct length mismatch: {len(confidences)} vs {len(correct)}"
        )
    n = len(confidences)
    if n == 0:
        raise ValueError("risk_coverage_curve requires at least one example")
    order = sorted(range(n), key=lambda i: confidences[i], reverse=True)
    sorted_correct = [bool(correct[i]) for i in order]
    coverages: list[float] = [0.0]
    risks: list[float] = [0.0]
    cum_errors = 0
    for k in range(1, n + 1):
        if not sorted_correct[k - 1]:
            cum_errors += 1
        coverages.append(k / n)
        risks.append(cum_errors / k)
    return coverages, risks


def aurc(coverages: Sequence[float], risks: Sequence[float]) -> float:
    """Area under the risk-coverage curve via trapezoidal integration.

    Lower is better. Inputs must be parallel sequences of equal length.
    """
    if len(coverages) != len(risks):
        raise ValueError(
            f"coverages and risks length mismatch: {len(coverages)} vs {len(risks)}"
        )
    if len(coverages) < 2:
        raise ValueError("aurc requires at least two points")
    cov_arr = np.asarray(coverages, dtype=float)
    if not np.all(np.diff(cov_arr) >= 0):
        raise ValueError("aurc requires coverages to be monotonically non-decreasing")
    trapz_fn = getattr(np, "trapezoid", None) or np.trapz  # numpy>=2 renamed trapz
    return float(trapz_fn(np.asarray(risks, dtype=float), cov_arr))
