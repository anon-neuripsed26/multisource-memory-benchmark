"""Tests for selective.thresholds.grid_search_f_beta."""

from __future__ import annotations

import numpy as np
import pytest

from survey2agent.selective.thresholds import grid_search_f_beta


def _f_beta(tp: float, fp: float, fn: float, beta: float = 0.5) -> float:
    if (tp + fp) == 0 or (tp + fn) == 0:
        return 0.0
    p = tp / (tp + fp)
    r = tp / (tp + fn)
    b2 = beta * beta
    return (1 + b2) * p * r / (b2 * p + r)


def test_single_axis_finds_optimum():
    # 10 examples; predicted_label always "A".
    # Top-5 (high conf) labels are "A" (correct), bottom-5 are "B" (wrong).
    confidences = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05])
    labels = np.array(["A"] * 5 + ["B"] * 5)
    predicted = np.array(["A"] * 10)

    scores = {"predicted_label": predicted, "conf": confidences}
    axes = {"tau": [0.0, 0.5, 1.0]}

    def skip_pred(*, scores, tau):
        return scores["conf"] < tau

    # Hand compute:
    # tau=0.0: TP=5, FP=5, FN=0 → P=0.5, R=1, F0.5 = 1.25*0.5/1.125 ≈ 0.5556
    # tau=0.5: skip 5 (conf<0.5: 0.4,0.3,0.2,0.1,0.05); answered top5 all correct.
    #          TP=5, FP=0, FN=5 → P=1, R=0.5, F0.5 = 0.625/0.75 ≈ 0.8333
    # tau=1.0: all skip → 0
    best_assignment, best_score = grid_search_f_beta(
        scores, labels, axes=axes, skip_predicate=skip_pred, beta=0.5
    )
    assert best_assignment == {"tau": 0.5}
    assert best_score == pytest.approx(_f_beta(5, 0, 5, 0.5))


def test_two_axis_grid():
    # Synthetic ABF-like 2-axis: skip if (best < theta_E) | (delta < theta_d).
    n = 6
    # 4 "should-answer" (high best, high delta, correct) and 2 "should-skip" (low both, wrong).
    best = np.array([3.0, 3.0, 3.0, 3.0, 0.5, 0.5])
    delta = np.array([1.5, 1.5, 1.5, 1.5, 0.1, 0.1])
    predicted = np.array(["A", "A", "A", "A", "A", "A"])
    labels = np.array(["A", "A", "A", "A", "B", "B"])  # last two wrong

    scores = {"predicted_label": predicted, "best": best, "delta": delta}
    axes = {"theta_E": [0.0, 1.0, 2.0], "theta_delta": [0.0, 0.5, 1.0]}

    def skip_pred(*, scores, theta_E, theta_delta):
        return (scores["best"] < theta_E) | (scores["delta"] < theta_delta)

    best_assignment, best_score = grid_search_f_beta(
        scores, labels, axes=axes, skip_predicate=skip_pred, beta=0.5
    )
    # Skipping the two wrong answers gives TP=4, FP=0, FN=2 → P=1, R=2/3,
    # F_0.5 = 1.25 * 1 * (2/3) / (0.25 + 2/3) = 10/11. SKIP still counts as FN
    # under the "skip is never GT" rule, so F=1.0 is unreachable here.
    expected = _f_beta(4, 0, 2, 0.5)
    assert best_score == pytest.approx(expected)
    assert best_score == pytest.approx(10 / 11)
    # First-encountered combo achieving the optimum is (theta_E=0.0, theta_delta=0.5).
    assert best_assignment == {"theta_E": 0.0, "theta_delta": 0.5}


def test_empty_axes_raises():
    scores = {"predicted_label": np.array(["A"])}
    with pytest.raises(ValueError):
        grid_search_f_beta(
            scores,
            np.array(["A"]),
            axes={},
            skip_predicate=lambda **_: np.array([False]),
        )


def test_axis_with_no_candidates_raises():
    scores = {"predicted_label": np.array(["A"])}
    with pytest.raises(ValueError):
        grid_search_f_beta(
            scores,
            np.array(["A"]),
            axes={"tau": []},
            skip_predicate=lambda **_: np.array([False]),
        )


def test_missing_predicted_label_raises():
    scores = {"conf": np.array([0.5])}
    with pytest.raises(ValueError):
        grid_search_f_beta(
            scores,
            np.array(["A"]),
            axes={"tau": [0.0]},
            skip_predicate=lambda *, scores, tau: scores["conf"] < tau,
        )


def test_length_mismatch_raises():
    scores = {"predicted_label": np.array(["A", "A"]), "conf": np.array([0.5])}
    with pytest.raises(ValueError):
        grid_search_f_beta(
            scores,
            np.array(["A", "A"]),
            axes={"tau": [0.0]},
            skip_predicate=lambda *, scores, tau: scores["conf"] < tau,
        )
