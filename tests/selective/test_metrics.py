"""Tests for selective.metrics (macro + micro + dispatchers)."""

from __future__ import annotations

import pytest

from survey2agent.methods.base import SKIP_SENTINEL, Prediction
from survey2agent.selective.metrics import (
    aurc,
    coverage,
    f_beta_selective,
    f_beta_selective_macro,
    f_beta_selective_micro,
    forced_accuracy,
    forced_accuracy_macro,
    forced_accuracy_micro,
    risk_coverage_curve,
    selective_accuracy,
    selective_accuracy_macro,
    selective_accuracy_micro,
)


def _ans(label: str) -> Prediction:
    return Prediction(answer=label, would_skip=False)


def _skip(raw: str | None = None) -> Prediction:
    return Prediction(answer=SKIP_SENTINEL, would_skip=True, raw_answer=raw)


# ---- coverage ----


def test_coverage_empty_raises():
    with pytest.raises(ValueError):
        coverage([])


def test_coverage_all_skip_zero():
    assert coverage([_skip(), _skip(), _skip()]) == 0.0


def test_coverage_all_answer_one():
    assert coverage([_ans("A"), _ans("B")]) == 1.0


def test_coverage_mixed():
    preds = [_ans("A"), _skip(), _ans("B"), _skip()]
    assert coverage(preds) == 0.5


# ---- selective_accuracy_micro ----


def test_selective_accuracy_micro_all_skip_returns_zero():
    preds = [_skip(), _skip()]
    assert selective_accuracy_micro(preds, ["A", "B"]) == 0.0


def test_selective_accuracy_micro_all_correct():
    preds = [_ans("A"), _ans("B"), _ans("C")]
    assert selective_accuracy_micro(preds, ["A", "B", "C"]) == 1.0


def test_selective_accuracy_micro_partial():
    preds = [_ans("A"), _ans("B"), _ans("X"), _skip()]
    labels = ["A", "B", "C", "D"]
    assert selective_accuracy_micro(preds, labels) == pytest.approx(2 / 3)


def test_selective_accuracy_micro_length_mismatch_raises():
    with pytest.raises(ValueError):
        selective_accuracy_micro([_ans("A")], ["A", "B"])


# ---- forced_accuracy_micro ----


def test_forced_accuracy_micro_uses_raw_answer_for_skip():
    preds = [
        Prediction(answer=SKIP_SENTINEL, would_skip=True, raw_answer="A"),
        Prediction(answer=SKIP_SENTINEL, would_skip=True, raw_answer="X"),
        _ans("B"),
    ]
    labels = ["A", "B", "B"]
    assert forced_accuracy_micro(preds, labels) == pytest.approx(2 / 3)


def test_forced_accuracy_micro_no_raw_answer_uses_answer():
    preds = [_ans("A"), _ans("B")]
    assert forced_accuracy_micro(preds, ["A", "B"]) == 1.0


def test_forced_accuracy_micro_skip_no_raw_counts_as_skip_label_wrong():
    preds = [_skip(), _ans("A")]
    assert forced_accuracy_micro(preds, ["A", "A"]) == 0.5


# ---- f_beta_selective_micro ----


def test_f_beta_micro_all_correct_no_skip():
    preds = [_ans("A"), _ans("B"), _ans("C")]
    assert f_beta_selective_micro(preds, ["A", "B", "C"]) == pytest.approx(1.0)


def test_f_beta_micro_all_wrong_no_skip():
    preds = [_ans("X"), _ans("Y")]
    assert f_beta_selective_micro(preds, ["A", "B"]) == 0.0


def test_f_beta_micro_all_skip_zero():
    preds = [_skip(), _skip()]
    assert f_beta_selective_micro(preds, ["A", "B"]) == 0.0


def test_f_beta_micro_hand_computed_3tp_1fp_2fn():
    preds = (
        [_ans("A")] * 3
        + [_ans("X")]
        + [_skip(), _skip()]
    )
    labels = ["A", "A", "A", "B", "C", "D"]
    expected = (1.25 * 0.75 * 0.6) / (0.25 * 0.75 + 0.6)
    assert f_beta_selective_micro(preds, labels, beta=0.5) == pytest.approx(expected)
    assert f_beta_selective_micro(preds, labels, beta=0.5) == pytest.approx(
        0.7142857, abs=1e-6
    )


# ---- selective_accuracy_macro ----


def test_selective_accuracy_macro_drops_empty_qids():
    """A qid with all-SKIP records must be excluded from the macro mean."""
    # Q1: 2 answered, 1 correct → 0.5
    # Q2: 2 answered, 2 correct → 1.0
    # Q3: 2 SKIP, 0 answered → DROPPED
    preds = [
        _ans("A"), _ans("X"),       # Q1: 1/2
        _ans("B"), _ans("B"),       # Q2: 2/2
        _skip(), _skip(),            # Q3: dropped
    ]
    labels = ["A", "A", "B", "B", "C", "C"]
    qids = ["Q1", "Q1", "Q2", "Q2", "Q3", "Q3"]
    # mean(0.5, 1.0) = 0.75 (Q3 dropped, NOT averaged in as 0.0)
    assert selective_accuracy_macro(preds, labels, qids) == pytest.approx(0.75)


def test_selective_accuracy_macro_all_qids_empty_returns_zero():
    preds = [_skip(), _skip()]
    labels = ["A", "B"]
    qids = ["Q1", "Q2"]
    assert selective_accuracy_macro(preds, labels, qids) == 0.0


def test_selective_accuracy_macro_qid_length_mismatch_raises():
    with pytest.raises(ValueError):
        selective_accuracy_macro([_ans("A")], ["A"], ["Q1", "Q2"])


# ---- forced_accuracy_macro ----


def test_forced_accuracy_macro_per_qid_mean():
    # Q1: 1/2 correct, Q2: 2/2 correct → mean = 0.75
    preds = [_ans("A"), _ans("X"), _ans("B"), _ans("B")]
    labels = ["A", "A", "B", "B"]
    qids = ["Q1", "Q1", "Q2", "Q2"]
    assert forced_accuracy_macro(preds, labels, qids) == pytest.approx(0.75)


def test_forced_accuracy_macro_uses_raw_answer():
    # SKIP with raw_answer="A" matches label "A" → counts as correct in forced
    preds = [
        Prediction(answer=SKIP_SENTINEL, would_skip=True, raw_answer="A"),
        _ans("X"),                # wrong
    ]
    labels = ["A", "B"]
    qids = ["Q1", "Q2"]
    # Q1: 1/1 = 1.0; Q2: 0/1 = 0.0 → mean 0.5
    assert forced_accuracy_macro(preds, labels, qids) == pytest.approx(0.5)


# ---- f_beta_selective_macro paper lock (Background §4) ----


@pytest.mark.parametrize(
    "name,sel,cov,forced,expected_f05",
    [
        # DSNBF Selective triple from the spec (Background §4):
        # P = 0.853; R = 0.853*0.783/0.803 ≈ 0.8318;
        # F0.5 = 1.25*P*R / (0.25*P + R) ≈ 0.849 → matches paper 2x2 Factorial Decomposition.
        ("DSNBF",       0.853, 0.783, 0.803, 0.849),
        # Formula identity: when sel=cov=forced=1, F0.5 = 1.
        ("perfect",     1.000, 1.000, 1.000, 1.000),
        # Formula sanity: P=R=0.5 → F0.5 = 0.5.
        # Set cov=forced=0.5 with sel=0.5 → R = 0.5*0.5/0.5 = 0.5.
        ("P=R=0.5",     0.500, 0.500, 0.500, 0.500),
    ],
)
def test_f_beta_macro_paper_lock(name, sel, cov, forced, expected_f05):
    """Macro-derived F0.5 closed-form reproduces paper 2x2 Factorial Decomposition within ±0.001
    for the one concrete triple documented in the DSNBF spec,
    plus formula identities.

    Forced-only baselines (SSB, GPT-Direct, GPT-Schema) are reported in
    2x2 Factorial Decomposition with their forced accuracy as the F0.5 column entry by
    convention (selective F0.5 does not meaningfully apply when no method
    can SKIP); see ``_validate_cp0119_paper_numbers.py`` for end-to-end
    reproduction of NBF / ABF and the forced-only baselines.
    """
    got = f_beta_selective_macro(sel, cov, forced, beta=0.5)
    assert got == pytest.approx(expected_f05, abs=1e-3), (
        f"{name}: got {got:.4f}, expected {expected_f05:.4f}"
    )


def test_f_beta_macro_zero_when_forced_zero():
    assert f_beta_selective_macro(0.5, 0.5, 0.0) == 0.0


def test_f_beta_macro_zero_when_sel_zero():
    assert f_beta_selective_macro(0.0, 0.5, 0.5) == 0.0


# ---- public dispatcher behavior ----


def test_selective_accuracy_default_warns_without_qids():
    preds = [_ans("A"), _ans("X")]
    labels = ["A", "A"]
    with pytest.warns(DeprecationWarning, match="macro reduction"):
        v = selective_accuracy(preds, labels)
    assert v == pytest.approx(0.5)  # micro fallback


def test_forced_accuracy_default_warns_without_qids():
    with pytest.warns(DeprecationWarning, match="macro reduction"):
        v = forced_accuracy([_ans("A"), _ans("X")], ["A", "A"])
    assert v == pytest.approx(0.5)


def test_f_beta_selective_default_warns_without_qids():
    preds = [_ans("A"), _ans("X")]
    labels = ["A", "A"]
    with pytest.warns(DeprecationWarning, match="macro reduction"):
        f_beta_selective(preds, labels)


def test_selective_accuracy_dispatch_macro_when_qids_supplied():
    preds = [_ans("A"), _ans("X"), _ans("B"), _ans("B")]
    labels = ["A", "A", "B", "B"]
    qids = ["Q1", "Q1", "Q2", "Q2"]
    # No warning expected
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any warning becomes an error
        v = selective_accuracy(preds, labels, qids)
    assert v == pytest.approx(0.75)


def test_f_beta_selective_macro_path_via_dispatcher():
    preds = [_ans("A"), _ans("X"), _ans("B"), _ans("B")]
    labels = ["A", "A", "B", "B"]
    qids = ["Q1", "Q1", "Q2", "Q2"]
    # All answered → cov=1.0; sel_macro=0.75; forced_macro=0.75.
    # R = sel*cov/forced = 0.75*1/0.75 = 1.0 (forced-only path).
    # F0.5 = 1.25*0.75*1 / (0.25*0.75 + 1) = 0.9375/1.1875 ≈ 0.7895.
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        v = f_beta_selective(preds, labels, qids, beta=0.5)
    assert v == pytest.approx(0.7894736842105263, abs=1e-6)


# ---- risk_coverage_curve ----


def test_risk_coverage_perfect_ranking_monotone():
    confidences = [0.9, 0.8, 0.7, 0.6, 0.5]
    correct = [True, True, True, False, False]
    cov, risk = risk_coverage_curve(confidences, correct)
    assert len(cov) == 6 and len(risk) == 6
    assert cov == sorted(cov)
    assert risk[0] == 0.0
    assert risk[1] == 0.0
    assert risk[2] == 0.0
    assert risk[3] == 0.0
    assert risk[4] == pytest.approx(1 / 4)
    assert risk[5] == pytest.approx(2 / 5)


def test_risk_coverage_unsorted_input_resorted_internally():
    confidences = [0.1, 0.9, 0.5]
    correct = [False, True, False]
    cov, risk = risk_coverage_curve(confidences, correct)
    assert risk == [0.0, 0.0, 0.5, pytest.approx(2 / 3)]
    assert cov == [0.0, pytest.approx(1 / 3), pytest.approx(2 / 3), 1.0]


def test_risk_coverage_length_mismatch_raises():
    with pytest.raises(ValueError):
        risk_coverage_curve([0.5], [True, False])


# ---- aurc ----


def test_aurc_three_point_trapezoid():
    assert aurc([0.0, 0.5, 1.0], [0.0, 0.0, 0.5]) == pytest.approx(0.125)


def test_aurc_perfect_predictor_zero():
    cov = [0.0, 0.25, 0.5, 0.75, 1.0]
    risk = [0.0, 0.0, 0.0, 0.0, 0.0]
    assert aurc(cov, risk) == 0.0


def test_aurc_length_mismatch_raises():
    with pytest.raises(ValueError):
        aurc([0.0, 1.0], [0.0])
