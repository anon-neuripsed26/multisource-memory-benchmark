"""Tests for the shared `bias_predict` helper (BCF / future ABF)."""

from __future__ import annotations

import pytest

from survey2agent.extraction.question_spec import QUESTIONS, get_bias
from survey2agent.methods import bias_predict


def _find_qid(predicate) -> str:
    for qid, q in QUESTIONS.items():
        if predicate(qid, q):
            return qid
    raise RuntimeError("no question matches predicate")


def test_ordinal_positive_bias_shifts_up() -> None:
    """For an ordinal question and a source with b=+1, bias_predict shifts
    the candidate's ordinal rank up by 1 and inverse-looks up the label."""
    qid = _find_qid(
        lambda qid, q: q["answer_space_type"] == "ordinal"
        and q["ordinal_encoding"] is not None
        and get_bias(qid).get("planner", 0) == 1
    )
    enc = QUESTIONS[qid]["ordinal_encoding"]
    inv = {pos: label for label, pos in enc.items()}
    # Take a low-rank label so shift+1 stays in-range.
    low_label = inv[1]
    expected = inv[2]
    assert bias_predict("planner", qid, low_label) == expected


def test_ordinal_negative_bias_shifts_down() -> None:
    """For an ordinal question and a source with b=-1, bias_predict shifts
    the candidate's ordinal rank down by 1."""
    qid = _find_qid(
        lambda qid, q: q["answer_space_type"] == "ordinal"
        and q["ordinal_encoding"] is not None
        and get_bias(qid).get("daily_self_report", 0) == -1
    )
    enc = QUESTIONS[qid]["ordinal_encoding"]
    inv = {pos: label for label, pos in enc.items()}
    K = len(enc)
    high_label = inv[K]
    expected = inv[K - 1]
    assert bias_predict("daily_self_report", qid, high_label) == expected


def test_ordinal_clamp_at_top() -> None:
    """A +1 source on the top-ranked candidate must clamp at K (no wrap)."""
    qid = _find_qid(
        lambda qid, q: q["answer_space_type"] == "ordinal"
        and q["ordinal_encoding"] is not None
        and get_bias(qid).get("planner", 0) == 1
    )
    enc = QUESTIONS[qid]["ordinal_encoding"]
    inv = {pos: label for label, pos in enc.items()}
    K = len(enc)
    top_label = inv[K]
    assert bias_predict("planner", qid, top_label) == top_label


def test_ordinal_clamp_at_bottom() -> None:
    """A -1 source on the bottom-ranked candidate must clamp at 1 (no wrap)."""
    qid = _find_qid(
        lambda qid, q: q["answer_space_type"] == "ordinal"
        and q["ordinal_encoding"] is not None
        and get_bias(qid).get("daily_self_report", 0) == -1
    )
    enc = QUESTIONS[qid]["ordinal_encoding"]
    inv = {pos: label for label, pos in enc.items()}
    bottom_label = inv[1]
    assert bias_predict("daily_self_report", qid, bottom_label) == bottom_label


def test_nominal_and_edge_options_are_identity() -> None:
    """Nominal questions, zero-bias sources, and edge options must all
    return the candidate unchanged."""
    # 1. Nominal question: identity for any source.
    nominal_qid = _find_qid(lambda qid, q: q["answer_space_type"] == "nominal")
    cand_n = QUESTIONS[nominal_qid]["answer_space"][0]
    assert bias_predict("planner", nominal_qid, cand_n) == cand_n

    # 2. Zero-bias source on an ordinal question: identity.
    ord_qid = _find_qid(
        lambda qid, q: q["answer_space_type"] == "ordinal"
        and q["ordinal_encoding"] is not None
    )
    enc = QUESTIONS[ord_qid]["ordinal_encoding"]
    cand_o = next(iter(enc.keys()))
    assert bias_predict("objective_log", ord_qid, cand_o) == cand_o

    # 3. Edge option (label not in ordinal_encoding) on an ordinal question
    #    with a non-zero-bias source: still identity.
    edge_qid = _find_qid(
        lambda qid, q: q["answer_space_type"] == "ordinal"
        and q["ordinal_encoding"] is not None
        and len(q["edge_options"]) > 0
    )
    edge_label = QUESTIONS[edge_qid]["edge_options"][0]
    assert bias_predict("planner", edge_qid, edge_label) == edge_label
