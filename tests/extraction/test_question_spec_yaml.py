"""Tests for the YAML-backed question spec loader."""

from __future__ import annotations

from collections import Counter

import pytest

from survey2agent.extraction.question_spec import (
    BIAS_DEFAULTS,
    BIAS_OVERRIDES,
    QUESTION_TEXT,
    QUESTIONS,
    SOURCE_NAMES,
    get_bias,
)


def test_question_spec_has_18_questions() -> None:
    assert len(QUESTIONS) == 18


def test_reasoning_type_distribution() -> None:
    counts = Counter(q["type"] for q in QUESTIONS.values())
    expected = {
        "arbitration": 3,
        "identity": 2,
        "plan_reality": 2,
        "trend": 2,
        "causal": 2,
        "missing_data": 3,
        "annotation": 2,
        "control": 2,
    }
    assert dict(counts) == expected
    assert sum(expected.values()) == 18


def test_edge_options_subset_of_answer_space() -> None:
    for qid, q in QUESTIONS.items():
        edges = set(q["edge_options"])
        space = set(q["answer_space"])
        assert edges <= space, f"{qid}: edge_options {edges - space} not in answer_space"


def test_ordinal_questions_have_ordered_labels() -> None:
    for qid, q in QUESTIONS.items():
        if q["answer_space_type"] == "ordinal":
            enc = q["ordinal_encoding"]
            assert isinstance(enc, dict) and enc, f"{qid}: ordinal needs non-empty encoding"
        elif q["answer_space_type"] == "nominal":
            assert q["ordinal_encoding"] is None, f"{qid}: nominal must have None encoding"
        else:
            raise AssertionError(f"{qid}: unknown answer_space_type {q['answer_space_type']!r}")


def test_question_text_keys_match_questions_keys() -> None:
    assert set(QUESTION_TEXT) == set(QUESTIONS)
    # SOURCE_NAMES should be the canonical 5-tuple
    assert len(SOURCE_NAMES) == 5


# ── bias_overrides resolution ───────────────────────────────────────────────


def test_bias_overrides_keys_are_known_qids_and_sources() -> None:
    for qid, per_q in BIAS_OVERRIDES.items():
        assert qid in QUESTIONS, f"override references unknown qid {qid!r}"
        for source in per_q:
            assert source in SOURCE_NAMES, (
                f"override {qid}.{source} references unknown source"
            )


def test_get_bias_resolution_order() -> None:
    # A2 has explicit override: daily_self_report=+1, but planner / others
    # fall back to topic-level defaults.
    bias = get_bias("A2")
    assert bias["daily_self_report"] == 1, "A2 override should apply"
    # planner has scalar default +1.
    assert bias["planner"] == BIAS_DEFAULTS["planner"]
    # profile_ltm / objective_log / device_log scalar 0.
    assert bias["profile_ltm"] == 0
    assert bias["objective_log"] == 0
    assert bias["device_log"] == 0


def test_get_bias_partial_override_falls_back_to_topic_default() -> None:
    # G2 overrides daily_self_report only; planner must still come from the
    # scalar topic default.
    bias = get_bias("G2")
    assert bias["daily_self_report"] == 1
    assert bias["planner"] == BIAS_DEFAULTS["planner"]


def test_get_bias_unoverridden_topic_dependent_uses_default() -> None:
    # D1 has no override and topic=social → daily_self_report should resolve
    # to BIAS_DEFAULTS["daily_self_report"]["social"].
    expected = BIAS_DEFAULTS["daily_self_report"]["social"]
    assert get_bias("D1")["daily_self_report"] == expected


def test_get_bias_F1_double_override() -> None:
    # F1 overrides BOTH planner and daily_self_report.
    bias = get_bias("F1")
    assert bias["planner"] == -1
    assert bias["daily_self_report"] == 1


def test_get_bias_unknown_qid_raises() -> None:
    with pytest.raises(KeyError):
        get_bias("NOT_A_REAL_QID")


def test_documented_overrides_are_present() -> None:
    # The four documented semantic-inversion fixes must be in the YAML.
    expected = {
        ("A2", "daily_self_report"): 1,
        ("C2", "daily_self_report"): 1,
        ("F1", "planner"): -1,
        ("F1", "daily_self_report"): 1,
        ("G2", "daily_self_report"): 1,
    }
    for (qid, source), value in expected.items():
        assert BIAS_OVERRIDES.get(qid, {}).get(source) == value, (
            f"missing or wrong override {qid}.{source}={value}"
        )
