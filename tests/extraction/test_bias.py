"""Tests for question_spec.get_bias (QBD-2 bias matrix loader)."""

from __future__ import annotations

from survey2agent.extraction.question_spec import (
    QUESTIONS,
    SOURCE_NAMES,
    get_bias,
)


def test_get_bias_returns_5_sources():
    for qid in QUESTIONS:
        bias = get_bias(qid)
        assert set(bias.keys()) == set(SOURCE_NAMES), (
            f"qid {qid}: keys {set(bias.keys())} != SOURCE_NAMES {set(SOURCE_NAMES)}"
        )


def test_objective_log_bias_is_zero():
    for qid in QUESTIONS:
        assert get_bias(qid)["objective_log"] == 0, (
            f"qid {qid}: objective_log bias != 0"
        )


def test_daily_self_report_topic_dependent():
    # Topic-level defaults: work=-1, diet=+1, social=-1, sleep=+1, exercise=+1
    # Per-question overrides (semantic-inversion fixes): A2=+1 (reversed
    # ordered_labels), C2=+1 (plan-realization), F1=+1 (unplanned-social),
    # G2=+1 (voluntary-vs-obligatory). See `bias_overrides` in
    # configs/questions.yaml.
    expected = {
        "A1": +1,   # sleep
        "A2": +1,   # work, override
        "A3": +1,   # diet
        "B2": +1,   # exercise
        "B3": -1,   # work
        "C2": +1,   # social, override
        "C3": +1,   # sleep
        "D1": -1,   # social
        "D2": +1,   # diet
        "E1": +1,   # sleep
        "E2": +1,   # exercise
        "F1": +1,   # social, override
        "F2": +1,   # exercise
        "F3": -1,   # work
        "G1": +1,   # exercise
        "G2": +1,   # social, override
        "Ctrl1": +1,  # diet
        "Ctrl2": +1,  # sleep
    }
    for qid, want in expected.items():
        got = get_bias(qid)["daily_self_report"]
        assert got == want, f"qid {qid}: daily_self_report bias {got} != {want}"


def test_planner_bias_is_plus_one_except_overrides():
    # F1 overrides planner to -1 because planners cannot record unplanned
    # social activity by construction. All other qids keep the +1 default.
    for qid in QUESTIONS:
        want = -1 if qid == "F1" else 1
        got = get_bias(qid)["planner"]
        assert got == want, f"qid {qid}: planner bias {got} != {want}"
