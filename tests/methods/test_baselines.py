"""Tests for the 6 trivial / single-source / fusion / oracle baselines."""

from __future__ import annotations

import pytest

from survey2agent.extraction.atoms import ExtractedAtom
from survey2agent.extraction.question_spec import QUESTIONS, SOURCE_NAMES
from survey2agent.methods import (
    SKIP_SENTINEL,
    MajorityClass,
    MajorityVote,
    OracleExtraction,
    Prediction,
    Random,
    SSB,
    SSBSelective,
)


# ── Synthetic-atom helpers ─────────────────────────────────────────────────


def _build_atom(persona: str, per_q_per_src: dict[str, dict[str, str | None]]) -> ExtractedAtom:
    """Build an ExtractedAtom from a partial spec, filling missing fields with None."""
    extraction: dict[str, dict[str, str | None]] = {}
    for qid in QUESTIONS:
        src_map = {src: None for src in SOURCE_NAMES}
        if qid in per_q_per_src:
            for src, val in per_q_per_src[qid].items():
                src_map[src] = val
        extraction[qid] = src_map
    return ExtractedAtom.from_json({"persona": persona, "extraction": extraction})


def _q_with_at_least(n_labels: int) -> str:
    """Return any qid whose answer_space has >= n_labels options."""
    for qid, q in QUESTIONS.items():
        if len(q["answer_space"]) >= n_labels:
            return qid
    raise RuntimeError(f"no question has >= {n_labels} answer-space labels")


# ── Random ─────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("qid", list(QUESTIONS.keys())[:5])
def test_random_returns_label_in_answer_space(qid: str) -> None:
    method = Random(seed=0)
    atom = _build_atom("p1", {})
    pred = method.predict_one(atom, qid)
    assert pred.answer in QUESTIONS[qid]["answer_space"]
    assert pred.would_skip is False


def test_random_seeded_is_deterministic() -> None:
    a, b = Random(seed=7), Random(seed=7)
    atom = _build_atom("p1", {})
    seq_a = [a.predict_one(atom, qid).answer for qid in QUESTIONS]
    seq_b = [b.predict_one(atom, qid).answer for qid in QUESTIONS]
    assert seq_a == seq_b


# ── MajorityClass ──────────────────────────────────────────────────────────


def test_majority_class_requires_fit_flag() -> None:
    assert MajorityClass.requires_fit is True
    assert MajorityClass.requires_calibration is False


def test_majority_class_predicts_train_mode() -> None:
    qid = "A1"
    space = QUESTIONS[qid]["answer_space"]
    target = space[0]
    other = space[1]
    records = [
        (_build_atom(f"p{i}", {}), {qid: target}) for i in range(3)
    ] + [
        (_build_atom("p3", {}), {qid: other}),
    ]
    method = MajorityClass()
    method.fit(records)
    pred = method.predict_one(_build_atom("test", {}), qid)
    assert pred.answer == target
    assert pred.would_skip is False


def test_majority_class_state_dict_roundtrip() -> None:
    qid = "A1"
    target = QUESTIONS[qid]["answer_space"][0]
    method = MajorityClass()
    method.fit([(_build_atom("p1", {}), {qid: target})])
    state = method.state_dict()
    restored = MajorityClass()
    restored.load_state_dict(state)
    assert restored.predict_one(_build_atom("p2", {}), qid).answer == target


# ── SSB ────────────────────────────────────────────────────────────────────


def test_ssb_picks_best_source_per_question() -> None:
    qid = "A1"
    space = QUESTIONS[qid]["answer_space"]
    target = space[0]
    distractor = space[1]
    # Construct a train batch where `objective_log` always matches GT and
    # other sources always disagree.
    winning_src = "objective_log"
    records = []
    for i in range(5):
        per_q = {qid: {src: distractor for src in SOURCE_NAMES}}
        per_q[qid][winning_src] = target
        records.append((_build_atom(f"p{i}", per_q), {qid: target}))
    method = SSB(seed=0)
    method.fit(records)
    assert method._best_source[qid] == winning_src
    test_atom = _build_atom("t", {qid: {winning_src: target}})
    pred = method.predict_one(test_atom, qid)
    assert pred.answer == target
    assert pred.would_skip is False


def test_ssb_random_fallback_when_best_source_null() -> None:
    qid = "A1"
    target = QUESTIONS[qid]["answer_space"][0]
    records = [
        (_build_atom(f"p{i}", {qid: {"objective_log": target}}), {qid: target})
        for i in range(3)
    ]
    method = SSB(seed=0)
    method.fit(records)
    # Test atom with no objective_log value -> random fallback (still a valid label).
    pred = method.predict_one(_build_atom("t", {}), qid)
    assert pred.answer in QUESTIONS[qid]["answer_space"]
    assert pred.would_skip is False


# ── SSBSelective ───────────────────────────────────────────────────────────


def test_ssb_selective_skips_when_below_theta() -> None:
    qid = "A1"
    space = QUESTIONS[qid]["answer_space"]
    target = space[0]
    other = space[1]
    winning_src = "objective_log"

    # Train: winning_src always correct.
    train = []
    for i in range(5):
        per_q = {qid: {src: other for src in SOURCE_NAMES}}
        per_q[qid][winning_src] = target
        train.append((_build_atom(f"tr{i}", per_q), {qid: target}))

    # Cal: build observations where low agreement (winner alone) is mostly
    # wrong, high agreement is mostly right -> theta should be > 0.
    cal = []
    # 5 records: winning_src says target, ALL other sources say `other`,
    # GT is `other` -> low-agree case is wrong.
    for i in range(5):
        per_q = {qid: {src: other for src in SOURCE_NAMES}}
        per_q[qid][winning_src] = target
        cal.append((_build_atom(f"cl{i}", per_q), {qid: other}))
    # 5 records: all sources agree on target -> high-agree case is right.
    for i in range(5):
        per_q = {qid: {src: target for src in SOURCE_NAMES}}
        cal.append((_build_atom(f"ch{i}", per_q), {qid: target}))

    method = SSBSelective(seed=0)
    method.fit(train)
    method.calibrate(cal)
    assert method._theta_agree > 0.0

    # Test atom matches the low-agree pattern -> SKIP.
    per_q = {qid: {src: other for src in SOURCE_NAMES}}
    per_q[qid][winning_src] = target
    test_atom = _build_atom("t", per_q)
    pred = method.predict_one(test_atom, qid)
    assert pred.answer == SKIP_SENTINEL
    assert pred.would_skip is True


def test_ssb_selective_skips_on_null_best_source() -> None:
    qid = "A1"
    target = QUESTIONS[qid]["answer_space"][0]
    train = [
        (_build_atom(f"p{i}", {qid: {"objective_log": target}}), {qid: target})
        for i in range(3)
    ]
    method = SSBSelective(seed=0)
    method.fit(train)
    method.calibrate(train)
    # Test atom without the chosen source -> C1 SKIP.
    pred = method.predict_one(_build_atom("t", {}), qid)
    assert pred.answer == SKIP_SENTINEL
    assert pred.would_skip is True


# ── MajorityVote ───────────────────────────────────────────────────────────


def test_majority_vote_returns_mode_of_active_sources() -> None:
    qid = "A1"
    space = QUESTIONS[qid]["answer_space"]
    winner = space[0]
    loser = space[1]
    per_q = {
        qid: {
            "objective_log": winner,
            "device_log": winner,
            "planner": winner,
            "daily_self_report": loser,
        }
    }
    method = MajorityVote(seed=0)
    pred = method.predict_one(_build_atom("p", per_q), qid)
    assert pred.answer == winner
    assert pred.would_skip is False


def test_majority_vote_handles_all_none() -> None:
    qid = "A1"
    method = MajorityVote(seed=0)
    pred = method.predict_one(_build_atom("p", {}), qid)
    assert pred.answer in QUESTIONS[qid]["answer_space"]
    assert pred.would_skip is False


# ── OracleExtraction ───────────────────────────────────────────────────────


def test_oracle_returns_attached_gt() -> None:
    qid = "A1"
    target = QUESTIONS[qid]["answer_space"][0]
    other = QUESTIONS[qid]["answer_space"][1]
    # Some source carries the GT.
    atom = _build_atom("p1", {qid: {"objective_log": target, "planner": other}})
    method = OracleExtraction()
    method.attach_gt({"p1": {qid: target}})
    pred = method.predict_one(atom, qid)
    assert pred.answer == target
    assert pred.would_skip is False


def test_oracle_falls_back_when_gt_missing_from_sources() -> None:
    qid = "A1"
    space = QUESTIONS[qid]["answer_space"]
    target = space[0]  # GT is target
    other = space[1]
    # No source carries GT; all say `other` -> majority vote -> other.
    atom = _build_atom("p1", {qid: {src: other for src in SOURCE_NAMES}})
    method = OracleExtraction(skip_on_miss=False)
    method.attach_gt({"p1": {qid: target}})
    pred = method.predict_one(atom, qid)
    assert pred.answer == other
    assert pred.would_skip is False


def test_oracle_skip_on_miss_variant_skips() -> None:
    qid = "A1"
    space = QUESTIONS[qid]["answer_space"]
    target = space[0]
    other = space[1]
    atom = _build_atom("p1", {qid: {src: other for src in SOURCE_NAMES}})
    method = OracleExtraction(skip_on_miss=True)
    method.attach_gt({"p1": {qid: target}})
    pred = method.predict_one(atom, qid)
    assert pred.answer == SKIP_SENTINEL
    assert pred.would_skip is True


# ── Base interface ─────────────────────────────────────────────────────────


def test_skip_sentinel_not_in_any_answer_space() -> None:
    for qid, q in QUESTIONS.items():
        assert SKIP_SENTINEL not in q["answer_space"], (
            f"answer_space for {qid} contains reserved sentinel {SKIP_SENTINEL!r}"
        )


def test_prediction_consistency_validates() -> None:
    with pytest.raises(ValueError):
        Prediction(answer=SKIP_SENTINEL, would_skip=False)
    with pytest.raises(ValueError):
        Prediction(answer="some_label", would_skip=True)
    # Valid constructions:
    Prediction(answer=SKIP_SENTINEL, would_skip=True)
    Prediction(answer="some_label", would_skip=False)
