"""Per-instance metadata enrichment for EvaluationResult.

Verifies the prerequisite for per-axis breakdown utilities:
:class:`EvaluationResult` carries
``reasoning_type``, ``topic``, and ``difficulty_class`` populated from
``configs/questions.yaml`` and the persona spec, so downstream breakdown
utilities can group results without re-loading upstream specs.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest

pytestmark = pytest.mark.needs_data

from survey2agent._paths import seed_dir
from survey2agent.evaluation.data_loaders import (
    build_training_records,
    load_atoms_for_seed,
    load_ground_truths,
    load_persona_difficulty_index,
    load_splits,
)
from survey2agent.evaluation.runner import EvaluationResult, run_method
from survey2agent.extraction.question_spec import QUESTIONS
from survey2agent.methods.base import Prediction
from survey2agent.methods.random_baseline import Random
from survey2agent.selective.protocol import EvaluationOutcome


SEED = "s20260321"


# ── EvaluationResult metadata field requirements ──────────────────────────


def test_evaluation_result_requires_metadata_fields():
    """Omitting any of the three new fields raises TypeError."""
    pred = Prediction(answer="X", would_skip=False)
    common = dict(
        method_name="m",
        persona_id="p",
        qid="A1",
        prediction=pred,
        label="X",
        outcome=EvaluationOutcome.TRUE_ANSWER,
    )
    for missing in ("reasoning_type", "topic", "difficulty_class"):
        kwargs = {
            "reasoning_type": "arbitration",
            "topic": "sleep",
            "difficulty_class": "stable",
            **common,
        }
        del kwargs[missing]
        with pytest.raises(TypeError):
            EvaluationResult(**kwargs)


def test_evaluation_result_accepts_metadata_fields():
    pred = Prediction(answer="X", would_skip=False)
    r = EvaluationResult(
        method_name="m",
        persona_id="p",
        qid="A1",
        prediction=pred,
        label="X",
        outcome=EvaluationOutcome.TRUE_ANSWER,
        reasoning_type="arbitration",
        topic="sleep",
        difficulty_class="stable",
    )
    assert r.reasoning_type == "arbitration"
    assert r.topic == "sleep"
    assert r.difficulty_class == "stable"


# ── load_persona_difficulty_index ─────────────────────────────────────────


def test_load_persona_difficulty_index_size_and_distribution():
    idx = load_persona_difficulty_index(SEED)
    assert len(idx) == 480
    counts = Counter(idx.values())
    assert counts == {"stable": 160, "temporal_shift": 160, "stated_vs_revealed": 160}


def test_load_persona_difficulty_index_matches_personas_json():
    """Spot-check values against the raw personas.json source of truth."""
    raw_path: Path = seed_dir(SEED) / "config" / "personas.json"
    with raw_path.open("r", encoding="utf-8") as fh:
        raw = json.load(fh)
    expected = {p["id"]: p["difficulty_type"] for p in raw["personas"]}
    idx = load_persona_difficulty_index(SEED)
    assert idx == expected


def test_load_persona_difficulty_index_default_seed_matches_explicit():
    """Persona ids are seed-stable; default arg should match explicit s20260321."""
    assert load_persona_difficulty_index() == load_persona_difficulty_index(SEED)


def test_load_persona_difficulty_index_unknown_seed_raises():
    with pytest.raises(FileNotFoundError):
        load_persona_difficulty_index("s99999999")


# ── build_training_records propagates metadata ────────────────────────────


def test_build_training_records_populates_metadata():
    splits = load_splits()
    atoms = load_atoms_for_seed(SEED)
    gts = load_ground_truths(SEED)
    diff_idx = load_persona_difficulty_index(SEED)

    records = build_training_records(
        atoms, gts, splits["test"], qids=["A1", "B2", "Ctrl1"]
    )
    assert len(records) > 0
    for r in records:
        # qid → reasoning_type / topic must come from QUESTIONS, not be defaulted.
        spec = QUESTIONS[r.qid]
        assert r.reasoning_type == spec["type"]
        assert r.topic == spec["topic"]
        # difficulty_class must match the persona spec.
        assert r.difficulty_class == diff_idx[r.atom.persona]
        assert r.difficulty_class in {"stable", "temporal_shift", "stated_vs_revealed"}


# ── run_method propagates metadata into EvaluationResult ──────────────────


def test_run_method_propagates_metadata_to_results():
    splits = load_splits()
    atoms = load_atoms_for_seed(SEED)
    gts = load_ground_truths(SEED)
    diff_idx = load_persona_difficulty_index(SEED)

    recs = build_training_records(atoms, gts, splits["test"], qids=["A1", "C2", "F3"])
    results = run_method(Random(seed=0), recs)
    assert len(results) == len(recs)

    for rec, res in zip(recs, results):
        assert res.reasoning_type == rec.reasoning_type
        assert res.topic == rec.topic
        assert res.difficulty_class == rec.difficulty_class
        # And it matches the upstream specs directly.
        spec = QUESTIONS[res.qid]
        assert res.reasoning_type == spec["type"]
        assert res.topic == spec["topic"]
        assert res.difficulty_class == diff_idx[res.persona_id]


def test_run_method_metadata_spot_check_known_persona():
    """A1 is arbitration/sleep; pick a test-split ``bench_stable_*`` persona
    and verify all three metadata fields land correctly on the result."""
    splits = load_splits()
    atoms = load_atoms_for_seed(SEED)
    gts = load_ground_truths(SEED)
    diff_idx = load_persona_difficulty_index(SEED)

    target_pid = next(
        (
            pid
            for pid in splits["test"]
            if pid in atoms
            and pid in gts
            and diff_idx.get(pid) == "stable"
        ),
        None,
    )
    if target_pid is None:
        pytest.skip("no stable persona available in test split + atoms + gts")

    recs = build_training_records(atoms, gts, [target_pid], qids=["A1"])
    assert len(recs) == 1
    results = run_method(Random(seed=0), recs)
    res = results[0]
    assert res.persona_id == target_pid
    assert res.qid == "A1"
    assert res.reasoning_type == "arbitration"
    assert res.topic == "sleep"
    assert res.difficulty_class == "stable"
