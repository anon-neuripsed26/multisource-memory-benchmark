"""Tests for evaluation.runner."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.needs_data

from survey2agent.evaluation.data_loaders import (
    build_training_records,
    load_atoms_for_seed,
    load_ground_truths,
    load_splits,
)
from survey2agent.evaluation.runner import EvaluationResult, run_method
from survey2agent.methods.majority_class import MajorityClass
from survey2agent.methods.random_baseline import Random
from survey2agent.selective.protocol import EvaluationOutcome


SEED = "s20260321"


def _test_records(qids=("A1", "A2", "A3")):
    splits = load_splits()
    atoms = load_atoms_for_seed(SEED)
    gts = load_ground_truths(SEED)
    return build_training_records(atoms, gts, splits["test"], qids=list(qids))


def test_run_method_random_no_fit():
    recs = _test_records()
    results = run_method(Random(seed=42), recs)
    assert len(results) == len(recs)
    for r in results:
        assert isinstance(r, EvaluationResult)
        assert isinstance(r.outcome, EvaluationOutcome)
        assert r.method_name == "Random"
        # Random never abstains.
        assert not r.prediction.would_skip
        assert r.outcome in {
            EvaluationOutcome.TRUE_ANSWER,
            EvaluationOutcome.FALSE_ANSWER,
        }


def test_run_method_majority_class_with_fit():
    recs = _test_records(qids=("A1", "A2"))
    # Train on the same test records as a stand-in (frozen set has no train atoms).
    results = run_method(MajorityClass(), recs, train_records=recs)
    assert len(results) == len(recs)
    # Each (persona, qid) pair appears once.
    pairs = {(r.persona_id, r.qid) for r in results}
    assert len(pairs) == len(results)


def test_run_method_missing_train_raises():
    recs = _test_records(qids=("A1",))
    with pytest.raises(ValueError, match="requires_fit"):
        run_method(MajorityClass(), recs)


def test_run_method_missing_cal_raises():
    """A method with requires_calibration=True must fail without cal_records."""
    recs = _test_records(qids=("A1",))

    class _NeedsCal(Random):
        name = "NeedsCal"
        requires_calibration = True

    with pytest.raises(ValueError, match="requires_calibration"):
        run_method(_NeedsCal(seed=0), recs)


def test_run_method_preserves_order():
    recs = _test_records(qids=("A1",))
    results = run_method(Random(seed=7), recs)
    for rec, res in zip(recs, results):
        assert res.persona_id == rec.atom.persona
        assert res.qid == rec.qid
        assert res.label == rec.label
