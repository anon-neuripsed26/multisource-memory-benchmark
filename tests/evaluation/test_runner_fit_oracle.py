"""End-to-end runner tests using oracle ExtractedAtoms.

Verifies that fit-required methods (BCF, NBFSelective, ABFSelective) can
consume the deterministic 480-persona oracle atoms produced by
:func:`survey2agent.evaluation.data_loaders.load_atoms_for_seed` with
``mode="oracle"``. Smoke tests only — assert the runner returns a result
per ``(persona, qid)`` and does not crash during fit / calibrate / predict.
"""

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
from survey2agent.extraction.atoms import EXPECTED_QUESTION_IDS
from survey2agent.methods.abf import ABFSelective
from survey2agent.methods.bcf import BCF
from survey2agent.methods.nbf import NBFSelective


SEED = "s20260321"
# Methods iterate all 18 qids during fit; partial GT triggers KeyError. Use full set.
QIDS = EXPECTED_QUESTION_IDS


@pytest.fixture(scope="module")
def oracle_records():
    splits = load_splits()
    atoms = load_atoms_for_seed(SEED, mode="oracle")
    assert len(atoms) == 480, f"oracle mode should return 480 atoms, got {len(atoms)}"
    gts = load_ground_truths(SEED)
    train = build_training_records(atoms, gts, splits["train"], qids=list(QIDS))
    cal = build_training_records(atoms, gts, splits["cal"], qids=list(QIDS))
    test = build_training_records(atoms, gts, splits["test"], qids=list(QIDS))
    return train, cal, test


def _check_results(results: list[EvaluationResult], n_expected: int) -> None:
    assert len(results) == n_expected
    for r in results:
        assert isinstance(r, EvaluationResult)
        assert r.qid in QIDS
        assert r.prediction is not None


def test_bcf_end_to_end_on_oracle_atoms(oracle_records):
    train, _cal, test = oracle_records
    results = run_method(BCF(), test, train_records=train)
    _check_results(results, len(test))


def test_nbf_selective_end_to_end_on_oracle_atoms(oracle_records):
    train, cal, test = oracle_records
    results = run_method(NBFSelective(), test, train_records=train, cal_records=cal)
    _check_results(results, len(test))


def test_abf_selective_end_to_end_on_oracle_atoms(oracle_records):
    train, cal, test = oracle_records
    results = run_method(
        ABFSelective(), test, train_records=train, cal_records=cal
    )
    _check_results(results, len(test))
