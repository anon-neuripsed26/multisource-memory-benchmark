"""Tests for evaluation.data_loaders."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.needs_data

from survey2agent.evaluation.data_loaders import (
    TrainingRecord,
    build_training_records,
    load_atoms_for_seed,
    load_ground_truths,
    load_splits,
)
from survey2agent.extraction.atoms import EXPECTED_QUESTION_IDS, ExtractedAtom


SEED = "s20260321"


def test_load_splits_keys_and_sizes():
    splits = load_splits()
    assert set(splits.keys()) == {"train", "dev", "cal", "test"}
    sizes = {k: len(v) for k, v in splits.items()}
    # Canonical 480 → 216/48/96/120.
    assert sizes == {"train": 216, "dev": 48, "cal": 96, "test": 120}
    total = sum(sizes.values())
    assert total <= 480 and total == 480
    # No persona appears in two splits.
    seen: set[str] = set()
    for ids in splits.values():
        for pid in ids:
            assert pid not in seen, pid
            seen.add(pid)


def test_load_atoms_for_seed_returns_test_split():
    atoms = load_atoms_for_seed(SEED)
    # In the frozen artifact set only the 120 test-split atoms are present.
    assert len(atoms) == 120
    sample_pid, sample_atom = next(iter(atoms.items()))
    assert isinstance(sample_atom, ExtractedAtom)
    assert sample_atom.persona == sample_pid


def test_load_atoms_for_seed_llm_explicit_matches_default():
    """``mode="llm"`` is the default; explicit form yields identical output."""
    a = load_atoms_for_seed(SEED)
    b = load_atoms_for_seed(SEED, mode="llm")
    assert set(a.keys()) == set(b.keys()) and len(a) == 120


def test_load_atoms_for_seed_oracle_returns_full_480():
    atoms = load_atoms_for_seed(SEED, mode="oracle")
    assert len(atoms) == 480
    sample_pid, sample_atom = next(iter(atoms.items()))
    assert isinstance(sample_atom, ExtractedAtom)
    assert sample_atom.persona == sample_pid


def test_load_atoms_for_seed_invalid_mode_raises():
    with pytest.raises(ValueError, match="mode must be"):
        load_atoms_for_seed(SEED, mode="bogus")  # type: ignore[arg-type]


def test_load_ground_truths_shape():
    gts = load_ground_truths(SEED)
    # 480 personas, each with all 18 qids.
    assert len(gts) == 480
    for pid, per_qid in gts.items():
        assert set(per_qid.keys()) == set(EXPECTED_QUESTION_IDS), pid
        for v in per_qid.values():
            assert isinstance(v, str)


def test_build_training_records_test_split_size():
    splits = load_splits()
    atoms = load_atoms_for_seed(SEED)
    gts = load_ground_truths(SEED)
    records = build_training_records(atoms, gts, splits["test"])
    assert len(records) == 120 * 18
    # All carry valid qids and labels.
    for r in records[:25]:
        assert r.qid in EXPECTED_QUESTION_IDS
        assert isinstance(r.label, str) and r.label
        assert isinstance(r.atom, ExtractedAtom)


def test_build_training_records_filters_missing_atoms():
    splits = load_splits()
    gts = load_ground_truths(SEED)
    atoms = load_atoms_for_seed(SEED)
    # Train personas have GT but no atoms in the frozen set → 0 records.
    records = build_training_records(atoms, gts, splits["train"])
    assert records == []


def test_build_training_records_qid_filter():
    splits = load_splits()
    atoms = load_atoms_for_seed(SEED)
    gts = load_ground_truths(SEED)
    records = build_training_records(atoms, gts, splits["test"], qids=["A1", "A2"])
    assert len(records) == 120 * 2
    assert {r.qid for r in records} == {"A1", "A2"}


def test_training_record_is_frozen():
    splits = load_splits()
    atoms = load_atoms_for_seed(SEED)
    gts = load_ground_truths(SEED)
    rec = build_training_records(atoms, gts, splits["test"], qids=["A1"])[0]
    assert isinstance(rec, TrainingRecord)
    with pytest.raises((AttributeError, Exception)):
        rec.qid = "Z9"  # type: ignore[misc]
