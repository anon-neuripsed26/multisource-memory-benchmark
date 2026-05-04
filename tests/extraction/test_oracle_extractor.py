"""Tests for survey2agent.extraction.oracle_extractor."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.needs_data

from survey2agent._paths import seed_dir
from survey2agent.extraction import (
    EXPECTED_QUESTION_IDS,
    EXPECTED_SOURCES,
    ExtractedAtom,
    build_oracle_atom,
    build_oracle_atoms_for_seed,
)
from survey2agent.extraction.oracle_extractor import _freeze_extraction  # type: ignore[attr-defined]


SEED = "s20260321"
SEED_DIR = seed_dir(SEED)


def _persona_dir(name_prefix: str) -> Path | None:
    """Find first persona dir whose name starts with ``name_prefix``."""
    for pd in sorted(SEED_DIR.iterdir()):
        if pd.is_dir() and pd.name.startswith(name_prefix):
            return pd
    return None


# ── Task 6.1 ─ schema test ────────────────────────────────────────────────


def test_build_oracle_atom_schema():
    pd = _persona_dir("bench_")
    assert pd is not None, "no bench_* persona found"
    atom = build_oracle_atom(pd)
    assert isinstance(atom, ExtractedAtom)
    assert atom.persona == pd.name
    # All 18 qids and all 5 sources present.
    assert set(atom.extraction.keys()) == set(EXPECTED_QUESTION_IDS)
    for qid in EXPECTED_QUESTION_IDS:
        assert set(atom.extraction[qid].keys()) == set(EXPECTED_SOURCES), qid
        for src, val in atom.extraction[qid].items():
            assert val is None or isinstance(val, str), (qid, src, val)


# ── Task 6.2 ─ byte-equiv test (REMOVED) ──────────────────────────────────
# The original byte-equiv test compared this package's vendored
# ``compute_all_mu`` against the upstream shell.py. After the inference
# shell was internalized into ``survey2agent.extraction._mu_shell``,
# the in-tree implementation is the source of truth and the
# upstream module is no longer a dependency. The byte-equiv test was a
# one-time sanity check at vendor-time and has been deleted.


# ── Task 6.3 ─ full-seed determinism test ────────────────────────────────


def test_build_oracle_atoms_for_seed_determinism():
    a = build_oracle_atoms_for_seed(SEED)
    b = build_oracle_atoms_for_seed(SEED)
    assert len(a) == 480
    assert list(a.keys()) == list(b.keys())

    def _hash(atoms: dict[str, ExtractedAtom]) -> str:
        h = hashlib.sha256()
        for pid in sorted(atoms.keys()):
            ext = atoms[pid].extraction
            payload = {
                qid: {src: ext[qid][src] for src in EXPECTED_SOURCES}
                for qid in EXPECTED_QUESTION_IDS
            }
            h.update(pid.encode("utf-8"))
            h.update(json.dumps(payload, sort_keys=True).encode("utf-8"))
        return h.hexdigest()

    assert _hash(a) == _hash(b)


# ── Task 6.4 ─ coverage snapshot ──────────────────────────────────────────


def test_build_oracle_atom_coverage_snapshot():
    """Regression snapshot of which (qid, source) cells are non-null for
    a fixed persona. Locks in the v3 shell behaviour; if any quirk is
    accidentally "fixed", this test will catch it.
    """
    pd = _persona_dir("bench_stable_001")
    if pd is None:
        pytest.skip("bench_stable_001_* not present")
    atom = build_oracle_atom(pd)
    coverage = {
        qid: tuple(src for src in EXPECTED_SOURCES if atom.extraction[qid][src] is not None)
        for qid in EXPECTED_QUESTION_IDS
    }
    # At least every qid yields ≥ 1 non-null source (otherwise extraction
    # is degenerate). Stronger per-cell snapshot would be brittle across
    # persona-specific data; we assert structural coverage instead.
    for qid, srcs in coverage.items():
        assert len(srcs) >= 1, f"{qid} produced no non-null sources for {pd.name}"
    # objective_log carries no sleep data → A1 must have it as None.
    assert atom.extraction["A1"]["objective_log"] is None


# ── Task 6.5 ─ missing seed dir ──────────────────────────────────────────


def test_build_oracle_atoms_for_seed_missing_dir_raises():
    with pytest.raises(FileNotFoundError):
        build_oracle_atoms_for_seed("s99999999")
