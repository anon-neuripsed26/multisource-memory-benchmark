"""Tests for methods._atom_adapter (ExtractedAtom → mu_q)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from survey2agent.extraction.atoms import ExtractedAtom
from survey2agent.extraction.question_spec import QUESTIONS, SOURCE_NAMES
from survey2agent.methods import atom_to_mu_q, iter_mu_q_per_question

_FIXTURE_PATH: Path = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "extracted_atoms"
    / "s20260321"
    / "bench_shift_121_avery_ellis.json"
)

pytestmark = pytest.mark.needs_data


def _load_atom() -> ExtractedAtom:
    with _FIXTURE_PATH.open("r", encoding="utf-8") as fh:
        raw = json.load(fh)
    return ExtractedAtom.from_json(raw)


def test_adapter_returns_dict_keyed_by_5_sources():
    atom = _load_atom()
    mu_q = atom_to_mu_q(atom, "A1")
    assert set(mu_q.keys()) == set(SOURCE_NAMES)
    assert isinstance(mu_q, dict)


def test_adapter_handles_null_values():
    # Construct a synthetic atom with explicit None values to lock behavior
    # independently of fixture contents.
    raw = {
        "persona": "synthetic",
        "extraction": {
            qid: {src: None for src in SOURCE_NAMES} for qid in QUESTIONS
        },
    }
    # Ensure at least one non-None value too so the fixture exercises both.
    raw["extraction"]["A1"]["objective_log"] = "10_to_19"
    atom = ExtractedAtom.from_json(raw)
    mu_q = atom_to_mu_q(atom, "A1")
    assert mu_q["objective_log"] == "10_to_19"
    for src in SOURCE_NAMES:
        if src == "objective_log":
            continue
        assert mu_q[src] is None, f"source {src} should be None, got {mu_q[src]!r}"


def test_iter_mu_q_per_question_yields_18_entries():
    atom = _load_atom()
    entries = list(iter_mu_q_per_question(atom))
    assert len(entries) == 18
    assert [qid for qid, _ in entries] == list(QUESTIONS.keys())
    for qid, mu_q in entries:
        assert set(mu_q.keys()) == set(SOURCE_NAMES)
