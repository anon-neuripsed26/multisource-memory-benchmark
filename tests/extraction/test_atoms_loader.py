"""Smoke tests for the extracted-atoms / method-prediction loader."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from survey2agent.extraction import (
    EXPECTED_QUESTION_IDS,
    EXPECTED_SOURCES,
    ExtractedAtom,
    MethodPrediction,
    load_atom,
    load_atoms_from_dir,
    load_method_prediction,
    load_method_predictions_from_dir,
)

REPO = Path(__file__).resolve().parents[2]  # repo root
ATOMS_ROOT = REPO / "data" / "extracted_atoms"
METHODS_ROOT = REPO / "data" / "method_outputs"

pytestmark = pytest.mark.needs_data


# ----- Atom schema --------------------------------------------------------- #


@pytest.mark.parametrize(
    "rel_path",
    [
        "s20260321/bench_stable_121_sam_bennett.json",
        "s20260321/bench_shift_121_avery_ellis.json",
        "s20260321/bench_stated_121_drew_garcia.json",
        "s20260324/bench_stable_121_sam_bennett.json",
        "s20260322/bench_shift_121_avery_ellis.json",
    ],
)
def test_extracted_atom_schema_validates(rel_path: str) -> None:
    atom = load_atom(ATOMS_ROOT / rel_path)
    assert isinstance(atom, ExtractedAtom)
    assert atom.persona  # nonempty
    assert set(atom.extraction.keys()) == set(EXPECTED_QUESTION_IDS)
    for qid in EXPECTED_QUESTION_IDS:
        assert set(atom.extraction[qid].keys()) == set(EXPECTED_SOURCES), (
            f"qid {qid} sources mismatch in {rel_path}"
        )


def test_extracted_atom_immutable() -> None:
    atom = load_atom(ATOMS_ROOT / "s20260321" / "bench_stable_121_sam_bennett.json")
    # frozen dataclass: cannot reassign top-level fields
    with pytest.raises((AttributeError, TypeError)):
        atom.extraction = {}  # type: ignore[misc]
    # MappingProxyType: cannot mutate inner mappings
    with pytest.raises(TypeError):
        atom.extraction["A1"] = {}  # type: ignore[index]
    with pytest.raises(TypeError):
        atom.extraction["A1"]["profile_ltm"] = "tampered"  # type: ignore[index]


def test_load_atoms_from_dir_count() -> None:
    atoms = load_atoms_from_dir(ATOMS_ROOT / "s20260321")
    assert len(atoms) == 120
    # Every key should match the persona inside the file
    for persona_id, atom in atoms.items():
        assert atom.persona == persona_id


def test_extracted_atom_rejects_missing_question() -> None:
    src = json.loads(
        (ATOMS_ROOT / "s20260321" / "bench_stable_121_sam_bennett.json").read_text(
            encoding="utf-8"
        )
    )
    bad = copy.deepcopy(src)
    del bad["extraction"]["A1"]
    with pytest.raises(ValueError, match=r"missing question A1"):
        ExtractedAtom.from_json(bad)


# ----- Method-prediction schema ------------------------------------------- #


@pytest.mark.parametrize(
    "rel_path",
    [
        "deepseek-v3.2/s20260321/direct/bench_stable_121_sam_bennett.json",
        "gemini_p2/s20260321/direct/bench_stable_121_sam_bennett.json",
        "qwen3-235b-a22b-2507/s20260321/schema-aware/bench_stable_121_sam_bennett.json",
    ],
)
def test_method_prediction_schema_validates(rel_path: str) -> None:
    pred = load_method_prediction(METHODS_ROOT / rel_path)
    assert isinstance(pred, MethodPrediction)
    assert pred.persona
    assert set(pred.answers.keys()) == set(EXPECTED_QUESTION_IDS)
    for qid in EXPECTED_QUESTION_IDS:
        entry = pred.answers[qid]
        assert "answer" in entry and isinstance(entry["answer"], str)
        assert "would_skip" in entry and isinstance(entry["would_skip"], bool)


def test_method_prediction_count() -> None:
    preds = load_method_predictions_from_dir(
        METHODS_ROOT / "deepseek-v3.2" / "s20260321" / "direct"
    )
    assert len(preds) == 120
    for persona_id, pred in preds.items():
        assert pred.persona == persona_id
