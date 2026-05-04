"""End-to-end smoke test.

Exercises load-atom -> instantiate-method -> predict_one on real frozen
artifacts. Covers two methods that bracket the spectrum of training
requirements: `Random` (no fit, no calibration) and `MajorityClass`
(requires fit). Synthetic GT keeps the test self-contained (independent
of the v1.0 reference dataset tree).

Atom source is `data/sample/extracted_atoms/s20260321/` by default so
this smoke test continues to work after the full benchmark moves to
Hugging Face at release time. If the sample directory is unavailable,
falls back to the in-tree full atom set.
"""

from __future__ import annotations

import sys
from pathlib import Path

from survey2agent.extraction import load_atoms_from_dir
from survey2agent.extraction.atoms import ExtractedAtom
from survey2agent.extraction.question_spec import QUESTIONS
from survey2agent.methods import (
    SKIP_SENTINEL,
    MajorityClass,
    Prediction,
    Random,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_SAMPLE_ATOM_DIR = (
    _PROJECT_ROOT / "data" / "sample" / "extracted_atoms" / "s20260321"
)
_FULL_ATOM_DIR = (
    _PROJECT_ROOT / "data" / "extracted_atoms" / "s20260321"
)


def _resolve_atom_dir() -> Path:
    """Prefer the lightweight release sample; fall back to full atoms."""
    if _SAMPLE_ATOM_DIR.is_dir() and any(_SAMPLE_ATOM_DIR.glob("*.json")):
        print(
            "[smoke] Reading sample atoms from "
            f"{_SAMPLE_ATOM_DIR.relative_to(_PROJECT_ROOT)}. "
            "This verifies the lightweight release sample only; "
            "paper reproduction requires python data/fetch_benchmark.py.",
            file=sys.stderr,
        )
        return _SAMPLE_ATOM_DIR
    print(
        "[smoke] Sample atoms unavailable; falling back to full frozen "
        f"atoms under {_FULL_ATOM_DIR.relative_to(_PROJECT_ROOT)}.",
        file=sys.stderr,
    )
    return _FULL_ATOM_DIR


def _smoke_qids() -> list[str]:
    return list(QUESTIONS.keys())[:3]


def _check_prediction(pred: Prediction, qid: str) -> None:
    assert isinstance(pred, Prediction)
    assert isinstance(pred.answer, str)
    assert isinstance(pred.would_skip, bool)
    if pred.would_skip:
        assert pred.answer == SKIP_SENTINEL
    else:
        assert pred.answer != SKIP_SENTINEL
        assert pred.answer in QUESTIONS[qid]["answer_space"]


def _load_two_atoms() -> list[ExtractedAtom]:
    atom_dir = _resolve_atom_dir()
    atoms = load_atoms_from_dir(atom_dir)
    assert atoms, f"no atoms found under {atom_dir}"
    assert len(atoms) >= 2, (
        f"smoke test requires at least 2 atom files under {atom_dir}; "
        f"found {len(atoms)}"
    )
    keys = sorted(atoms.keys())[:2]
    return [atoms[k] for k in keys]


def test_smoke_end_to_end() -> None:
    atoms = _load_two_atoms()
    qids = _smoke_qids()

    rng_method = Random(seed=42)
    for atom in atoms:
        for qid in qids:
            _check_prediction(rng_method.predict_one(atom, qid), qid)

    synthetic_gt = {qid: QUESTIONS[qid]["answer_space"][0] for qid in QUESTIONS}
    train_records = [(atom, dict(synthetic_gt)) for atom in atoms]
    mc = MajorityClass()
    mc.fit(train_records)
    for atom in atoms:
        for qid in qids:
            pred = mc.predict_one(atom, qid)
            _check_prediction(pred, qid)
            assert pred.answer == QUESTIONS[qid]["answer_space"][0]
            assert pred.would_skip is False
