"""Tests for `BCF` (paper Forced-Accuracy Main Table, 4-parameter Bias-Corrected Fusion)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from survey2agent.extraction.atoms import ExtractedAtom
from survey2agent.extraction.question_spec import QUESTIONS, SOURCE_NAMES
from survey2agent.methods import BCF, Prediction, load_persona_gt
from survey2agent.methods.bcf import _atom_to_mu_all


from survey2agent._paths import EXTRACTED_ATOMS_ROOT, seed_dir


ATOM_DIR = EXTRACTED_ATOMS_ROOT / "s20260321"
GT_DIR = seed_dir("s20260321")


@pytest.fixture(scope="module")
def small_records() -> list[tuple[ExtractedAtom, dict[str, str]]]:
    """30 personas (10 per difficulty class) for fitting + evaluation."""
    if not ATOM_DIR.exists() or not GT_DIR.exists():
        pytest.skip("benchmark data not available")
    by_diff: dict[str, list[Path]] = {"stable": [], "shift": [], "stated": []}
    for p in sorted(ATOM_DIR.glob("*.json")):
        prefix = p.stem.split("_")[1]
        if prefix in by_diff and len(by_diff[prefix]) < 10:
            by_diff[prefix].append(p)
    out: list[tuple[ExtractedAtom, dict[str, str]]] = []
    for paths in by_diff.values():
        for atom_path in paths:
            with atom_path.open() as fh:
                atom = ExtractedAtom.from_json(json.load(fh))
            gt = load_persona_gt(GT_DIR / atom_path.stem)
            out.append((atom, gt))
    return out


def test_bcf_requires_fit_flag() -> None:
    assert BCF.requires_fit is True
    assert BCF.requires_calibration is False
    assert BCF.name == "BCF(4p)"


@pytest.mark.needs_data
def test_bcf_predicts_label_in_answer_space(
    small_records: list[tuple[ExtractedAtom, dict[str, str]]],
) -> None:
    train, evalset = small_records[:20], small_records[20:]
    method = BCF(seed=42)
    method.fit(train)
    for atom, _gt in evalset:
        for qid in QUESTIONS:
            pred = method.predict_one(atom, qid)
            assert isinstance(pred, Prediction)
            assert pred.answer in QUESTIONS[qid]["answer_space"]
            assert pred.would_skip is False


@pytest.mark.needs_data
def test_bcf_deltas_in_unit_interval(
    small_records: list[tuple[ExtractedAtom, dict[str, str]]],
) -> None:
    train = small_records[:20]
    method = BCF(seed=42)
    method.fit(train)
    # Grid is {0.0, 0.1, ..., 0.5} → δ_s ∈ [0, 0.5] for the 4 learnable
    # sources, and δ_obj is fixed at 0.
    assert method._deltas["objective_log"] == 0.0
    for src in ("profile_ltm", "planner", "daily_self_report", "device_log"):
        assert 0.0 <= method._deltas[src] <= 0.5


@pytest.mark.needs_data
def test_bcf_state_dict_roundtrip(
    small_records: list[tuple[ExtractedAtom, dict[str, str]]],
) -> None:
    train, evalset = small_records[:20], small_records[20:]
    method = BCF(seed=42)
    method.fit(train)
    state = json.loads(json.dumps(method.state_dict()))

    twin = BCF(seed=42)
    twin.load_state_dict(state)

    for atom, _gt in evalset:
        for qid in QUESTIONS:
            assert (
                method.predict_one(atom, qid).answer
                == twin.predict_one(atom, qid).answer
            )


