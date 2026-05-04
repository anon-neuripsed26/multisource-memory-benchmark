"""Tests for `ABF` and `ABFSelective`."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from survey2agent.extraction.atoms import ExtractedAtom
from survey2agent.extraction.question_spec import QUESTIONS, SOURCE_NAMES
from survey2agent.methods import (
    ABF,
    ABFSelective,
    SKIP_SENTINEL,
    load_persona_gt,
)
from survey2agent.methods.abf import _atom_to_mu_all


from survey2agent._paths import EXTRACTED_ATOMS_ROOT, seed_dir

ATOM_DIR = EXTRACTED_ATOMS_ROOT / "s20260321"
GT_DIR = seed_dir("s20260321")


@pytest.fixture(scope="module")
def small_records() -> list[tuple[ExtractedAtom, dict[str, str]]]:
    """30 personas (10 per difficulty class) — train/cal/eval slices."""
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


# ── Tests ─────────────────────────────────────────────────────────────────


def test_abf_capability_flags() -> None:
    assert ABF.requires_fit is True
    assert ABF.requires_calibration is False
    assert ABFSelective.requires_fit is True
    assert ABFSelective.requires_calibration is True
    assert ABF.name == "ABF"
    assert ABFSelective.name == "ABF+SKIP"


@pytest.mark.needs_data
def test_abf_predicts_label_in_answer_space(
    small_records: list[tuple[ExtractedAtom, dict[str, str]]],
) -> None:
    train, evalset = small_records[:20], small_records[20:]
    method = ABF(seed=42)
    method.fit(train)
    for atom, _gt in evalset:
        for qid in QUESTIONS:
            pred = method.predict_one(atom, qid)
            assert pred.answer in QUESTIONS[qid]["answer_space"]
            assert pred.would_skip is False


@pytest.mark.needs_data
def test_abf_learned_params_in_bounds(
    small_records: list[tuple[ExtractedAtom, dict[str, str]]],
) -> None:
    train = small_records[:20]
    method = ABF(seed=42)
    method.fit(train)
    for s in ("profile_ltm", "planner", "daily_self_report", "device_log"):
        assert 0.0 <= method._deltas[s] <= 0.5
    assert method._deltas["objective_log"] == 0.0
    assert 0.5 <= method._alpha <= 10.0
    assert 0.0 <= method._pi <= 1.0


@pytest.mark.needs_data
def test_abf_state_dict_roundtrip(
    small_records: list[tuple[ExtractedAtom, dict[str, str]]],
) -> None:
    train, evalset = small_records[:20], small_records[20:]
    method = ABF(seed=42)
    method.fit(train)
    restored_state = json.loads(json.dumps(method.state_dict()))

    twin = ABF(seed=42)
    twin.load_state_dict(restored_state)

    for atom, _gt in evalset:
        for qid in QUESTIONS:
            assert (
                method.predict_one(atom, qid).answer
                == twin.predict_one(atom, qid).answer
            )


@pytest.mark.needs_data
def test_abf_selective_state_dict_includes_thetas_and_roundtrips(
    small_records: list[tuple[ExtractedAtom, dict[str, str]]],
) -> None:
    train, cal = small_records[:20], small_records[20:]
    method = ABFSelective(seed=42)
    method.fit(train)
    method.calibrate(cal)
    state = method.state_dict()
    assert "theta_E" in state
    assert "theta_delta" in state
    assert "deltas" in state and "alpha" in state and "pi" in state
    # legacy v3 compat field must be gone.
    assert "_theta_ED" not in state and "theta_ED" not in state

    restored = json.loads(json.dumps(state))
    twin = ABFSelective(seed=42)
    twin.load_state_dict(restored)
    for atom, _gt in cal:
        for qid in QUESTIONS:
            assert (
                method.predict_one(atom, qid).answer
                == twin.predict_one(atom, qid).answer
            )


@pytest.mark.needs_data
def test_abf_selective_calibrated_thetas_in_grid_range(
    small_records: list[tuple[ExtractedAtom, dict[str, str]]],
) -> None:
    train, cal = small_records[:20], small_records[20:]
    method = ABFSelective(seed=42)
    method.fit(train)
    method.calibrate(cal)
    assert 0.0 <= method._theta_E <= 5.0
    assert 0.0 <= method._theta_delta <= 2.0
    # Thetas must land on the declared grid (0.5 / 0.2 spacing respectively).
    assert abs((method._theta_E * 2) - round(method._theta_E * 2)) < 1e-9
    assert abs((method._theta_delta * 5) - round(method._theta_delta * 5)) < 1e-9


@pytest.mark.needs_data
def test_abf_selective_zero_active_always_skips(
    small_records: list[tuple[ExtractedAtom, dict[str, str]]],
) -> None:
    """|S|=0 → SKIP regardless of θ_E / θ_Δ."""
    train, cal = small_records[:20], small_records[20:]
    method = ABFSelective(seed=42)
    method.fit(train)
    method.calibrate(cal)

    # Build an atom whose extraction is all-None.
    template = small_records[0][0]
    blank_extraction = {
        qid: {src: None for src in SOURCE_NAMES} for qid in QUESTIONS
    }
    blank_atom = ExtractedAtom.from_json({
        "persona": template.persona,
        "extraction": blank_extraction,
    })
    # Even with θ_E and θ_Δ forced to 0 (most permissive), zero evidence must SKIP.
    method._theta_E = 0.0
    method._theta_delta = 0.0
    for qid in QUESTIONS:
        pred = method.predict_one(blank_atom, qid)
        assert pred.answer == SKIP_SENTINEL
        assert pred.would_skip is True


@pytest.mark.needs_data
def test_abf_selective_single_active_never_skips(
    small_records: list[tuple[ExtractedAtom, dict[str, str]]],
) -> None:
    """|S|=1 → never SKIP regardless of θ_E / θ_Δ."""
    train, cal = small_records[:20], small_records[20:]
    method = ABFSelective(seed=42)
    method.fit(train)
    method.calibrate(cal)
    method._theta_E = 99.0   # absurdly high: would SKIP everything if not guarded
    method._theta_delta = 99.0

    # Build an atom with exactly one active source per question (objective_log).
    src_keep = "objective_log"
    template = small_records[0][0]
    single_extraction = {}
    for qid in QUESTIONS:
        row = {src: None for src in SOURCE_NAMES}
        # Inject *some* observation if the template had one for objective_log.
        v = template.extraction[qid][src_keep]
        if v is None:
            # Fall back to the question's first answer — produces a valid label.
            v = QUESTIONS[qid]["answer_space"][0]
        row[src_keep] = v
        single_extraction[qid] = row
    single_atom = ExtractedAtom.from_json({
        "persona": template.persona,
        "extraction": single_extraction,
    })

    for qid in QUESTIONS:
        pred = method.predict_one(single_atom, qid)
        assert pred.answer in QUESTIONS[qid]["answer_space"]
        assert pred.would_skip is False

