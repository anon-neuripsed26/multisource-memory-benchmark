"""Tests for DSNBF and DSNBFSelective (T2 fusion, paper flagship)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from survey2agent.extraction.atoms import ExtractedAtom
from survey2agent.extraction.question_spec import QUESTIONS, SOURCE_NAMES
from survey2agent.methods import (
    DSNBF,
    DSNBFSelective,
    MethodTrainingRecord,
    SKIP_SENTINEL,
    load_persona_gt,
)


from survey2agent._paths import EXTRACTED_ATOMS_ROOT, seed_dir

ATOM_DIR = EXTRACTED_ATOMS_ROOT / "s20260321"
GT_DIR = seed_dir("s20260321")


# ── Fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def small_records() -> list[tuple[ExtractedAtom, dict[str, str]]]:
    """Load the first 30 real personas (10 per difficulty class) for fitting."""
    if not ATOM_DIR.exists() or not GT_DIR.exists():
        pytest.skip("benchmark data not available in this environment")
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


def test_dsnbf_requires_fit_flag() -> None:
    assert DSNBF.requires_fit is True
    assert DSNBF.requires_calibration is True
    assert DSNBFSelective.requires_fit is True
    assert DSNBFSelective.requires_calibration is True


@pytest.mark.needs_data
def test_dsnbf_predicts_label_in_answer_space(
    small_records: list[tuple[ExtractedAtom, dict[str, str]]],
) -> None:
    train, evalset = small_records[:20], small_records[20:]
    method = DSNBF(seed=42)
    method.fit(train)
    for atom, _gt in evalset:
        for qid in QUESTIONS:
            pred = method.predict_one(atom, qid)
            assert pred.answer in QUESTIONS[qid]["answer_space"]
            assert pred.would_skip is False


@pytest.mark.needs_data
def test_dsnbf_state_dict_roundtrip(
    small_records: list[tuple[ExtractedAtom, dict[str, str]]],
) -> None:
    train, evalset = small_records[:20], small_records[20:]
    method = DSNBF(seed=42)
    method.fit(train)
    state = method.state_dict()
    # Round-trip through JSON to enforce serialisability.
    restored_state = json.loads(json.dumps(state))

    twin = DSNBF(seed=42)
    twin.load_state_dict(restored_state)

    for atom, _gt in evalset:
        for qid in QUESTIONS:
            assert (
                method.predict_one(atom, qid).answer
                == twin.predict_one(atom, qid).answer
            )


@pytest.mark.needs_data
def test_dsnbf_uses_emission_temperature_to_change_predictions(
    small_records: list[tuple[ExtractedAtom, dict[str, str]]],
) -> None:
    """Bumping T (sharper emissions) must change at least one prediction.

    This is a sanity check that the calibrated hyperparameters actually feed
    through into the inference path (catches accidentally-frozen values).
    """
    train, evalset = small_records[:20], small_records[20:]
    method_a = DSNBF(seed=42)
    method_a.fit(train)
    method_b = DSNBF(seed=42)
    method_b.fit(train)
    method_b._T = 5.0  # very sharp emissions
    method_b._gw = 1.0  # pure global path so T fully drives the score

    differs = False
    for atom, _gt in evalset:
        for qid in QUESTIONS:
            a = method_a.predict_one(atom, qid).answer
            b = method_b.predict_one(atom, qid).answer
            if a != b:
                differs = True
                break
        if differs:
            break
    assert differs, "DSNBF predictions invariant to T — inference path broken"


@pytest.mark.needs_data
def test_dsnbf_selective_calibrate_yields_in_range_theta(
    small_records: list[tuple[ExtractedAtom, dict[str, str]]],
) -> None:
    train, cal = small_records[:20], small_records[20:]
    method = DSNBFSelective(seed=42)
    method.fit(train)
    method.calibrate(cal)
    assert 0.0 <= method._theta <= 1.0
    # Calibrated hyperparams must land inside their declared grids.
    assert 0.3 <= method._T <= 2.0
    assert 0.1 <= method._T_diff <= 2.0
    assert 0.0 <= method._gw <= 1.0


@pytest.mark.needs_data
def test_dsnbf_selective_skips_when_theta_high(
    small_records: list[tuple[ExtractedAtom, dict[str, str]]],
) -> None:
    """With theta forced to 1.0, every multi-source case must SKIP."""
    train, evalset = small_records[:20], small_records[20:]
    method = DSNBFSelective(seed=42)
    method.fit(train)
    method._theta = 1.0  # force SKIP on every margin < 1

    saw_skip = False
    for atom, _gt in evalset:
        for qid in QUESTIONS:
            n_act = sum(
                1 for s in SOURCE_NAMES if atom.extraction[qid][s] is not None
            )
            pred = method.predict_one(atom, qid)
            if n_act >= 2:
                # Margin between top-1 and top-2 cannot reach 1.0 in
                # practice — the multi-source case must abstain.
                assert pred.answer == SKIP_SENTINEL
                assert pred.would_skip is True
                saw_skip = True
            else:
                assert pred.answer in QUESTIONS[qid]["answer_space"]
                assert pred.would_skip is False
    assert saw_skip, "expected at least one multi-source case in eval set"


@pytest.mark.needs_data
def test_dsnbf_warns_on_nonconforming_persona_id(
    small_records: list[tuple[ExtractedAtom, dict[str, str]]],
) -> None:
    """Custom personas not matching `bench_<prefix>_NNN_<name>` must
    fall back to ``stable`` AND emit a one-time RuntimeWarning so misuse
    is visible without crashing the run."""
    atom, gt = small_records[0]
    # Build an atom whose persona id does not start with `bench_<prefix>_`.
    bad_atom = ExtractedAtom.from_json({
        "persona": "custom_persona_42",
        "extraction": {qid: dict(atom.extraction[qid]) for qid in QUESTIONS},
    })
    train = [(bad_atom, gt)]
    method = DSNBF(seed=0)
    with pytest.warns(RuntimeWarning, match="bench_<stable\\|shift\\|stated>"):
        method.fit(train)


@pytest.mark.needs_data
def test_dsnbf_prefers_runner_difficulty_metadata(
    small_records: list[tuple[ExtractedAtom, dict[str, str]]],
) -> None:
    """Official runner metadata must avoid parsing difficulty from persona_id."""
    atom, gt = small_records[0]
    custom_atom = ExtractedAtom.from_json({
        "persona": "custom_persona_42",
        "extraction": {qid: dict(atom.extraction[qid]) for qid in QUESTIONS},
    })
    record = MethodTrainingRecord(
        atom=custom_atom,
        gt=gt,
        difficulty_class="temporal_shift",
    )

    method = DSNBF(seed=0)
    method.fit([record])

    assert method._train_cache is not None
    assert method._train_cache[0]["difficulty"] == "temporal_shift"
