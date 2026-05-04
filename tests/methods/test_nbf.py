"""Tests for `NBF` (paper Forced-Accuracy Main Table NoSkip) and `NBFSelective` (paper 2x2 Factorial Decomposition)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from survey2agent.extraction.atoms import ExtractedAtom
from survey2agent.extraction.question_spec import QUESTIONS, SOURCE_NAMES
from survey2agent.methods import (
    NBF,
    SKIP_SENTINEL,
    NBFSelective,
    Prediction,
    load_persona_gt,
)
from survey2agent.methods.nbf import _atom_to_mu_all


from survey2agent._paths import EXTRACTED_ATOMS_ROOT, seed_dir


ATOM_DIR = EXTRACTED_ATOMS_ROOT / "s20260321"
GT_DIR = seed_dir("s20260321")


@pytest.fixture(scope="module")
def small_records() -> list[tuple[ExtractedAtom, dict[str, str]]]:
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


def test_nbf_requires_fit_flag() -> None:
    assert NBF.requires_fit is True
    assert NBF.name == "NBF"


def test_nbfselective_requires_calibration_flag() -> None:
    assert NBFSelective.requires_fit is True
    assert NBFSelective.requires_calibration is True
    assert NBFSelective.name == "NBF+SKIP"


@pytest.mark.needs_data
def test_nbf_predicts_label_in_answer_space(
    small_records: list[tuple[ExtractedAtom, dict[str, str]]],
) -> None:
    train, cal, evalset = (
        small_records[:15], small_records[15:22], small_records[22:],
    )
    method = NBF(seed=42)
    method.fit(train)
    method.calibrate(cal)
    for atom, _gt in evalset:
        for qid in QUESTIONS:
            pred = method.predict_one(atom, qid)
            assert isinstance(pred, Prediction)
            assert pred.answer in QUESTIONS[qid]["answer_space"]
            assert pred.would_skip is False


@pytest.mark.needs_data
def test_nbf_temperature_calibrated_to_macro_acc(
    small_records: list[tuple[ExtractedAtom, dict[str, str]]],
) -> None:
    """After `calibrate`, T must lie inside the {0.1, ..., 3.0} grid and
    pick a value that maximises macro-acc on the cal split (we sanity-check
    by ensuring at least one off-grid T does not beat the chosen one)."""
    train, cal = small_records[:15], small_records[15:]
    method = NBF(seed=42)
    method.fit(train)
    method.calibrate(cal)
    assert 0.1 <= method._temperature <= 3.0
    chosen = method._temperature

    def _macro_acc(T: float) -> float:
        method._temperature = T
        q_correct: dict[str, int] = {}
        q_total: dict[str, int] = {}
        for atom, gt in cal:
            mu_all = _atom_to_mu_all(atom)
            for qid in QUESTIONS:
                q_total[qid] = q_total.get(qid, 0) + 1
                log_posts, _ = method._log_posterior(qid, mu_all[qid])
                best = max(log_posts, key=log_posts.get)
                if best == gt[qid]:
                    q_correct[qid] = q_correct.get(qid, 0) + 1
        return sum(q_correct.get(q, 0) / q_total[q] for q in q_total) / len(q_total)

    chosen_acc = _macro_acc(chosen)
    # The grid is exhaustive over T_10 ∈ [1, 30], so chosen must be ≥ every
    # other grid point's accuracy.
    for T_10 in range(1, 31):
        assert chosen_acc + 1e-12 >= _macro_acc(T_10 * 0.1)


@pytest.mark.needs_data
def test_nbfselective_theta_in_unit_interval(
    small_records: list[tuple[ExtractedAtom, dict[str, str]]],
) -> None:
    train, cal = small_records[:15], small_records[15:]
    method = NBFSelective(seed=42)
    method.fit(train)
    method.calibrate(cal)
    assert 0.0 <= method._theta_margin <= 1.0
    assert 0.1 <= method._temperature <= 3.0


@pytest.mark.needs_data
def test_nbfselective_skip_only_when_multi_source(
    small_records: list[tuple[ExtractedAtom, dict[str, str]]],
) -> None:
    """Forcing θ=1.0 must SKIP exactly the multi-source cases and answer
    every single-source case."""
    train, evalset = small_records[:20], small_records[20:]
    method = NBFSelective(seed=42)
    method.fit(train)
    method._theta_margin = 1.0  # force SKIP on every margin < 1

    saw_skip = False
    for atom, _gt in evalset:
        for qid in QUESTIONS:
            n_act = sum(
                1 for s in SOURCE_NAMES if atom.extraction[qid][s] is not None
            )
            pred = method.predict_one(atom, qid)
            if n_act >= 2:
                assert pred.answer == SKIP_SENTINEL
                assert pred.would_skip is True
                saw_skip = True
            else:
                # Single-source cases must always answer (gating rule).
                assert pred.answer in QUESTIONS[qid]["answer_space"]
                assert pred.would_skip is False
    assert saw_skip, "expected ≥1 multi-source case in eval set"


@pytest.mark.needs_data
def test_nbf_state_dict_roundtrip(
    small_records: list[tuple[ExtractedAtom, dict[str, str]]],
) -> None:
    train, cal, evalset = (
        small_records[:15], small_records[15:22], small_records[22:],
    )
    for cls in (NBF, NBFSelective):
        method = cls(seed=42)
        method.fit(train)
        method.calibrate(cal)
        state = json.loads(json.dumps(method.state_dict()))
        twin = cls(seed=42)
        twin.load_state_dict(state)
        for atom, _gt in evalset:
            for qid in QUESTIONS:
                assert (
                    method.predict_one(atom, qid).answer
                    == twin.predict_one(atom, qid).answer
                )


