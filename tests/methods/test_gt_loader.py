"""Tests for methods._gt_loader."""

from __future__ import annotations

from pathlib import Path

import pytest

from survey2agent._paths import persona_dir
from survey2agent.methods import load_persona_gt

pytestmark = pytest.mark.needs_data

_PERSONA_DIR: Path = persona_dir("s20260321", "bench_stated_160_kai_garcia")


def test_load_persona_gt_returns_18_qids() -> None:
    gt = load_persona_gt(_PERSONA_DIR)
    assert len(gt) == 18
    expected = {
        "A1", "A2", "A3",
        "B2", "B3",
        "C2", "C3",
        "D1", "D2",
        "E1", "E2",
        "F1", "F2", "F3",
        "G1", "G2",
        "Ctrl1", "Ctrl2",
    }
    assert set(gt.keys()) == expected
    for qid, label in gt.items():
        assert isinstance(label, str), f"{qid} -> {label!r} is not str"
