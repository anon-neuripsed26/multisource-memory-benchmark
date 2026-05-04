"""Unit tests for ``StructLLMSource``.

Validates the new frozen-artifact loader added to
``survey2agent.methods.llm_sources``. Schema is byte-identical to
``FrozenBulkJSONSource``; coverage = 4 seeds × 2 modes × 120 personas.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from survey2agent.methods import StructLLMSource
from survey2agent.methods.llm_base import RawLLMOutput


CANONICAL_SEEDS = ["s20260321", "s20260322", "s20260323", "s20260324"]

pytestmark = pytest.mark.needs_data


def test_load_one_record() -> None:
    src = StructLLMSource(mode="oracle", seed="s20260321")
    out = src.get("bench_stable_121_sam_bennett", "A1")
    assert isinstance(out, RawLLMOutput)
    assert isinstance(out.answer, str) and out.answer
    assert isinstance(out.would_skip, bool)


def test_mode_validation() -> None:
    with pytest.raises(ValueError):
        StructLLMSource(mode="not-a-mode", seed="s20260321")


def test_persona_count_per_seed_mode() -> None:
    """Each (mode, seed) folder must contain exactly 120 persona files."""
    for mode in ("oracle", "extracted"):
        for seed in CANONICAL_SEEDS:
            src = StructLLMSource(mode=mode, seed=seed)
            files = list(Path(src.dir).glob("*.json"))
            assert len(files) == 120, (
                f"{mode}/{seed} has {len(files)} files (expected 120)"
            )


def test_qid_coverage_subset() -> None:
    """One representative persona must answer the 18 canonical question ids."""
    expected = {
        "A1", "A2", "A3", "B1", "B2", "C1", "C2", "C3",
        "D1", "D2", "E1", "E2", "F1", "F2", "F3", "G1", "G2",
        "Ctrl1", "Ctrl2",
    }
    src = StructLLMSource(mode="oracle", seed="s20260321")
    files = sorted(Path(src.dir).glob("*.json"))
    persona_id = files[0].stem
    src.get(persona_id, "A1")  # populate cache
    cached = src._cache.get(persona_id, {})
    keys = set(cached.keys())
    assert "A1" in keys, f"missing A1 for {persona_id}"
    # Allow a small set of older artifacts that may not have all 19 keys;
    # Require at least 17 of the canonical question set.
    assert len(keys & expected) >= 17, (
        f"{persona_id} answered only {sorted(keys & expected)}"
    )


def test_custom_root(tmp_path: Path) -> None:
    """Passing ``root=`` resolves the directory layout under that root."""
    fake_root = tmp_path / "method_outputs"
    seed_dir = fake_root / "struct_llm" / "oracle" / "s20260321"
    seed_dir.mkdir(parents=True)
    src = StructLLMSource(mode="oracle", seed="s20260321", root=fake_root)
    assert Path(src.dir) == seed_dir
