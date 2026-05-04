"""Tests for the frozen LLM source loaders."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from survey2agent._paths import METHOD_OUTPUTS_ROOT
from survey2agent.methods import (
    FrozenBulkJSONSource,
    FrozenFewShotDirSource,
    LiveSource,
    RawLLMOutput,
)

_BULK_ROOT = METHOD_OUTPUTS_ROOT
_FEWSHOT_DIR = _BULK_ROOT / "gpt-5.4" / "s20260321" / "few-shot"

_FIXTURE_PERSONA = "bench_shift_121_avery_ellis"
_FIXTURE_QID = "A1"


# ── FrozenBulkJSONSource ───────────────────────────────────────────────────


@pytest.mark.needs_data
def test_bulk_source_get_matches_disk() -> None:
    src = FrozenBulkJSONSource("gpt-5.4", "s20260321", "direct")
    out = src.get(_FIXTURE_PERSONA, _FIXTURE_QID)
    raw = json.loads((_BULK_ROOT / "gpt-5.4" / "s20260321" / "direct" / f"{_FIXTURE_PERSONA}.json").read_text(encoding="utf-8"))
    expected = raw["answers"][_FIXTURE_QID]
    assert isinstance(out, RawLLMOutput)
    assert out.answer == expected["answer"]
    assert out.would_skip == expected["would_skip"]


def test_bulk_source_missing_persona_raises_file_not_found() -> None:
    src = FrozenBulkJSONSource("gpt-5.4", "s20260321", "direct")
    with pytest.raises(FileNotFoundError):
        src.get("persona_that_does_not_exist", "A1")


def test_bulk_source_caches_per_persona(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Build a tiny fake variant dir so we can detect re-reads.
    variant_dir = tmp_path / "fake-model" / "s_test" / "direct"
    variant_dir.mkdir(parents=True)
    persona = "p1"
    payload = {"persona": persona, "answers": {"A1": {"answer": "X", "would_skip": False}}}
    (variant_dir / f"{persona}.json").write_text(json.dumps(payload), encoding="utf-8")

    src = FrozenBulkJSONSource("fake-model", "s_test", "direct", root=tmp_path)
    src.get(persona, "A1")

    # Now make the file unreadable; cached read should still succeed.
    (variant_dir / f"{persona}.json").unlink()
    out = src.get(persona, "A1")
    assert out.answer == "X"


def test_bulk_source_explicit_root_kwarg(tmp_path: Path) -> None:
    variant_dir = tmp_path / "m" / "s" / "direct"
    variant_dir.mkdir(parents=True)
    payload = {"persona": "p1", "answers": {"A1": {"answer": "Y", "would_skip": True}}}
    (variant_dir / "p1.json").write_text(json.dumps(payload), encoding="utf-8")
    src = FrozenBulkJSONSource("m", "s", "direct", root=tmp_path)
    out = src.get("p1", "A1")
    assert out == RawLLMOutput(answer="Y", would_skip=True)


# ── FrozenFewShotDirSource ─────────────────────────────────────────────────


@pytest.mark.needs_data
@pytest.mark.skipif(not _FEWSHOT_DIR.exists(), reason="few-shot results dir missing")
def test_fewshot_source_get_matches_disk() -> None:
    src = FrozenFewShotDirSource("gpt-5.4", "s20260321")
    out = src.get(_FIXTURE_PERSONA, _FIXTURE_QID)
    raw = json.loads((_FEWSHOT_DIR / f"{_FIXTURE_PERSONA}__{_FIXTURE_QID}.json").read_text(encoding="utf-8"))
    assert out.answer == raw["answer"]
    assert out.would_skip == raw["would_skip"]


def test_fewshot_source_explicit_results_dir(tmp_path: Path) -> None:
    src = FrozenFewShotDirSource(results_dir=tmp_path)
    with pytest.raises(FileNotFoundError):
        src.get("nope", "A1")


def test_fewshot_source_requires_model_or_results_dir() -> None:
    with pytest.raises(ValueError):
        FrozenFewShotDirSource()


# ── LiveSource ─────────────────────────────────────────────────────────────


def test_live_source_raises_not_implemented() -> None:
    src = LiveSource()
    with pytest.raises(NotImplementedError):
        src.get("p1", "A1")
