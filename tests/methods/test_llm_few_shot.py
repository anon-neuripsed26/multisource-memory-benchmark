"""Tests for `LLMFewShot` / `LLMFewShotSelective`."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from survey2agent._paths import METHOD_OUTPUTS_ROOT
from survey2agent.extraction.atoms import ExtractedAtom
from survey2agent.extraction.question_spec import QUESTIONS, SOURCE_NAMES
from survey2agent.methods import (
    FrozenFewShotDirSource,
    LLMFewShot,
    LLMFewShotSelective,
    RawLLMOutput,
    SKIP_SENTINEL,
)
from survey2agent.methods.llm_base import LLMSource


_FEWSHOT_DIR = METHOD_OUTPUTS_ROOT / "gpt-5.4" / "s20260321" / "few-shot"


def _atom(persona: str) -> ExtractedAtom:
    extraction: dict[str, dict[str, str | None]] = {
        qid: {src: None for src in SOURCE_NAMES} for qid in QUESTIONS
    }
    return ExtractedAtom.from_json({"persona": persona, "extraction": extraction})


class _StubSource(LLMSource):
    def __init__(self, raw: RawLLMOutput) -> None:
        self.raw = raw

    def get(self, persona_id: str, qid: str) -> RawLLMOutput:
        return self.raw


def test_few_shot_requires_fit_flag() -> None:
    m = LLMFewShot(source=_StubSource(RawLLMOutput("X", False)), model_display_name="GPT-5.4")
    assert m.requires_fit is True
    assert m.requires_calibration is False


def test_few_shot_fit_is_noop_records_train_size() -> None:
    m = LLMFewShot(source=_StubSource(RawLLMOutput("X", False)), model_display_name="GPT-5.4")
    m.fit([])  # empty record list still counts as a fit
    state = m.state_dict()
    assert state["train_size"] == 0
    assert state["fitted"] is True


def test_few_shot_fit_missing_manifest_raises(tmp_path: Path) -> None:
    bogus = tmp_path / "no_such_manifest.json"
    m = LLMFewShot(
        source=_StubSource(RawLLMOutput("X", False)),
        model_display_name="GPT-5.4",
        shot_manifest_path=bogus,
    )
    with pytest.raises(FileNotFoundError):
        m.fit([])


def test_few_shot_forced_discards_skip() -> None:
    src = _StubSource(RawLLMOutput(answer="X", would_skip=True))
    m = LLMFewShot(source=src, model_display_name="GPT-5.4")
    pred = m.predict_one(_atom("p1"), "A1")
    assert pred.answer == "X"
    assert pred.would_skip is False
    assert pred.raw_answer == "X"
    assert m.name == "GPT-5.4 Few-Shot"


def test_few_shot_selective_honors_skip() -> None:
    src = _StubSource(RawLLMOutput(answer="X", would_skip=True))
    m = LLMFewShotSelective(source=src, model_display_name="GPT-5.4")
    pred = m.predict_one(_atom("p1"), "A1")
    assert pred.answer == SKIP_SENTINEL
    assert pred.would_skip is True
    assert pred.raw_answer == "X"
    assert m.name == "GPT-5.4 Few-Shot (Selective)"


@pytest.mark.needs_data
@pytest.mark.skipif(not _FEWSHOT_DIR.exists(), reason="few-shot results dir missing")
def test_few_shot_byte_equiv_against_frozen_dir() -> None:
    src = FrozenFewShotDirSource("gpt-5.4", "s20260321")
    method = LLMFewShot(source=src, model_display_name="GPT-5.4")
    sample_files = sorted(_FEWSHOT_DIR.glob("bench_shift_121_avery_ellis__*.json"))
    assert sample_files, "expected at least one fixture file"
    for sf in sample_files[:5]:
        raw = json.loads(sf.read_text(encoding="utf-8"))
        persona = raw["persona"]
        qid = raw["question"]
        pred = method.predict_one(_atom(persona), qid)
        assert pred.raw_answer == raw["answer"]
        assert pred.answer == raw["answer"]
        assert pred.would_skip is False
