"""Tests for `LLMSchemaAware` / `LLMSchemaAwareSelective`."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from survey2agent._paths import METHOD_OUTPUTS_ROOT
from survey2agent.extraction.atoms import ExtractedAtom
from survey2agent.extraction.question_spec import QUESTIONS, SOURCE_NAMES
from survey2agent.methods import (
    FrozenBulkJSONSource,
    LLMSchemaAware,
    LLMSchemaAwareSelective,
    RawLLMOutput,
    SKIP_SENTINEL,
)
from survey2agent.methods.llm_base import LLMSource


_BULK_DIR = METHOD_OUTPUTS_ROOT / "gpt-5.4" / "s20260321" / "schema-aware"


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


def test_schema_forced_discards_skip() -> None:
    src = _StubSource(RawLLMOutput(answer="X", would_skip=True))
    m = LLMSchemaAware(source=src, model_display_name="GPT-5.4")
    pred = m.predict_one(_atom("p1"), "A1")
    assert pred.answer == "X"
    assert pred.would_skip is False
    assert pred.raw_answer == "X"
    assert m.name == "GPT-5.4 Schema"


def test_schema_selective_honors_skip() -> None:
    src = _StubSource(RawLLMOutput(answer="X", would_skip=True))
    m = LLMSchemaAwareSelective(source=src, model_display_name="GPT-5.4")
    pred = m.predict_one(_atom("p1"), "A1")
    assert pred.answer == SKIP_SENTINEL
    assert pred.would_skip is True
    assert pred.raw_answer == "X"
    assert m.name == "GPT-5.4 Schema (Selective)"


@pytest.mark.needs_data
@pytest.mark.skipif(not _BULK_DIR.exists(), reason="frozen schema-aware dir missing")
def test_schema_byte_equiv_against_frozen_json() -> None:
    src = FrozenBulkJSONSource("gpt-5.4", "s20260321", "schema-aware")
    method = LLMSchemaAware(source=src, model_display_name="GPT-5.4")
    persona_files = sorted(_BULK_DIR.glob("*.json"))[:5]
    assert len(persona_files) == 5
    for pf in persona_files:
        on_disk = json.loads(pf.read_text(encoding="utf-8"))
        persona = on_disk["persona"]
        for qid, entry in on_disk["answers"].items():
            pred = method.predict_one(_atom(persona), qid)
            assert pred.raw_answer == entry["answer"]
            assert pred.answer == entry["answer"]
            assert pred.would_skip is False
