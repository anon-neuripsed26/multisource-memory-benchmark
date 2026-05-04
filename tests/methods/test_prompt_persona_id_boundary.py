"""Regression tests for the method-facing persona-ID information boundary."""

from __future__ import annotations

import json
from pathlib import Path

from survey2agent._paths import PROJECT_ROOT
from survey2agent.api_clients import CompletionRequest
from survey2agent.extraction.atoms import EXPECTED_QUESTION_IDS, EXPECTED_SOURCES
from survey2agent.extraction.extractor import build_persona_extraction_request
from survey2agent.extraction.question_spec import QUESTIONS
from survey2agent.methods._llm_prompt_builders import (
    build_direct_request,
    build_few_shot_request,
    build_schema_aware_request,
    build_struct_llm_request,
)

SAMPLE_PERSONA_ID = "bench_shift_121_avery_ellis"
SAMPLE_PERSONA_DIR = (
    PROJECT_ROOT
    / "data"
    / "sample"
    / "benchmark"
    / "seeds"
    / "s20260321"
    / SAMPLE_PERSONA_ID
)
FEW_SHOT_CONFIGS = PROJECT_ROOT / "configs" / "few_shot"

FORBIDDEN_PROMPT_TOKENS = (
    SAMPLE_PERSONA_ID,
    "bench_shift_",
    "bench_stable_",
    "bench_stated_",
    "temporal_shift",
    "stated_vs_revealed",
    "Difficulty:",
)


def _prompt_text(request: CompletionRequest) -> str:
    return f"{request.system_prompt}\n{request.user_prompt}"


def _assert_no_persona_id_or_difficulty_tokens(prompt: str) -> None:
    for token in FORBIDDEN_PROMPT_TOKENS:
        assert token not in prompt


def test_direct_schema_and_extraction_prompts_do_not_expose_target_id() -> None:
    assert SAMPLE_PERSONA_DIR.is_dir(), f"missing sample persona: {SAMPLE_PERSONA_DIR}"

    requests = [
        build_direct_request(SAMPLE_PERSONA_DIR),
        build_schema_aware_request(SAMPLE_PERSONA_DIR),
        build_persona_extraction_request(SAMPLE_PERSONA_DIR),
    ]

    for request in requests:
        _assert_no_persona_id_or_difficulty_tokens(_prompt_text(request))


def test_struct_llm_prompt_uses_persona_id_only_for_file_validation(tmp_path: Path) -> None:
    grid = {
        qid: {source: QUESTIONS[qid]["answer_space"][0] for source in EXPECTED_SOURCES}
        for qid in EXPECTED_QUESTION_IDS
    }
    bundle = tmp_path / f"{SAMPLE_PERSONA_ID}.json"
    bundle.write_text(
        json.dumps({"persona": SAMPLE_PERSONA_ID, "extraction": grid}),
        encoding="utf-8",
    )

    request = build_struct_llm_request(SAMPLE_PERSONA_ID, bundle)

    _assert_no_persona_id_or_difficulty_tokens(_prompt_text(request))


def test_few_shot_prompt_redacts_exemplar_ids_and_target_id() -> None:
    assert SAMPLE_PERSONA_DIR.is_dir(), f"missing sample persona: {SAMPLE_PERSONA_DIR}"

    request = build_few_shot_request(SAMPLE_PERSONA_DIR, "A1", FEW_SHOT_CONFIGS)

    _assert_no_persona_id_or_difficulty_tokens(_prompt_text(request))
    assert "*(Example persona 1)*" in request.user_prompt
