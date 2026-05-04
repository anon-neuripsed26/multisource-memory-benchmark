"""End-to-end smoke test for `extract_atom` using a `MockSyncClient`.

This test verifies:

  1. `extract_atom` returns a validated `ExtractedAtom` (passes schema check).
  2. The mock client is invoked with the expected provider/model.
  3. Batch mode uses one persona-level call vs. single-question mode.
  4. Cache hits do not trigger live LLM calls on the second run.

The mock parses the prompt to recover which qids it must answer, then returns
canned responses (first option in each qid's `answer_space`).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import ClassVar

import pytest

from survey2agent.api_clients import (
    CacheStore,
    CompletionRequest,
    ModelConfig,
    SyncLLMClient,
)
from survey2agent.extraction import (
    EXPECTED_QUESTION_IDS,
    EXPECTED_SOURCES,
    SOURCE_QUESTION_MAP,
    ExtractedAtom,
    extract_atom,
)
from survey2agent.extraction.question_spec import QUESTIONS

# A real persona dir (read-only). Sourced from the internalized benchmark.
_PERSONA_ID = "bench_shift_121_avery_ellis"
from survey2agent._paths import persona_dir as _persona_dir  # noqa: E402
_PERSONA_DIR = _persona_dir("s20260321", _PERSONA_ID)

pytestmark = pytest.mark.needs_data


# ─── Mock client ────────────────────────────────────────────────────────────

class MockSyncClient(SyncLLMClient):
    """In-memory LLM stub.

    For persona-level extraction prompts, returns a JSON object covering every
    qid/source cell. For single-question prompts, returns a
    `REASONING: ...\\nANSWER: <label>` block. The chosen answer is always the
    first option in `QUESTIONS[qid]["answer_space"]`.
    """

    provider: ClassVar[str] = "mock"
    allowed_request_fields: ClassVar[frozenset[str]] = frozenset(
        {"user_prompt", "system_prompt", "temperature", "top_p", "seed"}
    )

    def __init__(self, model_config: ModelConfig, cache_store: CacheStore) -> None:
        super().__init__(model_config, cache_store)
        self.live_call_count = 0

    def _raw_complete(self, request: CompletionRequest) -> str:
        self.live_call_count += 1
        prompt = request.user_prompt
        if "## Extraction Spec (" in prompt:
            headers = re.findall(r"^### ([A-Za-z0-9]+)$", prompt, re.MULTILINE)
            qids_in_prompt = [h for h in headers if h in QUESTIONS]
            payload = {}
            for qid in qids_in_prompt:
                informative = set(SOURCE_QUESTION_MAP.get(qid, []))
                payload[qid] = {
                    source: (
                        QUESTIONS[qid]["answer_space"][0]
                        if source in informative else None
                    )
                    for source in EXPECTED_SOURCES
                }
            return json.dumps(payload)
        m = re.search(r"## Question \(([A-Za-z0-9]+)\)", prompt)
        if not m:
            return "ANSWER: null"
        qid = m.group(1)
        first = QUESTIONS[qid]["answer_space"][0]
        return f"REASONING: mock\nANSWER: {first}"


def _make_client(tmp_path: Path) -> MockSyncClient:
    cfg = ModelConfig(
        paper_alias="mock-1",
        provider="mock",
        api_model_id="mock-model",
        api_endpoint=None,
        default_params={},
    )
    cache = CacheStore(root=tmp_path / "cache")
    return MockSyncClient(cfg, cache)


# ─── Fixtures ───────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def persona_dir() -> Path:
    if not _PERSONA_DIR.is_dir():
        pytest.skip(f"persona dir not found: {_PERSONA_DIR}")
    return _PERSONA_DIR


# ─── Tests ──────────────────────────────────────────────────────────────────

def test_extract_atom_returns_validated_extracted_atom(
    persona_dir: Path, tmp_path: Path
) -> None:
    client = _make_client(tmp_path)
    atom = extract_atom(_PERSONA_ID, persona_dir, client, batch_mode=True)

    assert isinstance(atom, ExtractedAtom)
    assert atom.persona == _PERSONA_ID
    assert set(atom.extraction.keys()) == set(EXPECTED_QUESTION_IDS)
    for qid in EXPECTED_QUESTION_IDS:
        assert set(atom.extraction[qid].keys()) == set(EXPECTED_SOURCES)
        for src, val in atom.extraction[qid].items():
            assert val is None or isinstance(val, str)
            if val is not None:
                assert val in QUESTIONS[qid]["answer_space"]


def test_extract_atom_uses_provided_client_only(
    persona_dir: Path, tmp_path: Path
) -> None:
    client = _make_client(tmp_path)
    extract_atom(_PERSONA_ID, persona_dir, client, batch_mode=True)
    assert client.live_call_count > 0
    assert client.provider == "mock"
    assert client.model_config.api_model_id == "mock-model"


def test_extract_atom_batch_mode_reduces_calls(
    persona_dir: Path, tmp_path: Path
) -> None:
    batch_client = _make_client(tmp_path / "batch")
    extract_atom(_PERSONA_ID, persona_dir, batch_client, batch_mode=True)
    assert batch_client.live_call_count == 1

    single_client = _make_client(tmp_path / "single")
    extract_atom(_PERSONA_ID, persona_dir, single_client, batch_mode=False)
    expected_single_calls = sum(len(srcs) for srcs in SOURCE_QUESTION_MAP.values())
    assert single_client.live_call_count == expected_single_calls
    assert single_client.live_call_count > batch_client.live_call_count


def test_extract_atom_cache_hit_no_extra_call(
    persona_dir: Path, tmp_path: Path
) -> None:
    client = _make_client(tmp_path)
    atom1 = extract_atom(_PERSONA_ID, persona_dir, client, batch_mode=True)
    first_call_count = client.live_call_count
    assert first_call_count > 0

    atom2 = extract_atom(_PERSONA_ID, persona_dir, client, batch_mode=True)
    assert client.live_call_count == first_call_count
    assert atom1.persona == atom2.persona
    assert dict(atom1.extraction) == dict(atom2.extraction)
