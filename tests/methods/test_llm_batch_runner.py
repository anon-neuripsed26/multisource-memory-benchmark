"""Tests for ``methods.llm_batch_runner.run_llm_answers_batch``.

Covers the three variants (direct / schema_aware / struct_llm) via mock
clients. Bundles are validated against the frozen answers schema.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import ClassVar

import pytest

from survey2agent.api_clients import (
    BatchHandle,
    BatchLLMClient,
    BatchResultItem,
    BatchStatus,
    CacheStore,
    CompletionRequest,
    ModelConfig,
    SyncLLMClient,
)
from survey2agent.extraction.atoms import EXPECTED_QUESTION_IDS, EXPECTED_SOURCES
from survey2agent.extraction.question_spec import QUESTIONS
from survey2agent.methods.llm_batch_runner import (
    LLMAnswersBatchReport,
    run_llm_answers_batch,
)

_PERSONA_A = "bench_shift_121_avery_ellis"
_PERSONA_B = "bench_stable_110_clara_bennett"
from survey2agent._paths import seed_dir as _seed_dir  # noqa: E402
_DATASET_ROOT = _seed_dir("s20260321")

pytestmark = pytest.mark.needs_data


def _mc(**over) -> ModelConfig:
    base = dict(
        paper_alias="Mock-1", provider="mock", api_model_id="mock-1",
        api_endpoint=None, default_params={},
    )
    base.update(over)
    return ModelConfig(**base)


def _canned_answers_json() -> str:
    return json.dumps({
        "answers": {
            qid: {
                "answer": QUESTIONS[qid]["answer_space"][0],
                "would_skip": False,
            }
            for qid in EXPECTED_QUESTION_IDS
        }
    })


class _MockSync(SyncLLMClient):
    provider: ClassVar[str] = "mock"
    allowed_request_fields: ClassVar[frozenset[str]] = frozenset(
        {"user_prompt", "system_prompt", "temperature"}
    )

    def __init__(self, model_config, cache_store, *, fail_personas=()):
        super().__init__(model_config, cache_store)
        self._fail = set(fail_personas)
        self.calls = 0

    def _raw_complete(self, request: CompletionRequest) -> str:
        self.calls += 1
        assert request.custom_id is not None
        if request.custom_id in self._fail:
            return "not json at all"
        return _canned_answers_json()


class _MockBatch(BatchLLMClient):
    provider: ClassVar[str] = "mock"
    allowed_request_fields: ClassVar[frozenset[str]] = frozenset(
        {"user_prompt", "system_prompt", "temperature", "custom_id"}
    )

    def __init__(self, model_config, cache_store, *, fail_personas=()):
        super().__init__(model_config, cache_store)
        self._fail = set(fail_personas)
        self._submitted: list[CompletionRequest] = []

    def submit_batch(self, requests):
        self._submitted = list(requests)
        return BatchHandle(
            provider=self.provider, batch_id="mock-batch",
            model_id=self.model_config.api_model_id,
        )

    def poll_status(self, handle):
        return BatchStatus.COMPLETED

    def fetch_results(self, handle):
        out = []
        for req in self._submitted:
            assert req.custom_id
            if req.custom_id in self._fail:
                out.append(BatchResultItem(
                    custom_id=req.custom_id, text=None, finish_reason=None,
                    error_message="simulated provider error",
                ))
                continue
            out.append(BatchResultItem(
                custom_id=req.custom_id, text=_canned_answers_json(),
                finish_reason="stop", error_message=None,
            ))
        return out


@pytest.fixture(scope="module")
def persona_dirs() -> dict[str, Path]:
    if not _DATASET_ROOT.is_dir():
        pytest.skip(f"dataset dir missing: {_DATASET_ROOT}")
    dirs = {}
    for pid in (_PERSONA_A, _PERSONA_B):
        d = _DATASET_ROOT / pid
        if not (d / "structural_sources").is_dir():
            pytest.skip(f"persona dir missing: {d}")
        dirs[pid] = d
    return dirs


@pytest.fixture
def extraction_bundles(tmp_path, persona_dirs) -> Path:
    """Build a minimal extraction-bundle dir for the struct_llm variant."""
    out = tmp_path / "extraction_bundles"
    out.mkdir()
    for pid in persona_dirs:
        grid = {
            qid: {src: QUESTIONS[qid]["answer_space"][0] for src in EXPECTED_SOURCES}
            for qid in EXPECTED_QUESTION_IDS
        }
        (out / f"{pid}.json").write_text(
            json.dumps({"persona": pid, "extraction": grid}, ensure_ascii=False),
            encoding="utf-8",
        )
    return out


def _assert_answers_bundle_schema(path: Path) -> None:
    bundle = json.loads(path.read_text(encoding="utf-8"))
    assert set(bundle.keys()) == {"persona", "answers"}
    assert isinstance(bundle["persona"], str) and bundle["persona"]
    answers = bundle["answers"]
    assert set(answers.keys()) == set(EXPECTED_QUESTION_IDS)
    for qid in EXPECTED_QUESTION_IDS:
        entry = answers[qid]
        assert set(entry.keys()) == {"answer", "would_skip"}
        assert isinstance(entry["answer"], str)
        assert isinstance(entry["would_skip"], bool)


# ---------------------------------------------------------------------------
# Variant: direct
# ---------------------------------------------------------------------------


def test_direct_sync_writes_bundles(persona_dirs, tmp_path):
    client = _MockSync(_mc(), CacheStore(tmp_path / "cache"))
    report = run_llm_answers_batch(
        persona_ids=list(persona_dirs.keys()),
        persona_dirs=persona_dirs,
        variant="direct",
        client=client,
        output_dir=tmp_path / "out",
        allow_api_call=True,
    )
    assert isinstance(report, LLMAnswersBatchReport)
    assert report.n_success == 2 and report.n_failed == 0
    for pid in persona_dirs:
        _assert_answers_bundle_schema(tmp_path / "out" / f"{pid}.json")


def test_direct_batch_writes_bundles(persona_dirs, tmp_path):
    client = _MockBatch(_mc(), CacheStore(tmp_path / "cache"))
    report = run_llm_answers_batch(
        persona_ids=list(persona_dirs.keys()),
        persona_dirs=persona_dirs,
        variant="direct",
        client=client,
        output_dir=tmp_path / "out",
        allow_api_call=True,
    )
    assert report.n_success == 2
    for pid in persona_dirs:
        _assert_answers_bundle_schema(tmp_path / "out" / f"{pid}.json")


# ---------------------------------------------------------------------------
# Variant: schema_aware
# ---------------------------------------------------------------------------


def test_schema_aware_sync_writes_bundles(persona_dirs, tmp_path):
    client = _MockSync(_mc(), CacheStore(tmp_path / "cache"))
    report = run_llm_answers_batch(
        persona_ids=list(persona_dirs.keys()),
        persona_dirs=persona_dirs,
        variant="schema_aware",
        client=client,
        output_dir=tmp_path / "out",
        allow_api_call=True,
    )
    assert report.n_success == 2
    for pid in persona_dirs:
        _assert_answers_bundle_schema(tmp_path / "out" / f"{pid}.json")


# ---------------------------------------------------------------------------
# Variant: struct_llm
# ---------------------------------------------------------------------------


def test_struct_llm_requires_extraction_bundle_dir(persona_dirs, tmp_path):
    client = _MockSync(_mc(), CacheStore(tmp_path / "cache"))
    with pytest.raises(ValueError):
        run_llm_answers_batch(
            persona_ids=list(persona_dirs.keys()),
            persona_dirs=persona_dirs,
            variant="struct_llm",
            client=client,
            output_dir=tmp_path / "out",
            allow_api_call=True,
        )


def test_struct_llm_sync_writes_bundles(persona_dirs, tmp_path, extraction_bundles):
    client = _MockSync(_mc(), CacheStore(tmp_path / "cache"))
    report = run_llm_answers_batch(
        persona_ids=list(persona_dirs.keys()),
        persona_dirs=persona_dirs,
        variant="struct_llm",
        client=client,
        output_dir=tmp_path / "out",
        extraction_bundle_dir=extraction_bundles,
        allow_api_call=True,
    )
    assert report.n_success == 2
    for pid in persona_dirs:
        _assert_answers_bundle_schema(tmp_path / "out" / f"{pid}.json")


# ---------------------------------------------------------------------------
# Failure handling
# ---------------------------------------------------------------------------


def test_partial_failure_bundles_are_still_written(persona_dirs, tmp_path):
    client = _MockBatch(
        _mc(), CacheStore(tmp_path / "cache"),
        fail_personas={_PERSONA_A},
    )
    report = run_llm_answers_batch(
        persona_ids=list(persona_dirs.keys()),
        persona_dirs=persona_dirs,
        variant="direct",
        client=client,
        output_dir=tmp_path / "out",
        allow_api_call=True,
    )
    assert report.n_failed == 1
    assert _PERSONA_A in report.failed_personas
    fail_log = json.loads(report.failure_log_path.read_text(encoding="utf-8"))
    assert _PERSONA_A in fail_log
    # Both bundles must still exist and be schema-conformant.
    for pid in persona_dirs:
        _assert_answers_bundle_schema(tmp_path / "out" / f"{pid}.json")


def test_invalid_variant_rejected(persona_dirs, tmp_path):
    client = _MockSync(_mc(), CacheStore(tmp_path / "cache"))
    with pytest.raises(ValueError):
        run_llm_answers_batch(
            persona_ids=list(persona_dirs.keys()),
            persona_dirs=persona_dirs,
            variant="nonsense",  # type: ignore[arg-type]
            client=client,
            output_dir=tmp_path / "out",
            allow_api_call=True,
        )
