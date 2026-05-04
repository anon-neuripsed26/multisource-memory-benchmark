"""Tests for ``extraction.batch_pipeline.run_extraction_batch``.

Uses mock clients (sync and batch) so no network call is made. Bundles are
validated against the same frozen-artifact schema assertions used by
``tests/integration/test_frozen_artifact_schema_lock.py``.
"""

from __future__ import annotations

import json
import re
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
from survey2agent.extraction.batch_pipeline import (
    ExtractionBatchReport,
    run_extraction_batch,
)
from survey2agent.extraction.extractor import SOURCE_QUESTION_MAP
from survey2agent.extraction.question_spec import QUESTIONS


_PERSONA_A = "bench_shift_121_avery_ellis"
_PERSONA_B = "bench_stable_110_clara_bennett"
from survey2agent._paths import seed_dir as _seed_dir  # noqa: E402
_DATASET_ROOT = _seed_dir("s20260321")

pytestmark = pytest.mark.needs_data


def _mc(**over) -> ModelConfig:
    base = dict(
        paper_alias="Mock-1",
        provider="mock",
        api_model_id="mock-1",
        api_endpoint=None,
        default_params={},
    )
    base.update(over)
    return ModelConfig(**base)


def _qids_from_prompt(prompt: str) -> list[str]:
    """Recover the qid list embedded in a persona-level extraction prompt."""
    headers = re.findall(r"^### ([A-Za-z0-9]+)$", prompt, re.MULTILINE)
    return [h for h in headers if h in QUESTIONS]


def _canned_grid_for(qids: list[str]) -> str:
    """Return a JSON response assigning each informative cell its first label."""
    payload = {}
    for qid in qids:
        informative = set(SOURCE_QUESTION_MAP.get(qid, []))
        payload[qid] = {
            source: (
                QUESTIONS[qid]["answer_space"][0]
                if source in informative else None
            )
            for source in EXPECTED_SOURCES
        }
    return json.dumps(payload)


# ---------------------------------------------------------------------------
# Mock clients
# ---------------------------------------------------------------------------


class _MockSync(SyncLLMClient):
    provider: ClassVar[str] = "mock"
    allowed_request_fields: ClassVar[frozenset[str]] = frozenset(
        {"user_prompt", "system_prompt", "temperature"}
    )

    def __init__(self, model_config, cache_store, *, fail_personas=()):
        super().__init__(model_config, cache_store)
        self._fail_personas = set(fail_personas)
        self.calls = 0

    def _raw_complete(self, request: CompletionRequest) -> str:
        self.calls += 1
        assert request.custom_id is not None
        # Inject a parse-failing response for configured persona ids.
        if request.custom_id in self._fail_personas:
            return "(no json here)"
        qids = _qids_from_prompt(request.user_prompt)
        return _canned_grid_for(qids)


class _MockBatch(BatchLLMClient):
    provider: ClassVar[str] = "mock"
    allowed_request_fields: ClassVar[frozenset[str]] = frozenset(
        {"user_prompt", "system_prompt", "temperature", "response_schema", "custom_id"}
    )

    def __init__(self, model_config, cache_store, *, fail_personas=()):
        super().__init__(model_config, cache_store)
        self._fail_personas = set(fail_personas)
        self._submitted: list[CompletionRequest] = []

    def submit_batch(self, requests):
        self._submitted = list(requests)
        return BatchHandle(
            provider=self.provider, batch_id="mock-batch", model_id=self.model_config.api_model_id
        )

    def poll_status(self, handle):
        return BatchStatus.COMPLETED

    def fetch_results(self, handle):
        results = []
        for req in self._submitted:
            assert req.custom_id
            if req.custom_id in self._fail_personas:
                results.append(BatchResultItem(
                    custom_id=req.custom_id, text=None, finish_reason=None,
                    error_message="mock provider error",
                ))
                continue
            qids = _qids_from_prompt(req.user_prompt)
            results.append(BatchResultItem(
                custom_id=req.custom_id, text=_canned_grid_for(qids),
                finish_reason="stop", error_message=None,
            ))
        return results


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Schema assertions (mirror test_frozen_artifact_schema_lock)
# ---------------------------------------------------------------------------


def _assert_extraction_bundle_schema(path: Path) -> None:
    bundle = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(bundle, dict)
    assert set(bundle.keys()) == {"persona", "extraction"}
    assert isinstance(bundle["persona"], str) and bundle["persona"]
    extraction = bundle["extraction"]
    assert isinstance(extraction, dict)
    assert set(extraction.keys()) == set(EXPECTED_QUESTION_IDS)
    for qid in EXPECTED_QUESTION_IDS:
        per_qid = extraction[qid]
        assert set(per_qid.keys()) == set(EXPECTED_SOURCES)
        for src, val in per_qid.items():
            assert val is None or isinstance(val, str)


# ---------------------------------------------------------------------------
# Tests — sync client
# ---------------------------------------------------------------------------


def test_sync_writes_per_persona_bundles(persona_dirs, tmp_path):
    client = _MockSync(_mc(), CacheStore(tmp_path / "cache"))
    report = run_extraction_batch(
        persona_ids=list(persona_dirs.keys()),
        persona_dirs=persona_dirs,
        client=client,
        output_dir=tmp_path / "out",
        allow_api_call=True,
    )
    assert isinstance(report, ExtractionBatchReport)
    assert report.n_personas == 2
    assert report.n_success == 2
    assert report.n_failed == 0
    assert client.calls == 2
    for pid in persona_dirs:
        _assert_extraction_bundle_schema(tmp_path / "out" / f"{pid}.json")


def test_sync_cache_only_raises_per_persona(persona_dirs, tmp_path):
    # Cache empty + allow_api_call=False → each sync request surfaces as
    # a cache miss; bundles are still written with all-null cells, and
    # failures are logged per persona.
    client = _MockSync(_mc(), CacheStore(tmp_path / "cache"))
    report = run_extraction_batch(
        persona_ids=list(persona_dirs.keys()),
        persona_dirs=persona_dirs,
        client=client,
        output_dir=tmp_path / "out",
        allow_api_call=False,
    )
    assert report.n_failed == 2
    assert client.calls == 0
    fail_log = json.loads(report.failure_log_path.read_text(encoding="utf-8"))
    for pid in persona_dirs:
        assert pid in fail_log
        assert set(fail_log[pid]) == {"persona"}
    # Bundles written, all cells null.
    for pid in persona_dirs:
        bundle = json.loads((tmp_path / "out" / f"{pid}.json").read_text(encoding="utf-8"))
        for qid in EXPECTED_QUESTION_IDS:
            for src in EXPECTED_SOURCES:
                assert bundle["extraction"][qid][src] is None


def test_sync_partial_failure_is_logged(persona_dirs, tmp_path):
    client = _MockSync(
        _mc(), CacheStore(tmp_path / "cache"),
        fail_personas={_PERSONA_A},
    )
    report = run_extraction_batch(
        persona_ids=list(persona_dirs.keys()),
        persona_dirs=persona_dirs,
        client=client,
        output_dir=tmp_path / "out",
        allow_api_call=True,
    )
    assert report.n_failed == 0
    # Parse-failing JSON leaves a valid all-null grid rather than aborting.
    for pid in persona_dirs:
        _assert_extraction_bundle_schema(tmp_path / "out" / f"{pid}.json")


# ---------------------------------------------------------------------------
# Tests — batch client
# ---------------------------------------------------------------------------


def test_batch_writes_per_persona_bundles(persona_dirs, tmp_path):
    client = _MockBatch(_mc(), CacheStore(tmp_path / "cache"))
    report = run_extraction_batch(
        persona_ids=list(persona_dirs.keys()),
        persona_dirs=persona_dirs,
        client=client,
        output_dir=tmp_path / "out",
        allow_api_call=True,
    )
    assert report.n_success == 2
    assert report.n_failed == 0
    assert len(client._submitted) == 2
    for pid in persona_dirs:
        _assert_extraction_bundle_schema(tmp_path / "out" / f"{pid}.json")


def test_batch_provider_error_recorded_as_failure(persona_dirs, tmp_path):
    client = _MockBatch(
        _mc(), CacheStore(tmp_path / "cache"),
        fail_personas={_PERSONA_A, _PERSONA_B},
    )
    report = run_extraction_batch(
        persona_ids=list(persona_dirs.keys()),
        persona_dirs=persona_dirs,
        client=client,
        output_dir=tmp_path / "out",
        allow_api_call=True,
    )
    assert report.n_failed == 2
    fail_log = json.loads(report.failure_log_path.read_text(encoding="utf-8"))
    for pid in persona_dirs:
        assert set(fail_log[pid]) == {"persona"}
    # Bundles remain schema-conformant; failing personas hold null labels.
    for pid in persona_dirs:
        _assert_extraction_bundle_schema(tmp_path / "out" / f"{pid}.json")
        bundle = json.loads((tmp_path / "out" / f"{pid}.json").read_text(encoding="utf-8"))
        for qid in EXPECTED_QUESTION_IDS:
            for source in EXPECTED_SOURCES:
                assert bundle["extraction"][qid][source] is None


def test_batch_cache_only_mode_refuses(persona_dirs, tmp_path):
    client = _MockBatch(_mc(), CacheStore(tmp_path / "cache"))
    report = run_extraction_batch(
        persona_ids=list(persona_dirs.keys()),
        persona_dirs=persona_dirs,
        client=client,
        output_dir=tmp_path / "out",
        allow_api_call=False,
    )
    assert report.n_failed == 2
    # Client never submitted.
    assert client._submitted == []
