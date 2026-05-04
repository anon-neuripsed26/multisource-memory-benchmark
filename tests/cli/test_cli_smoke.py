"""Smoke tests for the ``survey2agent`` CLI.

Uses ``monkeypatch`` to swap the client factory for a mock so no network
call is made. The four producer subcommands are invoked through
``cli.main(argv=...)``; ``fetch-batch`` is exercised via a mock batch
client and a stub ``BatchHandle.to_json()`` payload.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import ClassVar

import pytest

from survey2agent import cli
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
from survey2agent.extraction.extractor import SOURCE_QUESTION_MAP
from survey2agent.extraction.question_spec import QUESTIONS


_PERSONA_A = "bench_shift_121_avery_ellis"
_PERSONA_B = "bench_stable_110_clara_bennett"
from survey2agent._paths import seed_dir as _seed_dir  # noqa: E402
_SEED_DIR = _seed_dir("s20260321")


def _mc() -> ModelConfig:
    return ModelConfig(
        paper_alias="Mock-1", provider="mock", api_model_id="mock-1",
        api_endpoint=None, default_params={},
    )


def _canned_answers() -> str:
    return json.dumps({
        "answers": {
            qid: {"answer": QUESTIONS[qid]["answer_space"][0], "would_skip": False}
            for qid in EXPECTED_QUESTION_IDS
        }
    })


def _canned_extraction_grid(qids: list[str]) -> str:
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


class _MockSync(SyncLLMClient):
    provider: ClassVar[str] = "mock"
    allowed_request_fields: ClassVar[frozenset[str]] = frozenset(
        {"user_prompt", "system_prompt", "temperature"}
    )

    def _raw_complete(self, request: CompletionRequest) -> str:
        import re
        assert request.custom_id is not None
        # Extraction prompt: persona-level "## Extraction Spec (N questions)".
        if "## Extraction Spec (" in request.user_prompt:
            qids = [q for q in re.findall(r"^### ([A-Za-z0-9]+)$", request.user_prompt, re.MULTILINE) if q in QUESTIONS]
            return _canned_extraction_grid(qids)
        return _canned_answers()


class _MockBatch(BatchLLMClient):
    provider: ClassVar[str] = "mock"
    allowed_request_fields: ClassVar[frozenset[str]] = frozenset(
        {"user_prompt", "system_prompt", "temperature", "response_schema", "custom_id"}
    )

    def __init__(self, model_config, cache_store):
        super().__init__(model_config, cache_store)
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
        import re
        out = []
        for req in self._submitted:
            assert req.custom_id
            if "## Extraction Spec (" in req.user_prompt:
                qids = [q for q in re.findall(r"^### ([A-Za-z0-9]+)$", req.user_prompt, re.MULTILINE) if q in QUESTIONS]
                text = _canned_extraction_grid(qids)
            else:
                text = _canned_answers()
            out.append(BatchResultItem(
                custom_id=req.custom_id, text=text, finish_reason="stop",
                error_message=None,
            ))
        return out


@pytest.fixture
def patched_factory(monkeypatch, tmp_path):
    """Replace the CLI's client factory with a mock-returning one."""
    def _factory(provider: str, model_key: str, cache_dir: Path | None):
        cache = CacheStore(root=cache_dir or tmp_path / "cache")
        if provider in ("openai", "google"):
            return _MockBatch(_mc(), cache)
        return _MockSync(_mc(), cache)
    monkeypatch.setattr(cli, "_make_client", _factory)
    return _factory


@pytest.fixture(scope="module")
def seed_dir() -> Path:
    if not (_SEED_DIR / _PERSONA_A / "structural_sources").is_dir():
        pytest.skip(f"dataset missing: {_SEED_DIR}")
    return _SEED_DIR


@pytest.fixture
def personas_file(tmp_path, seed_dir) -> Path:
    p = tmp_path / "personas.txt"
    p.write_text(f"{_PERSONA_A}\n{_PERSONA_B}\n", encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# Smoke: parser + help
# ---------------------------------------------------------------------------


def test_cli_help_exits_zero():
    rc = cli.main([])
    assert rc == 0


def test_cli_unknown_subcommand_exits_nonzero():
    with pytest.raises(SystemExit):
        cli.main(["not-a-real-subcommand"])


# ---------------------------------------------------------------------------
# Smoke: each producer subcommand
# ---------------------------------------------------------------------------


def _run_producer(cmd: str, *, seed_dir, personas_file, tmp_path, extra=()) -> Path:
    out = tmp_path / f"{cmd}-out"
    rc = cli.main([
        cmd,
        "--provider", "openrouter",
        "--model", "qwen3-235b",
        "--seed", str(seed_dir),
        "--personas", str(personas_file),
        "--output-dir", str(out),
        "--cache-dir", str(tmp_path / "cache"),
        "--allow-api-call",
        *extra,
    ])
    assert rc == 0
    return out


def _assert_answers_bundle(path: Path) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    assert set(data.keys()) == {"persona", "answers"}
    assert set(data["answers"].keys()) == set(EXPECTED_QUESTION_IDS)
    for qid in EXPECTED_QUESTION_IDS:
        assert set(data["answers"][qid].keys()) == {"answer", "would_skip"}


def _assert_extraction_bundle(path: Path) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    assert set(data.keys()) == {"persona", "extraction"}
    assert set(data["extraction"].keys()) == set(EXPECTED_QUESTION_IDS)
    for qid in EXPECTED_QUESTION_IDS:
        assert set(data["extraction"][qid].keys()) == set(EXPECTED_SOURCES)


@pytest.mark.needs_data
def test_cli_run_extraction(patched_factory, seed_dir, personas_file, tmp_path):
    out = _run_producer("run-extraction",
                        seed_dir=seed_dir, personas_file=personas_file, tmp_path=tmp_path)
    for pid in (_PERSONA_A, _PERSONA_B):
        _assert_extraction_bundle(out / f"{pid}.json")


@pytest.mark.needs_data
def test_cli_run_llm_direct(patched_factory, seed_dir, personas_file, tmp_path):
    out = _run_producer("run-llm-direct",
                        seed_dir=seed_dir, personas_file=personas_file, tmp_path=tmp_path)
    for pid in (_PERSONA_A, _PERSONA_B):
        _assert_answers_bundle(out / f"{pid}.json")


@pytest.mark.needs_data
def test_cli_run_schema_aware(patched_factory, seed_dir, personas_file, tmp_path):
    out = _run_producer("run-schema-aware",
                        seed_dir=seed_dir, personas_file=personas_file, tmp_path=tmp_path)
    for pid in (_PERSONA_A, _PERSONA_B):
        _assert_answers_bundle(out / f"{pid}.json")


@pytest.mark.needs_data
def test_cli_run_struct_llm(patched_factory, seed_dir, personas_file, tmp_path):
    # First populate an extraction bundle dir.
    extr_dir = tmp_path / "extr"
    _run_producer(
        "run-extraction",
        seed_dir=seed_dir, personas_file=personas_file, tmp_path=tmp_path,
    )
    # Re-use the run-extraction output as the struct-llm input dir.
    extr_dir = tmp_path / "run-extraction-out"
    out = _run_producer(
        "run-struct-llm",
        seed_dir=seed_dir, personas_file=personas_file, tmp_path=tmp_path,
        extra=("--extraction-bundle-dir", str(extr_dir)),
    )
    for pid in (_PERSONA_A, _PERSONA_B):
        _assert_answers_bundle(out / f"{pid}.json")


# ---------------------------------------------------------------------------
# Smoke: cache-only mode notice + fetch-batch
# ---------------------------------------------------------------------------


@pytest.mark.needs_data
def test_cli_cache_only_prints_notice(patched_factory, seed_dir, personas_file, tmp_path, capsys):
    cli.main([
        "run-llm-direct",
        "--provider", "openrouter",
        "--model", "qwen3-235b",
        "--seed", str(seed_dir),
        "--personas", str(personas_file),
        "--output-dir", str(tmp_path / "out"),
        "--cache-dir", str(tmp_path / "cache"),
    ])
    captured = capsys.readouterr()
    assert "cache-only mode" in captured.err.lower()


def test_cli_fetch_batch(patched_factory, tmp_path):
    handle = BatchHandle(provider="openai", batch_id="mock-batch", model_id="mock-1")
    handle_path = tmp_path / "handle.json"
    handle_path.write_text(handle.to_json(), encoding="utf-8")
    # The mock batch client has no _submitted content, so fetch_results
    # returns []. The CLI must still succeed and write an empty JSONL.
    rc = cli.main([
        "fetch-batch",
        "--provider", "openai",
        "--model", "gpt-5.4",
        "--handle", str(handle_path),
        "--output-dir", str(tmp_path / "fetched"),
        "--cache-dir", str(tmp_path / "cache"),
    ])
    assert rc == 0
    out = tmp_path / "fetched" / "batch_results.jsonl"
    assert out.exists()


# ---------------------------------------------------------------------------
# Smoke: module-as-script invocation (subprocess)
# ---------------------------------------------------------------------------


def test_cli_module_help_subprocess():
    result = subprocess.run(
        [sys.executable, "-m", "survey2agent.cli"],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0
    assert "survey2agent" in (result.stdout + result.stderr)
