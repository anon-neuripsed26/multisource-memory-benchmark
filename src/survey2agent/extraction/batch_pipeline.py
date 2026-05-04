"""Producer: drives persona-level extraction calls.

Given a set of personas, a bound LLM client, and an output directory, this
module issues one call per persona, parses the full 18-question x 5-source
response with :func:`extractor.parse_persona_extraction_response`, and writes
one frozen-artifact JSON bundle per persona::

    {output_dir}/{persona_id}.json
    {"persona": "...", "extraction": {qid: {source: label_or_null}}}

Two client kinds are supported:

* ``BatchLLMClient`` — all requests are collected and issued through a
  single ``run_batch_blocking`` call (chunked at
  :data:`MAX_REQUESTS_PER_BATCH` if needed).
* ``SyncLLMClient`` — requests are issued sequentially, allowing the cache
  layer to short-circuit repeats.

Failures (provider error or parse error) are recorded in a per-persona failure
log and do not abort the run; the persona's grid is left as ``null``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from survey2agent.api_clients import (
    BatchLLMClient,
    BatchResultItem,
    CompletionRequest,
    SyncLLMClient,
)

from .atoms import EXPECTED_QUESTION_IDS, EXPECTED_SOURCES
from .extractor import (
    build_persona_extraction_request,
    parse_persona_extraction_response,
)


#: Upper bound on requests per provider batch. OpenAI caps at 50k items per
#: batch; other providers are comparable. Oversize runs are auto-chunked.
MAX_REQUESTS_PER_BATCH = 50_000


@dataclass
class ExtractionBatchReport:
    """Summary of a single ``run_extraction_batch`` invocation."""

    n_personas: int
    n_success: int
    n_failed: int
    failed_personas: list[str] = field(default_factory=list)
    output_dir: Path | None = None
    failure_log_path: Path | None = None


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_extraction_batch(
    *,
    persona_ids: list[str],
    persona_dirs: dict[str, Path],
    client: BatchLLMClient | SyncLLMClient,
    output_dir: Path,
    allow_api_call: bool = False,
    failure_log_path: Path | None = None,
) -> ExtractionBatchReport:
    """Run extraction for every persona and write frozen bundles.

    Args:
        persona_ids: Personas to process, in the order the producer should
            visit them.
        persona_dirs: Mapping from persona id to the persona's on-disk
            directory (which must contain ``structural_sources/``).
        client: A bound ``BatchLLMClient`` or ``SyncLLMClient``. The caller
            is responsible for wiring the correct ``ModelConfig`` /
            ``CacheStore`` to the client.
        output_dir: Directory that will hold the per-persona JSON bundles
            (created on demand).
        allow_api_call: Propagated to the sync client. For batch clients,
            cache is not consulted per-item; an empty cache requires
            ``allow_api_call=True`` to reach the provider. In cache-only
            mode the sync path raises :class:`CacheMissError` on miss, the
            batch path refuses to submit (see below).
        failure_log_path: Path for the JSON failure log. Defaults to
            ``{output_dir}/_failures.json``.

    Returns:
        An :class:`ExtractionBatchReport` summarising the run.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if failure_log_path is None:
        failure_log_path = output_dir / "_failures.json"
    failure_log_path = Path(failure_log_path)

    include_schema = isinstance(client, BatchLLMClient)

    # Build the full request list.
    requests: list[CompletionRequest] = []
    origins: list[str] = []  # parallel to requests
    for persona_id in persona_ids:
        persona_dir = Path(persona_dirs[persona_id])
        req = build_persona_extraction_request(
            persona_dir,
            include_response_schema=include_schema,
        )
        req = _with_custom_id(req, persona_id)
        requests.append(req)
        origins.append(persona_id)

    # Dispatch to the client.
    results_by_custom_id: dict[str, BatchResultItem | Exception]
    if isinstance(client, BatchLLMClient):
        results_by_custom_id = _run_via_batch(client, requests, allow_api_call)
    else:
        results_by_custom_id = _run_via_sync(client, requests, allow_api_call)

    # Assemble per-persona bundles.
    bundles: dict[str, dict[str, dict[str, str | None]]] = {
        pid: _empty_grid() for pid in persona_ids
    }
    failures: dict[str, dict[str, str]] = {}
    for req, persona_id in zip(requests, origins):
        key = persona_id
        outcome = results_by_custom_id.get(key)
        if outcome is None:
            failures.setdefault(persona_id, {})["persona"] = "no result returned"
            continue
        if isinstance(outcome, Exception):
            failures.setdefault(persona_id, {})["persona"] = (
                f"{type(outcome).__name__}: {outcome}"
            )
            continue
        if outcome.error_message is not None or outcome.text is None:
            failures.setdefault(persona_id, {})["persona"] = (
                outcome.error_message or "empty text"
            )
            continue
        bundles[persona_id] = parse_persona_extraction_response(outcome.text)

    # Write bundles.
    n_failed_personas = 0
    failed_personas: list[str] = []
    for persona_id in persona_ids:
        out_path = output_dir / f"{persona_id}.json"
        bundle = {"persona": persona_id, "extraction": bundles[persona_id]}
        out_path.write_text(
            json.dumps(bundle, ensure_ascii=False, indent=2, sort_keys=False) + "\n",
            encoding="utf-8",
        )
        if persona_id in failures:
            n_failed_personas += 1
            failed_personas.append(persona_id)

    # Write failure log (always, even if empty → signals a clean run).
    failure_log_path.parent.mkdir(parents=True, exist_ok=True)
    failure_log_path.write_text(
        json.dumps(failures, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    return ExtractionBatchReport(
        n_personas=len(persona_ids),
        n_success=len(persona_ids) - n_failed_personas,
        n_failed=n_failed_personas,
        failed_personas=failed_personas,
        output_dir=output_dir,
        failure_log_path=failure_log_path,
    )


# ---------------------------------------------------------------------------
# Dispatch helpers
# ---------------------------------------------------------------------------


def _run_via_batch(
    client: BatchLLMClient,
    requests: list[CompletionRequest],
    allow_api_call: bool,
) -> dict[str, BatchResultItem | Exception]:
    out: dict[str, BatchResultItem | Exception] = {}
    if not requests:
        return out
    if not allow_api_call:
        # Batch clients have no cache short-circuit; refuse to submit.
        # Surface one synthetic failure per request so callers see a clean
        # "cache-only produced no output" state rather than an exception.
        err = RuntimeError(
            "batch client requires allow_api_call=True to dispatch; "
            "no cache-only path exists for batch APIs"
        )
        for req in requests:
            assert req.custom_id is not None
            out[req.custom_id] = err
        return out
    for chunk in _chunked(requests, MAX_REQUESTS_PER_BATCH):
        items = client.run_batch_blocking(chunk)
        for item in items:
            out[item.custom_id] = item
    return out


def _run_via_sync(
    client: SyncLLMClient,
    requests: list[CompletionRequest],
    allow_api_call: bool,
) -> dict[str, BatchResultItem | Exception]:
    out: dict[str, BatchResultItem | Exception] = {}
    for req in requests:
        assert req.custom_id is not None
        try:
            result = client.complete(req, allow_api_call=allow_api_call)
        except Exception as exc:  # provider errors, cache misses, parse errors
            out[req.custom_id] = exc
            continue
        out[req.custom_id] = BatchResultItem(
            custom_id=req.custom_id,
            text=result.text,
            finish_reason=result.finish_reason,
            error_message=None,
        )
    return out


def _chunked(seq: list[CompletionRequest], size: int) -> Iterable[list[CompletionRequest]]:
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _with_custom_id(req: CompletionRequest, custom_id: str) -> CompletionRequest:
    from dataclasses import replace
    return replace(req, custom_id=custom_id)


def _empty_grid() -> dict[str, dict[str, str | None]]:
    return {qid: {src: None for src in EXPECTED_SOURCES} for qid in EXPECTED_QUESTION_IDS}


__all__ = [
    "ExtractionBatchReport",
    "MAX_REQUESTS_PER_BATCH",
    "run_extraction_batch",
]
