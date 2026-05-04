"""Producer: drives LLM-Direct / Schema-Aware / Struct-LLM / Few-Shot answer runs.

For each persona this producer issues one or more calls that ask the LLM to answer
survey questions, then writes frozen-artifact JSON bundles.

The variants differ in prompt and output layout:

* ``"direct"`` / ``"schema_aware"`` / ``"struct_llm"`` — one call per persona
  answering all 18 questions. Output: ``{output_dir}/{persona_id}.json``.
* ``"few-shot"`` — one call per (persona, question). Output:
  ``{output_dir}/{persona_id}__{qid}.json``.

``BatchLLMClient`` and ``SyncLLMClient`` are both accepted.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Literal

from survey2agent.api_clients import (
    BatchLLMClient,
    BatchResultItem,
    CompletionRequest,
    SyncLLMClient,
)

from ._llm_prompt_builders import (
    FEW_SHOT_QIDS,
    Variant,
    build_direct_request,
    build_schema_aware_request,
    build_struct_llm_request,
    build_few_shot_request,
    parse_answers_response,
)
from ..extraction.atoms import EXPECTED_QUESTION_IDS
from ..extraction.batch_pipeline import MAX_REQUESTS_PER_BATCH
from ..extraction.question_spec import QUESTIONS


@dataclass
class LLMAnswersBatchReport:
    """Summary of a single ``run_llm_answers_batch`` invocation."""

    variant: Variant
    n_personas: int
    n_success: int
    n_failed: int
    failed_personas: list[str] = field(default_factory=list)
    output_dir: Path | None = None
    failure_log_path: Path | None = None


@dataclass
class LLMFewShotBatchReport:
    """Summary of a single ``run_few_shot_batch`` invocation."""

    n_personas: int
    n_questions: int
    n_total: int
    n_success: int
    n_failed: int
    output_dir: Path | None = None
    failure_log_path: Path | None = None


def run_llm_answers_batch(
    *,
    persona_ids: list[str],
    persona_dirs: dict[str, Path],
    variant: Literal["direct", "schema_aware", "struct_llm"],
    client: BatchLLMClient | SyncLLMClient,
    output_dir: Path,
    extraction_bundle_dir: Path | None = None,
    allow_api_call: bool = False,
    failure_log_path: Path | None = None,
) -> LLMAnswersBatchReport:
    """Run an LLM answer producer for every persona and write frozen bundles."""
    if variant not in ("direct", "schema_aware", "struct_llm"):
        raise ValueError(f"unknown variant: {variant!r}")
    if variant == "struct_llm" and extraction_bundle_dir is None:
        raise ValueError("variant='struct_llm' requires extraction_bundle_dir")
    if extraction_bundle_dir is not None:
        extraction_bundle_dir = Path(extraction_bundle_dir)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if failure_log_path is None:
        failure_log_path = output_dir / "_failures.json"
    failure_log_path = Path(failure_log_path)

    # Build requests.
    requests: list[CompletionRequest] = []
    for persona_id in persona_ids:
        persona_dir = Path(persona_dirs[persona_id])
        if variant == "direct":
            req = build_direct_request(persona_dir)
        elif variant == "schema_aware":
            req = build_schema_aware_request(persona_dir)
        else:
            assert extraction_bundle_dir is not None
            bundle_path = extraction_bundle_dir / f"{persona_id}.json"
            req = build_struct_llm_request(persona_id, bundle_path)
        requests.append(replace(req, custom_id=persona_id))

    # Dispatch.
    if isinstance(client, BatchLLMClient):
        results = _run_via_batch(client, requests, allow_api_call)
    else:
        results = _run_via_sync(client, requests, allow_api_call)

    # Write bundles.
    failures: dict[str, str] = {}
    for persona_id in persona_ids:
        outcome = results.get(persona_id)
        if outcome is None:
            failures[persona_id] = "no result returned"
            answers = _empty_answers()
        elif isinstance(outcome, Exception):
            failures[persona_id] = f"{type(outcome).__name__}: {outcome}"
            answers = _empty_answers()
        elif outcome.error_message is not None or outcome.text is None:
            failures[persona_id] = outcome.error_message or "empty text"
            answers = _empty_answers()
        else:
            answers = parse_answers_response(outcome.text)

        bundle = {"persona": persona_id, "answers": answers}
        (output_dir / f"{persona_id}.json").write_text(
            json.dumps(bundle, ensure_ascii=False, indent=2, sort_keys=False) + "\n",
            encoding="utf-8",
        )

    # Write failure log.
    failure_log_path.parent.mkdir(parents=True, exist_ok=True)
    failure_log_path.write_text(
        json.dumps(failures, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    return LLMAnswersBatchReport(
        variant=variant,
        n_personas=len(persona_ids),
        n_success=len(persona_ids) - len(failures),
        n_failed=len(failures),
        failed_personas=sorted(failures.keys()),
        output_dir=output_dir,
        failure_log_path=failure_log_path,
    )


def run_few_shot_batch(
    *,
    persona_ids: list[str],
    persona_dirs: dict[str, Path],
    qids: list[str] | None,
    client: BatchLLMClient | SyncLLMClient,
    output_dir: Path,
    configs_root: Path,
    allow_api_call: bool = False,
    failure_log_path: Path | None = None,
) -> LLMFewShotBatchReport:
    """Run the Few-Shot producer for (persona, qid) pairs."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if failure_log_path is None:
        failure_log_path = output_dir / "_failures.json"
    failure_log_path = Path(failure_log_path)
    
    selected_qids = qids if qids is not None else FEW_SHOT_QIDS
    
    requests: list[CompletionRequest] = []
    for pid in persona_ids:
        for qid in selected_qids:
            req = build_few_shot_request(persona_dirs[pid], qid, configs_root)
            requests.append(replace(req, custom_id=f"{pid}__{qid}"))
            
    if isinstance(client, BatchLLMClient):
        results = _run_via_batch(client, requests, allow_api_call)
    else:
        results = _run_via_sync(client, requests, allow_api_call)
        
    failures: dict[str, str] = {}
    n_success = 0
    
    for req in requests:
        cid = req.custom_id
        assert cid is not None
        pid, qid = cid.split("__")
        
        outcome = results.get(cid)
        data = None
        if outcome and not isinstance(outcome, Exception) and outcome.text:
            try:
                text = outcome.text.strip()
                fenced = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
                if fenced:
                    text = fenced.group(1).strip()
                data = json.loads(text)
            except (json.JSONDecodeError, ValueError):
                # Fallback: try to find any { ... } block
                m = re.search(r"\{.*\}", text, re.DOTALL)
                if m:
                    try:
                        data = json.loads(m.group())
                    except:
                        pass
        
        if data and isinstance(data, dict) and "answer" in data:
            n_success += 1
            final_json = {
                "persona": pid,
                "question": qid,
                "answer": data["answer"],
                "would_skip": data.get("would_skip", False)
            }
        else:
            err = "no valid response"
            if isinstance(outcome, Exception):
                err = str(outcome)
            elif outcome and outcome.error_message:
                err = outcome.error_message
            failures[cid] = err
            
            # Write fallback
            q_info = QUESTIONS[qid]
            final_json = {
                "persona": pid,
                "question": qid,
                "answer": q_info["answer_space"][0],
                "would_skip": True
            }
            
        (output_dir / f"{cid}.json").write_text(
            json.dumps(final_json, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8"
        )
        
    failure_log_path.write_text(
        json.dumps(failures, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8"
    )
    
    return LLMFewShotBatchReport(
        n_personas=len(persona_ids),
        n_questions=len(selected_qids),
        n_total=len(requests),
        n_success=n_success,
        n_failed=len(failures),
        output_dir=output_dir,
        failure_log_path=failure_log_path
    )


# ---------------------------------------------------------------------------
# Dispatch helpers (mirror batch_pipeline)
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
        err = RuntimeError(
            "batch client requires allow_api_call=True to dispatch; "
            "no cache-only path exists for batch APIs"
        )
        for req in requests:
            assert req.custom_id is not None
            out[req.custom_id] = err
        return out
    for i in range(0, len(requests), MAX_REQUESTS_PER_BATCH):
        chunk = requests[i : i + MAX_REQUESTS_PER_BATCH]
        for item in client.run_batch_blocking(chunk):
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
        except Exception as exc:
            out[req.custom_id] = exc
            continue
        out[req.custom_id] = BatchResultItem(
            custom_id=req.custom_id,
            text=result.text,
            finish_reason=result.finish_reason,
            error_message=None,
        )
    return out


def _empty_answers() -> dict[str, dict[str, str | bool]]:
    return {qid: {"answer": "", "would_skip": True} for qid in EXPECTED_QUESTION_IDS}


__all__ = [
    "LLMAnswersBatchReport",
    "LLMFewShotBatchReport",
    "run_llm_answers_batch",
    "run_few_shot_batch",
]
