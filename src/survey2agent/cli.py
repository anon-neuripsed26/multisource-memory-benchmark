"""``survey2agent`` CLI entry point.

Subcommands:

* ``run-extraction`` — for each persona, extract the full 18-question x
  5-source atom grid and write ``{persona}.json`` bundles matching the
  frozen extraction schema.
* ``run-llm-direct`` / ``run-schema-aware`` / ``run-struct-llm`` — answer
  all 18 survey questions for each persona in one LLM call and write
  ``{persona}.json`` bundles matching the frozen answers schema.
* ``run-few-shot`` — answer selected questions with the k-shot prompt
  template and write ``{persona}__{qid}.json`` bundles.
* ``fetch-batch`` — resume a batch previously submitted by any of the
  producers, given a saved ``BatchHandle.to_json()`` payload.

Every subcommand defaults to cache-only mode. Pass ``--allow-api-call``
to permit live provider dispatch.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

from survey2agent.api_clients import (
    BatchHandle,
    BatchLLMClient,
    CacheStore,
    SyncLLMClient,
    load_model_config,
)


_CONFIGS_MODELS_YAML = (
    Path(__file__).resolve().parents[2] / "configs" / "models.yaml"
)
_DEFAULT_CACHE_ROOT = (
    Path(__file__).resolve().parents[2]
    / "results"
    / "released"
    / "cached_api_outputs"
)
_DEFAULT_FEW_SHOT_CONFIGS_ROOT = (
    Path(__file__).resolve().parents[2] / "configs" / "few_shot"
)


# ---------------------------------------------------------------------------
# Arg parsing
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="survey2agent",
        description="Diagnostic testbed for selective QA over multi-source personal memory.",
    )
    sub = parser.add_subparsers(dest="command", required=False)

    for name, help_text in (
        ("run-extraction", "Persona-level atom extraction; writes extraction bundles."),
        ("run-llm-direct", "Whole-persona LLM-Direct answers; writes answer bundles."),
        ("run-schema-aware", "Whole-persona Schema-Aware answers; writes answer bundles."),
        ("run-struct-llm", "Whole-persona Struct-LLM answers (atom grid input)."),
    ):
        sp = sub.add_parser(name, help=help_text)
        sp.add_argument("--provider", required=True, choices=["openai", "google", "openrouter"])
        sp.add_argument("--model", required=True, help="Key in configs/models.yaml.")
        sp.add_argument(
            "--seed",
            required=True,
            type=Path,
            help="Seed dataset directory containing one sub-directory per persona.",
        )
        sp.add_argument(
            "--personas",
            required=True,
            help="One of: 'all' or a path to a CSV / newline-delimited id file.",
        )
        sp.add_argument("--output-dir", required=True, type=Path)
        sp.add_argument("--allow-api-call", action="store_true", default=False)
        sp.add_argument(
            "--cache-dir",
            type=Path,
            default=None,
            help=f"SHA256 cache root. Defaults to {_DEFAULT_CACHE_ROOT}.",
        )
        sp.add_argument(
            "--batch-handle-out",
            type=Path,
            default=None,
            help="If set, submit the batch, persist the returned BatchHandle "
                 "JSON to this path, and exit without polling.",
        )
        if name == "run-struct-llm":
            sp.add_argument(
                "--extraction-bundle-dir",
                type=Path,
                required=True,
                help="Directory of {persona}.json extraction bundles produced by run-extraction.",
            )

    few = sub.add_parser(
        "run-few-shot",
        help="Per-(persona, question) few-shot LLM answers; writes answer bundles.",
    )
    few.add_argument("--provider", required=True, choices=["openai", "google", "openrouter"])
    few.add_argument("--model", required=True, help="Key in configs/models.yaml.")
    few.add_argument(
        "--seed",
        required=True,
        type=Path,
        help="Seed dataset directory containing one sub-directory per persona.",
    )
    few.add_argument(
        "--personas",
        required=True,
        help="One of: 'all' or a path to a CSV / newline-delimited id file.",
    )
    few.add_argument(
        "--questions",
        default="all",
        help="One of: 'all', a comma-separated qid list, or a newline-delimited qid file.",
    )
    few.add_argument(
        "--configs-root",
        type=Path,
        default=_DEFAULT_FEW_SHOT_CONFIGS_ROOT,
        help=f"Few-shot prompt config root. Defaults to {_DEFAULT_FEW_SHOT_CONFIGS_ROOT}.",
    )
    few.add_argument("--output-dir", required=True, type=Path)
    few.add_argument("--allow-api-call", action="store_true", default=False)
    few.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help=f"SHA256 cache root. Defaults to {_DEFAULT_CACHE_ROOT}.",
    )
    few.add_argument(
        "--batch-handle-out",
        type=Path,
        default=None,
        help="If set, submit the batch, persist the returned BatchHandle "
             "JSON to this path, and exit without polling.",
    )

    fetch = sub.add_parser(
        "fetch-batch",
        help="Fetch and parse results for an already-submitted batch job.",
    )
    fetch.add_argument("--provider", required=True, choices=["openai", "google"])
    fetch.add_argument("--model", required=True, help="Key in configs/models.yaml.")
    fetch.add_argument(
        "--handle", required=True, type=Path, help="Path to a BatchHandle.to_json() payload."
    )
    fetch.add_argument("--output-dir", required=True, type=Path)
    fetch.add_argument("--cache-dir", type=Path, default=None)

    return parser


# ---------------------------------------------------------------------------
# Client factory
# ---------------------------------------------------------------------------


def _make_client(
    provider: str,
    model_key: str,
    cache_dir: Path | None,
) -> SyncLLMClient | BatchLLMClient:
    """Instantiate a provider client from ``configs/models.yaml``."""
    from survey2agent.api_clients import (
        GeminiBatchClient,
        OpenAIBatchClient,
        OpenRouterSyncClient,
    )

    cfg = load_model_config(_CONFIGS_MODELS_YAML, model_key)
    if cfg.provider != provider:
        raise SystemExit(
            f"--provider {provider!r} does not match configs/models.yaml entry "
            f"{model_key!r} (provider={cfg.provider!r})"
        )
    cache = CacheStore(root=cache_dir or _DEFAULT_CACHE_ROOT)
    if provider == "openai":
        return OpenAIBatchClient(cfg, cache)
    if provider == "google":
        return GeminiBatchClient(cfg, cache)
    if provider == "openrouter":
        return OpenRouterSyncClient(cfg, cache)
    raise SystemExit(f"unsupported provider: {provider}")


# ---------------------------------------------------------------------------
# Persona resolution
# ---------------------------------------------------------------------------


def _resolve_personas(
    seed_dir: Path, selector: str
) -> tuple[list[str], dict[str, Path]]:
    """Return (persona_ids, persona_dirs) for the requested selector."""
    seed_dir = Path(seed_dir)
    if not seed_dir.is_dir():
        raise SystemExit(f"seed dir does not exist: {seed_dir}")

    all_ids = sorted(
        p.name for p in seed_dir.iterdir()
        if p.is_dir() and (p / "structural_sources").is_dir()
    )
    all_dirs = {pid: seed_dir / pid for pid in all_ids}

    if selector == "all":
        return all_ids, all_dirs

    path = Path(selector)
    if path.is_file():
        ids: list[str] = []
        text = path.read_text(encoding="utf-8")
        if path.suffix.lower() == ".csv":
            reader = csv.DictReader(text.splitlines())
            if reader.fieldnames and "persona_id" in reader.fieldnames:
                ids = [row["persona_id"].strip() for row in reader if row.get("persona_id")]
            else:
                reader2 = csv.reader(text.splitlines())
                ids = [row[0].strip() for row in reader2 if row]
        else:
            ids = [line.strip() for line in text.splitlines() if line.strip()]
        missing = [pid for pid in ids if pid not in all_dirs]
        if missing:
            raise SystemExit(
                f"persona ids missing from seed dir {seed_dir}: "
                f"{missing[:5]}{'…' if len(missing) > 5 else ''}"
            )
        return ids, {pid: all_dirs[pid] for pid in ids}

    raise SystemExit(
        f"--personas must be 'all' or a file path; got {selector!r}"
    )


def _resolve_few_shot_qids(selector: str) -> list[str] | None:
    """Return qids for ``run-few-shot``; ``None`` means all supported qids."""
    from survey2agent.methods._llm_prompt_builders import FEW_SHOT_QIDS

    if selector == "all":
        return None

    path = Path(selector)
    if path.is_file():
        qids = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        qids = [part.strip() for part in selector.split(",") if part.strip()]

    allowed = set(FEW_SHOT_QIDS)
    unknown = [qid for qid in qids if qid not in allowed]
    if unknown:
        raise SystemExit(
            f"unsupported few-shot question ids: {unknown[:5]}"
            f"{'...' if len(unknown) > 5 else ''}; allowed: {', '.join(FEW_SHOT_QIDS)}"
        )
    return qids


# ---------------------------------------------------------------------------
# Subcommand handlers
# ---------------------------------------------------------------------------


def _emit_cache_only_notice(allow_api_call: bool) -> None:
    if not allow_api_call:
        print(
            "Running in cache-only mode. Use --allow-api-call to hit the API.",
            file=sys.stderr,
        )


def _cmd_run_extraction(args: argparse.Namespace) -> int:
    from survey2agent.extraction.batch_pipeline import run_extraction_batch

    client = _make_client(args.provider, args.model, args.cache_dir)
    persona_ids, persona_dirs = _resolve_personas(args.seed, args.personas)
    _emit_cache_only_notice(args.allow_api_call)

    if args.batch_handle_out is not None:
        _submit_and_save_handle_extraction(client, persona_ids, persona_dirs, args)
        return 0

    report = run_extraction_batch(
        persona_ids=persona_ids,
        persona_dirs=persona_dirs,
        client=client,
        output_dir=args.output_dir,
        allow_api_call=args.allow_api_call,
    )
    _print_extraction_report(report)
    return 0 if report.n_failed == 0 else 1


def _cmd_run_llm_answers(args: argparse.Namespace, variant: str) -> int:
    from survey2agent.methods.llm_batch_runner import run_llm_answers_batch

    client = _make_client(args.provider, args.model, args.cache_dir)
    persona_ids, persona_dirs = _resolve_personas(args.seed, args.personas)
    _emit_cache_only_notice(args.allow_api_call)

    extraction_dir: Path | None = getattr(args, "extraction_bundle_dir", None)

    if args.batch_handle_out is not None:
        _submit_and_save_handle_answers(
            client, persona_ids, persona_dirs, variant, extraction_dir, args
        )
        return 0

    report = run_llm_answers_batch(
        persona_ids=persona_ids,
        persona_dirs=persona_dirs,
        variant=variant,  # type: ignore[arg-type]
        client=client,
        output_dir=args.output_dir,
        extraction_bundle_dir=extraction_dir,
        allow_api_call=args.allow_api_call,
    )
    _print_answers_report(report)
    return 0 if report.n_failed == 0 else 1


def _cmd_run_few_shot(args: argparse.Namespace) -> int:
    from survey2agent.methods.llm_batch_runner import run_few_shot_batch

    client = _make_client(args.provider, args.model, args.cache_dir)
    persona_ids, persona_dirs = _resolve_personas(args.seed, args.personas)
    qids = _resolve_few_shot_qids(args.questions)
    configs_root = Path(args.configs_root)
    if not configs_root.is_dir():
        raise SystemExit(f"few-shot configs root does not exist: {configs_root}")
    _emit_cache_only_notice(args.allow_api_call)

    if args.batch_handle_out is not None:
        _submit_and_save_handle_few_shot(
            client, persona_ids, persona_dirs, qids, configs_root, args
        )
        return 0

    report = run_few_shot_batch(
        persona_ids=persona_ids,
        persona_dirs=persona_dirs,
        qids=qids,
        client=client,
        output_dir=args.output_dir,
        configs_root=configs_root,
        allow_api_call=args.allow_api_call,
    )
    _print_few_shot_report(report)
    return 0 if report.n_failed == 0 else 1


def _cmd_fetch_batch(args: argparse.Namespace) -> int:
    client = _make_client(args.provider, args.model, args.cache_dir)
    if not isinstance(client, BatchLLMClient):
        raise SystemExit(f"provider {args.provider!r} does not support batch fetch")
    handle = BatchHandle.from_json(Path(args.handle).read_text(encoding="utf-8"))
    items = client.fetch_results(handle)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir / "batch_results.jsonl"
    with out.open("w", encoding="utf-8") as fh:
        for item in items:
            fh.write(
                json.dumps(
                    {
                        "custom_id": item.custom_id,
                        "text": item.text,
                        "finish_reason": item.finish_reason,
                        "error_message": item.error_message,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    print(f"fetched {len(items)} items → {out}", file=sys.stderr)
    return 0


# ---------------------------------------------------------------------------
# Batch-handle persistence helpers
# ---------------------------------------------------------------------------


def _submit_and_save_handle_extraction(
    client: Any,
    persona_ids: list[str],
    persona_dirs: dict[str, Path],
    args: argparse.Namespace,
) -> None:
    if not isinstance(client, BatchLLMClient):
        raise SystemExit("--batch-handle-out requires a batch provider")
    from dataclasses import replace
    from survey2agent.extraction.extractor import build_persona_extraction_request

    requests = []
    for persona_id in persona_ids:
        req = build_persona_extraction_request(persona_dirs[persona_id])
        requests.append(replace(req, custom_id=persona_id))
    handle = client.submit_batch(requests)
    args.batch_handle_out.parent.mkdir(parents=True, exist_ok=True)
    args.batch_handle_out.write_text(handle.to_json(), encoding="utf-8")
    print(f"submitted batch; handle → {args.batch_handle_out}", file=sys.stderr)


def _submit_and_save_handle_answers(
    client: Any,
    persona_ids: list[str],
    persona_dirs: dict[str, Path],
    variant: str,
    extraction_dir: Path | None,
    args: argparse.Namespace,
) -> None:
    if not isinstance(client, BatchLLMClient):
        raise SystemExit("--batch-handle-out requires a batch provider")
    from dataclasses import replace
    from survey2agent.methods._llm_prompt_builders import (
        build_direct_request,
        build_schema_aware_request,
        build_struct_llm_request,
    )

    requests = []
    for persona_id in persona_ids:
        if variant == "direct":
            req = build_direct_request(persona_dirs[persona_id])
        elif variant == "schema_aware":
            req = build_schema_aware_request(persona_dirs[persona_id])
        else:
            assert extraction_dir is not None
            req = build_struct_llm_request(persona_id, extraction_dir / f"{persona_id}.json")
        requests.append(replace(req, custom_id=persona_id))
    handle = client.submit_batch(requests)
    args.batch_handle_out.parent.mkdir(parents=True, exist_ok=True)
    args.batch_handle_out.write_text(handle.to_json(), encoding="utf-8")
    print(f"submitted batch; handle → {args.batch_handle_out}", file=sys.stderr)


def _submit_and_save_handle_few_shot(
    client: Any,
    persona_ids: list[str],
    persona_dirs: dict[str, Path],
    qids: list[str] | None,
    configs_root: Path,
    args: argparse.Namespace,
) -> None:
    if not isinstance(client, BatchLLMClient):
        raise SystemExit("--batch-handle-out requires a batch provider")
    from dataclasses import replace
    from survey2agent.methods._llm_prompt_builders import (
        FEW_SHOT_QIDS,
        build_few_shot_request,
    )

    selected_qids = qids if qids is not None else FEW_SHOT_QIDS
    requests = []
    for persona_id in persona_ids:
        for qid in selected_qids:
            req = build_few_shot_request(persona_dirs[persona_id], qid, configs_root)
            requests.append(replace(req, custom_id=f"{persona_id}__{qid}"))

    handle = client.submit_batch(requests)
    args.batch_handle_out.parent.mkdir(parents=True, exist_ok=True)
    args.batch_handle_out.write_text(handle.to_json(), encoding="utf-8")
    print(f"submitted batch; handle → {args.batch_handle_out}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Report printers
# ---------------------------------------------------------------------------


def _print_extraction_report(report: Any) -> None:
    print(
        f"extraction: {report.n_success}/{report.n_personas} personas OK; "
        f"{report.n_failed} failed; output → {report.output_dir}",
        file=sys.stderr,
    )
    if report.failed_personas:
        print(
            f"failed personas: {report.failed_personas[:5]}"
            f"{'…' if len(report.failed_personas) > 5 else ''}; "
            f"see {report.failure_log_path}",
            file=sys.stderr,
        )


def _print_answers_report(report: Any) -> None:
    print(
        f"{report.variant}: {report.n_success}/{report.n_personas} personas OK; "
        f"{report.n_failed} failed; output → {report.output_dir}",
        file=sys.stderr,
    )
    if report.failed_personas:
        print(
            f"failed personas: {report.failed_personas[:5]}"
            f"{'…' if len(report.failed_personas) > 5 else ''}; "
            f"see {report.failure_log_path}",
            file=sys.stderr,
        )


def _print_few_shot_report(report: Any) -> None:
    print(
        f"few-shot: {report.n_success}/{report.n_total} persona-question calls OK; "
        f"{report.n_failed} failed; output → {report.output_dir}",
        file=sys.stderr,
    )
    if report.n_failed:
        print(f"see {report.failure_log_path}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


_DISPATCH = {
    "run-extraction": lambda a: _cmd_run_extraction(a),
    "run-llm-direct": lambda a: _cmd_run_llm_answers(a, "direct"),
    "run-schema-aware": lambda a: _cmd_run_llm_answers(a, "schema_aware"),
    "run-struct-llm": lambda a: _cmd_run_llm_answers(a, "struct_llm"),
    "run-few-shot": lambda a: _cmd_run_few_shot(a),
    "fetch-batch": lambda a: _cmd_fetch_batch(a),
}


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if not args.command:
        parser.print_help()
        return 0
    handler = _DISPATCH.get(args.command)
    if handler is None:
        print(f"unknown subcommand: {args.command}", file=sys.stderr)
        return 2
    return handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
