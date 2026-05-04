"""Shared utilities for paper-table reproduction scripts.

All numeric outputs are written to ``paper_artifacts/output/`` as a CSV +
Markdown pair. ``PAPER_TOLERANCE`` (±0.005) is the accept/reject band.

The CSV schema is fixed (column order matters for downstream parsing):

    row_id, method_label, mode, metric, point, ci_low, ci_high,
    n_seeds, n_personas, paper_value, paper_match

``ci_low`` / ``ci_high`` are blank for point-only cells (2x2 Factorial Decomposition).
``paper_match`` is ``"OK"`` if ``|point - paper_value| <= tolerance``,
else ``"FAIL d=<delta>"``. ``OK``/``FAIL`` are ASCII to keep the CSV
parseable on Windows.
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from survey2agent.evaluation.data_loaders import (
    build_training_records,
    load_atoms_for_seed,
    load_ground_truths,
    load_persona_difficulty_index,
    load_splits,
)
from survey2agent.evaluation.multi_seed import CANONICAL_SEEDS
from survey2agent.evaluation.runner import EvaluationResult, run_method
from survey2agent.methods import FrozenBulkJSONSource, StructLLMSource
from survey2agent.methods.base import Method

PACKAGE_DIR: Path = Path(__file__).parent
OUTPUT_DIR: Path = PACKAGE_DIR / "output"
PAPER_TOLERANCE: float = 0.005

CSV_COLUMNS: tuple[str, ...] = (
    "row_id",
    "method_label",
    "mode",
    "metric",
    "point",
    "ci_low",
    "ci_high",
    "n_seeds",
    "n_personas",
    "paper_value_low",
    "paper_value_point",
    "paper_value_high",
    "paper_match",
)


# ── Per-seed record builders (cache atom loads at module scope) ─────────


_OrSplits = dict[str, list[str]]
_DiffIdx = dict[str, str]
_Atoms = dict[str, dict]
_GTs = dict[str, dict[str, dict[str, str]]]


def load_shared_resources() -> tuple[_OrSplits, _DiffIdx, _GTs, _Atoms, _Atoms]:
    """Load splits, difficulty index, GTs, oracle atoms, and LLM atoms.

    Returns
    -------
    splits, difficulty_index, gts_by_seed, oracle_atoms_by_seed, llm_atoms_by_seed
    """
    splits = load_splits()
    diff_idx = load_persona_difficulty_index()
    gts = {s: load_ground_truths(s) for s in CANONICAL_SEEDS}
    oracle_atoms = {s: load_atoms_for_seed(s, mode="oracle") for s in CANONICAL_SEEDS}
    llm_atoms = {s: load_atoms_for_seed(s, mode="llm") for s in CANONICAL_SEEDS}
    return splits, diff_idx, gts, oracle_atoms, llm_atoms


def run_oracle_mode_across_seeds(
    method_factory: Callable[[], Method],
    *,
    oracle_atoms: Mapping[str, dict],
    gts: Mapping[str, dict[str, dict[str, str]]],
    splits: Mapping[str, list[str]],
    diff_idx: Mapping[str, str],
    split: str = "test",
) -> dict[str, list[EvaluationResult]]:
    """Fit / cal / predict all on oracle atoms (paper "μ* input" / direct-readout columns)."""
    out: dict[str, list[EvaluationResult]] = {}
    for seed in CANONICAL_SEEDS:
        atoms = oracle_atoms[seed]
        gt = gts[seed]
        train = build_training_records(atoms, gt, splits["train"], difficulty_index=diff_idx)
        cal = build_training_records(atoms, gt, splits["cal"], difficulty_index=diff_idx)
        evalr = build_training_records(atoms, gt, splits[split], difficulty_index=diff_idx)
        method = method_factory()
        kw: dict = {}
        if method.requires_fit:
            kw["train_records"] = train
        if method.requires_calibration:
            kw["cal_records"] = cal
        out[seed] = run_method(method, evalr, **kw)
    return out


def run_mixed_mode_across_seeds(
    method_factory: Callable[[], Method],
    *,
    oracle_atoms: Mapping[str, dict],
    llm_atoms: Mapping[str, dict],
    gts: Mapping[str, dict[str, dict[str, str]]],
    splits: Mapping[str, list[str]],
    diff_idx: Mapping[str, str],
    split: str = "test",
) -> dict[str, list[EvaluationResult]]:
    """Fit / cal on oracle atoms, predict on LLM-extracted atoms.

    Mirrors the pattern in
    ``tests/integration/test_paper_table5_reproduction.py`` and
    ``scripts/_validate_cp0119_paper_numbers.py``. Used for 2x2 Factorial Decomposition
    "Sel.Acc / Cov" columns and Forced-Accuracy Main Table fusion rows on extracted μ.
    """
    out: dict[str, list[EvaluationResult]] = {}
    for seed in CANONICAL_SEEDS:
        oa = oracle_atoms[seed]
        la = llm_atoms[seed]
        gt = gts[seed]
        train = build_training_records(oa, gt, splits["train"], difficulty_index=diff_idx)
        cal = build_training_records(oa, gt, splits["cal"], difficulty_index=diff_idx)
        evalr = build_training_records(la, gt, splits[split], difficulty_index=diff_idx)
        method = method_factory()
        kw: dict = {}
        if method.requires_fit:
            kw["train_records"] = train
        if method.requires_calibration:
            kw["cal_records"] = cal
        out[seed] = run_method(method, evalr, **kw)
    return out


def run_llm_across_seeds(
    *,
    model: str,
    variant: str,
    display_name: str,
    selective_cls: type[Method],
    llm_atoms: Mapping[str, dict],
    gts: Mapping[str, dict[str, dict[str, str]]],
    splits: Mapping[str, list[str]],
    diff_idx: Mapping[str, str],
    split: str = "test",
) -> dict[str, list[EvaluationResult]]:
    """LLM rows: source is seed-bound, no fit / cal."""
    out: dict[str, list[EvaluationResult]] = {}
    for seed in CANONICAL_SEEDS:
        atoms = llm_atoms[seed]
        gt = gts[seed]
        evalr = build_training_records(atoms, gt, splits[split], difficulty_index=diff_idx)
        method = selective_cls(
            source=FrozenBulkJSONSource(model=model, seed=seed, variant=variant),
            model_display_name=display_name,
        )
        out[seed] = run_method(method, evalr)
    return out


def run_struct_llm_across_seeds(
    *,
    mode: str,
    display_name: str,
    method_cls: type[Method],
    eval_atoms: Mapping[str, dict],
    gts: Mapping[str, dict[str, dict[str, str]]],
    splits: Mapping[str, list[str]],
    diff_idx: Mapping[str, str],
    split: str = "test",
) -> dict[str, list[EvaluationResult]]:
    """Struct-LLM rows: read frozen ``StructLLMSource`` per seed, no fit / cal.

    ``mode`` ∈ {``"oracle"``, ``"extracted"``} selects the structured-LLM
    artifact family (see ``StructLLMSource``). ``eval_atoms`` provides
    the per-seed atom dict used to build the eval records (the atoms
    themselves are only used to recover the ``persona_id``; the source
    delivers the answer). For 2x2 Factorial Decomposition ``llm_oracle`` cell pass oracle atoms;
    for ``llm_extracted`` pass LLM-extracted atoms — keeping eval-record
    construction symmetric with the matching fusion cell.
    """
    if mode not in ("oracle", "extracted"):
        raise ValueError(f"mode must be 'oracle' or 'extracted'; got {mode!r}")
    out: dict[str, list[EvaluationResult]] = {}
    for seed in CANONICAL_SEEDS:
        atoms = eval_atoms[seed]
        gt = gts[seed]
        evalr = build_training_records(atoms, gt, splits[split], difficulty_index=diff_idx)
        method = method_cls(
            source=StructLLMSource(mode=mode, seed=seed),
            model_display_name=display_name,
        )
        out[seed] = run_method(method, evalr)
    return out


# ── Output formatting ───────────────────────────────────────────────────


def paper_match_marker(
    point: float | None,
    paper_value: float | None,
    *,
    tolerance: float = PAPER_TOLERANCE,
) -> str:
    """Return ``"OK"`` if within tolerance, else ``"FAIL d=X.XXXX"``.

    Returns ``"NA"`` when either value is ``None`` (e.g. paper-cell-only
    rows like LLM cells in Per-Type Diagnostic Analysis).
    """
    if point is None or paper_value is None:
        return "NA"
    delta = abs(float(point) - float(paper_value))
    if delta <= tolerance:
        return "OK"
    return f"FAIL d={delta:.4f}"


def _fmt(value: float | None, *, digits: int = 4) -> str:
    if value is None:
        return ""
    return f"{float(value):.{digits}f}"


def _decimal_round(value: float, *, digits: int) -> str:
    """Format decimal-style table values with half-up rounding.

    Python's default float formatting uses round-half-even, which can display
    boundary cells such as 77.25 as 77.2. Paper tables use conventional
    half-up one-decimal percentages, so Markdown reproduction tables should
    match that display convention.
    """
    quantum = Decimal("1").scaleb(-digits)
    # Add a tiny epsilon before decimal conversion so binary floating-point
    # representations of exact table boundaries (e.g. 77.25 stored as
    # 77.24999999999999) do not round down in Markdown display.
    adjusted = float(value) + 1e-9
    return format(Decimal(str(adjusted)).quantize(quantum, rounding=ROUND_HALF_UP), "f")


def _md_fmt_pct(value: float | None, *, digits: int = 1) -> str:
    if value is None:
        return "—"
    return _decimal_round(100.0 * float(value), digits=digits)


def _md_fmt_ci(low: float | None, high: float | None, *, digits: int = 1) -> str:
    if low is None or high is None:
        return ""
    return (
        f"[{_decimal_round(100.0 * float(low), digits=digits)}, "
        f"{_decimal_round(100.0 * float(high), digits=digits)}]"
    )


def _header_comment(script_name: str) -> str:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return (
        f"# Generated by {script_name} at {ts}\n"
        f"# Tolerance: ±{PAPER_TOLERANCE} (absolute, on the unit scale)\n"
        "# Frozen artifacts: data/method_outputs/, "
        "data/extracted_atoms/\n"
    )


def write_outputs(
    table_id: str,
    rows: Sequence[Mapping[str, Any]],
    *,
    md_table: str,
    script_name: str,
    md_caption: str = "",
    md_footnotes: Sequence[str] = (),
    subdir: str = "",
) -> tuple[Path, Path]:
    """Write ``output[/subdir]/<table_id>.csv`` and ``output[/subdir]/<table_id>.md``.

    Each row in ``rows`` must contain at least the keys in :data:`CSV_COLUMNS`.
    Extra keys are ignored. ``md_table`` is the pre-rendered Markdown body
    (the script knows the paper layout best). ``subdir`` (e.g. ``"main"``
    or ``"appendix"``) selects a subdirectory under :data:`OUTPUT_DIR`;
    default writes to :data:`OUTPUT_DIR` itself for backward compat.
    """
    out_dir = OUTPUT_DIR / subdir if subdir else OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{table_id}.csv"
    md_path = out_dir / f"{table_id}.md"

    header = _header_comment(script_name)

    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        for line in header.splitlines():
            fh.write(line + "\n")
        writer = csv.DictWriter(fh, fieldnames=list(CSV_COLUMNS), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            normalized = {col: row.get(col, "") for col in CSV_COLUMNS}
            for k in (
                "point", "ci_low", "ci_high",
                "paper_value_low", "paper_value_point", "paper_value_high",
            ):
                v = normalized.get(k, "")
                if isinstance(v, float):
                    normalized[k] = _fmt(v)
                elif v is None:
                    normalized[k] = ""
            writer.writerow(normalized)

    md_lines: list[str] = []
    md_lines.append(header.rstrip())
    md_lines.append("")
    md_lines.append(f"# {table_id}")
    if md_caption:
        md_lines.append("")
        md_lines.append(md_caption)
    md_lines.append("")
    md_lines.append(md_table.rstrip())
    if md_footnotes:
        md_lines.append("")
        for fn in md_footnotes:
            md_lines.append(fn)
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    return csv_path, md_path


def emit_row(
    *,
    row_id: str,
    method_label: str,
    mode: str,
    metric: str,
    point: float | None,
    ci_low: float | None = None,
    ci_high: float | None = None,
    n_seeds: int = len(CANONICAL_SEEDS),
    n_personas: int | str = "",
    paper_point: float | None = None,
    paper_low: float | None = None,
    paper_high: float | None = None,
    tolerance: float = PAPER_TOLERANCE,
) -> dict[str, Any]:
    """Build one CSV row dict with the canonical 13 columns + paper match."""
    if paper_point is None:
        marker = "NA"
    else:
        marker = paper_match_marker(point, paper_point, tolerance=tolerance)
    return {
        "row_id": row_id,
        "method_label": method_label,
        "mode": mode,
        "metric": metric,
        "point": point,
        "ci_low": ci_low if ci_low is not None else "",
        "ci_high": ci_high if ci_high is not None else "",
        "n_seeds": n_seeds,
        "n_personas": n_personas,
        "paper_value_low": paper_low if paper_low is not None else "",
        "paper_value_point": paper_point if paper_point is not None else "",
        "paper_value_high": paper_high if paper_high is not None else "",
        "paper_match": marker,
    }


# Re-export numeric formatters for table scripts
__all__ = [
    "OUTPUT_DIR",
    "PACKAGE_DIR",
    "PAPER_TOLERANCE",
    "CSV_COLUMNS",
    "CANONICAL_SEEDS",
    "load_shared_resources",
    "run_oracle_mode_across_seeds",
    "run_mixed_mode_across_seeds",
    "run_llm_across_seeds",
    "run_struct_llm_across_seeds",
    "paper_match_marker",
    "write_outputs",
    "emit_row",
    "_md_fmt_pct",
    "_md_fmt_ci",
]
