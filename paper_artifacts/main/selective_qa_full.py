"""Reproduce paper Full Selective QA Table — Full Selective QA Table (4-seed pooled).

Source: ``Appendix selective-QA section`` lines 1-29
(``\\label{tab:D-1}``). Re-implementation of the deleted
``paper_artifacts/table_5_selective.py`` (the appendix
table lives in the ``main/`` layout for runtime convenience; users
can move the script into ``appendix/`` later if preferred).

Reduction: per-seed-then-mean (``aggregate_per_seed_then_average``).
Cell-level paper-lock test (``test_paper_main_tables_reproduction.py``)
asserts every cell within ±0.005 of the published numbers.

17 rows, 34 cells (selective_accuracy + coverage):
  - T1 SSB (ext, oracle)
  - T2 NBF / DSNBF / ABF (ext, oracle each)
  - T3 4 LLM families × 2 prompting variants (ext only)
  - Ref. Source Reachability (oracle only; GT-aided direct-readout reference, not deployable)

Usage::

    python -m paper_artifacts.main.selective_qa_full
    python -m paper_artifacts.main.selective_qa_full --rows ssb_ext,dsnbf_ext
"""

from __future__ import annotations

import argparse
import sys
import time

from survey2agent.evaluation.data_loaders import build_training_records
from survey2agent.evaluation.multi_seed import aggregate_per_seed_then_average
from survey2agent.evaluation.runner import EvaluationResult, run_method
from survey2agent.methods import (
    ABFSelective,
    DSNBFSelective,
    LLMDirectSelective,
    LLMSchemaAwareSelective,
    NBFSelective,
    OracleExtraction,
    SSBSelective,
)

from .._common import (
    CANONICAL_SEEDS,
    PAPER_TOLERANCE,
    emit_row,
    load_shared_resources,
    run_llm_across_seeds,
    run_mixed_mode_across_seeds,
    run_oracle_mode_across_seeds,
    write_outputs,
    _md_fmt_pct,
)


# ── Paper Full Selective QA Table numbers (appendix_d_selective_fewshot.tex L13-29) ──────


# row_id → (tier, method_label, mode, sel_acc_paper, cov_paper, builder_kind)
TABLE_F1_SPEC: list[dict] = [
    # T1
    {"row_id": "ssb_ext",     "tier": "T1", "method_label": "SSB",      "mode": "ext",
     "sel_acc_paper": 0.808, "cov_paper": 0.829, "builder": "fusion_ext",
     "method_factory": lambda: SSBSelective()},
    {"row_id": "ssb_oracle",  "tier": "T1", "method_label": "SSB",      "mode": "oracle",
     "sel_acc_paper": 0.816, "cov_paper": 0.836, "builder": "fusion_oracle",
     "method_factory": lambda: SSBSelective()},
    # T2
    {"row_id": "nbf_ext",     "tier": "T2", "method_label": "NBF",      "mode": "ext",
     "sel_acc_paper": 0.859, "cov_paper": 0.775, "builder": "fusion_ext",
     "method_factory": lambda: NBFSelective()},
    {"row_id": "nbf_oracle",  "tier": "T2", "method_label": "NBF",      "mode": "oracle",
     "sel_acc_paper": 0.885, "cov_paper": 0.779, "builder": "fusion_oracle",
     "method_factory": lambda: NBFSelective()},
    {"row_id": "dsnbf_ext",   "tier": "T2", "method_label": "DSNBF",    "mode": "ext",
     "sel_acc_paper": 0.853, "cov_paper": 0.783, "builder": "fusion_ext",
     "method_factory": lambda: DSNBFSelective()},
    {"row_id": "dsnbf_oracle","tier": "T2", "method_label": "DSNBF",    "mode": "oracle",
     "sel_acc_paper": 0.888, "cov_paper": 0.772, "builder": "fusion_oracle",
     "method_factory": lambda: DSNBFSelective()},
    {"row_id": "abf_ext",     "tier": "T2", "method_label": "ABF",      "mode": "ext",
     "sel_acc_paper": 0.830, "cov_paper": 0.620, "builder": "fusion_ext",
     "method_factory": lambda: ABFSelective()},
    {"row_id": "abf_oracle",  "tier": "T2", "method_label": "ABF",      "mode": "oracle",
     "sel_acc_paper": 0.834, "cov_paper": 0.602, "builder": "fusion_oracle",
     "method_factory": lambda: ABFSelective()},
    # T3 LLM
    {"row_id": "gpt54_direct",       "tier": "T3", "method_label": "GPT-5.4 Direct",       "mode": "ext",
     "sel_acc_paper": 0.703, "cov_paper": 0.923, "builder": "llm",
     "llm": ("gpt-5.4", "direct", "GPT-5.4", LLMDirectSelective)},
    {"row_id": "gpt54_schema",       "tier": "T3", "method_label": "GPT-5.4 Schema",       "mode": "ext",
     "sel_acc_paper": 0.710, "cov_paper": 0.954, "builder": "llm",
     "llm": ("gpt-5.4", "schema-aware", "GPT-5.4", LLMSchemaAwareSelective)},
    {"row_id": "gemini_direct",      "tier": "T3", "method_label": "Gemini 3.1 Direct",    "mode": "ext",
     "sel_acc_paper": 0.684, "cov_paper": 0.985, "builder": "llm",
     "llm": ("gemini_p2", "direct", "Gemini 3.1", LLMDirectSelective)},
    {"row_id": "gemini_schema",      "tier": "T3", "method_label": "Gemini 3.1 Schema",    "mode": "ext",
     "sel_acc_paper": 0.704, "cov_paper": 0.992, "builder": "llm",
     "llm": ("gemini_p2", "schema-aware", "Gemini 3.1", LLMSchemaAwareSelective)},
    {"row_id": "deepseek_direct",    "tier": "T3", "method_label": "DeepSeek V3.2 Direct", "mode": "ext",
     "sel_acc_paper": 0.561, "cov_paper": 1.000, "builder": "llm",
     "llm": ("deepseek-v3.2", "direct", "DeepSeek V3.2", LLMDirectSelective)},
    {"row_id": "deepseek_schema",    "tier": "T3", "method_label": "DeepSeek V3.2 Schema", "mode": "ext",
     "sel_acc_paper": 0.569, "cov_paper": 1.000, "builder": "llm",
     "llm": ("deepseek-v3.2", "schema-aware", "DeepSeek V3.2", LLMSchemaAwareSelective)},
    {"row_id": "qwen3_direct",       "tier": "T3", "method_label": "Qwen3 235B Direct",    "mode": "ext",
     "sel_acc_paper": 0.422, "cov_paper": 0.993, "builder": "llm",
     "llm": ("qwen3-235b-a22b-2507", "direct", "Qwen3 235B", LLMDirectSelective)},
    {"row_id": "qwen3_schema",       "tier": "T3", "method_label": "Qwen3 235B Schema",    "mode": "ext",
     "sel_acc_paper": 0.480, "cov_paper": 0.988, "builder": "llm",
     "llm": ("qwen3-235b-a22b-2507", "schema-aware", "Qwen3 235B", LLMSchemaAwareSelective)},
    # Ref. Source Reachability (GT-aided direct-readout reference, not a deployable selective method)
    {"row_id": "oracle_ext", "tier": "Ref.", "method_label": "Source Reachability", "mode": "oracle",
     "sel_acc_paper": 1.000, "cov_paper": 0.932, "builder": "oracle_ext"},
]


def _run_oracle_ext_across_seeds(*, oracle_atoms, gts, splits, diff_idx):
    out: dict[str, list[EvaluationResult]] = {}
    for seed in CANONICAL_SEEDS:
        atoms = oracle_atoms[seed]
        gt = gts[seed]
        evalr = build_training_records(atoms, gt, splits["test"], difficulty_index=diff_idx)
        method = OracleExtraction(skip_on_miss=True)
        method.attach_gt({pid: gt[pid] for pid in splits["test"] if pid in gt})
        out[seed] = run_method(method, evalr)
    return out


def _build_seed_results(spec, *, splits, diff_idx, gts, oracle_atoms, llm_atoms):
    builder = spec["builder"]
    if builder == "fusion_oracle":
        return run_oracle_mode_across_seeds(
            spec["method_factory"],
            oracle_atoms=oracle_atoms, gts=gts, splits=splits, diff_idx=diff_idx,
        )
    if builder == "fusion_ext":
        return run_mixed_mode_across_seeds(
            spec["method_factory"],
            oracle_atoms=oracle_atoms, llm_atoms=llm_atoms,
            gts=gts, splits=splits, diff_idx=diff_idx,
        )
    if builder == "llm":
        model, variant, display, cls = spec["llm"]
        return run_llm_across_seeds(
            model=model, variant=variant, display_name=display, selective_cls=cls,
            llm_atoms=llm_atoms, gts=gts, splits=splits, diff_idx=diff_idx,
        )
    if builder == "oracle_ext":
        return _run_oracle_ext_across_seeds(
            oracle_atoms=oracle_atoms, gts=gts, splits=splits, diff_idx=diff_idx,
        )
    raise ValueError(f"unknown builder kind: {builder!r}")


def _render_md_table(rows: list[dict]) -> str:
    by_row: dict[str, dict] = {}
    for r in rows:
        key = r["row_id"]
        by_row.setdefault(key, {})[r["metric"]] = r

    lines: list[str] = []
    lines.append("|     | Method                    | Sel. Acc (%) | Cov (%) | μ* Sel. Acc (%) | μ* Cov (%) |")
    lines.append("|:----|:--------------------------|:------------:|:-------:|:---------------:|:----------:|")

    pairs: dict[str, dict[str, dict]] = {}
    label_tier: dict[str, str] = {}
    for spec in TABLE_F1_SPEC:
        m = spec["method_label"]
        pairs.setdefault(m, {})[spec["mode"]] = spec
        label_tier[m] = spec["tier"]

    last_tier = ""
    seen: set[str] = set()
    for spec in TABLE_F1_SPEC:
        m = spec["method_label"]
        if m in seen:
            continue
        seen.add(m)
        tier = label_tier[m]
        tier_cell = f"**{tier}**" if tier != last_tier else ""
        last_tier = tier

        ext_spec = pairs[m].get("ext")
        oracle_spec = pairs[m].get("oracle")

        def cell_pair(s):
            if s is None:
                return "—", "—"
            row_dict = by_row.get(s["row_id"], {})
            sel = row_dict.get("selective_accuracy")
            cov = row_dict.get("coverage")
            sel_v = sel["point"] if sel else None
            cov_v = cov["point"] if cov else None
            return _md_fmt_pct(sel_v), _md_fmt_pct(cov_v)

        ext_sel, ext_cov = cell_pair(ext_spec)
        oracle_sel, oracle_cov = cell_pair(oracle_spec)
        # Bold for DSNBF best cells (matching paper)
        if m == "DSNBF":
            ext_sel = f"**{ext_sel}**"
            oracle_sel = f"**{oracle_sel}**"
        lines.append(f"| {tier_cell} | {m} | {ext_sel} | {ext_cov} | {oracle_sel} | {oracle_cov} |")

    return "\n".join(lines)


def main(rows: list[str] | None = None) -> int:
    """Run Full Selective QA Table reproduction and write CSV + MD outputs.

    Returns
    -------
    Number of failed cells (0 = all within ±0.005 of paper).
    """
    t0 = time.time()
    splits, diff_idx, gts, oracle_atoms, llm_atoms = load_shared_resources()

    selected = TABLE_F1_SPEC
    if rows:
        wanted = set(rows)
        selected = [s for s in TABLE_F1_SPEC if s["row_id"] in wanted]
        if not selected:
            print(f"[selective_qa_full] no rows matched: {sorted(wanted)}", file=sys.stderr)
            return 1

    output_rows: list[dict] = []
    fail_count = 0

    for spec in selected:
        sr = _build_seed_results(
            spec, splits=splits, diff_idx=diff_idx, gts=gts,
            oracle_atoms=oracle_atoms, llm_atoms=llm_atoms,
        )
        n_personas = sum(len({r.persona_id for r in lst}) for lst in sr.values())

        sel_actual = aggregate_per_seed_then_average(sr, metric="selective_accuracy")
        cov_actual = aggregate_per_seed_then_average(sr, metric="coverage")

        for metric_name, point, paper_v in (
            ("selective_accuracy", sel_actual, spec["sel_acc_paper"]),
            ("coverage",            cov_actual, spec["cov_paper"]),
        ):
            row = emit_row(
                row_id=spec["row_id"],
                method_label=spec["method_label"],
                mode=spec["mode"],
                metric=metric_name,
                point=point,
                n_personas=n_personas,
                paper_point=paper_v,
            )
            if row["paper_match"].startswith("FAIL"):
                fail_count += 1
            output_rows.append(row)

    md_table = _render_md_table(output_rows)
    caption = (
        "**Full Selective QA Table.** Full Selective QA results (4-seed pooled). Sel. Acc and Cov "
        "report each method under its canonical input (LLM-extracted μ for T1/T2, "
        "NL for T3). Fusion methods are additionally evaluated under oracle μ. "
        "T3 rows grouped by family. Reduction: mean of per-seed macro values."
    )
    footnotes = [
        f"*Paper location: `Appendix selective-QA section` (`tab:D-1`).*",
        f"*Reduction: `aggregate_per_seed_then_average` (per-seed point estimates, then arithmetic mean across 4 canonical seeds).*",
        f"*Reproduction tolerance: ±{PAPER_TOLERANCE} absolute. Cells outside tolerance are marked `FAIL d=...` in the companion CSV.*",
    ]

    csv_p, md_p = write_outputs(
        "selective_qa",
        output_rows,
        md_table=md_table,
        script_name="paper_artifacts.main.selective_qa_full",
        md_caption=caption,
        md_footnotes=footnotes,
        subdir="main",
    )

    elapsed = time.time() - t0
    n_pass = len(output_rows) - fail_count
    print(
        f"[selective_qa_full] wrote {len(output_rows)} cells "
        f"({n_pass} OK / {fail_count} FAIL) in {elapsed:.1f}s "
        f"-> {csv_p}, {md_p}"
    )
    return fail_count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rows", default="",
        help="comma-separated row_id filter (default: all 17 rows)",
    )
    args = parser.parse_args()
    row_filter = [r.strip() for r in args.rows.split(",") if r.strip()] or None
    sys.exit(0 if main(rows=row_filter) == 0 else 1)
