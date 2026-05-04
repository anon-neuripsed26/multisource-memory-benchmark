"""Reproduce paper Per-Type Accuracy — Full per-type macro accuracy (4-seed pooled).

Source: ``Appendix diagnostics section`` lines 1-22
(``\\label{tab:E1}``).

Per-type marginal of Tables E2a and E2b plus three additional columns
(GPT-μ*, Source Reachability, full LLM grid) that the per-difficulty tables drop.

8 rows × 13 numeric columns (104 paper-lock cells; the ``# Q`` column is
metadata, not a paper-lock target):

  Type | # Q | DSNBF | NBF | SSB | GPT-μ* | GPT-S | GPT-D | Gem-D |
  Gem-S | DS-D | DS-S | QW-D | QW-S | Source Reachability

  - T2 fusion (DSNBF, NBF, SSB) on **μ* (direct)**, forced.
  - GPT-μ* = ``StructLLMSource(mode="oracle")`` consumed by ``LLMDirect``,
    forced.
  - 8 LLM cells = 4 families × 2 prompting variants on NL input, forced.
  - Source Reachability = ``OracleExtraction(skip_on_miss=False)`` ceiling on μ* atoms.

Reduction: pool-then-aggregate per reasoning type (point estimate from
``aggregate_pooled_with_breakdown_ci`` with ``breakdown_key="reasoning_type"``,
``metric="forced_accuracy"``). No CIs displayed in this table.

Row order matches paper Per-Type Accuracy (DSNBF descending): Ident, Arbit, Ctrl,
Factor, Temp, P-R, Miss, Annot.
"""

from __future__ import annotations

import argparse
import sys
import time

from survey2agent.evaluation.multi_seed import aggregate_pooled_with_breakdown_ci
from survey2agent.methods import (
    DSNBF,
    LLMDirect,
    LLMSchemaAware,
    NBF,
    OracleExtraction,
    SSB,
)

from .._common import (
    PAPER_TOLERANCE,
    emit_row,
    load_shared_resources,
    run_llm_across_seeds,
    run_oracle_mode_across_seeds,
    run_struct_llm_across_seeds,
    write_outputs,
    _md_fmt_pct,
)
from ..main.per_type_accuracy import _run_oracle_ext_per_type


# ── Paper Per-Type Accuracy numbers (appendix_e_diagnostics.tex L13-21) ──────────────────


# (type_key, label, n_q, dsnbf, nbf, ssb, str_orc, gpt_s, gpt_d, gem_d, gem_s, ds_d, ds_s, qw_d, qw_s, oracle)
PAPER_TAB_PER_TYPE_MACRO_ACCURACY_FULL = [
    ("identity",     "B · Ident",  2, 0.968, 0.968, 0.964, 0.970, 0.945, 0.819, 0.716, 0.814, 0.774, 0.740, 0.581, 0.554, 0.997),
    ("arbitration",  "A · Arbit",  3, 0.875, 0.869, 0.767, 0.685, 0.733, 0.732, 0.720, 0.751, 0.609, 0.610, 0.496, 0.534, 0.941),
    ("control",      "Ctrl",       2, 0.857, 0.861, 0.865, 0.757, 0.765, 0.791, 0.762, 0.777, 0.570, 0.517, 0.485, 0.472, 0.960),
    ("causal",       "E · Factor", 2, 0.831, 0.809, 0.766, 0.706, 0.709, 0.708, 0.696, 0.702, 0.557, 0.597, 0.104, 0.205, 0.950),
    ("trend",        "D · Temp",   2, 0.822, 0.777, 0.767, 0.762, 0.686, 0.725, 0.707, 0.681, 0.615, 0.665, 0.523, 0.548, 0.900),
    ("plan_reality", "C · P-R",    2, 0.816, 0.828, 0.819, 0.734, 0.694, 0.698, 0.698, 0.693, 0.465, 0.482, 0.579, 0.578, 0.948),
    ("missing_data", "F · Miss",   3, 0.779, 0.786, 0.783, 0.715, 0.685, 0.642, 0.643, 0.674, 0.435, 0.478, 0.258, 0.460, 0.920),
    ("annotation",   "G · Annot",  2, 0.635, 0.657, 0.608, 0.585, 0.344, 0.375, 0.479, 0.499, 0.499, 0.486, 0.385, 0.470, 0.837),
]


# (model, variant, family_short, regime_short, cls)
LLM_VARIANTS_D1 = [
    ("gpt-5.4",                "schema-aware", "GPT", "S", LLMSchemaAware),
    ("gpt-5.4",                "direct",       "GPT", "D", LLMDirect),
    ("gemini_p2",              "direct",       "Gem", "D", LLMDirect),
    ("gemini_p2",              "schema-aware", "Gem", "S", LLMSchemaAware),
    ("deepseek-v3.2",          "direct",       "DS",  "D", LLMDirect),
    ("deepseek-v3.2",          "schema-aware", "DS",  "S", LLMSchemaAware),
    ("qwen3-235b-a22b-2507",   "direct",       "QW",  "D", LLMDirect),
    ("qwen3-235b-a22b-2507",   "schema-aware", "QW",  "S", LLMSchemaAware),
]

# Map (family, regime) tag -> column index (1-based: dsnbf=0..nbf=1..ssb=2..str_orc=3..GPT-S=4..)
LLM_TAG_TO_PAPER_IDX = {
    "GPT-S": 4, "GPT-D": 5, "Gem-D": 6, "Gem-S": 7,
    "DS-D": 8, "DS-S": 9, "QW-D": 10, "QW-S": 11,
}


def _per_type_points(seed_results) -> dict[str, float]:
    """Pool-then-aggregate per reasoning_type, return ``{type_key: point}``."""
    ci = aggregate_pooled_with_breakdown_ci(
        seed_results, breakdown_key="reasoning_type",
        metric="forced_accuracy", n_bootstrap=200, seed=42,
    )
    return {k: v["point"] for k, v in ci.items()}


def main() -> int:
    t0 = time.time()
    splits, diff_idx, gts, oracle_atoms, llm_atoms = load_shared_resources()

    print("[per_type_macro_accuracy_full] running DSNBF/NBF/SSB on oracle...", flush=True)
    sr_dsnbf = run_oracle_mode_across_seeds(
        DSNBF, oracle_atoms=oracle_atoms, gts=gts, splits=splits, diff_idx=diff_idx,
    )
    sr_nbf = run_oracle_mode_across_seeds(
        NBF, oracle_atoms=oracle_atoms, gts=gts, splits=splits, diff_idx=diff_idx,
    )
    sr_ssb = run_oracle_mode_across_seeds(
        SSB, oracle_atoms=oracle_atoms, gts=gts, splits=splits, diff_idx=diff_idx,
    )
    print("[per_type_macro_accuracy_full] running GPT-μ*...", flush=True)
    sr_struct = run_struct_llm_across_seeds(
        mode="oracle", display_name="GPT-μ* (direct-readout)",
        method_cls=LLMDirect, eval_atoms=oracle_atoms,
        gts=gts, splits=splits, diff_idx=diff_idx,
    )
    print("[per_type_macro_accuracy_full] running Source Reachability...", flush=True)
    sr_oracle_ext = _run_oracle_ext_per_type(
        oracle_atoms=oracle_atoms, gts=gts, splits=splits, diff_idx=diff_idx,
    )

    sr_llm_by_tag: dict[str, dict] = {}
    for model, variant, fam, reg, cls in LLM_VARIANTS_D1:
        tag = f"{fam}-{reg}"
        print(f"[per_type_macro_accuracy_full] running LLM {tag} ({model} {variant})...", flush=True)
        sr_llm_by_tag[tag] = run_llm_across_seeds(
            model=model, variant=variant, display_name=f"{fam} {variant}",
            selective_cls=cls, llm_atoms=llm_atoms, gts=gts,
            splits=splits, diff_idx=diff_idx,
        )

    # Compute per-type points for every column
    points_dsnbf  = _per_type_points(sr_dsnbf)
    points_nbf    = _per_type_points(sr_nbf)
    points_ssb    = _per_type_points(sr_ssb)
    points_struct = _per_type_points(sr_struct)
    points_oracle = _per_type_points(sr_oracle_ext)
    points_llm: dict[str, dict[str, float]] = {
        tag: _per_type_points(sr) for tag, sr in sr_llm_by_tag.items()
    }

    output_rows: list[dict] = []
    fail_count = 0

    def add_cell(*, type_key, label, col_label, point, paper_v, mode):
        nonlocal fail_count
        row = emit_row(
            row_id=f"{type_key}__{col_label}",
            method_label=f"{col_label} :: {label}",
            mode=mode, metric="forced_accuracy",
            point=point, n_personas="",
            paper_point=paper_v,
        )
        output_rows.append(row)
        if row["paper_match"].startswith("FAIL"):
            fail_count += 1

    for spec in PAPER_TAB_PER_TYPE_MACRO_ACCURACY_FULL:
        (type_key, label, n_q,
         p_dsnbf, p_nbf, p_ssb, p_struct,
         p_gpt_s, p_gpt_d, p_gem_d, p_gem_s,
         p_ds_d, p_ds_s, p_qw_d, p_qw_s,
         p_oracle) = spec
        add_cell(type_key=type_key, label=label, col_label="DSNBF",
                 point=points_dsnbf.get(type_key), paper_v=p_dsnbf, mode="oracle")
        add_cell(type_key=type_key, label=label, col_label="NBF",
                 point=points_nbf.get(type_key), paper_v=p_nbf, mode="oracle")
        add_cell(type_key=type_key, label=label, col_label="SSB",
                 point=points_ssb.get(type_key), paper_v=p_ssb, mode="oracle")
        add_cell(type_key=type_key, label=label, col_label="GPT-μ*",
                 point=points_struct.get(type_key), paper_v=p_struct, mode="oracle")
        # 8 LLM columns in paper order (GPT-S, GPT-D, Gem-D, Gem-S, DS-D, DS-S, QW-D, QW-S)
        for tag, paper_v in (
            ("GPT-S", p_gpt_s), ("GPT-D", p_gpt_d),
            ("Gem-D", p_gem_d), ("Gem-S", p_gem_s),
            ("DS-D",  p_ds_d),  ("DS-S",  p_ds_s),
            ("QW-D",  p_qw_d),  ("QW-S",  p_qw_s),
        ):
            add_cell(type_key=type_key, label=label, col_label=tag,
                     point=points_llm[tag].get(type_key), paper_v=paper_v, mode="ext")
        add_cell(type_key=type_key, label=label, col_label="Source Reachability",
                 point=points_oracle.get(type_key), paper_v=p_oracle, mode="oracle")

    # ── Render Markdown ───────────────────────────────────────────
    by_id = {r["row_id"]: r for r in output_rows}
    cols = ["DSNBF", "NBF", "SSB", "GPT-μ*", "GPT-S", "GPT-D", "Gem-D", "Gem-S",
            "DS-D", "DS-S", "QW-D", "QW-S", "Source Reachability"]
    md_lines = [
        "| Type        | # Q | " + " | ".join(cols) + " |",
        "|:------------|:---:|" + "|".join([":---:"] * len(cols)) + "|",
    ]
    for spec in PAPER_TAB_PER_TYPE_MACRO_ACCURACY_FULL:
        type_key, label, n_q = spec[0], spec[1], spec[2]
        cells = [
            _md_fmt_pct(by_id[f"{type_key}__{c}"]["point"]) for c in cols
        ]
        md_lines.append(f"| {label} | {n_q} | " + " | ".join(cells) + " |")
    md_table = "\n".join(md_lines)

    caption = (
        "**Per-Type Accuracy.** Full per-type macro accuracy (%, answer-only mode, 4-seed pooled). "
        "T1/T2 methods (DSNBF, NBF, SSB) use μ* atoms; T3 LLM methods (8 cells) use NL input. "
        "GPT-μ* = GPT-5.4 reading the same μ* atoms as structured prompt input. Source Reachability = "
        "`OracleExtraction` reference (GT-aided direct readout, not deployable) on μ* atoms. Types ordered by paper DSNBF descending. "
        "GPT = GPT-5.4; Gem = Gemini 3.1; DS = DeepSeek V3.2; QW = Qwen3 235B. "
        "D = Direct; S = Schema."
    )
    footnotes = [
        "*Paper location: `Appendix diagnostics section` (`tab:E1`).*",
        "*Reduction: per-type pool-then-aggregate "
        "(`aggregate_pooled_with_breakdown_ci`, `breakdown_key=\"reasoning_type\"`, "
        "`metric=\"forced_accuracy\"`).*",
        f"*Reproduction tolerance: ±{PAPER_TOLERANCE} absolute on per-type points.*",
    ]

    csv_p, md_p = write_outputs(
        "per_type_macro_accuracy_full",
        output_rows,
        md_table=md_table,
        script_name="paper_artifacts.appendix.per_type_macro_accuracy_full",
        md_caption=caption,
        md_footnotes=footnotes,
        subdir="appendix",
    )
    elapsed = time.time() - t0
    n_pass = len(output_rows) - fail_count
    print(
        f"[per_type_macro_accuracy_full] wrote {len(output_rows)} cells "
        f"({n_pass} OK / {fail_count} FAIL) in {elapsed:.1f}s "
        f"-> {csv_p}, {md_p}"
    )
    return fail_count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    sys.exit(0 if main() == 0 else 1)
