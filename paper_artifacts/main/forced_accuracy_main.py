"""Reproduce paper Forced-Accuracy Main Table — Macro accuracy (answer-only mode, 4-seed pooled).

Source: ``Experiments section`` lines 7-32 (``\\label{tab:4}``).

Reduction: pool-then-bootstrap (``aggregate_pooled_with_ci``) with
``metric="forced_accuracy"``, persona-cluster, ``n_bootstrap=2000``,
``seed=42``, 95% percentile CI.

Panel A rows (T0 / T1 / T2 / reference):

  - T0: Random, Majority Class
  - T1: SSB (extracted μ + direct-readout μ*)
  - T2: MV / ArgRAG-style adaptation, BCF, NBF, DSNBF, ABF (extracted + direct-readout μ* each)
  - Reference: Source Reachability (GT-aided direct-readout probe)

Panel B (T3 LLM rows): GPT-5.4 / Gemini 3.1 / DeepSeek V3.2 / Qwen3 235B,
each under Direct + Schema-Aware variants (forced_accuracy on extracted μ).

ArgRAG-style adaptation = Majority Vote under the equal-weight,
5-source closed-class atom representation used here (paper §3). Forced-Accuracy Main Table lists a single shared row;
no full document-level ArgRAG method class is wired.

Note: T0 ``Random`` is constructed with ``seed=42`` so the point estimate
is deterministic across reruns. The paper's Random row (30.1%) was also
seeded; the ±0.005 tolerance allows for a different seed value.
"""

from __future__ import annotations

import argparse
import sys
import time

from survey2agent.evaluation.data_loaders import build_training_records
from survey2agent.evaluation.multi_seed import aggregate_pooled_with_ci
from survey2agent.evaluation.runner import EvaluationResult, run_method
from survey2agent.methods import (
    ABF,
    BCF,
    DSNBF,
    LLMDirect,
    LLMSchemaAware,
    MajorityClass,
    MajorityVote,
    NBF,
    OracleExtraction,
    Random,
    SSB,
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
    _md_fmt_ci,
)


# ── Paper Forced-Accuracy Main Table numbers (s4_experiments.tex L13-32) ────────────────────


# Panel A: tier, method_label, ext (acc, lo, hi) | direct-readout μ* (acc, lo, hi)
PANEL_A_PAPER: list[dict] = [
    {"tier": "T0", "method_label": "Random",
     "ext": (0.301, 0.291, 0.310), "oracle": None,
     "row_id_ext": "t0_random", "row_id_oracle": None},
    {"tier": "T0", "method_label": "Majority Class",
     "ext": (0.571, 0.559, 0.584), "oracle": None,
     "row_id_ext": "t0_majclass", "row_id_oracle": None},
    {"tier": "T1", "method_label": "SSB",
     "ext": (0.773, 0.764, 0.781), "oracle": (0.790, 0.782, 0.798),
     "row_id_ext": "t1_ssb", "row_id_oracle": "t1_ssb_oracle"},
    {"tier": "T2", "method_label": "Majority Vote / ArgRAG-style",
     "ext": (0.688, 0.676, 0.699), "oracle": (0.695, 0.684, 0.705),
     "row_id_ext": "t2_mv_argrag", "row_id_oracle": "t2_mv_argrag_oracle"},
    {"tier": "T2", "method_label": "BCF",
     "ext": (0.692, 0.680, 0.703), "oracle": (0.698, 0.688, 0.707),
     "row_id_ext": "t2_bcf", "row_id_oracle": "t2_bcf_oracle"},
    {"tier": "T2", "method_label": "NBF",
     "ext": (0.798, 0.789, 0.806), "oracle": (0.821, 0.812, 0.829),
     "row_id_ext": "t2_nbf", "row_id_oracle": "t2_nbf_oracle"},
    {"tier": "T2", "method_label": "DSNBF",
     "ext": (0.803, 0.795, 0.812), "oracle": (0.823, 0.816, 0.831),
     "row_id_ext": "t2_dsnbf", "row_id_oracle": "t2_dsnbf_oracle"},
    {"tier": "T2", "method_label": "ABF",
     "ext": (0.720, 0.709, 0.729), "oracle": (0.720, 0.711, 0.729),
     "row_id_ext": "t2_abf", "row_id_oracle": "t2_abf_oracle"},
    {"tier": "Ref.", "method_label": "Source Reachability",
     "ext": None, "oracle": (0.932, None, None),
     "row_id_ext": None, "row_id_oracle": "t4_oracle_ext"},
]


PANEL_B_PAPER: list[dict] = [
    {"row_id": "t3_gpt_direct",      "family": "GPT-5.4",        "variant": "Direct",
     "model": "gpt-5.4",            "ext_variant": "direct",
     "paper": (0.686, None, None),  "cls": LLMDirect},
    {"row_id": "t3_gpt_schema",      "family": "GPT-5.4",        "variant": "Schema",
     "model": "gpt-5.4",            "ext_variant": "schema-aware",
     "paper": (0.697, None, None),  "cls": LLMSchemaAware},
    {"row_id": "t3_gemini_direct",   "family": "Gemini 3.1 Pro", "variant": "Direct",
     "model": "gemini_p2",          "ext_variant": "direct",
     "paper": (0.678, None, None),  "cls": LLMDirect},
    {"row_id": "t3_gemini_schema",   "family": "Gemini 3.1 Pro", "variant": "Schema",
     "model": "gemini_p2",          "ext_variant": "schema-aware",
     "paper": (0.700, None, None),  "cls": LLMSchemaAware},
    {"row_id": "t3_deepseek_direct", "family": "DeepSeek V3.2",  "variant": "Direct",
     "model": "deepseek-v3.2",      "ext_variant": "direct",
     "paper": (0.561, None, None),  "cls": LLMDirect},
    {"row_id": "t3_deepseek_schema", "family": "DeepSeek V3.2",  "variant": "Schema",
     "model": "deepseek-v3.2",      "ext_variant": "schema-aware",
     "paper": (0.569, None, None),  "cls": LLMSchemaAware},
    {"row_id": "t3_qwen3_direct",    "family": "Qwen3 235B",     "variant": "Direct",
     "model": "qwen3-235b-a22b-2507","ext_variant": "direct",
     "paper": (0.421, None, None),  "cls": LLMDirect},
    {"row_id": "t3_qwen3_schema",    "family": "Qwen3 235B",     "variant": "Schema",
     "model": "qwen3-235b-a22b-2507","ext_variant": "schema-aware",
     "paper": (0.480, None, None),  "cls": LLMSchemaAware},
]


_PANEL_A_FACTORIES = {
    "Random":                  lambda: Random(seed=42),
    "Majority Class":          lambda: MajorityClass(),
    "SSB":                     lambda: SSB(),
    "Majority Vote / ArgRAG-style":  lambda: MajorityVote(),
    "BCF":                     lambda: BCF(),
    "NBF":                     lambda: NBF(),
    "DSNBF":                   lambda: DSNBF(),
    "ABF":                     lambda: ABF(),
}


def _run_oracle_ext_across_seeds(*, oracle_atoms, gts, splits, diff_idx):
    out: dict[str, list[EvaluationResult]] = {}
    for seed in CANONICAL_SEEDS:
        atoms = oracle_atoms[seed]
        gt = gts[seed]
        evalr = build_training_records(atoms, gt, splits["test"], difficulty_index=diff_idx)
        method = OracleExtraction(skip_on_miss=False)
        method.attach_gt({pid: gt[pid] for pid in splits["test"] if pid in gt})
        out[seed] = run_method(method, evalr)
    return out


def _ci_dict(seed_results) -> dict:
    return aggregate_pooled_with_ci(
        seed_results,
        metric="forced_accuracy",
        n_bootstrap=2000,
        confidence=0.95,
        seed=42,
    )


def _row_from_ci(*, row_id, method_label, mode, ci, paper_acc, paper_lo, paper_hi):
    return emit_row(
        row_id=row_id,
        method_label=method_label,
        mode=mode,
        metric="forced_accuracy",
        point=ci["point"],
        ci_low=ci["ci_low"],
        ci_high=ci["ci_high"],
        n_seeds=ci.get("n_seeds", len(CANONICAL_SEEDS)),
        n_personas=ci.get("n_personas_pooled", 0),
        paper_point=paper_acc,
        paper_low=paper_lo,
        paper_high=paper_hi,
    )


def _render_panel_a_md(rows_by_id: dict[str, dict]) -> str:
    lines = [
        "*Panel A: Fusion and source-selection methods (T0–T2) plus reference*",
        "",
        "|     | Method                  | Acc (%) | 95% CI       | Acc on μ* (%) | 95% CI       |",
        "|:----|:------------------------|:-------:|:------------:|:----------------:|:------------:|",
    ]
    last_tier = ""
    for spec in PANEL_A_PAPER:
        tier = spec["tier"]
        tier_cell = f"**{tier}**" if tier != last_tier else ""
        last_tier = tier

        def cell(row_id):
            if row_id is None:
                return "—", ""
            r = rows_by_id.get(row_id)
            if r is None:
                return "—", ""
            return _md_fmt_pct(r["point"]), _md_fmt_ci(r["ci_low"], r["ci_high"])

        ext_acc, ext_ci = cell(spec["row_id_ext"])
        oracle_acc, oracle_ci = cell(spec["row_id_oracle"])
        # Bold DSNBF row best cells per paper formatting
        if spec["method_label"] == "DSNBF":
            ext_acc = f"**{ext_acc}**"
            oracle_acc = f"**{oracle_acc}**"
        lines.append(
            f"| {tier_cell} | {spec['method_label']} | {ext_acc} | {ext_ci} | {oracle_acc} | {oracle_ci} |"
        )
    return "\n".join(lines)


def _render_panel_b_md(rows_by_id: dict[str, dict]) -> str:
    lines = [
        "*Panel B: LLM baselines (T3) — per family, two prompting regimes*",
        "",
        "|     | LLM Family       | Direct Acc (%) | Schema Acc (%) | Δ (pp) |",
        "|:----|:-----------------|:--------------:|:--------------:|:------:|",
    ]
    by_family: dict[str, dict[str, dict]] = {}
    for spec in PANEL_B_PAPER:
        by_family.setdefault(spec["family"], {})[spec["variant"]] = (spec, rows_by_id.get(spec["row_id"]))
    for family, vmap in by_family.items():
        d_spec, d_row = vmap.get("Direct", (None, None))
        s_spec, s_row = vmap.get("Schema", (None, None))
        d_acc = _md_fmt_pct(d_row["point"]) if d_row else "—"
        s_acc = _md_fmt_pct(s_row["point"]) if s_row else "—"
        if d_row and s_row:
            delta = 100.0 * (s_row["point"] - d_row["point"])
            delta_str = f"{delta:+.1f}"
        else:
            delta_str = ""
        lines.append(
            f"| **T3** | {family} | {d_acc} | {s_acc} | {delta_str} |"
        )
    return "\n".join(lines)


def main(rows: list[str] | None = None) -> int:
    """Reproduce Forced-Accuracy Main Table and write CSV + MD outputs."""
    t0 = time.time()
    splits, diff_idx, gts, oracle_atoms, llm_atoms = load_shared_resources()

    output_rows: list[dict] = []
    fail_count = 0

    def _allow(row_id) -> bool:
        return rows is None or row_id in rows

    # ── Panel A ───────────────────────────────────────────────────────
    for spec in PANEL_A_PAPER:
        label = spec["method_label"]
        if spec["row_id_oracle"] == "t4_oracle_ext":
            if _allow(spec["row_id_oracle"]):
                sr = _run_oracle_ext_across_seeds(
                    oracle_atoms=oracle_atoms, gts=gts, splits=splits, diff_idx=diff_idx,
                )
                ci = _ci_dict(sr)
                paper_acc, paper_lo, paper_hi = spec["oracle"]
                row = _row_from_ci(
                    row_id=spec["row_id_oracle"], method_label=label, mode="oracle",
                    ci=ci, paper_acc=paper_acc, paper_lo=paper_lo, paper_hi=paper_hi,
                )
                output_rows.append(row)
                if row["paper_match"].startswith("FAIL"):
                    fail_count += 1
            continue

        factory = _PANEL_A_FACTORIES[label]

        if spec["ext"] is not None and spec["row_id_ext"] is not None and _allow(spec["row_id_ext"]):
            sr = run_mixed_mode_across_seeds(
                factory,
                oracle_atoms=oracle_atoms, llm_atoms=llm_atoms,
                gts=gts, splits=splits, diff_idx=diff_idx,
            )
            ci = _ci_dict(sr)
            paper_acc, paper_lo, paper_hi = spec["ext"]
            row = _row_from_ci(
                row_id=spec["row_id_ext"], method_label=label, mode="ext",
                ci=ci, paper_acc=paper_acc, paper_lo=paper_lo, paper_hi=paper_hi,
            )
            output_rows.append(row)
            if row["paper_match"].startswith("FAIL"):
                fail_count += 1

        if spec["oracle"] is not None and spec["row_id_oracle"] is not None and _allow(spec["row_id_oracle"]):
            sr = run_oracle_mode_across_seeds(
                factory,
                oracle_atoms=oracle_atoms, gts=gts, splits=splits, diff_idx=diff_idx,
            )
            ci = _ci_dict(sr)
            paper_acc, paper_lo, paper_hi = spec["oracle"]
            row = _row_from_ci(
                row_id=spec["row_id_oracle"], method_label=label, mode="oracle",
                ci=ci, paper_acc=paper_acc, paper_lo=paper_lo, paper_hi=paper_hi,
            )
            output_rows.append(row)
            if row["paper_match"].startswith("FAIL"):
                fail_count += 1

    # ── Panel B (LLM rows) ────────────────────────────────────────────
    for spec in PANEL_B_PAPER:
        if not _allow(spec["row_id"]):
            continue
        sr = run_llm_across_seeds(
            model=spec["model"], variant=spec["ext_variant"],
            display_name=spec["family"], selective_cls=spec["cls"],
            llm_atoms=llm_atoms, gts=gts, splits=splits, diff_idx=diff_idx,
        )
        ci = _ci_dict(sr)
        paper_acc, paper_lo, paper_hi = spec["paper"]
        row = _row_from_ci(
            row_id=spec["row_id"],
            method_label=f"{spec['family']} {spec['variant']}",
            mode="ext",
            ci=ci, paper_acc=paper_acc, paper_lo=paper_lo, paper_hi=paper_hi,
        )
        output_rows.append(row)
        if row["paper_match"].startswith("FAIL"):
            fail_count += 1

    # ── Render Markdown ──────────────────────────────────────────────
    rows_by_id = {r["row_id"]: r for r in output_rows}
    panel_a = _render_panel_a_md(rows_by_id)
    panel_b = _render_panel_b_md(rows_by_id)
    md_table = panel_a + "\n\n" + panel_b

    caption = (
        "**Forced-Accuracy Main Table.** Macro accuracy (%, answer-only mode), 4-seed pooled, "
        "8,640 question instances (480 seed-persona test clusters × 18 questions). "
        "T0/T1/T2 use their canonical input; the last column uses direct-readout μ*. "
        "T3: each LLM family is evaluated under two prompting regimes (Direct and Schema-Aware), "
        "reported in separate columns with the schema gain (Δ). Reduction: pool-then-bootstrap, "
        "persona-clustered, 95% percentile CI from B=2,000 resamples (seed=42). DSNBF cells in **bold**."
    )
    footnotes = [
        f"*Paper location: `Experiments section` (`tab:4`).*",
        "*MV / ArgRAG-style: this is the paper's equal-weight closed-class "
        "ArgRAG adaptation; it is identical to Majority Vote under the "
        "5-source atom representation (§3), so a single shared row "
        "reuses the MajorityVote computation.*",
        f"*Reduction: `aggregate_pooled_with_ci` (pool-then-bootstrap, persona-clustered, "
        f"`n_bootstrap=2000`, `seed=42`, 95% percentile CI).*",
        f"*Reproduction tolerance: ±{PAPER_TOLERANCE} absolute (±0.5 pp). "
        "Random uses `seed=42` for determinism.*",
    ]

    csv_p, md_p = write_outputs(
        "forced_accuracy",
        output_rows,
        md_table=md_table,
        script_name="paper_artifacts.main.forced_accuracy_main",
        md_caption=caption,
        md_footnotes=footnotes,
        subdir="main",
    )
    elapsed = time.time() - t0
    n_pass = len(output_rows) - fail_count
    print(
        f"[forced_accuracy_main] wrote {len(output_rows)} cells "
        f"({n_pass} OK / {fail_count} FAIL) in {elapsed:.1f}s "
        f"-> {csv_p}, {md_p}"
    )
    return fail_count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", default="", help="comma-separated row_id filter")
    args = parser.parse_args()
    row_filter = [r.strip() for r in args.rows.split(",") if r.strip()] or None
    sys.exit(0 if main(rows=row_filter) == 0 else 1)
