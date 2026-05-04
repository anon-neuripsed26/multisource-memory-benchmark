r"""Reproduce paper 2x2 Factorial Decomposition — 2×2 factorial decomposition (Resolver × Input).

Source: ``Experiments section`` lines 45-58
(``\\label{tab:5}``).

Cells (95% pool-then-bootstrap CI, persona-clustered, B=2000, seed=42):

  ┌───────────────┬───────────────────────────┬───────────────────────────┐
  │ Input \ Resol │ Fusion (DSNBF-NoSkip)     │ LLM (GPT-μ*)              │
  ├───────────────┼───────────────────────────┼───────────────────────────┤
  │ μ* (direct)   │ 82.3 [81.6, 83.1]   (a)   │ 73.5 [72.6, 74.4]   (b)   │
  │ Extracted μ̂  │ 80.3 [79.5, 81.2]   (c)   │ 71.4 [70.4, 72.3]   (d)   │
  └───────────────┴───────────────────────────┴───────────────────────────┘

Decomposition (printed as MD footnote):
  - total gap a−d
  - resolver effect ((a−b) + (c−d)) / 2
  - input effect ((a−c) + (b−d)) / 2
  - interaction (a − b − c + d)

Resolution map:
  - **Fusion** = ``DSNBF()`` (forced, ignores ``would_skip``)
  - **LLM** = ``StructLLMSource`` (oracle/extracted) wrapped by ``LLMDirect``
"""

from __future__ import annotations

import argparse
import sys
import time

from survey2agent.evaluation.multi_seed import aggregate_pooled_with_ci
from survey2agent.methods import DSNBF, LLMDirect

from .._common import (
    PAPER_TOLERANCE,
    emit_row,
    load_shared_resources,
    run_mixed_mode_across_seeds,
    run_oracle_mode_across_seeds,
    run_struct_llm_across_seeds,
    write_outputs,
    _md_fmt_pct,
    _md_fmt_ci,
)


# Paper 2x2 Factorial Decomposition numbers
PAPER_CELLS = {
    "fusion_oracle":    (0.823, 0.816, 0.831),
    "llm_oracle":       (0.735, 0.726, 0.744),
    "fusion_extracted": (0.803, 0.795, 0.812),
    "llm_extracted":    (0.714, 0.704, 0.723),
}


def _ci(seed_results) -> dict:
    return aggregate_pooled_with_ci(
        seed_results,
        metric="forced_accuracy",
        n_bootstrap=2000,
        confidence=0.95,
        seed=42,
    )


def main() -> int:
    t0 = time.time()
    splits, diff_idx, gts, oracle_atoms, llm_atoms = load_shared_resources()

    output_rows: list[dict] = []
    points: dict[str, float] = {}
    fail_count = 0

    # (a) fusion_oracle ─────────────────────────────────────────────
    sr = run_oracle_mode_across_seeds(
        DSNBF, oracle_atoms=oracle_atoms, gts=gts, splits=splits, diff_idx=diff_idx,
    )
    ci = _ci(sr)
    p_acc, p_lo, p_hi = PAPER_CELLS["fusion_oracle"]
    row = emit_row(
        row_id="fusion_oracle", method_label="DSNBF (Fusion)", mode="oracle",
        metric="forced_accuracy",
        point=ci["point"], ci_low=ci["ci_low"], ci_high=ci["ci_high"],
        n_personas=ci["n_personas_pooled"],
        paper_point=p_acc, paper_low=p_lo, paper_high=p_hi,
    )
    output_rows.append(row); points["a"] = ci["point"]
    if row["paper_match"].startswith("FAIL"):
        fail_count += 1

    # (b) llm_oracle ────────────────────────────────────────────────
    sr = run_struct_llm_across_seeds(
        mode="oracle", display_name="GPT-μ* (direct-readout)",
        method_cls=LLMDirect, eval_atoms=oracle_atoms,
        gts=gts, splits=splits, diff_idx=diff_idx,
    )
    ci = _ci(sr)
    p_acc, p_lo, p_hi = PAPER_CELLS["llm_oracle"]
    row = emit_row(
        row_id="llm_oracle", method_label="GPT-μ* (direct-readout)", mode="oracle",
        metric="forced_accuracy",
        point=ci["point"], ci_low=ci["ci_low"], ci_high=ci["ci_high"],
        n_personas=ci["n_personas_pooled"],
        paper_point=p_acc, paper_low=p_lo, paper_high=p_hi,
    )
    output_rows.append(row); points["b"] = ci["point"]
    if row["paper_match"].startswith("FAIL"):
        fail_count += 1

    # (c) fusion_extracted ──────────────────────────────────────────
    sr = run_mixed_mode_across_seeds(
        DSNBF, oracle_atoms=oracle_atoms, llm_atoms=llm_atoms,
        gts=gts, splits=splits, diff_idx=diff_idx,
    )
    ci = _ci(sr)
    p_acc, p_lo, p_hi = PAPER_CELLS["fusion_extracted"]
    row = emit_row(
        row_id="fusion_extracted", method_label="DSNBF (Fusion)", mode="ext",
        metric="forced_accuracy",
        point=ci["point"], ci_low=ci["ci_low"], ci_high=ci["ci_high"],
        n_personas=ci["n_personas_pooled"],
        paper_point=p_acc, paper_low=p_lo, paper_high=p_hi,
    )
    output_rows.append(row); points["c"] = ci["point"]
    if row["paper_match"].startswith("FAIL"):
        fail_count += 1

    # (d) llm_extracted ─────────────────────────────────────────────
    sr = run_struct_llm_across_seeds(
        mode="extracted", display_name="GPT-μ* (extracted)",
        method_cls=LLMDirect, eval_atoms=llm_atoms,
        gts=gts, splits=splits, diff_idx=diff_idx,
    )
    ci = _ci(sr)
    p_acc, p_lo, p_hi = PAPER_CELLS["llm_extracted"]
    row = emit_row(
        row_id="llm_extracted", method_label="GPT-μ* (extracted)", mode="ext",
        metric="forced_accuracy",
        point=ci["point"], ci_low=ci["ci_low"], ci_high=ci["ci_high"],
        n_personas=ci["n_personas_pooled"],
        paper_point=p_acc, paper_low=p_lo, paper_high=p_hi,
    )
    output_rows.append(row); points["d"] = ci["point"]
    if row["paper_match"].startswith("FAIL"):
        fail_count += 1

    # ── Decomposition (% point) ─────────────────────────────────────
    a, b, c, d = points["a"], points["b"], points["c"], points["d"]
    total_gap = (a - d) * 100
    resolver_effect = ((a - b) + (c - d)) / 2 * 100
    input_effect = ((a - c) + (b - d)) / 2 * 100
    interaction = (a - b - c + d) * 100

    # Paper checks (within ±0.5 pp)
    paper_decomp = {
        "total_gap":       (10.9, total_gap),
        "resolver_effect": (8.85, resolver_effect),
        "input_effect":    (2.05, input_effect),
        "interaction":     (-0.1, interaction),
    }

    # ── Render Markdown ─────────────────────────────────────────────
    rows_by_id = {r["row_id"]: r for r in output_rows}

    def fmt_cell(rid):
        r = rows_by_id[rid]
        return f"{_md_fmt_pct(r['point'])} {_md_fmt_ci(r['ci_low'], r['ci_high'])}"

    md_table = (
        "|                  | Fusion (DSNBF-NoSkip)            | LLM (GPT-μ*)                     |\n"
        "|:-----------------|:--------------------------------:|:--------------------------------:|\n"
        f"| **μ\\* (direct)** | {fmt_cell('fusion_oracle')}      | {fmt_cell('llm_oracle')}         |\n"
        f"| **Extracted μ̂** | {fmt_cell('fusion_extracted')}   | {fmt_cell('llm_extracted')}      |\n"
    )

    decomp_lines = [
        "",
        "**Decomposition** (paper §4.2; values in pp):",
        "",
        "| Effect            | Reproduced | Paper   | |Δ| pp |",
        "|:------------------|:----------:|:-------:|:------:|",
    ]
    decomp_fail = 0
    for name, (paper_v, repro_v) in paper_decomp.items():
        delta = abs(repro_v - paper_v)
        marker = "" if delta <= 0.5 else " ⚠"
        if delta > 0.5:
            decomp_fail += 1
        decomp_lines.append(
            f"| {name.replace('_', ' ')} | {repro_v:+.2f} | {paper_v:+.2f} | {delta:.2f}{marker} |"
        )
    md_table = md_table + "\n" + "\n".join(decomp_lines)

    caption = (
        "**2x2 Factorial Decomposition.** 2×2 factorial decomposition (Resolver × Input). "
        "Fusion = DSNBF-NoSkip (answer-only), LLM = GPT-5.4 reading structured μ via "
        "`StructLLMSource` (answer-only). 4-seed pool, persona-clustered 95% "
        "bootstrap CI from B=2,000 resamples (seed=42)."
    )
    footnotes = [
        "*Paper location: `Experiments section` (`tab:5`).*",
        "*Reduction: `aggregate_pooled_with_ci(metric=\"forced_accuracy\", n_bootstrap=2000, seed=42)`.*",
        f"*Reproduction tolerance: ±{PAPER_TOLERANCE} absolute on cell points (±0.5 pp on decomposition values).*",
    ]

    csv_p, md_p = write_outputs(
        "factorial",
        output_rows,
        md_table=md_table,
        script_name="paper_artifacts.main.factorial_decomposition",
        md_caption=caption,
        md_footnotes=footnotes,
        subdir="main",
    )
    elapsed = time.time() - t0
    n_pass = len(output_rows) - fail_count
    print(
        f"[factorial_decomposition] wrote {len(output_rows)} cells "
        f"+ 4 decomposition values "
        f"({n_pass}/4 cells OK, {4 - decomp_fail}/4 decomp OK) in {elapsed:.1f}s "
        f"-> {csv_p}, {md_p}"
    )
    return fail_count + decomp_fail


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    sys.exit(0 if main() == 0 else 1)
