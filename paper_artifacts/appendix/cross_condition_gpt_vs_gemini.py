"""Reproduce paper GPT-5.4 vs Gemini Cross-Condition — GPT-5.4 vs Gemini cross-condition (seed 1).

Source: ``Appendix robustness section`` (lines 48-67).

4 rows × 3 columns (GPT, Gemini, Δ) = 12 paper-lock cells.

Sources (seed 1 only — paper restricts GPT-5.4 vs Gemini Cross-Condition to s20260321):
  * LLM Direct (NL):
      GPT  -> ``bootstrap_ci_4seed.json.per_seed[20260321]
                [LLM-Direct].point_estimate``
      Gem  -> computed via FrozenBulkJSONSource('gemini_p2', s1, 'direct')
  * Schema Aware:
      GPT  -> ``...[Schema-Aware-Direct]``
      Gem  -> FrozenBulkJSONSource('gemini_p2', s1, 'schema-aware')
  * Struct Extracted:
      GPT  -> ``bootstrap_ci_all_methods_v2.json.per_seed[20260321]
                [Struct-Ext-Direct].point_estimate``
      Gem  -> FrozenBulkJSONSource pointing at
              ``data/benchmark/results/gemini_p2/extract-struct/`` (seed-agnostic
              dir; only seed 1 was processed for this artifact family).
  * Struct Oracle:
      GPT  -> ``...[Struct-Oracle-Direct]``
      Gem  -> ``...gemini_p2/oracle-struct/``
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

from survey2agent.evaluation.aggregator import aggregate_metrics
from survey2agent.evaluation.data_loaders import (
    build_training_records,
    load_atoms_for_seed,
    load_ground_truths,
    load_persona_difficulty_index,
    load_splits,
)
from survey2agent.evaluation.runner import run_method
from survey2agent.methods import FrozenBulkJSONSource, LLMDirect, LLMSchemaAware

from .._common import emit_row, write_outputs, _md_fmt_pct, PAPER_TOLERANCE
from .._appendix_helpers import (
    RESULTS_DATA_DIR,
    load_bootstrap_ci,
)


SEED1: str = "s20260321"
SEED1_KEY: str = "20260321"

# (row, col) -> paper value (fraction); col ∈ {gpt, gem, delta}
PAPER_TAB_CROSS_CONDITION_GPT_VS_GEMINI: dict[tuple[str, str], float] = {
    ("LLM Direct (NL)",  "gpt"):   0.689,
    ("LLM Direct (NL)",  "gem"):   0.681,
    ("LLM Direct (NL)",  "delta"): -0.008,
    ("Schema Aware",     "gpt"):   0.694,
    ("Schema Aware",     "gem"):   0.700,
    ("Schema Aware",     "delta"): 0.006,
    ("Struct Extracted", "gpt"):   0.733,
    ("Struct Extracted", "gem"):   0.755,
    ("Struct Extracted", "delta"): 0.022,
    ("Struct Oracle",    "gpt"):   0.742,
    ("Struct Oracle",    "gem"):   0.746,
    ("Struct Oracle",    "delta"): 0.004,
}

ROW_ORDER: list[str] = [
    "LLM Direct (NL)", "Schema Aware", "Struct Extracted", "Struct Oracle"
]
DELTA_TOLERANCE: float = 0.002


def _build_gemini_source(variant_dir: str) -> FrozenBulkJSONSource:
    """Construct a FrozenBulkJSONSource pointing at the seed-agnostic
    Gemini struct artifact directory (``data/benchmark/results/gemini_p2/
    {extract-struct,oracle-struct}/``).

    The struct-LLM Gemini run was executed once on seed 1 only (no per-seed
    subdir exists), so we only call this for ``seed='s20260321'`` evaluation.
    Override ``self.dir`` directly after construction; the standard
    ``base/model/seed/variant`` template does not fit this layout.
    """
    src = FrozenBulkJSONSource(model="gemini_p2", seed=SEED1, variant=variant_dir)
    src.dir = RESULTS_DATA_DIR / "gemini_p2" / variant_dir
    return src


def _compute_macro(method, evalr) -> float:
    return aggregate_metrics(run_method(method, evalr))["forced_accuracy"]


def main() -> int:
    t0 = time.time()
    boot = load_bootstrap_ci(all_methods_v2=True)
    ps1 = boot["per_seed"][SEED1_KEY]

    print("[cross_condition_gpt_vs_gemini] loading atoms for s1...", flush=True)
    splits = load_splits()
    diff_idx = load_persona_difficulty_index()
    gt = load_ground_truths(SEED1)
    llm_atoms = load_atoms_for_seed(SEED1, mode="llm")
    evalr = build_training_records(llm_atoms, gt, splits["test"], difficulty_index=diff_idx)

    point_by_cell: dict[tuple[str, str], float] = {}

    # GPT cells from bootstrap pooled JSON
    point_by_cell[("LLM Direct (NL)",  "gpt")] = ps1["LLM-Direct"]["point_estimate"]
    point_by_cell[("Schema Aware",     "gpt")] = ps1["Schema-Aware-Direct"]["point_estimate"]
    point_by_cell[("Struct Extracted", "gpt")] = ps1["Struct-Ext-Direct"]["point_estimate"]
    point_by_cell[("Struct Oracle",    "gpt")] = ps1["Struct-Oracle-Direct"]["point_estimate"]

    # Gemini Direct/Schema cells via runner on bulk artifact
    print("[cross_condition_gpt_vs_gemini] computing Gemini Direct...", flush=True)
    point_by_cell[("LLM Direct (NL)", "gem")] = _compute_macro(
        LLMDirect(
            source=FrozenBulkJSONSource(model="gemini_p2", seed=SEED1, variant="direct"),
            model_display_name="Gem-Direct",
        ),
        evalr,
    )
    print("[cross_condition_gpt_vs_gemini] computing Gemini Schema...", flush=True)
    point_by_cell[("Schema Aware", "gem")] = _compute_macro(
        LLMSchemaAware(
            source=FrozenBulkJSONSource(model="gemini_p2", seed=SEED1, variant="schema-aware"),
            model_display_name="Gem-Schema",
        ),
        evalr,
    )
    print("[cross_condition_gpt_vs_gemini] computing Gemini Struct Extracted...", flush=True)
    point_by_cell[("Struct Extracted", "gem")] = _compute_macro(
        LLMDirect(
            source=_build_gemini_source("extract-struct"),
            model_display_name="Gem-Struct-Ext",
        ),
        evalr,
    )
    print("[cross_condition_gpt_vs_gemini] computing Gemini Struct Oracle...", flush=True)
    point_by_cell[("Struct Oracle", "gem")] = _compute_macro(
        LLMDirect(
            source=_build_gemini_source("oracle-struct"),
            model_display_name="Gem-Struct-Oracle",
        ),
        evalr,
    )

    # Δ cells
    for row in ROW_ORDER:
        gpt = point_by_cell[(row, "gpt")]
        gem = point_by_cell[(row, "gem")]
        point_by_cell[(row, "delta")] = gem - gpt

    output_rows: list[dict] = []
    fail_count = skip_count = 0
    for row in ROW_ORDER:
        for col in ("gpt", "gem", "delta"):
            paper_v = PAPER_TAB_CROSS_CONDITION_GPT_VS_GEMINI.get((row, col))
            if paper_v is None:
                skip_count += 1; continue
            tol = DELTA_TOLERANCE if col == "delta" else PAPER_TOLERANCE
            r = emit_row(
                row_id=f"{row.replace(' ', '_')}__{col}",
                method_label=f"{row} :: {col}",
                mode="ext", metric="macro_accuracy" if col != "delta" else "macro_accuracy_delta",
                point=point_by_cell.get((row, col)),
                paper_point=paper_v,
                tolerance=tol,
            )
            output_rows.append(r)
            if r["paper_match"].startswith("FAIL"):
                fail_count += 1

    md_lines = [
        "| Information Condition | GPT-5.4 (%) | Gemini (%) | Δ (pp) |",
        "|:---|:---:|:---:|:---:|",
    ]
    for row in ROW_ORDER:
        gpt = point_by_cell[(row, "gpt")]
        gem = point_by_cell[(row, "gem")]
        delta = point_by_cell[(row, "delta")]
        md_lines.append(
            f"| {row} | {100*gpt:.1f} | {100*gem:.1f} | {100*delta:+.1f} |"
        )
    md_table = "\n".join(md_lines)

    csv_p, md_p = write_outputs(
        "cross_condition_gpt_vs_gemini",
        output_rows,
        md_table=md_table,
        script_name="paper_artifacts.appendix.cross_condition_gpt_vs_gemini",
        md_caption=(
            "**GPT-5.4 vs Gemini Cross-Condition.** GPT-5.4 vs. Gemini 3.1 Pro (seed 1, all four "
            "information conditions). Δ = Gemini − GPT in pp."
        ),
        md_footnotes=[
            "*Paper location: `Appendix robustness section` (`tab:F3`).*",
            f"*Tolerance: ±{PAPER_TOLERANCE} on point cells; ±{DELTA_TOLERANCE} on Δ cells.*",
        ],
        subdir="appendix",
    )
    elapsed = time.time() - t0
    n_pass = len(output_rows) - fail_count
    print(
        f"[cross_condition_gpt_vs_gemini] OK={n_pass} FAIL={fail_count} SKIP={skip_count} "
        f"total={len(output_rows)} ({elapsed:.1f}s) -> {csv_p}, {md_p}"
    )
    return fail_count


if __name__ == "__main__":
    sys.exit(0 if main() == 0 else 1)
