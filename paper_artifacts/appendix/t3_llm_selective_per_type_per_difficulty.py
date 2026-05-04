"""Reproduce paper Difficulty-Class Breakdown (T3 +SKIP) — T3 LLM +SKIP per-type × per-difficulty (NL input).

Source: ``Appendix diagnostics section`` lines 162-208
(``\\label{tab:E2b-skip}``).

Per-type × per-difficulty-class **selective accuracy** (%, 4-seed mean,
NL input) with paired coverage (%) per cell. 8 LLM cells × 24 (type,
diff) cells = 192 sel_acc + 192 cov + 8 footer drops = 392 paper-lock
cells. Footer reports overall (stable → svr) drop in selective accuracy.

Method spec: ``LLMDirectSelective`` and ``LLMSchemaAwareSelective`` for
each of 4 LLM families.

Per the paper text (appendix_e_diagnostics.tex L131): "DeepSeek and Qwen3
essentially never abstain (coverage ≈ 100%)", so their selective
accuracy approximately equals answer-only accuracy from Difficulty-Class Breakdown (T3 LLM). GPT
and Gemini abstain selectively (coverage 85-100%).

Reduction: pool across 4 seeds (``pool_results_across_seeds``), group
by ``(reasoning_type, difficulty_class)``, then **micro selective
accuracy** per cell. Footer drop = ``stable_overall - svr_overall`` of
pooled-within-diff micro selective accuracy. Paper sign convention:
positive = "drops X pp from stable to svr" (Schema overcorrection on
GPT/Gemini gives +13.5/+13.1; DS/QW improve on svr, giving negative
drop values).

This is the same reduction as Difficulty-Class Breakdown (T2 +SKIP); per-seed-then-mean does
*not* reproduce these cells because coverage varies across seeds.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict

from survey2agent.evaluation.aggregator import aggregate_metrics
from survey2agent.evaluation.multi_seed import pool_results_across_seeds
from survey2agent.methods import (
    LLMDirectSelective,
    LLMSchemaAwareSelective,
)

from .._common import (
    PAPER_TOLERANCE,
    emit_row,
    load_shared_resources,
    run_llm_across_seeds,
    write_outputs,
    _md_fmt_pct,
)


# ── LLM column spec (paper Difficulty-Class Breakdown (T3 +SKIP) column order) ─────────────────


# (col_label, model, variant, display, method_cls). All Selective.
LLM_COLS = [
    ("GPT-D+S",  "gpt-5.4",                "direct",       "GPT-5.4 Direct (Selective)",       LLMDirectSelective),
    ("GPT-S+S",  "gpt-5.4",                "schema-aware", "GPT-5.4 Schema (Selective)",       LLMSchemaAwareSelective),
    ("Gem-D+S",  "gemini_p2",              "direct",       "Gemini 3.1 Direct (Selective)",    LLMDirectSelective),
    ("Gem-S+S",  "gemini_p2",              "schema-aware", "Gemini 3.1 Schema (Selective)",    LLMSchemaAwareSelective),
    ("DS-D+S",   "deepseek-v3.2",          "direct",       "DeepSeek V3.2 Direct (Selective)", LLMDirectSelective),
    ("DS-S+S",   "deepseek-v3.2",          "schema-aware", "DeepSeek V3.2 Schema (Selective)", LLMSchemaAwareSelective),
    ("QW-D+S",   "qwen3-235b-a22b-2507",   "direct",       "Qwen3 235B Direct (Selective)",    LLMDirectSelective),
    ("QW-S+S",   "qwen3-235b-a22b-2507",   "schema-aware", "Qwen3 235B Schema (Selective)",    LLMSchemaAwareSelective),
]


# ── Paper Difficulty-Class Breakdown (T3 +SKIP) numbers (appendix_e_diagnostics.tex L172-208) ──────


# Row order: same as paper Difficulty-Class Breakdown (T3 LLM) (alphabetic-by-letter): A, B, C, D, E, F, G, Ctrl.
TYPE_ORDER = [
    ("identity",     "Ident"),
    ("arbitration",  "Arbit"),
    ("control",      "Ctrl"),
    ("causal",       "Factor"),
    ("trend",        "Temp"),
    ("plan_reality", "P-R"),
    ("missing_data", "Miss"),
    ("annotation",   "Annot"),
]

DIFFS = [
    ("stable",              "stable"),
    ("temporal_shift",      "ts"),
    ("stated_vs_revealed",  "svr"),
]

# (type_key, diff_key) -> tuple of (sel_acc, coverage) per LLM column.
# Order: GPT-D+S, GPT-S+S, Gem-D+S, Gem-S+S, DS-D+S, DS-S+S, QW-D+S, QW-S+S.
PAPER_TAB_T3_LLM_SELECTIVE_PER_TYPE_PER_DIFFICULTY: dict[tuple[str, str], tuple[tuple[float, float], ...]] = {
    ("identity", "stable"): (
        (0.869, 0.928), (0.978, 0.988), (0.668, 0.988), (0.822, 1.000),
        (0.856, 1.000), (0.741, 1.000), (0.503, 1.000), (0.459, 1.000),
    ),
    ("identity", "temporal_shift"): (
        (0.783, 0.922), (0.955, 0.972), (0.650, 0.991), (0.766, 1.000),
        (0.750, 1.000), (0.631, 1.000), (0.394, 1.000), (0.381, 1.000),
    ),
    ("identity", "stated_vs_revealed"): (
        (0.871, 0.875), (0.929, 0.962), (0.841, 1.000), (0.853, 1.000),
        (0.716, 1.000), (0.847, 1.000), (0.847, 1.000), (0.822, 1.000),
    ),
    ("arbitration", "stable"): (
        (0.773, 0.956), (0.783, 0.971), (0.749, 0.996), (0.793, 0.996),
        (0.665, 1.000), (0.662, 1.000), (0.558, 0.994), (0.605, 0.998),
    ),
    ("arbitration", "temporal_shift"): (
        (0.831, 0.940), (0.829, 0.962), (0.812, 0.996), (0.845, 0.996),
        (0.642, 1.000), (0.644, 1.000), (0.589, 0.988), (0.653, 0.979),
    ),
    ("arbitration", "stated_vs_revealed"): (
        (0.625, 0.933), (0.612, 0.956), (0.605, 0.988), (0.616, 0.992),
        (0.521, 1.000), (0.525, 1.000), (0.343, 0.954), (0.344, 0.969),
    ),
    ("control", "stable"): (
        (0.781, 0.928), (0.737, 0.988), (0.727, 0.997), (0.759, 1.000),
        (0.491, 1.000), (0.434, 1.000), (0.245, 0.997), (0.206, 1.000),
    ),
    ("control", "temporal_shift"): (
        (0.862, 0.884), (0.798, 0.959), (0.790, 0.997), (0.797, 1.000),
        (0.491, 1.000), (0.438, 1.000), (0.459, 1.000), (0.453, 1.000),
    ),
    ("control", "stated_vs_revealed"): (
        (0.829, 0.875), (0.782, 0.962), (0.766, 1.000), (0.775, 1.000),
        (0.728, 1.000), (0.678, 1.000), (0.753, 1.000), (0.759, 0.997),
    ),
    ("causal", "stable"): (
        (0.865, 0.831), (0.846, 0.875), (0.806, 0.966), (0.770, 0.991),
        (0.447, 1.000), (0.466, 1.000), (0.066, 0.994), (0.210, 0.997),
    ),
    ("causal", "temporal_shift"): (
        (0.869, 0.859), (0.852, 0.906), (0.788, 0.988), (0.830, 0.994),
        (0.488, 1.000), (0.534, 1.000), (0.212, 0.988), (0.268, 0.991),
    ),
    ("causal", "stated_vs_revealed"): (
        (0.518, 0.978), (0.533, 0.956), (0.516, 1.000), (0.514, 0.997),
        (0.738, 1.000), (0.791, 1.000), (0.035, 0.984), (0.129, 0.972),
    ),
    ("trend", "stable"): (
        (0.793, 0.953), (0.853, 0.975), (0.755, 0.997), (0.822, 1.000),
        (0.694, 1.000), (0.709, 1.000), (0.653, 1.000), (0.603, 1.000),
    ),
    ("trend", "temporal_shift"): (
        (0.820, 0.938), (0.833, 0.975), (0.772, 1.000), (0.822, 1.000),
        (0.544, 1.000), (0.775, 1.000), (0.628, 1.000), (0.491, 1.000),
    ),
    ("trend", "stated_vs_revealed"): (
        (0.609, 0.878), (0.384, 0.959), (0.601, 0.994), (0.400, 1.000),
        (0.606, 1.000), (0.509, 1.000), (0.288, 1.000), (0.550, 1.000),
    ),
    ("plan_reality", "stable"): (
        (0.728, 0.884), (0.735, 0.909), (0.738, 0.906), (0.725, 0.953),
        (0.466, 1.000), (0.478, 1.000), (0.614, 0.997), (0.628, 0.975),
    ),
    ("plan_reality", "temporal_shift"): (
        (0.801, 0.928), (0.809, 0.947), (0.800, 0.953), (0.798, 0.959),
        (0.469, 1.000), (0.506, 1.000), (0.519, 0.994), (0.508, 0.934),
    ),
    ("plan_reality", "stated_vs_revealed"): (
        (0.607, 0.859), (0.622, 0.859), (0.628, 0.866), (0.614, 0.922),
        (0.459, 1.000), (0.462, 1.000), (0.603, 0.959), (0.611, 0.941),
    ),
    ("missing_data", "stable"): (
        (0.638, 0.983), (0.700, 0.971), (0.624, 0.998), (0.674, 0.998),
        (0.408, 1.000), (0.444, 1.000), (0.258, 1.000), (0.482, 0.998),
    ),
    ("missing_data", "temporal_shift"): (
        (0.663, 0.990), (0.710, 0.983), (0.654, 1.000), (0.678, 0.998),
        (0.408, 1.000), (0.496, 1.000), (0.269, 0.998), (0.504, 0.983),
    ),
    ("missing_data", "stated_vs_revealed"): (
        (0.628, 0.979), (0.661, 0.996), (0.650, 1.000), (0.667, 1.000),
        (0.490, 1.000), (0.496, 1.000), (0.246, 1.000), (0.402, 0.985),
    ),
    ("annotation", "stable"): (
        # Tex L222: 35.5 (93.4) 34.3 (94.7) 58.5 (98.8) 56.4 (99.7) 58.1 (100.0) 47.8 (100.0) 37.8 (100.0) 49.1 (100.0)
        (0.355, 0.934), (0.343, 0.947), (0.585, 0.988), (0.564, 0.997),
        (0.581, 1.000), (0.478, 1.000), (0.378, 1.000), (0.491, 1.000),
    ),
    ("annotation", "temporal_shift"): (
        # Tex L223: 39.5 (90.9) 36.6 (93.1) 52.2 (100.0) 52.5 (100.0) 46.9 (100.0) 47.2 (100.0) 37.8 (100.0) 49.4 (100.0)
        (0.395, 0.909), (0.366, 0.931), (0.522, 1.000), (0.525, 1.000),
        (0.469, 1.000), (0.472, 1.000), (0.378, 1.000), (0.494, 1.000),
    ),
    ("annotation", "stated_vs_revealed"): (
        # Tex L224: 39.1 (87.8) 33.7 (92.8) 32.8 (99.1) 40.9 (100.0) 44.7 (100.0) 50.9 (100.0) 39.8 (99.7) 42.5 (100.0)
        (0.391, 0.878), (0.337, 0.928), (0.328, 0.991), (0.409, 1.000),
        (0.447, 1.000), (0.509, 1.000), (0.398, 0.997), (0.425, 1.000),
    ),
}

# Footer: Overall sel-acc drop (stable -> svr), stable_overall - svr_overall.
# Paper convention: positive = degradation, negative = improvement.
# GPT/Gem schema overcorrects on svr → large positive; DS/QW improve on svr → negative.
PAPER_FOOTER_DROP = {
    "GPT-D+S": +0.089,
    "GPT-S+S": +0.135,
    "Gem-D+S": +0.086,
    "Gem-S+S": +0.131,
    "DS-D+S":  -0.007,
    "DS-S+S":  -0.040,
    "QW-D+S":  -0.014,
    "QW-S+S":  -0.022,
}


def _pool_per_cell_and_overall(seed_results):
    pooled = pool_results_across_seeds(seed_results)
    by_cell: dict[tuple[str, str], list] = defaultdict(list)
    by_diff: dict[str, list] = defaultdict(list)
    for r in pooled:
        by_cell[(r.reasoning_type, r.difficulty_class)].append(r)
        by_diff[r.difficulty_class].append(r)
    per_cell = {k: aggregate_metrics(v) for k, v in by_cell.items()}
    per_diff = {k: aggregate_metrics(v) for k, v in by_diff.items()}
    return per_cell, per_diff


def main() -> int:
    t0 = time.time()
    splits, diff_idx, gts, oracle_atoms, llm_atoms = load_shared_resources()

    sel_by_col: dict[str, dict[tuple[str, str], float]] = {}
    cov_by_col: dict[str, dict[tuple[str, str], float]] = {}
    sel_overall_by_col: dict[str, dict[str, float]] = {}
    for col_label, model, variant, display, cls in LLM_COLS:
        print(f"[t3_llm_selective_per_type_per_difficulty] running {col_label} ({model} {variant})...", flush=True)
        sr = run_llm_across_seeds(
            model=model, variant=variant, display_name=display,
            selective_cls=cls, llm_atoms=llm_atoms, gts=gts,
            splits=splits, diff_idx=diff_idx,
        )
        per_cell, per_diff = _pool_per_cell_and_overall(sr)
        sel_by_col[col_label] = {k: v["selective_accuracy_micro"] for k, v in per_cell.items()}
        cov_by_col[col_label] = {k: v["coverage"] for k, v in per_cell.items()}
        sel_overall_by_col[col_label] = {k: v["selective_accuracy_micro"] for k, v in per_diff.items()}

    output_rows: list[dict] = []
    fail_count = 0

    def add_cell(*, type_key, type_label, diff_key, diff_label, col_label,
                 metric_name, point, paper_v):
        nonlocal fail_count
        row = emit_row(
            row_id=f"{type_key}__{diff_key}__{col_label}__{metric_name}",
            method_label=f"{col_label} :: {type_label} / {diff_label}",
            mode="ext", metric=metric_name,
            point=point, n_personas="",
            paper_point=paper_v,
        )
        output_rows.append(row)
        if row["paper_match"].startswith("FAIL"):
            fail_count += 1

    for type_key, type_label in TYPE_ORDER:
        for diff_key, diff_label in DIFFS:
            paper_pairs = PAPER_TAB_T3_LLM_SELECTIVE_PER_TYPE_PER_DIFFICULTY[(type_key, diff_key)]
            for (col_label, *_rest), (paper_sel, paper_cov) in zip(LLM_COLS, paper_pairs):
                add_cell(
                    type_key=type_key, type_label=type_label,
                    diff_key=diff_key, diff_label=diff_label,
                    col_label=col_label, metric_name="selective_accuracy",
                    point=sel_by_col[col_label].get((type_key, diff_key)),
                    paper_v=paper_sel,
                )
                add_cell(
                    type_key=type_key, type_label=type_label,
                    diff_key=diff_key, diff_label=diff_label,
                    col_label=col_label, metric_name="coverage",
                    point=cov_by_col[col_label].get((type_key, diff_key)),
                    paper_v=paper_cov,
                )

    # Footer: overall sel-acc drop (stable -> svr) per LLM cell, pool-micro.
    drop_actual: dict[str, float] = {}
    for col_label, *_rest in LLM_COLS:
        ov = sel_overall_by_col[col_label]
        drop = ov.get("stable", 0.0) - ov.get("stated_vs_revealed", 0.0)
        drop_actual[col_label] = drop
        paper_drop = PAPER_FOOTER_DROP[col_label]
        row = emit_row(
            row_id=f"_footer_drop__{col_label}",
            method_label=f"{col_label} :: overall sel-acc drop (stable->svr)",
            mode="ext", metric="selective_accuracy_drop",
            point=drop, n_personas="",
            paper_point=paper_drop,
        )
        output_rows.append(row)
        if row["paper_match"].startswith("FAIL"):
            fail_count += 1

    # ── Render Markdown ───────────────────────────────────────────
    by_id = {r["row_id"]: r for r in output_rows}
    cols = [c for c, *_ in LLM_COLS]
    md_lines = [
        "| Type    | Diff | " + " | ".join(cols) + " |",
        "|:--------|:-----|" + "|".join([":---:"] * len(cols)) + "|",
    ]
    for type_key, type_label in TYPE_ORDER:
        for diff_key, diff_label in DIFFS:
            cells_md: list[str] = []
            for c in cols:
                sel = by_id[f"{type_key}__{diff_key}__{c}__selective_accuracy"]["point"]
                cov = by_id[f"{type_key}__{diff_key}__{c}__coverage"]["point"]
                cells_md.append(f"{_md_fmt_pct(sel)} ({_md_fmt_pct(cov)})")
            md_lines.append(f"| {type_label} | {diff_label} | " + " | ".join(cells_md) + " |")
    drop_cells = [f"{drop_actual[c] * 100:+.1f}" for c in cols]
    md_lines.append(f"| **Sel-acc drop (stable→svr)** |  | " + " | ".join(drop_cells) + " |")
    md_table = "\n".join(md_lines)

    caption = (
        "**Difficulty-Class Breakdown (T3 +SKIP).** T3 LLM methods with selective abstention "
        "(SKIP variants): per-type × per-difficulty-class selective accuracy "
        "(%, 4-seed mean, NL input). Each cell is `sel_acc (coverage)`. "
        "GPT = GPT-5.4; Gem = Gemini 3.1; DS = DeepSeek V3.2; QW = Qwen3 235B. "
        "D = Direct; S = Schema; +S = SKIP variant. DeepSeek and Qwen3 "
        "essentially never abstain (coverage ≈ 100%), so their selective "
        "accuracy approximately equals answer-only accuracy."
    )
    footnotes = [
        "*Paper location: `Appendix diagnostics section` (`tab:E2b-skip`).*",
        "*Reduction: pool across 4 seeds, then micro selective accuracy per "
        "(type, diff) cell. Footer = `stable_overall - svr_overall` of "
        "pooled-within-diff micro selective accuracy.*",
        f"*Reproduction tolerance: ±{PAPER_TOLERANCE} absolute on every cell point "
        f"and footer drop.*",
    ]

    csv_p, md_p = write_outputs(
        "t3_llm_selective_per_type_per_difficulty",
        output_rows,
        md_table=md_table,
        script_name="paper_artifacts.appendix.t3_llm_selective_per_type_per_difficulty",
        md_caption=caption,
        md_footnotes=footnotes,
        subdir="appendix",
    )
    elapsed = time.time() - t0
    n_pass = len(output_rows) - fail_count
    print(
        f"[t3_llm_selective_per_type_per_difficulty] wrote {len(output_rows)} cells "
        f"({n_pass} OK / {fail_count} FAIL) in {elapsed:.1f}s "
        f"-> {csv_p}, {md_p}"
    )
    return fail_count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    sys.exit(0 if main() == 0 else 1)
