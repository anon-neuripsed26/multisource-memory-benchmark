"""Reproduce paper Difficulty-Class Breakdown (T3 LLM) — T3 LLM methods per-type × per-difficulty (NL input).

Source: ``Appendix diagnostics section`` lines 68-110
(``\\label{tab:E2b}``).

Per-type × per-difficulty-class macro accuracy (%, answer-only mode,
4-seed mean, NL input). 8 LLM cells (4 families × 2 prompting variants),
8 reasoning types × 3 difficulty classes (24 rows). Footer reports
overall (stable → svr) drop per LLM cell.

Reduction: per-cell per-seed-then-mean
(``aggregate_per_seed_then_average_with_breakdown`` with
``breakdown_keys=("reasoning_type", "difficulty_class")``,
``metric="forced_accuracy"``). Same reduction as Difficulty-Class Breakdown (T2 fusion).

Row order matches paper Difficulty-Class Breakdown (T3 LLM): A-Arbit, B-Ident, C-P-R, D-Temp,
E-Factor, F-Miss, G-Annot, Ctrl. (Different from Per-Type Accuracy / Difficulty-Class Breakdown (T2 fusion)
ordering — paper alphabetizes by reasoning-type letter here.)

LLM methods are frozen-artifact backed (``FrozenBulkJSONSource``); they
are deterministic per seed and require no fit / cal step.
"""

from __future__ import annotations

import argparse
import sys
import time

from survey2agent.evaluation.multi_seed import (
    aggregate_per_seed_then_average_with_breakdown,
)
from survey2agent.methods import (
    LLMDirect,
    LLMSchemaAware,
)

from .._common import (
    PAPER_TOLERANCE,
    emit_row,
    load_shared_resources,
    run_llm_across_seeds,
    write_outputs,
    _md_fmt_pct,
)


# ── LLM column spec (paper Difficulty-Class Breakdown (T3 LLM) column order) ──────────────────────


# (col_label, model, variant, display, method_cls)
# Method class is the FORCED variant (LLMDirect / LLMSchemaAware) — the
# selective variants live in t3_llm_selective_per_type_per_difficulty.py.
LLM_COLS = [
    ("GPT-D",  "gpt-5.4",                "direct",       "GPT-5.4 Direct",       LLMDirect),
    ("GPT-S",  "gpt-5.4",                "schema-aware", "GPT-5.4 Schema",       LLMSchemaAware),
    ("Gem-D",  "gemini_p2",              "direct",       "Gemini 3.1 Direct",    LLMDirect),
    ("Gem-S",  "gemini_p2",              "schema-aware", "Gemini 3.1 Schema",    LLMSchemaAware),
    ("DS-D",   "deepseek-v3.2",          "direct",       "DeepSeek V3.2 Direct", LLMDirect),
    ("DS-S",   "deepseek-v3.2",          "schema-aware", "DeepSeek V3.2 Schema", LLMSchemaAware),
    ("QW-D",   "qwen3-235b-a22b-2507",   "direct",       "Qwen3 235B Direct",    LLMDirect),
    ("QW-S",   "qwen3-235b-a22b-2507",   "schema-aware", "Qwen3 235B Schema",    LLMSchemaAware),
]


# ── Paper Difficulty-Class Breakdown (T3 LLM) numbers (appendix_e_diagnostics.tex L77-105) ────────────


# Row order: paper Difficulty-Class Breakdown (T3 LLM) alphabetic-by-letter.
TYPE_ORDER = [
    ("arbitration",  "A-Arbit"),
    ("identity",     "B-Ident"),
    ("plan_reality", "C-P-R"),
    ("trend",        "D-Temp"),
    ("causal",       "E-Factor"),
    ("missing_data", "F-Miss"),
    ("annotation",   "G-Annot"),
    ("control",      "Ctrl"),
]

DIFFS = [
    ("stable",              "stable"),
    ("temporal_shift",      "ts"),
    ("stated_vs_revealed",  "svr"),
]

# (type_key, diff_key) -> (GPT-D, GPT-S, Gem-D, Gem-S, DS-D, DS-S, QW-D, QW-S)
PAPER_TAB_T3_LLM_PER_TYPE_PER_DIFFICULTY: dict[tuple[str, str], tuple[float, ...]] = {
    ("arbitration",  "stable"):              (0.758, 0.777, 0.746, 0.792, 0.665, 0.662, 0.560, 0.604),
    ("arbitration",  "temporal_shift"):      (0.817, 0.819, 0.808, 0.844, 0.642, 0.644, 0.590, 0.658),
    ("arbitration",  "stated_vs_revealed"):  (0.621, 0.604, 0.604, 0.617, 0.521, 0.525, 0.338, 0.340),
    ("identity",     "stable"):              (0.853, 0.975, 0.662, 0.822, 0.856, 0.741, 0.503, 0.459),
    ("identity",     "temporal_shift"):      (0.772, 0.947, 0.644, 0.766, 0.750, 0.631, 0.394, 0.381),
    ("identity",     "stated_vs_revealed"):  (0.831, 0.912, 0.841, 0.853, 0.716, 0.847, 0.847, 0.822),
    ("plan_reality", "stable"):              (0.706, 0.706, 0.709, 0.703, 0.466, 0.478, 0.616, 0.628),
    ("plan_reality", "temporal_shift"):      (0.800, 0.794, 0.794, 0.788, 0.469, 0.506, 0.519, 0.497),
    ("plan_reality", "stated_vs_revealed"):  (0.588, 0.581, 0.591, 0.588, 0.459, 0.462, 0.603, 0.609),
    ("trend",        "stable"):              (0.778, 0.841, 0.753, 0.822, 0.694, 0.709, 0.653, 0.603),
    ("trend",        "temporal_shift"):      (0.800, 0.825, 0.772, 0.822, 0.544, 0.775, 0.628, 0.491),
    ("trend",        "stated_vs_revealed"):  (0.597, 0.394, 0.597, 0.400, 0.606, 0.509, 0.288, 0.550),
    ("causal",       "stable"):              (0.778, 0.784, 0.788, 0.762, 0.447, 0.466, 0.066, 0.209),
    ("causal",       "temporal_shift"):      (0.831, 0.822, 0.784, 0.828, 0.488, 0.534, 0.212, 0.269),
    ("causal",       "stated_vs_revealed"):  (0.516, 0.522, 0.516, 0.516, 0.738, 0.791, 0.034, 0.138),
    ("missing_data", "stable"):              (0.635, 0.688, 0.625, 0.675, 0.408, 0.444, 0.258, 0.483),
    ("missing_data", "temporal_shift"):      (0.660, 0.706, 0.654, 0.679, 0.408, 0.496, 0.269, 0.500),
    ("missing_data", "stated_vs_revealed"):  (0.629, 0.660, 0.650, 0.667, 0.490, 0.496, 0.246, 0.396),
    ("annotation",   "stable"):              (0.353, 0.341, 0.584, 0.562, 0.581, 0.478, 0.378, 0.491),
    ("annotation",   "temporal_shift"):      (0.400, 0.353, 0.522, 0.525, 0.469, 0.472, 0.378, 0.494),
    ("annotation",   "stated_vs_revealed"):  (0.372, 0.338, 0.331, 0.409, 0.447, 0.509, 0.400, 0.425),
    ("control",      "stable"):              (0.762, 0.741, 0.728, 0.759, 0.491, 0.434, 0.244, 0.206),
    ("control",      "temporal_shift"):      (0.819, 0.784, 0.791, 0.797, 0.491, 0.438, 0.459, 0.453),
    ("control",      "stated_vs_revealed"):  (0.791, 0.769, 0.766, 0.775, 0.728, 0.678, 0.753, 0.756),
}

# Footer: Overall drop (stable -> svr) per LLM cell.
PAPER_FOOTER_DROP = {
    "GPT-D": -0.085,
    "GPT-S": -0.134,
    "Gem-D": -0.088,
    "Gem-S": -0.134,
    "DS-D":  +0.012,
    "DS-S":  +0.051,
    "QW-D":  +0.029,
    "QW-S":  +0.044,
}


def _per_cell_points(seed_results) -> dict[tuple[str, str], float]:
    raw = aggregate_per_seed_then_average_with_breakdown(
        seed_results,
        breakdown_keys=("reasoning_type", "difficulty_class"),
        metric="forced_accuracy",
    )
    return {k: v["point"] for k, v in raw.items()}


def main() -> int:
    t0 = time.time()
    splits, diff_idx, gts, oracle_atoms, llm_atoms = load_shared_resources()

    points_by_col: dict[str, dict[tuple[str, str], float]] = {}
    for col_label, model, variant, display, cls in LLM_COLS:
        print(f"[t3_llm_per_type_per_difficulty] running {col_label} ({model} {variant})...", flush=True)
        sr = run_llm_across_seeds(
            model=model, variant=variant, display_name=display,
            selective_cls=cls, llm_atoms=llm_atoms, gts=gts,
            splits=splits, diff_idx=diff_idx,
        )
        points_by_col[col_label] = _per_cell_points(sr)

    output_rows: list[dict] = []
    fail_count = 0

    def add_cell(*, type_key, type_label, diff_key, diff_label, col_label,
                 point, paper_v):
        nonlocal fail_count
        row = emit_row(
            row_id=f"{type_key}__{diff_key}__{col_label}",
            method_label=f"{col_label} :: {type_label} / {diff_label}",
            mode="ext", metric="forced_accuracy",
            point=point, n_personas="",
            paper_point=paper_v,
        )
        output_rows.append(row)
        if row["paper_match"].startswith("FAIL"):
            fail_count += 1

    for type_key, type_label in TYPE_ORDER:
        for diff_key, diff_label in DIFFS:
            paper_tuple = PAPER_TAB_T3_LLM_PER_TYPE_PER_DIFFICULTY[(type_key, diff_key)]
            for (col_label, *_rest), paper_v in zip(LLM_COLS, paper_tuple):
                point = points_by_col[col_label].get((type_key, diff_key))
                add_cell(
                    type_key=type_key, type_label=type_label,
                    diff_key=diff_key, diff_label=diff_label,
                    col_label=col_label, point=point, paper_v=paper_v,
                )

    # Footer: overall drop (stable -> svr) per LLM cell.
    # Paper definition: arithmetic mean of the 8 per-type cells per diff
    # column, then svr_mean - stable_mean. This is the mean over the
    # rendered table cells (not a separate per-seed overall reduction —
    # the two reductions coincide for fusion methods on this dataset but
    # diverge for LLMs whose per-(seed, type, diff) macro masks per-cell
    # variance). Verified against paper Difficulty-Class Breakdown (T3 LLM) QW-S = +4.4 pp.
    drop_actual: dict[str, float] = {}
    for col_label, *_rest in LLM_COLS:
        cells = points_by_col[col_label]
        stable_vals = [cells[(t, "stable")] for t, _ in TYPE_ORDER if (t, "stable") in cells]
        svr_vals = [cells[(t, "stated_vs_revealed")] for t, _ in TYPE_ORDER if (t, "stated_vs_revealed") in cells]
        drop = (sum(svr_vals) / len(svr_vals)) - (sum(stable_vals) / len(stable_vals))
        drop_actual[col_label] = drop
        paper_drop = PAPER_FOOTER_DROP[col_label]
        row = emit_row(
            row_id=f"_footer_drop__{col_label}",
            method_label=f"{col_label} :: overall drop (stable->svr)",
            mode="ext", metric="forced_accuracy_drop",
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
            cells_md = [
                _md_fmt_pct(by_id[f"{type_key}__{diff_key}__{c}"]["point"])
                for c in cols
            ]
            md_lines.append(f"| {type_label} | {diff_label} | " + " | ".join(cells_md) + " |")
    drop_cells = [f"{drop_actual[c] * 100:+.1f}" for c in cols]
    md_lines.append(f"| **Drop (stable→svr)** |  | " + " | ".join(drop_cells) + " |")
    md_table = "\n".join(md_lines)

    caption = (
        "**Difficulty-Class Breakdown (T3 LLM).** T3 LLM methods: per-type × per-difficulty-class accuracy "
        "(%, 4-seed mean, NL input). 8 LLM cells (4 families × 2 prompting variants). "
        "GPT = GPT-5.4; Gem = Gemini 3.1; DS = DeepSeek V3.2; QW = Qwen3 235B. "
        "D = Direct; S = Schema. svr = stated_vs_revealed; ts = temporal_shift. "
        "Footer: overall drop (stable → svr) per LLM cell."
    )
    footnotes = [
        "*Paper location: `Appendix diagnostics section` (`tab:E2b`).*",
        "*Reduction: per-cell per-seed-then-mean "
        "(`aggregate_per_seed_then_average_with_breakdown`, "
        "`breakdown_keys=(\"reasoning_type\", \"difficulty_class\")`, "
        "`metric=\"forced_accuracy\"`).*",
        f"*Reproduction tolerance: ±{PAPER_TOLERANCE} absolute on per-cell points "
        f"and on footer drop values.*",
    ]

    csv_p, md_p = write_outputs(
        "t3_llm_per_type_per_difficulty",
        output_rows,
        md_table=md_table,
        script_name="paper_artifacts.appendix.t3_llm_per_type_per_difficulty",
        md_caption=caption,
        md_footnotes=footnotes,
        subdir="appendix",
    )
    elapsed = time.time() - t0
    n_pass = len(output_rows) - fail_count
    print(
        f"[t3_llm_per_type_per_difficulty] wrote {len(output_rows)} cells "
        f"({n_pass} OK / {fail_count} FAIL) in {elapsed:.1f}s "
        f"-> {csv_p}, {md_p}"
    )
    return fail_count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    sys.exit(0 if main() == 0 else 1)
