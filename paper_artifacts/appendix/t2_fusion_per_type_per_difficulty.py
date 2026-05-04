"""Reproduce paper Difficulty-Class Breakdown (T2 fusion) — T2 fusion methods per-type × per-difficulty (extracted μ).

Source: ``Appendix diagnostics section`` lines 24-66
(``\\label{tab:E2a}``).

Per-type × per-difficulty-class macro accuracy (%, answer-only mode, 4-seed
mean, extracted μ̂). 6 fusion methods, 8 reasoning types × 3 difficulty
classes (24 rows). Footer reports overall (stable → svr) drop per method.

Reduction: per-cell per-seed-then-mean
(``aggregate_per_seed_then_average_with_breakdown`` with
``breakdown_keys=("reasoning_type", "difficulty_class")``,
``metric="forced_accuracy"``). For each (type, diff) cell: macro-over-qid
within (seed, type, diff), then arithmetic mean across 4 seeds. This
matches the legacy ``_d2_skip_recompute.py`` reduction.

Footer "Overall drop" = (overall svr accuracy − overall stable accuracy),
where "overall" is the per-seed-then-mean of forced_accuracy on the
diff-class subset (no per-type macro), computed via
``aggregate_per_seed_then_average_with_breakdown`` with
``breakdown_keys="difficulty_class"``.

Row order matches paper Difficulty-Class Breakdown (T2 fusion) (DSNBF descending, identical to Per-Type Accuracy):
Ident, Arbit, Ctrl, Factor, Temp, P-R, Miss, Annot.
"""

from __future__ import annotations

import argparse
import sys
import time

from survey2agent.evaluation.multi_seed import (
    aggregate_per_seed_then_average_with_breakdown,
)
from survey2agent.methods import (
    ABF,
    BCF,
    DSNBF,
    MajorityVote,
    NBF,
    SSB,
)

from .._common import (
    PAPER_TOLERANCE,
    emit_row,
    load_shared_resources,
    run_mixed_mode_across_seeds,
    write_outputs,
    _md_fmt_pct,
)


# ── Method spec ─────────────────────────────────────────────────────


# Column order matches paper Difficulty-Class Breakdown (T2 fusion) header.
#
# SSB and MajorityVote take an explicit ``seed=42`` to match the
# v1.0 reference generation script (``v1.0 reference: methods/baselines.py``
# defaults: ``SingleSourceBest(seed=42)``, ``MajorityVote(seed=42)``).
# Without the explicit seed the per-cell RNG variance (random fallback on
# null source for SSB; random tie-break for MV) drifts cells by 1-3 pp,
# which exceeds the ±0.005 paper-lock tolerance on per-(type, difficulty)
# breakdowns even though the full-pool macro is within tolerance. The
# Release class defaults are ``seed=None``; passing
# ``seed=42`` here matches the upstream spec used to compute the paper
# numbers, not a patch.
METHODS = [
    ("DSNBF", lambda: DSNBF()),
    ("NBF",   lambda: NBF()),
    ("ABF",   lambda: ABF()),
    ("SSB",   lambda: SSB(seed=42)),
    ("BCF",   lambda: BCF()),
    ("MV",    lambda: MajorityVote(seed=42)),
]


# ── Paper Difficulty-Class Breakdown (T2 fusion) numbers (appendix_e_diagnostics.tex L33-58) ────────────


# Row order: same as Per-Type Accuracy (DSNBF descending).
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

# (type_key, diff_key) -> (DSNBF, NBF, ABF, SSB, BCF, MV) in column order.
PAPER_TAB_T2_FUSION_PER_TYPE_PER_DIFFICULTY: dict[tuple[str, str], tuple[float, ...]] = {
    ("identity",     "stable"):              (0.891, 0.884, 0.891, 0.887, 0.853, 0.900),
    ("identity",     "temporal_shift"):      (0.872, 0.872, 0.872, 0.866, 0.847, 0.841),
    ("identity",     "stated_vs_revealed"):  (0.978, 0.975, 0.950, 0.956, 0.853, 0.872),
    ("arbitration",  "stable"):              (0.852, 0.865, 0.706, 0.706, 0.698, 0.729),
    ("arbitration",  "temporal_shift"):      (0.840, 0.831, 0.665, 0.710, 0.608, 0.637),
    ("arbitration",  "stated_vs_revealed"):  (0.815, 0.831, 0.463, 0.815, 0.460, 0.435),
    ("control",      "stable"):              (0.884, 0.881, 0.709, 0.859, 0.822, 0.744),
    ("control",      "temporal_shift"):      (0.816, 0.834, 0.781, 0.806, 0.772, 0.769),
    ("control",      "stated_vs_revealed"):  (0.847, 0.850, 0.850, 0.872, 0.841, 0.850),
    ("causal",       "stable"):              (0.809, 0.769, 0.741, 0.762, 0.769, 0.741),
    ("causal",       "temporal_shift"):      (0.794, 0.744, 0.653, 0.684, 0.678, 0.656),
    ("causal",       "stated_vs_revealed"):  (0.819, 0.844, 0.575, 0.878, 0.628, 0.616),
    ("trend",        "stable"):              (0.859, 0.834, 0.834, 0.844, 0.841, 0.816),
    ("trend",        "temporal_shift"):      (0.844, 0.853, 0.850, 0.850, 0.853, 0.828),
    ("trend",        "stated_vs_revealed"):  (0.762, 0.641, 0.628, 0.628, 0.338, 0.472),
    ("plan_reality", "stable"):              (0.791, 0.794, 0.738, 0.784, 0.784, 0.681),
    ("plan_reality", "temporal_shift"):      (0.878, 0.869, 0.847, 0.894, 0.878, 0.809),
    ("plan_reality", "stated_vs_revealed"):  (0.681, 0.706, 0.594, 0.691, 0.691, 0.569),
    ("missing_data", "stable"):              (0.800, 0.804, 0.783, 0.779, 0.681, 0.740),
    ("missing_data", "temporal_shift"):      (0.800, 0.794, 0.783, 0.806, 0.708, 0.735),
    ("missing_data", "stated_vs_revealed"):  (0.717, 0.723, 0.667, 0.696, 0.444, 0.490),
    ("annotation",   "stable"):              (0.694, 0.700, 0.647, 0.650, 0.653, 0.644),
    ("annotation",   "temporal_shift"):      (0.666, 0.659, 0.619, 0.613, 0.613, 0.628),
    ("annotation",   "stated_vs_revealed"):  (0.559, 0.562, 0.550, 0.562, 0.562, 0.484),
}

# Footer: Overall drop (stable -> svr) per method.
PAPER_FOOTER_DROP = {
    "DSNBF": -0.051,
    "NBF":   -0.051,
    "ABF":   -0.106,
    "SSB":   -0.018,
    "BCF":   -0.169,
    "MV":    -0.164,
}

# Tolerance for footer drops: ±0.005 on the absolute drop value
# (= ±0.5 pp). Same as cell tolerance.


def _per_cell_points(seed_results) -> dict[tuple[str, str], float]:
    """Return ``{(type_key, diff_key): point}`` from per-seed-then-mean breakdown."""
    raw = aggregate_per_seed_then_average_with_breakdown(
        seed_results,
        breakdown_keys=("reasoning_type", "difficulty_class"),
        metric="forced_accuracy",
    )
    return {k: v["point"] for k, v in raw.items()}


def main() -> int:
    t0 = time.time()
    splits, diff_idx, gts, oracle_atoms, llm_atoms = load_shared_resources()

    # Run all 6 fusion methods on extracted μ
    points_by_method: dict[str, dict[tuple[str, str], float]] = {}
    seed_results_by_method: dict[str, dict] = {}
    for col_label, factory in METHODS:
        print(f"[t2_fusion_per_type_per_difficulty] running {col_label} on extracted μ...", flush=True)
        sr = run_mixed_mode_across_seeds(
            factory, oracle_atoms=oracle_atoms, llm_atoms=llm_atoms,
            gts=gts, splits=splits, diff_idx=diff_idx,
        )
        seed_results_by_method[col_label] = sr
        points_by_method[col_label] = _per_cell_points(sr)

    # Build CSV rows
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
            paper_tuple = PAPER_TAB_T2_FUSION_PER_TYPE_PER_DIFFICULTY[(type_key, diff_key)]
            for (col_label, _factory), paper_v in zip(METHODS, paper_tuple):
                point = points_by_method[col_label].get((type_key, diff_key))
                add_cell(
                    type_key=type_key, type_label=type_label,
                    diff_key=diff_key, diff_label=diff_label,
                    col_label=col_label, point=point, paper_v=paper_v,
                )

    # Footer: overall drop (stable -> svr) per method.
    # Paper Difficulty-Class Breakdown (T2 fusion) footer uses **pooled-on-diff** reduction:
    # ``aggregate_per_seed_then_average_with_breakdown(breakdown_keys="difficulty_class")``
    # then ``svr_overall - stable_overall``. NOT mean-of-the-8-displayed-cells.
    # (Paper Difficulty-Class Breakdown (T3 LLM) for T3 LLMs uses mean-of-cells; the two tables disagree on
    # convention.) Verified: pooled-on-diff matches paper
    # to within 0.02 pp for all 6 methods including ABF/BCF/MV which previously
    # failed mean-of-cells by 0.9-1.3 pp.
    drop_actual: dict[str, float] = {}
    for col_label, _factory in METHODS:
        sr = seed_results_by_method[col_label]
        per_diff = aggregate_per_seed_then_average_with_breakdown(
            sr, breakdown_keys="difficulty_class", metric="forced_accuracy",
        )
        stable_overall = per_diff[("stable",)]["point"]
        svr_overall = per_diff[("stated_vs_revealed",)]["point"]
        drop = svr_overall - stable_overall
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
    cols = [c for c, _ in METHODS]
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
    # Footer
    drop_cells = [
        f"{drop_actual[c] * 100:+.1f}" for c in cols
    ]
    md_lines.append(f"| **Drop (stable→svr)** |  | " + " | ".join(drop_cells) + " |")
    md_table = "\n".join(md_lines)

    caption = (
        "**Difficulty-Class Breakdown (T2 fusion).** T2 fusion methods: per-type × per-difficulty-class accuracy "
        "(%, 4-seed mean, extracted μ̂). svr = stated_vs_revealed; ts = temporal_shift. "
        "Types in Per-Type Accuracy order (DSNBF descending). Footer: overall drop (stable → svr) "
        "per method."
    )
    footnotes = [
        "*Paper location: `Appendix diagnostics section` (`tab:E2a`).*",
        "*Reduction: per-cell per-seed-then-mean "
        "(`aggregate_per_seed_then_average_with_breakdown`, "
        "`breakdown_keys=(\"reasoning_type\", \"difficulty_class\")`, "
        "`metric=\"forced_accuracy\"`). Footer drop computed from "
        "`breakdown_keys=\"difficulty_class\"` overall.*",
        f"*Reproduction tolerance: ±{PAPER_TOLERANCE} absolute on per-cell points "
        f"and on footer drop values.*",
    ]

    csv_p, md_p = write_outputs(
        "t2_fusion_per_type_per_difficulty",
        output_rows,
        md_table=md_table,
        script_name="paper_artifacts.appendix.t2_fusion_per_type_per_difficulty",
        md_caption=caption,
        md_footnotes=footnotes,
        subdir="appendix",
    )
    elapsed = time.time() - t0
    n_pass = len(output_rows) - fail_count
    print(
        f"[t2_fusion_per_type_per_difficulty] wrote {len(output_rows)} cells "
        f"({n_pass} OK / {fail_count} FAIL) in {elapsed:.1f}s "
        f"-> {csv_p}, {md_p}"
    )
    return fail_count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    sys.exit(0 if main() == 0 else 1)
