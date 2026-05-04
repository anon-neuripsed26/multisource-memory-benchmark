"""Reproduce paper Difficulty-Class Breakdown (T2 +SKIP) — T2 fusion +SKIP per-type × per-difficulty (extracted μ).

Source: ``Appendix diagnostics section`` lines 119-160
(``\\label{tab:E2a-skip}``).

Per-type × per-difficulty-class **selective accuracy** (%, 4-seed mean,
extracted μ), with paired coverage (%) per cell. 6 selective fusion
methods, 8 reasoning types × 3 difficulty classes (24 rows). Footer
reports overall (stable → svr) drop in selective accuracy per method
(positive = svr easier than stable on answered cells, which is the case
for learned-bias methods that abstain on the hard cells).

Method spec:
  - DSNBF+S = ``DSNBFSelective``
  - NBF+S   = ``NBFSelective``
  - ABF+S   = ``ABFSelective``
  - SSB+S   = ``SSBSelective`` (with seed=42 to match v1.0 reference)
  - BCF+S   = ``BCF`` (no Selective subclass; never abstains; coverage=100%)
  - MV+S    = ``MajorityVote(seed=42)`` (no Selective subclass; coverage=100%)

The "+S" suffix on BCF and MV is paper notation for the SKIP-eligible
column slot; per the paper text these methods degenerate to no-skip
(``BCF+S coverage=100%``, ``MV+S coverage=100%``), so their selective
accuracy equals their forced accuracy from Difficulty-Class Breakdown (T2 fusion).

Reduction: pool across 4 seeds (``pool_results_across_seeds``), group
by ``(reasoning_type, difficulty_class)``, then **micro selective
accuracy** per cell (= ``correct_answered / answered_count`` summed
across all qids in the cell). Coverage is pooled likewise. Verified
against ``data/benchmark/results/d2_skip_perq_perdiff_4seed.json`` (e.g.
Trend stable DSNBF+S = 92.95%, paper rounds to 92.9). Per-seed-then-mean
*does not* reproduce paper Difficulty-Class Breakdown (T2 +SKIP) cells because per-seed answered
counts vary (unlike Difficulty-Class Breakdown (T2 fusion) forced where coverage=100% throughout, so
both reductions coincide).

Footer drop = ``stable_overall - svr_overall`` of pooled-within-diff
micro selective accuracy. Paper sign convention: positive = "drops X
pp from stable to svr" (BCF+S +17.8 = magnitude of the forced drop in
Difficulty-Class Breakdown (T2 fusion); for fusion methods that abstain effectively, e.g. DSNBF+S
+1.5, the drop magnitude is small).
"""

from __future__ import annotations

import argparse
import sys
import time

from collections import defaultdict

from survey2agent.evaluation.aggregator import aggregate_metrics
from survey2agent.evaluation.multi_seed import pool_results_across_seeds
from survey2agent.methods import (
    ABFSelective,
    BCF,
    DSNBFSelective,
    MajorityVote,
    NBFSelective,
    SSBSelective,
)

from .._common import (
    PAPER_TOLERANCE,
    emit_row,
    load_shared_resources,
    run_mixed_mode_across_seeds,
    write_outputs,
    _md_fmt_pct,
)


# ── Method spec (column order matches paper Difficulty-Class Breakdown (T2 +SKIP) header) ──────


METHODS = [
    ("DSNBF+S", lambda: DSNBFSelective()),
    ("NBF+S",   lambda: NBFSelective()),
    ("ABF+S",   lambda: ABFSelective()),
    ("SSB+S",   lambda: SSBSelective(seed=42)),
    ("BCF+S",   lambda: BCF()),                    # degenerate: never skips
    ("MV+S",    lambda: MajorityVote(seed=42)),    # degenerate: never skips
]


# ── Paper Difficulty-Class Breakdown (T2 +SKIP) numbers (appendix_e_diagnostics.tex L130-156) ──────


# Row order: same as Difficulty-Class Breakdown (T2 fusion) (DSNBF descending).
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

# (type_key, diff_key) -> tuple of (sel_acc, coverage) per method.
# Order: DSNBF+S, NBF+S, ABF+S, SSB+S, BCF+S, MV+S.
PAPER_TAB_T2_FUSION_SELECTIVE_PER_TYPE_PER_DIFFICULTY: dict[tuple[str, str], tuple[tuple[float, float], ...]] = {
    ("identity", "stable"): (
        (0.903, 0.969), (0.903, 0.972), (0.918, 0.844), (0.890, 0.997), (0.853, 1.000), (0.900, 1.000),
    ),
    ("identity", "temporal_shift"): (
        (0.878, 0.975), (0.893, 0.938), (0.905, 0.787), (0.871, 0.991), (0.847, 1.000), (0.841, 1.000),
    ),
    ("identity", "stated_vs_revealed"): (
        (0.978, 1.000), (0.975, 1.000), (0.959, 0.841), (0.961, 0.969), (0.853, 1.000), (0.872, 1.000),
    ),
    ("arbitration", "stable"): (
        (0.876, 0.860), (0.874, 0.829), (0.759, 0.785), (0.734, 0.933), (0.698, 1.000), (0.729, 1.000),
    ),
    ("arbitration", "temporal_shift"): (
        (0.901, 0.825), (0.881, 0.842), (0.740, 0.650), (0.729, 0.875), (0.608, 1.000), (0.637, 1.000),
    ),
    ("arbitration", "stated_vs_revealed"): (
        (0.849, 0.908), (0.858, 0.910), (0.620, 0.554), (0.851, 0.558), (0.460, 1.000), (0.435, 1.000),
    ),
    ("control", "stable"): (
        (0.917, 0.906), (0.914, 0.912), (0.918, 0.531), (0.954, 0.681), (0.822, 1.000), (0.744, 1.000),
    ),
    ("control", "temporal_shift"): (
        (0.874, 0.891), (0.881, 0.891), (0.925, 0.456), (0.917, 0.750), (0.772, 1.000), (0.769, 1.000),
    ),
    ("control", "stated_vs_revealed"): (
        (0.926, 0.884), (0.898, 0.922), (0.938, 0.653), (0.925, 0.875), (0.841, 1.000), (0.850, 1.000),
    ),
    ("causal", "stable"): (
        (0.875, 0.772), (0.864, 0.756), (0.791, 0.778), (0.831, 0.831), (0.769, 1.000), (0.741, 1.000),
    ),
    ("causal", "temporal_shift"): (
        (0.854, 0.750), (0.861, 0.741), (0.772, 0.672), (0.762, 0.775), (0.678, 1.000), (0.656, 1.000),
    ),
    ("causal", "stated_vs_revealed"): (
        (0.884, 0.859), (0.882, 0.691), (0.704, 0.697), (0.909, 0.653), (0.628, 1.000), (0.616, 1.000),
    ),
    ("trend", "stable"): (
        (0.929, 0.753), (0.909, 0.787), (0.930, 0.756), (0.879, 0.881), (0.841, 1.000), (0.816, 1.000),
    ),
    ("trend", "temporal_shift"): (
        (0.902, 0.797), (0.915, 0.809), (0.931, 0.769), (0.875, 0.900), (0.853, 1.000), (0.828, 1.000),
    ),
    ("trend", "stated_vs_revealed"): (
        (0.849, 0.641), (0.691, 0.738), (0.460, 0.394), (0.676, 0.819), (0.338, 1.000), (0.472, 1.000),
    ),
    ("plan_reality", "stable"): (
        (0.816, 0.884), (0.819, 0.828), (0.847, 0.713), (0.809, 0.769), (0.784, 1.000), (0.681, 1.000),
    ),
    ("plan_reality", "temporal_shift"): (
        (0.913, 0.894), (0.923, 0.853), (0.950, 0.750), (0.936, 0.784), (0.878, 1.000), (0.809, 1.000),
    ),
    ("plan_reality", "stated_vs_revealed"): (
        (0.756, 0.703), (0.757, 0.734), (0.780, 0.553), (0.756, 0.603), (0.691, 1.000), (0.569, 1.000),
    ),
    ("missing_data", "stable"): (
        (0.847, 0.750), (0.862, 0.787), (0.917, 0.504), (0.814, 0.931), (0.681, 1.000), (0.740, 1.000),
    ),
    ("missing_data", "temporal_shift"): (
        (0.850, 0.748), (0.866, 0.729), (0.941, 0.498), (0.835, 0.908), (0.708, 1.000), (0.735, 1.000),
    ),
    ("missing_data", "stated_vs_revealed"): (
        (0.872, 0.617), (0.868, 0.648), (0.906, 0.244), (0.735, 0.746), (0.444, 1.000), (0.490, 1.000),
    ),
    ("annotation", "stable"): (
        (0.855, 0.559), (0.907, 0.469), (0.739, 0.647), (0.662, 0.934), (0.653, 1.000), (0.644, 1.000),
    ),
    ("annotation", "temporal_shift"): (
        (0.772, 0.494), (0.816, 0.459), (0.682, 0.600), (0.624, 0.947), (0.613, 1.000), (0.628, 1.000),
    ),
    ("annotation", "stated_vs_revealed"): (
        (0.574, 0.338), (0.663, 0.316), (0.607, 0.438), (0.590, 0.800), (0.562, 1.000), (0.484, 1.000),
    ),
}

# Footer: Overall sel-acc drop (stable -> svr) per method.
PAPER_FOOTER_DROP = {
    "DSNBF+S": +0.015,
    "NBF+S":   +0.033,
    "ABF+S":   +0.085,
    "SSB+S":   +0.011,
    "BCF+S":   +0.169,
    "MV+S":    +0.164,
}


def _pool_per_cell_and_overall(
    seed_results,
) -> tuple[
    dict[tuple[str, str], dict[str, float]],
    dict[str, dict[str, float]],
]:
    r"""Pool across 4 seeds, group by (type, diff) and by diff alone.

    Returns ``(per_cell, per_diff_overall)`` where each value is a dict
    of headline metrics from :func:`aggregate_metrics` (notably
    ``selective_accuracy``, ``selective_accuracy_micro``, ``coverage``).

    The paper Difficulty-Class Breakdown (T2 +SKIP) / Difficulty-Class Breakdown (T3 +SKIP) cells are computed as
    pool-across-seeds then **micro selective accuracy pooled across qids
    in the (type, diff) cell**
    (= ``correct_answered / answered_count`` summed over all records).
    Verified against ``data/benchmark/results/d2_skip_perq_perdiff_4seed.json``
    (Trend stable DSNBF+S = 92.95\%, paper rounds to 92.9).
    """
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

    sel_by_method: dict[str, dict[tuple[str, str], float]] = {}
    cov_by_method: dict[str, dict[tuple[str, str], float]] = {}
    sel_overall_by_method: dict[str, dict[str, float]] = {}
    for col_label, factory in METHODS:
        print(f"[t2_fusion_selective_per_type_per_difficulty] running {col_label} on extracted μ...", flush=True)
        sr = run_mixed_mode_across_seeds(
            factory, oracle_atoms=oracle_atoms, llm_atoms=llm_atoms,
            gts=gts, splits=splits, diff_idx=diff_idx,
        )
        per_cell, per_diff = _pool_per_cell_and_overall(sr)
        sel_by_method[col_label] = {k: v["selective_accuracy_micro"] for k, v in per_cell.items()}
        cov_by_method[col_label] = {k: v["coverage"] for k, v in per_cell.items()}
        sel_overall_by_method[col_label] = {k: v["selective_accuracy_micro"] for k, v in per_diff.items()}

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
            paper_pairs = PAPER_TAB_T2_FUSION_SELECTIVE_PER_TYPE_PER_DIFFICULTY[(type_key, diff_key)]
            for (col_label, _factory), (paper_sel, paper_cov) in zip(METHODS, paper_pairs):
                add_cell(
                    type_key=type_key, type_label=type_label,
                    diff_key=diff_key, diff_label=diff_label,
                    col_label=col_label, metric_name="selective_accuracy",
                    point=sel_by_method[col_label].get((type_key, diff_key)),
                    paper_v=paper_sel,
                )
                add_cell(
                    type_key=type_key, type_label=type_label,
                    diff_key=diff_key, diff_label=diff_label,
                    col_label=col_label, metric_name="coverage",
                    point=cov_by_method[col_label].get((type_key, diff_key)),
                    paper_v=paper_cov,
                )

    # Footer: overall sel-acc drop (stable -> svr) per method.
    # Paper convention: positive = "drops X pp from stable to svr".
    #   paper_drop = stable_overall - svr_overall
    # where overall is per-seed-then-mean pooled-within-diff (selective
    # accuracy = correct_answered / answered_count summed across all
    # types in the diff class). Cell-mean does NOT match here because
    # selective accuracy weights vary with coverage (different from
    # forced accuracy where coverage=100% throughout).
    drop_actual: dict[str, float] = {}
    for col_label, _factory in METHODS:
        ov = sel_overall_by_method[col_label]
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
    cols = [c for c, _ in METHODS]
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
        "**Difficulty-Class Breakdown (T2 +SKIP).** T2 fusion methods with selective abstention "
        "(SKIP variants): per-type × per-difficulty-class selective accuracy "
        "(%, 4-seed mean, extracted μ). Each cell is `sel_acc (coverage)`. "
        "BCF+S and MV+S degenerate to no-skip (coverage=100%, sel_acc = forced "
        "accuracy from Difficulty-Class Breakdown (T2 fusion)). Footer drop computed on selective accuracy."
    )
    footnotes = [
        "*Paper location: `Appendix diagnostics section` (`tab:E2a-skip`).*",
        "*Reduction: per-cell per-seed-then-mean "
        "(`aggregate_per_seed_then_average_with_breakdown`) on both "
        "`selective_accuracy` and `coverage`. Each cell emits two CSV rows.*",
        "*Footer drop = mean(svr cells) − mean(stable cells) of selective_accuracy. "
        "BCF+S and MV+S are sign-flipped (paper writes magnitude of forced drop "
        "since they never abstain).*",
        f"*Reproduction tolerance: ±{PAPER_TOLERANCE} absolute on every cell point "
        f"and footer drop.*",
    ]

    csv_p, md_p = write_outputs(
        "t2_fusion_selective_per_type_per_difficulty",
        output_rows,
        md_table=md_table,
        script_name="paper_artifacts.appendix.t2_fusion_selective_per_type_per_difficulty",
        md_caption=caption,
        md_footnotes=footnotes,
        subdir="appendix",
    )
    elapsed = time.time() - t0
    n_pass = len(output_rows) - fail_count
    print(
        f"[t2_fusion_selective_per_type_per_difficulty] wrote {len(output_rows)} cells "
        f"({n_pass} OK / {fail_count} FAIL) in {elapsed:.1f}s "
        f"-> {csv_p}, {md_p}"
    )
    return fail_count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    sys.exit(0 if main() == 0 else 1)
