"""Reproduce paper Training Size Sensitivity — Training size sensitivity (4-seed mean ± σ).

Source: ``Appendix robustness section`` (lines 27-46)
plus ``data/benchmark/results/ablation_experiments_s2026032{1-4}.json``
(``ablation`` block keyed by training size in {50, 100, 150, 216}).

4 rows × 6 methods × 2 (point + σ) = 48 paper-lock cells.

Paper-label -> JSON-key:
  DSNBF -> DSNBF-NoSkip   NBF -> NBF-NoSkip   ABF -> PRISM-NoSkip
  BCF   -> BCF(4p)        MV  -> Majority-Vote  SSB -> Single-Source-Best

σ uses population stddev across the 4 seeds (DSNBF@216 σ_pop = 0.62,
paper says 0.7 — sample stddev would give 0.72; σ tolerance ±0.002
absorbs the rounding inconsistency).
"""

from __future__ import annotations

import sys
import time

from .._common import emit_row, write_outputs, _md_fmt_pct, PAPER_TOLERANCE
from .._appendix_helpers import (
    CANONICAL_SEED_STRS,
    PAPER_LABEL_TO_JSON_KEY,
    load_ablation,
    mean_std,
)


# (train_size, method) -> {"point": fraction, "sigma": fraction}
PAPER_TAB_TRAIN_SIZE_ABLATION: dict[tuple[str, str], dict[str, float]] = {
    ("50",  "DSNBF"): {"point": 0.809, "sigma": 0.006},
    ("50",  "NBF"):   {"point": 0.805, "sigma": 0.006},
    ("50",  "ABF"):   {"point": 0.718, "sigma": 0.010},
    ("50",  "BCF"):   {"point": 0.693, "sigma": 0.007},
    ("50",  "MV"):    {"point": 0.695, "sigma": 0.006},
    ("50",  "SSB"):   {"point": 0.782, "sigma": 0.005},

    ("100", "DSNBF"): {"point": 0.820, "sigma": 0.007},
    ("100", "NBF"):   {"point": 0.817, "sigma": 0.006},
    ("100", "ABF"):   {"point": 0.718, "sigma": 0.009},
    ("100", "BCF"):   {"point": 0.693, "sigma": 0.007},
    ("100", "MV"):    {"point": 0.695, "sigma": 0.006},
    ("100", "SSB"):   {"point": 0.788, "sigma": 0.006},

    ("150", "DSNBF"): {"point": 0.823, "sigma": 0.007},
    ("150", "NBF"):   {"point": 0.819, "sigma": 0.006},
    ("150", "ABF"):   {"point": 0.718, "sigma": 0.009},
    ("150", "BCF"):   {"point": 0.694, "sigma": 0.007},
    ("150", "MV"):    {"point": 0.695, "sigma": 0.006},
    ("150", "SSB"):   {"point": 0.788, "sigma": 0.006},

    ("216", "DSNBF"): {"point": 0.823, "sigma": 0.007},
    ("216", "NBF"):   {"point": 0.820, "sigma": 0.009},
    ("216", "ABF"):   {"point": 0.718, "sigma": 0.010},
    ("216", "BCF"):   {"point": 0.694, "sigma": 0.007},
    ("216", "MV"):    {"point": 0.695, "sigma": 0.006},
    ("216", "SSB"):   {"point": 0.790, "sigma": 0.005},
}

ROW_ORDER: list[str] = ["50", "100", "150", "216"]
COL_ORDER: list[str] = ["DSNBF", "NBF", "ABF", "BCF", "MV", "SSB"]
SIGMA_TOLERANCE: float = 0.002


def main() -> int:
    t0 = time.time()
    print("[train_size_ablation] loading ablation JSONs...", flush=True)
    abls_per_seed = {sd: load_ablation(sd) for sd in CANONICAL_SEED_STRS}

    point_by_cell: dict[tuple[str, str], float] = {}
    sigma_by_cell: dict[tuple[str, str], float] = {}
    for size in ROW_ORDER:
        for method in COL_ORDER:
            json_key = PAPER_LABEL_TO_JSON_KEY[method]
            per_seed = [abls_per_seed[sd]["ablation"][size][json_key] for sd in CANONICAL_SEED_STRS]
            mean, sigma = mean_std(per_seed)
            point_by_cell[(size, method)] = mean
            sigma_by_cell[(size, method)] = sigma

    output_rows: list[dict] = []
    fail_count = skip_count = 0
    for size in ROW_ORDER:
        for method in COL_ORDER:
            paper = PAPER_TAB_TRAIN_SIZE_ABLATION.get((size, method))
            if paper is None:
                skip_count += 2
                continue
            r_point = emit_row(
                row_id=f"N{size}__{method}__point",
                method_label=f"N={size} :: {method}",
                mode="oracle", metric="macro_accuracy",
                point=point_by_cell[(size, method)],
                paper_point=paper["point"],
                tolerance=PAPER_TOLERANCE,
            )
            r_sigma = emit_row(
                row_id=f"N{size}__{method}__sigma",
                method_label=f"N={size} :: {method} σ",
                mode="oracle", metric="macro_accuracy_sigma",
                point=sigma_by_cell[(size, method)],
                paper_point=paper["sigma"],
                tolerance=SIGMA_TOLERANCE,
            )
            output_rows.extend([r_point, r_sigma])
            for r in (r_point, r_sigma):
                if r["paper_match"].startswith("FAIL"):
                    fail_count += 1

    md_lines = [
        "| N | " + " | ".join(COL_ORDER) + " |",
        "|:---:|" + "|".join([":---:"] * len(COL_ORDER)) + "|",
    ]
    for size in ROW_ORDER:
        cells = []
        for method in COL_ORDER:
            p = point_by_cell[(size, method)]
            s = sigma_by_cell[(size, method)]
            cells.append(f"{100*p:.1f}±{100*s:.1f}")
        md_lines.append(f"| {size} | " + " | ".join(cells) + " |")
    md_table = "\n".join(md_lines)

    csv_p, md_p = write_outputs(
        "train_size_ablation",
        output_rows,
        md_table=md_table,
        script_name="paper_artifacts.appendix.train_size_ablation",
        md_caption=(
            "**Training Size Sensitivity.** Macro accuracy (%, 4-seed mean ± σ) vs. training size "
            "(oracle μ). σ is population stddev."
        ),
        md_footnotes=[
            "*Paper location: `Appendix robustness section` (`tab:F2`).*",
            f"*Tolerance: ±{PAPER_TOLERANCE} on point cells; ±{SIGMA_TOLERANCE} on σ cells.*",
        ],
        subdir="appendix",
    )
    elapsed = time.time() - t0
    n_pass = len(output_rows) - fail_count
    print(
        f"[train_size_ablation] OK={n_pass} FAIL={fail_count} SKIP={skip_count} "
        f"total={len(output_rows)} ({elapsed:.1f}s) -> {csv_p}, {md_p}"
    )
    return fail_count


if __name__ == "__main__":
    sys.exit(0 if main() == 0 else 1)
