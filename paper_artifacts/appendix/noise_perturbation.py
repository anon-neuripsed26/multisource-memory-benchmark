"""Reproduce paper Extraction Noise Tolerance — Extraction noise tolerance (4-seed mean).

Source: ``Appendix robustness section`` (lines 76-94)
plus ``data/benchmark/results/ablation_experiments_s2026032{1-4}.json``
(``perturbed`` block keyed by flip rate ε ∈ {0.0, 0.1, 0.2, 0.3, 0.5}).

5 rows × 6 methods × 1 (point only, no σ in paper) = 30 paper-lock cells.

Paper-label -> JSON-key:
  DSNBF -> DSNBF-NoSkip   NBF -> NBF-NoSkip   ABF -> PRISM-NoSkip
  BCF   -> BCF(4p)        MV  -> Majority-Vote  MC  -> Majority-Class
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


# (eps_str, method) -> paper point (fraction)
PAPER_TAB_NOISE_PERTURBATION: dict[tuple[str, str], float] = {
    ("0.0", "DSNBF"): 0.823, ("0.0", "NBF"): 0.820, ("0.0", "ABF"): 0.720,
    ("0.0", "BCF"):   0.698, ("0.0", "MV"):  0.695, ("0.0", "MC"):  0.571,
    ("0.1", "DSNBF"): 0.748, ("0.1", "NBF"): 0.755, ("0.1", "ABF"): 0.680,
    ("0.1", "BCF"):   0.648, ("0.1", "MV"):  0.656, ("0.1", "MC"):  0.571,
    ("0.2", "DSNBF"): 0.670, ("0.2", "NBF"): 0.685, ("0.2", "ABF"): 0.630,
    ("0.2", "BCF"):   0.605, ("0.2", "MV"):  0.610, ("0.2", "MC"):  0.571,
    ("0.3", "DSNBF"): 0.597, ("0.3", "NBF"): 0.623, ("0.3", "ABF"): 0.580,
    ("0.3", "BCF"):   0.559, ("0.3", "MV"):  0.559, ("0.3", "MC"):  0.571,
    ("0.5", "DSNBF"): 0.423, ("0.5", "NBF"): 0.470, ("0.5", "ABF"): 0.464,
    ("0.5", "BCF"):   0.447, ("0.5", "MV"):  0.452, ("0.5", "MC"):  0.571,
}

ROW_ORDER: list[str] = ["0.0", "0.1", "0.2", "0.3", "0.5"]
COL_ORDER: list[str] = ["DSNBF", "NBF", "ABF", "BCF", "MV", "MC"]


def main() -> int:
    t0 = time.time()
    print("[noise_perturbation] loading ablation JSONs...", flush=True)
    abls_per_seed = {sd: load_ablation(sd) for sd in CANONICAL_SEED_STRS}

    point_by_cell: dict[tuple[str, str], float] = {}
    for eps in ROW_ORDER:
        for method in COL_ORDER:
            json_key = PAPER_LABEL_TO_JSON_KEY[method]
            per_seed = [
                abls_per_seed[sd]["perturbed"][eps][json_key] for sd in CANONICAL_SEED_STRS
            ]
            mean, _ = mean_std(per_seed)
            point_by_cell[(eps, method)] = mean

    output_rows: list[dict] = []
    fail_count = skip_count = 0
    for eps in ROW_ORDER:
        for method in COL_ORDER:
            paper_v = PAPER_TAB_NOISE_PERTURBATION.get((eps, method))
            if paper_v is None:
                skip_count += 1
                continue
            r = emit_row(
                row_id=f"eps{eps}__{method}",
                method_label=f"ε={eps} :: {method}",
                mode="oracle_perturbed", metric="macro_accuracy",
                point=point_by_cell[(eps, method)],
                paper_point=paper_v,
                tolerance=PAPER_TOLERANCE,
            )
            output_rows.append(r)
            if r["paper_match"].startswith("FAIL"):
                fail_count += 1

    md_lines = [
        "| ε | " + " | ".join(COL_ORDER) + " |",
        "|:---:|" + "|".join([":---:"] * len(COL_ORDER)) + "|",
    ]
    for eps in ROW_ORDER:
        cells = [_md_fmt_pct(point_by_cell[(eps, m)]) for m in COL_ORDER]
        md_lines.append(f"| {eps} | " + " | ".join(cells) + " |")
    md_table = "\n".join(md_lines)

    csv_p, md_p = write_outputs(
        "noise_perturbation",
        output_rows,
        md_table=md_table,
        script_name="paper_artifacts.appendix.noise_perturbation",
        md_caption=(
            "**Extraction Noise Tolerance.** Macro accuracy (%, 4-seed mean) under random label "
            "flips applied to oracle μ. MC = Majority-Class (noise-invariant)."
        ),
        md_footnotes=[
            "*Paper location: `Appendix robustness section` (`tab:F5`).*",
            f"*Tolerance: ±{PAPER_TOLERANCE} on all cells.*",
        ],
        subdir="appendix",
    )
    elapsed = time.time() - t0
    n_pass = len(output_rows) - fail_count
    print(
        f"[noise_perturbation] OK={n_pass} FAIL={fail_count} SKIP={skip_count} "
        f"total={len(output_rows)} ({elapsed:.1f}s) -> {csv_p}, {md_p}"
    )
    return fail_count


if __name__ == "__main__":
    sys.exit(0 if main() == 0 else 1)
