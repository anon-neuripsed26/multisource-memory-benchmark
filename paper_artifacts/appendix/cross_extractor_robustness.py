"""Reproduce paper Cross-Extractor Robustness — Cross-extractor robustness (4-seed mean).

Source: ``Appendix robustness section`` (lines 116-138).

6 rows × 4 columns (μ*, μ̂GPT, μ̂Gem, Δ) = 24 paper-lock cells.

Sources:
  μ*       -> ``4seed_results.json.per_seed[sd][JSON_KEY].overall_macro``
  μ̂GPT     -> ``full_comparison_<sd>.json.extraction[JSON_KEY].overall_macro``
  μ̂Gem     -> ``full_comparison_gemini_<sd>.json.extraction[JSON_KEY].overall_macro``
  Δ (pp)   -> μ̂Gem − μ̂GPT (in fraction units; paper rounds to 0.1pp)

Paper-label -> JSON-key:
  DSNBF-NS -> DSNBF-NoSkip   NBF-NS -> NBF-NoSkip   ABF-NS -> PRISM-NoSkip
  BCF(4p)  -> BCF(4p)        MajVote -> Majority-Vote  SSB -> Single-Source-Best

Δ tolerance is relaxed (±0.002) because the paper rounds Δ to 0.1pp from
two independent 0.1pp-rounded means.
"""

from __future__ import annotations

import sys
import time

from .._common import emit_row, write_outputs, _md_fmt_pct, PAPER_TOLERANCE
from .._appendix_helpers import (
    CANONICAL_SEED_KEYS,
    CANONICAL_SEED_STRS,
    PAPER_LABEL_TO_JSON_KEY,
    load_4seed_results,
    load_full_comparison,
    mean_std,
)


# (paper_label, col) -> paper value (fraction); col ∈ {oracle, gpt, gem, delta}
PAPER_TAB_CROSS_EXTRACTOR_ROBUSTNESS: dict[tuple[str, str], float] = {
    ("DSNBF-NS", "oracle"): 0.823, ("DSNBF-NS", "gpt"): 0.803,
    ("DSNBF-NS", "gem"):    0.796, ("DSNBF-NS", "delta"): -0.007,
    ("NBF-NS",   "oracle"): 0.821, ("NBF-NS",   "gpt"): 0.798,
    ("NBF-NS",   "gem"):    0.789, ("NBF-NS",   "delta"): -0.009,
    ("SSB",      "oracle"): 0.790, ("SSB",      "gpt"): 0.772,
    ("SSB",      "gem"):    0.749, ("SSB",      "delta"): -0.023,
    ("ABF-NS",   "oracle"): 0.720, ("ABF-NS",   "gpt"): 0.720,
    ("ABF-NS",   "gem"):    0.724, ("ABF-NS",   "delta"):  0.004,
    ("BCF(4p)",  "oracle"): 0.698, ("BCF(4p)",  "gpt"): 0.692,
    ("BCF(4p)",  "gem"):    0.683, ("BCF(4p)",  "delta"): -0.009,
    ("MajVote",  "oracle"): 0.695, ("MajVote",  "gpt"): 0.688,
    ("MajVote",  "gem"):    0.675, ("MajVote",  "delta"): -0.013,
}

# Map paper_label -> internal key in PAPER_LABEL_TO_JSON_KEY
ROW_TO_PAPER_KEY: dict[str, str] = {
    "DSNBF-NS": "DSNBF",
    "NBF-NS":   "NBF",
    "SSB":      "SSB",
    "ABF-NS":   "ABF",
    "BCF(4p)":  "BCF",
    "MajVote":  "MV",
}
ROW_ORDER: list[str] = ["DSNBF-NS", "NBF-NS", "SSB", "ABF-NS", "BCF(4p)", "MajVote"]
DELTA_TOLERANCE: float = 0.002


def _mean_oracle(json_key: str, fs: dict) -> float:
    vals = [fs["per_seed"][k][json_key]["overall_macro"] for k in CANONICAL_SEED_KEYS]
    return mean_std(vals)[0]


def _mean_extraction(json_key: str, *, gemini: bool) -> float:
    vals = []
    for sd in CANONICAL_SEED_STRS:
        d = load_full_comparison(sd, gemini=gemini)
        vals.append(d["extraction"][json_key]["overall_macro"])
    return mean_std(vals)[0]


def main() -> int:
    t0 = time.time()
    print("[cross_extractor_robustness] loading static JSON...", flush=True)
    fs = load_4seed_results()

    point_by_cell: dict[tuple[str, str], float] = {}
    for paper_label in ROW_ORDER:
        json_key = PAPER_LABEL_TO_JSON_KEY[ROW_TO_PAPER_KEY[paper_label]]
        oracle = _mean_oracle(json_key, fs)
        gpt = _mean_extraction(json_key, gemini=False)
        gem = _mean_extraction(json_key, gemini=True)
        point_by_cell[(paper_label, "oracle")] = oracle
        point_by_cell[(paper_label, "gpt")]    = gpt
        point_by_cell[(paper_label, "gem")]    = gem
        point_by_cell[(paper_label, "delta")]  = gem - gpt

    output_rows: list[dict] = []
    fail_count = skip_count = 0
    for paper_label in ROW_ORDER:
        for col in ("oracle", "gpt", "gem", "delta"):
            paper_v = PAPER_TAB_CROSS_EXTRACTOR_ROBUSTNESS.get((paper_label, col))
            if paper_v is None:
                skip_count += 1
                continue
            tol = DELTA_TOLERANCE if col == "delta" else PAPER_TOLERANCE
            r = emit_row(
                row_id=f"{paper_label}__{col}",
                method_label=f"{paper_label} :: μ-{col}",
                mode="oracle" if col == "oracle" else ("ext-gpt" if col == "gpt" else ("ext-gem" if col == "gem" else "delta")),
                metric="macro_accuracy" if col != "delta" else "macro_accuracy_delta",
                point=point_by_cell[(paper_label, col)],
                paper_point=paper_v,
                tolerance=tol,
            )
            output_rows.append(r)
            if r["paper_match"].startswith("FAIL"):
                fail_count += 1

    md_lines = [
        "| Method | μ* | μ̂GPT | μ̂Gem | Δ (pp) |",
        "|:---|:---:|:---:|:---:|:---:|",
    ]
    for paper_label in ROW_ORDER:
        oracle = point_by_cell[(paper_label, "oracle")]
        gpt = point_by_cell[(paper_label, "gpt")]
        gem = point_by_cell[(paper_label, "gem")]
        delta = point_by_cell[(paper_label, "delta")]
        md_lines.append(
            f"| {paper_label} | {100*oracle:.1f} | {100*gpt:.1f} | {100*gem:.1f} | "
            f"{100*delta:+.1f} |"
        )
    md_table = "\n".join(md_lines)

    csv_p, md_p = write_outputs(
        "cross_extractor_robustness",
        output_rows,
        md_table=md_table,
        script_name="paper_artifacts.appendix.cross_extractor_robustness",
        md_caption=(
            "**Cross-Extractor Robustness.** Fusion macro accuracy (%, 4-seed mean) under two "
            "extraction backends. μ* = oracle structured read-out; μ̂GPT / μ̂Gem "
            "= GPT-5.4 / Gemini 3.1 Pro extraction. Δ = Gemini − GPT."
        ),
        md_footnotes=[
            "*Paper location: `Appendix robustness section` (`tab:F6`).*",
            f"*Tolerance: ±{PAPER_TOLERANCE} on point cells; ±{DELTA_TOLERANCE} on Δ cells.*",
        ],
        subdir="appendix",
    )
    elapsed = time.time() - t0
    n_pass = len(output_rows) - fail_count
    print(
        f"[cross_extractor_robustness] OK={n_pass} FAIL={fail_count} SKIP={skip_count} "
        f"total={len(output_rows)} ({elapsed:.1f}s) -> {csv_p}, {md_p}"
    )
    return fail_count


if __name__ == "__main__":
    sys.exit(0 if main() == 0 else 1)
