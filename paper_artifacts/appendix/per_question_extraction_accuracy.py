"""Reproduce paper Per-Question Extraction Accuracy — Per-question per-difficulty extraction accuracy.

Source: ``Appendix robustness section`` (lines 174-220)
plus ``data/benchmark/results/c6b_perq_perdiff_4seed.json``.

18 questions × 3 difficulties × 2 extractors (GPT, Gemini) plus per-cell
Δ (Gem − GPT, in pp) and 3 overall summary rows.

Per-cell paper-lock budget:
  18 × 3 × 2 (point) = 108 cells
  18 × 3 × 1 (Δ)     =  54 cells
  3 (overall GPT)    +  3 (overall Gem) +  3 (overall Δ) = 9 cells
  Plus σ for the 18×3×2 cells (paper reports point ± σ): 108 σ cells.
  Total: 108 + 108 + 54 + 9 = 279 paper-lock cells.

JSON value units: ``c6b_perq_perdiff_4seed.json[Q__diff__ext]`` =
  {mean: percent (e.g. 92.03), std: percent, per_seed: [...], total_per_seed: [160,160,160,160]}

We convert percent → fraction (÷100) before paper-match comparison.
σ tolerance ±0.002 absolute (paper rounds to 0.1pp = 0.001, so ±0.2pp safe).
Δ tolerance ±0.002 absolute (Δ is difference of two 0.1pp-rounded means).
"""

from __future__ import annotations

import sys
import time

from .._common import emit_row, write_outputs, _md_fmt_pct, PAPER_TOLERANCE
from .._appendix_helpers import load_c6b_perq_perdiff_4seed


QUESTIONS: list[str] = [
    "A1", "A2", "A3", "B2", "B3", "C2", "C3", "D1", "D2",
    "E1", "E2", "F1", "F2", "F3", "G1", "G2", "Ctrl1", "Ctrl2",
]
DIFFS: list[str] = ["stable", "temporal_shift", "stated_vs_revealed"]
DIFF_SHORT: dict[str, str] = {"stable": "St", "temporal_shift": "TS", "stated_vs_revealed": "SvR"}
EXTRACTORS: list[str] = ["gpt", "gem"]

SIGMA_TOLERANCE: float = 0.002
DELTA_TOLERANCE: float = 0.002


# Paper Per-Question Extraction Accuracy values (from appendix_f_robustness.tex L174-225).
# Schema: (Q, diff) -> {"gpt": (mean_frac, sigma_frac), "gem": (mean_frac, sigma_frac), "delta": delta_frac}
# delta_frac is mean_gem - mean_gpt, both in fractions; paper reports in pp.
PAPER_TAB_PER_QUESTION_EXTRACTION_ACCURACY: dict[tuple[str, str], dict] = {
    ("A1","stable"): {"gpt":(0.920,0.009), "gem":(0.916,0.011), "delta":-0.005},
    ("A1","temporal_shift"): {"gpt":(0.939,0.019), "gem":(0.939,0.015), "delta":0.000},
    ("A1","stated_vs_revealed"): {"gpt":(0.992,0.010), "gem":(0.995,0.005), "delta":0.003},

    ("A2","stable"): {"gpt":(0.874,0.006), "gem":(0.928,0.018), "delta":0.054},
    ("A2","temporal_shift"): {"gpt":(0.862,0.015), "gem":(0.926,0.016), "delta":0.064},
    ("A2","stated_vs_revealed"): {"gpt":(0.772,0.008), "gem":(0.764,0.007), "delta":-0.009},

    ("A3","stable"): {"gpt":(0.992,0.007), "gem":(0.983,0.012), "delta":-0.009},
    ("A3","temporal_shift"): {"gpt":(0.980,0.005), "gem":(0.986,0.008), "delta":0.006},
    ("A3","stated_vs_revealed"): {"gpt":(0.888,0.013), "gem":(0.955,0.017), "delta":0.067},

    ("B2","stable"): {"gpt":(0.958,0.015), "gem":(0.971,0.006), "delta":0.014},
    ("B2","temporal_shift"): {"gpt":(0.959,0.014), "gem":(0.970,0.015), "delta":0.011},
    ("B2","stated_vs_revealed"): {"gpt":(0.968,0.013), "gem":(0.990,0.004), "delta":0.022},

    ("B3","stable"): {"gpt":(0.836,0.052), "gem":(0.788,0.013), "delta":-0.048},
    ("B3","temporal_shift"): {"gpt":(0.835,0.033), "gem":(0.808,0.020), "delta":-0.027},
    ("B3","stated_vs_revealed"): {"gpt":(0.950,0.016), "gem":(0.855,0.014), "delta":-0.095},

    ("C2","stable"): {"gpt":(0.983,0.013), "gem":(0.988,0.004), "delta":0.004},
    ("C2","temporal_shift"): {"gpt":(0.984,0.015), "gem":(0.990,0.003), "delta":0.006},
    ("C2","stated_vs_revealed"): {"gpt":(0.965,0.018), "gem":(0.954,0.012), "delta":-0.010},

    ("C3","stable"): {"gpt":(0.769,0.023), "gem":(0.778,0.024), "delta":0.009},
    ("C3","temporal_shift"): {"gpt":(0.922,0.010), "gem":(0.944,0.019), "delta":0.022},
    ("C3","stated_vs_revealed"): {"gpt":(0.712,0.036), "gem":(0.744,0.047), "delta":0.031},

    ("D1","stable"): {"gpt":(0.925,0.016), "gem":(0.944,0.016), "delta":0.019},
    ("D1","temporal_shift"): {"gpt":(0.919,0.016), "gem":(0.927,0.014), "delta":0.008},
    ("D1","stated_vs_revealed"): {"gpt":(0.896,0.018), "gem":(0.917,0.019), "delta":0.021},

    ("D2","stable"): {"gpt":(0.994,0.011), "gem":(1.000,0.000), "delta":0.006},
    ("D2","temporal_shift"): {"gpt":(1.000,0.000), "gem":(0.994,0.006), "delta":-0.006},
    ("D2","stated_vs_revealed"): {"gpt":(0.962,0.018), "gem":(0.988,0.000), "delta":0.025},

    ("E1","stable"): {"gpt":(0.881,0.048), "gem":(0.733,0.003), "delta":-0.148},
    ("E1","temporal_shift"): {"gpt":(0.861,0.027), "gem":(0.728,0.013), "delta":-0.133},
    ("E1","stated_vs_revealed"): {"gpt":(0.825,0.017), "gem":(0.708,0.049), "delta":-0.117},

    ("E2","stable"): {"gpt":(0.885,0.020), "gem":(0.895,0.024), "delta":0.010},
    ("E2","temporal_shift"): {"gpt":(0.898,0.025), "gem":(0.899,0.022), "delta":0.001},
    ("E2","stated_vs_revealed"): {"gpt":(0.947,0.026), "gem":(0.964,0.008), "delta":0.017},

    ("F1","stable"): {"gpt":(0.916,0.013), "gem":(0.922,0.013), "delta":0.006},
    ("F1","temporal_shift"): {"gpt":(0.923,0.020), "gem":(0.927,0.021), "delta":0.003},
    ("F1","stated_vs_revealed"): {"gpt":(0.960,0.024), "gem":(0.981,0.015), "delta":0.021},

    ("F2","stable"): {"gpt":(0.771,0.034), "gem":(0.689,0.022), "delta":-0.082},
    ("F2","temporal_shift"): {"gpt":(0.829,0.021), "gem":(0.720,0.042), "delta":-0.109},
    ("F2","stated_vs_revealed"): {"gpt":(0.840,0.019), "gem":(0.754,0.012), "delta":-0.086},

    ("F3","stable"): {"gpt":(0.931,0.015), "gem":(0.838,0.031), "delta":-0.094},
    ("F3","temporal_shift"): {"gpt":(0.966,0.024), "gem":(0.902,0.032), "delta":-0.064},
    ("F3","stated_vs_revealed"): {"gpt":(0.969,0.013), "gem":(0.952,0.009), "delta":-0.017},

    ("G1","stable"): {"gpt":(0.979,0.012), "gem":(0.985,0.011), "delta":0.006},
    ("G1","temporal_shift"): {"gpt":(0.994,0.004), "gem":(0.992,0.006), "delta":-0.002},
    ("G1","stated_vs_revealed"): {"gpt":(0.979,0.013), "gem":(0.998,0.004), "delta":0.019},

    ("G2","stable"): {"gpt":(0.861,0.010), "gem":(0.909,0.017), "delta":0.048},
    ("G2","temporal_shift"): {"gpt":(0.862,0.004), "gem":(0.880,0.028), "delta":0.017},
    ("G2","stated_vs_revealed"): {"gpt":(0.844,0.022), "gem":(0.835,0.034), "delta":-0.008},

    ("Ctrl1","stable"): {"gpt":(0.985,0.012), "gem":(0.785,0.009), "delta":-0.200},
    ("Ctrl1","temporal_shift"): {"gpt":(0.975,0.019), "gem":(0.802,0.033), "delta":-0.173},
    ("Ctrl1","stated_vs_revealed"): {"gpt":(0.977,0.009), "gem":(0.769,0.074), "delta":-0.208},

    ("Ctrl2","stable"): {"gpt":(0.994,0.011), "gem":(0.988,0.004), "delta":-0.006},
    ("Ctrl2","temporal_shift"): {"gpt":(0.994,0.007), "gem":(0.979,0.017), "delta":-0.015},
    ("Ctrl2","stated_vs_revealed"): {"gpt":(0.931,0.030), "gem":(0.942,0.025), "delta":0.010},
}

# Overall (per-difficulty) paper rows: (diff, ext) -> (mean_frac, sigma_frac);
# (diff, "delta") -> delta_frac
PAPER_OVERALL: dict[tuple[str, str], object] = {
    ("stable",             "gpt"):   (0.909, 0.006),
    ("stable",             "gem"):   (0.886, 0.006),
    ("stable",             "delta"): -0.023,
    ("temporal_shift",     "gpt"):   (0.920, 0.007),
    ("temporal_shift",     "gem"):   (0.898, 0.004),
    ("temporal_shift",     "delta"): -0.022,
    ("stated_vs_revealed", "gpt"):   (0.909, 0.011),
    ("stated_vs_revealed", "gem"):   (0.889, 0.004),
    ("stated_vs_revealed", "delta"): -0.020,
}


def main() -> int:
    t0 = time.time()
    print("[per_question_extraction_accuracy] loading c6b JSON...", flush=True)
    src = load_c6b_perq_perdiff_4seed()

    # Build computed values (convert percent → fraction)
    point_by_cell: dict[tuple[str, str, str], float] = {}
    sigma_by_cell: dict[tuple[str, str, str], float] = {}
    delta_by_cell: dict[tuple[str, str], float] = {}
    for q in QUESTIONS:
        for d in DIFFS:
            for ext in EXTRACTORS:
                key = f"{q}__{d}__{ext}"
                if key not in src:
                    continue
                point_by_cell[(q, d, ext)] = src[key]["mean"] / 100.0
                sigma_by_cell[(q, d, ext)] = src[key]["std"] / 100.0
            if (q, d, "gpt") in point_by_cell and (q, d, "gem") in point_by_cell:
                delta_by_cell[(q, d)] = (
                    point_by_cell[(q, d, "gem")] - point_by_cell[(q, d, "gpt")]
                )

    # Overall rows
    overall_point: dict[tuple[str, str], float] = {}
    overall_sigma: dict[tuple[str, str], float] = {}
    overall_delta: dict[str, float] = {}
    for d in DIFFS:
        for ext in EXTRACTORS:
            key = f"_OVERALL__{d}__{ext}"
            if key in src:
                overall_point[(d, ext)] = src[key]["mean"] / 100.0
                overall_sigma[(d, ext)] = src[key]["std"] / 100.0
        if (d, "gpt") in overall_point and (d, "gem") in overall_point:
            overall_delta[d] = overall_point[(d, "gem")] - overall_point[(d, "gpt")]

    # Emit rows
    output_rows: list[dict] = []
    fail_count = skip_count = 0

    def push(row_id: str, label: str, mode: str, metric: str,
             point: float | None, paper_v: float | None, tol: float) -> None:
        nonlocal fail_count, skip_count
        if paper_v is None:
            skip_count += 1
            return
        r = emit_row(
            row_id=row_id, method_label=label, mode=mode, metric=metric,
            point=point, paper_point=paper_v, tolerance=tol,
        )
        output_rows.append(r)
        if r["paper_match"].startswith("FAIL"):
            fail_count += 1

    for q in QUESTIONS:
        for d in DIFFS:
            paper = PAPER_TAB_PER_QUESTION_EXTRACTION_ACCURACY.get((q, d))
            if paper is None:
                skip_count += 5  # 2 ext × 2 (point+sigma) + 1 delta
                continue
            for ext in EXTRACTORS:
                paper_mean, paper_sigma = paper[ext]
                push(f"{q}__{DIFF_SHORT[d]}__{ext}__point",
                     f"{q} :: {DIFF_SHORT[d]} :: {ext}",
                     mode="ext", metric="extraction_accuracy",
                     point=point_by_cell.get((q, d, ext)),
                     paper_v=paper_mean, tol=PAPER_TOLERANCE)
                push(f"{q}__{DIFF_SHORT[d]}__{ext}__sigma",
                     f"{q} :: {DIFF_SHORT[d]} :: {ext} σ",
                     mode="ext", metric="extraction_accuracy_sigma",
                     point=sigma_by_cell.get((q, d, ext)),
                     paper_v=paper_sigma, tol=SIGMA_TOLERANCE)
            push(f"{q}__{DIFF_SHORT[d]}__delta",
                 f"{q} :: {DIFF_SHORT[d]} :: Δ",
                 mode="delta", metric="extraction_accuracy_delta",
                 point=delta_by_cell.get((q, d)),
                 paper_v=paper["delta"], tol=DELTA_TOLERANCE)

    # Overall rows
    for d in DIFFS:
        for ext in EXTRACTORS:
            key = (d, ext)
            paper_v = PAPER_OVERALL.get(key)
            if paper_v is None:
                skip_count += 2; continue
            paper_mean, paper_sigma = paper_v
            push(f"OVERALL__{DIFF_SHORT[d]}__{ext}__point",
                 f"Overall :: {DIFF_SHORT[d]} :: {ext}",
                 mode="ext", metric="extraction_accuracy",
                 point=overall_point.get((d, ext)),
                 paper_v=paper_mean, tol=PAPER_TOLERANCE)
            push(f"OVERALL__{DIFF_SHORT[d]}__{ext}__sigma",
                 f"Overall :: {DIFF_SHORT[d]} :: {ext} σ",
                 mode="ext", metric="extraction_accuracy_sigma",
                 point=overall_sigma.get((d, ext)),
                 paper_v=paper_sigma, tol=SIGMA_TOLERANCE)
        push(f"OVERALL__{DIFF_SHORT[d]}__delta",
             f"Overall :: {DIFF_SHORT[d]} :: Δ",
             mode="delta", metric="extraction_accuracy_delta",
             point=overall_delta.get(d),
             paper_v=PAPER_OVERALL[(d, "delta")], tol=DELTA_TOLERANCE)

    # Markdown — compact per-Q × diff × ext table
    md_lines = ["| Q | Diff | GPT | Gem | Δ |", "|:---|:---:|:---:|:---:|:---:|"]
    for q in QUESTIONS:
        for d in DIFFS:
            gpt = point_by_cell.get((q, d, "gpt"))
            gem = point_by_cell.get((q, d, "gem"))
            delta = delta_by_cell.get((q, d))
            sg = sigma_by_cell.get((q, d, "gpt"))
            sm = sigma_by_cell.get((q, d, "gem"))
            md_lines.append(
                f"| {q} | {DIFF_SHORT[d]} | "
                f"{100*gpt:.1f}±{100*sg:.1f} | {100*gem:.1f}±{100*sm:.1f} | "
                f"{100*delta:+.1f} |"
            )
    md_lines.append("| | | | | |")
    for d in DIFFS:
        gp = overall_point.get((d, "gpt")); gm = overall_point.get((d, "gem"))
        sg = overall_sigma.get((d, "gpt")); sm = overall_sigma.get((d, "gem"))
        dl = overall_delta.get(d)
        md_lines.append(
            f"| **Overall** | **{DIFF_SHORT[d]}** | "
            f"{100*gp:.1f}±{100*sg:.1f} | {100*gm:.1f}±{100*sm:.1f} | "
            f"{100*dl:+.1f} |"
        )
    md_table = "\n".join(md_lines)

    csv_p, md_p = write_outputs(
        "per_question_extraction_accuracy",
        output_rows,
        md_table=md_table,
        script_name="paper_artifacts.appendix.per_question_extraction_accuracy",
        md_caption=(
            "**Per-Question Extraction Accuracy.** Per-question × per-difficulty extraction accuracy "
            "(%, 4-seed mean ± σ, 160 test personas per (Q, diff) cell). "
            "Δ = Gemini − GPT in pp."
        ),
        md_footnotes=[
            "*Paper location: `Appendix robustness section` (`tab:F6b`).*",
            f"*Tolerance: ±{PAPER_TOLERANCE} on point cells; ±{SIGMA_TOLERANCE} on σ; "
            f"±{DELTA_TOLERANCE} on Δ.*",
        ],
        subdir="appendix",
    )
    elapsed = time.time() - t0
    n_pass = len(output_rows) - fail_count
    print(
        f"[per_question_extraction_accuracy] OK={n_pass} FAIL={fail_count} SKIP={skip_count} "
        f"total={len(output_rows)} ({elapsed:.1f}s) -> {csv_p}, {md_p}"
    )
    return fail_count


if __name__ == "__main__":
    sys.exit(0 if main() == 0 else 1)
