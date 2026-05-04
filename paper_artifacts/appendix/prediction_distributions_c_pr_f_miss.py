"""Reproduce paper Prediction Distributions (C/F-type) — Prediction distributions (%) on C-type and
F-type failure questions (C2, F3) by difficulty class.

Source: ``Appendix diagnostics section`` lines 285-323
(``\\label{tab:E3b}``).

Same conventions, sources, and tolerance as
``prediction_distributions_e_causal``. See that module's docstring for
full semantics.

Cell count: C2 has 3 answers × 3 diffs + 1 acc × 3 diffs = 12 rows;
F3 has 3 answers × 3 diffs + 1 acc × 3 diffs = 12 rows. Across 4
sources that is 24 × 4 = 96 cells.
"""

from __future__ import annotations

import argparse
import sys
import time

from .._common import OUTPUT_DIR
from .prediction_distributions_e_causal import (
    ANSWER_ORDER as _E_ANSWER_ORDER,  # noqa: F401  (kept for parity / smoke import)
    DIFF_SHORT_TO_LONG,
    SOURCE_ORDER,
    SOURCE_SPEC,
    TOLERANCE_PCT,
    _diff_buckets,
    _gt_histogram,
    _model_accuracy,
    _model_histogram,
    _paper_match,
    _fmt_pct,
)
from .._common import PAPER_TOLERANCE
from survey2agent.evaluation.data_loaders import (
    load_ground_truths,
    load_persona_difficulty_index,
    load_splits,
)
from survey2agent.evaluation.multi_seed import CANONICAL_SEEDS
from survey2agent.methods import FrozenBulkJSONSource

import csv
import math
from datetime import datetime, timezone
from pathlib import Path


NAN = float("nan")


# ── Paper Prediction Distributions (C/F-type) numbers (appendix_e_diagnostics.tex L285-323) ───────────────

PAPER_TAB_PREDICTION_DISTRIBUTIONS_C_PR_F_MISS: dict[str, dict[str, dict[tuple[str, str], float]]] = {
    "C2": {
        "25_to_50_pct": {
            ("St",  "GT"):    57.5, ("St",  "GPT"): 55.0, ("St",  "DS"): 1.2,  ("St",  "Qwen3"): 98.1,
            ("TS",  "GT"):    35.0, ("TS",  "GPT"): 52.5, ("TS",  "DS"): NAN,  ("TS",  "Qwen3"): 98.8,
            ("SvR", "GT"):    47.5, ("SvR", "GPT"): 22.5, ("SvR", "DS"): 0.6,  ("SvR", "Qwen3"): 99.4,
        },
        "below_25_pct": {
            ("St",  "GT"):    30.0, ("St",  "GPT"): 10.0, ("St",  "DS"): 26.9, ("St",  "Qwen3"): 1.9,
            ("TS",  "GT"):    47.5, ("TS",  "GPT"): 17.5, ("TS",  "DS"): 41.9, ("TS",  "Qwen3"): 1.2,
            ("SvR", "GT"):    22.5, ("SvR", "GPT"): NAN,  ("SvR", "DS"): 37.5, ("SvR", "Qwen3"): 0.6,
        },
        "above_50_pct": {
            ("St",  "GT"):    12.5, ("St",  "GPT"): 35.0, ("St",  "DS"): 71.9, ("St",  "Qwen3"): NAN,
            ("TS",  "GT"):    17.5, ("TS",  "GPT"): 30.0, ("TS",  "DS"): 58.1, ("TS",  "Qwen3"): NAN,
            ("SvR", "GT"):    30.0, ("SvR", "GPT"): 77.5, ("SvR", "DS"): 61.9, ("SvR", "Qwen3"): NAN,
        },
        "Accuracy": {
            ("St",  "GT"):    NAN,  ("St",  "GPT"): 62.5, ("St",  "DS"): 20.0, ("St",  "Qwen3"): 50.0,
            ("TS",  "GT"):    NAN,  ("TS",  "GPT"): 62.5, ("TS",  "DS"): 33.1, ("TS",  "Qwen3"): 37.5,
            ("SvR", "GT"):    NAN,  ("SvR", "GPT"): 35.0, ("SvR", "DS"): 26.2, ("SvR", "Qwen3"): 55.6,
        },
    },
    "F3": {
        "both_occurred": {
            ("St",  "GT"):    57.5, ("St",  "GPT"): 70.0, ("St",  "DS"): 23.1, ("St",  "Qwen3"): NAN,
            ("TS",  "GT"):    70.0, ("TS",  "GPT"): 57.5, ("TS",  "DS"): 13.1, ("TS",  "Qwen3"): NAN,
            ("SvR", "GT"):    87.5, ("SvR", "GPT"): 80.0, ("SvR", "DS"): 47.5, ("SvR", "Qwen3"): NAN,
        },
        # Paper text abbreviates this label as "yes_worked"; canonical
        # data label (matches GT and frozen-LLM artifacts) is
        # "yes_worked_despite_no_entry".
        "yes_worked_despite_no_entry": {
            ("St",  "GT"):    30.0, ("St",  "GPT"): 30.0, ("St",  "DS"): 11.9, ("St",  "Qwen3"): 100.0,
            ("TS",  "GT"):    27.5, ("TS",  "GPT"): 42.5, ("TS",  "DS"): 44.4, ("TS",  "Qwen3"): 100.0,
            ("SvR", "GT"):    12.5, ("SvR", "GPT"): 20.0, ("SvR", "DS"): 44.4, ("SvR", "Qwen3"): 100.0,
        },
        "truly_off": {
            ("St",  "GT"):    12.5, ("St",  "GPT"): NAN,  ("St",  "DS"): 65.0, ("St",  "Qwen3"): NAN,
            ("TS",  "GT"):    2.5,  ("TS",  "GPT"): NAN,  ("TS",  "DS"): 42.5, ("TS",  "Qwen3"): NAN,
            ("SvR", "GT"):    NAN,  ("SvR", "GPT"): NAN,  ("SvR", "DS"): 8.1,  ("SvR", "Qwen3"): NAN,
        },
        "Accuracy": {
            ("St",  "GT"):    NAN,  ("St",  "GPT"): 62.5, ("St",  "DS"): 28.7, ("St",  "Qwen3"): 34.4,
            ("TS",  "GT"):    NAN,  ("TS",  "GPT"): 72.5, ("TS",  "DS"): 25.6, ("TS",  "Qwen3"): 26.2,
            ("SvR", "GT"):    NAN,  ("SvR", "GPT"): 87.5, ("SvR", "DS"): 46.9, ("SvR", "Qwen3"): 13.8,
        },
    },
}


ANSWER_ORDER: dict[str, list[str]] = {
    "C2": ["25_to_50_pct", "below_25_pct", "above_50_pct", "Accuracy"],
    "F3": ["both_occurred", "yes_worked_despite_no_entry", "truly_off", "Accuracy"],
}

# Display alias: paper abbreviates "yes_worked_despite_no_entry" as
# "yes_worked" to fit the table column. Canonical data label is the long
# form; we display the short form to mirror paper Prediction Distributions (C/F-type).
DISPLAY_LABEL: dict[str, str] = {
    "yes_worked_despite_no_entry": "yes_worked",
}


def _render_md(rows: list[dict]) -> str:
    lines: list[str] = []
    lines.append("| Q | Answer | Diff | GT | GPT | DS | Qwen3 |")
    lines.append("|:--|:-------|:----:|:--:|:---:|:--:|:-----:|")
    by_key: dict[tuple[str, str, str], dict[str, float]] = {}
    for r in rows:
        by_key.setdefault(
            (r["question_id"], r["answer_label"], r["difficulty"]), {}
        )[r["source"]] = r["pct_actual"]
    for qid in ("C2", "F3"):
        first_row_for_q = True
        for ans in ANSWER_ORDER[qid]:
            for diff_short, diff_long in DIFF_SHORT_TO_LONG.items():
                cells = by_key.get((qid, ans, diff_long), {})
                q_label = f"**{qid}**" if first_row_for_q else ""
                first_row_for_q = False
                ans_label = (
                    "*Accuracy*" if ans == "Accuracy"
                    else DISPLAY_LABEL.get(ans, ans)
                )
                cell_strs = [_fmt_pct(cells.get(s, float("nan"))) for s in SOURCE_ORDER]
                lines.append(
                    f"| {q_label} | {ans_label} | {diff_short} | "
                    + " | ".join(cell_strs) + " |"
                )
    return "\n".join(lines)


def _write_outputs(rows: list[dict]) -> tuple[Path, Path]:
    table_id = "prediction_distributions_c_pr_f_miss"
    script_name = "paper_artifacts.appendix.prediction_distributions_c_pr_f_miss"
    out_dir = OUTPUT_DIR / "appendix"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"{table_id}.csv"
    md_path = out_dir / f"{table_id}.md"

    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    header = (
        f"# Generated by {script_name} at {ts}\n"
        f"# Tolerance: ±{TOLERANCE_PCT:.3f} pp absolute (= ±{PAPER_TOLERANCE} on unit scale)\n"
        f"# Frozen artifacts: data/method_outputs/(gpt-5.4|deepseek-v3.2|qwen3-235b-a22b-2507)/"
        "(s20260321..s20260324)/schema-aware/\n"
    )

    cols = (
        "row_id", "question_id", "answer_label", "difficulty",
        "source", "pct_actual", "pct_paper", "paper_match",
    )
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        for line in header.splitlines():
            fh.write(line + "\n")
        writer = csv.DictWriter(fh, fieldnames=list(cols), extrasaction="ignore")
        writer.writeheader()
        for r in rows:
            row_out = {
                "row_id": r["row_id"],
                "question_id": r["question_id"],
                "answer_label": r["answer_label"],
                "difficulty": r["difficulty"],
                "source": r["source"],
                "pct_actual": "" if math.isnan(r["pct_actual"]) else f"{r['pct_actual']:.4f}",
                "pct_paper":  "" if math.isnan(r["pct_paper"])  else f"{r['pct_paper']:.4f}",
                "paper_match": r["paper_match"],
            }
            writer.writerow(row_out)

    md_table = _render_md(rows)
    n_total = len(rows)
    n_pass = sum(1 for r in rows if r["paper_match"] == "OK")
    n_fail = sum(1 for r in rows if r["paper_match"].startswith("FAIL"))
    n_skip = sum(1 for r in rows if r["paper_match"] == "SKIP_NaN")
    caption = (
        "**Prediction Distributions (C/F-type).** Prediction distributions (%) on C-type and F-type "
        "failure questions (C2, F3) by difficulty class. Same conventions as "
        "Prediction Distributions (E-type) (GT seed-1 / GPT seed-1 / DS, Qwen3 4-seed pooled, all "
        "Schema-Aware). `—` = paper marks `---` (0% or missing answer key); "
        "`paper_match=SKIP_NaN`."
    )
    md_lines = [
        header.rstrip(),
        "",
        f"# {table_id}",
        "",
        caption,
        "",
        md_table,
        "",
        f"*{n_total} cells: {n_pass} OK, {n_fail} FAIL, {n_skip} SKIP_NaN.*",
        "",
        "*Paper location: `Appendix diagnostics section` (`tab:E3b`).*",
    ]
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    return csv_path, md_path


def main() -> int:
    t0 = time.time()
    splits = load_splits()
    diff_idx = load_persona_difficulty_index()
    test_personas = splits["test"]
    diff_buckets = _diff_buckets(test_personas, diff_idx)
    for d, ps in diff_buckets.items():
        if len(ps) != 40:
            raise RuntimeError(
                f"unexpected diff bucket size for {d!r}: got {len(ps)}, expected 40"
            )

    gt_by_seed: dict[str, dict[str, dict[str, str]]] = {
        s: load_ground_truths(s) for s in CANONICAL_SEEDS
    }
    gt_seed1 = gt_by_seed["s20260321"]

    sources_per_tag: dict[str, dict[str, FrozenBulkJSONSource]] = {}
    for tag, (model, seed_set, variant) in SOURCE_SPEC.items():
        sources_per_tag[tag] = {
            s: FrozenBulkJSONSource(model=model, seed=s, variant=variant)
            for s in seed_set
        }

    rows: list[dict] = []
    for qid, by_answer in PAPER_TAB_PREDICTION_DISTRIBUTIONS_C_PR_F_MISS.items():
        for answer_label in ANSWER_ORDER[qid]:
            paper_diffs = by_answer[answer_label]
            for diff_short, diff_long in DIFF_SHORT_TO_LONG.items():
                for source in SOURCE_ORDER:
                    paper_pct = paper_diffs[(diff_short, source)]
                    if source == "GT":
                        if answer_label == "Accuracy":
                            actual = float("nan")
                        else:
                            actual = _gt_histogram(
                                qid, answer_label, diff_long,
                                diff_buckets=diff_buckets, gt_seed1=gt_seed1,
                            )
                    else:
                        srcs = sources_per_tag[source]
                        if answer_label == "Accuracy":
                            actual = _model_accuracy(
                                qid, diff_long,
                                sources_by_seed=srcs,
                                diff_buckets=diff_buckets,
                                gt_by_seed=gt_by_seed,
                            )
                        else:
                            actual = _model_histogram(
                                qid, answer_label, diff_long,
                                sources_by_seed=srcs,
                                diff_buckets=diff_buckets,
                            )
                    if math.isnan(actual):
                        match = "SKIP_NaN" if math.isnan(paper_pct) else "FAIL d=NaN"
                    else:
                        match = _paper_match(actual, paper_pct, tol=TOLERANCE_PCT)
                    rows.append({
                        "row_id":       f"{qid}__{answer_label}__{diff_short}__{source}",
                        "question_id":  qid,
                        "answer_label": answer_label,
                        "difficulty":   diff_long,
                        "source":       source,
                        "pct_actual":   actual,
                        "pct_paper":    paper_pct,
                        "paper_match":  match,
                    })

    csv_p, md_p = _write_outputs(rows)
    n_total = len(rows)
    n_pass = sum(1 for r in rows if r["paper_match"] == "OK")
    n_fail = sum(1 for r in rows if r["paper_match"].startswith("FAIL"))
    n_skip = sum(1 for r in rows if r["paper_match"] == "SKIP_NaN")
    elapsed = time.time() - t0
    print(
        f"[prediction_distributions_c_pr_f_miss] wrote {n_total} cells "
        f"({n_pass} OK / {n_fail} FAIL / {n_skip} SKIP_NaN) "
        f"in {elapsed:.1f}s -> {csv_p}, {md_p}"
    )
    return n_fail


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    sys.exit(0 if main() == 0 else 1)
