"""Reproduce paper Prediction Distributions (E-type) — Prediction distributions (%) on E-type
failure questions (E1, E2) by difficulty class.

Source: ``Appendix diagnostics section`` lines 235-282
(``\\label{tab:E3}``).

This is a histogram of the model's ``answer`` field over personas in a
(question, difficulty) bucket — NOT a metric aggregate. For each
``(question_id, answer_label, difficulty)`` cell we count personas whose
prediction equals ``answer_label`` and divide by the total personas in
the difficulty bucket. An "Accuracy" row per (question, difficulty)
holds the macro accuracy.

Columns:
  GT     = ground-truth distribution from seed 1 only (40 / diff).
  GPT    = ``gpt-5.4`` Schema-Aware seed 1 only (40 / diff).
  DS     = ``deepseek-v3.2`` Schema-Aware 4-seed pooled (160 / diff).
  Qwen3  = ``qwen3-235b-a22b-2507`` Schema-Aware 4-seed pooled (160 / diff).

For the Accuracy row the GT column is vacuous (the row is "prediction
equals GT", which is 100% by construction); the paper marks it ``---``
(NaN) and so do we.

Paper convention: ``---`` marks "0% or missing answer key". We treat any
NaN in the paper truth as a SKIP cell for paper-lock purposes (we still
compute and emit the actual percentage, but do NOT validate against
NaN). Non-NaN cells are paper-locked at ±0.5 pp absolute (= the canonical
``PAPER_TOLERANCE = 0.005`` after the percent→fraction rescale).

Cell count: E1 has 4 answers × 3 diffs + 1 acc × 3 diffs = 15 rows;
E2 has 3 answers × 3 diffs + 1 acc × 3 diffs = 12 rows. Across 4
sources that is (15 + 12) × 4 = 108 cells. Of these, 18 are paper-NaN
(= SKIP_NaN); the remaining 90 are paper-locked.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from survey2agent.evaluation.data_loaders import (
    load_ground_truths,
    load_persona_difficulty_index,
    load_splits,
)
from survey2agent.evaluation.multi_seed import CANONICAL_SEEDS
from survey2agent.methods import FrozenBulkJSONSource

from .._common import OUTPUT_DIR, PAPER_TOLERANCE


# Tolerance on the percent scale (0-100) — matches the canonical
# unit-scale tolerance after percent→fraction rescale.
TOLERANCE_PCT: float = PAPER_TOLERANCE * 100.0


# ── Paper Prediction Distributions (E-type) numbers (appendix_e_diagnostics.tex L235-282) ────────────────
#
# Schema: PAPER_TAB_PREDICTION_DISTRIBUTIONS_E_CAUSAL[qid][answer_label] = {(diff_short, source): pct or NaN}
# Diff codes: "St" = stable, "TS" = temporal_shift, "SvR" = stated_vs_revealed.
# Sources: "GT", "GPT", "DS", "Qwen3".
# float('nan') marks paper "---" (0% or missing answer key).

NAN = float("nan")

PAPER_TAB_PREDICTION_DISTRIBUTIONS_E_CAUSAL: dict[str, dict[str, dict[tuple[str, str], float]]] = {
    "E1": {
        "no_single_factor": {
            ("St",  "GT"):    20.0, ("St",  "GPT"): 27.5, ("St",  "DS"): 90.0, ("St",  "Qwen3"): 54.4,
            ("TS",  "GT"):    25.0, ("TS",  "GPT"): 17.5, ("TS",  "DS"): 64.4, ("TS",  "Qwen3"): 36.9,
            ("SvR", "GT"):    92.5, ("SvR", "GPT"): 12.5, ("SvR", "DS"): 78.8, ("SvR", "Qwen3"): 20.0,
        },
        "no_late_nights": {
            ("St",  "GT"):    60.0, ("St",  "GPT"): 37.5, ("St",  "DS"): NAN,  ("St",  "Qwen3"): NAN,
            ("TS",  "GT"):    27.5, ("TS",  "GPT"): 25.0, ("TS",  "DS"): 0.6,  ("TS",  "Qwen3"): 0.6,
            ("SvR", "GT"):    NAN,  ("SvR", "GPT"): NAN,  ("SvR", "DS"): NAN,  ("SvR", "Qwen3"): NAN,
        },
        "social_activity": {
            ("St",  "GT"):    20.0, ("St",  "GPT"): 35.0, ("St",  "DS"): 1.9,  ("St",  "Qwen3"): NAN,
            ("TS",  "GT"):    27.5, ("TS",  "GPT"): 40.0, ("TS",  "DS"): 0.6,  ("TS",  "Qwen3"): NAN,
            ("SvR", "GT"):    5.0,  ("SvR", "GPT"): 87.5, ("SvR", "DS"): NAN,  ("SvR", "Qwen3"): NAN,
        },
        "work_activity": {
            ("St",  "GT"):    NAN,  ("St",  "GPT"): NAN,  ("St",  "DS"): 8.1,  ("St",  "Qwen3"): 45.6,
            ("TS",  "GT"):    20.0, ("TS",  "GPT"): 17.5, ("TS",  "DS"): 34.4, ("TS",  "Qwen3"): 62.5,
            ("SvR", "GT"):    2.5,  ("SvR", "GPT"): NAN,  ("SvR", "DS"): 21.2, ("SvR", "Qwen3"): 80.0,
        },
        "Accuracy": {
            ("St",  "GT"):    NAN,  ("St",  "GPT"): 62.5, ("St",  "DS"): 24.4, ("St",  "Qwen3"): 15.6,
            ("TS",  "GT"):    NAN,  ("TS",  "GPT"): 85.0, ("TS",  "DS"): 46.9, ("TS",  "Qwen3"): 33.1,
            ("SvR", "GT"):    NAN,  ("SvR", "GPT"): 17.5, ("SvR", "DS"): 75.0, ("SvR", "Qwen3"): 22.5,
        },
    },
    "E2": {
        "no_fewer_than_30": {
            ("St",  "GT"):    72.5, ("St",  "GPT"): 72.5, ("St",  "DS"): 93.8, ("St",  "Qwen3"): NAN,
            ("TS",  "GT"):    65.0, ("TS",  "GPT"): 70.0, ("TS",  "DS"): 72.5, ("TS",  "Qwen3"): NAN,
            ("SvR", "GT"):    90.0, ("SvR", "GPT"): 82.5, ("SvR", "DS"): 91.2, ("SvR", "Qwen3"): NAN,
        },
        "between_30_60": {
            ("St",  "GT"):    25.0, ("St",  "GPT"): 20.0, ("St",  "DS"): 3.8,  ("St",  "Qwen3"): 97.5,
            ("TS",  "GT"):    30.0, ("TS",  "GPT"): 30.0, ("TS",  "DS"): 3.1,  ("TS",  "Qwen3"): 74.4,
            ("SvR", "GT"):    10.0, ("SvR", "GPT"): 15.0, ("SvR", "DS"): 5.0,  ("SvR", "Qwen3"): 81.2,
        },
        "yes_more_than_60": {
            ("St",  "GT"):    2.5,  ("St",  "GPT"): 7.5,  ("St",  "DS"): 2.5,  ("St",  "Qwen3"): 2.5,
            ("TS",  "GT"):    5.0,  ("TS",  "GPT"): NAN,  ("TS",  "DS"): 24.4, ("TS",  "Qwen3"): 25.6,
            ("SvR", "GT"):    NAN,  ("SvR", "GPT"): 2.5,  ("SvR", "DS"): 3.8,  ("SvR", "Qwen3"): 18.8,
        },
        "Accuracy": {
            ("St",  "GT"):    NAN,  ("St",  "GPT"): 90.0, ("St",  "DS"): 68.8, ("St",  "Qwen3"): 26.2,
            ("TS",  "GT"):    NAN,  ("TS",  "GPT"): 87.5, ("TS",  "DS"): 60.0, ("TS",  "Qwen3"): 20.6,
            ("SvR", "GT"):    NAN,  ("SvR", "GPT"): 82.5, ("SvR", "DS"): 83.1, ("SvR", "Qwen3"): 5.0,
        },
    },
}


# Display order for answer rows per question (paper order).
ANSWER_ORDER: dict[str, list[str]] = {
    "E1": ["no_single_factor", "no_late_nights", "social_activity", "work_activity", "Accuracy"],
    "E2": ["no_fewer_than_30", "between_30_60", "yes_more_than_60", "Accuracy"],
}

DIFF_SHORT_TO_LONG: dict[str, str] = {
    "St":  "stable",
    "TS":  "temporal_shift",
    "SvR": "stated_vs_revealed",
}

# Source -> (model, seed_set, variant). seed_set is either ("s20260321",)
# for seed-1-only or CANONICAL_SEEDS for 4-seed pooled.
SOURCE_SPEC: dict[str, tuple[str, tuple[str, ...], str]] = {
    "GPT":   ("gpt-5.4",              ("s20260321",), "schema-aware"),
    "DS":    ("deepseek-v3.2",        CANONICAL_SEEDS, "schema-aware"),
    "Qwen3": ("qwen3-235b-a22b-2507", CANONICAL_SEEDS, "schema-aware"),
}

# Source order for column rendering and CSV emission (paper order).
SOURCE_ORDER: tuple[str, ...] = ("GT", "GPT", "DS", "Qwen3")


# ── Computation ──────────────────────────────────────────────────────────


def _diff_buckets(test_personas: list[str], diff_idx: dict[str, str]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {v: [] for v in DIFF_SHORT_TO_LONG.values()}
    for p in test_personas:
        out[diff_idx[p]].append(p)
    return out


def _gt_histogram(
    qid: str,
    answer_label: str,
    diff_long: str,
    *,
    diff_buckets: dict[str, list[str]],
    gt_seed1: dict[str, dict[str, str]],
) -> float:
    """GT distribution from seed 1 only (40 personas per diff)."""
    bucket = diff_buckets[diff_long]
    n = len(bucket)
    if n == 0:
        return 0.0
    matches = sum(1 for p in bucket if gt_seed1.get(p, {}).get(qid) == answer_label)
    return 100.0 * matches / n


def _gt_accuracy_is_vacuous() -> float:
    """GT 'Accuracy' row is vacuous (predicting GT against GT = 100%)."""
    return float("nan")


def _model_histogram(
    qid: str,
    answer_label: str,
    diff_long: str,
    *,
    sources_by_seed: dict[str, FrozenBulkJSONSource],
    diff_buckets: dict[str, list[str]],
) -> float:
    """Pooled histogram over all (seed, persona) pairs in the diff bucket.

    For seed-1-only sources ``sources_by_seed`` has one entry; for 4-seed
    pooled it has four. Personas in each diff bucket are the same across
    seeds (seed-stable persona ids), giving 40 × n_seeds total samples.
    """
    bucket = diff_buckets[diff_long]
    total = 0
    matches = 0
    for seed, src in sources_by_seed.items():
        for p in bucket:
            try:
                pred = src.get(p, qid).answer
            except (KeyError, FileNotFoundError):
                # Persona's bundle missing this qid — count toward total
                # but not toward matches. Should not happen for the test
                # split; surfaces immediately if it does.
                total += 1
                continue
            total += 1
            if pred == answer_label:
                matches += 1
    if total == 0:
        return 0.0
    return 100.0 * matches / total


def _model_accuracy(
    qid: str,
    diff_long: str,
    *,
    sources_by_seed: dict[str, FrozenBulkJSONSource],
    diff_buckets: dict[str, list[str]],
    gt_by_seed: dict[str, dict[str, dict[str, str]]],
) -> float:
    bucket = diff_buckets[diff_long]
    total = 0
    matches = 0
    for seed, src in sources_by_seed.items():
        gt_seed = gt_by_seed[seed]
        for p in bucket:
            gold = gt_seed.get(p, {}).get(qid)
            if gold is None:
                continue
            try:
                pred = src.get(p, qid).answer
            except (KeyError, FileNotFoundError):
                total += 1
                continue
            total += 1
            if pred == gold:
                matches += 1
    if total == 0:
        return 0.0
    return 100.0 * matches / total


# ── Output ────────────────────────────────────────────────────────────────


def _paper_match(actual_pct: float, paper_pct: float, *, tol: float) -> str:
    if math.isnan(paper_pct):
        return "SKIP_NaN"
    delta = abs(actual_pct - paper_pct)
    if delta <= tol:
        return "OK"
    return f"FAIL d={delta:.4f}"


def _fmt_pct(x: float) -> str:
    if math.isnan(x):
        return "—"
    return f"{x:.1f}"


def _render_md(rows: list[dict]) -> str:
    """Render Markdown table mirroring paper Prediction Distributions (E-type) layout."""
    lines: list[str] = []
    lines.append("| Q | Answer | Diff | GT | GPT | DS | Qwen3 |")
    lines.append("|:--|:-------|:----:|:--:|:---:|:--:|:-----:|")
    by_key: dict[tuple[str, str, str], dict[str, float]] = {}
    for r in rows:
        by_key.setdefault(
            (r["question_id"], r["answer_label"], r["difficulty"]), {}
        )[r["source"]] = r["pct_actual"]
    for qid in ("E1", "E2"):
        first_row_for_q = True
        for ans in ANSWER_ORDER[qid]:
            for diff_short, diff_long in DIFF_SHORT_TO_LONG.items():
                cells = by_key.get((qid, ans, diff_long), {})
                q_label = f"**{qid}**" if first_row_for_q else ""
                first_row_for_q = False
                ans_label = ans if ans != "Accuracy" else "*Accuracy*"
                cell_strs = [_fmt_pct(cells.get(s, float("nan"))) for s in SOURCE_ORDER]
                lines.append(
                    f"| {q_label} | {ans_label} | {diff_short} | "
                    + " | ".join(cell_strs) + " |"
                )
    return "\n".join(lines)


def _write_outputs(rows: list[dict], *, table_id: str, script_name: str) -> tuple[Path, Path]:
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
        "**Prediction Distributions (E-type).** Prediction distributions (%) on E-type failure questions "
        "(E1, E2) by difficulty class. GT = seed-1 ground-truth distribution "
        "(40 personas / diff). GPT = `gpt-5.4` Schema-Aware seed 1 (40 / diff). "
        "DS = `deepseek-v3.2` Schema-Aware 4-seed pooled (160 / diff). "
        "Qwen3 = `qwen3-235b-a22b-2507` Schema-Aware 4-seed pooled (160 / diff). "
        "`—` = paper marks `---` (0% or missing answer key); these cells are "
        "emitted with the actual computed pct but `paper_match=SKIP_NaN`."
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
        "*Paper location: `Appendix diagnostics section` (`tab:E3`).*",
    ]
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")
    return csv_path, md_path


def main() -> int:
    t0 = time.time()
    splits = load_splits()
    diff_idx = load_persona_difficulty_index()
    test_personas = splits["test"]
    diff_buckets = _diff_buckets(test_personas, diff_idx)

    # Sanity: 40 personas per diff (seed-1-only sources rely on this).
    for d, ps in diff_buckets.items():
        if len(ps) != 40:
            raise RuntimeError(
                f"unexpected diff bucket size for {d!r}: got {len(ps)}, expected 40"
            )

    # Load GT for every seed (DS/Qwen3 accuracy rows pool over 4 seeds).
    gt_by_seed: dict[str, dict[str, dict[str, str]]] = {
        s: load_ground_truths(s) for s in CANONICAL_SEEDS
    }
    gt_seed1 = gt_by_seed["s20260321"]

    # Build sources per source-tag.
    sources_per_tag: dict[str, dict[str, FrozenBulkJSONSource]] = {}
    for tag, (model, seed_set, variant) in SOURCE_SPEC.items():
        sources_per_tag[tag] = {
            s: FrozenBulkJSONSource(model=model, seed=s, variant=variant)
            for s in seed_set
        }

    rows: list[dict] = []

    for qid, by_answer in PAPER_TAB_PREDICTION_DISTRIBUTIONS_E_CAUSAL.items():
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
                        # Only the GT/Accuracy combo produces NaN actual.
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

    csv_p, md_p = _write_outputs(
        rows, table_id="prediction_distributions_e_causal",
        script_name="paper_artifacts.appendix.prediction_distributions_e_causal",
    )

    n_total = len(rows)
    n_pass = sum(1 for r in rows if r["paper_match"] == "OK")
    n_fail = sum(1 for r in rows if r["paper_match"].startswith("FAIL"))
    n_skip = sum(1 for r in rows if r["paper_match"] == "SKIP_NaN")
    elapsed = time.time() - t0
    print(
        f"[prediction_distributions_e_causal] wrote {n_total} cells "
        f"({n_pass} OK / {n_fail} FAIL / {n_skip} SKIP_NaN) "
        f"in {elapsed:.1f}s -> {csv_p}, {md_p}"
    )
    return n_fail


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    sys.exit(0 if main() == 0 else 1)
