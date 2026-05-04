"""Reproduce paper Cross-Seed Stability — Cross-seed stability (per-seed macro accuracy).

Source: ``Appendix robustness section`` (lines 1-25).

10 rows × 5 columns (4 per-seed point cells + 1 σ cell = 50 paper-lock
cells total):

  Method          | seed1 | seed2 | seed3 | seed4 | σ

T1/T2 fusion rows (DSNBF, NBF) read ``DSNBF-NoSkip`` / ``NBF-NoSkip``
from ``4seed_results.json`` — these are the no-skip variants whose
``overall_macro`` is the column the paper labels just "DSNBF" / "NBF".

LLM rows:
  * GPT-5.4 Direct/Schema  -> ``bootstrap_ci_4seed.json.per_seed[sd]
                              [{'LLM-Direct','Schema-Aware-Direct'}].point_estimate``
  * DeepSeek/Qwen Direct/Schema  -> ``openweight_4seed_eval.json``
  * Gemini Direct/Schema  -> computed via ``run_llm_across_seeds``
                            (FrozenBulkJSONSource('gemini_p2', ...))
                            because no aggregated JSON contains them
                            (the ``llm`` block of full_comparison_gemini_*
                            is a copy of GPT values, not Gemini).

σ uses population stddev (divisor N) — matches paper rounding for most
rows; emit with ±0.002 absolute tolerance.
"""

from __future__ import annotations

import sys
import time
from collections import defaultdict

from survey2agent.evaluation.aggregator import aggregate_metrics
from survey2agent.evaluation.data_loaders import (
    build_training_records,
    load_atoms_for_seed,
    load_ground_truths,
    load_persona_difficulty_index,
    load_splits,
)
from survey2agent.methods import FrozenBulkJSONSource, LLMDirect, LLMSchemaAware
from survey2agent.evaluation.runner import run_method

from .._common import emit_row, write_outputs, _md_fmt_pct, PAPER_TOLERANCE
from .._appendix_helpers import (
    CANONICAL_SEED_KEYS,
    CANONICAL_SEED_STRS,
    PAPER_LABEL_TO_JSON_KEY,
    load_4seed_results,
    load_bootstrap_ci,
    load_openweight_4seed,
    mean_std,
)


# (row_label, seed_idx_or_sigma) -> paper value (fraction)
PAPER_TAB_CROSS_SEED_STABILITY: dict[tuple[str, str], float] = {
    # DSNBF
    ("DSNBF", "seed1"): 0.830, ("DSNBF", "seed2"): 0.814,
    ("DSNBF", "seed3"): 0.828, ("DSNBF", "seed4"): 0.822, ("DSNBF", "sigma"): 0.006,
    # NBF
    ("NBF", "seed1"): 0.829, ("NBF", "seed2"): 0.810,
    ("NBF", "seed3"): 0.817, ("NBF", "seed4"): 0.826, ("NBF", "sigma"): 0.008,
    # GPT-5.4 Direct
    ("GPT-5.4 Direct", "seed1"): 0.689, ("GPT-5.4 Direct", "seed2"): 0.693,
    ("GPT-5.4 Direct", "seed3"): 0.676, ("GPT-5.4 Direct", "seed4"): 0.686,
    ("GPT-5.4 Direct", "sigma"): 0.006,
    # GPT-5.4 Schema
    ("GPT-5.4 Schema", "seed1"): 0.694, ("GPT-5.4 Schema", "seed2"): 0.702,
    ("GPT-5.4 Schema", "seed3"): 0.689, ("GPT-5.4 Schema", "seed4"): 0.702,
    ("GPT-5.4 Schema", "sigma"): 0.006,
    # Gemini Direct
    ("Gemini Direct", "seed1"): 0.681, ("Gemini Direct", "seed2"): 0.673,
    ("Gemini Direct", "seed3"): 0.669, ("Gemini Direct", "seed4"): 0.690,
    ("Gemini Direct", "sigma"): 0.009,
    # Gemini Schema
    ("Gemini Schema", "seed1"): 0.700, ("Gemini Schema", "seed2"): 0.706,
    ("Gemini Schema", "seed3"): 0.690, ("Gemini Schema", "seed4"): 0.705,
    ("Gemini Schema", "sigma"): 0.006,
    # DeepSeek Direct
    ("DeepSeek Direct", "seed1"): 0.557, ("DeepSeek Direct", "seed2"): 0.571,
    ("DeepSeek Direct", "seed3"): 0.556, ("DeepSeek Direct", "seed4"): 0.559,
    ("DeepSeek Direct", "sigma"): 0.006,
    # DeepSeek Schema
    ("DeepSeek Schema", "seed1"): 0.584, ("DeepSeek Schema", "seed2"): 0.571,
    ("DeepSeek Schema", "seed3"): 0.556, ("DeepSeek Schema", "seed4"): 0.565,
    ("DeepSeek Schema", "sigma"): 0.011,
    # Qwen3 Direct
    ("Qwen3 Direct", "seed1"): 0.416, ("Qwen3 Direct", "seed2"): 0.414,
    ("Qwen3 Direct", "seed3"): 0.424, ("Qwen3 Direct", "seed4"): 0.429,
    ("Qwen3 Direct", "sigma"): 0.006,
    # Qwen3 Schema
    ("Qwen3 Schema", "seed1"): 0.473, ("Qwen3 Schema", "seed2"): 0.487,
    ("Qwen3 Schema", "seed3"): 0.478, ("Qwen3 Schema", "seed4"): 0.481,
    ("Qwen3 Schema", "sigma"): 0.005,
}

ROW_ORDER: list[str] = [
    "DSNBF", "NBF",
    "GPT-5.4 Direct", "GPT-5.4 Schema",
    "Gemini Direct", "Gemini Schema",
    "DeepSeek Direct", "DeepSeek Schema",
    "Qwen3 Direct", "Qwen3 Schema",
]
COL_ORDER: list[str] = ["seed1", "seed2", "seed3", "seed4", "sigma"]
SIGMA_TOLERANCE: float = 0.002


def _compute_gemini_per_seed(variant: str, *, splits, diff_idx, gts, llm_atoms) -> list[float]:
    """Return per-seed forced macro accuracy for Gemini under given variant.

    ``variant`` ∈ {"direct", "schema-aware"}. Uses LLMDirect for both
    (forced_accuracy macro is invariant to the selective wrapper as long
    as no skip is emitted, which is true for both Direct and
    Schema-Aware Gemini bulk artifacts).
    """
    cls = LLMDirect if variant == "direct" else LLMSchemaAware
    out: list[float] = []
    for seed in CANONICAL_SEED_STRS:
        atoms = llm_atoms[seed]
        gt = gts[seed]
        evalr = build_training_records(atoms, gt, splits["test"], difficulty_index=diff_idx)
        method = cls(
            source=FrozenBulkJSONSource(model="gemini_p2", seed=seed, variant=variant),
            model_display_name=f"Gem-{variant}",
        )
        results = run_method(method, evalr)
        out.append(aggregate_metrics(results)["forced_accuracy"])
    return out


def _per_seed_from_4seed(json_key: str, fs: dict) -> list[float]:
    return [fs["per_seed"][k][json_key]["overall_macro"] for k in CANONICAL_SEED_KEYS]


def _per_seed_from_bootstrap(name: str, b: dict) -> list[float]:
    return [b["per_seed"][k][name]["point_estimate"] for k in CANONICAL_SEED_KEYS]


def _per_seed_from_openweight(name: str, ow: dict) -> list[float]:
    return [ow[name]["per_seed"][k]["macro_accuracy"] / 100.0 for k in CANONICAL_SEED_STRS]


def main() -> int:
    t0 = time.time()
    print("[cross_seed_stability] loading static JSON...", flush=True)
    fs = load_4seed_results()
    boot = load_bootstrap_ci()
    ow = load_openweight_4seed()

    point_by_cell: dict[tuple[str, str], float] = {}
    sigma_by_row: dict[str, float] = {}

    # --- Static-JSON rows ---------------------------------------------------
    static_rows: list[tuple[str, list[float]]] = [
        ("DSNBF",            _per_seed_from_4seed(PAPER_LABEL_TO_JSON_KEY["DSNBF"], fs)),
        ("NBF",              _per_seed_from_4seed(PAPER_LABEL_TO_JSON_KEY["NBF"], fs)),
        ("GPT-5.4 Direct",   _per_seed_from_bootstrap("LLM-Direct", boot)),
        ("GPT-5.4 Schema",   _per_seed_from_bootstrap("Schema-Aware-Direct", boot)),
        ("DeepSeek Direct",  _per_seed_from_openweight("DeepSeek-V3.2-Direct", ow)),
        ("DeepSeek Schema",  _per_seed_from_openweight("DeepSeek-V3.2-Schema", ow)),
        ("Qwen3 Direct",     _per_seed_from_openweight("Qwen3-235B-Direct", ow)),
        ("Qwen3 Schema",     _per_seed_from_openweight("Qwen3-235B-Schema", ow)),
    ]
    for label, vals in static_rows:
        for ci, val in enumerate(vals):
            point_by_cell[(label, f"seed{ci+1}")] = val
        _, sigma = mean_std(vals)
        sigma_by_row[label] = sigma

    # --- Runner-computed Gemini rows ----------------------------------------
    print("[cross_seed_stability] loading atoms for Gemini computation...", flush=True)
    splits = load_splits()
    diff_idx = load_persona_difficulty_index()
    gts = {s: load_ground_truths(s) for s in CANONICAL_SEED_STRS}
    llm_atoms = {s: load_atoms_for_seed(s, mode="llm") for s in CANONICAL_SEED_STRS}

    for label, variant in (("Gemini Direct", "direct"), ("Gemini Schema", "schema-aware")):
        print(f"[cross_seed_stability] computing {label}...", flush=True)
        vals = _compute_gemini_per_seed(
            variant, splits=splits, diff_idx=diff_idx, gts=gts, llm_atoms=llm_atoms,
        )
        for ci, val in enumerate(vals):
            point_by_cell[(label, f"seed{ci+1}")] = val
        _, sigma = mean_std(vals)
        sigma_by_row[label] = sigma

    # --- Emit cells ---------------------------------------------------------
    output_rows: list[dict] = []
    fail_count = skip_count = 0
    for row in ROW_ORDER:
        for col in COL_ORDER:
            paper_v = PAPER_TAB_CROSS_SEED_STABILITY.get((row, col))
            if col == "sigma":
                point = sigma_by_row.get(row)
                tol = SIGMA_TOLERANCE
            else:
                point = point_by_cell.get((row, col))
                tol = PAPER_TOLERANCE
            if paper_v is None:
                skip_count += 1
                continue
            r = emit_row(
                row_id=f"{row}__{col}",
                method_label=f"{row} :: {col}",
                mode="oracle" if row in {"DSNBF", "NBF"} else "ext",
                metric="macro_accuracy" if col != "sigma" else "macro_accuracy_sigma",
                point=point,
                paper_point=paper_v,
                tolerance=tol,
            )
            output_rows.append(r)
            if r["paper_match"].startswith("FAIL"):
                fail_count += 1

    # --- Markdown ----------------------------------------------------------
    md_lines = [
        "| Method | " + " | ".join(COL_ORDER) + " |",
        "|:---|" + "|".join([":---:"] * len(COL_ORDER)) + "|",
    ]
    for row in ROW_ORDER:
        cells = []
        for col in COL_ORDER:
            if col == "sigma":
                v = sigma_by_row.get(row)
                cells.append(f"{100.0 * v:.1f}" if v is not None else "—")
            else:
                v = point_by_cell.get((row, col))
                cells.append(_md_fmt_pct(v))
        md_lines.append(f"| {row} | " + " | ".join(cells) + " |")
    md_table = "\n".join(md_lines)

    csv_p, md_p = write_outputs(
        "cross_seed_stability",
        output_rows,
        md_table=md_table,
        script_name="paper_artifacts.appendix.cross_seed_stability",
        md_caption=(
            "**Cross-Seed Stability.** Per-seed macro accuracy (%) for 10 methods. "
            "Fusion rows use oracle μ; LLM rows use NL input. "
            "σ is population stddev across the 4 seeds."
        ),
        md_footnotes=[
            "*Paper location: `Appendix robustness section` "
            "(`tab:F1`).*",
            f"*Tolerance: ±{PAPER_TOLERANCE} on point cells; ±{SIGMA_TOLERANCE} on σ cells.*",
        ],
        subdir="appendix",
    )
    elapsed = time.time() - t0
    n_pass = len(output_rows) - fail_count
    print(
        f"[cross_seed_stability] OK={n_pass} FAIL={fail_count} SKIP={skip_count} "
        f"total={len(output_rows)} ({elapsed:.1f}s) -> {csv_p}, {md_p}"
    )
    return fail_count


if __name__ == "__main__":
    sys.exit(0 if main() == 0 else 1)
