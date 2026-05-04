"""Reproduce paper Per-Type Diagnostic Analysis — Per-type diagnostic analysis (oracle input).

Source: ``Experiments section`` lines 67-79
(``\\label{tab:6}``).

For each of the 8 reasoning types (sorted by paper DSNBF descending):

  - **# Q**: number of question ids in that reasoning type
  - **DSNBF** (μ* input, answer-only)
  - **GPT-μ***: GPT-5.4 ``StructLLMSource(mode="oracle")`` (answer-only)
  - **Δ_matched** = DSNBF − GPT-μ*, paired persona-clustered bootstrap
    (B=2,000, seed=42, percentile 95% CI in pp)
  - **Best LLM**: max accuracy among 8 LLM cells (4 families × 2 prompting
    regimes), each on NL input. Best cell tag (e.g. ``GPT-S``, ``Gem-D``)
    is reported alongside the numeric value.
  - **Oracle**: ``OracleExtraction(skip_on_miss=False)`` ceiling on oracle μ.

Reduction: pool-then-bootstrap with ``breakdown_key="reasoning_type"``
for marginal CIs; for paired Δ a custom inline bootstrap aligns
DSNBF and GPT-μ* by ``(seed, persona, qid)``.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict

import numpy as np

from survey2agent.evaluation.aggregator import aggregate_metrics
from survey2agent.evaluation.data_loaders import build_training_records
from survey2agent.evaluation.multi_seed import (
    aggregate_pooled_with_breakdown_ci,
    pool_results_across_seeds,
)
from survey2agent.evaluation.runner import EvaluationResult, run_method
from survey2agent.methods import (
    DSNBF,
    LLMDirect,
    LLMSchemaAware,
    OracleExtraction,
)

from .._common import (
    CANONICAL_SEEDS,
    PAPER_TOLERANCE,
    emit_row,
    load_shared_resources,
    run_llm_across_seeds,
    run_oracle_mode_across_seeds,
    run_struct_llm_across_seeds,
    write_outputs,
    _md_fmt_pct,
)


# ── Paper Per-Type Diagnostic Analysis numbers (s4_experiments.tex L70-78) ──────────────────


PAPER_TAB_PER_TYPE_ACCURACY = [
    # (type_key,            label,         n_q, dsnbf, str_orc, delta, ci_lo, ci_hi, best_llm_acc, best_llm_tag, oracle)
    ("identity",            "B · Ident",     2, 0.968, 0.970, -0.002, -0.010,  0.006, 0.945, "GPT-S",  0.997),
    ("arbitration",         "A · Arbit",     3, 0.875, 0.685,  0.190,  0.170,  0.212, 0.751, "Gem-S",  0.941),
    ("control",             "Ctrl",          2, 0.857, 0.757,  0.100,  0.069,  0.130, 0.791, "GPT-D",  0.960),
    ("causal",              "E · Factor",    2, 0.831, 0.706,  0.125,  0.096,  0.155, 0.709, "GPT-S",  0.950),
    ("trend",               "D · Temp",      2, 0.822, 0.762,  0.059,  0.034,  0.084, 0.725, "GPT-D",  0.900),
    ("plan_reality",        "C · P-R",       2, 0.816, 0.734,  0.081,  0.049,  0.114, 0.698, "GPT-D",  0.948),
    ("missing_data",        "F · Miss",      3, 0.779, 0.715,  0.065,  0.040,  0.088, 0.685, "GPT-S",  0.920),
    ("annotation",          "G · Annot",     2, 0.635, 0.585,  0.050,  0.008,  0.092, 0.499, "Gem-S",  0.837),
]

LLM_VARIANTS = [
    # (model, variant, family_short, regime_short, cls)
    ("gpt-5.4",                "direct",       "GPT", "D", LLMDirect),
    ("gpt-5.4",                "schema-aware", "GPT", "S", LLMSchemaAware),
    ("gemini_p2",              "direct",       "Gem", "D", LLMDirect),
    ("gemini_p2",              "schema-aware", "Gem", "S", LLMSchemaAware),
    ("deepseek-v3.2",          "direct",       "DSk", "D", LLMDirect),
    ("deepseek-v3.2",          "schema-aware", "DSk", "S", LLMSchemaAware),
    ("qwen3-235b-a22b-2507",   "direct",       "Qwn", "D", LLMDirect),
    ("qwen3-235b-a22b-2507",   "schema-aware", "Qwn", "S", LLMSchemaAware),
]


def _run_oracle_ext_per_type(*, oracle_atoms, gts, splits, diff_idx):
    out: dict[str, list[EvaluationResult]] = {}
    for seed in CANONICAL_SEEDS:
        atoms = oracle_atoms[seed]
        gt = gts[seed]
        evalr = build_training_records(atoms, gt, splits["test"], difficulty_index=diff_idx)
        method = OracleExtraction(skip_on_miss=False)
        method.attach_gt({pid: gt[pid] for pid in splits["test"] if pid in gt})
        out[seed] = run_method(method, evalr)
    return out


def _per_type_point(seed_results, *, breakdown_key="reasoning_type",
                    metric="forced_accuracy") -> dict[str, float]:
    """Return ``{type: point_value}`` from pool-then-aggregate (no CI)."""
    pooled = pool_results_across_seeds(seed_results)
    cells: dict[str, list[EvaluationResult]] = defaultdict(list)
    for r in pooled:
        cells[getattr(r, breakdown_key)].append(r)
    out: dict[str, float] = {}
    for cell, rows in cells.items():
        agg = aggregate_metrics(rows)
        out[cell] = float(agg[metric])
    return out


def _paired_delta_ci(
    dsnbf_seed_results,
    struct_seed_results,
    *,
    type_key: str,
    n_bootstrap: int = 2000,
    seed: int = 42,
    confidence: float = 0.95,
) -> dict[str, float]:
    """Paired persona-clustered bootstrap of (DSNBF − GPT-μ*) on a type subset.

    Pools both methods, indexes by (seed, persona). For each bootstrap
    iteration, resamples persona ids with replacement and computes
    forced_accuracy on each method's matching subset, then takes Δ.
    """
    dsnbf_pooled = pool_results_across_seeds(dsnbf_seed_results)
    struct_pooled = pool_results_across_seeds(struct_seed_results)

    # Filter to type
    d_filt = [r for r in dsnbf_pooled if r.reasoning_type == type_key]
    s_filt = [r for r in struct_pooled if r.reasoning_type == type_key]

    # Index by persona_id (already seed-prefixed by pool_results_across_seeds)
    d_by_p: dict[str, list[EvaluationResult]] = defaultdict(list)
    s_by_p: dict[str, list[EvaluationResult]] = defaultdict(list)
    for r in d_filt:
        d_by_p[r.persona_id].append(r)
    for r in s_filt:
        s_by_p[r.persona_id].append(r)

    # Personas present in both (typical: identical splits so all align)
    personas = sorted(set(d_by_p) & set(s_by_p))
    if not personas:
        return {"point": 0.0, "ci_low": 0.0, "ci_high": 0.0, "n_bootstrap": 0,
                "n_personas": 0}

    point_d = float(aggregate_metrics(d_filt)["forced_accuracy"])
    point_s = float(aggregate_metrics(s_filt)["forced_accuracy"])
    point_delta = point_d - point_s

    rng = np.random.default_rng(seed)
    n_p = len(personas)
    boot = np.empty(n_bootstrap, dtype=float)
    for b in range(n_bootstrap):
        idx = rng.integers(0, n_p, size=n_p)
        d_sample: list[EvaluationResult] = []
        s_sample: list[EvaluationResult] = []
        for i in idx:
            pid = personas[i]
            d_sample.extend(d_by_p[pid])
            s_sample.extend(s_by_p[pid])
        if not d_sample or not s_sample:
            boot[b] = point_delta
            continue
        ad = float(aggregate_metrics(d_sample)["forced_accuracy"])
        asr = float(aggregate_metrics(s_sample)["forced_accuracy"])
        boot[b] = ad - asr

    alpha = 1.0 - confidence
    return {
        "point": point_delta,
        "ci_low": float(np.quantile(boot, alpha / 2.0)),
        "ci_high": float(np.quantile(boot, 1.0 - alpha / 2.0)),
        "n_bootstrap": n_bootstrap,
        "n_personas": n_p,
    }


def main() -> int:
    t0 = time.time()
    splits, diff_idx, gts, oracle_atoms, llm_atoms = load_shared_resources()

    # ── Run all 4 method families once, then breakdown by type ─────
    print("[per_type_accuracy] running DSNBF (oracle)...", flush=True)
    sr_dsnbf = run_oracle_mode_across_seeds(
        DSNBF, oracle_atoms=oracle_atoms, gts=gts, splits=splits, diff_idx=diff_idx,
    )

    print("[per_type_accuracy] running Struct-LLM (oracle)...", flush=True)
    sr_struct = run_struct_llm_across_seeds(
        mode="oracle", display_name="GPT-μ* (direct-readout)",
        method_cls=LLMDirect, eval_atoms=oracle_atoms,
        gts=gts, splits=splits, diff_idx=diff_idx,
    )

    print("[per_type_accuracy] running Source Reachability...", flush=True)
    sr_oracle_ext = _run_oracle_ext_per_type(
        oracle_atoms=oracle_atoms, gts=gts, splits=splits, diff_idx=diff_idx,
    )

    # 8 LLM variants × per-type points (compute once each, take per-type max)
    llm_points_per_type: dict[str, list[tuple[float, str]]] = defaultdict(list)
    for model, variant, fam, reg, cls in LLM_VARIANTS:
        tag = f"{fam}-{reg}"
        print(f"[per_type_accuracy] running LLM {model} {variant}...", flush=True)
        sr = run_llm_across_seeds(
            model=model, variant=variant, display_name=f"{fam} {variant}",
            selective_cls=cls, llm_atoms=llm_atoms, gts=gts,
            splits=splits, diff_idx=diff_idx,
        )
        per_t = _per_type_point(sr)
        for t, p in per_t.items():
            llm_points_per_type[t].append((p, tag))

    # ── Per-type breakdowns ────────────────────────────────────────
    dsnbf_per_type = aggregate_pooled_with_breakdown_ci(
        sr_dsnbf, breakdown_key="reasoning_type", metric="forced_accuracy",
        n_bootstrap=2000, seed=42,
    )
    struct_per_type = aggregate_pooled_with_breakdown_ci(
        sr_struct, breakdown_key="reasoning_type", metric="forced_accuracy",
        n_bootstrap=2000, seed=42,
    )
    oracle_per_type = _per_type_point(sr_oracle_ext)

    output_rows: list[dict] = []
    fail_count = 0

    for spec in PAPER_TAB_PER_TYPE_ACCURACY:
        type_key, label, n_q, p_dsnbf, p_str, p_delta, p_lo, p_hi, p_best, p_best_tag, p_oracle = spec

        # DSNBF cell
        d_ci = dsnbf_per_type.get(type_key, {})
        row = emit_row(
            row_id=f"{type_key}__dsnbf", method_label=f"DSNBF :: {label}",
            mode="oracle", metric="forced_accuracy",
            point=d_ci.get("point"), ci_low=d_ci.get("ci_low"), ci_high=d_ci.get("ci_high"),
            n_personas=d_ci.get("n_personas_pooled", 0),
            paper_point=p_dsnbf,
        )
        output_rows.append(row)
        if row["paper_match"].startswith("FAIL"):
            fail_count += 1

        # GPT-μ* cell
        s_ci = struct_per_type.get(type_key, {})
        row = emit_row(
            row_id=f"{type_key}__str_orc", method_label=f"GPT-μ* :: {label}",
            mode="oracle", metric="forced_accuracy",
            point=s_ci.get("point"), ci_low=s_ci.get("ci_low"), ci_high=s_ci.get("ci_high"),
            n_personas=s_ci.get("n_personas_pooled", 0),
            paper_point=p_str,
        )
        output_rows.append(row)
        if row["paper_match"].startswith("FAIL"):
            fail_count += 1

        # Δ_matched cell (paired bootstrap CI)
        delta = _paired_delta_ci(sr_dsnbf, sr_struct, type_key=type_key)
        row = emit_row(
            row_id=f"{type_key}__delta", method_label=f"Δ_matched :: {label}",
            mode="oracle", metric="forced_accuracy_delta",
            point=delta["point"], ci_low=delta["ci_low"], ci_high=delta["ci_high"],
            n_personas=delta["n_personas"],
            paper_point=p_delta, paper_low=p_lo, paper_high=p_hi,
        )
        output_rows.append(row)
        if row["paper_match"].startswith("FAIL"):
            fail_count += 1

        # Best LLM cell
        candidates = llm_points_per_type.get(type_key, [])
        if candidates:
            best_acc, best_tag = max(candidates, key=lambda x: x[0])
        else:
            best_acc, best_tag = None, ""
        row = emit_row(
            row_id=f"{type_key}__best_llm",
            method_label=f"Best LLM ({best_tag}) :: {label}",
            mode="ext", metric="forced_accuracy",
            point=best_acc, n_personas="",
            paper_point=p_best,
        )
        output_rows.append(row)
        if row["paper_match"].startswith("FAIL"):
            fail_count += 1

        # Source Reachability cell
        row = emit_row(
            row_id=f"{type_key}__oracle",
            method_label=f"Source Reachability :: {label}",
            mode="oracle", metric="forced_accuracy",
            point=oracle_per_type.get(type_key), n_personas="",
            paper_point=p_oracle,
        )
        output_rows.append(row)
        if row["paper_match"].startswith("FAIL"):
            fail_count += 1

    # ── Render Markdown ───────────────────────────────────────────
    by_id = {r["row_id"]: r for r in output_rows}
    md_lines = [
        "| Type        | # Q | DSNBF | GPT-μ* | Δ_matched | 95% CI       | Best LLM       | Source Reachability |",
        "|:------------|:---:|:-----:|:-------:|:---------:|:------------:|:---------------|:-------------------:|",
    ]
    for spec in PAPER_TAB_PER_TYPE_ACCURACY:
        type_key, label, n_q, *_rest = spec
        d = by_id[f"{type_key}__dsnbf"]
        s = by_id[f"{type_key}__str_orc"]
        delta = by_id[f"{type_key}__delta"]
        bllm = by_id[f"{type_key}__best_llm"]
        ora = by_id[f"{type_key}__oracle"]
        delta_pp = (delta["point"] or 0.0) * 100
        ci_pp = f"[{(delta['ci_low'] or 0)*100:+.1f}, {(delta['ci_high'] or 0)*100:+.1f}]"
        # Best LLM tag is in method_label as "Best LLM (<tag>) :: ..."
        tag_part = bllm["method_label"].split("(", 1)[1].split(")", 1)[0] if "(" in bllm["method_label"] else ""
        bllm_str = f"{_md_fmt_pct(bllm['point'])} ({tag_part})"
        md_lines.append(
            f"| {label} | {n_q} | {_md_fmt_pct(d['point'])} | {_md_fmt_pct(s['point'])} | "
            f"{delta_pp:+.1f} | {ci_pp} | {bllm_str} | {_md_fmt_pct(ora['point'])} |"
        )
    md_table = "\n".join(md_lines)

    caption = (
        "**Per-Type Diagnostic Analysis.** Per-type accuracy (%, answer-only mode, 4-seed pooled). "
        "GPT-μ* = GPT-5.4 reading direct-readout μ* atoms as structured prompt input. "
        "Δ_matched = DSNBF − GPT-μ*, paired persona-clustered bootstrap "
        "(B=2,000, seed=42), 95% percentile CI in pp. Best LLM = max over "
        "4 LLM families × 2 prompting regimes (NL input). Source Reachability = "
        "GT-aided direct-readout reference (not deployable). Types ordered by paper DSNBF descending."
    )
    footnotes = [
        "*Paper location: `Experiments section` (`tab:6`).*",
        "*Reduction: per-type pool-then-bootstrap "
        "(`aggregate_pooled_with_breakdown_ci`, `breakdown_key=\"reasoning_type\"`, "
        "B=2000, seed=42); paired Δ via custom persona-clustered bootstrap.*",
        f"*Reproduction tolerance: ±{PAPER_TOLERANCE} absolute on per-type points "
        f"and Δ values.*",
    ]

    csv_p, md_p = write_outputs(
        "per_type_accuracy",
        output_rows,
        md_table=md_table,
        script_name="paper_artifacts.main.per_type_accuracy",
        md_caption=caption,
        md_footnotes=footnotes,
        subdir="main",
    )
    elapsed = time.time() - t0
    n_pass = len(output_rows) - fail_count
    print(
        f"[per_type_accuracy] wrote {len(output_rows)} cells "
        f"({n_pass} OK / {fail_count} FAIL) in {elapsed:.1f}s "
        f"-> {csv_p}, {md_p}"
    )
    return fail_count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    sys.exit(0 if main() == 0 else 1)
