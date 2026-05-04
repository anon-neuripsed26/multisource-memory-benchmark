"""Reproduce paper subsection "Cross-Parameter Transfer Without Refit" — DSNBF cross-bias transfer table.

Source: ``Appendix robustness section`` (subsection "Cross-Parameter Transfer Without Refit").

What this tests
---------------
The locked cross-bias robustness reference (``robustness_*.json``) retrains
each resolver inside each bias/dropout variant. That measures model-class
stability but does not test whether fitted parameters transfer across
source-projection settings. This script runs the missing transfer check:

  1. For each seed, fit and calibrate DSNBF once on the default variant
     ``b=1.0, d=1.0``.
  2. Rebuild the nine ``b x d`` source-projection variants in memory.
  3. Evaluate the default-fitted DSNBF on each shifted test split without
     refitting (``transfer`` arm).
  4. On the same rebuilt variant, fit a fresh target-variant DSNBF and
     compare no-refit transfer against that target-retrained score.

Both arms use the same rebuilt in-memory variant data so the comparison is
on identical test splits. All rows use direct-readout atoms ``μ*``.

Interpretation guardrail
------------------------
This is a distribution-shift transfer check. Strong transfer supports
cross-parameter stability; large negative gaps indicate that
target-distribution calibration data still helps. It is **not** evidence
that DSNBF reverse-engineers the DGP.

Lock contract
-------------
9 variants × 3 metrics (``transfer_mean``, ``retrained_mean``, ``gap_mean_pp``)
= 27 paper-lock cells against ``PAPER_TAB_CROSS_BIAS_TRANSFER``. Standard
deviations are emitted into the CSV/Markdown outputs but are not paper-locked
because their cross-seed stochastic spread exceeds the standard ±0.005
tolerance band (the paper itself reports σ at 0.1pp precision).

Runtime
-------
~9 minutes for the full 4-seed × 9-variant sweep on a laptop. Each variant
rebuilds source projections in memory and re-fits DSNBF (target-retrained
arm), so the cost scales with the number of variants times seeds.
"""

from __future__ import annotations

import json
import math
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from survey2agent._paths import persona_dir, seed_dir
from survey2agent.data_generation.ground_truth import compute_all_ground_truths
from survey2agent.data_generation.source_projector import _infer_knobs, project_all_sources
from survey2agent.evaluation.aggregator import aggregate_metrics
from survey2agent.evaluation.data_loaders import (
    TrainingRecord,
    build_training_records,
    load_persona_difficulty_index,
    load_splits,
)
from survey2agent.evaluation.runner import EvaluationResult
from survey2agent.extraction._mu_shell import compute_all_mu
from survey2agent.extraction.atoms import ExtractedAtom, _freeze_extraction
from survey2agent.methods import DSNBF
from survey2agent.selective.protocol import classify_prediction

from .._common import (
    CANONICAL_SEEDS,
    PAPER_TOLERANCE,
    emit_row,
    write_outputs,
)


# Paper numbers (fraction units, 4-seed mean) read directly from the paper
# table. Each variant has three locked cells: transfer mean, target-retrained
# mean, gap mean (in *pp*, transfer − target-retrained).
PAPER_TAB_CROSS_BIAS_TRANSFER: dict[tuple[str, str], float] = {
    ("b0.5_d0.5", "transfer"):    0.837, ("b0.5_d0.5", "retrained"):    0.844, ("b0.5_d0.5", "gap_pp"):    -0.7,
    ("b0.5_d1.0", "transfer"):    0.831, ("b0.5_d1.0", "retrained"):    0.831, ("b0.5_d1.0", "gap_pp"):     0.0,
    ("b0.5_d2.0", "transfer"):    0.816, ("b0.5_d2.0", "retrained"):    0.836, ("b0.5_d2.0", "gap_pp"):    -2.0,
    ("b1.0_d0.5", "transfer"):    0.829, ("b1.0_d0.5", "retrained"):    0.842, ("b1.0_d0.5", "gap_pp"):    -1.4,
    ("b1.0_d1.0", "transfer"):    0.823, ("b1.0_d1.0", "retrained"):    0.823, ("b1.0_d1.0", "gap_pp"):     0.0,
    ("b1.0_d2.0", "transfer"):    0.809, ("b1.0_d2.0", "retrained"):    0.814, ("b1.0_d2.0", "gap_pp"):    -0.5,
    ("b2.0_d0.5", "transfer"):    0.802, ("b2.0_d0.5", "retrained"):    0.840, ("b2.0_d0.5", "gap_pp"):    -3.8,
    ("b2.0_d1.0", "transfer"):    0.800, ("b2.0_d1.0", "retrained"):    0.824, ("b2.0_d1.0", "gap_pp"):    -2.4,
    ("b2.0_d2.0", "transfer"):    0.792, ("b2.0_d2.0", "retrained"):    0.808, ("b2.0_d2.0", "gap_pp"):    -1.6,
}

# Tolerance for the gap column (expressed in *pp*, not fraction). 0.5 pp
# matches the rounding precision of the paper table and the 4-seed σ
# reported alongside each cell.
GAP_TOLERANCE_PP: float = 0.5

VARIANTS: tuple[tuple[float, float], ...] = (
    (0.5, 0.5), (0.5, 1.0), (0.5, 2.0),
    (1.0, 0.5), (1.0, 1.0), (1.0, 2.0),
    (2.0, 0.5), (2.0, 1.0), (2.0, 2.0),
)
DEFAULT_VARIANT: tuple[float, float] = (1.0, 1.0)
SEEDS: tuple[str, ...] = tuple(CANONICAL_SEEDS)

BIAS_KNOBS: tuple[str, ...] = (
    "self_report_conflict_rate",
    "self_report_underreport_bias",
    "self_report_overreport_bias",
    "planner_optimism_bias",
    "planner_behavior_gap_rate",
)
DROPOUT_KNOBS: tuple[str, ...] = (
    "device_dropout_rate",
    "device_noise_rate",
    "objective_dropout_rate",
    "objective_noise_rate",
)

_HERE = Path(__file__).resolve()


@dataclass(frozen=True)
class _VariantData:
    atoms: dict[str, ExtractedAtom]
    gts: dict[str, dict[str, str]]


def _variant_key(bias_scale: float, dropout_scale: float) -> str:
    return f"b{bias_scale:.1f}_d{dropout_scale:.1f}"


def _seed_int(seed: str) -> int:
    if not seed.startswith("s"):
        raise ValueError(f"seed must look like 's20260321', got {seed!r}")
    return int(seed[1:])


def _load_personas(seed: str) -> list[dict[str, Any]]:
    path = seed_dir(seed) / "config" / "personas.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    return list(payload["personas"])


def _load_event_table(seed: str, persona_id: str) -> list[dict[str, Any]]:
    path = persona_dir(seed, persona_id) / "event_table.json"
    return list(json.loads(path.read_text(encoding="utf-8")))


def _scaled_knob_overrides(
    persona: Mapping[str, Any],
    records: Sequence[dict[str, Any]],
    *,
    bias_scale: float,
    dropout_scale: float,
) -> dict[str, Any] | None:
    if bias_scale == 1.0 and dropout_scale == 1.0:
        return None
    defaults = _infer_knobs(dict(persona), len(records))
    overrides: dict[str, Any] = {}
    for key in BIAS_KNOBS:
        if key in defaults:
            overrides[key] = min(1.0, defaults[key] * bias_scale)
    for key in DROPOUT_KNOBS:
        if key in defaults:
            overrides[key] = min(0.50, defaults[key] * dropout_scale)
    return overrides


def _source_records_for_gt(sources: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "planner": sources["planner"].get("records", []),
        "device_log": sources["device_log"].get("records", []),
        "objective_log": sources["objective_log"].get("records", []),
        "profile_ltm": sources["profile_ltm"],
    }


def _source_records_for_mu(sources: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "profile_ltm": sources["profile_ltm"],
        "planner": sources["planner"].get("records", []),
        "daily_self_report": sources["daily_self_report"].get("records", []),
        "objective_log": sources["objective_log"].get("records", []),
        "device_log": sources["device_log"].get("records", []),
    }


def _build_variant_data(seed: str, bias_scale: float, dropout_scale: float) -> _VariantData:
    base_seed = _seed_int(seed)
    atoms: dict[str, ExtractedAtom] = {}
    gts: dict[str, dict[str, str]] = {}
    for persona in _load_personas(seed):
        persona_id = persona["id"]
        events = _load_event_table(seed, persona_id)
        sources = project_all_sources(
            persona,
            events,
            base_seed=base_seed,
            knob_overrides=_scaled_knob_overrides(
                persona,
                events,
                bias_scale=bias_scale,
                dropout_scale=dropout_scale,
            ),
        )
        gt_full = compute_all_ground_truths(events, persona, _source_records_for_gt(sources))
        gts[persona_id] = {
            qid: entry["answer"]
            for qid, entry in gt_full.items()
            if isinstance(entry, dict) and isinstance(entry.get("answer"), str)
        }
        mu_all = compute_all_mu(_source_records_for_mu(sources))
        atoms[persona_id] = ExtractedAtom(
            persona=persona_id,
            extraction=_freeze_extraction(mu_all),
        )
    return _VariantData(atoms=atoms, gts=gts)


def _to_method_records(records: Sequence[TrainingRecord]) -> list[tuple]:
    grouped: dict[str, tuple[ExtractedAtom, dict[str, str]]] = {}
    for record in records:
        persona_id = record.atom.persona
        if persona_id not in grouped:
            grouped[persona_id] = (record.atom, {})
        grouped[persona_id][1][record.qid] = record.label
    return list(grouped.values())


def _fit_dsnbf(
    data: _VariantData,
    *,
    splits: Mapping[str, list[str]],
    difficulty_index: Mapping[str, str],
) -> DSNBF:
    train = build_training_records(
        data.atoms, data.gts, splits["train"], difficulty_index=difficulty_index,
    )
    cal = build_training_records(
        data.atoms, data.gts, splits["cal"], difficulty_index=difficulty_index,
    )
    method = DSNBF()
    method.fit(_to_method_records(train))
    method.calibrate(_to_method_records(cal))
    return method


def _evaluate(
    method: DSNBF,
    data: _VariantData,
    *,
    splits: Mapping[str, list[str]],
    difficulty_index: Mapping[str, str],
) -> float:
    # DSNBF caches inferred difficulty by persona id. Reset before each
    # variant because the same persona ids receive different shifted atoms.
    method._cached_persona = None  # noqa: SLF001
    method._diff_probs = None      # noqa: SLF001
    eval_records = build_training_records(
        data.atoms, data.gts, splits["test"], difficulty_index=difficulty_index,
    )
    results: list[EvaluationResult] = []
    for record in eval_records:
        pred = method.predict_one(record.atom, record.qid)
        results.append(
            EvaluationResult(
                method_name=method.name,
                persona_id=record.atom.persona,
                qid=record.qid,
                prediction=pred,
                label=record.label,
                outcome=classify_prediction(pred, record.label),
                reasoning_type=record.reasoning_type,
                topic=record.topic,
                difficulty_class=record.difficulty_class,
            )
        )
    return float(aggregate_metrics(results)["forced_accuracy"])


def _mean_std(values: Sequence[float]) -> tuple[float, float]:
    if not values:
        return math.nan, math.nan
    return statistics.mean(values), statistics.pstdev(values)


def main() -> int:
    t0 = time.time()
    splits = load_splits()

    # per_variant_per_seed[(variant_key, seed)] -> {"transfer", "retrained"}
    per_variant_per_seed: dict[tuple[str, str], dict[str, float]] = {}

    for seed in SEEDS:
        print(f"[cross_bias_transfer] seed {seed}: fitting default DSNBF", flush=True)
        difficulty_index = load_persona_difficulty_index(seed)
        default_data = _build_variant_data(seed, *DEFAULT_VARIANT)
        default_fit_method = _fit_dsnbf(
            default_data, splits=splits, difficulty_index=difficulty_index,
        )

        for bias_scale, dropout_scale in VARIANTS:
            key = _variant_key(bias_scale, dropout_scale)
            print(f"[cross_bias_transfer] seed {seed}: evaluating {key}", flush=True)
            data = (
                default_data
                if (bias_scale, dropout_scale) == DEFAULT_VARIANT
                else _build_variant_data(seed, bias_scale, dropout_scale)
            )
            transfer_acc = _evaluate(
                default_fit_method, data, splits=splits, difficulty_index=difficulty_index,
            )
            target_fit_method = _fit_dsnbf(
                data, splits=splits, difficulty_index=difficulty_index,
            )
            retrained_acc = _evaluate(
                target_fit_method, data, splits=splits, difficulty_index=difficulty_index,
            )
            per_variant_per_seed[(key, seed)] = {
                "transfer": transfer_acc,
                "retrained": retrained_acc,
            }

    # Aggregate across seeds and emit paper-lock rows.
    output_rows: list[dict[str, Any]] = []
    fail_count = 0
    skip_count = 0

    summary_by_variant: dict[str, dict[str, float]] = {}
    for bias_scale, dropout_scale in VARIANTS:
        key = _variant_key(bias_scale, dropout_scale)
        transfer_vals = [per_variant_per_seed[(key, s)]["transfer"] for s in SEEDS]
        retrained_vals = [per_variant_per_seed[(key, s)]["retrained"] for s in SEEDS]
        gap_pp_vals = [
            100.0 * (t - r) for t, r in zip(transfer_vals, retrained_vals, strict=True)
        ]
        transfer_mean, transfer_sd = _mean_std(transfer_vals)
        retrained_mean, retrained_sd = _mean_std(retrained_vals)
        gap_mean_pp, gap_sd_pp = _mean_std(gap_pp_vals)
        summary_by_variant[key] = {
            "transfer_mean": transfer_mean,
            "transfer_sd": transfer_sd,
            "retrained_mean": retrained_mean,
            "retrained_sd": retrained_sd,
            "gap_mean_pp": gap_mean_pp,
            "gap_sd_pp": gap_sd_pp,
        }

        # Emit three locked rows per variant.
        for col, point, paper_v, tol in (
            ("transfer", transfer_mean, PAPER_TAB_CROSS_BIAS_TRANSFER.get((key, "transfer")), PAPER_TOLERANCE),
            ("retrained", retrained_mean, PAPER_TAB_CROSS_BIAS_TRANSFER.get((key, "retrained")), PAPER_TOLERANCE),
            # gap is in pp units; tolerance is 0.5 pp -> 0.005 in fraction equivalent
            ("gap_pp", gap_mean_pp, PAPER_TAB_CROSS_BIAS_TRANSFER.get((key, "gap_pp")), GAP_TOLERANCE_PP),
        ):
            if paper_v is None:
                skip_count += 1
                continue
            r = emit_row(
                row_id=f"{key}__{col}",
                method_label=f"DSNBF :: {key} :: {col}",
                mode="oracle",
                metric=(
                    "macro_accuracy"
                    if col in ("transfer", "retrained")
                    else "macro_accuracy_gap_pp"
                ),
                point=point,
                paper_point=paper_v,
                tolerance=tol,
            )
            output_rows.append(r)
            if r["paper_match"].startswith("FAIL"):
                fail_count += 1

    # Build human-readable Markdown table mirroring the paper.
    md_lines = [
        "| Variant | Transfer | Target-retrained | Gap (pp) |",
        "|:---|:---:|:---:|:---:|",
    ]
    for bias_scale, dropout_scale in VARIANTS:
        key = _variant_key(bias_scale, dropout_scale)
        s = summary_by_variant[key]
        md_lines.append(
            f"| {key.replace('_', ', ')} | "
            f"{100*s['transfer_mean']:.1f} ± {100*s['transfer_sd']:.1f} | "
            f"{100*s['retrained_mean']:.1f} ± {100*s['retrained_sd']:.1f} | "
            f"{s['gap_mean_pp']:+.1f} ± {abs(s['gap_sd_pp']):.1f} |"
        )
    md_table = "\n".join(md_lines)

    csv_p, md_p = write_outputs(
        "cross_bias_transfer",
        output_rows,
        md_table=md_table,
        script_name="paper_artifacts.appendix.cross_bias_transfer",
        md_caption=(
            "**Cross-Parameter Transfer Without Refit.** DSNBF macro accuracy "
            "(%, 4-seed mean ± σ). Transfer = fit-and-calibrate once on the "
            "default ``b=1.0, d=1.0`` projection setting, then evaluate on the "
            "target variant without refit. Target-retrained = fit and calibrate "
            "on each target variant. Gap = transfer − target-retrained."
        ),
        md_footnotes=[
            "*Paper location: `Appendix robustness section` "
            "(subsection \"Cross-Parameter Transfer Without Refit\").*",
            f"*Tolerance: ±{PAPER_TOLERANCE} on transfer / target-retrained "
            f"means; ±{GAP_TOLERANCE_PP} pp on Gap.*",
        ],
        subdir="appendix",
    )
    elapsed = time.time() - t0
    n_pass = len(output_rows) - fail_count
    print(
        f"[cross_bias_transfer] OK={n_pass} FAIL={fail_count} SKIP={skip_count} "
        f"total={len(output_rows)} ({elapsed:.1f}s) -> {csv_p}, {md_p}"
    )
    return fail_count


if __name__ == "__main__":
    sys.exit(0 if main() == 0 else 1)
