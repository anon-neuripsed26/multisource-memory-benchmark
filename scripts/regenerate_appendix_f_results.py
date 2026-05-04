"""Regenerate deterministic Appendix-F result JSONs from the released code.

This script replaces the legacy prototype-only robustness/ablation runners
for the deterministic, non-LLM parts of Appendix F. It uses only public
release modules and the released benchmark/source artifacts:

* train-size ablation and oracle-mu noise perturbation
* source-projection DGP perturbation
* four-seed oracle/extracted fusion summaries used by cross-seed and
  cross-extractor appendix tables
* per-question, per-difficulty extraction accuracy for GPT and Gemini

No LLM APIs are called. GPT-extracted atoms are read from
``data/extracted_atoms``; Gemini-extracted atoms are read from
``data/method_outputs/gemini_p2/{seed}/extract``; oracle/direct-readout
atoms are rebuilt deterministically from ``data/benchmark/seeds``.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import statistics
import time
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations, product
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from scipy.stats import kendalltau

from survey2agent._paths import METHOD_OUTPUTS_ROOT, RESULTS_ROOT, persona_dir, seed_dir
from survey2agent.data_generation.ground_truth import compute_all_ground_truths
from survey2agent.data_generation.source_projector import _infer_knobs, project_all_sources
from survey2agent.evaluation.aggregator import aggregate_metrics
from survey2agent.evaluation.data_loaders import (
    TrainingRecord,
    build_training_records,
    load_atoms_for_seed,
    load_ground_truths,
    load_persona_difficulty_index,
    load_splits,
)
from survey2agent.evaluation.runner import EvaluationResult, run_method
from survey2agent.extraction._mu_shell import compute_all_mu
from survey2agent.extraction.atoms import (
    EXPECTED_QUESTION_IDS,
    ExtractedAtom,
    EXPECTED_SOURCES,
    _freeze_extraction,
    load_atoms_from_dir,
)
from survey2agent.extraction.question_spec import QUESTIONS
from survey2agent.methods import (
    ABF,
    ABFSelective,
    BCF,
    DSNBF,
    DSNBFSelective,
    MajorityClass,
    MajorityVote,
    NBF,
    NBFSelective,
    OracleExtraction,
    Random,
    SSB,
    SSBGlobal,
    SSBSelective,
)
from survey2agent.methods.base import Method

CANONICAL_SEEDS: tuple[str, ...] = (
    "s20260321",
    "s20260322",
    "s20260323",
    "s20260324",
)
TRAIN_SIZES: tuple[int, ...] = (50, 100, 150, 216)
NOISE_EPS: tuple[float, ...] = (0.0, 0.1, 0.2, 0.3, 0.5)
BIAS_SCALES: tuple[float, ...] = (0.5, 1.0, 2.0)
DROPOUT_SCALES: tuple[float, ...] = (0.5, 1.0, 2.0)

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

_MethodFactory = Callable[[], Method]


@dataclass(frozen=True)
class _SeedData:
    seed: str
    splits: Mapping[str, list[str]]
    difficulty_index: Mapping[str, str]
    gts: dict[str, dict[str, str]]
    oracle_atoms: dict[str, ExtractedAtom]
    llm_atoms: dict[str, ExtractedAtom]
    gemini_atoms: dict[str, ExtractedAtom]


@dataclass(frozen=True)
class _VariantData:
    atoms: dict[str, ExtractedAtom]
    gts: dict[str, dict[str, str]]


def _seed_int(seed: str) -> int:
    if not seed.startswith("s"):
        raise ValueError(f"seed must look like 's20260321', got {seed!r}")
    return int(seed[1:])


def _variant_key(bias_scale: float, dropout_scale: float) -> str:
    return f"b{bias_scale:.1f}_d{dropout_scale:.1f}"


def _load_seed_data(seed: str, splits: Mapping[str, list[str]]) -> _SeedData:
    return _SeedData(
        seed=seed,
        splits=splits,
        difficulty_index=load_persona_difficulty_index(seed),
        gts=load_ground_truths(seed),
        oracle_atoms=load_atoms_for_seed(seed, mode="oracle"),
        llm_atoms=load_atoms_for_seed(seed, mode="llm"),
        gemini_atoms=load_atoms_from_dir(METHOD_OUTPUTS_ROOT / "gemini_p2" / seed / "extract"),
    )


def _to_method_records(records: Sequence[TrainingRecord]) -> list[tuple[ExtractedAtom, dict[str, str]]]:
    grouped: dict[str, tuple[ExtractedAtom, dict[str, str]]] = {}
    for record in records:
        persona_id = record.atom.persona
        if persona_id not in grouped:
            grouped[persona_id] = (record.atom, {})
        grouped[persona_id][1][record.qid] = record.label
    return list(grouped.values())


def _fit_cal_predict(
    method_factory: _MethodFactory,
    *,
    train_atoms: Mapping[str, ExtractedAtom],
    cal_atoms: Mapping[str, ExtractedAtom],
    eval_atoms: Mapping[str, ExtractedAtom],
    gts: Mapping[str, Mapping[str, str]],
    splits: Mapping[str, list[str]],
    difficulty_index: Mapping[str, str],
    eval_split: str = "test",
) -> list[EvaluationResult]:
    train_records = build_training_records(
        train_atoms, gts, splits["train"], difficulty_index=difficulty_index,
    )
    cal_records = build_training_records(
        cal_atoms, gts, splits["cal"], difficulty_index=difficulty_index,
    )
    eval_records = build_training_records(
        eval_atoms, gts, splits[eval_split], difficulty_index=difficulty_index,
    )

    method = method_factory()
    if isinstance(method, OracleExtraction):
        method.attach_gt({pid: dict(gts[pid]) for pid in splits[eval_split] if pid in gts})

    kwargs: dict[str, Sequence[TrainingRecord]] = {}
    if method.requires_fit:
        kwargs["train_records"] = train_records
    if method.requires_calibration:
        kwargs["cal_records"] = cal_records
    return run_method(method, eval_records, **kwargs)


def _metric_summary(results: Sequence[EvaluationResult]) -> dict[str, Any]:
    overall = aggregate_metrics(results)
    by_qid: dict[str, list[EvaluationResult]] = defaultdict(list)
    by_type: dict[str, list[EvaluationResult]] = defaultdict(list)
    by_diff: dict[str, list[EvaluationResult]] = defaultdict(list)
    by_qid_diff: dict[tuple[str, str], list[EvaluationResult]] = defaultdict(list)

    for row in results:
        by_qid[row.qid].append(row)
        by_type[row.reasoning_type].append(row)
        by_diff[row.difficulty_class].append(row)
        by_qid_diff[(row.qid, row.difficulty_class)].append(row)

    per_qid = {
        key: aggregate_metrics(value)["forced_accuracy"]
        for key, value in sorted(by_qid.items())
    }
    per_type = {
        key: aggregate_metrics(value)["forced_accuracy"]
        for key, value in sorted(by_type.items())
    }
    per_difficulty = {
        key: aggregate_metrics(value)["forced_accuracy"]
        for key, value in sorted(by_diff.items())
    }
    per_question_difficulty: dict[str, dict[str, float]] = {}
    for (qid, diff), value in sorted(by_qid_diff.items()):
        per_question_difficulty.setdefault(qid, {})[diff] = aggregate_metrics(value)[
            "forced_accuracy"
        ]

    return {
        "per_question": per_qid,
        "per_type": per_type,
        "per_difficulty": per_difficulty,
        "per_question_difficulty": per_question_difficulty,
        "overall_macro": overall["forced_accuracy"],
        "overall_micro": overall["forced_accuracy_micro"],
        "selective_macro": overall["selective_accuracy"],
        "coverage": overall["coverage"],
    }


def _json_key(method: Method) -> str:
    """Paper-result JSON key for a method instance.

    The primary keys use the public paper names. A few legacy aliases are
    added later for backward compatibility with older appendix scripts.
    """
    name = method.name
    return {
        "MajorityClass": "Majority-Class",
        "SSB": "Single-Source-Best",
        "SSBSelective": "SSB+SKIP",
        "MajorityVote": "Majority-Vote",
        "BCF(4p)": "BCF",
        "NBF": "NBF-NoSkip",
        "NBF+SKIP": "NBF",
        "DSNBF": "DSNBF-NoSkip",
        "DSNBFSelective": "DSNBF",
        "ABF": "ABF-NoSkip",
        "ABF+SKIP": "ABF",
        "OracleExtraction": "Source-Reachability",
    }.get(name, name)


def _add_legacy_aliases(row: dict[str, Any]) -> dict[str, Any]:
    """Expose historical keys until all released readers are migrated."""
    aliases = {
        "BCF": "BCF(4p)",
        "ABF-NoSkip": "PRISM-NoSkip",
        "ABF": "PRISM",
        "Source-Reachability": "Oracle-Ext",
    }
    out = dict(row)
    for public_key, legacy_key in aliases.items():
        if public_key in out and legacy_key not in out:
            out[legacy_key] = copy.deepcopy(out[public_key])
    return out


def _factory_key(factory: _MethodFactory) -> str:
    return _json_key(factory())


NO_SKIP_FACTORIES: tuple[_MethodFactory, ...] = (
    lambda: Random(seed=42),
    MajorityClass,
    SSB,
    MajorityVote,
    BCF,
    NBF,
    DSNBF,
    ABF,
)
FULL_COMPARISON_FACTORIES: tuple[_MethodFactory, ...] = (
    lambda: Random(seed=42),
    MajorityClass,
    SSB,
    SSBSelective,
    MajorityVote,
    BCF,
    NBF,
    NBFSelective,
    DSNBF,
    DSNBFSelective,
    ABF,
    ABFSelective,
    lambda: OracleExtraction(skip_on_miss=False),
)
DGP_FACTORIES: tuple[_MethodFactory, ...] = (
    lambda: Random(seed=42),
    MajorityClass,
    SSBGlobal,
    MajorityVote,
    BCF,
    ABF,
    NBF,
    DSNBF,
    lambda: OracleExtraction(skip_on_miss=False),
)


def _stratified_subsample(
    train_ids: Sequence[str],
    difficulty_index: Mapping[str, str],
    n: int,
    *,
    seed: int,
) -> list[str]:
    by_diff: dict[str, list[str]] = defaultdict(list)
    for persona_id in train_ids:
        by_diff[difficulty_index[persona_id]].append(persona_id)

    n_per_class = n // len(by_diff)
    remainder = n - n_per_class * len(by_diff)
    rng = random.Random(seed)

    result: list[str] = []
    for idx, diff in enumerate(sorted(by_diff)):
        group = sorted(by_diff[diff])
        k = min(len(group), n_per_class + (1 if idx < remainder else 0))
        result.extend(rng.sample(group, k))
    return sorted(result)


def _run_forced_accuracy(
    factory: _MethodFactory,
    *,
    train_atoms: Mapping[str, ExtractedAtom],
    cal_atoms: Mapping[str, ExtractedAtom],
    eval_atoms: Mapping[str, ExtractedAtom],
    gts: Mapping[str, Mapping[str, str]],
    splits: Mapping[str, list[str]],
    difficulty_index: Mapping[str, str],
    train_ids: Sequence[str] | None = None,
    eval_split: str = "test",
) -> float:
    local_splits = {
        key: list(value)
        for key, value in splits.items()
    }
    if train_ids is not None:
        local_splits["train"] = list(train_ids)
    results = _fit_cal_predict(
        factory,
        train_atoms=train_atoms,
        cal_atoms=cal_atoms,
        eval_atoms=eval_atoms,
        gts=gts,
        splits=local_splits,
        difficulty_index=difficulty_index,
        eval_split=eval_split,
    )
    return float(aggregate_metrics(results)["forced_accuracy"])


def _run_train_size_ablation(seed_data: _SeedData) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    full_train_ids = seed_data.splits["train"]
    for train_size in TRAIN_SIZES:
        n_runs = 1 if train_size >= len(full_train_ids) else 5
        by_method: dict[str, list[float]] = defaultdict(list)
        for run_idx in range(n_runs):
            train_ids = (
                full_train_ids
                if train_size >= len(full_train_ids)
                else _stratified_subsample(
                    full_train_ids,
                    seed_data.difficulty_index,
                    train_size,
                    seed=42 + run_idx,
                )
            )
            for factory in NO_SKIP_FACTORIES:
                key = _factory_key(factory)
                by_method[key].append(
                    _run_forced_accuracy(
                        factory,
                        train_atoms=seed_data.oracle_atoms,
                        cal_atoms=seed_data.oracle_atoms,
                        eval_atoms=seed_data.oracle_atoms,
                        gts=seed_data.gts,
                        splits=seed_data.splits,
                        difficulty_index=seed_data.difficulty_index,
                        train_ids=train_ids,
                    )
                )
        out[str(train_size)] = {
            method: statistics.mean(values)
            for method, values in sorted(by_method.items())
        }
        out[str(train_size)] = _add_legacy_aliases(out[str(train_size)])
    return out


def _perturb_atoms(
    atoms: Mapping[str, ExtractedAtom],
    *,
    epsilon: float,
    seed: int = 12345,
) -> dict[str, ExtractedAtom]:
    if epsilon == 0.0:
        return dict(atoms)

    rng = random.Random(seed)
    ordinal_sorted: dict[str, list[str]] = {}
    nominal_questions: set[str] = set()
    answer_spaces: dict[str, list[str]] = {}
    for qid, spec in QUESTIONS.items():
        answer_spaces[qid] = list(spec["answer_space"])
        enc = spec.get("ordinal_encoding") or {}
        if spec.get("answer_space_type") == "nominal":
            nominal_questions.add(qid)
        elif enc:
            ordinal_sorted[qid] = sorted(enc, key=lambda label: enc[label])

    out: dict[str, ExtractedAtom] = {}
    for persona_id, atom in atoms.items():
        extraction = {
            qid: dict(source_map)
            for qid, source_map in atom.extraction.items()
        }
        for qid in EXPECTED_QUESTION_IDS:
            source_map = extraction[qid]
            for source in list(source_map):
                value = source_map[source]
                if value is None or rng.random() >= epsilon:
                    continue
                labels = ordinal_sorted.get(qid)
                if labels and value in labels:
                    pos = labels.index(value)
                    if pos == 0:
                        new_pos = 1
                    elif pos == len(labels) - 1:
                        new_pos = pos - 1
                    else:
                        new_pos = pos + rng.choice([-1, 1])
                    source_map[source] = labels[new_pos]
                elif qid in nominal_questions:
                    alternatives = [x for x in answer_spaces[qid] if x != value]
                    if alternatives:
                        source_map[source] = rng.choice(alternatives)
        out[persona_id] = ExtractedAtom(
            persona=persona_id,
            extraction=_freeze_extraction(extraction),
        )
    return out


def _run_noise_perturbation(seed_data: _SeedData) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for epsilon in NOISE_EPS:
        eval_atoms = _perturb_atoms(seed_data.oracle_atoms, epsilon=epsilon)
        row: dict[str, float] = {}
        for factory in NO_SKIP_FACTORIES:
            row[_factory_key(factory)] = _run_forced_accuracy(
                factory,
                train_atoms=seed_data.oracle_atoms,
                cal_atoms=seed_data.oracle_atoms,
                eval_atoms=eval_atoms,
                gts=seed_data.gts,
                splits=seed_data.splits,
                difficulty_index=seed_data.difficulty_index,
            )
        out[f"{epsilon:.1f}"] = _add_legacy_aliases(row)
    return out


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


def _load_personas(seed: str) -> list[dict[str, Any]]:
    payload = json.loads((seed_dir(seed) / "config" / "personas.json").read_text(encoding="utf-8"))
    return list(payload["personas"])


def _load_event_table(seed: str, persona_id: str) -> list[dict[str, Any]]:
    return list(json.loads((persona_dir(seed, persona_id) / "event_table.json").read_text(encoding="utf-8")))


def _build_variant_data(seed: str, bias_scale: float, dropout_scale: float) -> _VariantData:
    atoms: dict[str, ExtractedAtom] = {}
    gts: dict[str, dict[str, str]] = {}
    base_seed = _seed_int(seed)
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


def _run_dgp_robustness(seed: str, splits: Mapping[str, list[str]]) -> dict[str, Any]:
    variant_results: dict[str, dict[str, float]] = {}
    diff_idx = load_persona_difficulty_index(seed)
    start = time.time()

    for bias_scale, dropout_scale in product(BIAS_SCALES, DROPOUT_SCALES):
        key = _variant_key(bias_scale, dropout_scale)
        data = _build_variant_data(seed, bias_scale, dropout_scale)
        row: dict[str, float] = {}
        for factory in DGP_FACTORIES:
            row[_factory_key(factory)] = _run_forced_accuracy(
                factory,
                train_atoms=data.atoms,
                cal_atoms=data.atoms,
                eval_atoms=data.atoms,
                gts=data.gts,
                splits=splits,
                difficulty_index=diff_idx,
            )
        variant_results[key] = _add_legacy_aliases(row)

    public_methods = sorted(next(iter(variant_results.values())).keys())
    variant_names = sorted(variant_results)
    matrix = np.array(
        [
            [variant_results[variant][method] for method in public_methods]
            for variant in variant_names
        ]
    )
    tau_matrix = np.ones((len(variant_names), len(variant_names)))
    p_matrix = np.zeros_like(tau_matrix)
    for i, j in combinations(range(len(variant_names)), 2):
        tau, p_val = kendalltau(matrix[i], matrix[j])
        tau_matrix[i, j] = tau_matrix[j, i] = float(tau)
        p_matrix[i, j] = p_matrix[j, i] = float(p_val)
    off_diag = tau_matrix[np.triu_indices(len(variant_names), k=1)]

    return {
        "experiment": "distribution_robustness",
        "base_seed": _seed_int(seed),
        "bias_scales": list(BIAS_SCALES),
        "dropout_scales": list(DROPOUT_SCALES),
        "variant_results": variant_results,
        "analysis": {
            "variant_names": variant_names,
            "method_names": public_methods,
            "tau_matrix": tau_matrix.tolist(),
            "p_value_matrix": p_matrix.tolist(),
            "mean_tau": float(np.mean(off_diag)),
            "min_tau": float(np.min(off_diag)),
            "max_tau": float(np.max(off_diag)),
            "std_tau": float(np.std(off_diag)),
            "all_significant": bool(np.all(p_matrix[np.triu_indices(len(variant_names), k=1)] < 0.05)),
        },
        "total_time_sec": time.time() - start,
    }


def _run_full_comparison(
    seed_data: _SeedData,
    *,
    extracted_atoms: Mapping[str, ExtractedAtom],
    old_path: Path,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "n_test": len(seed_data.splits["test"]),
        "personas": list(seed_data.splits["test"]),
        "oracle": {},
        "extraction": {},
    }
    for factory in FULL_COMPARISON_FACTORIES:
        key = _factory_key(factory)
        oracle_results = _fit_cal_predict(
            factory,
            train_atoms=seed_data.oracle_atoms,
            cal_atoms=seed_data.oracle_atoms,
            eval_atoms=seed_data.oracle_atoms,
            gts=seed_data.gts,
            splits=seed_data.splits,
            difficulty_index=seed_data.difficulty_index,
        )
        payload["oracle"][key] = _metric_summary(oracle_results)

        if key == "Source-Reachability":
            continue
        extracted_results = _fit_cal_predict(
            factory,
            train_atoms=seed_data.oracle_atoms,
            cal_atoms=seed_data.oracle_atoms,
            eval_atoms=extracted_atoms,
            gts=seed_data.gts,
            splits=seed_data.splits,
            difficulty_index=seed_data.difficulty_index,
        )
        payload["extraction"][key] = _metric_summary(extracted_results)

    payload["oracle"] = _add_legacy_aliases(payload["oracle"])
    payload["extraction"] = _add_legacy_aliases(payload["extraction"])

    # Preserve cached LLM block if present; it is generated from frozen LLM
    # outputs and is not affected by the BCF/ABF prior update.
    if old_path.exists():
        old = json.loads(old_path.read_text(encoding="utf-8"))
        if "llm" in old:
            payload["llm"] = old["llm"]
    return payload


def _aggregate_4seed(full_by_seed: Mapping[str, dict[str, Any]]) -> dict[str, Any]:
    per_seed: dict[str, dict[str, Any]] = {}
    seed_keys = [str(_seed_int(seed)) for seed in CANONICAL_SEEDS]

    for seed in CANONICAL_SEEDS:
        seed_key = str(_seed_int(seed))
        per_seed[seed_key] = {}
        for method, summary in full_by_seed[seed]["oracle"].items():
            per_seed[seed_key][method] = summary

    method_names = sorted({m for rows in per_seed.values() for m in rows})
    aggregate: dict[str, Any] = {}
    for method in method_names:
        vals = [
            per_seed[seed_key][method]["overall_macro"]
            for seed_key in seed_keys
            if method in per_seed[seed_key]
        ]
        if not vals:
            continue
        mean = statistics.mean(vals)
        sigma = statistics.pstdev(vals)
        per_qid: dict[str, list[float]] = defaultdict(list)
        per_diff: dict[str, list[float]] = defaultdict(list)
        for seed_key in seed_keys:
            if method not in per_seed[seed_key]:
                continue
            summary = per_seed[seed_key][method]
            for qid, value in summary.get("per_question", {}).items():
                per_qid[qid].append(value)
            for diff, value in summary.get("per_difficulty", {}).items():
                per_diff[diff].append(value)
        aggregate[method] = {
            "per_seed_macro": {
                seed_key: per_seed[seed_key][method]["overall_macro"]
                for seed_key in seed_keys
                if method in per_seed[seed_key]
            },
            "mean_macro": mean,
            "std_macro": sigma,
            "ci95_half": 1.96 * sigma / math.sqrt(len(vals)) if vals else 0.0,
            "ci95": [
                mean - (1.96 * sigma / math.sqrt(len(vals)) if vals else 0.0),
                mean + (1.96 * sigma / math.sqrt(len(vals)) if vals else 0.0),
            ],
            "mean_micro": statistics.mean(
                per_seed[seed_key][method]["overall_micro"]
                for seed_key in seed_keys
                if method in per_seed[seed_key]
            ),
            "mean_selective_macro": statistics.mean(
                per_seed[seed_key][method]["selective_macro"]
                for seed_key in seed_keys
                if method in per_seed[seed_key]
            ),
            "mean_coverage": statistics.mean(
                per_seed[seed_key][method]["coverage"]
                for seed_key in seed_keys
                if method in per_seed[seed_key]
            ),
            "per_question_mean": {
                qid: statistics.mean(values) for qid, values in sorted(per_qid.items())
            },
            "per_question_std": {
                qid: statistics.pstdev(values) for qid, values in sorted(per_qid.items())
            },
            "per_difficulty_mean": {
                diff: statistics.mean(values) for diff, values in sorted(per_diff.items())
            },
            "per_difficulty_std": {
                diff: statistics.pstdev(values) for diff, values in sorted(per_diff.items())
            },
        }

    return {
        "seeds": [_seed_int(seed) for seed in CANONICAL_SEEDS],
        "n_seeds": len(CANONICAL_SEEDS),
        "per_seed": per_seed,
        "aggregate": aggregate,
    }


def _extraction_accuracy_cell(
    *,
    oracle_atoms: Mapping[str, ExtractedAtom],
    extracted_atoms: Mapping[str, ExtractedAtom],
    persona_ids: Sequence[str],
    qid: str,
) -> tuple[int, int]:
    """Return ``(correct, total)`` over non-null direct-readout source cells.

    Appendix F's extraction-accuracy table measures whether GPT/Gemini
    reproduced the direct-readout source atom whenever the direct-readout
    atom exists. Null oracle cells are excluded from the denominator; this
    is why, for example, A1 has 160 rather than 200 cells per
    seed/difficulty (40 personas times four informative sources).
    """
    correct = 0
    total = 0
    for persona_id in persona_ids:
        oracle = oracle_atoms[persona_id].extraction[qid]
        extracted = extracted_atoms[persona_id].extraction[qid]
        for source in EXPECTED_SOURCES:
            oracle_value = oracle[source]
            if oracle_value is None:
                continue
            total += 1
            if extracted[source] == oracle_value:
                correct += 1
    return correct, total


def _run_per_question_extraction_accuracy(
    seeds: Sequence[str],
    seed_data_by_seed: Mapping[str, _SeedData],
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    diffs = ("stable", "temporal_shift", "stated_vs_revealed")
    extractors = {
        "gpt": lambda data: data.llm_atoms,
        "gem": lambda data: data.gemini_atoms,
    }

    for qid in EXPECTED_QUESTION_IDS:
        for diff in diffs:
            for ext_name, get_atoms in extractors.items():
                per_seed: list[float] = []
                total_per_seed: list[int] = []
                for seed in seeds:
                    data = seed_data_by_seed[seed]
                    persona_ids = [
                        pid
                        for pid in data.splits["test"]
                        if data.difficulty_index[pid] == diff
                    ]
                    correct, total = _extraction_accuracy_cell(
                        oracle_atoms=data.oracle_atoms,
                        extracted_atoms=get_atoms(data),
                        persona_ids=persona_ids,
                        qid=qid,
                    )
                    if total == 0:
                        raise ValueError(f"zero extraction denominator for {qid}/{diff}/{seed}")
                    per_seed.append(100.0 * correct / total)
                    total_per_seed.append(total)
                out[f"{qid}__{diff}__{ext_name}"] = {
                    "mean": statistics.mean(per_seed),
                    "std": statistics.pstdev(per_seed),
                    "per_seed": per_seed,
                    "total_per_seed": total_per_seed,
                }

    for diff in diffs:
        for ext_name, get_atoms in extractors.items():
            per_seed: list[float] = []
            total_per_seed: list[int] = []
            for seed in seeds:
                data = seed_data_by_seed[seed]
                correct = 0
                total = 0
                persona_ids = [
                    pid
                    for pid in data.splits["test"]
                    if data.difficulty_index[pid] == diff
                ]
                extracted_atoms = get_atoms(data)
                for qid in EXPECTED_QUESTION_IDS:
                    c, t = _extraction_accuracy_cell(
                        oracle_atoms=data.oracle_atoms,
                        extracted_atoms=extracted_atoms,
                        persona_ids=persona_ids,
                        qid=qid,
                    )
                    correct += c
                    total += t
                if total == 0:
                    raise ValueError(f"zero extraction denominator for overall/{diff}/{seed}")
                per_seed.append(100.0 * correct / total)
                total_per_seed.append(total)
            out[f"_OVERALL__{diff}__{ext_name}"] = {
                "mean": statistics.mean(per_seed),
                "std": statistics.pstdev(per_seed),
                "per_seed": per_seed,
            }
    return out


def _write_json(path: Path, payload: Mapping[str, Any], *, dry_run: bool) -> None:
    if dry_run:
        print(f"[dry-run] would write {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[write] {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="*", default=list(CANONICAL_SEEDS))
    parser.add_argument("--out-dir", type=Path, default=RESULTS_ROOT)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--only",
        nargs="*",
        choices=("ablation", "robustness", "full-comparison", "extraction-accuracy", "all"),
        default=["all"],
    )
    args = parser.parse_args()

    selected = set(args.only)
    if "all" in selected:
        selected = {"ablation", "robustness", "full-comparison", "extraction-accuracy"}

    t0 = time.time()
    splits = load_splits()
    full_by_seed: dict[str, dict[str, Any]] = {}
    seed_data_by_seed: dict[str, _SeedData] = {}

    for seed in args.seeds:
        print(f"[regenerate] loading {seed}", flush=True)
        seed_data = _load_seed_data(seed, splits)
        seed_data_by_seed[seed] = seed_data

        if "ablation" in selected:
            print(f"[regenerate] {seed}: train-size + noise", flush=True)
            payload = {
                "ablation": _run_train_size_ablation(seed_data),
                "perturbed": _run_noise_perturbation(seed_data),
            }
            _write_json(args.out_dir / f"ablation_experiments_{seed}.json", payload, dry_run=args.dry_run)

        if "robustness" in selected:
            print(f"[regenerate] {seed}: DGP robustness grid", flush=True)
            payload = _run_dgp_robustness(seed, splits)
            fname = "robustness_results.json" if seed == "s20260321" else f"robustness_{seed}.json"
            _write_json(args.out_dir / fname, payload, dry_run=args.dry_run)

        if "full-comparison" in selected:
            print(f"[regenerate] {seed}: full comparison (oracle + GPT extracted)", flush=True)
            gpt_payload = _run_full_comparison(
                seed_data,
                extracted_atoms=seed_data.llm_atoms,
                old_path=RESULTS_ROOT / f"full_comparison_{seed}.json",
            )
            full_by_seed[seed] = gpt_payload
            _write_json(args.out_dir / f"full_comparison_{seed}.json", gpt_payload, dry_run=args.dry_run)

            print(f"[regenerate] {seed}: full comparison (oracle + Gemini extracted)", flush=True)
            gemini_payload = _run_full_comparison(
                seed_data,
                extracted_atoms=seed_data.gemini_atoms,
                old_path=RESULTS_ROOT / f"full_comparison_gemini_{seed}.json",
            )
            _write_json(
                args.out_dir / f"full_comparison_gemini_{seed}.json",
                gemini_payload,
                dry_run=args.dry_run,
            )

    if "full-comparison" in selected and set(args.seeds) == set(CANONICAL_SEEDS):
        _write_json(args.out_dir / "4seed_results.json", _aggregate_4seed(full_by_seed), dry_run=args.dry_run)

    if "extraction-accuracy" in selected:
        print("[regenerate] per-question extraction accuracy (GPT + Gemini)", flush=True)
        payload = _run_per_question_extraction_accuracy(args.seeds, seed_data_by_seed)
        _write_json(args.out_dir / "c6b_perq_perdiff_4seed.json", payload, dry_run=args.dry_run)

    print(f"[regenerate] done in {(time.time() - t0) / 60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
