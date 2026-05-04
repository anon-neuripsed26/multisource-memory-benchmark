"""4-seed pooled evaluation utilities.

Implements the paper's 4-seed pooled CI convention (s5_experiments.md):

    "We pool over four random seeds (8,640 question instances on the
    480 seed-persona test clusters) and report 95%
    bootstrap percentile confidence intervals (B = 2,000 persona-level
    resamples)."

Reuses :func:`survey2agent.evaluation.bootstrap.bootstrap_ci` verbatim. Persona ids are prefixed with the seed token so the
persona-clustered bootstrap treats each ``(seed, persona)`` tuple as a
distinct cluster (mirrors
v1.0 reference).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
from typing import Callable, Literal, Mapping, Sequence

from survey2agent.methods.base import Method

from .aggregator import aggregate_metrics
from .bootstrap import bootstrap_ci
from .data_loaders import (
    TrainingRecord,
    build_training_records,
    load_atoms_for_seed,
    load_ground_truths,
    load_splits,
)
from .runner import EvaluationResult, run_method

__all__ = [
    "CANONICAL_SEEDS",
    "aggregate_per_seed_then_average",
    "aggregate_per_seed_then_average_with_breakdown",
    "aggregate_pooled_with_breakdown_ci",
    "aggregate_pooled_with_ci",
    "aggregate_pooled_with_qid_difficulty_ci",
    "evaluate_method_across_seeds",
    "pool_results_across_seeds",
]

_BREAKDOWN_KEYS = ("reasoning_type", "topic", "difficulty_class")

CANONICAL_SEEDS: tuple[str, ...] = (
    "s20260321",
    "s20260322",
    "s20260323",
    "s20260324",
)


def pool_results_across_seeds(
    seed_results: Mapping[str, Sequence[EvaluationResult]],
) -> list[EvaluationResult]:
    """Concatenate per-seed result lists, prefixing ``persona_id`` with the seed.

    For example, persona ``"p_001"`` under seed ``"s20260321"`` becomes
    ``"s20260321_p_001"``. This makes each ``(seed, persona)`` tuple a
    distinct cluster for the persona-clustered bootstrap, mirroring the
    paper's pool-then-bootstrap convention.

    All other ``EvaluationResult`` fields (``qid``, ``prediction``,
    ``label``, ``outcome``, ``reasoning_type``, ``topic``,
    ``difficulty_class``, ``method_name``) are preserved verbatim via
    :func:`dataclasses.replace`. Per-seed iteration order is preserved.
    Empty mapping → empty list.
    """
    pooled: list[EvaluationResult] = []
    for seed, results in seed_results.items():
        for r in results:
            pooled.append(replace(r, persona_id=f"{seed}_{r.persona_id}"))
    return pooled


def _build_records_for_seed(
    seed: str,
    atoms_mode: Literal["llm", "oracle"],
    split: str,
) -> tuple[list[TrainingRecord], list[TrainingRecord], list[TrainingRecord]]:
    """Return ``(train, cal, eval)`` records for one seed.

    ``eval`` records target the requested split (default ``"test"``).
    ``train`` and ``cal`` are always built so callers can pass them to
    fit-/calibration-required methods without re-loading.
    """
    splits = load_splits()
    if split not in splits:
        raise ValueError(
            f"split must be one of {sorted(splits)}; got {split!r}"
        )
    atoms = load_atoms_for_seed(seed, mode=atoms_mode)
    gts = load_ground_truths(seed)
    eval_records = build_training_records(atoms, gts, splits[split])
    train_records = build_training_records(atoms, gts, splits["train"])
    cal_records = build_training_records(atoms, gts, splits["cal"])
    return train_records, cal_records, eval_records


def evaluate_method_across_seeds(
    method_factory: Callable[[], Method],
    *,
    seeds: Sequence[str] = CANONICAL_SEEDS,
    atoms_mode: Literal["llm", "oracle"] = "oracle",
    split: str = "test",
) -> dict[str, list[EvaluationResult]]:
    """Run ``method_factory()`` independently on each seed.

    For each seed:

      1. Load atoms (``mode=atoms_mode``) + ground truth + canonical splits.
      2. Build ``train`` / ``cal`` / ``eval`` records via
         :func:`build_training_records`.
      3. Instantiate a fresh method via ``method_factory()`` (separate
         fit / calibrate / predict cycle per seed — matches the paper's
         independent-seed assumption).
      4. ``train_records`` / ``cal_records`` are passed only when the
         method's ``requires_fit`` / ``requires_calibration`` flag is set.

    Parameters
    ----------
    method_factory : zero-arg callable returning a fresh :class:`Method`.
    seeds : non-empty subset of :data:`CANONICAL_SEEDS`.
    atoms_mode : ``"oracle"`` (480 personas / seed, all splits) or
        ``"llm"`` (frozen test-only set; fit-required methods will see
        empty train / cal records).
    split : key into :func:`load_splits` (default ``"test"``).

    Returns
    -------
    dict
        ``{seed: [EvaluationResult, ...]}`` in the order of ``seeds``.
    """
    if len(seeds) == 0:
        raise ValueError("seeds must be a non-empty sequence")
    unknown = [s for s in seeds if s not in CANONICAL_SEEDS]
    if unknown:
        raise ValueError(
            f"unknown seed(s) {unknown}; expected subset of {CANONICAL_SEEDS}"
        )

    out: dict[str, list[EvaluationResult]] = {}
    for seed in seeds:
        train, cal, evalr = _build_records_for_seed(seed, atoms_mode, split)
        method = method_factory()
        run_kwargs: dict = {}
        if method.requires_fit:
            run_kwargs["train_records"] = train
        if method.requires_calibration:
            run_kwargs["cal_records"] = cal
        out[seed] = run_method(method, evalr, **run_kwargs)
    return out


def aggregate_per_seed_then_average(
    seed_results: Mapping[str, Sequence[EvaluationResult]],
    *,
    metric: str = "selective_accuracy",
    beta: float = 0.5,
) -> float:
    """Compute per-seed ``aggregate_metrics``, then arithmetic-average across seeds.

    This is the reduction used by paper Table 5 point estimates::

        result = (1 / N_seeds) * sum_s aggregate_metrics(seed_results[s])[metric]

    For Table 4 / 6 CI numbers, use :func:`aggregate_pooled_with_ci` instead
    (pool-then-bootstrap). This helper exists so paper-lock tests can assert
    Table 5 cells without depending on bootstrap noise (``aggregate_pooled_with_ci``
    happens to produce the same point estimate for symmetric metrics on this
    data, but the per-seed-then-mean reduction is what the paper text reports).

    Parameters
    ----------
    seed_results
        Mapping of seed name to its list of :class:`EvaluationResult`.
    metric
        Any key returned by :func:`aggregate_metrics` (e.g.
        ``"selective_accuracy"``, ``"coverage"``, ``"forced_accuracy"``,
        ``"f_beta"`` and their ``*_micro`` siblings).
    beta
        Forwarded to :func:`aggregate_metrics` (only matters for ``f_beta``
        keys).

    Returns
    -------
    float
        Arithmetic mean of the per-seed metric values.

    Raises
    ------
    ValueError
        If ``seed_results`` is empty, if any seed has no results, or if any
        per-seed metric value is ``NaN`` / ``None``.
    KeyError
        If ``metric`` is not a key produced by :func:`aggregate_metrics`.
    """
    import math

    if len(seed_results) == 0:
        raise ValueError("seed_results must be a non-empty mapping")

    per_seed_values: list[float] = []
    for seed, results in seed_results.items():
        if len(results) == 0:
            raise ValueError(f"seed {seed!r} has no results")
        agg = aggregate_metrics(results, beta=beta)
        if metric not in agg:
            raise KeyError(
                f"metric {metric!r} not in aggregate_metrics output; "
                f"available keys: {sorted(agg)}"
            )
        value = agg[metric]
        if value is None or (isinstance(value, float) and math.isnan(value)):
            raise ValueError(
                f"seed {seed!r}: metric {metric!r} is {value!r} "
                f"(refusing to silently skip)"
            )
        per_seed_values.append(float(value))

    return sum(per_seed_values) / len(per_seed_values)


def aggregate_pooled_with_ci(
    seed_results: Mapping[str, Sequence[EvaluationResult]],
    *,
    metric: str = "f_beta",
    n_bootstrap: int = 2000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict[str, float]:
    """Pool per-seed results and compute a persona-clustered bootstrap CI.

    Pipeline: :func:`pool_results_across_seeds` →
    :func:`bootstrap_ci` (``cluster_by="persona"``).

    The seed-prefixed persona ids guarantee each ``(seed, persona)`` tuple
    becomes its own cluster, exactly reproducing the paper's pool-then-CI
    procedure.

    Returns the standard ``bootstrap_ci`` dict augmented with
    ``n_seeds`` (number of seeds in input) and ``n_personas_pooled``
    (count of unique seed-prefixed persona ids in the pooled list).
    """
    pooled = pool_results_across_seeds(seed_results)
    ci = bootstrap_ci(
        pooled,
        metric=metric,
        n_bootstrap=n_bootstrap,
        confidence=confidence,
        seed=seed,
        cluster_by="persona",
    )
    n_personas_pooled = len({r.persona_id for r in pooled})
    return {
        **ci,
        "n_seeds": len(seed_results),
        "n_personas_pooled": int(n_personas_pooled),
    }


def _pooled_ci_for_subset(
    subset: Sequence[EvaluationResult],
    *,
    n_seeds: int,
    metric: str,
    n_bootstrap: int,
    confidence: float,
    seed: int,
) -> dict[str, float]:
    """Run persona-clustered bootstrap CI on a (possibly filtered) pooled subset.

    The caller is responsible for ensuring ``subset`` already has
    seed-prefixed ``persona_id`` values (i.e., it came out of
    :func:`pool_results_across_seeds`).
    """
    ci = bootstrap_ci(
        subset,
        metric=metric,
        n_bootstrap=n_bootstrap,
        confidence=confidence,
        seed=seed,
        cluster_by="persona",
    )
    n_personas_pooled = len({r.persona_id for r in subset})
    return {
        **ci,
        "n_seeds": int(n_seeds),
        "n_personas_pooled": int(n_personas_pooled),
    }


def aggregate_pooled_with_breakdown_ci(
    seed_results: Mapping[str, Sequence[EvaluationResult]],
    *,
    breakdown_key: Literal["reasoning_type", "topic", "difficulty_class"],
    metric: str = "f_beta",
    n_bootstrap: int = 2000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict[str, dict[str, float]]:
    """Per-cell pooled CI: pool across seeds, group by ``breakdown_key``,
    bootstrap each cell.

    For each observed value of ``breakdown_key`` (one of
    ``"reasoning_type"`` / ``"topic"`` / ``"difficulty_class"``):

      1. :func:`pool_results_across_seeds` is applied once.
      2. The pooled list is filtered to records whose
         ``getattr(r, breakdown_key)`` matches the cell value.
      3. :func:`bootstrap_ci` (``cluster_by="persona"``,
         ``n_bootstrap=2000`` by default) runs on the cell's subset.

    Because the pooled persona ids are seed-prefixed, every
    ``(seed, persona)`` tuple becomes its own cluster within each cell —
    matching the paper's pool-then-CI convention applied per breakdown
    cell (Tables 6 and 7).

    Returns
    -------
    ``{cell_value: {point, mean, std, ci_low, ci_high, n_bootstrap,
    cluster_by, n_seeds, n_personas_pooled}}``. Cells with no records
    after pooling are omitted. Output is keyed in sorted order.
    """
    if breakdown_key not in _BREAKDOWN_KEYS:
        raise ValueError(
            f"breakdown_key must be one of {_BREAKDOWN_KEYS!r}; got {breakdown_key!r}"
        )

    pooled = pool_results_across_seeds(seed_results)
    cells: dict[str, list[EvaluationResult]] = defaultdict(list)
    for r in pooled:
        cells[getattr(r, breakdown_key)].append(r)

    n_seeds = len(seed_results)
    return {
        cell: _pooled_ci_for_subset(
            cells[cell],
            n_seeds=n_seeds,
            metric=metric,
            n_bootstrap=n_bootstrap,
            confidence=confidence,
            seed=seed,
        )
        for cell in sorted(cells)
    }


def aggregate_pooled_with_qid_difficulty_ci(
    seed_results: Mapping[str, Sequence[EvaluationResult]],
    *,
    metric: str = "f_beta",
    n_bootstrap: int = 2000,
    confidence: float = 0.95,
    seed: int = 42,
) -> dict[str, dict[str, dict[str, float]]]:
    """Cross-product (qid × difficulty_class) pooled bootstrap CI.

    Pool across seeds, group by ``(qid, difficulty_class)``, then bootstrap
    each non-empty cell with ``cluster_by="persona"``.

    Returns ``{qid: {difficulty_class: ci_dict}}`` where ``ci_dict`` has
    the same 9 keys as :func:`aggregate_pooled_with_breakdown_ci` cells.
    Empty cells are omitted; both dict levels iterate in sorted order.
    """
    pooled = pool_results_across_seeds(seed_results)
    cells: dict[tuple[str, str], list[EvaluationResult]] = defaultdict(list)
    for r in pooled:
        cells[(r.qid, r.difficulty_class)].append(r)

    n_seeds = len(seed_results)
    out: dict[str, dict[str, dict[str, float]]] = {}
    for qid, diff in sorted(cells):
        out.setdefault(qid, {})[diff] = _pooled_ci_for_subset(
            cells[(qid, diff)],
            n_seeds=n_seeds,
            metric=metric,
            n_bootstrap=n_bootstrap,
            confidence=confidence,
            seed=seed,
        )
    return out


def aggregate_per_seed_then_average_with_breakdown(
    seed_results: Mapping[str, Sequence[EvaluationResult]],
    *,
    breakdown_keys: str | Sequence[str],
    metric: str = "forced_accuracy",
    beta: float = 0.5,
) -> dict[tuple[str, ...], dict[str, float]]:
    """Per-cell per-seed-then-mean reduction over one or more breakdown axes.

    For each seed, group ``EvaluationResult``s by the tuple of
    ``(getattr(r, k) for k in breakdown_keys)``, call
    :func:`aggregate_metrics` per group, and extract ``metric``. Then take
    the arithmetic mean across seeds **per cell**.

    This is the reduction reported in paper Tables E2a / E2b / E2a-skip /
    E2b-skip (paper Appendix diagnostics section: "4-seed
    mean"). It mirrors the per-(seed, type, diff) reduction used in
    ``_d2_skip_recompute.py`` (the v1.0 reference generation script): per
    (seed, cell) macro-over-qid metric, then unweighted mean across seeds.

    Parameters
    ----------
    seed_results
        Mapping of seed name to its list of :class:`EvaluationResult`.
    breakdown_keys
        Either a single ``EvaluationResult`` attribute name (str) or a
        sequence of names (e.g. ``("reasoning_type", "difficulty_class")``).
        Each must be one of ``"reasoning_type"``, ``"topic"``,
        ``"difficulty_class"``, or ``"qid"``.
    metric
        Any key returned by :func:`aggregate_metrics`.
    beta
        Forwarded to :func:`aggregate_metrics` (only matters for ``f_beta``
        keys).

    Returns
    -------
    ``dict[cell_key, {"point": float, "n_seeds_with_cell": int,
    "per_seed_values": list[float]}]`` where ``cell_key`` is the tuple
    of breakdown values (length == len(breakdown_keys)). For a single
    breakdown key, ``cell_key`` is still a 1-tuple. Cells absent from a
    seed contribute no value to that seed's slot; the mean uses only
    seeds that have at least one record in the cell. Cells with zero
    seeds present (impossible if ``seed_results`` is consistent) are
    omitted. Output is keyed in sorted order.

    Raises
    ------
    ValueError
        If ``seed_results`` is empty or any ``breakdown_key`` is not one
        of the allowed attribute names, or if a per-seed metric is
        ``NaN``/``None``.
    KeyError
        If ``metric`` is not a key produced by :func:`aggregate_metrics`.
    """
    import math

    if len(seed_results) == 0:
        raise ValueError("seed_results must be a non-empty mapping")

    if isinstance(breakdown_keys, str):
        keys: tuple[str, ...] = (breakdown_keys,)
    else:
        keys = tuple(breakdown_keys)
    if not keys:
        raise ValueError("breakdown_keys must be a non-empty sequence")
    allowed = (*_BREAKDOWN_KEYS, "qid")
    for k in keys:
        if k not in allowed:
            raise ValueError(
                f"breakdown key {k!r} not in {allowed!r}"
            )

    # Per-seed: cell_key -> list[EvaluationResult]
    per_seed_cells: dict[str, dict[tuple[str, ...], list[EvaluationResult]]] = {}
    all_cells: set[tuple[str, ...]] = set()
    for s, results in seed_results.items():
        cells: dict[tuple[str, ...], list[EvaluationResult]] = defaultdict(list)
        for r in results:
            cell_key = tuple(getattr(r, k) for k in keys)
            cells[cell_key].append(r)
            all_cells.add(cell_key)
        per_seed_cells[s] = cells

    out: dict[tuple[str, ...], dict[str, float]] = {}
    for cell_key in sorted(all_cells):
        per_seed_values: list[float] = []
        for s in seed_results:
            sub = per_seed_cells[s].get(cell_key)
            if not sub:
                continue
            agg = aggregate_metrics(sub, beta=beta)
            if metric not in agg:
                raise KeyError(
                    f"metric {metric!r} not in aggregate_metrics output; "
                    f"available keys: {sorted(agg)}"
                )
            value = agg[metric]
            if value is None or (isinstance(value, float) and math.isnan(value)):
                raise ValueError(
                    f"seed {s!r}, cell {cell_key!r}: metric {metric!r} is "
                    f"{value!r} (refusing to silently skip)"
                )
            per_seed_values.append(float(value))
        if not per_seed_values:
            continue
        out[cell_key] = {
            "point": sum(per_seed_values) / len(per_seed_values),
            "n_seeds_with_cell": len(per_seed_values),
            "per_seed_values": list(per_seed_values),
        }
    return out
