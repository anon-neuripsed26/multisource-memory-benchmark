"""Bootstrap CI for evaluation metrics.

Default is **persona-clustered** resampling to match the v1.0 reference
and the paper's reported convention.

Persona-cluster bootstrap:
    1. Index ``results`` by ``persona_id`` → list[EvaluationResult].
    2. Let ``N`` = number of unique personas.
    3. For each replicate, sample ``N`` personas with replacement and
       concatenate their result lists (so each replicate has a variable
       record count but a fixed persona count).
    4. Aggregate the replicate via :func:`aggregate_metrics` (macro by
       default) and read the requested metric.

Defaults match paper bootstrap CI: ``n_bootstrap=2000``, ``seed=42``,
``confidence=0.95``, percentile method.

``cluster_by="iid"`` preserves legacy flat-row resampling as an escape
hatch.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Callable, Mapping, Sequence

import numpy as np

from .aggregator import aggregate_metrics
from .runner import EvaluationResult

__all__ = ["bootstrap_ci"]


def _empty_ci() -> dict[str, float]:
    return {
        "point": 0.0,
        "mean": 0.0,
        "std": 0.0,
        "ci_low": 0.0,
        "ci_high": 0.0,
        "n_bootstrap": 0,
        "cluster_by": "persona",
    }


def _index_by_persona(
    results: Sequence[EvaluationResult],
) -> "OrderedDict[str, list[EvaluationResult]]":
    """Group ``results`` by ``persona_id``, preserving first-seen order."""
    grouped: "OrderedDict[str, list[EvaluationResult]]" = OrderedDict()
    for r in results:
        grouped.setdefault(r.persona_id, []).append(r)
    return grouped


def bootstrap_ci(
    results: Sequence[EvaluationResult],
    *,
    metric: str = "f_beta",
    metric_fn: Callable[[Sequence[EvaluationResult]], float] | None = None,
    n_bootstrap: int = 2000,
    confidence: float = 0.95,
    seed: int = 42,
    cluster_by: str = "persona",
    aggregator_kwargs: Mapping | None = None,
) -> dict[str, float]:
    """Percentile bootstrap CI for a scalar metric on ``results``.

    Two metric modes:
    - ``metric_fn``: a custom callable ``Sequence[EvaluationResult] → float``.
    - ``metric``: a key into :func:`aggregate_metrics` (default ``"f_beta"``,
      which is macro-derived).

    Two clustering modes:
    - ``cluster_by="persona"`` (default, paper convention): resample
      ``N`` personas with replacement (``N`` = unique personas in input).
    - ``cluster_by="iid"``: legacy flat-row resampling.

    Returns ``point`` (metric on full sample), ``mean``, ``std``,
    ``ci_low`` / ``ci_high`` (percentile bounds), ``n_bootstrap``,
    ``cluster_by``. Determinism via ``np.random.default_rng(seed)``.

    Empty input → all zeros, no exception.
    """
    if not (0.0 < confidence < 1.0):
        raise ValueError(f"confidence must be in (0, 1); got {confidence!r}")
    if n_bootstrap <= 0:
        raise ValueError(f"n_bootstrap must be positive; got {n_bootstrap!r}")
    if cluster_by not in {"persona", "iid"}:
        raise ValueError(
            f"cluster_by must be 'persona' or 'iid'; got {cluster_by!r}"
        )

    n = len(results)
    if n == 0:
        out = _empty_ci()
        out["cluster_by"] = cluster_by
        return out

    agg_kwargs: dict = dict(aggregator_kwargs) if aggregator_kwargs else {}

    def _eval(sample: Sequence[EvaluationResult]) -> float:
        if metric_fn is not None:
            return float(metric_fn(sample))
        m = aggregate_metrics(sample, **agg_kwargs)
        if metric not in m:
            raise KeyError(
                f"metric {metric!r} not in aggregate_metrics keys {sorted(m)}"
            )
        return float(m[metric])

    point = _eval(results)
    rng = np.random.default_rng(seed)
    boot_vals = np.empty(n_bootstrap, dtype=float)

    if cluster_by == "persona":
        persona_index = _index_by_persona(results)
        persona_ids = list(persona_index.keys())
        n_personas = len(persona_ids)
        for b in range(n_bootstrap):
            idx = rng.integers(0, n_personas, size=n_personas)
            sample: list[EvaluationResult] = []
            for i in idx:
                sample.extend(persona_index[persona_ids[i]])
            boot_vals[b] = _eval(sample)
    else:  # iid
        results_list = list(results)
        for b in range(n_bootstrap):
            idx = rng.integers(0, n, size=n)
            sample = [results_list[i] for i in idx]
            boot_vals[b] = _eval(sample)

    alpha = 1.0 - confidence
    lo = float(np.quantile(boot_vals, alpha / 2.0))
    hi = float(np.quantile(boot_vals, 1.0 - alpha / 2.0))

    return {
        "point": float(point),
        "mean": float(boot_vals.mean()),
        "std": float(boot_vals.std(ddof=0)),
        "ci_low": lo,
        "ci_high": hi,
        "n_bootstrap": int(n_bootstrap),
        "cluster_by": cluster_by,
    }
