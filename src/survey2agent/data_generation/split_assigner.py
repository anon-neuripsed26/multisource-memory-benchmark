"""Stratified split assignment: 216/48/96/120 (train/dev/cal/test).

The split is stratified by difficulty class so that each of the three classes
(stable, temporal_shift, stated_vs_revealed) has the same proportional
representation in each split.

Split ratios:  train=45%, dev=10%, cal=20%, test=25%
Per difficulty class (160 each):  72 / 16 / 32 / 40
Total (480):                     216 / 48 / 96 / 120
"""

from __future__ import annotations

import math
from typing import Any

from .constants import DIFFICULTY_TYPES


# ── Split configuration ──────────────────────────────────────────────────────

SPLIT_RATIOS: tuple[tuple[str, float], ...] = (
    ("train",       0.45),
    ("dev",         0.10),
    ("calibration", 0.20),
    ("test",        0.25),
)


# ── Core logic ───────────────────────────────────────────────────────────────

def _proportional_sizes(total: int) -> dict[str, int]:
    """Deterministically distribute *total* into split buckets.

    Uses largest-remainder method so sizes sum to exactly *total*.
    """
    exact = [(name, total * ratio) for name, ratio in SPLIT_RATIOS]
    base = {name: math.floor(size) for name, size in exact}
    remainder = total - sum(base.values())
    if remainder > 0:
        ranked = sorted(
            ((name, size - base[name], i) for i, (name, size) in enumerate(exact)),
            key=lambda t: (-t[1], t[2]),
        )
        for name, _, _ in ranked[:remainder]:
            base[name] += 1
    return {name: base[name] for name, _ in SPLIT_RATIOS}


def assign_splits(personas: list[dict[str, Any]]) -> dict[str, Any]:
    """Assign each persona to a split, stratified by difficulty class.

    Parameters
    ----------
    personas : list[dict]
        Full persona records.  Each must have ``"id"`` and ``"difficulty_type"`` keys.

    Returns
    -------
    dict with keys ``mapping`` (persona_id → split), ``summary`` (split → count),
    and ``per_track_summary`` (difficulty → split → count).
    """
    by_type: dict[str, list[dict[str, Any]]] = {dt: [] for dt in DIFFICULTY_TYPES}
    for p in personas:
        dt = p["difficulty_type"]
        if dt not in by_type:
            raise ValueError(f"Unknown difficulty_type: {dt!r}")
        by_type[dt].append(p)

    mapping: dict[str, str] = {}
    summary: dict[str, int] = {name: 0 for name, _ in SPLIT_RATIOS}
    per_track: dict[str, dict[str, int]] = {}

    for dt in DIFFICULTY_TYPES:
        items = sorted(by_type[dt], key=lambda p: p["id"])
        sizes = _proportional_sizes(len(items))
        per_track[dt] = sizes

        cursor = 0
        for split_name, _ in SPLIT_RATIOS:
            end = cursor + sizes[split_name]
            for p in items[cursor:end]:
                mapping[p["id"]] = split_name
                summary[split_name] += 1
            cursor = end

    return {
        "mapping": mapping,
        "summary": summary,
        "per_track_summary": per_track,
    }
