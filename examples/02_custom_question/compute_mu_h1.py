"""Reference implementation of the per-source direct readout μ*(s, H1).

This is the deterministic answer that source ``s`` would give to question
H1 if it had perfect access to its own raw fields. ``μ*`` is consumed by
the Source Reachability reference rows and by the source-reachability diagnostic table in
``paper_artifacts/appendix/source_ceiling_complement_table.py``; without
it, your new question still works for answer-only methods that consume
LLM atoms but Source Reachability / Struct-LLM rows will report ``None`` for it.

In production this function would be inlined as ``_mu_h1`` in
``src/survey2agent/extraction/_mu_shell.py`` and registered in the
dispatcher ``compute_all_mu``. We keep it standalone here so the
example does not depend on a non-shipped registration.

Per-source semantics for H1 (Social — distinct activity-type count
over the last 14 days):

* ``daily_self_report``: count distinct activity strings across the
  last 14 daily reports. **Bias direction b = -1** (under-reports
  social activity per QBD-2).
* ``planner``: count distinct planned activities across the last 14
  planner entries. **Bias direction b = +1** (planner is optimistic).
* ``profile_ltm``: read the long-term ``stated_profile.social`` baseline
  if available; otherwise return ``None``. Idealized self-image, so
  bias is mixture-absorbed by learned ``δ_prof``.
* ``objective_log``: ⊥. Objective-log does not record social activity
  types in this benchmark.
* ``device_log``: ⊥. Device-log does not record social activity types
  either.

The function returns ``Mu = dict[str, str | None]`` with one entry per
evidence stream. ``None`` is the canonical ⊥ "this stream cannot
answer" sentinel.
"""

from __future__ import annotations

from typing import Any

Mu = dict[str, str | None]

EXPECTED_SOURCES: tuple[str, ...] = (
    "profile_ltm",
    "planner",
    "daily_self_report",
    "objective_log",
    "device_log",
)


def _classify(n: int) -> str:
    if n <= 1:
        return "0_to_1"
    if n <= 3:
        return "2_to_3"
    return "4_or_more"


def _empty_mu() -> Mu:
    return {s: None for s in EXPECTED_SOURCES}


def compute_mu_h1(sources: dict[str, Any]) -> Mu:
    """Per-source direct readout for H1."""
    mu = _empty_mu()

    sr = sources.get("daily_self_report") or []
    if sr:
        last_14 = sorted(sr, key=lambda r: int(r["day_index"]))[-14:]
        types = {
            a
            for r in last_14
            for a in (((r.get("social") or {}).get("activities")) or [])
        }
        mu["daily_self_report"] = _classify(len(types))

    pl = sources.get("planner") or []
    if pl:
        last_14 = sorted(pl, key=lambda r: int(r["day_index"]))[-14:]
        types = {
            a
            for r in last_14
            for a in (((r.get("social_plan") or {}).get("activities")) or [])
        }
        mu["planner"] = _classify(len(types))

    profile = sources.get("profile_ltm") or {}
    base = (
        ((profile.get("stated_profile") or {}).get("social") or {})
        .get("types_per_fortnight")
    )
    if base is not None:
        mu["profile_ltm"] = _classify(int(base))

    return mu
