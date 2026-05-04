"""Reference implementation of the H1 ground-truth rule.

H1 | Social — distinct social-activity types engaged in during the last 14 days.

Input contract mirrors the rules in
``src/survey2agent/data_generation/ground_truth.py``:

    events:  list[dict]  — 30 daily event dicts, indexed by ``day_index``
    persona: dict        — the persona JSON (unused here; shown for contract)
    sources: dict        — per-stream raw output (unused here)

Each daily event is shaped like::

    {
      "day_index": <int 0..29>,
      "social": {
          "activities": list[str],   # activity name strings
          "supporting_other": bool,
      },
      # plus "sleep", "work", "diet", "exercise", "mood" — ignored by H1.
      ...
    }

The function returns ``{"answer": <label>, "derivation_detail": <str>}``,
matching every other ``compute_*`` in the main GT registry.
"""

from __future__ import annotations

from typing import Any


def compute_h1(
    events: list[dict[str, Any]],
    persona: dict[str, Any],
    sources: dict[str, Any],
) -> dict[str, Any]:
    """H1 | Social — distinct activity-type count over the last 14 days."""
    # Last 14 days by day_index. Events are ordered 0..29 by construction,
    # but sort defensively so this still works on a re-ordered fixture.
    last_14 = sorted(events, key=lambda d: int(d["day_index"]))[-14:]

    distinct_types = {
        activity
        for d in last_14
        for activity in d["social"].get("activities", [])
    }
    n = len(distinct_types)

    if n <= 1:
        answer = "0_to_1"
    elif n <= 3:
        answer = "2_to_3"
    else:
        answer = "4_or_more"

    return {
        "answer": answer,
        "derivation_detail": f"distinct_activity_types={n} window_days=14",
    }
