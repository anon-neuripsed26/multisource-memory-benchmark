"""Toy projector: latent event table -> calendar_hint stream records.

This is a SANDBOXED demo of the projector pattern described in
EXTENDING.md §4.1. It does NOT touch
``src/survey2agent/data_generation/source_projector.py`` and is not
wired into the production generator. See README.md for the integration
checklist that would be required to deploy this stream for real.

Production projectors take an L2 event table row and emit an L3 source
view. Here we mirror the same shape with a self-contained fixture so
that the example runs offline.

Bias direction
--------------
calendar_hint records what the persona *planned* to do. Because plans
do not always materialise, the planned-event count systematically
exceeds the actual-event count. We model this with **bias direction
b = +1** on social-activity questions: distinct planned activity types
>= distinct actually-happened types. The fixture in
``fixture_events.json`` already encodes this gap (9 planned distinct
types vs 7 happened distinct types).
"""

from __future__ import annotations

from typing import Any

CALENDAR_HINT_SCHEMA_FIELDS: tuple[str, ...] = (
    "day_index",
    "slot",
    "activity_type",
    "intended",
)


def project_calendar_hint(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Project the latent event table into a calendar_hint stream view.

    The output mirrors the shape that
    ``data/benchmark/seeds/<seed>/<persona_id>/calendar_hint.json``
    would have under the production schema:
    ``{"entries": [{day_index, slot, activity_type, intended}, ...]}``.

    Only events with ``planned == True`` are visible to the calendar
    stream — calendar_hint cannot see un-planned spontaneous events.
    The ``intended`` field records the persona's pre-commitment level
    (always ``True`` here because this is a planning surface).
    """
    entries: list[dict[str, Any]] = []
    for ev in events:
        if not ev.get("planned"):
            continue
        entries.append(
            {
                "day_index": int(ev["day_index"]),
                "slot": str(ev["slot"]),
                "activity_type": str(ev["activity_type"]),
                "intended": True,
            }
        )
    return {"entries": entries}
