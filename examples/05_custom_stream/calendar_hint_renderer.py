"""Toy NL renderer for the calendar_hint stream.

Sandboxed demo of the renderer pattern. Production renderers live
under ``src/survey2agent/data_generation/nl_render/`` and are folded
into ``nl_memory_renderer.render_full_memory``. To deploy this stream
for real you would add a ``render_calendar_hint(records, ...) -> str``
function there and call it from ``render_full_memory``.

The render keeps each entry on its own line so an LLM extractor can
unambiguously count distinct activity types and infer the +1 bias
direction (planned >= happened) without prose ambiguity.
"""

from __future__ import annotations

from typing import Any


def render_calendar_hint(stream: dict[str, Any]) -> str:
    """Render a calendar_hint stream view as a NL paragraph."""
    entries = list(stream.get("entries") or [])
    if not entries:
        return "Calendar (last fortnight): no entries logged.\n"

    entries_sorted = sorted(entries, key=lambda e: int(e["day_index"]))
    lines = ["Calendar (last fortnight, planned-event view):"]
    for ev in entries_sorted:
        lines.append(
            f"  Day {int(ev['day_index']):02d} ({ev['slot']}): "
            f"planned to {ev['activity_type'].replace('_', ' ')}."
        )
    lines.append(
        "Note: this is the planning surface; not every entry necessarily "
        "took place."
    )
    return "\n".join(lines) + "\n"
