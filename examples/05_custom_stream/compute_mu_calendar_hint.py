"""Per-source direct readout μ*(calendar_hint, H1) demo.

Sandboxed demo of the μ* readout pattern described in EXTENDING.md
§4.6. This is the deterministic answer that the calendar_hint stream
would give to question H1 (Social — distinct activity-type count over
the last 14 days, defined in
``examples/02_custom_question/h1_question.yaml``) if it had perfect
access to its own raw fields.

In production, this readout would be inlined as
``_mu_calendar_hint_h1`` in
``src/survey2agent/extraction/_mu_shell.py`` and registered in
``compute_all_mu``. Doing so is **out of scope for this sandbox** —
see EXTENDING.md §4.6 for the integration steps that would also
require touching ``_source_loader.load_sources`` to actually read
``calendar_hint.json``.

H1 answer space (matches examples/02): ``0_to_1`` | ``2_to_3`` |
``4_or_more``.

Bias direction
--------------
calendar_hint over-reports compared to objective truth: planned events
include some that did not actually happen. For H1, this means
``μ*(calendar_hint, H1)`` will count distinct *planned* activity types,
which is >= the distinct activities that actually occurred. ABF and
other bias-aware fusion methods would learn ``δ_calendar_hint > 0`` to
correct for this.
"""

from __future__ import annotations

from typing import Any

H1_ANSWER_SPACE: tuple[str, ...] = ("0_to_1", "2_to_3", "4_or_more")


def _classify(n: int) -> str:
    if n <= 1:
        return "0_to_1"
    if n <= 3:
        return "2_to_3"
    return "4_or_more"


def compute_mu_calendar_hint_h1(stream: dict[str, Any]) -> str | None:
    """Return μ*(calendar_hint, H1).

    Counts distinct ``activity_type`` values across every entry in the
    calendar (treated as the last fortnight). Returns ``None`` if the
    stream has no entries (canonical ⊥ "this stream cannot answer").
    """
    entries = list(stream.get("entries") or [])
    if not entries:
        return None
    distinct = {str(ev["activity_type"]) for ev in entries}
    return _classify(len(distinct))
