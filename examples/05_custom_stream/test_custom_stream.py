"""Tests for the calendar_hint sandboxed custom-stream example."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import calendar_hint_projector as _proj
import calendar_hint_renderer as _render
import compute_mu_calendar_hint as _mu


def _load_fixture() -> dict[str, Any]:
    with (_HERE / "fixture_events.json").open("r", encoding="utf-8") as fh:
        return json.load(fh)


def test_projector_emits_expected_schema() -> None:
    fixture = _load_fixture()
    stream = _proj.project_calendar_hint(fixture["events"])
    assert "entries" in stream
    assert isinstance(stream["entries"], list)
    assert len(stream["entries"]) >= 1
    expected_fields = set(_proj.CALENDAR_HINT_SCHEMA_FIELDS)
    for ev in stream["entries"]:
        assert set(ev.keys()) == expected_fields, (
            f"calendar_hint entry has unexpected fields: {set(ev.keys())} != {expected_fields}"
        )
        assert isinstance(ev["day_index"], int)
        assert ev["intended"] is True


def test_projector_bias_direction_plus_one() -> None:
    """+1 bias: distinct planned types >= distinct happened types."""
    fixture = _load_fixture()
    stream = _proj.project_calendar_hint(fixture["events"])

    distinct_planned = {ev["activity_type"] for ev in stream["entries"]}
    distinct_happened = {
        ev["activity_type"] for ev in fixture["events"] if ev.get("happened")
    }
    assert distinct_planned >= distinct_happened, (
        "calendar_hint should over-report (planned superset of happened)"
    )
    # Strict overcount in this fixture: book_club + museum_trip planned
    # but never happened on any day.
    extra = distinct_planned - distinct_happened
    assert extra == {"book_club", "museum_trip"}, (
        f"expected exactly 2 phantom planned types, got: {extra}"
    )


def test_renderer_returns_nonempty_nl() -> None:
    fixture = _load_fixture()
    stream = _proj.project_calendar_hint(fixture["events"])
    text = _render.render_calendar_hint(stream)
    assert isinstance(text, str)
    assert len(text.strip()) > 0
    assert "Calendar" in text
    # At least one entry rendered with day prefix.
    assert "Day 01" in text


def test_mu_calendar_hint_h1_in_answer_space() -> None:
    fixture = _load_fixture()
    stream = _proj.project_calendar_hint(fixture["events"])
    mu = _mu.compute_mu_calendar_hint_h1(stream)
    assert mu in _mu.H1_ANSWER_SPACE


def test_mu_calendar_hint_h1_value_matches_fixture() -> None:
    """Fixture has 9 distinct planned activity types -> '4_or_more'."""
    fixture = _load_fixture()
    stream = _proj.project_calendar_hint(fixture["events"])
    distinct = {ev["activity_type"] for ev in stream["entries"]}
    assert len(distinct) == 9
    assert _mu.compute_mu_calendar_hint_h1(stream) == "4_or_more"


def test_mu_returns_none_on_empty_stream() -> None:
    assert _mu.compute_mu_calendar_hint_h1({"entries": []}) is None
    assert _mu.compute_mu_calendar_hint_h1({}) is None
