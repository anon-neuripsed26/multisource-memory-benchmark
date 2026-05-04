"""Runnable example: add a 19th question (H1) without touching the main benchmark.

This test validates three independent pieces of the contract described in
[EXTENDING.md §2](../../EXTENDING.md):

1. ``h1_question.yaml`` conforms to
   ``schemas/question_definition.schema.json`` — the same schema that
   guards the real 18-question benchmark spec.
2. ``compute_h1.compute_h1`` produces the expected label on a fixture
   that matches the real event schema used by
   ``src/survey2agent/data_generation/ground_truth.py`` (sorted by
   ``day_index``; ``d["social"]["activities"]`` is ``list[str]``).
3. The H1 ``answer_space`` contains no reserved ``"SKIP"`` sentinel —
   an import-time guard in
   [`methods/base.py`](../../src/survey2agent/methods/base.py) would
   refuse to load the package otherwise.

Run: ``pytest examples/02_custom_question/test_custom_question.py -v``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


HERE = Path(__file__).parent
ROOT = HERE.parents[1]

# Allow ``import compute_h1`` without requiring a package install.
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

# ``import compute_h1`` works because we just prepended ``HERE`` to sys.path.
import compute_h1 as _compute_h1_mod  # noqa: E402
import compute_mu_h1 as _compute_mu_h1_mod  # noqa: E402


def _load_h1_spec() -> dict:
    yaml = pytest.importorskip("yaml")
    with (HERE / "h1_question.yaml").open() as f:
        data = yaml.safe_load(f)
    assert "H1" in data, "h1_question.yaml must define a top-level H1 entry"
    return data["H1"]


def _load_fixture_events() -> list[dict]:
    with (HERE / "fixture_events.json").open() as f:
        payload = json.load(f)
    return payload["events"]


def test_h1_schema_valid() -> None:
    """h1_question.yaml conforms to the production question schema."""
    jsonschema = pytest.importorskip("jsonschema")
    spec = _load_h1_spec()
    schema_path = ROOT / "schemas" / "question_definition.schema.json"
    with schema_path.open() as f:
        schema = json.load(f)
    jsonschema.validate(spec, schema)


def test_h1_no_skip_collision() -> None:
    """Reserved 'SKIP' sentinel must not appear in the answer space."""
    spec = _load_h1_spec()
    assert "SKIP" not in spec["answer_space"]
    # Ordered labels are the substantive (non-edge) portion.
    assert "SKIP" not in spec.get("ordered_labels", [])


def test_compute_h1_on_fixture_matches_expected_label() -> None:
    """compute_h1 produces '2_to_3' on the shipped fixture.

    The fixture's last-14-day window (day_index 16..29) contains
    three distinct activity types: dinner_with_friends, movie, board_games.
    """
    events = _load_fixture_events()
    result = _compute_h1_mod.compute_h1(events=events, persona={}, sources={})
    assert result["answer"] == "2_to_3", result
    # Derivation detail echoes the raw count for debugging.
    assert "distinct_activity_types=3" in result["derivation_detail"]
    # Returned label is a member of the declared answer space.
    spec = _load_h1_spec()
    assert result["answer"] in spec["answer_space"]


def test_compute_h1_boundary_empty_window() -> None:
    """Empty activities in the last 14 days → '0_to_1'."""
    events = [
        {"day_index": i, "social": {"activities": [], "supporting_other": False}}
        for i in range(30)
    ]
    result = _compute_h1_mod.compute_h1(events=events, persona={}, sources={})
    assert result["answer"] == "0_to_1", result


def test_compute_h1_boundary_many_types() -> None:
    """Five distinct activity types in the last 14 days → '4_or_more'."""
    events = [
        {"day_index": i, "social": {"activities": [], "supporting_other": False}}
        for i in range(16)
    ] + [
        {"day_index": 16, "social": {"activities": ["a"], "supporting_other": False}},
        {"day_index": 17, "social": {"activities": ["b"], "supporting_other": False}},
        {"day_index": 18, "social": {"activities": ["c"], "supporting_other": False}},
        {"day_index": 19, "social": {"activities": ["d"], "supporting_other": False}},
        {"day_index": 20, "social": {"activities": ["e"], "supporting_other": False}},
    ] + [
        {"day_index": i, "social": {"activities": [], "supporting_other": False}}
        for i in range(21, 30)
    ]
    result = _compute_h1_mod.compute_h1(events=events, persona={}, sources={})
    assert result["answer"] == "4_or_more", result


# ──────────────────────────────────────────────────────────────────────────
# μ*(s, H1) — per-source direct readout
# ──────────────────────────────────────────────────────────────────────────


def _make_sources_two_types() -> dict:
    """30-day source bundle whose last 14 days exhibit 2 distinct types."""
    return {
        "daily_self_report": [
            {"day_index": i, "social": {"activities": []}}
            for i in range(16)
        ] + [
            {"day_index": 16, "social": {"activities": ["dinner"]}},
            {"day_index": 17, "social": {"activities": ["movie"]}},
        ] + [
            {"day_index": i, "social": {"activities": []}}
            for i in range(18, 30)
        ],
        "planner": [
            {"day_index": i, "social_plan": {"activities": []}}
            for i in range(16)
        ] + [
            {"day_index": 16, "social_plan": {"activities": ["dinner", "movie", "concert"]}},
        ] + [
            {"day_index": i, "social_plan": {"activities": []}}
            for i in range(17, 30)
        ],
        "profile_ltm": {"stated_profile": {"social": {"types_per_fortnight": 1}}},
        "objective_log": [{"day_index": i} for i in range(30)],
        "device_log": [{"day_index": i, "available": True} for i in range(30)],
    }


def test_mu_h1_returns_one_entry_per_stream() -> None:
    """μ* must return a dict with exactly the five canonical stream keys."""
    mu = _compute_mu_h1_mod.compute_mu_h1(_make_sources_two_types())
    assert set(mu) == set(_compute_mu_h1_mod.EXPECTED_SOURCES)


def test_mu_h1_values_are_in_answer_space_or_none() -> None:
    """Every μ* entry must be in the answer space ∪ {None}."""
    spec = _load_h1_spec()
    allowed = set(spec["answer_space"]) | {None}
    mu = _compute_mu_h1_mod.compute_mu_h1(_make_sources_two_types())
    for stream, label in mu.items():
        assert label in allowed, (stream, label)


def test_mu_h1_streams_with_no_signal_emit_none() -> None:
    """objective_log and device_log have no H1 signal → must be None."""
    mu = _compute_mu_h1_mod.compute_mu_h1(_make_sources_two_types())
    assert mu["objective_log"] is None
    assert mu["device_log"] is None


def test_mu_h1_planner_optimism_vs_self_report() -> None:
    """Bias direction sanity check: planner (b=+1, optimistic) ≥ self-report (b=-1)."""
    mu = _compute_mu_h1_mod.compute_mu_h1(_make_sources_two_types())
    encoding = {"0_to_1": 1, "2_to_3": 2, "4_or_more": 3}
    assert encoding[mu["planner"]] >= encoding[mu["daily_self_report"]]


def test_mu_h1_profile_ltm_falls_back_to_none_when_missing() -> None:
    """profile_ltm without stated_profile.social.types_per_fortnight → None."""
    sources = _make_sources_two_types()
    sources["profile_ltm"] = {}
    mu = _compute_mu_h1_mod.compute_mu_h1(sources)
    assert mu["profile_ltm"] is None
