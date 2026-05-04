"""Dataclass definitions for daily event records (Layer 2 output).

Each persona gets a 30-day event table where every day contains structured
events across 6 behavioural domains plus derived and semantic annotation fields.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any


# ── Per-domain events ────────────────────────────────────────────────────────

@dataclass
class SleepEvent:
    bedtime: str                     # HH:MM (24h, may exceed 24:00 for next-day)
    wake_time: str                   # HH:MM
    duration_h: float                # hours, rounded to 1 decimal
    quality: int                     # 1–5 ordinal
    interruptions: int               # count
    trouble_falling_asleep: bool
    screen_before_bed: bool
    # Derived fields for GT computation
    short_night: bool                # duration_h < 5.5
    weekend_late_bed: bool           # weekend AND bedtime >= 23:45
    met_target_bedtime: bool         # bedtime <= (mean - 20min)
    target_bedtime_minutes: int      # reference for GT


@dataclass
class ExerciseEvent:
    did_exercise: bool
    exercise_type: str | None = None     # from EXERCISE_TYPES
    duration_min: int | None = None
    intentional: bool = False        # planned/deliberate vs incidental


@dataclass
class DietEvent:
    meals: int                       # total meals today
    home_cooked: int                 # subset of meals
    fast_food: int                   # subset of non-home meals
    coffee_cups: int
    # Semantic fields for source disagreement
    food_orders: list[dict[str, Any]] = field(default_factory=list)
    # Each order: {"beneficiary": "self"|"other"|"shared", "meat_included": bool,
    #              "context_tag": str}


@dataclass
class WorkEvent:
    hours: float
    overtime: bool                   # hours > 9
    is_weekend: bool
    worked_on_weekend: bool
    stress_events: list[str] = field(default_factory=list)
    work_life_balance: int = 3       # 1–5 derived
    # Semantic: afterhours motivation
    afterhours_reason: str | None = None  # "self_project"|"team_need"|None


@dataclass
class SocialEvent:
    activities: list[str] = field(default_factory=list)
    supporting_other: bool = False   # social for obligation vs self-initiated


@dataclass
class MoodEvent:
    overall: int                     # 1–5
    stressed: bool
    energy: int                      # 1–5
    loneliness: int = 3              # 1–5 derived
    routine_satisfaction: int = 3    # 1–5 derived
    relaxation_activity: str | None = None


# ── Full daily record ────────────────────────────────────────────────────────

@dataclass
class DailyRecord:
    date: str                        # YYYY-MM-DD
    day_of_week: str                 # e.g. "monday"
    day_index: int                   # 1-based
    persona_id: str
    sleep: SleepEvent
    exercise: ExerciseEvent
    diet: DietEvent
    work: WorkEvent
    social: SocialEvent
    mood: MoodEvent
    context_tags: list[str] = field(default_factory=list)  # weekend, social, work_stress, ...


# ── Serialisation ────────────────────────────────────────────────────────────

def daily_record_to_dict(r: DailyRecord) -> dict[str, Any]:
    """Convert a DailyRecord to a JSON-serialisable dict."""
    return dataclasses.asdict(r)
