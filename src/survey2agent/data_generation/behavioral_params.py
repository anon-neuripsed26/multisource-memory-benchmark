"""Domain-specific behavioural parameter sampling for the six persona domains.

Each ``sample_*`` function draws from pre-defined ranges using a
``numpy.random.Generator`` instance for full reproducibility.  The helper
``clamp`` ensures post-shift values stay within physically meaningful bounds.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.random import Generator

from .constants import (
    EXERCISE_TYPES,
    RELAXATION_ACTIVITIES,
    SHIFT_SCENARIOS,
    SOCIAL_ACTIVITY_POOL,
    WEEKDAYS,
)
from .persona_schema import (
    BehavioralParams,
    DietParams,
    ExerciseParams,
    MoodParams,
    SleepParams,
    SocialParams,
    TemporalShift,
    WorkParams,
    behavioral_params_from_dict,
    _to_dict,
)


# ── Utilities ────────────────────────────────────────────────────────────────

def clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _sample_preferred_days(rng: Generator, days_per_week: float) -> list[str] | None:
    """Pick *target* preferred weekdays, sorted in calendar order."""
    target = max(1, min(6, int(round(days_per_week))))
    indices = rng.choice(len(WEEKDAYS), size=target, replace=False)
    return [WEEKDAYS[i] for i in sorted(indices)]


def _sample_social_activities(rng: Generator, k: int = 4) -> list[str]:
    indices = rng.choice(len(SOCIAL_ACTIVITY_POOL), size=k, replace=False)
    return [SOCIAL_ACTIVITY_POOL[i] for i in indices]


def _sample_relaxation_activities(rng: Generator, k: int = 3) -> list[str]:
    indices = rng.choice(len(RELAXATION_ACTIVITIES), size=k, replace=False)
    return [RELAXATION_ACTIVITIES[i] for i in indices]


def sample_primary_exercise(rng: Generator) -> str:
    return EXERCISE_TYPES[rng.integers(len(EXERCISE_TYPES))]


# ── Stable behavioural parameters (healthy / normal range) ────────────────────

def sample_stable_params(rng: Generator, primary_exercise: str) -> BehavioralParams:
    """Sample behavioural params for the *stable* and *temporal_shift* (pre-shift) tracks."""
    days_per_week = int(rng.choice([2, 3, 4, 5]))
    bedtime_choices = ["22:15", "22:30", "23:00", "23:15", "23:30"]

    return BehavioralParams(
        sleep=SleepParams(
            bedtime_mean=bedtime_choices[rng.integers(len(bedtime_choices))],
            bedtime_std_min=int(rng.choice([15, 20, 25, 30])),
            duration_mean_h=round(float(rng.uniform(6.8, 8.0)), 1),
            duration_std_h=round(float(rng.uniform(0.2, 0.5)), 1),
            quality_mean=float(rng.choice([3.5, 4.0, 4.2])),
            quality_std=round(float(rng.uniform(0.3, 0.6)), 1),
            trouble_falling_asleep_prob=round(float(rng.uniform(0.03, 0.12)), 2),
            screen_before_bed_prob=round(float(rng.uniform(0.20, 0.70)), 2),
        ),
        exercise=ExerciseParams(
            days_per_week=days_per_week,
            preferred_days=_sample_preferred_days(rng, days_per_week),
            type=primary_exercise,
            duration_mean_min=int(rng.choice([30, 40, 45, 50, 60])),
            duration_std_min=int(rng.choice([8, 10, 12, 15])),
            skip_prob=round(float(rng.uniform(0.05, 0.15)), 2),
        ),
        diet=DietParams(
            meals_per_day=float(rng.choice([3.0, 3.0, 3.0, 2.5])),
            home_cook_prob=round(float(rng.uniform(0.55, 0.80)), 2),
            fast_food_prob=round(float(rng.uniform(0.05, 0.20)), 2),
            coffee_cups_mean=round(float(rng.uniform(0.5, 2.5)), 1),
            coffee_cups_std=round(float(rng.uniform(0.3, 0.8)), 1),
        ),
        work=WorkParams(
            hours_mean=round(float(rng.uniform(7.5, 9.0)), 1),
            hours_std=round(float(rng.uniform(0.4, 0.8)), 1),
            overtime_prob=round(float(rng.uniform(0.05, 0.18)), 2),
            weekend_work_prob=round(float(rng.uniform(0.02, 0.12)), 2),
            stress_prob=round(float(rng.uniform(0.08, 0.18)), 2),
        ),
        social=SocialParams(
            meetup_prob_per_day=round(float(rng.uniform(0.20, 0.38)), 2),
            weekend_social_boost=round(float(rng.uniform(0.15, 0.35)), 2),
            social_activities=_sample_social_activities(rng),
        ),
        mood=MoodParams(
            overall_mean=round(float(rng.uniform(3.6, 4.4)), 1),
            overall_std=round(float(rng.uniform(0.3, 0.7)), 1),
            stress_prob=round(float(rng.uniform(0.08, 0.18)), 2),
            energy_mean=round(float(rng.uniform(3.4, 4.4)), 1),
            energy_std=round(float(rng.uniform(0.3, 0.7)), 1),
            relaxation_prob=round(float(rng.uniform(0.35, 0.65)), 2),
            relaxation_activities=_sample_relaxation_activities(rng),
        ),
    )


# ── Actual behaviour for stated-vs-revealed (worse than claimed) ──────────────

def sample_actual_params_for_stated(rng: Generator, primary_exercise: str) -> BehavioralParams:
    """Sample behavioural params that diverge from the idealised stated profile."""
    days_per_week = float(rng.choice([0.5, 1.0, 1.5, 2.0, 3.0]))
    preferred_days = _sample_preferred_days(rng, max(1.0, days_per_week)) if days_per_week >= 1.0 else None
    late_bedtimes = ["23:45", "00:00", "00:30", "01:00"]

    extra_social = ["late-night drinks", "networking meetup"]

    return BehavioralParams(
        sleep=SleepParams(
            bedtime_mean=late_bedtimes[rng.integers(len(late_bedtimes))],
            bedtime_std_min=int(rng.choice([30, 40, 45, 60])),
            duration_mean_h=round(float(rng.uniform(5.5, 7.0)), 1),
            duration_std_h=round(float(rng.uniform(0.6, 1.1)), 1),
            quality_mean=round(float(rng.uniform(2.5, 3.5)), 1),
            quality_std=round(float(rng.uniform(0.8, 1.1)), 1),
            trouble_falling_asleep_prob=round(float(rng.uniform(0.20, 0.45)), 2),
            screen_before_bed_prob=round(float(rng.uniform(0.55, 0.90)), 2),
        ),
        exercise=ExerciseParams(
            days_per_week=days_per_week,
            preferred_days=preferred_days,
            type=primary_exercise,
            duration_mean_min=int(rng.choice([20, 25, 30, 35, 40])),
            duration_std_min=int(rng.choice([10, 12, 15, 18])),
            skip_prob=round(float(rng.uniform(0.25, 0.55)), 2),
        ),
        diet=DietParams(
            meals_per_day=float(rng.choice([2.0, 2.5, 3.0])),
            home_cook_prob=round(float(rng.uniform(0.20, 0.45)), 2),
            fast_food_prob=round(float(rng.uniform(0.25, 0.50)), 2),
            coffee_cups_mean=round(float(rng.uniform(2.0, 4.0)), 1),
            coffee_cups_std=round(float(rng.uniform(0.6, 1.2)), 1),
        ),
        work=WorkParams(
            hours_mean=round(float(rng.uniform(6.0, 8.5)), 1),
            hours_std=round(float(rng.uniform(1.5, 3.0)), 1),
            overtime_prob=round(float(rng.uniform(0.12, 0.35)), 2),
            weekend_work_prob=round(float(rng.uniform(0.12, 0.35)), 2),
            stress_prob=round(float(rng.uniform(0.25, 0.50)), 2),
        ),
        social=SocialParams(
            meetup_prob_per_day=round(float(rng.uniform(0.25, 0.50)), 2),
            weekend_social_boost=round(float(rng.uniform(0.15, 0.30)), 2),
            social_activities=_sample_social_activities(rng) + [extra_social[rng.integers(len(extra_social))]],
        ),
        mood=MoodParams(
            overall_mean=round(float(rng.uniform(2.5, 3.5)), 1),
            overall_std=round(float(rng.uniform(0.8, 1.1)), 1),
            stress_prob=round(float(rng.uniform(0.25, 0.50)), 2),
            energy_mean=round(float(rng.uniform(2.2, 3.4)), 1),
            energy_std=round(float(rng.uniform(0.8, 1.1)), 1),
            relaxation_prob=round(float(rng.uniform(0.20, 0.45)), 2),
            relaxation_activities=_sample_relaxation_activities(rng),
        ),
    )


# ── Stated profile (idealised self-narrative) ────────────────────────────────

def build_stated_profile(actual: BehavioralParams, rng: Generator) -> dict[str, str]:
    """Construct the self-reported narrative that contradicts *actual*."""
    preferred_days = actual.exercise.preferred_days or ["weekdays"]
    day_text = ", ".join(d.capitalize() for d in preferred_days[:3])

    social_options = [
        "I'm pretty introverted and usually prefer quiet evenings at home.",
        "I don't socialize that much and mostly keep to myself during the week.",
    ]
    return {
        "sleep_description": "I'm an early riser and usually asleep by 10:30pm so I can get a full 8 hours.",
        "exercise_description": f"I work out almost every morning, usually {actual.exercise.type} on {day_text}.",
        "diet_description": "I'm pretty health-conscious, cook at home often, and mostly avoid junk food.",
        "work_description": "I keep a disciplined work routine with clear boundaries and a reliable daytime schedule.",
        "social_description": social_options[rng.integers(len(social_options))],
        "mood_description": "I'm generally balanced, low-stress, and pretty satisfied with my routines.",
    }


# ── Temporal shift application ───────────────────────────────────────────────

def apply_shift(base: BehavioralParams, scenario: dict[str, Any], rng: Generator) -> BehavioralParams:
    """Apply a shift scenario to *base* params, producing post-shift params."""
    # Convert to dict, mutate, convert back
    d = _to_dict(base)

    # Sleep
    d["sleep"]["duration_mean_h"] = round(clamp(base.sleep.duration_mean_h + scenario["sleep_delta"], 4.5, 9.0), 1)
    d["sleep"]["quality_mean"] = round(clamp(base.sleep.quality_mean + scenario["quality_delta"], 1.5, 5.0), 1)
    if scenario["sleep_delta"] < 0:
        late = ["00:00", "00:30", "01:00"]
        d["sleep"]["bedtime_mean"] = late[rng.integers(len(late))]
    else:
        early = ["21:45", "22:00", "22:15"]
        d["sleep"]["bedtime_mean"] = early[rng.integers(len(early))]
    d["sleep"]["trouble_falling_asleep_prob"] = round(clamp(
        base.sleep.trouble_falling_asleep_prob + max(0.0, scenario["stress_boost"]), 0.02, 0.75), 2)
    d["sleep"]["screen_before_bed_prob"] = round(clamp(
        base.sleep.screen_before_bed_prob + max(0.0, scenario["stress_boost"]), 0.10, 0.95), 2)

    # Exercise
    new_dpw = round(clamp(base.exercise.days_per_week * scenario["exercise_multiplier"], 0.5, 6.0), 1)
    d["exercise"]["days_per_week"] = new_dpw
    d["exercise"]["preferred_days"] = _sample_preferred_days(rng, max(1.0, new_dpw))
    d["exercise"]["duration_mean_min"] = int(clamp(
        base.exercise.duration_mean_min * scenario["exercise_multiplier"], 15, 80))
    d["exercise"]["skip_prob"] = round(clamp(
        base.exercise.skip_prob + (0.20 if scenario["exercise_multiplier"] < 1.0 else -0.05), 0.02, 0.75), 2)

    # Diet
    d["diet"]["fast_food_prob"] = round(clamp(
        base.diet.fast_food_prob + scenario["diet_fast_food_boost"], 0.02, 0.75), 2)
    d["diet"]["home_cook_prob"] = round(clamp(
        base.diet.home_cook_prob - scenario["diet_fast_food_boost"], 0.05, 0.90), 2)
    d["diet"]["coffee_cups_mean"] = round(clamp(
        base.diet.coffee_cups_mean + scenario["coffee_delta"], 0.0, 5.0), 1)

    # Work
    d["work"]["hours_mean"] = round(clamp(
        base.work.hours_mean + scenario["work_hours_delta"], 4.0, 12.5), 1)
    d["work"]["overtime_prob"] = round(clamp(
        base.work.overtime_prob + scenario["overtime_boost"], 0.02, 0.90), 2)
    d["work"]["weekend_work_prob"] = round(clamp(
        base.work.weekend_work_prob + scenario["weekend_work_boost"], 0.02, 0.85), 2)
    d["work"]["stress_prob"] = round(clamp(
        base.work.stress_prob + scenario["stress_boost"], 0.02, 0.90), 2)

    # Social
    d["social"]["meetup_prob_per_day"] = round(clamp(
        base.social.meetup_prob_per_day * scenario["social_multiplier"], 0.02, 0.60), 2)
    d["social"]["weekend_social_boost"] = round(clamp(
        base.social.weekend_social_boost * scenario["social_multiplier"], 0.02, 0.40), 2)

    # Mood
    d["mood"]["overall_mean"] = round(clamp(
        base.mood.overall_mean + scenario["mood_delta"], 1.5, 5.0), 1)
    d["mood"]["stress_prob"] = round(clamp(
        base.mood.stress_prob + scenario["stress_boost"], 0.02, 0.90), 2)
    d["mood"]["energy_mean"] = round(clamp(
        base.mood.energy_mean + scenario["mood_delta"], 1.5, 5.0), 1)
    if scenario["mood_delta"] < 0:
        d["mood"]["relaxation_prob"] = round(clamp(base.mood.relaxation_prob - 0.15, 0.05, 0.80), 2)
        d["mood"]["relaxation_activities"] = [
            "scrolling phone", "watching TV",
            RELAXATION_ACTIVITIES[rng.integers(len(RELAXATION_ACTIVITIES))],
        ]
    else:
        d["mood"]["relaxation_prob"] = round(clamp(base.mood.relaxation_prob + 0.10, 0.05, 0.80), 2)

    return behavioral_params_from_dict(d)


def sample_shift_scenario(rng: Generator) -> tuple[dict[str, Any], int]:
    """Pick a random shift scenario and shift day (10–20)."""
    idx = int(rng.integers(len(SHIFT_SCENARIOS)))
    shift_day = int(rng.integers(10, 21))  # [10, 20] inclusive
    return dict(SHIFT_SCENARIOS[idx]), shift_day
