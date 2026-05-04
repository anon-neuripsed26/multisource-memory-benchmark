"""Core event table generation logic (Layer 2).

``generate_event_table`` is the single entry point.  For each persona it
produces 30 ``DailyRecord`` entries using the persona's behavioural parameters
and seeded RNG for full determinism.

Temporal-shift personas switch from base to shifted parameters at ``shift_day``.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timedelta
from typing import Any

import numpy as np
from numpy.random import Generator, PCG64

from .constants import WEEKDAYS
from .event_schema import (
    DailyRecord,
    DietEvent,
    ExerciseEvent,
    MoodEvent,
    SleepEvent,
    SocialEvent,
    WorkEvent,
    daily_record_to_dict,
)


# ── Utilities ────────────────────────────────────────────────────────────────

def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _time_str_to_minutes(time_str: str, *, bedtime: bool = False) -> int:
    """Convert 'HH:MM' to minutes since midnight.

    When *bedtime* is True, early-morning times (< 12:00) are mapped to
    24h+ notation so they sort after 23:59.
    """
    h, m = map(int, time_str.split(":"))
    minutes = h * 60 + m
    if bedtime and minutes < 12 * 60:
        minutes += 24 * 60
    return minutes


def _minutes_to_time_str(minutes: int) -> str:
    """Convert minutes since midnight to 'HH:MM'."""
    minutes = minutes % (24 * 60)
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def _stable_seed_offset(persona_id: str, modulus: int = 10_000) -> int:
    """Hash-based offset that is reproducible across Python processes."""
    digest = hashlib.sha256(persona_id.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % modulus


def _is_weekend(date: datetime) -> bool:
    return date.weekday() >= 5


def _day_of_week(date: datetime) -> str:
    return date.strftime("%A").lower()


# ── Parameter accessor (handles temporal shift) ─────────────────────────────

def _get_params(persona: dict[str, Any], day_index: int) -> dict[str, Any]:
    """Return behavioural params for *day_index* (1-based).

    For temporal-shift personas, switched to ``shifted_params`` on and after
    ``shift_day``.
    """
    base = persona["behavioral_params"]
    shift = persona.get("temporal_shift")

    if shift and day_index >= shift["shift_day"]:
        shifted = shift["shifted_params"]
        merged: dict[str, Any] = {}
        for domain in base:
            if domain in shifted:
                merged[domain] = {**base[domain], **shifted[domain]}
            else:
                merged[domain] = base[domain]
        return merged
    return base


# ── Domain generators ────────────────────────────────────────────────────────

STRESS_EVENT_POOL = (
    "tight deadline", "difficult client call", "unexpected bug",
    "last-minute changes", "team conflict", "presentation prep",
    "budget concerns", "overloaded inbox",
)


def _gen_sleep(
    params: dict[str, Any],
    rng: Generator,
    date: datetime,
    persona_base_bedtime_min: int,
) -> SleepEvent:
    bedtime_mean_min = _time_str_to_minutes(params["bedtime_mean"], bedtime=True)
    bedtime_std = float(params["bedtime_std_min"])

    bedtime_min = int(rng.normal(bedtime_mean_min, bedtime_std))
    if _is_weekend(date):
        bedtime_min += int(rng.integers(15, 46))  # 15-45 inclusive
    bedtime_min = int(_clamp(bedtime_min, 20 * 60, 28 * 60))

    duration_h = float(rng.normal(params["duration_mean_h"], params["duration_std_h"]))
    duration_h = round(_clamp(duration_h, 3.0, 11.0), 1)

    wake_min = bedtime_min + int(duration_h * 60)

    quality = int(round(float(rng.normal(params["quality_mean"], params["quality_std"]))))
    quality = int(_clamp(quality, 1, 5))

    interruptions = 0
    if quality <= 2:
        interruptions = int(rng.integers(1, 4))
    elif quality <= 3:
        interruptions = int(rng.integers(0, 2))

    trouble = float(rng.random()) < params["trouble_falling_asleep_prob"]
    screen = float(rng.random()) < params["screen_before_bed_prob"]

    # Derived fields
    short_night = duration_h < 5.5
    weekend_late_bed = _is_weekend(date) and bedtime_min >= _time_str_to_minutes("23:45", bedtime=True)
    target_bedtime = max(20 * 60, persona_base_bedtime_min - 20)
    met_target = bedtime_min <= target_bedtime

    return SleepEvent(
        bedtime=_minutes_to_time_str(bedtime_min),
        wake_time=_minutes_to_time_str(wake_min),
        duration_h=duration_h,
        quality=quality,
        interruptions=interruptions,
        trouble_falling_asleep=trouble,
        screen_before_bed=screen,
        short_night=short_night,
        weekend_late_bed=weekend_late_bed,
        met_target_bedtime=met_target,
        target_bedtime_minutes=target_bedtime,
    )


def _gen_exercise(
    params: dict[str, Any],
    rng: Generator,
    date: datetime,
) -> ExerciseEvent:
    day_name = _day_of_week(date)
    preferred = params.get("preferred_days")

    if preferred:
        base_prob = 0.85 if day_name in preferred else 0.05
    else:
        base_prob = float(params["days_per_week"]) / 7.0

    if float(rng.random()) < params["skip_prob"]:
        base_prob *= 0.3

    did_exercise = float(rng.random()) < base_prob
    if not did_exercise:
        return ExerciseEvent(did_exercise=False)

    duration = int(rng.normal(params["duration_mean_min"], params["duration_std_min"]))
    duration = int(_clamp(duration, 10, 120))

    # Intentionality requires sufficient duration (≥30 min) AND either:
    #   - a match with preferred exercise days, OR
    #   - no preferred-day pattern (duration alone signals deliberate exercise)
    # The AND-only condition with duration>=30 is the published
    # convention; it correctly handles personas without a preferred-day
    # pattern (where the day-match check is vacuously True).
    day_match = (not preferred) or (day_name in preferred)
    intentional = day_match and duration >= 30

    return ExerciseEvent(
        did_exercise=True,
        exercise_type=params["type"],
        duration_min=duration,
        intentional=bool(intentional),
    )


def _gen_diet(
    params: dict[str, Any],
    rng: Generator,
    date: datetime,
    semantic_profile: dict[str, Any],
    context_tags: list[str] | None = None,
) -> DietEvent:
    meals = int(round(float(rng.normal(params["meals_per_day"], 0.4))))
    meals = int(_clamp(meals, 1, 5))

    home_cooked = sum(1 for _ in range(meals) if float(rng.random()) < params["home_cook_prob"])
    remaining = meals - home_cooked
    denom = max(1.0 - float(params["home_cook_prob"]) + 0.01, 0.01)

    # Takeout modulation by stress context
    cond_pref = semantic_profile.get("conditional_preference_conflict", {})
    tags_set = set(context_tags or [])
    base_ff_prob = float(params["fast_food_prob"])
    if "work_stress" in tags_set:
        ff_prob = max(base_ff_prob, float(cond_pref.get("takeout_on_stress_prob", 0.4)))
    else:
        ff_prob = base_ff_prob

    fast_food = sum(
        1 for _ in range(remaining)
        if float(rng.random()) < ff_prob / denom
    )

    coffee = int(round(float(rng.normal(params["coffee_cups_mean"], params["coffee_cups_std"]))))
    coffee = int(_clamp(coffee, 0, 8))

    # Context-aware meat probability (conditional preference conflict)
    exception_contexts = cond_pref.get("exception_contexts", [])
    exception_active = bool(tags_set & set(exception_contexts))
    if exception_active:
        meat_prob = float(cond_pref.get("self_meat_on_exception_prob", 0.15))
    else:
        meat_prob = float(cond_pref.get("self_meat_baseline_prob", 0.05))

    # Generate food orders with beneficiary attribution
    food_orders: list[dict[str, Any]] = []
    attr = semantic_profile.get("attribution_conflict", {})
    other_prob = float(attr.get("for_others_food_order_prob", 0.0))
    shared_prob = float(attr.get("shared_order_prob", 0.0))

    for _ in range(meals):
        r = float(rng.random())
        if r < other_prob:
            beneficiary = "other"
        elif r < other_prob + shared_prob:
            beneficiary = "shared"
        else:
            beneficiary = "self"

        meat = float(rng.random()) < meat_prob
        food_orders.append({
            "beneficiary": beneficiary,
            "meat_included": meat,
            "context_tag": context_tags[0] if context_tags else "routine",
        })

    return DietEvent(
        meals=meals,
        home_cooked=home_cooked,
        fast_food=fast_food,
        coffee_cups=coffee,
        food_orders=food_orders,
    )


def _gen_work(
    params: dict[str, Any],
    rng: Generator,
    date: datetime,
    semantic_profile: dict[str, Any],
) -> WorkEvent:
    weekend = _is_weekend(date)

    if weekend:
        works_today = float(rng.random()) < params["weekend_work_prob"]
        if not works_today:
            return WorkEvent(
                hours=0.0, overtime=False, is_weekend=True,
                worked_on_weekend=False,
            )
        hours = float(rng.normal(float(params["hours_mean"]) * 0.5, float(params["hours_std"])))
    else:
        hours = float(rng.normal(params["hours_mean"], params["hours_std"]))

    hours = round(_clamp(hours, 0.0, 16.0), 1)
    overtime = hours > 9.0

    stress_events: list[str] = []
    stressed = float(rng.random()) < params["stress_prob"]
    if stressed:
        k = int(rng.integers(1, 3))  # 1 or 2
        indices = rng.choice(len(STRESS_EVENT_POOL), size=min(k, len(STRESS_EVENT_POOL)), replace=False)
        stress_events = [STRESS_EVENT_POOL[i] for i in indices]

    # Work-life balance score (derived)
    wlb = 3.6
    wlb += -0.9 if overtime else 0.2
    wlb += -0.7 if (weekend and hours > 0) else 0.0
    wlb += -0.5 if stress_events else 0.2
    wlb += -0.4 if hours >= 9.5 else 0.2
    wlb = int(_clamp(round(wlb), 1, 5))

    # Afterhours reason
    afterhours_reason: str | None = None
    if overtime or hours >= 9.0:
        work_sem = semantic_profile.get("work_semantic_patterns", {})
        self_prob = float(work_sem.get("afterhours_for_self_prob", 0.4))
        if float(rng.random()) < self_prob:
            afterhours_reason = "self_project"
        else:
            afterhours_reason = "team_need"

    return WorkEvent(
        hours=hours,
        overtime=overtime,
        is_weekend=weekend,
        worked_on_weekend=weekend and hours > 0,
        stress_events=stress_events,
        work_life_balance=wlb,
        afterhours_reason=afterhours_reason,
    )


def _gen_social(
    params: dict[str, Any],
    rng: Generator,
    date: datetime,
    semantic_profile: dict[str, Any],
    work: WorkEvent | None = None,
) -> SocialEvent:
    prob = float(params["meetup_prob_per_day"])
    if _is_weekend(date):
        prob += float(params["weekend_social_boost"])

    has_social = float(rng.random()) < prob
    if not has_social:
        return SocialEvent()

    num = 1 if float(rng.random()) < 0.7 else 2
    pool = list(params["social_activities"])
    num = min(num, len(pool))
    indices = rng.choice(len(pool), size=num, replace=False)
    activities = [pool[i] for i in indices]

    # Single-parameter direct mapping (panel decision: eliminate compound OR)
    social_sem = semantic_profile.get("social_semantic_patterns", {})
    oblig_prob = float(social_sem.get("obligation_social_prob", 0.3))
    supporting = float(rng.random()) < oblig_prob

    return SocialEvent(activities=activities, supporting_other=supporting)


def _gen_mood(
    params: dict[str, Any],
    rng: Generator,
    date: datetime,
    work: WorkEvent,
    sleep: SleepEvent,
    social: SocialEvent,
    persona_profile: dict[str, Any],
) -> MoodEvent:
    overall = float(rng.normal(params["overall_mean"], params["overall_std"]))
    if work.stress_events:
        overall -= 0.5
    if sleep.duration_h < 5.5:
        overall -= 0.5
    if sleep.quality <= 2:
        overall -= 0.3
    overall = int(_clamp(round(overall), 1, 5))

    stressed = float(rng.random()) < params["stress_prob"] or bool(work.stress_events)

    energy = float(rng.normal(params["energy_mean"], params["energy_std"]))
    if sleep.duration_h < 6.0:
        energy -= 1.0
    energy = int(_clamp(round(energy), 1, 5))

    # Loneliness (derived)
    loneliness = 2.2
    if persona_profile.get("relationship") == "single":
        loneliness += 0.6
    if not social.activities:
        loneliness += 0.9
    else:
        loneliness -= 0.5
    loneliness += 0.3 if overall <= 2 else -0.2
    loneliness += 0.2 if energy <= 2 else 0.0
    loneliness = int(_clamp(round(loneliness), 1, 5))

    # Routine satisfaction (derived)
    routine_sat = 3.0
    routine_sat += 0.5 if sleep.duration_h >= 7.0 else -0.6
    routine_sat += 0.4 if sleep.quality >= 4 else (-0.3 if sleep.quality <= 2 else 0.0)
    routine_sat += 0.4 if work.hours > 0 and not work.overtime else (-0.7 if work.overtime else 0.0)
    routine_sat += 0.2 if social.activities else -0.1
    routine_sat = int(_clamp(round(routine_sat), 1, 5))

    did_relax = float(rng.random()) < params["relaxation_prob"]
    relax_act: str | None = None
    if did_relax and params["relaxation_activities"]:
        pool = params["relaxation_activities"]
        relax_act = pool[int(rng.integers(len(pool)))]

    return MoodEvent(
        overall=overall,
        stressed=stressed,
        energy=energy,
        loneliness=loneliness,
        routine_satisfaction=routine_sat,
        relaxation_activity=relax_act,
    )


# ── Context tags ─────────────────────────────────────────────────────────────

def _build_context_tags(date: datetime, work: WorkEvent, social: SocialEvent) -> list[str]:
    tags: list[str] = []
    if _is_weekend(date):
        tags.append("weekend")
    if social.activities:
        tags.append("social")
    if work.stress_events or work.overtime:
        tags.append("work_stress")
    if not tags:
        tags.append("routine")
    return sorted(tags)


# ── Public API ───────────────────────────────────────────────────────────────

def generate_event_table(
    persona: dict[str, Any],
    *,
    start_date: str,
    num_days: int,
    base_seed: int,
) -> list[dict[str, Any]]:
    """Generate a 30-day event table for one persona.

    Parameters
    ----------
    persona : dict
        Serialised persona record (from ``personas.json``).
    start_date : str
        ISO date string for Day 1.
    num_days : int
        Number of days to generate (default 30).
    base_seed : int
        Global dataset seed; combined with persona ID for per-persona seed.

    Returns
    -------
    list[dict]
        List of daily record dicts, one per day.
    """
    seed = base_seed + _stable_seed_offset(persona["id"])
    rng = Generator(PCG64(seed))

    start = datetime.strptime(start_date, "%Y-%m-%d")
    semantic_profile = persona.get("semantic_conflict_profile", {})
    profile = persona.get("profile", {})

    # Base bedtime for target computation (always use pre-shift value)
    base_bedtime_min = _time_str_to_minutes(
        persona["behavioral_params"]["sleep"]["bedtime_mean"], bedtime=True,
    )

    records: list[dict[str, Any]] = []

    for day_idx in range(num_days):
        date = start + timedelta(days=day_idx)
        day_1based = day_idx + 1
        params = _get_params(persona, day_1based)

        sleep = _gen_sleep(params["sleep"], rng, date, base_bedtime_min)
        exercise = _gen_exercise(params["exercise"], rng, date)
        work = _gen_work(params["work"], rng, date, semantic_profile)
        social = _gen_social(params["social"], rng, date, semantic_profile, work)
        ctx = _build_context_tags(date, work, social)
        diet = _gen_diet(params["diet"], rng, date, semantic_profile, ctx)
        mood = _gen_mood(params["mood"], rng, date, work, sleep, social, profile)

        record = DailyRecord(
            date=date.strftime("%Y-%m-%d"),
            day_of_week=_day_of_week(date),
            day_index=day_1based,
            persona_id=persona["id"],
            sleep=sleep,
            exercise=exercise,
            diet=diet,
            work=work,
            social=social,
            mood=mood,
            context_tags=ctx,
        )
        records.append(daily_record_to_dict(record))

    return records
