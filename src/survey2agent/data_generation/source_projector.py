"""Layer 3 — Source projection from latent event tables.

Five sources with distinct distortion models:

  1. profile_ltm        — stale snapshot anchored to Days 1..anchor_end
  2. planner            — aspirational daily plans from behavioral_params (b=+1)
  3. daily_self_report  — topic-dependent bias (Work:-1, Diet:+1, Social:-1,
                          Sleep:+1, Exercise:+1)
  4. objective_log      — accurate with small noise (b=0, δ=0 fixed)
  5. device_log         — accurate but high field-level dropout (b=0)
"""

from __future__ import annotations

import hashlib
from collections import Counter
from typing import Any

import numpy as np
from numpy.random import Generator, PCG64


SCHEMA_VERSION = "v2.source_projection"


# ── RNG helpers ──────────────────────────────────────────────────────────────

def _stable_seed(base_seed: int, persona_id: str, namespace: str) -> int:
    """Deterministic 32-bit seed keyed by (base_seed, persona_id, namespace)."""
    digest = hashlib.sha256(
        f"{base_seed}:{persona_id}:{namespace}".encode("utf-8")
    ).hexdigest()
    return int(digest[:16], 16) & 0xFFFFFFFF


def _make_rng(base_seed: int, persona_id: str, namespace: str) -> Generator:
    return Generator(PCG64(_stable_seed(base_seed, persona_id, namespace)))


# ── Numeric helpers ──────────────────────────────────────────────────────────

def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def _mean(values: list[float]) -> float:
    return round(sum(values) / len(values), 3) if values else 0.0


def _minutes_to_hhmm(minutes: int) -> str:
    minutes = int(minutes) % (24 * 60)
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def _time_to_minutes(time_str: str, *, bedtime: bool = False) -> int:
    h, m = map(int, time_str.split(":"))
    minutes = h * 60 + m
    if bedtime and minutes < 12 * 60:
        minutes += 24 * 60
    return minutes


# ── Difficulty-conditioned source knobs ──────────────────────────────────────

def _infer_knobs(persona: dict[str, Any], total_days: int) -> dict[str, Any]:
    difficulty = persona["difficulty_type"]
    shift = persona.get("temporal_shift") or {}

    if difficulty == "temporal_shift":
        shift_day = int(shift.get("shift_day", max(8, total_days // 2)))
        anchor_end = max(2, shift_day - 2)
        knobs = {
            "profile_anchor_end_day": anchor_end,
            "self_report_conflict_rate": 0.20,
            "self_report_underreport_bias": 0.18,
            "self_report_overreport_bias": 0.12,
            "planner_optimism_bias": 0.20,
            "planner_behavior_gap_rate": 0.26,
            "device_dropout_rate": 0.14,
            "device_noise_rate": 0.13,
            "objective_dropout_rate": 0.12,
            "objective_noise_rate": 0.10,
        }

    elif difficulty == "stated_vs_revealed":
        anchor_end = max(5, min(10, total_days))
        knobs = {
            "profile_anchor_end_day": anchor_end,
            "self_report_conflict_rate": 0.34,
            "self_report_underreport_bias": 0.10,
            "self_report_overreport_bias": 0.24,
            "planner_optimism_bias": 0.28,
            "planner_behavior_gap_rate": 0.22,
            "device_dropout_rate": 0.16,
            "device_noise_rate": 0.14,
            "objective_dropout_rate": 0.15,
            "objective_noise_rate": 0.11,
        }

    else:  # stable
        anchor_end = max(6, min(12, total_days))
        knobs = {
            "profile_anchor_end_day": anchor_end,
            "self_report_conflict_rate": 0.16,
            "self_report_underreport_bias": 0.12,
            "self_report_overreport_bias": 0.10,
            "planner_optimism_bias": 0.14,
            "planner_behavior_gap_rate": 0.12,
            "device_dropout_rate": 0.10,
            "device_noise_rate": 0.08,
            "objective_dropout_rate": 0.09,
            "objective_noise_rate": 0.07,
        }

    # Persona-level sparsity modulation (stated_vs_revealed only)
    svr = persona.get("stated_vs_revealed") or {}
    sparsity = svr.get("sparsity_injection") or {}
    extra_day = float(sparsity.get("missing_day_prob", 0.0))
    knobs["device_dropout_rate"] = min(0.50, knobs["device_dropout_rate"] + extra_day)
    knobs["objective_dropout_rate"] = min(0.50, knobs["objective_dropout_rate"] + extra_day)

    return knobs


# ── Anchor-window aggregation (profile_ltm) ─────────────────────────────────

def _summarize_window(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate event-table records into compact routine snapshot."""
    if not records:
        return {"days": 0}

    sleep_rows = [r["sleep"] for r in records]
    exercise_rows = [r["exercise"] for r in records]
    diet_rows = [r["diet"] for r in records]
    work_rows = [r["work"] for r in records]
    social_rows = [r["social"] for r in records]
    mood_rows = [r["mood"] for r in records]

    bedtimes = [_time_to_minutes(s["bedtime"], bedtime=True) for s in sleep_rows]
    wakes = [_time_to_minutes(s["wake_time"]) for s in sleep_rows]
    durations = [float(s["duration_h"]) for s in sleep_rows]
    qualities = [float(s["quality"]) for s in sleep_rows]

    ex_active = [e for e in exercise_rows if e.get("did_exercise")]
    ex_durations = [float(e["duration_min"]) for e in ex_active]
    ex_types: Counter[str] = Counter(
        e.get("exercise_type") for e in ex_active if e.get("exercise_type")
    )

    work_hours = [float(w["hours"]) for w in work_rows]
    overtime_days = sum(1 for w in work_rows if w.get("overtime"))
    weekend_work = sum(1 for w in work_rows if w.get("worked_on_weekend"))
    stress_days = sum(1 for w in work_rows if w.get("stress_events"))

    meals = [float(d["meals"]) for d in diet_rows]
    home_cooked = [float(d["home_cooked"]) for d in diet_rows]
    fast_food = [float(d["fast_food"]) for d in diet_rows]
    coffee = [float(d["coffee_cups"]) for d in diet_rows]

    social_active = [s for s in social_rows if s.get("activities")]
    social_acts: Counter[str] = Counter(
        a for s in social_rows for a in s.get("activities", [])
    )

    mood_overall = [float(m["overall"]) for m in mood_rows]
    mood_energy = [float(m["energy"]) for m in mood_rows]
    mood_stressed = sum(1 for m in mood_rows if m.get("stressed"))

    n = len(records)
    return {
        "days": n,
        "sleep": {
            "bedtime_mean": _minutes_to_hhmm(int(round(_mean(bedtimes)))),
            "wake_time_mean": _minutes_to_hhmm(int(round(_mean(wakes)))),
            "duration_h_mean": round(_mean(durations), 1),
            "quality_mean": round(_mean(qualities), 1),
        },
        "exercise": {
            "days_active": len(ex_active),
            "days_per_week": round(len(ex_active) * 7.0 / n, 1),
            "type_mode": ex_types.most_common(1)[0][0] if ex_types else None,
            "duration_min_mean": round(_mean(ex_durations), 1) if ex_durations else 0.0,
        },
        "diet": {
            "meals_per_day_mean": round(_mean(meals), 1),
            "home_cooked_mean": round(_mean(home_cooked), 1),
            "fast_food_mean": round(_mean(fast_food), 1),
            "coffee_cups_mean": round(_mean(coffee), 1),
        },
        "work": {
            "hours_mean": round(_mean(work_hours), 1),
            "overtime_share": round(overtime_days / n, 2),
            "weekend_work_share": round(weekend_work / n, 2),
            "stress_share": round(stress_days / n, 2),
        },
        "social": {
            "active_days": len(social_active),
            "active_days_per_week": round(len(social_active) * 7.0 / n, 1),
            "activity_mode": social_acts.most_common(1)[0][0] if social_acts else None,
        },
        "mood": {
            "overall_mean": round(_mean(mood_overall), 1),
            "energy_mean": round(_mean(mood_energy), 1),
            "stressed_share": round(mood_stressed / n, 2),
        },
    }


# ══════════════════════════════════════════════════════════════════════════════
# 1.  profile_ltm  — stale early-window summary
# ══════════════════════════════════════════════════════════════════════════════

def build_profile_ltm(
    persona: dict[str, Any],
    records: list[dict[str, Any]],
    knobs: dict[str, Any],
) -> dict[str, Any]:
    total_days = len(records)
    anchor_end = int(knobs["profile_anchor_end_day"])
    anchor_records = [r for r in records if int(r["day_index"]) <= anchor_end]
    anchor_summary = _summarize_window(anchor_records)

    profile = persona["profile"]
    traits = persona["stable_traits"]
    difficulty = persona["difficulty_type"]

    # For stated_vs_revealed: use idealized stated_profile instead of actual data
    if difficulty == "stated_vs_revealed":
        stated = persona["stated_vs_revealed"]["stated_profile"]
        routine_snapshot = {
            "sleep": {
                "bedtime": "22:30",
                "duration_h": 8.0,
                "quality_label": "good",
            },
            "exercise": {
                "days_per_week": 5.0,
                "type": traits["primary_exercise"],
                "session_minutes": 45,
            },
            "diet": {
                "meals_per_day": 3.0,
                "home_cooked_mean": 2.1,  # "often" = 70% of 3 meals
                "home_cooked_bias": "often",
                "coffee_cups_mean": 1.0,
            },
            "work": {
                "hours_mean": 8.0,
                "overtime_bias": "rare",
            },
            "social": {
                "self_description": stated["social_description"],
                "close_friends": traits["close_friends"],
            },
            "mood": {
                "self_description": stated["mood_description"],
            },
        }
    else:
        # Anchor to early-window actuals (stale for temporal_shift)
        params = persona["behavioral_params"]
        routine_snapshot = {
            "sleep": {
                "bedtime": anchor_summary["sleep"]["bedtime_mean"]
                    or params["sleep"]["bedtime_mean"],
                "duration_h": anchor_summary["sleep"]["duration_h_mean"]
                    or params["sleep"]["duration_mean_h"],
                "quality_mean": anchor_summary["sleep"]["quality_mean"]
                    or params["sleep"]["quality_mean"],
            },
            "exercise": {
                "days_per_week": anchor_summary["exercise"]["days_per_week"]
                    or params["exercise"]["days_per_week"],
                "type": anchor_summary["exercise"]["type_mode"]
                    or params["exercise"]["type"],
                "session_minutes": anchor_summary["exercise"]["duration_min_mean"]
                    or params["exercise"]["duration_mean_min"],
            },
            "diet": {
                "meals_per_day": anchor_summary["diet"]["meals_per_day_mean"]
                    or params["diet"]["meals_per_day"],
                "home_cooked_mean": anchor_summary["diet"]["home_cooked_mean"]
                    or round(
                        float(params["diet"]["home_cook_prob"])
                        * float(params["diet"]["meals_per_day"]),
                        1,
                    ),
                "coffee_cups_mean": anchor_summary["diet"]["coffee_cups_mean"]
                    or params["diet"]["coffee_cups_mean"],
            },
            "work": {
                "hours_mean": anchor_summary["work"]["hours_mean"]
                    or params["work"]["hours_mean"],
                "overtime_share": anchor_summary["work"]["overtime_share"]
                    or params["work"].get("overtime_prob", 0.0),
            },
            "social": {
                "active_days_per_week": anchor_summary["social"]["active_days_per_week"]
                    or round(
                        float(params["social"]["meetup_prob_per_day"]) * 7.0, 1
                    ),
                "activity_mode": anchor_summary["social"]["activity_mode"]
                    or (
                        params["social"]["social_activities"][0]
                        if params["social"]["social_activities"]
                        else None
                    ),
            },
            "mood": {
                "overall_mean": anchor_summary["mood"]["overall_mean"]
                    or params["mood"]["overall_mean"],
                "energy_mean": anchor_summary["mood"]["energy_mean"]
                    or params["mood"]["energy_mean"],
                "stressed_share": anchor_summary["mood"]["stressed_share"]
                    or params["mood"]["stress_prob"],
            },
        }

        # Add derived mood/work fields from anchor window
        wlb_values = [
            float(r["work"].get("work_life_balance", 3))
            for r in anchor_records
        ]
        loneliness_values = [
            float(r["mood"].get("loneliness", 3)) for r in anchor_records
        ]
        satisfaction_values = [
            float(r["mood"].get("routine_satisfaction", 3))
            for r in anchor_records
        ]
        if wlb_values:
            routine_snapshot["work"]["work_life_balance_mean"] = round(
                _mean(wlb_values), 1
            )
        if loneliness_values:
            routine_snapshot["mood"]["loneliness_mean"] = round(
                _mean(loneliness_values), 1
            )
        if satisfaction_values:
            routine_snapshot["mood"]["routine_satisfaction_mean"] = round(
                _mean(satisfaction_values), 1
            )

    # Derive afterhours_work_style identity from anchor window behaviour
    if difficulty == "stated_vs_revealed":
        # Idealized self-image: claims strict work-life boundaries
        afterhours_work_style: str | None = "strict_boundary"
    else:
        anchor_weekends = [
            r for r in anchor_records if r["work"]["is_weekend"]
        ]
        if anchor_weekends:
            ww_rate = (
                sum(1 for r in anchor_weekends if r["work"]["worked_on_weekend"])
                / len(anchor_weekends)
            )
            if ww_rate <= 0.15:
                afterhours_work_style = "strict_boundary"
            elif ww_rate >= 0.35:
                afterhours_work_style = "flexible"
            else:
                afterhours_work_style = None
        else:
            afterhours_work_style = None

    return {
        "schema_version": SCHEMA_VERSION,
        "source_type": "profile_ltm",
        "persona_id": persona["id"],
        "persona_name": persona["name"],
        "difficulty_type": difficulty,
        "anchor_window": {
            "start_day_index": 1,
            "end_day_index": anchor_end,
            "total_days": total_days,
            "staleness_days": max(0, total_days - anchor_end),
        },
        "facts": {
            "identity": {
                "name": persona["name"],
                "age": profile["age"],
                "occupation": profile["occupation"],
                "city": profile["city"],
                "relationship": profile["relationship"],
            },
            "traits": {
                "dietary_restriction": traits["dietary_restriction"],
                "social_preference": traits["social_preference"],
                "close_friends": traits["close_friends"],
                "primary_exercise": traits["primary_exercise"],
                "afterhours_work_style": afterhours_work_style,
            },
            "routine_snapshot": routine_snapshot,
        },
    }


# ══════════════════════════════════════════════════════════════════════════════
# 2.  planner  — aspirational daily plans (b = +1 optimistic)
# ══════════════════════════════════════════════════════════════════════════════

def build_planner(
    persona: dict[str, Any],
    records: list[dict[str, Any]],
    knobs: dict[str, Any],
    base_seed: int = 0,
) -> dict[str, Any]:
    rng = _make_rng(base_seed, persona["id"], "planner")
    params = persona["behavioral_params"]
    profile = persona["profile"]
    traits = persona["stable_traits"]

    base_sleep = params["sleep"]
    base_exercise = params["exercise"]
    base_diet = params["diet"]
    base_work = params["work"]
    base_social = params["social"]
    base_mood = params["mood"]

    output_records: list[dict[str, Any]] = []

    for record in records:
        day_index = int(record["day_index"])
        weekend = record["day_of_week"] in {"saturday", "sunday"}

        # Sleep target — aspirational
        target_duration = round(
            _clamp(
                float(base_sleep["duration_mean_h"])
                + knobs["planner_optimism_bias"],
                5.5,
                9.0,
            ),
            1,
        )
        target_bedtime_min = (
            _time_to_minutes(base_sleep["bedtime_mean"], bedtime=True)
            - int(rng.integers(10, 36))
        )

        # Exercise target — slightly more ambitious
        exercise_planned = float(rng.random()) < min(
            0.95, float(base_exercise["days_per_week"]) / 7.0 + 0.15
        )
        exercise_duration = int(
            _clamp(
                round(
                    float(base_exercise["duration_mean_min"])
                    * (1.0 + knobs["planner_optimism_bias"] * 0.5)
                ),
                10,
                120,
            )
        )

        # Work target — plan to work less
        work_limit = round(
            _clamp(
                float(base_work["hours_mean"]) - float(rng.uniform(0.2, 0.8)),
                4.0,
                12.0,
            ),
            1,
        )

        # Social intent
        social_intent = float(rng.random()) < max(
            0.2, float(base_social["meetup_prob_per_day"]) + 0.10
        )

        # Gap injection: make plan more aspirational with some probability
        if float(rng.random()) < knobs["planner_behavior_gap_rate"]:
            target_duration = round(
                _clamp(target_duration + float(rng.uniform(0.2, 0.6)), 5.5, 9.0),
                1,
            )
            work_limit = round(
                _clamp(work_limit - float(rng.uniform(0.4, 1.1)), 0.0, 12.0), 1
            )
            if not exercise_planned:
                exercise_planned = True
                exercise_duration = int(
                    _clamp(exercise_duration + int(rng.integers(5, 16)), 10, 120)
                )
            if float(rng.random()) < 0.5:
                social_intent = True

        if weekend:
            if float(rng.random()) < 0.75:
                exercise_planned = True
            work_limit = round(
                _clamp(work_limit - float(rng.uniform(0.5, 1.0)), 0.0, 10.0), 1
            )

        output_records.append({
            "date": record["date"],
            "day_index": day_index,
            "sleep_target": {
                "bedtime": _minutes_to_hhmm(target_bedtime_min),
                "duration_h": target_duration,
                "wake_time": _minutes_to_hhmm(
                    target_bedtime_min + int(round(target_duration * 60))
                ),
            },
            "exercise_target": {
                "intended": exercise_planned,
                "type": base_exercise["type"],
                "duration_min": exercise_duration if exercise_planned else 0,
            },
            "work_target": {
                "hours_limit": work_limit,
                "avoid_overtime": True,
                "finish_by": "18:00" if not weekend else "16:00",
            },
            "diet_target": {
                "meals": int(
                    _clamp(round(float(base_diet["meals_per_day"])), 1, 5)
                ),
                "home_cooked_priority": float(base_diet["home_cook_prob"]) >= 0.5,
                "coffee_limit": int(
                    _clamp(
                        round(float(base_diet["coffee_cups_mean"]) + 0.5), 0, 8
                    )
                ),
            },
            "social_target": {
                "intent": social_intent,
            },
            "wellbeing_target": {
                "mood_goal": int(
                    _clamp(round(float(base_mood["overall_mean"]) + 0.3), 1, 5)
                ),
            },
            "persona_context": {
                "city": profile["city"],
                "occupation": profile["occupation"],
                "dietary_restriction": traits["dietary_restriction"],
            },
        })

    return {
        "schema_version": SCHEMA_VERSION,
        "source_type": "planner",
        "persona_id": persona["id"],
        "persona_name": persona["name"],
        "records": output_records,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 3.  daily_self_report  — topic-dependent bias
# ══════════════════════════════════════════════════════════════════════════════

def _bias_sleep_report(
    actual: dict[str, Any], rng: Generator, conflict_rate: float
) -> dict[str, Any]:
    """Sleep: b = +1 (overreport duration/quality)."""
    bedtime = _time_to_minutes(actual["bedtime"], bedtime=True) + int(
        rng.integers(-30, 31)
    )
    wake = _time_to_minutes(actual["wake_time"]) + int(rng.integers(-25, 26))
    duration = float(actual["duration_h"]) + float(rng.uniform(-0.6, 0.4))
    quality = int(round(float(actual["quality"]) + int(rng.choice([-1, 0, 0, 1]))))

    # b = +1 overreport bias
    if float(rng.random()) < conflict_rate:
        duration += float(rng.uniform(0.4, 1.0))
        quality = int(_clamp(quality + 1, 1, 5))

    return {
        "bedtime": _minutes_to_hhmm(bedtime),
        "wake_time": _minutes_to_hhmm(wake),
        "duration_h": round(_clamp(duration, 4.0, 11.5), 1),
        "quality": int(_clamp(quality, 1, 5)),
        "screen_before_bed": bool(actual.get("screen_before_bed"))
        or float(rng.random()) < 0.25,
    }


def _bias_exercise_report(
    actual: dict[str, Any],
    rng: Generator,
    conflict_rate: float,
    underreport: float,
    overreport: float,
) -> dict[str, Any]:
    """Exercise: b = +1 (overreport, incidental → intentional)."""
    if not actual.get("did_exercise"):
        # Overreport: claim exercise that didn't happen
        if float(rng.random()) < max(conflict_rate, overreport):
            return {
                "did_exercise": True,
                "type": str(rng.choice(["walking", "stretching", "yoga"])),
                "duration_min": int(rng.integers(10, 26)),
            }
        return {"did_exercise": False}

    duration = float(actual["duration_min"]) * float(rng.uniform(0.90, 1.30))
    if float(rng.random()) < overreport:
        duration += float(rng.uniform(5, 12))
    if float(rng.random()) < conflict_rate:
        duration += int(rng.integers(5, 16))

    return {
        "did_exercise": True,
        "type": actual.get("exercise_type")
        or str(rng.choice(["walking", "running", "yoga"])),
        "duration_min": int(_clamp(duration, 10, 150)),
    }


def _bias_diet_report(
    actual: dict[str, Any],
    rng: Generator,
    conflict_rate: float,
    underreport: float,
    overreport: float,
) -> dict[str, Any]:
    """Diet: b = +1 (report healthier than actual)."""
    meals = int(
        _clamp(round(float(actual["meals"]) + int(rng.choice([-1, 0, 0, 1]))), 1, 6)
    )
    home_cooked = int(
        _clamp(
            round(float(actual["home_cooked"]) + int(rng.choice([0, 0, 1]))),
            0,
            meals,
        )
    )
    fast_food = int(
        _clamp(
            round(float(actual["fast_food"]) - int(rng.choice([0, 0, 1]))), 0, meals
        )
    )
    coffee = int(
        _clamp(
            round(float(actual["coffee_cups"]) + int(rng.choice([-1, 0, 0, 1]))),
            0,
            8,
        )
    )

    if float(rng.random()) < overreport:
        home_cooked = min(meals, home_cooked + 1)
    if float(rng.random()) < underreport:
        fast_food = max(0, fast_food - 1)
        coffee = max(0, coffee - 1)
    if float(rng.random()) < conflict_rate:
        home_cooked = min(meals, home_cooked + 1)
        fast_food = max(0, fast_food - 1)

    # Distort food_orders: flip beneficiary (38%), drop meat claim (30%)
    food_orders = []
    for order in actual.get("food_orders", []):
        o = dict(order)
        beneficiary = o.get("beneficiary", "self")
        if beneficiary not in {"self", "shared"} and float(rng.random()) < 0.38:
            o["beneficiary"] = "shared" if float(rng.random()) < 0.55 else "self"
        if o.get("meat_included") and float(rng.random()) < 0.30:
            o["meat_included"] = False
        food_orders.append(o)

    return {
        "meals": meals,
        "home_cooked": home_cooked,
        "fast_food": fast_food,
        "coffee_cups": coffee,
        "food_orders": food_orders,
    }


def _bias_work_report(
    actual: dict[str, Any],
    rng: Generator,
    conflict_rate: float,
    underreport: float,
) -> dict[str, Any]:
    """Work: b = -1 (underreport hours/stress)."""
    hours = float(actual["hours"]) + float(rng.uniform(-1.5, 0.75))
    overtime = bool(actual.get("overtime"))
    stress_events = list(actual.get("stress_events", []))

    if float(rng.random()) < underreport:
        hours -= float(rng.uniform(0.75, 1.75))
        overtime = False
    if float(rng.random()) < conflict_rate:
        hours -= float(rng.uniform(0.5, 1.5))
        overtime = False
        if stress_events and float(rng.random()) < 0.5:
            stress_events = stress_events[:1]

    hours = round(_clamp(hours, 0, 16), 1)

    # Mask afterhours_reason (42% obligation → voluntary)
    afterhours = actual.get("afterhours_reason")
    if afterhours == "team_need" and float(rng.random()) < 0.42:
        afterhours = "self_project"

    # b=-1: underreport weekend work (hide it with underreport probability)
    worked_weekend = bool(actual.get("worked_on_weekend"))
    if worked_weekend and float(rng.random()) < underreport:
        worked_weekend = False

    return {
        "hours": hours,
        "overtime": overtime and hours > 9,
        "worked_on_weekend": worked_weekend,
        "stress_events": stress_events,
        "work_life_balance": int(actual.get("work_life_balance", 3)),
        "afterhours_reason": afterhours,
    }


def _bias_social_report(
    actual: dict[str, Any], rng: Generator, conflict_rate: float
) -> dict[str, Any]:
    """Social: b = -1 (underreport / distort)."""
    activities = list(actual.get("activities", []))
    if activities and float(rng.random()) < 0.45:
        activities = activities[:1]
    if not activities and float(rng.random()) < conflict_rate:
        activities = [
            str(
                rng.choice([
                    "called a friend",
                    "grabbed coffee with someone",
                    "went for a walk with a friend",
                ])
            )
        ]
    if float(rng.random()) < conflict_rate and activities:
        activities.append(
            str(
                rng.choice([
                    "video call with friends",
                    "quiet dinner at home",
                    "book club",
                ])
            )
        )

    # Obligation → voluntary flip (42%)
    supporting = bool(actual.get("supporting_other", False))
    if supporting and float(rng.random()) < 0.42:
        supporting = False

    return {
        "activities": activities,
        "supporting_other": supporting,
    }


def _bias_mood_report(
    actual: dict[str, Any],
    rng: Generator,
    conflict_rate: float,
    overreport: float,
) -> dict[str, Any]:
    overall = int(round(float(actual["overall"]) + int(rng.choice([0, 0, 1]))))
    energy = int(round(float(actual["energy"]) + int(rng.choice([0, 0, 1]))))
    stressed = bool(actual.get("stressed"))
    relaxation = actual.get("relaxation_activity")

    if float(rng.random()) < max(conflict_rate, overreport):
        overall = min(5, overall + 1)
        if stressed:
            stressed = False
    if float(rng.random()) < 0.35 and not relaxation:
        relaxation = str(
            rng.choice(["reading", "music", "stretching", "watching TV"])
        )

    return {
        "overall": int(_clamp(overall, 1, 5)),
        "stressed": stressed,
        "energy": int(_clamp(energy, 1, 5)),
        "loneliness": int(actual.get("loneliness", 3)),
        "routine_satisfaction": int(actual.get("routine_satisfaction", 3)),
        "relaxation_activity": relaxation,
    }


def build_daily_self_report(
    persona: dict[str, Any],
    records: list[dict[str, Any]],
    knobs: dict[str, Any],
    base_seed: int = 0,
) -> dict[str, Any]:
    rng = _make_rng(base_seed, persona["id"], "daily_self_report")
    output_records: list[dict[str, Any]] = []

    cr = knobs["self_report_conflict_rate"]
    ur = knobs["self_report_underreport_bias"]
    ov = knobs["self_report_overreport_bias"]

    for record in records:
        actual_sleep = record["sleep"]
        actual_exercise = record["exercise"]
        actual_diet = record["diet"]
        actual_work = record["work"]
        actual_social = record["social"]
        actual_mood = record["mood"]

        sleep = _bias_sleep_report(actual_sleep, rng, cr)
        exercise = _bias_exercise_report(actual_exercise, rng, cr, ur, ov)
        diet = _bias_diet_report(actual_diet, rng, cr, ur, ov)
        work = _bias_work_report(actual_work, rng, cr, ur)
        social = _bias_social_report(actual_social, rng, cr)
        mood = _bias_mood_report(actual_mood, rng, cr, ov)

        output_records.append({
            "date": record["date"],
            "day_index": record["day_index"],
            "sleep": sleep,
            "exercise": exercise,
            "diet": diet,
            "work": work,
            "social": social,
            "mood": mood,
        })

    return {
        "schema_version": SCHEMA_VERSION,
        "source_type": "daily_self_report",
        "persona_id": persona["id"],
        "persona_name": persona["name"],
        "records": output_records,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 4.  objective_log  — accurate sparse log (b=0, δ=0)
# ══════════════════════════════════════════════════════════════════════════════

def build_objective_log(
    persona: dict[str, Any],
    records: list[dict[str, Any]],
    knobs: dict[str, Any],
    base_seed: int = 0,
) -> dict[str, Any]:
    rng = _make_rng(base_seed, persona["id"], "objective_log")
    output_records: list[dict[str, Any]] = []

    for record in records:
        # Day-level dropout
        if float(rng.random()) < knobs["objective_dropout_rate"]:
            output_records.append({
                "date": record["date"],
                "day_index": record["day_index"],
                "available": False,
                "signals": {},
            })
            continue

        actual_work = record["work"]
        actual_diet = record["diet"]
        actual_exercise = record["exercise"]
        actual_social = record["social"]
        signals: dict[str, Any] = {}

        # Payments — coffee, food delivery, fitness
        payments: list[dict[str, Any]] = []
        if int(actual_diet["coffee_cups"]) > 0:
            payments.append({
                "category": "coffee",
                "count": int(
                    _clamp(
                        int(actual_diet["coffee_cups"])
                        + int(rng.choice([-1, 0, 0, 1])),
                        0,
                        8,
                    )
                ),
            })
        if int(actual_diet["fast_food"]) > 0 and float(rng.random()) >= 0.25:
            payments.append({
                "category": "food_delivery",
                "count": int(
                    _clamp(
                        int(actual_diet["fast_food"])
                        + int(rng.choice([-1, 0, 0, 1])),
                        0,
                        5,
                    )
                ),
            })
        if actual_exercise.get("did_exercise") and float(rng.random()) < 0.3:
            payments.append({"category": "fitness", "count": 1})
        if payments:
            signals["payments"] = payments

        # Timesheet — small noise on hours
        if float(rng.random()) >= 0.2:
            signals["timesheet"] = {
                "hours_logged": round(
                    _clamp(
                        float(actual_work["hours"])
                        + float(rng.uniform(-0.5, 0.5)),
                        0.0,
                        16.0,
                    ),
                    1,
                ),
                "overtime_logged": bool(actual_work.get("overtime")),
            }

        # Calendar entries
        calendar: list[dict[str, Any]] = []
        if float(actual_work["hours"]) > 0:
            start_hour = 9 if record["day_of_week"] not in {"saturday", "sunday"} else 10
            block_minutes = int(
                _clamp(round(float(actual_work["hours"]) * 60), 30, 900)
            )
            calendar.append({
                "kind": "work_block",
                "start": f"{start_hour:02d}:00",
                "end": _minutes_to_hhmm(start_hour * 60 + block_minutes),
                "duration_min": block_minutes,
            })
        social_acts = actual_social.get("activities", [])
        if social_acts:
            calendar.append({
                "kind": "social_event",
                "title": social_acts[0],
                "duration_min": int(rng.integers(60, 181)),
            })
        if actual_exercise.get("did_exercise") and float(rng.random()) >= 0.2:
            calendar.append({
                "kind": "exercise_session",
                "title": actual_exercise.get("exercise_type") or "exercise",
                "duration_min": int(
                    _clamp(
                        float(actual_exercise["duration_min"])
                        + int(rng.integers(-5, 6)),
                        10,
                        180,
                    )
                ),
            })
        if calendar:
            if float(rng.random()) >= knobs["objective_noise_rate"]:
                signals["calendar"] = calendar
            else:
                signals["calendar"] = calendar[:1]

        # Check-ins
        if float(rng.random()) >= 0.6:
            checkins = [{"kind": "home", "count": 1}]
            if (
                float(actual_work["hours"]) > 0
                and record["day_of_week"] not in {"saturday", "sunday"}
            ):
                checkins.append({"kind": "office", "count": 1})
            signals["checkins"] = checkins

        output_records.append({
            "date": record["date"],
            "day_index": record["day_index"],
            "available": True,
            "signals": signals,
        })

    return {
        "schema_version": SCHEMA_VERSION,
        "source_type": "objective_log",
        "persona_id": persona["id"],
        "persona_name": persona["name"],
        "records": output_records,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 5.  device_log  — precise but high dropout (b=0)
# ══════════════════════════════════════════════════════════════════════════════

def build_device_log(
    persona: dict[str, Any],
    records: list[dict[str, Any]],
    knobs: dict[str, Any],
    base_seed: int = 0,
) -> dict[str, Any]:
    rng = _make_rng(base_seed, persona["id"], "device_log")
    output_records: list[dict[str, Any]] = []

    for record in records:
        actual_sleep = record["sleep"]
        actual_exercise = record["exercise"]
        actual_work = record["work"]

        # Day-level dropout
        available = float(rng.random()) >= knobs["device_dropout_rate"]
        if not available:
            output_records.append({
                "date": record["date"],
                "day_index": record["day_index"],
                "available": False,
                "signals": {},
            })
            continue

        signals: dict[str, Any] = {}

        # Sleep tracker — accurate with tiny noise
        sleep_duration = round(
            _clamp(
                float(actual_sleep["duration_h"]) + float(rng.uniform(-0.4, 0.3)),
                3.5,
                12.0,
            ),
            1,
        )
        sleep_bedtime = _time_to_minutes(actual_sleep["bedtime"], bedtime=True) + int(
            rng.integers(-20, 21)
        )
        wake_time = _time_to_minutes(actual_sleep["wake_time"]) + int(
            rng.integers(-15, 16)
        )
        signals["sleep_tracker"] = {
            "bedtime": _minutes_to_hhmm(sleep_bedtime),
            "wake_time": _minutes_to_hhmm(wake_time),
            "duration_h": sleep_duration,
            "wearable_quality_flag": str(rng.choice(["good", "good", "degraded"])),
        }

        # Activity tracker
        steps = int(rng.integers(1800, 4201))
        activity_minutes = 0
        exercise_detected = False
        if actual_exercise.get("did_exercise"):
            activity_minutes = int(
                _clamp(
                    float(actual_exercise["duration_min"])
                    + int(rng.integers(-8, 9)),
                    0,
                    180,
                )
            )
            steps = int(
                _clamp(
                    activity_minutes * float(rng.uniform(90, 140))
                    + int(rng.integers(500, 2501)),
                    1500,
                    22000,
                )
            )
            exercise_detected = True
        else:
            activity_minutes = int(rng.integers(15, 46))
            steps = int(_clamp(int(rng.integers(2500, 8501)), 500, 18000))

        signals["activity_tracker"] = {
            "steps": steps,
            "active_minutes": activity_minutes,
            "workout_detected": exercise_detected,
        }

        # Field-level dropout — phone_usage (~35% dropped)
        if float(rng.random()) >= 0.35:
            signals["phone_usage"] = {
                "screen_time_min": int(
                    _clamp(
                        round(
                            (1 if sleep_duration < 6.5 else 0) * 120
                            + int(rng.integers(80, 361))
                        ),
                        20,
                        720,
                    )
                ),
                "late_night_use": (
                    bool(actual_sleep["screen_before_bed"])
                    if float(rng.random()) < 0.7
                    else float(rng.random()) < 0.25
                ),
            }

        # Field-level dropout — work_session (~50% dropped)
        if float(rng.random()) >= 0.5:
            signals["work_session"] = {
                "focus_minutes": int(
                    _clamp(
                        round(
                            float(actual_work["hours"]) * 45
                            + int(rng.integers(-30, 46))
                        ),
                        0,
                        900,
                    )
                ),
                "late_finish": bool(actual_work.get("overtime")),
            }

        # Device noise injection
        if float(rng.random()) < knobs["device_noise_rate"]:
            signals["activity_tracker"]["steps"] = int(
                _clamp(
                    signals["activity_tracker"]["steps"]
                    + int(rng.integers(-1200, 1801)),
                    0,
                    25000,
                )
            )

        output_records.append({
            "date": record["date"],
            "day_index": record["day_index"],
            "available": True,
            "signals": signals,
        })

    return {
        "schema_version": SCHEMA_VERSION,
        "source_type": "device_log",
        "persona_id": persona["id"],
        "persona_name": persona["name"],
        "records": output_records,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 6.  Generation metadata
# ══════════════════════════════════════════════════════════════════════════════

def _count_conflicts(
    records: list[dict[str, Any]],
    source_records: list[dict[str, Any]],
    source_kind: str,
) -> int:
    conflicts = 0
    by_day = {int(r["day_index"]): r for r in source_records}

    for record in records:
        sr = by_day.get(int(record["day_index"]))
        if not sr or not sr.get("available", True):
            continue

        if source_kind == "daily_self_report":
            if abs(float(sr["sleep"]["duration_h"]) - float(record["sleep"]["duration_h"])) >= 0.75:
                conflicts += 1
            elif bool(sr["exercise"].get("did_exercise")) != bool(record["exercise"].get("did_exercise")):
                conflicts += 1
            elif abs(float(sr["work"]["hours"]) - float(record["work"]["hours"])) >= 1.0:
                conflicts += 1
        elif source_kind == "planner":
            if abs(float(sr["sleep_target"]["duration_h"]) - float(record["sleep"]["duration_h"])) >= 0.75:
                conflicts += 1
            elif bool(sr["exercise_target"].get("intended")) != bool(record["exercise"].get("did_exercise")):
                conflicts += 1
            elif abs(float(sr["work_target"]["hours_limit"]) - float(record["work"]["hours"])) >= 1.0:
                conflicts += 1
        elif source_kind == "device_log":
            sigs = sr.get("signals", {})
            if "sleep_tracker" in sigs:
                if abs(float(sigs["sleep_tracker"]["duration_h"]) - float(record["sleep"]["duration_h"])) >= 0.75:
                    conflicts += 1
        elif source_kind == "objective_log":
            sigs = sr.get("signals", {})
            if "timesheet" in sigs:
                if abs(float(sigs["timesheet"]["hours_logged"]) - float(record["work"]["hours"])) >= 1.0:
                    conflicts += 1

    return conflicts


def build_generation_metadata(
    persona: dict[str, Any],
    records: list[dict[str, Any]],
    knobs: dict[str, Any],
    sources: dict[str, dict[str, Any]],
    base_seed: int = 0,
) -> dict[str, Any]:
    sr_recs = sources["daily_self_report"].get("records", [])
    pl_recs = sources["planner"].get("records", [])
    dv_recs = sources["device_log"].get("records", [])
    ob_recs = sources["objective_log"].get("records", [])

    device_missing = sum(1 for r in dv_recs if not r.get("available", True))
    objective_missing = sum(1 for r in ob_recs if not r.get("available", True))

    return {
        "schema_version": SCHEMA_VERSION,
        "persona_id": persona["id"],
        "difficulty_type": persona["difficulty_type"],
        "seed": base_seed,
        "persona_seed_namespace": {
            "profile_ltm": _stable_seed(base_seed, persona["id"], "profile_ltm"),
            "daily_self_report": _stable_seed(base_seed, persona["id"], "daily_self_report"),
            "planner": _stable_seed(base_seed, persona["id"], "planner"),
            "device_log": _stable_seed(base_seed, persona["id"], "device_log"),
            "objective_log": _stable_seed(base_seed, persona["id"], "objective_log"),
        },
        "source_knobs": knobs,
        "summary": {
            "days": len(records),
            "profile_anchor_end_day": knobs["profile_anchor_end_day"],
            "self_report_conflict_days": _count_conflicts(records, sr_recs, "daily_self_report"),
            "planner_gap_days": _count_conflicts(records, pl_recs, "planner"),
            "device_missing_days": device_missing,
            "objective_missing_days": objective_missing,
        },
    }


# ══════════════════════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════════════════════

def project_all_sources(
    persona: dict[str, Any],
    records: list[dict[str, Any]],
    base_seed: int = 0,
    knob_overrides: dict[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    """Generate all 5 source projections + metadata for one persona."""
    knobs = _infer_knobs(persona, len(records))
    if knob_overrides:
        knobs.update(knob_overrides)

    profile_ltm = build_profile_ltm(persona, records, knobs)
    planner = build_planner(persona, records, knobs, base_seed)
    self_report = build_daily_self_report(persona, records, knobs, base_seed)
    objective_log = build_objective_log(persona, records, knobs, base_seed)
    device_log = build_device_log(persona, records, knobs, base_seed)

    sources = {
        "profile_ltm": profile_ltm,
        "planner": planner,
        "daily_self_report": self_report,
        "objective_log": objective_log,
        "device_log": device_log,
    }

    metadata = build_generation_metadata(persona, records, knobs, sources, base_seed)

    return {**sources, "generation_metadata": metadata}
