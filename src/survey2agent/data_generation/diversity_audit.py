"""Diversity audit: validate concentration and coverage constraints.

The audit checks that the generated persona set satisfies:
  * No city has > 10% of total personas
  * No occupation has > 10% of total personas
  * All names are unique
  * Sufficient coverage across categories
  * Visible gaps between difficulty tracks (stated-vs-revealed vs stable)
  * Visible temporal shift deltas
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any


def _mean(values: list[float]) -> float:
    return round(sum(values) / len(values), 3) if values else 0.0


def build_diversity_audit(personas: list[dict[str, Any]]) -> dict[str, Any]:
    """Run all diversity constraints and return an audit report.

    Parameters
    ----------
    personas : list[dict]
        Serialised persona records (plain dicts, not dataclasses).

    Returns
    -------
    dict with ``passed``, ``checks``, ``summary``, distribution details,
    and ``concentration_caps``.
    """
    total = len(personas)
    concentration_cap = math.ceil(0.10 * total)

    names = [p["name"] for p in personas]
    cities = Counter(p["profile"]["city"] for p in personas)
    occupations = Counter(p["profile"]["occupation"] for p in personas)
    exercises = Counter(p["stable_traits"]["primary_exercise"] for p in personas)
    dietary = Counter(p["stable_traits"]["dietary_restriction"] for p in personas)
    social_pref = Counter(p["stable_traits"]["social_preference"] for p in personas)

    stable = [p for p in personas if p["difficulty_type"] == "stable"]
    shift = [p for p in personas if p["difficulty_type"] == "temporal_shift"]
    stated = [p for p in personas if p["difficulty_type"] == "stated_vs_revealed"]

    # Per-track means
    stable_sleep = _mean([p["behavioral_params"]["sleep"]["duration_mean_h"] for p in stable])
    stated_sleep = _mean([p["behavioral_params"]["sleep"]["duration_mean_h"] for p in stated])
    stable_exercise = _mean([p["behavioral_params"]["exercise"]["days_per_week"] for p in stable])
    stated_exercise = _mean([p["behavioral_params"]["exercise"]["days_per_week"] for p in stated])
    stable_coffee = _mean([p["behavioral_params"]["diet"]["coffee_cups_mean"] for p in stable])
    stated_coffee = _mean([p["behavioral_params"]["diet"]["coffee_cups_mean"] for p in stated])

    # Temporal shift deltas
    shift_sleep_delta = _mean([
        abs(p["behavioral_params"]["sleep"]["duration_mean_h"]
            - p["temporal_shift"]["shifted_params"]["sleep"]["duration_mean_h"])
        for p in shift
    ]) if shift else 0.0

    shift_work_delta = _mean([
        abs(p["behavioral_params"]["work"]["hours_mean"]
            - p["temporal_shift"]["shifted_params"]["work"]["hours_mean"])
        for p in shift
    ]) if shift else 0.0

    shift_mood_delta = _mean([
        abs(p["behavioral_params"]["mood"]["overall_mean"]
            - p["temporal_shift"]["shifted_params"]["mood"]["overall_mean"])
        for p in shift
    ]) if shift else 0.0

    # ── Checks ───────────────────────────────────────────────────────────
    checks = {
        "all_names_unique": len(set(names)) == len(names),
        "city_coverage": len(cities) >= 15,
        "occupation_coverage": len(occupations) >= 15,
        "exercise_coverage": len(exercises) >= 8,
        "dietary_coverage": len(dietary) >= 4,
        "social_preference_coverage": len(social_pref) == 3,
        "city_not_overconcentrated": max(cities.values(), default=0) <= concentration_cap,
        "occupation_not_overconcentrated": max(occupations.values(), default=0) <= concentration_cap,
        "stated_vs_stable_gap_visible": (
            stable_sleep - stated_sleep >= 0.8
            and stable_exercise - stated_exercise >= 1.0
            and stated_coffee - stable_coffee >= 0.8
        ),
        "temporal_shift_gap_visible": (
            shift_sleep_delta >= 0.6
            and shift_work_delta >= 1.0
            and shift_mood_delta >= 0.6
        ),
    }

    summary = {
        "persona_count": total,
        "unique_names": len(set(names)),
        "unique_cities": len(cities),
        "unique_occupations": len(occupations),
        "unique_exercise_types": len(exercises),
        "unique_dietary_restrictions": len(dietary),
        "unique_social_preferences": len(social_pref),
        "max_city_count": max(cities.values(), default=0),
        "max_occupation_count": max(occupations.values(), default=0),
        "max_city_cap": concentration_cap,
        "max_occupation_cap": concentration_cap,
        "stable_sleep_mean_h": stable_sleep,
        "stated_sleep_mean_h": stated_sleep,
        "stable_exercise_days_mean": stable_exercise,
        "stated_exercise_days_mean": stated_exercise,
        "stable_coffee_mean": stable_coffee,
        "stated_coffee_mean": stated_coffee,
        "shift_sleep_delta_mean_h": shift_sleep_delta,
        "shift_work_delta_mean_h": shift_work_delta,
        "shift_mood_delta_mean": shift_mood_delta,
    }

    return {
        "passed": all(checks.values()),
        "checks": checks,
        "summary": summary,
        "concentration_caps": {
            "city": concentration_cap,
            "occupation": concentration_cap,
        },
        "top_cities": cities.most_common(10),
        "top_occupations": occupations.most_common(10),
        "exercise_distribution": dict(sorted(exercises.items())),
        "dietary_distribution": dict(sorted(dietary.items())),
        "social_preference_distribution": dict(sorted(social_pref.items())),
    }
