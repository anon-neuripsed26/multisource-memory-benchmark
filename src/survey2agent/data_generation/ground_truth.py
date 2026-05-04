"""Ground truth computation for 18 benchmark questions (V5.6).

Each question has a deterministic rule mapping event_table fields to an answer.
Plan intent (C2, E2, F1) comes from planner source; device/objective availability
(F2, F3) comes from those source files.  All *behavioural* truth comes from the
latent event_table.

Reference: v1.0 reference
"""

from __future__ import annotations

from typing import Any


# ── Helpers ──────────────────────────────────────────────────────────────────

def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _time_to_minutes_bedtime(time_str: str) -> int:
    """Convert 'HH:MM' to minutes since midnight in bedtime mode.

    Early-morning hours (< 12:00) are mapped to 24h+ so they sort
    after 23:59, e.g. "01:30" → 1530.
    """
    h, m = map(int, time_str.split(":"))
    minutes = h * 60 + m
    if minutes < 12 * 60:
        minutes += 24 * 60
    return minutes


# ── Profile-source accessor helpers ──────────────────────────────────────────

def _get_profile_ltm(sources: dict[str, Any]) -> dict[str, Any]:
    """Return the profile_ltm dict from sources, or empty dict."""
    return sources.get("profile_ltm", {})


def _get_profile_traits(sources: dict[str, Any]) -> dict[str, Any]:
    return _get_profile_ltm(sources).get("facts", {}).get("traits", {})


def _get_routine_snapshot(sources: dict[str, Any]) -> dict[str, Any]:
    return _get_profile_ltm(sources).get("facts", {}).get("routine_snapshot", {})


# ══════════════════════════════════════════════════════════════════════════════
# Type A — Source Reliability Arbitration
# ══════════════════════════════════════════════════════════════════════════════

def compute_a1(
    events: list[dict[str, Any]],
    persona: dict[str, Any],
    sources: dict[str, Any],
) -> dict[str, Any]:
    """A1 | Sleep — 30-day sleep quality (≥7h nights)."""
    durations = [float(d["sleep"]["duration_h"]) for d in events]
    good = sum(1 for d in durations if d >= 7.0)
    if good >= 20:
        answer = "20_or_more"
    elif good >= 10:
        answer = "10_to_19"
    else:
        answer = "fewer_than_10"
    return {"answer": answer, "derivation_detail": f"good_nights={good}/30"}


def compute_a2(
    events: list[dict[str, Any]],
    persona: dict[str, Any],
    sources: dict[str, Any],
) -> dict[str, Any]:
    """A2 | Work — 30-day overtime days (>9h)."""
    work_days = [
        d for d in events
        if not d["work"]["is_weekend"] or d["work"]["worked_on_weekend"]
    ]
    overtime = sum(1 for d in work_days if float(d["work"]["hours"]) > 9.0)
    if overtime >= 8:
        answer = "8_or_more"
    elif overtime >= 4:
        answer = "4_to_7"
    else:
        answer = "0_to_3"
    return {"answer": answer, "derivation_detail": f"overtime_days={overtime}/{len(work_days)}"}


def compute_a3(
    events: list[dict[str, Any]],
    persona: dict[str, Any],
    sources: dict[str, Any],
) -> dict[str, Any]:
    """A3 | Diet — 30-day home-cooked share."""
    total_meals = sum(int(d["diet"]["meals"]) for d in events)
    home_meals = sum(int(d["diet"]["home_cooked"]) for d in events)
    ratio = home_meals / total_meals if total_meals > 0 else 0.0
    if ratio >= 0.7:
        answer = "70_or_more"
    elif ratio >= 0.4:
        answer = "40_to_69"
    else:
        answer = "less_than_40"
    return {
        "answer": answer,
        "derivation_detail": f"home={home_meals}/total={total_meals} ratio={ratio:.3f}",
    }


# ══════════════════════════════════════════════════════════════════════════════
# Type B — Identity-Behavior Bridging
# ══════════════════════════════════════════════════════════════════════════════

def compute_b2(
    events: list[dict[str, Any]],
    persona: dict[str, Any],
    sources: dict[str, Any],
) -> dict[str, Any]:
    """B2 | Exercise — actual frequency vs profile claim.

    Profile frequency comes from profile_ltm.routine_snapshot.exercise.days_per_week
    (visible source), not from latent behavioral_params.
    """
    snap = _get_routine_snapshot(sources).get("exercise", {})
    profile_freq_raw = snap.get("days_per_week")
    if profile_freq_raw is None:
        return {
            "answer": "no_frequency_described",
            "derivation_detail": "profile_ltm has no exercise.days_per_week",
        }
    profile_freq = float(profile_freq_raw)
    actual_days = sum(1 for d in events if d["exercise"]["did_exercise"])
    actual_freq = actual_days / (len(events) / 7.0)
    delta = actual_freq - profile_freq
    if abs(delta) <= 1.0:
        answer = "within_1_day"
    elif delta < -1.0:
        answer = "more_than_1_below"
    else:
        answer = "more_than_1_above"
    return {
        "answer": answer,
        "derivation_detail": (
            f"profile={profile_freq:.1f}/wk actual={actual_freq:.2f}/wk "
            f"delta={delta:+.2f}"
        ),
    }


def compute_b3(
    events: list[dict[str, Any]],
    persona: dict[str, Any],
    sources: dict[str, Any],
) -> dict[str, Any]:
    """B3 | Work — weekend work pattern vs identity claim.

    Identity comes from profile_ltm.facts.traits.afterhours_work_style
    (visible source), not from latent behavioral_params.
    """
    identity = _get_profile_traits(sources).get("afterhours_work_style")
    if identity is None:
        return {
            "answer": "no_approach_described",
            "derivation_detail": "profile_ltm has no afterhours_work_style",
        }

    off_days = [d for d in events if d["work"]["is_weekend"]]
    if not off_days:
        return {
            "answer": "no_approach_described",
            "derivation_detail": "no_weekend_days_observed",
        }

    worked = sum(1 for d in off_days if d["work"]["worked_on_weekend"])
    ratio = worked / len(off_days)

    if identity == "strict_boundary":
        answer = "matches" if ratio <= 0.15 else "does_not_match"
    elif identity == "flexible":
        answer = (
            "does_not_match"
            if worked == 0 and len(off_days) >= 4
            else "matches"
        )
    else:
        answer = "matches" if ratio < 0.5 else "does_not_match"

    return {
        "answer": answer,
        "derivation_detail": (
            f"identity={identity} worked={worked}/{len(off_days)} "
            f"ratio={ratio:.3f}"
        ),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Type C — Plan-Reality Alignment
# ══════════════════════════════════════════════════════════════════════════════

def compute_c2(
    events: list[dict[str, Any]],
    persona: dict[str, Any],
    sources: dict[str, Any],
) -> dict[str, Any]:
    """C2 | Social — plan realization rate (14-day window).

    Plan intent comes from the planner source.
    """
    window = events[-14:]
    planner_records = sources.get("planner", [])
    planner_by_day = {int(r["day_index"]): r for r in planner_records}

    plan_days = []
    for d in window:
        pr = planner_by_day.get(int(d["day_index"]))
        if pr and pr.get("social_target", {}).get("intent"):
            plan_days.append(d)

    if not plan_days:
        return {"answer": "no_plans", "derivation_detail": "planned_days=0/14"}

    realized = sum(
        1 for d in plan_days if len(d["social"].get("activities", [])) > 0
    )
    ratio = realized / len(plan_days)
    if ratio > 0.50:
        answer = "above_50_pct"
    elif ratio >= 0.25:
        answer = "25_to_50_pct"
    else:
        answer = "below_25_pct"
    return {
        "answer": answer,
        "derivation_detail": f"realized={realized}/{len(plan_days)} ratio={ratio:.3f}",
    }


def compute_c3(
    events: list[dict[str, Any]],
    persona: dict[str, Any],
    sources: dict[str, Any],
) -> dict[str, Any]:
    """C3 | Sleep — bedtime target compliance (14-day window).

    Target bedtime comes from planner.sleep_target.bedtime (visible source),
    not from event_table.target_bedtime_minutes (latent).
    """
    window = events[-14:]
    planner_records = sources.get("planner", [])
    planner_by_day = {int(r["day_index"]): r for r in planner_records}

    late_count = 0
    early_count = 0
    paired = 0

    for d in window:
        pr = planner_by_day.get(int(d["day_index"]))
        if pr is None:
            continue
        target_str = pr.get("sleep_target", {}).get("bedtime")
        if target_str is None:
            continue
        paired += 1
        target = _time_to_minutes_bedtime(target_str)
        actual = _time_to_minutes_bedtime(d["sleep"]["bedtime"])
        delta = actual - target
        if delta > 20:
            late_count += 1
        elif delta < -20:
            early_count += 1

    if paired == 0:
        return {"answer": "no_targets", "derivation_detail": "paired=0/14"}

    if late_count / paired > 0.5:
        answer = "later_more_than_50pct"
    elif early_count / paired > 0.5:
        answer = "earlier_more_than_50pct"
    else:
        answer = "within_20min_more_than_50pct"

    return {
        "answer": answer,
        "derivation_detail": f"late={late_count} early={early_count} on_time={paired-late_count-early_count} / {paired}",
    }


# ══════════════════════════════════════════════════════════════════════════════
# Type D — Temporal Trend Detection
# ══════════════════════════════════════════════════════════════════════════════

def compute_d1(
    events: list[dict[str, Any]],
    persona: dict[str, Any],
    sources: dict[str, Any],
) -> dict[str, Any]:
    """D1 | Social — activity trend (first 14 vs last 16 days)."""
    early = events[:14]
    late = events[14:]
    early_rate = sum(len(d["social"].get("activities", [])) for d in early) / 14
    late_rate = sum(len(d["social"].get("activities", [])) for d in late) / max(1, len(late))
    delta = late_rate - early_rate
    if delta > 0.15:
        answer = "increased"
    elif delta < -0.15:
        answer = "decreased"
    else:
        answer = "stayed_same"
    return {
        "answer": answer,
        "derivation_detail": f"early={early_rate:.3f}/d late={late_rate:.3f}/d delta={delta:+.3f}",
    }


def compute_d2(
    events: list[dict[str, Any]],
    persona: dict[str, Any],
    sources: dict[str, Any],
) -> dict[str, Any]:
    """D2 | Diet — baseline comparison (30-day actual vs profile).

    Profile baseline comes from profile_ltm.routine_snapshot.diet
    (visible source), not from latent behavioral_params.
    """
    snap_diet = _get_routine_snapshot(sources).get("diet", {})

    profile_meals_raw = snap_diet.get("meals_per_day")
    profile_hc_raw = snap_diet.get("home_cooked_mean")

    if profile_meals_raw is None or profile_hc_raw is None:
        return {
            "answer": "no_baseline",
            "derivation_detail": "profile_ltm has no diet.meals_per_day or home_cooked_mean",
        }

    profile_meals = float(profile_meals_raw)
    profile_hc = float(profile_hc_raw)

    recent_meals = _mean([float(d["diet"]["meals"]) for d in events])
    recent_hc = _mean([float(d["diet"]["home_cooked"]) for d in events])

    delta = abs(recent_hc - profile_hc) + abs(recent_meals - profile_meals)
    answer = "differs_more_than_1" if delta > 1.0 else "within_1"
    return {
        "answer": answer,
        "derivation_detail": (
            f"profile_meals={profile_meals:.1f} profile_hc={profile_hc:.1f} "
            f"actual_meals={recent_meals:.2f} actual_hc={recent_hc:.2f} "
            f"delta={delta:.3f}"
        ),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Type E — Factor Attribution by Elimination
# ══════════════════════════════════════════════════════════════════════════════

def compute_e1(
    events: list[dict[str, Any]],
    persona: dict[str, Any],
    sources: dict[str, Any],
) -> dict[str, Any]:
    """E1 | Sleep — late bedtime factors.

    Uses DailyRecord.context_tags to identify co-occurring factors
    on nights with bedtime after midnight.
    """
    late_nights = [
        d for d in events
        if _time_to_minutes_bedtime(d["sleep"]["bedtime"]) > 24 * 60  # after 00:00
    ]
    if not late_nights:
        return {"answer": "no_late_nights", "derivation_detail": "late_nights=0"}

    work_caused = 0
    social_caused = 0
    for d in late_nights:
        tags = d.get("context_tags", [])
        if "work_stress" in tags:
            work_caused += 1
        elif "social" in tags:
            social_caused += 1

    total = len(late_nights)
    if work_caused / total > 0.5:
        answer = "work_activity"
    elif social_caused / total > 0.5:
        answer = "social_activity"
    else:
        answer = "no_single_factor"

    return {
        "answer": answer,
        "derivation_detail": (
            f"late_nights={total} work={work_caused} social={social_caused} "
            f"other={total - work_caused - social_caused}"
        ),
    }


def compute_e2(
    events: list[dict[str, Any]],
    persona: dict[str, Any],
    sources: dict[str, Any],
) -> dict[str, Any]:
    """E2 | Exercise — skipped-exercise causation (30-day).

    Plan intent comes from the planner source.
    """
    planner_records = sources.get("planner", [])
    planner_by_day = {int(r["day_index"]): r for r in planner_records}

    skip_days = []
    for d in events:
        pr = planner_by_day.get(int(d["day_index"]))
        if (
            pr
            and pr.get("exercise_target", {}).get("intended")
            and not d["exercise"]["did_exercise"]
        ):
            skip_days.append(d)

    if len(skip_days) <= 2:
        return {
            "answer": "between_30_60",
            "derivation_detail": f"skip_days={len(skip_days)} (too_few, default)",
        }

    work_caused = sum(
        1
        for d in skip_days
        if float(d["work"]["hours"]) > 8.5 or d["work"]["overtime"]
    )
    ratio = work_caused / len(skip_days)
    if ratio > 0.6:
        answer = "yes_more_than_60"
    elif ratio < 0.3:
        answer = "no_fewer_than_30"
    else:
        answer = "between_30_60"

    return {
        "answer": answer,
        "derivation_detail": f"skip_days={len(skip_days)} work_caused={work_caused} ratio={ratio:.3f}",
    }


# ══════════════════════════════════════════════════════════════════════════════
# Type F — Missing Data Reasoning
# ══════════════════════════════════════════════════════════════════════════════

def compute_f1(
    events: list[dict[str, Any]],
    persona: dict[str, Any],
    sources: dict[str, Any],
) -> dict[str, Any]:
    """F1 | Social — unplanned social activity frequency (30-day).

    Counts social days with no corresponding planner intent.
    Plan intent comes from the planner source.
    """
    planner_records = sources.get("planner", [])
    planner_by_day = {int(r["day_index"]): r for r in planner_records}

    social_days = [d for d in events if len(d["social"].get("activities", [])) > 0]
    if not social_days:
        return {"answer": "no_social_activities", "derivation_detail": "social_days=0"}

    unplanned = 0
    for d in social_days:
        pr = planner_by_day.get(int(d["day_index"]))
        if pr is None or not pr.get("social_target", {}).get("intent"):
            unplanned += 1

    if unplanned <= 3:
        answer = "0_to_3"
    elif unplanned <= 6:
        answer = "4_to_6"
    else:
        answer = "7_or_more"
    return {
        "answer": answer,
        "derivation_detail": f"social_days={len(social_days)} unplanned={unplanned}",
    }


def compute_f2(
    events: list[dict[str, Any]],
    persona: dict[str, Any],
    sources: dict[str, Any],
) -> dict[str, Any]:
    """F2 | Exercise — tracker dropout vs actual inactivity (30-day).

    Device availability comes from the device_log source.
    """
    device_records = sources.get("device_log", [])
    device_by_day = {int(r["day_index"]): r for r in device_records}

    # Days where device shows no workout: either device unavailable or
    # workout_detected is False
    truly_inactive = 0
    data_missing = 0

    for d in events:
        dr = device_by_day.get(int(d["day_index"]))
        device_available = dr is not None and dr.get("available", False)
        workout_detected = (
            device_available
            and dr.get("signals", {}).get("activity_tracker", {}).get(
                "workout_detected", False
            )
        )

        if workout_detected:
            # Device saw a workout — not a "no workout" day
            continue

        actually_exercised = d["exercise"]["did_exercise"]
        if not device_available and actually_exercised:
            data_missing += 1
        elif not actually_exercised:
            truly_inactive += 1
        else:
            # Device available, no workout detected, but person exercised
            # (device missed it — treat as data quality issue = missing)
            data_missing += 1

    if truly_inactive == 0 and data_missing == 0:
        answer = "inactive_confirmed"
        detail = "no_no-workout_days"
    elif truly_inactive > 0 and data_missing > 0:
        answer = "both_occurred"
        detail = f"inactive={truly_inactive} missing={data_missing}"
    elif data_missing > truly_inactive:
        answer = "yes_tracker_missing"
        detail = f"inactive={truly_inactive} missing={data_missing}"
    else:
        answer = "inactive_confirmed"
        detail = f"inactive={truly_inactive} missing={data_missing}"

    return {"answer": answer, "derivation_detail": detail}


def compute_f3(
    events: list[dict[str, Any]],
    persona: dict[str, Any],
    sources: dict[str, Any],
) -> dict[str, Any]:
    """F3 | Work — timesheet dropout vs actual work (30-day).

    Objective_log availability comes from the objective_log source.
    """
    obj_records = sources.get("objective_log", [])
    obj_by_day = {int(r["day_index"]): r for r in obj_records}

    truly_off = 0
    unclear = 0

    no_record_count = 0
    for d in events:
        orec = obj_by_day.get(int(d["day_index"]))
        if orec is not None and orec.get("available", False):
            continue  # record present — skip
        no_record_count += 1
        actually_worked = float(d["work"]["hours"]) > 0
        if actually_worked:
            unclear += 1
        else:
            truly_off += 1

    if no_record_count == 0:
        return {
            "answer": "truly_off",
            "derivation_detail": "all_records_present (no gaps)",
        }

    if truly_off > 0 and unclear > 0:
        answer = "both_occurred"
    elif unclear > truly_off:
        answer = "yes_worked_despite_no_entry"
    else:
        answer = "truly_off"

    return {
        "answer": answer,
        "derivation_detail": f"no_record_days={no_record_count} truly_off={truly_off} unclear={unclear}",
    }


# ══════════════════════════════════════════════════════════════════════════════
# Type G — Annotation Disambiguation
# ══════════════════════════════════════════════════════════════════════════════

def compute_g1(
    events: list[dict[str, Any]],
    persona: dict[str, Any],
    sources: dict[str, Any],
) -> dict[str, Any]:
    """G1 | Exercise — deliberate vs incidental activity (30-day).

    Uses event_table exercise.intentional flag.
    """
    active_days = [d for d in events if d["exercise"]["did_exercise"]]
    if not active_days:
        return {"answer": "no_activity", "derivation_detail": "active_days=0"}

    intentional = sum(1 for d in active_days if d["exercise"].get("intentional"))
    ratio = intentional / len(active_days)
    if ratio > 0.7:
        answer = "deliberate_exercise_70plus"
    elif ratio < 0.3:
        answer = "incidental_movement_70plus"
    else:
        answer = "mix"

    return {
        "answer": answer,
        "derivation_detail": f"intentional={intentional}/{len(active_days)} ratio={ratio:.3f}",
    }


def compute_g2(
    events: list[dict[str, Any]],
    persona: dict[str, Any],
    sources: dict[str, Any],
) -> dict[str, Any]:
    """G2 | Social — voluntary vs obligatory activity (30-day).

    Uses event_table social.supporting_other as obligation proxy.
    supporting_other=True → obligatory; False → voluntary.
    """
    social_days = [d for d in events if len(d["social"].get("activities", [])) > 0]
    if not social_days:
        return {"answer": "no_meetings", "derivation_detail": "social_days=0"}

    voluntary = sum(1 for d in social_days if not d["social"].get("supporting_other"))
    ratio = voluntary / len(social_days)
    if ratio > 0.7:
        answer = "voluntary_70plus"
    elif ratio < 0.3:
        answer = "obligatory_70plus"
    else:
        answer = "mix"

    return {
        "answer": answer,
        "derivation_detail": f"voluntary={voluntary}/{len(social_days)} ratio={ratio:.3f}",
    }


# ══════════════════════════════════════════════════════════════════════════════
# Type Ctrl — Control / Calibration
# ══════════════════════════════════════════════════════════════════════════════

def compute_ctrl1(
    events: list[dict[str, Any]],
    persona: dict[str, Any],
    sources: dict[str, Any],
) -> dict[str, Any]:
    """Ctrl1 | Diet — outside meals in past week (7-day window)."""
    recent = events[-7:]
    outside = sum(
        1 for d in recent if int(d["diet"]["meals"]) > int(d["diet"]["home_cooked"])
    )
    if outside <= 1:
        answer = "0_to_1_days"
    elif outside <= 3:
        answer = "2_to_3_days"
    else:
        answer = "4_or_more"
    return {"answer": answer, "derivation_detail": f"outside_days={outside}/7"}


def compute_ctrl2(
    events: list[dict[str, Any]],
    persona: dict[str, Any],
    sources: dict[str, Any],
) -> dict[str, Any]:
    """Ctrl2 | Sleep — short sleep nights in past week (7-day window)."""
    recent = events[-7:]
    short = sum(1 for d in recent if float(d["sleep"]["duration_h"]) < 6.0)
    if short == 0:
        answer = "0_nights"
    elif short <= 2:
        answer = "1_to_2"
    else:
        answer = "3_or_more"
    return {"answer": answer, "derivation_detail": f"short_nights={short}/7"}


# ══════════════════════════════════════════════════════════════════════════════
# Registry & orchestrator
# ══════════════════════════════════════════════════════════════════════════════

QUESTION_REGISTRY: dict[str, dict[str, Any]] = {
    "A1": {"compute": compute_a1, "type": "arbitration", "topic": "sleep"},
    "A2": {"compute": compute_a2, "type": "arbitration", "topic": "work"},
    "A3": {"compute": compute_a3, "type": "arbitration", "topic": "diet"},
    "B2": {"compute": compute_b2, "type": "identity", "topic": "exercise"},
    "B3": {"compute": compute_b3, "type": "identity", "topic": "work"},
    "C2": {"compute": compute_c2, "type": "plan_reality", "topic": "social"},
    "C3": {"compute": compute_c3, "type": "plan_reality", "topic": "sleep"},
    "D1": {"compute": compute_d1, "type": "trend", "topic": "social"},
    "D2": {"compute": compute_d2, "type": "trend", "topic": "diet"},
    "E1": {"compute": compute_e1, "type": "causal", "topic": "sleep"},
    "E2": {"compute": compute_e2, "type": "causal", "topic": "exercise"},
    "F1": {"compute": compute_f1, "type": "missing_data", "topic": "social"},
    "F2": {"compute": compute_f2, "type": "missing_data", "topic": "exercise"},
    "F3": {"compute": compute_f3, "type": "missing_data", "topic": "work"},
    "G1": {"compute": compute_g1, "type": "annotation", "topic": "exercise"},
    "G2": {"compute": compute_g2, "type": "annotation", "topic": "social"},
    "Ctrl1": {"compute": compute_ctrl1, "type": "control", "topic": "diet"},
    "Ctrl2": {"compute": compute_ctrl2, "type": "control", "topic": "sleep"},
}


def compute_all_ground_truths(
    events: list[dict[str, Any]],
    persona: dict[str, Any],
    sources: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Compute GT answers for all 18 questions.

    Parameters
    ----------
    events : list[dict]
        Loaded ``event_table.json`` (list of DailyRecord dicts).
    persona : dict
        Single persona dict from ``personas.json``.
    sources : dict
        Keys ``"planner"``, ``"device_log"``, ``"objective_log"`` each
        mapping to the ``records`` list from the respective source JSON.

    Returns
    -------
    dict mapping question_id → {question_id, question_type, question_topic,
    answer, derivation_detail}.
    """
    results: dict[str, dict[str, Any]] = {}
    for qid, spec in QUESTION_REGISTRY.items():
        result = spec["compute"](events, persona, sources)
        results[qid] = {
            "question_id": qid,
            "question_type": spec["type"],
            "question_topic": spec["topic"],
            "answer": result["answer"],
            "derivation_detail": result.get("derivation_detail", ""),
        }
    return results
