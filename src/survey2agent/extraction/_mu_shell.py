"""Shell: per-source ordinal summaries μ(s,Q) for all 18 questions.

Oracle-Structured path: reads source JSONs directly (no atom layer).
Produces μ(s,Q) ∈ Ω_Q ∪ {⊥} for each (source, question) pair.

Architecture (direct-readout atom specification):
  Phase 0: Per-day micro-fusion (implicit — source JSONs have one record/day)
  Phase 1: Aggregation plan execution per (source, question)

Source data convention:
  sources = {
      "profile_ltm": {<full profile_ltm.json>},
      "planner": [<records list from planner.json>],
      "daily_self_report": [<records list from daily_self_report.json>],
      "objective_log": [<records list from objective_log.json>],
      "device_log": [<records list from device_log.json>],
  }
"""

from __future__ import annotations

from datetime import date as date_cls
from typing import Any

from survey2agent.extraction.atoms import EXPECTED_SOURCES as SOURCE_NAMES

Mu = dict[str, str | None]  # {source_name: answer_label or None (=⊥)}


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _is_weekend(date_str: str) -> bool:
    y, m, d = map(int, date_str.split("-"))
    return date_cls(y, m, d).weekday() >= 5


def _bedtime_minutes(time_str: str) -> int:
    """HH:MM → minutes, early-morning hours shifted +24h for sort."""
    h, m = map(int, time_str.split(":"))
    mins = h * 60 + m
    if mins < 12 * 60:
        mins += 24 * 60
    return mins


def _safe(d: Any, *keys: str, default: Any = None) -> Any:
    """Safe nested dict access."""
    for k in keys:
        if isinstance(d, dict):
            d = d.get(k)
        else:
            return default
    return d if d is not None else default


def _empty_mu() -> Mu:
    return {s: None for s in SOURCE_NAMES}


def _index_by_day(records: list[dict]) -> dict[int, dict]:
    return {int(r["day_index"]): r for r in records}


# ══════════════════════════════════════════════════════════════════════════════
# Type A — Source Reliability Arbitration
# ══════════════════════════════════════════════════════════════════════════════

def _classify_good_nights(count: int) -> str:
    if count >= 20:
        return "20_or_more"
    if count >= 10:
        return "10_to_19"
    return "fewer_than_10"


def _mu_a1(sources: dict) -> Mu:
    """A1 | Sleep — 30-day ≥7h nights count per source."""
    mu = _empty_mu()

    # daily_self_report: count all days with sleep.duration_h >= 7
    sr = sources.get("daily_self_report", [])
    if sr:
        good = sum(1 for r in sr
                   if float(_safe(r, "sleep", "duration_h", default=0)) >= 7.0)
        mu["daily_self_report"] = _classify_good_nights(good)

    # device_log: count available days with sleep_tracker.duration_h >= 7,
    # extrapolate to 30 days
    dl = sources.get("device_log", [])
    avail = [r for r in dl if r.get("available")]
    sleep_avail = [r for r in avail
                   if _safe(r, "signals", "sleep_tracker", "duration_h") is not None]
    if sleep_avail:
        good = sum(1 for r in sleep_avail
                   if float(_safe(r, "signals", "sleep_tracker", "duration_h",
                                  default=0)) >= 7.0)
        estimated = round(good / len(sleep_avail) * 30)
        mu["device_log"] = _classify_good_nights(estimated)

    # planner: planner's planned sleep duration (optimistic view)
    pl = sources.get("planner", [])
    if pl:
        good = sum(1 for r in pl
                   if float(_safe(r, "sleep_target", "duration_h", default=0)) >= 7.0)
        mu["planner"] = _classify_good_nights(good)

    # profile_ltm: baseline average sleep duration
    avg = _safe(sources.get("profile_ltm", {}),
                "facts", "routine_snapshot", "sleep", "duration_h")
    if avg is not None:
        avg = float(avg)
        if avg >= 7.3:
            mu["profile_ltm"] = "20_or_more"
        elif avg >= 6.5:
            mu["profile_ltm"] = "10_to_19"
        else:
            mu["profile_ltm"] = "fewer_than_10"

    # objective_log: no sleep data
    return mu


def _classify_overtime(count: int) -> str:
    if count >= 8:
        return "8_or_more"
    if count >= 4:
        return "4_to_7"
    return "0_to_3"


def _mu_a2(sources: dict) -> Mu:
    """A2 | Work — 30-day overtime days (>9h) per source."""
    mu = _empty_mu()

    # daily_self_report: work.hours > 9 on work days
    sr = sources.get("daily_self_report", [])
    if sr:
        overtime = 0
        for r in sr:
            is_wknd = _is_weekend(r.get("date", "2026-01-01"))
            worked_wknd = r.get("work", {}).get("worked_on_weekend", False)
            if is_wknd and not worked_wknd:
                continue
            if float(_safe(r, "work", "hours", default=0)) > 9.0:
                overtime += 1
        mu["daily_self_report"] = _classify_overtime(overtime)

    # objective_log: calendar.work_block duration_min / 60 > 9
    obj = sources.get("objective_log", [])
    if obj:
        overtime = 0
        for r in obj:
            if not r.get("available"):
                continue
            calendar = _safe(r, "signals", "calendar", default=[])
            for entry in calendar:
                if entry.get("kind") == "work_block":
                    dur_min = entry.get("duration_min", 0)
                    if dur_min / 60 > 9.0:
                        overtime += 1
                    break
        mu["objective_log"] = _classify_overtime(overtime)

    # planner: work_target.hours_limit — planner plans to limit hours
    pl = sources.get("planner", [])
    if pl:
        overtime = sum(1 for r in pl
                       if float(_safe(r, "work_target", "hours_limit",
                                      default=0)) > 9.0)
        mu["planner"] = _classify_overtime(overtime)

    # profile_ltm: hours_mean
    hmean = _safe(sources.get("profile_ltm", {}),
                  "facts", "routine_snapshot", "work", "hours_mean")
    if hmean is not None:
        hmean = float(hmean)
        if hmean >= 9.5:
            mu["profile_ltm"] = "8_or_more"
        elif hmean >= 8.0:
            mu["profile_ltm"] = "4_to_7"
        else:
            mu["profile_ltm"] = "0_to_3"

    # device_log: work_session.focus_minutes (noisy proxy)
    dl = sources.get("device_log", [])
    avail = [r for r in dl if r.get("available")]
    if avail:
        overtime = sum(1 for r in avail
                       if _safe(r, "signals", "work_session", "focus_minutes",
                                default=0) > 540)
        mu["device_log"] = _classify_overtime(overtime)

    return mu


def _mu_a3(sources: dict) -> Mu:
    """A3 | Diet — 30-day home-cooked share per source."""
    mu = _empty_mu()

    def _classify_hc(ratio: float) -> str:
        if ratio >= 0.7:
            return "70_or_more"
        if ratio >= 0.4:
            return "40_to_69"
        return "less_than_40"

    # daily_self_report
    sr = sources.get("daily_self_report", [])
    if sr:
        total = sum(int(_safe(r, "diet", "meals", default=0)) for r in sr)
        home = sum(int(_safe(r, "diet", "home_cooked", default=0)) for r in sr)
        if total > 0:
            mu["daily_self_report"] = _classify_hc(home / total)

    # planner: diet_target.home_cooked_priority (boolean, not a count)
    # Planner believes its plans will be met → if priority=True, expects high ratio
    pl = sources.get("planner", [])
    if pl:
        priority_days = sum(1 for r in pl
                            if _safe(r, "diet_target", "home_cooked_priority",
                                     default=False))
        ratio = priority_days / len(pl) if pl else 0
        if ratio >= 0.7:
            mu["planner"] = "70_or_more"
        elif ratio >= 0.4:
            mu["planner"] = "40_to_69"
        else:
            mu["planner"] = "less_than_40"

    # profile_ltm: routine_snapshot.diet
    snap = _safe(sources.get("profile_ltm", {}),
                 "facts", "routine_snapshot", "diet", default={})
    hc_mean = snap.get("home_cooked_mean")
    meals_mean = snap.get("meals_per_day")
    if hc_mean is not None and meals_mean is not None:
        r = float(hc_mean) / float(meals_mean) if float(meals_mean) > 0 else 0
        mu["profile_ltm"] = _classify_hc(r)

    # objective_log: food_delivery count is counter-evidence
    # High deliveries → low home-cooked. Very approximate.
    obj = sources.get("objective_log", [])
    avail_obj = [r for r in obj if r.get("available")]
    if avail_obj:
        total_delivery = 0
        for r in avail_obj:
            payments = _safe(r, "signals", "payments", default=[])
            for p in payments:
                if p.get("category") == "food_delivery":
                    total_delivery += p.get("count", 0)
        # Rough: assume ~3 meals/day, 30 days = 90 meals total.
        # Delivery meals ≈ total_delivery (extrapolated from available days)
        if len(avail_obj) > 0:
            ext_delivery = total_delivery / len(avail_obj) * 30
            est_ratio = max(0, 1.0 - ext_delivery / 90)
            mu["objective_log"] = _classify_hc(est_ratio)

    return mu


# ══════════════════════════════════════════════════════════════════════════════
# Type B — Identity-Behavior Bridging
# ══════════════════════════════════════════════════════════════════════════════

def _mu_b2(sources: dict) -> Mu:
    """B2 | Exercise — actual frequency vs profile claim per source.

    Profile provides identity claim (shared context).
    Each behavioral source provides observed exercise frequency.
    """
    mu = _empty_mu()
    profile = sources.get("profile_ltm", {})
    profile_freq = _safe(profile, "facts", "routine_snapshot", "exercise",
                         "days_per_week")
    if profile_freq is None:
        # All sources get the edge label
        for s in SOURCE_NAMES:
            mu[s] = "no_frequency_described"
        return mu
    profile_freq = float(profile_freq)

    def _classify_b2(actual_freq: float) -> str:
        delta = actual_freq - profile_freq
        if abs(delta) <= 1.0:
            return "within_1_day"
        if delta < -1.0:
            return "more_than_1_below"
        return "more_than_1_above"

    # daily_self_report
    sr = sources.get("daily_self_report", [])
    if sr:
        days = sum(1 for r in sr if _safe(r, "exercise", "did_exercise",
                                          default=False))
        freq = days / (len(sr) / 7.0)
        mu["daily_self_report"] = _classify_b2(freq)

    # device_log: workout_detected on available days, extrapolate
    dl = sources.get("device_log", [])
    avail = [r for r in dl if r.get("available")]
    if avail:
        days = sum(1 for r in avail
                   if _safe(r, "signals", "activity_tracker", "workout_detected",
                            default=False))
        freq = days / (len(avail) / 7.0) if avail else 0
        mu["device_log"] = _classify_b2(freq)

    # objective_log: calendar.exercise_session count
    obj = sources.get("objective_log", [])
    avail_obj = [r for r in obj if r.get("available")]
    if avail_obj:
        days = 0
        for r in avail_obj:
            calendar = _safe(r, "signals", "calendar", default=[])
            if any(e.get("kind") == "exercise_session" for e in calendar):
                days += 1
        freq = days / (len(avail_obj) / 7.0) if avail_obj else 0
        mu["objective_log"] = _classify_b2(freq)

    # planner: exercise_target.intended days → planned frequency
    pl = sources.get("planner", [])
    if pl:
        days = sum(1 for r in pl
                   if _safe(r, "exercise_target", "intended", default=False))
        freq = days / (len(pl) / 7.0) if pl else 0
        mu["planner"] = _classify_b2(freq)

    # profile_ltm: trivially "within_1_day" (self-consistency)
    mu["profile_ltm"] = "within_1_day"

    return mu


def _mu_b3(sources: dict) -> Mu:
    """B3 | Work — weekend work pattern vs identity claim per source.

    Profile provides identity (shared context): afterhours_work_style.
    Behavioral sources provide weekend work observations.
    """
    mu = _empty_mu()
    profile = sources.get("profile_ltm", {})
    identity = _safe(profile, "facts", "traits", "afterhours_work_style")
    if identity is None:
        for s in SOURCE_NAMES:
            mu[s] = "no_approach_described"
        return mu

    def _classify_b3(worked: int, total_off: int) -> str:
        if total_off == 0:
            return "no_approach_described"
        ratio = worked / total_off
        if identity == "strict_boundary":
            return "matches" if ratio <= 0.15 else "does_not_match"
        if identity == "flexible":
            return ("does_not_match"
                    if worked == 0 and total_off >= 4 else "matches")
        return "matches" if ratio < 0.5 else "does_not_match"

    # daily_self_report
    sr = sources.get("daily_self_report", [])
    if sr:
        off = [r for r in sr if _is_weekend(r.get("date", "2026-01-01"))]
        worked = sum(1 for r in off if r.get("work", {}).get(
            "worked_on_weekend", False))
        mu["daily_self_report"] = _classify_b3(worked, len(off))

    # objective_log: work_block on weekends
    obj = sources.get("objective_log", [])
    if obj:
        off = [r for r in obj
               if _is_weekend(r.get("date", "2026-01-01")) and r.get("available")]
        worked = 0
        for r in off:
            calendar = _safe(r, "signals", "calendar", default=[])
            if any(e.get("kind") == "work_block" for e in calendar):
                worked += 1
        if off:
            mu["objective_log"] = _classify_b3(worked, len(off))

    # device_log: work_session.focus_minutes > 0 on weekends
    dl = sources.get("device_log", [])
    avail_wknd = [r for r in dl
                  if r.get("available")
                  and _is_weekend(r.get("date", "2026-01-01"))]
    if avail_wknd:
        worked = sum(1 for r in avail_wknd
                     if _safe(r, "signals", "work_session", "focus_minutes",
                              default=0) > 0)
        mu["device_log"] = _classify_b3(worked, len(avail_wknd))

    # profile_ltm: trivially "matches"
    mu["profile_ltm"] = "matches"

    return mu


# ══════════════════════════════════════════════════════════════════════════════
# Type C — Plan-Reality Alignment
# ══════════════════════════════════════════════════════════════════════════════

def _mu_c2(sources: dict) -> Mu:
    """C2 | Social — plan realization rate (14-day).

    Planner provides intent (context). Behavioral sources provide
    activity observations. μ(s,C2) = realization rate using planner intent
    + source s's social activity data.
    """
    mu = _empty_mu()
    pl = sources.get("planner", [])
    pl_by_day = _index_by_day(pl)

    # Find the last 14 days' day_indices.
    # Planner records cover all 30 days; pick last 14 by day_index.
    all_days = sorted(pl_by_day.keys())
    window_days = set(all_days[-14:]) if len(all_days) >= 14 else set(all_days)

    plan_day_indices = [
        di for di in window_days
        if _safe(pl_by_day.get(di, {}), "social_target", "intent", default=False)
    ]

    if not plan_day_indices:
        # No social plans → all sources agree on edge label
        for s in SOURCE_NAMES:
            mu[s] = "no_plans"
        return mu

    def _classify_c2(ratio: float) -> str:
        if ratio > 0.50:
            return "above_50_pct"
        if ratio >= 0.25:
            return "25_to_50_pct"
        return "below_25_pct"

    # daily_self_report: social.activities on plan days
    sr = sources.get("daily_self_report", [])
    sr_by_day = _index_by_day(sr)
    if sr:
        realized = sum(1 for di in plan_day_indices
                       if len(_safe(sr_by_day.get(di, {}), "social",
                                    "activities", default=[])) > 0)
        mu["daily_self_report"] = _classify_c2(realized / len(plan_day_indices))

    # planner: planner believes all plans realized (b=+1 optimism)
    mu["planner"] = "above_50_pct"

    # objective_log: social-related payments on plan days (partial)
    obj = sources.get("objective_log", [])
    obj_by_day = _index_by_day(obj)
    avail_plan = [di for di in plan_day_indices
                  if obj_by_day.get(di, {}).get("available")]
    if avail_plan:
        realized = 0
        for di in avail_plan:
            payments = _safe(obj_by_day[di], "signals", "payments", default=[])
            calendar = _safe(obj_by_day[di], "signals", "calendar", default=[])
            has_social = any(p.get("category") in ("restaurant", "entertainment",
                                                    "social_venue")
                            for p in payments)
            has_social = has_social or any(
                e.get("kind") in ("social_event", "social_venue")
                for e in calendar)
            if has_social:
                realized += 1
        mu["objective_log"] = _classify_c2(realized / len(plan_day_indices))

    return mu


def _mu_c3(sources: dict) -> Mu:
    """C3 | Sleep — bedtime target compliance (14-day).

    Planner provides target bedtime (context). Sleep sources provide
    actual bedtime observations.
    """
    mu = _empty_mu()
    pl = sources.get("planner", [])
    pl_by_day = _index_by_day(pl)
    all_days = sorted(pl_by_day.keys())
    window_days = set(all_days[-14:]) if len(all_days) >= 14 else set(all_days)

    def _classify_c3(late: int, early: int, paired: int) -> str:
        if paired == 0:
            return "no_targets"
        if late / paired > 0.5:
            return "later_more_than_50pct"
        if early / paired > 0.5:
            return "earlier_more_than_50pct"
        return "within_20min_more_than_50pct"

    def _compute_compliance(day_records: dict[int, dict],
                            time_key_path: list[str]) -> str | None:
        late = early = paired = 0
        for di in window_days:
            pr = pl_by_day.get(di)
            if pr is None:
                continue
            target_str = _safe(pr, "sleep_target", "bedtime")
            if target_str is None:
                continue
            rec = day_records.get(di)
            if rec is None:
                continue
            actual_str = _safe(rec, *time_key_path)
            if actual_str is None:
                continue
            paired += 1
            target = _bedtime_minutes(target_str)
            actual = _bedtime_minutes(actual_str)
            delta = actual - target
            if delta > 20:
                late += 1
            elif delta < -20:
                early += 1
        if paired == 0:
            return None
        return _classify_c3(late, early, paired)

    # daily_self_report: sleep.bedtime
    sr = sources.get("daily_self_report", [])
    sr_by_day = _index_by_day(sr)
    if sr:
        result = _compute_compliance(sr_by_day, ["sleep", "bedtime"])
        if result:
            mu["daily_self_report"] = result

    # device_log: signals.sleep_tracker.bedtime
    dl = sources.get("device_log", [])
    dl_avail = {int(r["day_index"]): r for r in dl if r.get("available")}
    if dl_avail:
        result = _compute_compliance(
            dl_avail, ["signals", "sleep_tracker", "bedtime"])
        if result:
            mu["device_log"] = result

    # If no paired data from any source, check if planner even had targets
    has_targets = any(
        _safe(pl_by_day.get(di, {}), "sleep_target", "bedtime") is not None
        for di in window_days
    )
    if not has_targets:
        for s in SOURCE_NAMES:
            mu[s] = "no_targets"

    return mu


# ══════════════════════════════════════════════════════════════════════════════
# Type D — Temporal Trend Detection
# ══════════════════════════════════════════════════════════════════════════════

def _mu_d1(sources: dict) -> Mu:
    """D1 | Social — activity trend (first 14 vs last 16 days)."""
    mu = _empty_mu()

    def _classify_d1(delta: float) -> str:
        if delta > 0.15:
            return "increased"
        if delta < -0.15:
            return "decreased"
        return "stayed_same"

    # daily_self_report
    sr = sources.get("daily_self_report", [])
    if sr:
        sr_sorted = sorted(sr, key=lambda r: int(r["day_index"]))
        early = sr_sorted[:14]
        late = sr_sorted[14:]
        if early and late:
            e_rate = sum(len(_safe(r, "social", "activities", default=[]))
                         for r in early) / len(early)
            l_rate = sum(len(_safe(r, "social", "activities", default=[]))
                         for r in late) / len(late)
            mu["daily_self_report"] = _classify_d1(l_rate - e_rate)

    # planner: social_target.intent rate (early vs late)
    pl = sources.get("planner", [])
    if pl:
        pl_sorted = sorted(pl, key=lambda r: int(r["day_index"]))
        early = pl_sorted[:14]
        late = pl_sorted[14:]
        if early and late:
            e_rate = sum(1 for r in early
                         if _safe(r, "social_target", "intent",
                                  default=False)) / len(early)
            l_rate = sum(1 for r in late
                         if _safe(r, "social_target", "intent",
                                  default=False)) / len(late)
            mu["planner"] = _classify_d1(l_rate - e_rate)

    # objective_log: social calendar events trend (noisy)
    obj = sources.get("objective_log", [])
    avail = [r for r in obj if r.get("available")]
    if avail:
        avail_sorted = sorted(avail, key=lambda r: int(r["day_index"]))
        mid = len(avail_sorted) // 2
        early = avail_sorted[:mid]
        late = avail_sorted[mid:]
        if early and late:
            def _social_events(rec):
                calendar = _safe(rec, "signals", "calendar", default=[])
                return sum(1 for e in calendar
                           if e.get("kind") in ("social_event",
                                                 "social_venue"))
            e_rate = sum(_social_events(r) for r in early) / len(early)
            l_rate = sum(_social_events(r) for r in late) / len(late)
            mu["objective_log"] = _classify_d1(l_rate - e_rate)

    return mu


def _mu_d2(sources: dict) -> Mu:
    """D2 | Diet — baseline comparison (30-day actual vs profile)."""
    mu = _empty_mu()
    profile = sources.get("profile_ltm", {})
    snap = _safe(profile, "facts", "routine_snapshot", "diet", default={})
    p_meals = snap.get("meals_per_day")
    p_hc = snap.get("home_cooked_mean")

    if p_meals is None or p_hc is None:
        for s in SOURCE_NAMES:
            mu[s] = "no_baseline"
        return mu

    p_meals, p_hc = float(p_meals), float(p_hc)

    def _classify_d2(actual_meals: float, actual_hc: float) -> str:
        delta = abs(actual_hc - p_hc) + abs(actual_meals - p_meals)
        return "differs_more_than_1" if delta > 1.0 else "within_1"

    # daily_self_report
    sr = sources.get("daily_self_report", [])
    if sr:
        a_meals = sum(int(_safe(r, "diet", "meals", default=0))
                      for r in sr) / len(sr)
        a_hc = sum(int(_safe(r, "diet", "home_cooked", default=0))
                   for r in sr) / len(sr)
        mu["daily_self_report"] = _classify_d2(a_meals, a_hc)

    # profile_ltm: trivially "within_1" (self-consistency)
    mu["profile_ltm"] = "within_1"

    return mu


# ══════════════════════════════════════════════════════════════════════════════
# Type E — Factor Attribution by Elimination
# ══════════════════════════════════════════════════════════════════════════════

def _mu_e1(sources: dict) -> Mu:
    """E1 | Sleep — late bedtime factors.

    Each source gives its own view of what causes late bedtimes.
    """
    mu = _empty_mu()

    # daily_self_report: identify late nights (bedtime after midnight)
    # then check work overtime / social activity as proxy for context_tags
    sr = sources.get("daily_self_report", [])
    if sr:
        late_nights = [r for r in sr
                       if _bedtime_minutes(
                           _safe(r, "sleep", "bedtime", default="23:00")
                       ) > 24 * 60]
        if not late_nights:
            mu["daily_self_report"] = "no_late_nights"
        else:
            work_caused = 0
            social_caused = 0
            for r in late_nights:
                has_overtime = r.get("work", {}).get("overtime", False)
                has_social = len(_safe(r, "social", "activities",
                                       default=[])) > 0
                if has_overtime:
                    work_caused += 1
                elif has_social:
                    social_caused += 1
            total = len(late_nights)
            if work_caused / total > 0.5:
                mu["daily_self_report"] = "work_activity"
            elif social_caused / total > 0.5:
                mu["daily_self_report"] = "social_activity"
            else:
                mu["daily_self_report"] = "no_single_factor"

    # device_log: late bedtimes + work_session.late_finish as cause proxy
    dl = sources.get("device_log", [])
    avail = [r for r in dl if r.get("available")]
    if avail:
        late_recs = [r for r in avail
                     if _safe(r, "signals", "sleep_tracker", "bedtime")
                     is not None
                     and _bedtime_minutes(
                         _safe(r, "signals", "sleep_tracker", "bedtime",
                               default="23:00")
                     ) > 24 * 60]
        if not late_recs:
            mu["device_log"] = "no_late_nights"
        else:
            work_late = sum(
                1 for r in late_recs
                if _safe(r, "signals", "work_session", "late_finish",
                         default=False)
            )
            total = len(late_recs)
            if work_late / total > 0.5:
                mu["device_log"] = "work_activity"
            else:
                # Device can detect late bedtime but can't identify social
                mu["device_log"] = "no_single_factor"

    # planner: sleep_target bedtime vs work_target hours_limit
    pl = sources.get("planner", [])
    if pl:
        late_plan = [r for r in pl
                     if _bedtime_minutes(
                         _safe(r, "sleep_target", "bedtime", default="22:00")
                     ) > 24 * 60]
        if not late_plan:
            mu["planner"] = "no_late_nights"
        else:
            work_conflict = sum(
                1 for r in late_plan
                if float(_safe(r, "work_target", "hours_limit", default=0)) > 8
                or not _safe(r, "work_target", "avoid_overtime", default=True)
            )
            if work_conflict / len(late_plan) > 0.5:
                mu["planner"] = "work_activity"
            else:
                mu["planner"] = "no_single_factor"

    # objective_log: late work_block or social_event in calendar
    obj = sources.get("objective_log", [])
    if obj:
        late_work = 0
        late_social = 0
        total_days = 0
        for r in obj:
            if not r.get("available"):
                continue
            total_days += 1
            calendar = _safe(r, "signals", "calendar", default=[])
            has_long_work = any(
                e.get("kind") == "work_block"
                and e.get("duration_min", 0) > 540
                for e in calendar
            )
            has_social = any(
                e.get("kind") == "social_event" for e in calendar
            )
            if has_long_work:
                late_work += 1
            elif has_social:
                late_social += 1
        if total_days > 0:
            if late_work / total_days > 0.3:
                mu["objective_log"] = "work_activity"
            elif late_social / total_days > 0.3:
                mu["objective_log"] = "social_activity"
            else:
                mu["objective_log"] = "no_single_factor"

    return mu


def _mu_e2(sources: dict) -> Mu:
    """E2 | Exercise — skipped-exercise causation (30-day).

    Question: what fraction of skipped exercises were caused by work?
    Each source gives its own view of work-vs-exercise conflict.
    """
    mu = _empty_mu()
    pl = sources.get("planner", [])
    pl_by_day = _index_by_day(pl)

    def _classify_e2(ratio: float) -> str:
        if ratio > 0.6:
            return "yes_more_than_60"
        if ratio < 0.3:
            return "no_fewer_than_30"
        return "between_30_60"

    # daily_self_report: find days where planner intended but didn't exercise
    sr = sources.get("daily_self_report", [])
    sr_by_day = _index_by_day(sr)

    if sr and pl:
        skip_days = []
        for di, pr in pl_by_day.items():
            if not _safe(pr, "exercise_target", "intended", default=False):
                continue
            sr_rec = sr_by_day.get(di)
            if sr_rec is None:
                continue
            if not _safe(sr_rec, "exercise", "did_exercise", default=False):
                skip_days.append(sr_rec)

        if len(skip_days) <= 2:
            mu["daily_self_report"] = "between_30_60"
        else:
            work_caused = sum(
                1 for d in skip_days
                if float(_safe(d, "work", "hours", default=0)) > 8.5
                or d.get("work", {}).get("overtime", False)
            )
            mu["daily_self_report"] = _classify_e2(work_caused / len(skip_days))

    # planner: work-exercise schedule conflict (optimistic, bias +1)
    # Days where exercise intended AND work_target.hours_limit > 8 → conflict
    if pl:
        intended = [r for r in pl
                    if _safe(r, "exercise_target", "intended", default=False)]
        if intended:
            conflict = sum(
                1 for r in intended
                if float(_safe(r, "work_target", "hours_limit", default=0)) > 8
            )
            # Planner underestimates conflict (optimistic bias +1)
            mu["planner"] = _classify_e2(conflict / len(intended))

    # device_log: days without workout where work_session was heavy
    dl = sources.get("device_log", [])
    avail = [r for r in dl if r.get("available")]
    if avail:
        no_workout = [r for r in avail
                      if not _safe(r, "signals", "activity_tracker",
                                   "workout_detected", default=False)]
        if no_workout:
            work_heavy = sum(
                1 for r in no_workout
                if _safe(r, "signals", "work_session", "late_finish",
                         default=False)
                or int(_safe(r, "signals", "work_session", "focus_minutes",
                             default=0)) > 480
            )
            mu["device_log"] = _classify_e2(
                work_heavy / len(no_workout) if no_workout else 0
            )

    # objective_log: work_block vs exercise_session in calendar
    obj = sources.get("objective_log", [])
    if obj:
        no_exercise_days = 0
        work_heavy_days = 0
        for r in obj:
            if not r.get("available"):
                continue
            calendar = _safe(r, "signals", "calendar", default=[])
            has_exercise = any(e.get("kind") == "exercise_session"
                               for e in calendar)
            if has_exercise:
                continue
            no_exercise_days += 1
            has_heavy_work = any(
                e.get("kind") == "work_block"
                and e.get("duration_min", 0) > 540
                for e in calendar
            )
            if has_heavy_work:
                work_heavy_days += 1
        if no_exercise_days > 0:
            mu["objective_log"] = _classify_e2(
                work_heavy_days / no_exercise_days
            )

    # profile_ltm: overtime_share as baseline conflict estimate
    snap = _safe(sources.get("profile_ltm", {}),
                 "facts", "routine_snapshot", "work", default={})
    ot_share = snap.get("overtime_share")
    if ot_share is not None:
        mu["profile_ltm"] = _classify_e2(float(ot_share))

    return mu


# ══════════════════════════════════════════════════════════════════════════════
# Type F — Missing Data Reasoning
# ══════════════════════════════════════════════════════════════════════════════

def _mu_f1(sources: dict) -> Mu:
    """F1 | Social — unplanned social activity frequency (30-day).

    Each source gives its own view of social activity patterns;
    the "unplanned" aspect emerges from cross-source disagreement.
    """
    mu = _empty_mu()
    pl = sources.get("planner", [])
    pl_by_day = _index_by_day(pl)

    def _classify_f1(unplanned: int) -> str:
        if unplanned <= 3:
            return "0_to_3"
        if unplanned <= 6:
            return "4_to_6"
        return "7_or_more"

    # daily_self_report: social days with no planner intent
    sr = sources.get("daily_self_report", [])
    if sr:
        social_days = [r for r in sr
                       if len(_safe(r, "social", "activities", default=[])) > 0]
        if not social_days:
            mu["daily_self_report"] = "no_social_activities"
        else:
            unplanned = 0
            for r in social_days:
                di = int(r["day_index"])
                pr = pl_by_day.get(di)
                if pr is None or not _safe(pr, "social_target", "intent",
                                           default=False):
                    unplanned += 1
            mu["daily_self_report"] = _classify_f1(unplanned)

    # objective_log: calendar social_events cross-ref with planner
    obj = sources.get("objective_log", [])
    if obj:
        obj_by_day = _index_by_day(obj)
        social_days_obj = 0
        unplanned_obj = 0
        for di, orec in obj_by_day.items():
            if not orec.get("available"):
                continue
            calendar = _safe(orec, "signals", "calendar", default=[])
            has_social = any(e.get("kind") == "social_event" for e in calendar)
            if has_social:
                social_days_obj += 1
                pr = pl_by_day.get(di)
                if pr is None or not _safe(pr, "social_target", "intent",
                                           default=False):
                    unplanned_obj += 1
        if social_days_obj == 0:
            mu["objective_log"] = "no_social_activities"
        else:
            mu["objective_log"] = _classify_f1(unplanned_obj)

    # planner: planned social count → inverse signal (bias +1, optimistic)
    # Planner knows what was planned; everything outside plan is invisible
    if pl:
        planned = sum(1 for r in pl
                      if _safe(r, "social_target", "intent", default=False))
        # More planned social → planner thinks less is unplanned
        # Optimistic view: if planning 10+ social days, assumes few spontaneous
        if planned >= 10:
            mu["planner"] = "0_to_3"
        elif planned >= 5:
            mu["planner"] = "4_to_6"
        else:
            # Low planning → planner has no view → default to low unplanned
            mu["planner"] = "0_to_3"

    # profile_ltm: social frequency baseline (staleness bias)
    snap = _safe(sources.get("profile_ltm", {}),
                 "facts", "routine_snapshot", "social", default={})
    dpw = snap.get("active_days_per_week")
    if dpw is not None:
        dpw = float(dpw)
        if dpw == 0:
            mu["profile_ltm"] = "no_social_activities"
        else:
            # Estimate monthly social days, assume ~half are unplanned
            monthly = dpw * 4.3
            est_unplanned = int(monthly * 0.5)
            mu["profile_ltm"] = _classify_f1(est_unplanned)

    return mu


def _mu_f2(sources: dict) -> Mu:
    """F2 | Exercise — tracker dropout vs actual inactivity (30-day).

    Each source gives its own view of exercise patterns;
    discrepancies reveal tracker dropout vs true inactivity.
    """
    mu = _empty_mu()
    dl = sources.get("device_log", [])
    dl_by_day = _index_by_day(dl)

    sr = sources.get("daily_self_report", [])
    sr_by_day = _index_by_day(sr)

    if dl and sr:
        truly_inactive = 0
        data_missing = 0

        for di, dr in dl_by_day.items():
            avail = dr.get("available", False)
            workout = (avail and _safe(dr, "signals", "activity_tracker",
                                       "workout_detected", default=False))
            if workout:
                continue  # device saw workout, skip

            sr_rec = sr_by_day.get(di)
            actually_ex = (_safe(sr_rec, "exercise", "did_exercise",
                                 default=False) if sr_rec else False)

            if not avail and actually_ex:
                data_missing += 1
            elif not actually_ex:
                truly_inactive += 1
            else:
                data_missing += 1

        if truly_inactive == 0 and data_missing == 0:
            mu["daily_self_report"] = "inactive_confirmed"
        elif truly_inactive > 0 and data_missing > 0:
            mu["daily_self_report"] = "both_occurred"
        elif data_missing > truly_inactive:
            mu["daily_self_report"] = "yes_tracker_missing"
        else:
            mu["daily_self_report"] = "inactive_confirmed"

    # device_log: own availability vs workout rate (knows its own dropout)
    if dl:
        total = len(dl)
        unavail = sum(1 for r in dl if not r.get("available"))
        workouts = sum(1 for r in dl
                       if r.get("available")
                       and _safe(r, "signals", "activity_tracker",
                                 "workout_detected", default=False))
        if unavail == 0:
            mu["device_log"] = "inactive_confirmed"  # no dropout
        elif workouts > 0 and unavail > 0:
            mu["device_log"] = "both_occurred"
        elif unavail > total * 0.3:
            mu["device_log"] = "yes_tracker_missing"
        else:
            mu["device_log"] = "inactive_confirmed"

    # planner: exercise intent frequency (optimistic bias +1)
    pl = sources.get("planner", [])
    if pl:
        intended = sum(1 for r in pl
                       if _safe(r, "exercise_target", "intended", default=False))
        # High intent → planner expects exercise → dropout more likely if
        # device doesn't see it → shifts toward "yes_tracker_missing"
        if intended >= 15:
            mu["planner"] = "yes_tracker_missing"
        elif intended >= 5:
            mu["planner"] = "both_occurred"
        else:
            mu["planner"] = "inactive_confirmed"

    # objective_log: calendar exercise_session count vs device dropout
    obj = sources.get("objective_log", [])
    if obj:
        exercise_days = 0
        avail_days = 0
        for r in obj:
            if not r.get("available"):
                continue
            avail_days += 1
            calendar = _safe(r, "signals", "calendar", default=[])
            if any(e.get("kind") == "exercise_session" for e in calendar):
                exercise_days += 1
        if avail_days > 0:
            rate = exercise_days / avail_days
            if rate > 0.3:
                # Calendar shows exercise → if tracker misses it, that's dropout
                mu["objective_log"] = "yes_tracker_missing"
            elif rate > 0.1:
                mu["objective_log"] = "both_occurred"
            else:
                mu["objective_log"] = "inactive_confirmed"

    # profile_ltm: exercise frequency baseline (staleness bias)
    snap = _safe(sources.get("profile_ltm", {}),
                 "facts", "routine_snapshot", "exercise", default={})
    dpw = snap.get("days_per_week")
    if dpw is not None:
        dpw = float(dpw)
        if dpw >= 3:
            mu["profile_ltm"] = "yes_tracker_missing"
        elif dpw >= 1:
            mu["profile_ltm"] = "both_occurred"
        else:
            mu["profile_ltm"] = "inactive_confirmed"

    return mu


def _mu_f3(sources: dict) -> Mu:
    """F3 | Work — timesheet dropout vs actual work (30-day).

    Each source gives its view of work patterns;
    discrepancies reveal objective_log dropout vs actual off days.
    """
    mu = _empty_mu()
    obj = sources.get("objective_log", [])
    obj_by_day = _index_by_day(obj)

    sr = sources.get("daily_self_report", [])
    sr_by_day = _index_by_day(sr)

    if obj and sr:
        truly_off = 0
        unclear = 0
        no_record_count = 0

        for di, orec in obj_by_day.items():
            if orec.get("available", False):
                continue
            no_record_count += 1
            sr_rec = sr_by_day.get(di)
            actually_worked = (float(_safe(sr_rec, "work", "hours", default=0)) > 0
                               if sr_rec else False)
            if actually_worked:
                unclear += 1
            else:
                truly_off += 1

        if no_record_count == 0:
            mu["daily_self_report"] = "truly_off"
        elif truly_off > 0 and unclear > 0:
            mu["daily_self_report"] = "both_occurred"
        elif unclear > truly_off:
            mu["daily_self_report"] = "yes_worked_despite_no_entry"
        else:
            mu["daily_self_report"] = "truly_off"

    # device_log: work_session on days obj_log is unavailable
    dl = sources.get("device_log", [])
    if dl and obj:
        dl_by_day = _index_by_day(dl)
        no_obj_days = 0
        worked_per_device = 0
        for di, orec in obj_by_day.items():
            if orec.get("available", False):
                continue
            no_obj_days += 1
            dr = dl_by_day.get(di)
            if dr and dr.get("available"):
                focus = int(_safe(dr, "signals", "work_session",
                                  "focus_minutes", default=0))
                if focus > 30:
                    worked_per_device += 1
        if no_obj_days == 0:
            mu["device_log"] = "truly_off"
        elif worked_per_device > 0 and (no_obj_days - worked_per_device) > 0:
            mu["device_log"] = "both_occurred"
        elif worked_per_device > no_obj_days * 0.5:
            mu["device_log"] = "yes_worked_despite_no_entry"
        else:
            mu["device_log"] = "truly_off"

    # planner: work_target on days obj_log is unavailable (optimistic bias +1)
    pl = sources.get("planner", [])
    if pl and obj:
        pl_by_day = _index_by_day(pl)
        no_obj_days = 0
        planned_work = 0
        for di, orec in obj_by_day.items():
            if orec.get("available", False):
                continue
            no_obj_days += 1
            pr = pl_by_day.get(di)
            if pr:
                limit = float(_safe(pr, "work_target", "hours_limit", default=0))
                if limit > 0:
                    planned_work += 1
        if no_obj_days == 0:
            mu["planner"] = "truly_off"
        elif planned_work > 0 and (no_obj_days - planned_work) > 0:
            mu["planner"] = "both_occurred"
        elif planned_work > no_obj_days * 0.5:
            mu["planner"] = "yes_worked_despite_no_entry"
        else:
            mu["planner"] = "truly_off"

    # profile_ltm: work baseline (staleness bias)
    snap = _safe(sources.get("profile_ltm", {}),
                 "facts", "routine_snapshot", "work", default={})
    hmean = snap.get("hours_mean")
    ot = snap.get("overtime_share")
    if hmean is not None:
        hmean = float(hmean)
        if hmean > 6:
            # Active worker → missing entries are likely dropout
            mu["profile_ltm"] = "yes_worked_despite_no_entry"
        elif hmean > 3:
            mu["profile_ltm"] = "both_occurred"
        else:
            mu["profile_ltm"] = "truly_off"

    return mu


# ══════════════════════════════════════════════════════════════════════════════
# Type G — Annotation Disambiguation
# ══════════════════════════════════════════════════════════════════════════════

def _mu_g1(sources: dict) -> Mu:
    """G1 | Exercise — deliberate vs incidental activity (30-day).

    Source JSONs lack event_table's exercise.intentional flag.
    Proxy: duration_min >= 30 → intentional (matches event generator logic).
    Planner's exercise_target.intended provides plan-based view.
    """
    mu = _empty_mu()

    def _classify_g1(intentional: int, total: int) -> str:
        if total == 0:
            return "no_activity"
        ratio = intentional / total
        if ratio > 0.7:
            return "deliberate_exercise_70plus"
        if ratio < 0.3:
            return "incidental_movement_70plus"
        return "mix"

    # daily_self_report: duration proxy
    sr = sources.get("daily_self_report", [])
    if sr:
        active = [r for r in sr
                  if _safe(r, "exercise", "did_exercise", default=False)]
        if not active:
            mu["daily_self_report"] = "no_activity"
        else:
            intentional = sum(
                1 for r in active
                if int(_safe(r, "exercise", "duration_min", default=0)) >= 30
            )
            mu["daily_self_report"] = _classify_g1(intentional, len(active))

    # planner: exercise_target.intended (planner-only data)
    # Planner only knows about planned exercise days; all are "intentional"
    # by definition. Denominator = total planned days (planner has no
    # concept of unplanned exercise).
    pl = sources.get("planner", [])
    if pl:
        planned = sum(1 for r in pl
                      if _safe(r, "exercise_target", "intended", default=False))
        if planned > 0:
            # All planned exercise is intentional from planner's view
            mu["planner"] = _classify_g1(planned, planned)

    # device_log: active_minutes ≥ 30 → intentional proxy
    dl = sources.get("device_log", [])
    avail = [r for r in dl if r.get("available")]
    if avail:
        active = [r for r in avail
                  if _safe(r, "signals", "activity_tracker", "workout_detected",
                           default=False)]
        if not active:
            # Check for any activity at all
            any_active = [r for r in avail
                          if _safe(r, "signals", "activity_tracker",
                                   "active_minutes", default=0) > 10]
            if not any_active:
                mu["device_log"] = "no_activity"
        else:
            intentional = sum(
                1 for r in active
                if int(_safe(r, "signals", "activity_tracker",
                             "active_minutes", default=0)) >= 30
            )
            mu["device_log"] = _classify_g1(intentional, len(active))

    return mu


def _mu_g2(sources: dict) -> Mu:
    """G2 | Social — voluntary vs obligatory activity (30-day)."""
    mu = _empty_mu()

    def _classify_g2(voluntary: int, total: int) -> str:
        if total == 0:
            return "no_meetings"
        ratio = voluntary / total
        if ratio > 0.7:
            return "voluntary_70plus"
        if ratio < 0.3:
            return "obligatory_70plus"
        return "mix"

    # daily_self_report: supporting_other flag distinguishes vol/oblig
    sr = sources.get("daily_self_report", [])
    if sr:
        social_days = [r for r in sr
                       if len(_safe(r, "social", "activities", default=[])) > 0]
        if not social_days:
            mu["daily_self_report"] = "no_meetings"
        else:
            voluntary = sum(
                1 for r in social_days
                if not _safe(r, "social", "supporting_other", default=False)
            )
            mu["daily_self_report"] = _classify_g2(voluntary, len(social_days))

    # objective_log: calendar social_event count (no vol/oblig distinction)
    obj = sources.get("objective_log", [])
    if obj:
        social_count = 0
        for r in obj:
            if not r.get("available"):
                continue
            for ev in _safe(r, "signals", "calendar", default=[]):
                if ev.get("kind") == "social_event":
                    social_count += 1
        if social_count == 0:
            mu["objective_log"] = "no_meetings"
        else:
            # Calendar records all social events neutrally; defaults to mix
            mu["objective_log"] = "mix"

    # planner: social_target.intent → all planned social is voluntary (bias +1)
    pl = sources.get("planner", [])
    if pl:
        planned = sum(1 for r in pl
                      if _safe(r, "social_target", "intent", default=False))
        if planned == 0:
            mu["planner"] = "no_meetings"
        else:
            # Planner only sees self-initiated social → optimistic voluntary view
            mu["planner"] = "voluntary_70plus"

    # profile_ltm: social baseline (staleness, no vol/oblig info)
    snap = _safe(sources.get("profile_ltm", {}),
                 "facts", "routine_snapshot", "social", default={})
    dpw = snap.get("active_days_per_week")
    if dpw is not None:
        if float(dpw) == 0:
            mu["profile_ltm"] = "no_meetings"
        else:
            mu["profile_ltm"] = "mix"

    return mu


# ══════════════════════════════════════════════════════════════════════════════
# Control — Calibration Baseline
# ══════════════════════════════════════════════════════════════════════════════

def _mu_ctrl1(sources: dict) -> Mu:
    """Ctrl1 | Diet — outside meals in past week (7-day)."""
    mu = _empty_mu()

    def _classify_ctrl1(outside: int) -> str:
        if outside <= 1:
            return "0_to_1_days"
        if outside <= 3:
            return "2_to_3_days"
        return "4_or_more"

    # daily_self_report: last 7 records
    sr = sources.get("daily_self_report", [])
    if sr:
        recent = sorted(sr, key=lambda r: int(r["day_index"]))[-7:]
        outside = sum(
            1 for r in recent
            if int(_safe(r, "diet", "meals", default=0))
            > int(_safe(r, "diet", "home_cooked", default=0))
        )
        mu["daily_self_report"] = _classify_ctrl1(outside)

    # planner: diet_target — if home_cooked_priority=True, plans 0 outside
    pl = sources.get("planner", [])
    if pl:
        recent = sorted(pl, key=lambda r: int(r["day_index"]))[-7:]
        outside = sum(
            1 for r in recent
            if not _safe(r, "diet_target", "home_cooked_priority", default=True)
        )
        mu["planner"] = _classify_ctrl1(outside)

    # profile_ltm: baseline estimate
    snap = _safe(sources.get("profile_ltm", {}),
                 "facts", "routine_snapshot", "diet", default={})
    meals = snap.get("meals_per_day")
    hc = snap.get("home_cooked_mean")
    if meals is not None and hc is not None:
        outside_rate = max(0, float(meals) - float(hc))
        est_outside = round(outside_rate * 7)
        mu["profile_ltm"] = _classify_ctrl1(est_outside)

    return mu


def _mu_ctrl2(sources: dict) -> Mu:
    """Ctrl2 | Sleep — short sleep nights in past week (7-day)."""
    mu = _empty_mu()

    def _classify_ctrl2(short: int) -> str:
        if short == 0:
            return "0_nights"
        if short <= 2:
            return "1_to_2"
        return "3_or_more"

    # daily_self_report: last 7 records
    sr = sources.get("daily_self_report", [])
    if sr:
        recent = sorted(sr, key=lambda r: int(r["day_index"]))[-7:]
        short = sum(
            1 for r in recent
            if float(_safe(r, "sleep", "duration_h", default=8)) < 6.0
        )
        mu["daily_self_report"] = _classify_ctrl2(short)

    # device_log: sleep_tracker on last 7 available
    dl = sources.get("device_log", [])
    avail = sorted([r for r in dl if r.get("available")],
                   key=lambda r: int(r["day_index"]))
    recent_avail = avail[-7:] if avail else []
    if recent_avail:
        short = sum(
            1 for r in recent_avail
            if float(_safe(r, "signals", "sleep_tracker", "duration_h",
                           default=8)) < 6.0
        )
        mu["device_log"] = _classify_ctrl2(short)

    # planner: planned sleep duration < 6 (unlikely but possible)
    pl = sources.get("planner", [])
    if pl:
        recent = sorted(pl, key=lambda r: int(r["day_index"]))[-7:]
        short = sum(
            1 for r in recent
            if float(_safe(r, "sleep_target", "duration_h", default=8)) < 6.0
        )
        mu["planner"] = _classify_ctrl2(short)

    return mu


# ══════════════════════════════════════════════════════════════════════════════
# Registry & Orchestrator
# ══════════════════════════════════════════════════════════════════════════════

MU_REGISTRY: dict[str, Any] = {
    "A1": _mu_a1, "A2": _mu_a2, "A3": _mu_a3,
    "B2": _mu_b2, "B3": _mu_b3,
    "C2": _mu_c2, "C3": _mu_c3,
    "D1": _mu_d1, "D2": _mu_d2,
    "E1": _mu_e1, "E2": _mu_e2,
    "F1": _mu_f1, "F2": _mu_f2, "F3": _mu_f3,
    "G1": _mu_g1, "G2": _mu_g2,
    "Ctrl1": _mu_ctrl1, "Ctrl2": _mu_ctrl2,
}


def compute_all_mu(sources: dict) -> dict[str, Mu]:
    """Compute μ(s,Q) for all 18 questions × 5 sources.

    Parameters
    ----------
    sources : dict
        Keys: "profile_ltm" (dict), "planner" (list), "daily_self_report" (list),
        "objective_log" (list), "device_log" (list).

    Returns
    -------
    dict mapping question_id → {source_name: answer_label or None (=⊥)}
    """
    return {qid: fn(sources) for qid, fn in MU_REGISTRY.items()}
