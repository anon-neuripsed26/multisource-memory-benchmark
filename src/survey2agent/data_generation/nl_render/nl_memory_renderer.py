"""Deterministic NL memory renderer for the released benchmark.

Converts structured source JSONs to natural language memory documents.
No schema tokens leak — all field names are translated to natural descriptions.

Output style follows the same section structure as the original batch NL renders:
  1. Long-Term Background and Habits  (profile_ltm)
  2. Plans and Intentions             (planner)
  3. Daily Self-Reports               (daily_self_report)
  4. Objective Records                (objective_log)
  5. Device and Activity Records      (device_log)
"""

from __future__ import annotations

import json
from datetime import datetime as _dt
from pathlib import Path
from typing import Any


def _safe(obj: Any, *keys: str, default: Any = None) -> Any:
    for k in keys:
        if isinstance(obj, dict):
            obj = obj.get(k, default)
        else:
            return default
    return obj


def _fmt_time(t: str | None) -> str:
    """Format time string, handling None."""
    return t if t else "unknown"


def _quality_word(q: int | float | None) -> str:
    """Convert numeric quality to natural description."""
    if q is None:
        return "unknown"
    q = float(q)
    if q >= 4.0:
        return "good"
    if q >= 3.0:
        return "fair"
    return "poor"


def _bool_phrase(val: bool | None, true_s: str, false_s: str) -> str:
    if val is True:
        return true_s
    if val is False:
        return false_s
    return ""


# ── Profile Renderer ─────────────────────────────────────────────────────────

def _date_heading(date_str: str) -> str:
    """Format date string as '#### 2026-01-03 (Saturday)'."""
    try:
        d = _dt.strptime(date_str, "%Y-%m-%d")
        return f"#### {date_str} ({d.strftime('%A')})"
    except (ValueError, TypeError):
        return f"#### {date_str}"


def render_profile(data: dict) -> str:
    """Render profile_ltm.json to NL paragraph."""
    facts = data.get("facts", {})
    identity = facts.get("identity", {})
    traits = facts.get("traits", {})
    routine = facts.get("routine_snapshot", {})

    lines = ["## Long-Term Background and Habits", ""]

    # Identity
    name = identity.get("name", "This person")
    age = identity.get("age", "")
    occupation = identity.get("occupation", "")
    city = identity.get("city", "")
    relationship = identity.get("relationship", "")
    intro = f"{name} is"
    if age:
        intro += f" a {age}-year-old"
    if occupation:
        intro += f" {occupation}"
    if city:
        intro += f" in {city}"
    intro += "."
    if relationship:
        intro += f" Relationship status: {relationship}."
    lines.append(intro)

    # Traits
    trait_parts = []
    dr = traits.get("dietary_restriction")
    if dr:
        trait_parts.append(f"dietary restriction: {dr}")
    else:
        trait_parts.append("no dietary restrictions")
    sp = traits.get("social_preference")
    if sp:
        trait_parts.append(f"social preference toward {sp}")
    cf = traits.get("close_friends")
    if cf is not None:
        trait_parts.append(f"{cf} close friends")
    pe = traits.get("primary_exercise")
    if pe:
        trait_parts.append(f"primary exercise: {pe}")
    ahs = traits.get("afterhours_work_style")
    if ahs:
        style_desc = {
            "strict_boundary": "maintains strict work-life boundaries",
            "flexible": "has a flexible approach to after-hours work",
            "occasional": "occasionally works after hours",
        }
        trait_parts.append(style_desc.get(ahs, f"after-hours style: {ahs}"))
    if trait_parts:
        lines.append("Traits: " + "; ".join(trait_parts) + ".")

    # Routine snapshot
    sleep = routine.get("sleep", {})
    if sleep:
        parts = []
        if sleep.get("bedtime"):
            parts.append(f"typically goes to bed around {sleep['bedtime']}")
        if sleep.get("duration_h"):
            parts.append(f"sleeps about {sleep['duration_h']:.1f} hours")
        if parts:
            lines.append("Sleep habits: " + ", ".join(parts) + ".")

    exercise = routine.get("exercise", {})
    if exercise:
        parts = []
        if exercise.get("days_per_week"):
            parts.append(
                f"exercises about {exercise['days_per_week']:.1f} days per week"
            )
        if exercise.get("type"):
            parts.append(f"usually through {exercise['type']}")
        if parts:
            lines.append("Exercise: " + ", ".join(parts) + ".")

    diet = routine.get("diet", {})
    if diet:
        parts = []
        if diet.get("meals_per_day"):
            parts.append(f"averages {diet['meals_per_day']:.1f} meals per day")
        if diet.get("home_cooked_mean"):
            parts.append(
                f"about {diet['home_cooked_mean']:.1f} of those are home-cooked"
            )
        if parts:
            lines.append("Diet: " + ", ".join(parts) + ".")

    work = routine.get("work", {})
    if work:
        parts = []
        if work.get("hours_mean"):
            parts.append(f"works about {work['hours_mean']:.1f} hours per day")
        if work.get("overtime_share"):
            pct = work["overtime_share"] * 100
            parts.append(f"overtime on about {pct:.0f}% of days")
        if parts:
            lines.append("Work: " + ", ".join(parts) + ".")

    social = routine.get("social", {})
    if social:
        parts = []
        if social.get("active_days_per_week"):
            parts.append(
                f"socially active about {social['active_days_per_week']:.1f} "
                f"days per week"
            )
        if parts:
            lines.append("Social life: " + ", ".join(parts) + ".")

    return "\n".join(lines)


# ── Planner Renderer ─────────────────────────────────────────────────────────

def render_planner(records: list[dict]) -> str:
    """Render planner.json records to NL day-by-day entries."""
    lines = ["## Plans and Intentions", ""]

    for rec in records:
        date = rec.get("date", "unknown")
        parts = []

        # Sleep target
        st = rec.get("sleep_target", {})
        if st:
            bed = st.get("bedtime")
            wake = st.get("wake_time")
            dur = st.get("duration_h")
            if bed and dur:
                parts.append(
                    f"Planned bedtime {bed}, wake {_fmt_time(wake)}, "
                    f"targeting {dur:.1f} hours of sleep"
                )

        # Exercise target
        et = rec.get("exercise_target", {})
        if et:
            if et.get("intended"):
                ex_type = et.get("type", "exercise")
                dur = et.get("duration_min")
                parts.append(
                    f"Planned {ex_type}"
                    + (f" for {dur} minutes" if dur else "")
                )
            else:
                parts.append("No workout planned")

        # Work target
        wt = rec.get("work_target", {})
        if wt:
            limit = wt.get("hours_limit")
            finish = wt.get("finish_by")
            if limit:
                s = f"Work capped at {limit:.1f} hours"
                if finish:
                    s += f", finish by {finish}"
                parts.append(s)

        # Diet target
        dt = rec.get("diet_target", {})
        if dt:
            hc = dt.get("home_cooked_priority")
            if hc is True:
                parts.append("Home-cooked food prioritized")
            elif hc is False:
                parts.append("Home-cooked food not prioritized")

        # Social target
        soc = rec.get("social_target", {})
        if soc:
            if soc.get("intent"):
                parts.append("Social activity planned")
            else:
                parts.append("No social activity planned")

        lines.append(_date_heading(date))
        if parts:
            for p in parts:
                lines.append(f"- {p}.")
        else:
            lines.append("- No specific plans recorded.")
        lines.append("")

    return "\n".join(lines)


# ── Self-Report Renderer ─────────────────────────────────────────────────────

def render_self_report(records: list[dict]) -> str:
    """Render daily_self_report.json records to NL day-by-day entries."""
    lines = ["## Daily Self-Reports", ""]

    for rec in records:
        date = rec.get("date", "unknown")
        lines.append(_date_heading(date))
        day_parts: list[str] = []

        # Sleep
        sl = rec.get("sleep", {})
        if sl:
            bed = sl.get("bedtime")
            wake = sl.get("wake_time")
            dur = sl.get("duration_h")
            qual = sl.get("quality")
            s = "- Sleep:"
            if bed:
                s += f" went to bed at {bed}"
            if wake:
                s += f", woke at {wake}"
            if dur is not None:
                s += f", slept {dur:.1f} hours"
            if qual is not None:
                s += f", quality {_quality_word(qual)}"
            s += "."
            day_parts.append(s)

        # Exercise
        ex = rec.get("exercise", {})
        if ex:
            did = ex.get("did_exercise")
            if did:
                ex_type = ex.get("type", "exercise")
                dur = ex.get("duration_min")
                s = f"- Exercise: {ex_type}"
                if dur:
                    s += f", {dur} minutes"
                s += "."
                day_parts.append(s)
            else:
                day_parts.append("- Exercise: none.")

        # Diet
        di = rec.get("diet", {})
        if di:
            meals = di.get("meals")
            hc = di.get("home_cooked")
            s = "- Diet:"
            if meals is not None:
                s += f" {meals} meals total"
            if hc is not None:
                s += f", {hc} home-cooked"
            s += "."
            day_parts.append(s)

        # Work
        wk = rec.get("work", {})
        if wk:
            hrs = wk.get("hours")
            ot = wk.get("overtime")
            wknd = wk.get("worked_on_weekend")
            s = "- Work:"
            if hrs is not None:
                s += f" {hrs:.1f} hours"
            if ot is True:
                s += ", overtime"
            if wknd is True:
                s += ", worked on weekend"
            elif wknd is False:
                # Check if this is a weekend day — show explicit flag
                try:
                    d = _dt.strptime(date, "%Y-%m-%d")
                    if d.weekday() >= 5:  # Saturday=5, Sunday=6
                        s += ", no weekend work commitment"
                except (ValueError, TypeError):
                    pass
            s += "."
            day_parts.append(s)

        # Social
        soc = rec.get("social", {})
        if soc:
            activities = soc.get("activities", [])
            supporting = soc.get("supporting_other")
            if activities:
                act_strs = []
                for a in activities:
                    if isinstance(a, dict):
                        act_strs.append(
                            a.get("type", a.get("name", str(a)))
                        )
                    else:
                        act_strs.append(str(a))
                s = f"- Social: {', '.join(act_strs)}"
            else:
                s = "- Social: no activities"
            if supporting is True:
                s += ". Was supporting someone else"
            s += "."
            day_parts.append(s)

        for p in day_parts:
            lines.append(p)
        lines.append("")

    return "\n".join(lines)


# ── Objective Log Renderer ───────────────────────────────────────────────────

def render_objective_log(records: list[dict]) -> str:
    """Render objective_log.json records to NL day-by-day entries."""
    lines = ["## Objective Records", ""]

    for rec in records:
        date = rec.get("date", "unknown")
        avail = rec.get("available", True)

        lines.append(_date_heading(date))
        if not avail:
            lines.append("- Records unavailable for this date.")
            lines.append("")
            continue

        signals = rec.get("signals", {})
        day_parts: list[str] = []

        # Timesheet
        ts = signals.get("timesheet", {})
        if ts:
            hrs = ts.get("hours_logged")
            ot = ts.get("overtime_logged")
            s = f"- Timesheet: {hrs:.1f} hours logged"
            if ot:
                s += ", overtime reported"
            else:
                s += ", no overtime"
            s += "."
            day_parts.append(s)

        # Payments
        payments = signals.get("payments", [])
        if payments:
            pay_strs = []
            for p in payments:
                cat = p.get("category", "unknown").replace("_", " ")
                count = p.get("count", 1)
                pay_strs.append(
                    f"{count} {cat} purchase{'s' if count > 1 else ''}"
                )
            day_parts.append(f"- Payments: {', '.join(pay_strs)}.")

        # Calendar
        calendar = signals.get("calendar", [])
        if calendar:
            cal_strs = []
            for e in calendar:
                kind = e.get("kind", "event")
                friendly = kind.replace("_", " ")
                title = e.get("title", "")
                start = e.get("start", "")
                end = e.get("end", "")
                dur = e.get("duration_min")
                s = friendly
                if title:
                    s += f": {title}"
                if start and end:
                    s += f" ({start}-{end}"
                    if dur:
                        s += f", {dur} min"
                    s += ")"
                elif dur:
                    s += f" ({dur} min)"
                cal_strs.append(s)
            day_parts.append(f"- Calendar: {'; '.join(cal_strs)}.")

        if not day_parts:
            day_parts.append("- No records for this date.")

        for p in day_parts:
            lines.append(p)
        lines.append("")

    return "\n".join(lines)


# ── Device Log Renderer ──────────────────────────────────────────────────────

def render_device_log(records: list[dict]) -> str:
    """Render device_log.json records to NL day-by-day entries."""
    lines = ["## Device and Activity Records", ""]

    for rec in records:
        date = rec.get("date", "unknown")
        avail = rec.get("available", True)

        lines.append(_date_heading(date))
        if not avail:
            lines.append("- Records unavailable for this date.")
            lines.append("")
            continue

        signals = rec.get("signals", {})
        day_parts: list[str] = []

        # Sleep tracker
        st = signals.get("sleep_tracker", {})
        if st:
            bed = st.get("bedtime")
            wake = st.get("wake_time")
            dur = st.get("duration_h")
            s = "- Sleep tracker:"
            if bed:
                s += f" bed {bed}"
            if wake:
                s += f", wake {wake}"
            if dur is not None:
                s += f", {dur:.1f} hours"
            s += "."
            day_parts.append(s)

        # Activity tracker
        at = signals.get("activity_tracker", {})
        if at:
            active = at.get("active_minutes")
            workout = at.get("workout_detected")
            s = "- Activity:"
            if active is not None:
                s += f" {active} active minutes"
            if workout is True:
                s += ", workout detected"
            elif workout is False:
                s += ", no workout detected"
            s += "."
            day_parts.append(s)

        # Work session
        ws = signals.get("work_session", {})
        if ws:
            focus = ws.get("focus_minutes")
            late = ws.get("late_finish")
            if focus is not None:
                s = f"- Work session: {focus} minutes of focus time"
                if late is True:
                    s += ", finished late"
                elif late is False:
                    s += ", no late finish"
                s += "."
                day_parts.append(s)

        if not day_parts:
            day_parts.append("- No device records for this date.")

        for p in day_parts:
            lines.append(p)
        lines.append("")

    return "\n".join(lines)


# ── Top-Level API ────────────────────────────────────────────────────────────

SOURCE_RENDERERS = {
    "profile_ltm": render_profile,
    "planner": render_planner,
    "daily_self_report": render_self_report,
    "objective_log": render_objective_log,
    "device_log": render_device_log,
}


def render_source(source_name: str, data: Any) -> str:
    """Render a single source to NL text."""
    renderer = SOURCE_RENDERERS.get(source_name)
    if renderer is None:
        raise ValueError(f"Unknown source: {source_name}")
    return renderer(data)


def render_full_memory(sources: dict[str, Any]) -> str:
    """Render all 5 sources into a single NL memory document."""
    sections = []
    for sname in [
        "profile_ltm", "planner", "daily_self_report",
        "objective_log", "device_log",
    ]:
        data = sources.get(sname)
        if data is not None:
            sections.append(render_source(sname, data))
        else:
            sections.append(f"## {sname}\n\nData not available.")
    return "\n\n".join(sections)


def render_persona(persona_dir: Path) -> str:
    """Load and render all sources for a persona."""
    from .llm_extractor import load_sources_raw
    sources = load_sources_raw(persona_dir)
    return render_full_memory(sources)


def render_persona_to_file(persona_dir: Path, output_dir: Path) -> Path:
    """Render persona to NL and save as .md file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    pid = persona_dir.name
    out_file = output_dir / f"{pid}.md"
    text = render_persona(persona_dir)
    out_file.write_text(text, encoding="utf-8")
    return out_file
