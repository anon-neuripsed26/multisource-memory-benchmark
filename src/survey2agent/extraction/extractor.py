"""LLM-based atom extraction.

For each persona, this module:
  1. Renders the five source streams to NL text via
     `data_generation.nl_render.render_source`.
  2. Builds either a legacy single-question prompt or the released
     persona-level extraction prompt.
  3. Calls a `SyncLLMClient.complete()` (cache-aware).
  4. Parses the response into the frozen 18-question × 5-source atom grid.

Public API:
    extract_atom(persona_id, persona_dir, client, *, batch_mode=True) -> ExtractedAtom
    build_extraction_request(persona_dir, question_id, source) -> CompletionRequest
    build_batch_extraction_request(persona_dir, source, qids) -> CompletionRequest
    build_persona_extraction_request(persona_dir) -> CompletionRequest

The module is provider- and model-agnostic: the model is fully determined by
the `SyncLLMClient` instance passed in. For batch-only providers (OpenAI Batch
API, Gemini batches), call `build_persona_extraction_request` directly to
assemble one `CompletionRequest` per persona for batch submission.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from survey2agent.api_clients import (
    BatchLLMClient,
    CompletionRequest,
    SyncLLMClient,
)
from survey2agent.data_generation.nl_render.nl_memory_renderer import render_source

from .atoms import EXPECTED_QUESTION_IDS, EXPECTED_SOURCES, ExtractedAtom
from ._pydantic_schemas import build_persona_extraction_response_model
from .question_spec import QUESTIONS, QUESTION_TEXT


# ── Source-Question Relevance Map ────────────────────────────────────────────
# Which sources can produce a non-null observation for which question.
# Lifted from v1.0 reference

SOURCE_QUESTION_MAP: dict[str, list[str]] = {
    "A1": ["daily_self_report", "device_log", "planner", "profile_ltm"],
    "A2": ["daily_self_report", "device_log", "planner", "profile_ltm",
            "objective_log"],
    "A3": ["daily_self_report", "planner", "profile_ltm", "objective_log"],
    "B2": ["daily_self_report", "device_log", "planner", "profile_ltm",
            "objective_log"],
    "B3": ["daily_self_report", "device_log", "profile_ltm", "objective_log"],
    "C2": ["daily_self_report", "planner", "objective_log"],
    "C3": ["daily_self_report", "device_log"],
    "D1": ["daily_self_report", "planner", "objective_log"],
    "D2": ["daily_self_report", "profile_ltm"],
    "E1": ["daily_self_report", "device_log", "planner", "objective_log"],
    "E2": ["daily_self_report", "device_log", "planner", "objective_log",
            "profile_ltm"],
    "F1": ["daily_self_report", "objective_log", "planner", "profile_ltm"],
    "F2": ["daily_self_report", "device_log", "planner", "objective_log",
            "profile_ltm"],
    "F3": ["daily_self_report", "device_log", "planner", "profile_ltm"],
    "G1": ["daily_self_report", "planner", "device_log"],
    "G2": ["daily_self_report", "objective_log", "planner", "profile_ltm"],
    "Ctrl1": ["daily_self_report", "planner", "profile_ltm"],
    "Ctrl2": ["daily_self_report", "device_log", "planner"],
}

# Inverted: source -> qids that consume it (used in batch mode).
_SOURCE_TO_QIDS: dict[str, list[str]] = {}
for _qid, _srcs in SOURCE_QUESTION_MAP.items():
    for _src in _srcs:
        _SOURCE_TO_QIDS.setdefault(_src, []).append(_qid)


# ── Cross-source context map (some questions need a second source's text) ────

CONTEXT_MAP: dict[str, tuple[str, str]] = {
    "B2": ("profile_ltm",
           "The profile describes the person's exercise frequency as "
           "{exercise_days_per_week} days per week. Compare the observed "
           "frequency from the current source to this baseline."),
    "B3": ("profile_ltm",
           "The profile describes the person's after-hours work style as: "
           "\"{afterhours_work_style}\". Determine whether the weekend work "
           "pattern from the current source matches or contradicts this."),
    "C2": ("planner",
           "The planner shows social activity intentions. Use the planner "
           "to identify which days had social plans, then check whether this "
           "source confirms those plans were realized."),
    "C3": ("planner",
           "The planner shows target bedtimes. Use the planner targets to "
           "determine whether bedtimes in this source were within 20 minutes "
           "of the target."),
    "D2": ("profile_ltm",
           "The profile describes baseline meal habits: {meals_per_day} "
           "meals/day with {home_cooked_mean} home-cooked meals/day. "
           "Determine whether this source's 30-day averages differ from "
           "the profile by more than 1 meal/day."),
    "F1": ("planner",
           "The planner shows social activity intentions. An 'unplanned' "
           "social activity is one that occurred (per this source) on a day "
           "when the planner showed NO social intent."),
    "E2": ("planner",
           "The planner shows exercise intentions. Use the planner to "
           "identify which days had exercise planned, then check whether "
           "this source shows the exercise was actually done or skipped."),
    "F2": ("device_log",
           "The device/activity log shows which days had workout detected "
           "and which days had records unavailable (data dropout). Compare "
           "this source's exercise evidence against the device log's "
           "coverage to determine if missing tracker data = true inactivity "
           "or tracker dropout."),
    "F3": ("objective_log",
           "The objective records show which days had timesheet entries "
           "(available) and which were unavailable. Compare this source's "
           "work evidence on days when the objective log was unavailable "
           "to determine if missing entries = truly off or just timesheet "
           "dropout."),
}


# ── Per-question extraction hints ────────────────────────────────────────────

EXTRACTION_HINTS: dict[str, str] = {
    "A1": ("Count the number of nights with sleep duration ≥ 7 hours in the "
           "30-day window. Classify: fewer_than_10 (0-9), 10_to_19 (10-19), "
           "20_or_more (20-30). For device data with missing days, extrapolate "
           "to 30 days. For profile: if average sleep ≥ 7h, most nights ≥ 7h "
           "→ '20_or_more'. If average sleep < 6h → 'fewer_than_10'. "
           "Between 6-7h → '10_to_19'. For planner: if target sleep ≥ 7h, "
           "infer planned compliance → '20_or_more'."),
    "A2": ("Count the number of days with work hours > 9 in the 30-day window. "
           "Classify: 0_to_3 (0-3 days), 4_to_7 (4-7 days), 8_or_more (≥8 days). "
           "Only count WEEKDAYS (Mon-Fri) where work was actually reported. "
           "Use the date's day of week (shown in parentheses) to identify weekdays. "
           "Ignore weekend days unless they show work."),
    "A3": ("Calculate the share of home-cooked meals among all meals in 30 days. "
           "Classify: less_than_40 (<40%), 40_to_69 (40%-69%), 70_or_more (≥70%). "
           "For objective_log, use food delivery payment frequency as inverse proxy."),
    "B2": ("Compute the observed exercise frequency (days per week with exercise) "
           "from this source over 30 days. Compare to the profile frequency. "
           "Classify: more_than_1_below (>1 day/week less), within_1_day (±1 day), "
           "more_than_1_above (>1 day/week more), no_frequency_described (if "
           "profile has no exercise frequency). For profile source, answer "
           "'within_1_day' (self-consistency)."),
    "B3": ("Determine if weekend/after-hours work pattern matches the profile's "
           "stated work style. Use the day-of-week shown in parentheses next to "
           "each date to identify WEEKEND days (Saturday/Sunday). "
           "For self-report: look for explicit 'worked on weekend' markers "
           "(yes/no flag) on weekend days — do NOT infer from work hours. "
           "For device_log: check if work activity (focused work time) "
           "appears on weekend days. "
           "Count the fraction of weekend days with work. "
           "strict_boundary style: 'matches' if ≤15% of weekends had work; "
           "flexible style: 'does_not_match' only if zero work on ≥4 off-days. "
           "If the profile does not describe an after-hours work style, answer "
           "'no_approach_described'. For profile source itself, answer 'matches' "
           "(self-consistency)."),
    "C2": ("Identify days with social plans in the planner (last 14 days). "
           "For each planned day, check if this source shows social activity. "
           "Classify realization rate: below_25_pct, 25_to_50_pct, above_50_pct. "
           "If no planned social days, answer 'no_plans'. "
           "Planner itself is optimistic: answer 'above_50_pct'."),
    "C3": ("Last 14 days: compare actual bedtime to planner's target bedtime. "
           "Compute the absolute difference in minutes. If planned and actual "
           "are both near midnight (e.g., planned 23:50, actual 00:10 next day), "
           "the difference is 20 minutes (not 23 hours). Handle cross-midnight "
           "by treating times as continuous (e.g., 00:10 = 24h10m). "
           "Within 20 minutes = compliant. Classify: "
           "within_20min_more_than_50pct (>50% within), "
           "later_more_than_50pct (>50% later by >20min), "
           "earlier_more_than_50pct (>50% earlier by >20min), "
           "no_targets (planner has no bedtime targets)."),
    "D1": ("Split the 30-day window: first 14 days vs last 16 days. "
           "For self-report: count the total number of social activities "
           "in each half (sum up all activities, not just days with any). "
           "For planner: count the number of days with social intent "
           "in each half. "
           "Compute the RATE (count / number of days) for each half. "
           "If the rate changed by ≥0.15 (absolute), classify as "
           "'increased' (second half higher) or 'decreased' (second half "
           "lower). Otherwise, 'stayed_same'. "
           "Be precise: count each half separately and compute the ratio."),
    "D2": ("Compute 30-day average daily meals and home-cooked meals from "
           "this source. Compare to profile baseline. If the SUM of absolute "
           "differences (|actual_meals - profile_meals| + |actual_hc - profile_hc|) "
           "exceeds 1, answer 'differs_more_than_1'. Otherwise 'within_1'. "
           "If no baseline in profile, 'no_baseline'. "
           "Profile itself: 'within_1' (self-consistency)."),
    "E1": ("Find nights with bedtime after midnight (>24:00/00:00). "
           "For each late night, check if the person had OVERTIME "
           "(worked overtime that day or the 'overtime' flag was set) or "
           "had social activity. "
           "If >50% of late nights had overtime → 'work_activity'. "
           "If >50% had social activity → 'social_activity'. "
           "Otherwise 'no_single_factor'. If no late nights, 'no_late_nights'. "
           "Note: regular work activity (non-overtime) does NOT count — "
           "only overtime or unusually long work hours (>8.5h) count as "
           "a work-related cause of late bedtime."),
    "E2": ("Find days when exercise was planned (planner intent) but NOT done. "
           "On those skip days, check if work hours exceeded 8.5 hours. "
           "Classify share of skip-days with heavy work: "
           "yes_more_than_60 (>60%), between_30_60 (30-60%), "
           "no_fewer_than_30 (<30%)."),
    "F1": ("Count days with social activity in this source where the planner "
           "showed NO social intent. Classify: 0_to_3, 4_to_6, 7_or_more. "
           "If no social activities at all, 'no_social_activities'. "
           "For planner: few unplanned days (inverse signal)."),
    "F2": ("Assess exercise evidence by comparing THIS source against the "
           "device/activity log (provided as context). Focus on days when the "
           "device log shows 'Records unavailable' or no workout detected. "
           "If THIS source shows exercise on those days, the tracker likely "
           "missed it → 'yes_tracker_missing'. If THIS source also shows no "
           "exercise → 'inactive_confirmed'. If mixed → 'both_occurred'. "
           "For device_log itself: if many days are unavailable (>30%), "
           "answer 'yes_tracker_missing'; if all available and few workouts, "
           "'inactive_confirmed'. "
           "For profile: if exercise frequency ≥3 days/week → "
           "'yes_tracker_missing'; 1-2 days → 'both_occurred'; <1 → "
           "'inactive_confirmed'."),
    "F3": ("Assess work evidence by comparing THIS source against the "
           "objective records (provided as context). Focus on days when the "
           "objective log shows 'Records unavailable'. "
           "If THIS source shows work activity on those days (e.g., work "
           "hours > 0 in self-report, or more than 30 minutes of focused "
           "work activity on the device), "
           "the timesheet was just missing → 'yes_worked_despite_no_entry'. "
           "If THIS source also shows no work → 'truly_off'. "
           "If mixed → 'both_occurred'. "
           "For profile: if average work hours > 6 → "
           "'yes_worked_despite_no_entry'; 3-6 → 'both_occurred'; <3 → "
           "'truly_off'."),
    "G1": ("On days with physical activity, was it deliberate exercise "
           "(sessions ≥30 min) or incidental movement (<30 min like walking "
           "to work)? Classify: deliberate_exercise_70plus (≥70% deliberate), "
           "incidental_movement_70plus (≥70% incidental), mix. "
           "If no activity days, 'no_activity'. "
           "For self-report: use the exercise duration (30+ minutes = deliberate). "
           "For device_log: use 'workout detected' as proxy for deliberate "
           "exercise. If no workout detected but active minutes present, "
           "treat as incidental. "
           "For planner: all planned exercise sessions are deliberate."),
    "G2": ("For social activities attended, were they voluntary (by choice) "
           "or obligatory (work events, family commitments)? Classify: "
           "voluntary_70plus (≥70% voluntary), obligatory_70plus (≥70% "
           "obligatory), mix. If no social activities, 'no_meetings'. "
           "Classification rule: a day's social activity is OBLIGATORY only "
           "if the activity was for someone else's benefit (helping/caregiving, "
           "work-mandated events, family obligations) rather than by personal "
           "choice. All other social activities "
           "(dinner with friends, book club, bar trivia, coffee catch-up) "
           "are VOLUNTARY. "
           "For objective_log (calendar records): calendar doesn't distinguish "
           "voluntary/obligatory → answer 'mix'. "
           "For planner: planned social is self-initiated → 'voluntary_70plus'. "
           "For profile: no vol/oblig info → 'mix'."),
    "Ctrl1": ("Last 7 days: count days with at least one meal eaten outside "
              "the home (i.e., meals > home_cooked on that day, or food_orders "
              "present). Classify: 0_to_1_days, 2_to_3_days, 4_or_more."),
    "Ctrl2": ("Last 7 days/records: count nights with sleep duration < 6 hours. "
              "Classify: 0_nights, 1_to_2, 3_or_more."),
}


SOURCE_SPECIFIC_HINTS: dict[tuple[str, str], str] = {
    ("E1", "objective_log"): (
        "For objective_log specifically: this source has NO sleep/bedtime data. "
        "Instead, use work block duration and social calendar entries as proxy. "
        "Count days with timesheet hours > 9.0 (long work day) and days with "
        "social_event calendar entries. If long_work_days / total_available > 0.3 "
        "→ 'work_activity'. If social_event_days / total_available > 0.3 "
        "→ 'social_activity'. Otherwise → 'no_single_factor'. "
        "If no data available → null."
    ),
    ("G1", "device_log"): (
        "For device_log specifically: consider ONLY days where 'workout detected' "
        "is True as your sample (denominator). Among those workout days: "
        "active_minutes ≥ 30 = deliberate exercise, active_minutes < 30 = "
        "incidental movement. Days with active minutes but NO workout detected "
        "should be IGNORED entirely — they are not exercise days."
    ),
    ("E2", "profile_ltm"): (
        "For profile_ltm specifically: use the overtime percentage from the "
        "work routine. If overtime share > 60% → 'yes_more_than_60'. "
        "If 30-60% → 'between_30_60'. If < 30% → 'no_fewer_than_30'. "
        "If no overtime info available → null."
    ),
    ("E2", "planner"): (
        "For planner specifically: count days where BOTH exercise was planned "
        "(intended=true) AND work hours limit > 8. The ratio of such conflict "
        "days to total exercise-intended days is the work-exercise conflict rate. "
        "If conflict rate > 60% → 'yes_more_than_60'. If 30-60% → "
        "'between_30_60'. If < 30% → 'no_fewer_than_30'."
    ),
    ("F2", "planner"): (
        "For planner specifically: count the number of days with exercise "
        "intended (intended=true) out of total days. "
        "If ≥ 15 out of 30 → 'yes_tracker_missing' (active intent suggests "
        "tracker undercount). If 5-14 out of 30 → 'both_occurred'. "
        "If < 5 out of 30 → 'inactive_confirmed'."
    ),
    ("B3", "objective_log"): (
        "For objective_log specifically: timesheet entries with hours_logged "
        "≤ 0.5 on weekend days should NOT be counted as weekend work. Only "
        "formal work blocks or timesheet entries > 0.5 hours count as "
        "meaningful weekend work when determining the work pattern."
    ),
    ("F1", "profile_ltm"): (
        "For profile_ltm specifically: estimate that roughly 50% of total "
        "social activities are unplanned. Use the social active_days_per_week "
        "from the profile. If social frequency ≥ 3 days/week → '7_or_more' "
        "unplanned activities. If 1.5-3 days/week → '4_to_6'. "
        "If < 1.5 days/week → '0_to_3'."
    ),
}


# ── System prompts ───────────────────────────────────────────────────────────

SYSTEM_EXTRACT = """\
You are an expert data analyst extracting structured observations from \
personal memory data. You are given data from a SINGLE source about one person.

Your task: Read the source data and determine the best answer to a specific \
question based on information available in this source.

Rules:
1. Use ONLY the data from the specified source. Do not infer from missing data \
unless explicitly instructed.
2. If this source contains NO relevant information for the question, answer: null
3. When counting events over multiple days, be precise.
4. For device/objective data with missing days (available=false), extrapolate \
proportionally to the full time window if there are enough available days.
5. Respond with EXACTLY one answer label from the provided options, or "null".

Format your response as:
REASONING: <1-3 sentences explaining your analysis>
ANSWER: <exactly one option label or null>

IMPORTANT: Your ANSWER must be EXACTLY one of the option labels listed \
in the question (with underscores, e.g., '20_or_more'), or 'null'. \
Do NOT use the letter labels (a, b, c) in your answer.
"""


SYSTEM_BATCH_EXTRACT = """\
You are an expert data analyst extracting structured observations from \
personal memory data. You are given data from a PRIMARY source about one \
person, plus optional reference data from other sources for context.

Your task: For each question listed, determine the best answer based on \
information available in the PRIMARY source.

Rules:
1. Use ONLY the primary source data for observation. Reference data is only \
for context (e.g., comparing to baselines).
2. If the primary source contains NO relevant information for a question, \
answer null for that question.
3. When counting events over multiple days, be precise.
4. For device/objective data with missing days (available=false), extrapolate \
proportionally to the full time window if there are enough available days.
5. Respond with a JSON object mapping each question ID to exactly one answer \
label from the provided options, or null.
"""


SYSTEM_PERSONA_EXTRACT = """\
You are an expert data analyst extracting structured observations from \
multi-source personal memory data. You are given all five memory sources for \
one person.

Your task: For every listed question, extract what each source says. Do NOT \
resolve conflicts across sources. Each source cell should contain the best \
closed-class label supported by that source, or null if that source is not \
informative for the question.

Rules:
1. Fill the full 18-question x 5-source grid.
2. Use each source only for its own observation. Other sources may be used \
only where the question guidance explicitly asks for comparison context.
3. For sources that are not listed as informative for a question, output null.
4. When counting events over multiple days, be precise.
5. Respond with JSON only: {question_id: {source_name: label_or_null}}.
"""


# ── Prompt builders ─────────────────────────────────────────────────────────

def _format_answer_options(qid: str) -> str:
    """Format the answer space as lettered options."""
    options = QUESTIONS[qid]["answer_space"]
    letters = "abcdefghijklmnop"
    lines = []
    for i, opt in enumerate(options):
        label = opt.replace("_", " ")
        lines.append(f"  ({letters[i]}) {opt} — {label}")
    return "\n".join(lines)


def _fill_profile_template(desc_template: str, profile_data: Any) -> str:
    """Substitute profile-derived placeholders in a context description."""
    if not isinstance(profile_data, dict):
        return desc_template
    facts = profile_data.get("facts", {})
    routine = facts.get("routine_snapshot", {})
    desc = desc_template
    desc = desc.replace(
        "{exercise_days_per_week}",
        str(routine.get("exercise", {}).get("days_per_week", "unknown")),
    )
    desc = desc.replace(
        "{afterhours_work_style}",
        str(facts.get("traits", {}).get("afterhours_work_style", "unknown")),
    )
    desc = desc.replace(
        "{meals_per_day}",
        str(routine.get("diet", {}).get("meals_per_day", "unknown")),
    )
    desc = desc.replace(
        "{home_cooked_mean}",
        str(routine.get("diet", {}).get("home_cooked_mean", "unknown")),
    )
    return desc


def _get_context_text(
    qid: str,
    source_name: str,
    sources_raw: dict[str, Any],
) -> str | None:
    """Render the cross-source context block for `qid`, or `None` if unused."""
    if qid not in CONTEXT_MAP:
        return None
    ctx_source, desc_template = CONTEXT_MAP[qid]
    # Don't add context when extracting FROM the context source itself.
    if source_name == ctx_source:
        return None
    ctx_data = sources_raw.get(ctx_source)
    if ctx_data is None:
        return None
    ctx_text = render_source(ctx_source, ctx_data)
    desc = desc_template
    if ctx_source == "profile_ltm":
        desc = _fill_profile_template(desc, ctx_data)
    return f"## Context from {ctx_source}\n{desc}\n\n{ctx_text}"


def build_extraction_prompt(
    qid: str,
    source_name: str,
    source_nl: str,
    context_text: str | None = None,
) -> str:
    """Build the user prompt for a single (qid, source) extraction call."""
    q = QUESTIONS[qid]
    time_window = q["time_window"]
    question_text = QUESTION_TEXT[qid]
    options = _format_answer_options(qid)
    hint = EXTRACTION_HINTS.get(qid, "")
    src_hint = SOURCE_SPECIFIC_HINTS.get((qid, source_name), "")
    if src_hint:
        hint = (hint + "\n\n" + src_hint) if hint else src_hint
    edge = q.get("edge_options", [])

    parts: list[str] = [f"## Source: {source_name}", source_nl, ""]
    if context_text:
        parts.extend([context_text, ""])
    parts.extend([
        f"## Question ({qid})",
        question_text,
        "",
        f"Time window: past {time_window} days",
        "",
        "## Answer Options",
        options,
        "",
    ])
    if edge:
        parts.append(
            f"Edge-case options ({', '.join(edge)}): use these ONLY when the "
            "prerequisite condition genuinely does not exist in the data "
            "(e.g., no plans, no baseline described, no activity at all)."
        )
        parts.append("")
    if hint:
        parts.extend([f"## Extraction Guidance\n{hint}", ""])
    parts.append(
        "Based on the data from this source, provide your answer.\n"
        "REASONING: <brief analysis>\n"
        "ANSWER: <one option label or null>"
    )
    return "\n".join(parts)


def build_batch_extraction_prompt(
    source_name: str,
    source_nl: str,
    qids: list[str],
    sources_raw: dict[str, Any],
) -> str:
    """Build the user prompt for batched extraction across many qids of one source."""
    parts: list[str] = [f"## Primary Source: {source_name}", source_nl, ""]

    # Render unique context-source NL bodies.
    ctx_renders: dict[str, str] = {}
    for qid in qids:
        if qid in CONTEXT_MAP:
            ctx_source, _ = CONTEXT_MAP[qid]
            if ctx_source != source_name and ctx_source not in ctx_renders:
                ctx_data = sources_raw.get(ctx_source)
                if ctx_data is not None:
                    ctx_renders[ctx_source] = render_source(ctx_source, ctx_data)
    if ctx_renders:
        parts.append("## Reference Data (from other sources)")
        for ctx_src, ctx_text in ctx_renders.items():
            parts.extend([f"### {ctx_src}", ctx_text, ""])

    parts.append(f"## Questions ({len(qids)} total)")
    parts.append("")
    for qid in qids:
        q = QUESTIONS[qid]
        question_text = QUESTION_TEXT[qid]
        options = _format_answer_options(qid)
        hint = EXTRACTION_HINTS.get(qid, "")
        src_hint = SOURCE_SPECIFIC_HINTS.get((qid, source_name), "")
        if src_hint:
            hint = (hint + " " + src_hint) if hint else src_hint
        edge = q.get("edge_options", [])
        time_window = q["time_window"]

        parts.append(f"### {qid}")
        parts.append(question_text)
        parts.append(f"Time window: past {time_window} days")

        if qid in CONTEXT_MAP:
            ctx_source, desc_template = CONTEXT_MAP[qid]
            if ctx_source != source_name:
                desc = desc_template
                if ctx_source == "profile_ltm":
                    desc = _fill_profile_template(desc, sources_raw.get(ctx_source))
                parts.append(
                    f"Context: Refer to {ctx_source} in Reference Data. {desc}"
                )

        parts.append("Options:")
        parts.append(options)
        if edge:
            parts.append(
                f"Edge-case options ({', '.join(edge)}): only when applicable."
            )
        if hint:
            parts.append(f"Guidance: {hint}")
        parts.append("")

    parts.append(
        "## Output\n"
        "Respond with a JSON object mapping each question ID to exactly one "
        "answer option label, or null if the source has no relevant data.\n"
        'Example: {"A1": "20_or_more", "A2": null}\n\n'
        "JSON:"
    )
    return "\n".join(parts)


def build_persona_extraction_prompt(sources_raw: dict[str, Any]) -> str:
    """Build the released persona-level extraction prompt.

    The prompt contains all five rendered sources and asks for one full atom
    grid: ``{qid: {source: label_or_null}}``.
    """
    parts: list[str] = ["## Memory Sources", ""]
    for source in EXPECTED_SOURCES:
        parts.extend([
            f"### Source: {source}",
            render_source(source, sources_raw.get(source, {} if source == "profile_ltm" else [])),
            "",
        ])

    parts.append(f"## Extraction Spec ({len(EXPECTED_QUESTION_IDS)} questions)")
    parts.append("")
    for qid in EXPECTED_QUESTION_IDS:
        q = QUESTIONS[qid]
        question_text = QUESTION_TEXT[qid]
        options = _format_answer_options(qid)
        informative = SOURCE_QUESTION_MAP.get(qid, [])
        hint = EXTRACTION_HINTS.get(qid, "")
        edge = q.get("edge_options", [])

        parts.append(f"### {qid}")
        parts.append(question_text)
        parts.append(f"Time window: past {q['time_window']} days")
        parts.append(f"Informative sources: {', '.join(informative)}")
        parts.append("All other sources must be null for this question.")

        if qid in CONTEXT_MAP:
            ctx_source, desc_template = CONTEXT_MAP[qid]
            desc = desc_template
            if ctx_source == "profile_ltm":
                desc = _fill_profile_template(desc, sources_raw.get(ctx_source))
            parts.append(f"Cross-source context allowed: {ctx_source}. {desc}")

        parts.append("Options:")
        parts.append(options)
        if edge:
            parts.append(
                f"Edge-case options ({', '.join(edge)}): only when applicable."
            )
        if hint:
            parts.append(f"General guidance: {hint}")

        source_hints = [
            f"- {src}: {text}"
            for (hint_qid, src), text in SOURCE_SPECIFIC_HINTS.items()
            if hint_qid == qid
        ]
        if source_hints:
            parts.append("Source-specific guidance:")
            parts.extend(source_hints)
        parts.append("")

    source_keys = ", ".join(EXPECTED_SOURCES)
    parts.extend([
        "## Output",
        "Return JSON only. The top-level keys must be the question IDs. "
        "Each question maps to an object with exactly these source keys: "
        f"{source_keys}.",
        "Use one canonical option label from that question's answer options, "
        "or null. Do not include reasoning text.",
        'Example: {"A1": {"profile_ltm": "20_or_more", "planner": null, '
        '"daily_self_report": "10_to_19", "objective_log": null, '
        '"device_log": "20_or_more"}}',
        "",
        "JSON:",
    ])
    return "\n".join(parts)


# ── Response parsers ────────────────────────────────────────────────────────

def parse_extraction_response(response: str, qid: str) -> str | None:
    """Parse a single-question extraction response into an answer label or None.

    Looks for the last `ANSWER:` line. Performs exact-match, letter-label,
    and substring fuzzy match against the question's answer space. Returns
    None for explicit null tokens or when no match can be made.
    """
    answer_space = QUESTIONS[qid]["answer_space"]
    for line in reversed(response.strip().split("\n")):
        line = line.strip()
        if not line.upper().startswith("ANSWER:"):
            continue
        label = line.split(":", 1)[1].strip().lower().strip("() ")
        if label in ("null", "none", "n/a", "⊥", "abstain"):
            return None
        for opt in answer_space:
            if label == opt or label == opt.replace("_", " "):
                return opt
        if len(label) == 1 and label.isalpha():
            idx = ord(label) - ord("a")
            if 0 <= idx < len(answer_space):
                return answer_space[idx]
        for opt in answer_space:
            if opt in label or opt.replace("_", " ") in label:
                return opt
        return None
    return None


def _strip_json_fence(response: str) -> str:
    text = response.strip()
    fenced = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if fenced:
        return fenced.group(1).strip()
    return text


def _canonicalize_label(qid: str, val: Any) -> str | None:
    if val is None:
        return None
    if not isinstance(val, str):
        return None
    label = val.strip().lower()
    if label in ("null", "none", "n/a", "⊥", "abstain"):
        return None
    answer_space = QUESTIONS[qid]["answer_space"]
    for opt in answer_space:
        if label == opt or label == opt.replace("_", " "):
            return opt
    for opt in answer_space:
        if opt in label or opt.replace("_", " ") in label:
            return opt
    return None


def parse_batch_response(response: str, qids: list[str]) -> dict[str, str | None]:
    """Parse a batched extraction JSON response.

    Tolerates ```json``` fenced blocks and bare-object fragments. Unparseable
    or unmatchable answers default to None.
    """
    result: dict[str, str | None] = {qid: None for qid in qids}
    text = _strip_json_fence(response)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{[^{}]*\}", text)
        if m:
            try:
                data = json.loads(m.group())
            except json.JSONDecodeError:
                return result
        else:
            return result
    if not isinstance(data, dict):
        return result
    for qid in qids:
        result[qid] = _canonicalize_label(qid, data.get(qid))
    return result


def parse_persona_extraction_response(
    response: str,
) -> dict[str, dict[str, str | None]]:
    """Parse the released persona-level extraction JSON response.

    Accepts either ``{qid: {source: label_or_null}}`` or a wrapped
    ``{"extraction": ...}`` object. Missing, malformed, non-informative, and
    unmatchable cells default to null.
    """
    result = _empty_atom_grid()
    text = _strip_json_fence(response)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return result
    if not isinstance(data, dict):
        return result
    if isinstance(data.get("extraction"), dict):
        data = data["extraction"]

    for qid in EXPECTED_QUESTION_IDS:
        per_qid = data.get(qid)
        if not isinstance(per_qid, dict):
            continue
        informative = set(SOURCE_QUESTION_MAP.get(qid, []))
        for source in EXPECTED_SOURCES:
            if source not in informative:
                result[qid][source] = None
                continue
            result[qid][source] = _canonicalize_label(qid, per_qid.get(source))
    return result


# ── Source loader ───────────────────────────────────────────────────────────

def load_sources_raw(persona_dir: Path) -> dict[str, Any]:
    """Load the 5 structural source JSONs for one persona.

    `profile_ltm.json` is returned as the full dict; the other four return
    their `.records` list. Missing files yield an empty list/dict so that
    downstream code never crashes on incomplete personas.
    """
    persona_dir = Path(persona_dir)
    src_dir = persona_dir / "structural_sources"
    out: dict[str, Any] = {}
    for name, fname in (
        ("profile_ltm", "profile_ltm.json"),
        ("planner", "planner.json"),
        ("daily_self_report", "daily_self_report.json"),
        ("objective_log", "objective_log.json"),
        ("device_log", "device_log.json"),
    ):
        fpath = src_dir / fname
        if fpath.exists():
            raw = json.loads(fpath.read_text(encoding="utf-8"))
            out[name] = raw if name == "profile_ltm" else raw.get("records", [])
        else:
            out[name] = {} if name == "profile_ltm" else []
    return out


# ── Public extraction entry point ────────────────────────────────────────────


def build_extraction_request(
    persona_dir: Path,
    question_id: str,
    source: str,
) -> CompletionRequest:
    """Build a single-question `CompletionRequest` for one (qid, source) pair.

    Pure: reads only `persona_dir/structural_sources/*.json` and the static
    question / source spec. Suitable for batch submission.
    """
    persona_dir = Path(persona_dir)
    sources_raw = load_sources_raw(persona_dir)
    if source not in sources_raw:
        raise KeyError(f"source {source!r} not found under {persona_dir}")
    source_nl_text = render_source(source, sources_raw[source])
    ctx = _get_context_text(question_id, source, sources_raw)
    user_prompt = build_extraction_prompt(question_id, source, source_nl_text, context_text=ctx)
    return CompletionRequest(
        user_prompt=user_prompt,
        system_prompt=SYSTEM_EXTRACT,
    )


def build_batch_extraction_request(
    persona_dir: Path,
    source: str,
    qids: list[str],
) -> CompletionRequest:
    """Build a single batched `CompletionRequest` covering many qids of one source."""
    persona_dir = Path(persona_dir)
    sources_raw = load_sources_raw(persona_dir)
    if source not in sources_raw:
        raise KeyError(f"source {source!r} not found under {persona_dir}")
    source_nl_text = render_source(source, sources_raw[source])
    user_prompt = build_batch_extraction_prompt(source, source_nl_text, qids, sources_raw)
    return CompletionRequest(
        user_prompt=user_prompt,
        system_prompt=SYSTEM_BATCH_EXTRACT,
    )


def build_persona_extraction_request(
    persona_dir: Path,
    *,
    include_response_schema: bool = True,
) -> CompletionRequest:
    """Build one released extraction request for a persona's full atom grid."""
    persona_dir = Path(persona_dir)
    sources_raw = load_sources_raw(persona_dir)
    user_prompt = build_persona_extraction_prompt(sources_raw)
    response_schema = None
    if include_response_schema:
        response_schema = build_persona_extraction_response_model(
            EXPECTED_QUESTION_IDS,
            EXPECTED_SOURCES,
        )
    return CompletionRequest(
        user_prompt=user_prompt,
        system_prompt=SYSTEM_PERSONA_EXTRACT,
        response_schema=response_schema,
    )


def _empty_atom_grid() -> dict[str, dict[str, str | None]]:
    """Initialize a fresh {qid: {source: None}} grid for all 18 × 5 cells."""
    return {qid: {src: None for src in EXPECTED_SOURCES} for qid in EXPECTED_QUESTION_IDS}


def _extract_single_mode(
    persona_id: str,
    source_nl: dict[str, str],
    sources_raw: dict[str, Any],
    client: SyncLLMClient,
) -> dict[str, dict[str, str | None]]:
    grid = _empty_atom_grid()
    for qid in EXPECTED_QUESTION_IDS:
        for sname in SOURCE_QUESTION_MAP.get(qid, []):
            if sname not in source_nl:
                continue
            ctx = _get_context_text(qid, sname, sources_raw)
            user_prompt = build_extraction_prompt(qid, sname, source_nl[sname], context_text=ctx)
            request = CompletionRequest(
                user_prompt=user_prompt,
                system_prompt=SYSTEM_EXTRACT,
            )
            result = client.complete(request, allow_api_call=True)
            grid[qid][sname] = parse_extraction_response(result.text, qid)
    return grid


def _extract_batch_mode(
    persona_id: str,
    source_nl: dict[str, str],
    sources_raw: dict[str, Any],
    client: SyncLLMClient,
) -> dict[str, dict[str, str | None]]:
    request = CompletionRequest(
        user_prompt=build_persona_extraction_prompt(sources_raw),
        system_prompt=SYSTEM_PERSONA_EXTRACT,
    )
    result = client.complete(request, allow_api_call=True)
    return parse_persona_extraction_response(result.text)


def extract_atom(
    persona_id: str,
    persona_dir: Path,
    client: SyncLLMClient,
    *,
    batch_mode: bool = True,
) -> ExtractedAtom:
    """Extract an atom for one persona using a synchronous LLM client.

    Args:
        persona_id: Logical id (e.g., ``"bench_shift_121_avery_ellis"``).
            Stored on the returned `ExtractedAtom`.
        persona_dir: Directory containing ``structural_sources/*.json``.
        client: A bound `SyncLLMClient` instance. Caching and provider routing
            are handled by the client. For batch-only providers, call
            `build_persona_extraction_request` directly to assemble requests
            for batch submission.
        batch_mode: If True (default), issue one persona-level LLM call that
            returns the full 18-question x 5-source grid. If False, issue one
            call per (source, qid).

    Returns:
        A validated frozen `ExtractedAtom` with all 18 qids and 5 sources
        populated. Sources outside `SOURCE_QUESTION_MAP[qid]` are `None`.
    """
    if isinstance(client, BatchLLMClient):
        raise NotImplementedError(
            "extract_atom() expects a SyncLLMClient. Use "
            "build_persona_extraction_request to assemble CompletionRequest "
            "objects for batch submission."
        )
    persona_dir = Path(persona_dir)
    sources_raw = load_sources_raw(persona_dir)
    source_nl = {sname: render_source(sname, sdata) for sname, sdata in sources_raw.items()}

    if batch_mode:
        grid = _extract_batch_mode(persona_id, source_nl, sources_raw, client)
    else:
        grid = _extract_single_mode(persona_id, source_nl, sources_raw, client)

    return ExtractedAtom.from_json({"persona": persona_id, "extraction": grid})
