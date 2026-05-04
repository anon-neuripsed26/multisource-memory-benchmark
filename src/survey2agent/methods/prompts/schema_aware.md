# Survey Questions — Schema-Aware Variant

Answer ALL 18 questions. For each question, select exactly one answer label from the answer space.

You are provided with **source relevance** (which sources are informative), **context instructions** (cross-source comparisons needed), and **reasoning hints** (how to compute or judge the answer). Use all of this to guide your synthesis.

---

## Source Relevance Map

Which sources are informative for which questions:

| Question | profile_ltm | planner | daily_self_report | objective_log | device_log |
|----------|:-----------:|:-------:|:-----------------:|:-------------:|:----------:|
| A1       | ✓ | ✓ | ✓ |   | ✓ |
| A2       | ✓ | ✓ | ✓ | ✓ | ✓ |
| A3       | ✓ | ✓ | ✓ | ✓ |   |
| B2       | ✓ | ✓ | ✓ | ✓ | ✓ |
| B3       | ✓ |   | ✓ | ✓ | ✓ |
| C2       |   | ✓ | ✓ | ✓ |   |
| C3       |   |   | ✓ |   | ✓ |
| D1       |   | ✓ | ✓ | ✓ |   |
| D2       | ✓ |   | ✓ |   |   |
| E1       |   | ✓ | ✓ | ✓ | ✓ |
| E2       | ✓ | ✓ | ✓ | ✓ | ✓ |
| F1       | ✓ | ✓ | ✓ | ✓ |   |
| F2       | ✓ | ✓ | ✓ | ✓ | ✓ |
| F3       | ✓ | ✓ | ✓ |   | ✓ |
| G1       |   | ✓ | ✓ |   | ✓ |
| G2       | ✓ | ✓ | ✓ | ✓ |   |
| Ctrl1    | ✓ | ✓ | ✓ |   |   |
| Ctrl2    |   | ✓ | ✓ |   | ✓ |

---

## Cross-Source Context Instructions

Some questions require comparing information across sources. Follow these when answering:

| Question | Context Source | Instruction |
|----------|--------------|-------------|
| B2 | profile_ltm | The profile describes exercise frequency as N days/week. Compare the observed frequency from other sources to this baseline. |
| B3 | profile_ltm | The profile describes the person's after-hours work style. Determine whether the weekend work pattern from other sources matches or contradicts this. |
| C2 | planner | The planner shows social activity intentions. Identify which days had social plans, then check whether other sources confirm those plans were realized. |
| C3 | planner | The planner shows target bedtimes. Determine whether actual bedtimes were within 20 minutes of the target. |
| D2 | profile_ltm | The profile describes baseline meal habits (meals/day and home-cooked meals/day). Determine whether the 30-day averages from other sources differ by more than 1 meal/day. |
| E2 | planner | The planner shows exercise intentions. Identify which days had exercise planned, then check whether other sources confirm the exercise was done or skipped. |
| F1 | planner | The planner shows social activity intentions. An 'unplanned' social activity is one that occurred on a day when the planner showed NO social intent. |
| F2 | device_log | The device/activity log shows which days had workout detected and which days had records unavailable. Compare exercise evidence from other sources against device log coverage. |
| F3 | objective_log | The objective records show which days had timesheet entries and which were unavailable. Compare work evidence from other sources on days when the objective log was unavailable. |

---

## Source Bias Characteristics

Each data source has characteristic biases that affect how reliably it reflects reality. Consider these when sources disagree:

| Source | General Bias Pattern |
|--------|---------------------|
| **Profile/Background** | May be outdated or idealized. People describe themselves as they want to be, not always as they are. Stated habits (exercise frequency, diet quality) are often more favorable than actual behavior. |
| **Planner** | Optimistic by nature — plans often exceed what actually happens. Social plans, exercise plans, and healthy eating intentions frequently go unrealized. |
| **Daily Self-Report** | Direction varies by topic. People tend to over-report diet quality and exercise effort, but under-report social isolation and work-related stress. Self-reports are most accurate for routine events but less reliable for unusual or embarrassing events. |
| **Objective Records** | Most factually accurate when present, but may have gaps (days with no entry). An absent record does NOT necessarily mean inactivity — the recording system itself may have failed. |
| **Device/Activity Log** | Precise when working, but prone to data dropout (some days may have no data). A missing device record should not be interpreted as evidence of inactivity — the device may simply not have been worn or synced. |

When sources disagree, use these bias patterns to judge which source is more likely to reflect reality for that specific question topic.

---

## Questions with Hints

### A1 — Sleep Duration Count (Sleep, 30 days)
**Text**: In the past 30 days, how many nights did this person sleep 7 hours or more?
**Answer space**: `fewer_than_10`, `10_to_19`, `20_or_more`
**Source relevance**: profile_ltm, planner, daily_self_report, device_log

**Hint**: Count nights with sleep duration ≥ 7 hours in the 30-day window. Classify: fewer_than_10 (0-9), 10_to_19 (10-19), 20_or_more (20-30). For device data with missing days, extrapolate to 30 days. Profile average sleep ≥ 7h → most nights ≥ 7h. Profile average < 6h → few nights. Planner target ≥ 7h → planned compliance. When sources disagree, consider each source's bias characteristics (see above) to judge which is most likely correct.

---

### A2 — Long Work Days Count (Work, 30 days)
**Text**: In the past 30 days, on how many days did this person work more than 9 hours?
**Answer space**: `0_to_3`, `4_to_7`, `8_or_more`
**Source relevance**: profile_ltm, planner, daily_self_report, objective_log, device_log

**Hint**: Count days with work hours > 9 in the 30-day window, including worked weekends. Do not count non-work weekend days as long-work days. Use the date's day of week (shown in parentheses) only to interpret weekend context. When sources disagree, consider each source's bias characteristics to judge which is most likely correct.

---

### A3 — Home-Cooked Share (Diet, 30 days)
**Text**: In the past 30 days, on what share of meal occasions did this person eat home-cooked food?
**Answer space**: `less_than_40`, `40_to_69`, `70_or_more`
**Source relevance**: profile_ltm, planner, daily_self_report, objective_log

**Hint**: Calculate the share of home-cooked meals among all meals in 30 days. For objective_log, use food delivery payment frequency as inverse proxy. When sources disagree, consider each source's bias characteristics to judge which is most likely correct.

---

### B2 — Exercise Frequency vs Profile (Exercise, 30 days)
**Text**: This person's profile describes exercising a certain number of days per week. In the past 30 days, how did their actual exercise frequency compare to that description?
**Answer space**: `more_than_1_below`, `within_1_day`, `more_than_1_above`, `no_frequency_described`
**Source relevance**: profile_ltm, planner, daily_self_report, objective_log, device_log
**Context**: Compare against profile's stated exercise frequency.

**Hint**: Compute observed exercise frequency (days per week with exercise) from behavioral sources over 30 days. Compare to profile frequency. Classify: more_than_1_below (>1 day/week less), within_1_day (±1 day), more_than_1_above (>1 day/week more), no_frequency_described (if profile has no exercise frequency). Consider each source's bias characteristics when determining actual frequency.

---

### B3 — Weekend Work vs Profile (Work, 30 days)
**Text**: This person's profile describes a specific approach to weekend work. In the past 30 days, did their actual weekend work pattern match that description?
**Answer space**: `matches`, `does_not_match`, `no_approach_described`
**Source relevance**: profile_ltm, daily_self_report, objective_log, device_log
**Context**: Compare against profile's stated work style.

**Hint**: Determine if weekend/after-hours work pattern matches the profile's stated work style. Use the day-of-week shown in parentheses next to each date to identify WEEKEND days (Saturday/Sunday). For self-report: look for explicit 'worked on weekend' markers. For device_log: check if work activity appears on weekend days. If the profile does not describe an after-hours work style, answer 'no_approach_described'.

---

### C2 — Social Plan Realization (Social, 14 days)
**Text**: In the past 14 days, on days when this person planned a social activity, what share of those plans were actually realized?
**Answer space**: `below_25_pct`, `25_to_50_pct`, `above_50_pct`, `no_plans`
**Source relevance**: planner, daily_self_report, objective_log
**Context**: Identify planned social days from planner, check realization in other sources.

**Hint**: Identify days with social plans in the planner (last 14 days). For each planned day, check if self-report or objective records show social activity. Classify realization rate: below_25_pct, 25_to_50_pct, above_50_pct. If no planned social days, answer 'no_plans'.

---

### C3 — Bedtime vs Target (Sleep, 14 days)
**Text**: In the past 14 days, on days when this person set a target bedtime, did they go to bed within 20 minutes of that target?
**Answer space**: `within_20min_more_than_50pct`, `later_more_than_50pct`, `earlier_more_than_50pct`, `no_targets`
**Source relevance**: daily_self_report, device_log
**Context**: Compare against planner's target bedtimes.

**Hint**: Last 14 days: compare actual bedtime to planner's target bedtime. Handle cross-midnight by treating times as continuous (e.g., 00:10 = 24h10m). Within 20 minutes = compliant. Classify: within_20min_more_than_50pct (>50% within), later_more_than_50pct (>50% later by >20min), earlier_more_than_50pct (>50% earlier by >20min), no_targets (planner has no bedtime targets).

---

### D1 — Social Trend (Social, 30 days)
**Text**: Comparing the first 14 days to the last 16 days of the 30-day window: did this person's number of social activities per day increase, decrease, or stay the same?
**Answer space**: `decreased`, `stayed_same`, `increased`
**Source relevance**: planner, daily_self_report, objective_log

**Hint**: Split the 30-day window: first 14 days vs last 16 days. Count activities in each half and compute RATE (count / days). If the rate changed meaningfully, classify as 'increased' or 'decreased'. Small fluctuations should be classified as 'stayed_same'. When sources disagree, consider each source's bias characteristics to judge which is most likely correct.

---

### D2 — Diet vs Baseline (Diet, 30 days)
**Text**: In the past 30 days, did this person's daily home-cooked meal count and total meal count differ from the averages described in their profile by more than 1 meal/day?
**Answer space**: `within_1`, `differs_more_than_1`, `no_baseline`
**Source relevance**: profile_ltm, daily_self_report
**Context**: Compare against profile's stated meal baseline.

**Hint**: Estimate 30-day average daily meals and home-cooked meals from the available behavioral evidence, accounting for self-report bias. Compare to profile baseline. Consider whether actual behavior deviates noticeably from the stated baseline across both dimensions (total meals and home-cooked meals). If the combined deviation is more than minor, classify as 'differs_more_than_1'. Otherwise 'within_1'. If no baseline in profile, 'no_baseline'.

---

### E1 — Late Night Cause (Sleep, 30 days)
**Text**: In the past 30 days, on nights when this person went to bed after midnight, which co-occurring factor appeared on more than 50% of those nights?
**Answer space**: `work_activity`, `social_activity`, `no_single_factor`, `no_late_nights`
**Source relevance**: planner, daily_self_report, objective_log, device_log

**Hint**: Find nights with bedtime after midnight. For each late night, check if the person had OVERTIME (>8.5h work) or social activity that day. Regular work (non-overtime) does NOT count. If >50% of late nights had overtime → 'work_activity'. If >50% had social → 'social_activity'. Otherwise 'no_single_factor'. If no late nights, 'no_late_nights'.

---

### E2 — Exercise Skip Cause (Exercise, 30 days)
**Text**: In the past 30 days, on days when this person had planned to exercise but did not, did they work more than 8.5 hours that same day?
**Answer space**: `no_fewer_than_30`, `between_30_60`, `yes_more_than_60`
**Source relevance**: profile_ltm, planner, daily_self_report, objective_log, device_log
**Context**: Identify planned exercise days from planner, check skip reasons.

**Hint**: Find days when exercise was planned (planner intent) but NOT done (self-report/device show no exercise). On those skip days, check if work hours exceeded 8.5 hours. Classify share of skip-days with heavy work: yes_more_than_60 (>60%), between_30_60 (30-60%), no_fewer_than_30 (<30%).

---

### F1 — Unplanned Social Count (Social, 30 days)
**Text**: In the past 30 days, on how many days did this person attend social activities when the planner showed no social intent?
**Answer space**: `0_to_3`, `4_to_6`, `7_or_more`, `no_social_activities`
**Source relevance**: profile_ltm, planner, daily_self_report, objective_log
**Context**: Compare social activities against planner's social intentions.

**Hint**: Count days with social activity (from self-report/objective records) where the planner showed NO social intent. If no social activities at all, 'no_social_activities'.

---

### F2 — Tracker Missing vs Inactive (Exercise, 30 days)
**Text**: In the past 30 days, on days when the fitness tracker recorded no workout, did other sources indicate that this person exercised?
**Answer space**: `inactive_confirmed`, `both_occurred`, `yes_tracker_missing`
**Source relevance**: profile_ltm, planner, daily_self_report, objective_log, device_log
**Context**: Compare exercise evidence from all sources against device_log coverage.

**Hint**: Focus on days when the device log shows 'Records unavailable' or no workout detected. If self-report or other sources show exercise on those days → tracker missed it. Consider the proportion of days where device_log data is unavailable — this affects how confidently you can assess exercise patterns from tracker data alone. If all sources agree no exercise → 'inactive_confirmed'. Mixed → 'both_occurred'.

---

### F3 — Timesheet Missing vs Off (Work, 30 days)
**Text**: In the past 30 days, on days when the work timesheet had no entry, did other sources indicate that this person worked?
**Answer space**: `truly_off`, `both_occurred`, `yes_worked_despite_no_entry`
**Source relevance**: profile_ltm, planner, daily_self_report, device_log
**Context**: Compare work evidence from all sources against objective_log availability.

**Hint**: Focus on days when the objective log shows 'Records unavailable'. If self-report shows work hours > 0, or device shows >30 min focused work activity on those days → timesheet was just missing. If all sources agree no work → 'truly_off'. Mixed → 'both_occurred'.

---

### G1 — Deliberate vs Incidental Exercise (Exercise, 30 days)
**Text**: In the past 30 days, on days when this person was physically active, was the activity deliberate exercise or incidental movement?
**Answer space**: `incidental_movement_70plus`, `mix`, `deliberate_exercise_70plus`, `no_activity`
**Source relevance**: planner, daily_self_report, device_log

**Hint**: On days with physical activity, distinguish planned workouts from incidental movement. Duration can be a useful cue, but intent language and workout markers are more direct evidence of deliberate exercise. For device_log: 'workout detected' usually indicates deliberate exercise. For planner: planned exercise indicates deliberate intent. Synthesize across sources. If no activity days, 'no_activity'.

---

### G2 — Voluntary vs Obligatory Social (Social, 30 days)
**Text**: In the past 30 days, when this person attended social activities, were those activities voluntary or obligatory?
**Answer space**: `obligatory_70plus`, `mix`, `voluntary_70plus`, `no_meetings`
**Source relevance**: profile_ltm, planner, daily_self_report, objective_log

**Hint**: OBLIGATORY = activity for someone else's benefit (helping/caregiving, work-mandated events, family obligations). VOLUNTARY = personal choice (dinner with friends, book club, coffee catch-up). Classify: voluntary_70plus (≥70% voluntary), obligatory_70plus (≥70% obligatory), mix. If no social activities, 'no_meetings'. Weight self-report for activity descriptions, planner for intent (planned = self-initiated → voluntary).

---

### Ctrl1 — Outside Meals (Diet, 7 days)
**Text**: In the past week, how many days did this person eat food prepared outside the home?
**Answer space**: `0_to_1_days`, `2_to_3_days`, `4_or_more`
**Source relevance**: profile_ltm, planner, daily_self_report

**Hint**: Last 7 days: count days with at least one meal eaten outside the home. When sources disagree, consider each source's bias characteristics to judge which is most likely correct.

---

### Ctrl2 — Short Sleep Nights (Sleep, 7 days)
**Text**: In the past week, how many nights did this person sleep less than 6 hours?
**Answer space**: `0_nights`, `1_to_2`, `3_or_more`
**Source relevance**: planner, daily_self_report, device_log

**Hint**: Last 7 days: count nights with sleep duration < 6 hours. When sources disagree, consider each source's bias characteristics to judge which is most likely correct.


---

# Output Rules — Schema-Aware LLM-Direct

## Output Format

Each task produces ONE JSON file. The file contains a single JSON object:

```json
{
  "persona": "bench_example_000_jordan_kim",
  "answers": {
    "A1": {"answer": "10_to_19", "would_skip": false},
    "A2": {"answer": "4_to_7", "would_skip": false},
    "A3": {"answer": "40_to_69", "would_skip": true},
    "B2": {"answer": "within_1_day", "would_skip": false},
    "B3": {"answer": "does_not_match", "would_skip": false},
    "C2": {"answer": "25_to_50_pct", "would_skip": true},
    "C3": {"answer": "later_more_than_50pct", "would_skip": false},
    "D1": {"answer": "increased", "would_skip": false},
    "D2": {"answer": "within_1", "would_skip": false},
    "E1": {"answer": "no_single_factor", "would_skip": true},
    "E2": {"answer": "between_30_60", "would_skip": false},
    "F1": {"answer": "4_to_6", "would_skip": false},
    "F2": {"answer": "both_occurred", "would_skip": false},
    "F3": {"answer": "yes_worked_despite_no_entry", "would_skip": true},
    "G1": {"answer": "mix", "would_skip": false},
    "G2": {"answer": "mix", "would_skip": false},
    "Ctrl1": {"answer": "2_to_3_days", "would_skip": false},
    "Ctrl2": {"answer": "1_to_2", "would_skip": false}
  }
}
```

## Rules

1. The top-level `persona` field must match the persona directory name.
2. The `answers` object has exactly 18 keys (A1, A2, A3, B2, B3, C2, C3, D1, D2, E1, E2, F1, F2, F3, G1, G2, Ctrl1, Ctrl2).
3. Each value is an object with two fields:
   - `answer`: a valid answer label string from that question's answer space (REQUIRED, never null)
   - `would_skip`: a boolean indicating whether you would prefer to abstain if allowed (REQUIRED)
4. The `answer` field must always contain a valid label — you are forced to answer even if you mark `would_skip: true`.
5. Set `would_skip: true` only when evidence is genuinely too conflicting or insufficient for a confident judgment.
6. Answer labels MUST use exact strings with underscores (e.g., `"20_or_more"`, NOT `"20 or more"`).
7. The JSON must be valid and properly formatted.

## Acceptance Criteria

A task is DONE when:
- The output JSON file exists at the specified path
- It contains valid JSON matching the schema above
- All 18 questions are present with valid `answer` and `would_skip` fields
- No nulls or missing questions
