# AGENTS.md — LLM Few-Shot Baseline (Single Question per Task)

## Goal

Answer ONE survey question about a persona by reading their complete natural-language memory document + 3 worked examples (exemplars) showing how other personas were scored for the same question.

## Read First

1. `specs/questions.md` — all 18 questions with answer spaces (find YOUR assigned question)
2. `specs/output-rules.md` — JSON output format
3. The exemplar file for your assigned question: `exemplars/{QID}.md`
4. The test persona's NL render file (specified in the task prompt)

## Workflow

### Step 1 — Read specs
Read `specs/questions.md` to understand all question definitions and answer spaces. Identify the specific question you are assigned (given in the task prompt).

### Step 2 — Read the exemplar file
Read `exemplars/{QID}.md` for your assigned question. It contains 3 complete worked examples: for each example persona, you get their full memory document and the correct answer.

Study the exemplars carefully:
- Understand HOW the correct answer was determined from the memory content
- Note what information from which source sections was key to the judgment
- Notice patterns across the worked examples; do not rely on persona identifiers or split metadata.

### Step 3 — Read the test persona's NL render
Read the test persona's NL render file. It contains 5 sections:
- `## Long-Term Background and Habits` (profile/background)
- `## Plans and Intentions` (planner)
- `## Daily Self-Reports` (daily journal)
- `## Objective Records` (timesheet/measurements)
- `## Device and Activity Records` (wearable/tracker)

### Step 4 — Answer the question and write output
Apply the reasoning pattern you learned from the exemplars to the test persona:
1. Identify the relevant evidence from the test persona's memory
2. Select exactly one answer label from the answer space (**forced answer** — you MUST pick one)
3. Also decide: if you had the option to abstain (skip), would you? Record as `would_skip` (true/false)
4. Write the result JSON to the output file specified in the task prompt

### Answering Principles

- **Learn from exemplars**: The exemplars show you how to map memory content → answer for this question type. Use the same reasoning pattern for the test persona.
- **Synthesize across sources**: Consider ALL five source sections together. Different sources may tell different stories — resolve conflicts using common sense and your judgment about source reliability.
- **No single source presumed correct**: Every data source has potential biases. Objective records may have gaps, self-reports may be optimistic or pessimistic, profiles may be outdated, plans may not match reality, device data may have dropout.
- **Forced answer required**: You MUST select exactly one answer from the options, even if evidence is ambiguous. Pick the best-supported answer.
- **Skip decision**: Separately, indicate whether you would PREFER to skip if allowed. Skip when evidence is genuinely too conflicting or insufficient.
- **Use exact labels**: Answer must be one of the labels listed in the answer space (e.g., `"20_or_more"`, not `"20 or more"`). Use underscores exactly as shown.
- **Time windows matter**: Each question specifies a time window (7, 14, or 30 days). Only consider data within that window.
- **Cross-midnight times**: When comparing bedtimes near midnight, treat 00:10 as 24:10 (not 0:10). The difference between 23:50 and 00:10 is 20 minutes, not 23 hours.

### FORBIDDEN actions
- Using any scripting, programming, or command-line data processing to analyze the NL render
- Reading raw JSON source files (use the NL render only)
- Modifying any file other than the specified output file
- Answering questions other than the one assigned to this task
- Reading any files other than the specs, your exemplar file, and the test persona's NL render
