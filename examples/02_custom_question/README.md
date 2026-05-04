# Example 02 — Add a custom question type

This example walks through adding a 19th question (`H1`) to the
benchmark. Unlike example 01 (which adds a method) this one adds a
**question**, which means changes propagate through several layers:
question spec → ground-truth rule → bias matrix → re-generation →
re-evaluation.

Estimated wall-clock: ~1 hour (most of which is re-running
`generate_ground_truth` for one seed; the other three pipeline stages
are unchanged).

The full cross-cutting reference is
[../../EXTENDING.md §2](../../EXTENDING.md#2-add-a-new-question-type);
the snippets here are concrete examples you can copy.

## Try it immediately

This directory ships a runnable H1 end-to-end — a spec file, a GT rule,
a 30-day event fixture, and a five-test suite — so you can see the
whole contract exercised before editing anything in `configs/` or
`src/`:

```bash
pytest examples/02_custom_question/test_custom_question.py -v
```

Files:

* [`h1_question.yaml`](h1_question.yaml) — the H1 spec (stand-alone; **not** merged into `configs/questions.yaml`, so the 18-question benchmark is unchanged).
* [`compute_h1.py`](compute_h1.py) — the ground-truth rule, written against the real event schema used by `src/survey2agent/data_generation/ground_truth.py`.
* [`fixture_events.json`](fixture_events.json) — a 30-day toy event table.
* [`test_custom_question.py`](test_custom_question.py) — five tests: schema validity, SKIP collision check, expected-label on the fixture, and two boundary cases.

## Step 1 — Define the question

The shipped [`h1_question.yaml`](h1_question.yaml) is exactly the block
you would append to [`configs/questions.yaml`](../../configs/questions.yaml):

```yaml
H1:
  question_text: "How many distinct types of social activities did the persona engage in during the last 14 days?"
  type: D                        # Temporal / count family
  topic: social
  answer_space:
    - "0_to_1"
    - "2_to_3"
    - "4_or_more"
    - "uncertain"
  answer_space_type: ordinal
  ordered_labels:
    - "0_to_1"
    - "2_to_3"
    - "4_or_more"
  edge_options:
    - "uncertain"
  time_window: 14
```

The shape is enforced by
[`schemas/question_definition.schema.json`](../../schemas/question_definition.schema.json).
`test_h1_schema_valid` in this directory validates the spec against that
schema; an equivalent one-liner is:

```bash
python3 -c "
import json, yaml, jsonschema
spec = yaml.safe_load(open('examples/02_custom_question/h1_question.yaml'))['H1']
schema = json.load(open('schemas/question_definition.schema.json'))
jsonschema.validate(spec, schema)
print('OK')
"
```

## Step 2 — Implement the ground-truth rule

GT rules live in
[`src/survey2agent/data_generation/ground_truth.py`](../../src/survey2agent/data_generation/ground_truth.py).
All rules share one signature; follow the existing pattern (look at
`compute_a1`, `compute_b2`, or `compute_d1` for filtering, persona
lookup, and label-emission patterns). The shipped
[`compute_h1.py`](compute_h1.py) is a verbatim template:

```python
from typing import Any


def compute_h1(
    events: list[dict[str, Any]],
    persona: dict[str, Any],
    sources: dict[str, Any],
) -> dict[str, Any]:
    """H1 | Social — distinct activity-type count over the last 14 days."""
    last_14 = sorted(events, key=lambda d: int(d["day_index"]))[-14:]

    distinct_types = {
        activity
        for d in last_14
        for activity in d["social"].get("activities", [])
    }
    n = len(distinct_types)

    if n <= 1:
        answer = "0_to_1"
    elif n <= 3:
        answer = "2_to_3"
    else:
        answer = "4_or_more"

    return {
        "answer": answer,
        "derivation_detail": f"distinct_activity_types={n} window_days=14",
    }
```

Event-schema canonical accessors used above:

* `d["day_index"]: int` — 0..29, one entry per simulated day.
* `d["social"]["activities"]: list[str]` — activity name strings; see `compute_d1` and `compute_f1` in the main GT file for identical access patterns.
* `persona` and `sources` are available for rules that need them (for instance `compute_f1` consults `sources["planner"]`); H1 ignores both.

Then register the rule in the `QUESTION_REGISTRY` dict at the bottom
of the same file:

```python
QUESTION_REGISTRY: dict[str, dict[str, Any]] = {
    ...,
    "H1": {"compute": compute_h1, "type": "D", "topic": "social"},
}
```

The dispatcher `compute_all_ground_truths` iterates this dict and
picks up new entries automatically.

## Step 3 — Update the bias matrix

If H1 introduces a new (source × topic) combination not yet covered by
QBD-2 (the per-topic bias direction table), extend `bias_defaults` in
`configs/questions.yaml`. For H1 (topic=social), the existing
`social: -1` entry under `daily_self_report` is sufficient and no
change is needed.

## Step 4 — Re-generate one seed

```bash
SEED=20260321
DATASET_DIR=data/benchmark/seeds/s${SEED}

python3 -m survey2agent.data_generation.generate_ground_truth \
    --dataset-dir ${DATASET_DIR}
```

Only `generate_ground_truth` needs to re-run — personas, events, and
sources are unchanged.

## Step 5 — Re-evaluate

```bash
make smoke           # confirms the new GT field doesn't crash any method
make reproduce-main  # only if you want updated paper tables
```

All 13 methods will now emit a prediction for H1 because they iterate
over `QUESTIONS.keys()` rather than hard-coding the 18-question list.

## Step 6 — Implement the per-source direct readout `μ*(s, H1)`

`μ*(s, q)` is the deterministic answer that source `s` would give to
`q` if it had perfect access to its own raw fields. The Source
Reachability reference rows and the source-reachability diagnostic
table in
[`paper_artifacts/appendix/source_ceiling_complement_table.py`](../../paper_artifacts/appendix/source_ceiling_complement_table.py)
read this output. **Skipping this step does not break answer-only
methods, but Source Reachability / Struct-LLM / source-reachability diagnostic rows
will silently report `None` for H1.**

In production the readout is added as `_mu_h1` in
[`src/survey2agent/extraction/_mu_shell.py`](../../src/survey2agent/extraction/_mu_shell.py)
and registered in the `compute_all_mu` dispatcher at the bottom of the
same file. The shipped [`compute_mu_h1.py`](compute_mu_h1.py) is a
standalone copy of that function, callable without merging the spec.
The five tests under "μ*(s, H1)" in
[`test_custom_question.py`](test_custom_question.py) check:

| # | Property |
|---|---|
| 1 | Output is a `dict` with exactly the five canonical stream keys |
| 2 | Every value is in `answer_space ∪ {None}` |
| 3 | Streams with no H1 signal (`objective_log`, `device_log`) emit `None` |
| 4 | Bias-direction sanity: `planner` (`b=+1`) ≥ `daily_self_report` (`b=-1`) |
| 5 | `profile_ltm` falls back to `None` when `stated_profile.social` missing |

## Step 7 — Verification checklist

The shipped tests cover checks 1 and 2 of the §2.6 verification
checklist in `EXTENDING.md` (GT label, μ\* readout shape). The remaining
four checks require merging H1 into the main spec and running the
standard producer chain:

| # | Check | Demonstrated here? |
|---|---|---|
| 1 | GT label produced for every persona | ✅ `test_compute_h1_*` |
| 2 | μ\* readout returns expected fields | ✅ `test_mu_h1_*` |
| 3 | NL render and extraction run (or stub) | requires merging H1 into `configs/questions.yaml` |
| 4 | At least one fusion method evaluates the new question | same |
| 5 | At least one LLM-path method evaluates the new question | same |
| 6 | Output artifacts include per-instance GT / atoms / predictions / aggregate | same |

For 3–6, follow the per-difficulty tiny-seed loop documented in
[`EXTENDING.md` §2.5](../../EXTENDING.md#25--re-generate-the-benchmark-for-one-seed)
and §2.6.

## Common mistakes

- Forgetting `ordered_labels` for an ordinal question. The schema
  validator catches this at PR time.
- Including `"SKIP"` in `answer_space`. The import-time guard in
  [`methods/base.py`](../../src/survey2agent/methods/base.py) will
  raise `RuntimeError` and refuse to load.
- Hard-coding the question id in any method. Methods must work
  uniformly across all questions.
- Assuming a bespoke event schema. Always check the canonical
  accessors in existing `compute_*` functions before reaching for a
  new field name.

