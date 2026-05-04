# Extending the Benchmark

This document is a step-by-step guide for the five most common extensions:

1. [Add a new method](#1-add-a-new-method) to the comparison matrix.
2. [Add a new question type](#2-add-a-new-question-type) to the benchmark.
3. [Add a new persona generator seed](#3-add-a-new-persona-generator-seed).
4. [Add a new evidence stream](#4-add-a-new-evidence-stream).
5. [Plug in a new LLM backend](#5-plug-in-a-new-llm-backend).

All commands assume you are at the repository root and have already
run `make install` (see [README.md](README.md)). For governance and code
style, see [CONTRIBUTING.md](CONTRIBUTING.md).

> **Reuse point.** This testbed is designed to be forked and extended.
> The five extension paths above are the same touchpoints we used
> internally when shaping the current benchmark; each one is wired into the
> reproduction harness so that — once your extension lands — the same
> 23-table reproduction pipeline runs against your modified setup. If you
> are adding a new question or stream as part of a follow-up evaluation,
> §2 and §4 are the canonical entry points and each ends with a
> [verification checklist](#26--verification-checklist) that mirrors the
> reproducibility contract used for the published numbers.

## Information boundary for new methods

Persona IDs are bookkeeping keys only. They may appear in filenames,
frozen-output bundles, cache custom IDs, and reporting metadata, but they
must not be used as method features or inserted into model prompts for the
target persona. In particular, do not parse strings such as `bench_stable`,
`bench_shift`, `bench_stated`, seed names, split names, or directory paths
to infer difficulty.

Your method should consume only the artifact type it is assigned: an atom
table for T1/T2, a rendered NL memory document for T3, or a structured atom
grid for matched-input diagnostics. If your extension needs difficulty for
training or stratified reporting, keep that label on the train/cal/reporting
side of the pipeline and verify it is not available inside `predict_one` or
inside any LLM prompt.

---

## 1. Add a New Method

The contract is one abstract base class plus three optional flags. The
runner does the rest (training, calibration, batched inference,
selective-layer wiring, paper-table integration).

### 1.1  Subclass `Method`

```python
# src/survey2agent/methods/your_method.py
from __future__ import annotations

from survey2agent.extraction.atoms import ExtractedAtom
from survey2agent.extraction.question_spec import QUESTIONS

from .base import Method, Prediction, SKIP_SENTINEL


class YourMethod(Method):
    """One-line description that becomes the paper-facing identifier."""

    name = "YourMethod"          # paper alias; must be unique across methods
    requires_fit = False         # set True if you implement `fit`
    requires_calibration = False # set True if you implement `calibrate`

    def predict_one(self, atom: ExtractedAtom, qid: str) -> Prediction:
        # Your logic here. `atom` is the per-persona ExtractedAtom for one
        # source set; `qid` is one of the 18 question IDs in QUESTIONS.
        chosen = QUESTIONS[qid]["answer_space"][0]
        return Prediction(answer=chosen, would_skip=False)
```

### 1.2  (Optional) Implement `fit` / `calibrate`

```python
def fit(self, records):  # records: Sequence[(ExtractedAtom, GroundTruth)]
    # Estimate parameters from the train split (216 personas).
    ...

def calibrate(self, records):  # records: Sequence[(ExtractedAtom, GroundTruth)]
    # Tune thresholds on the cal split (96 personas).
    ...
```

The runner calls `fit` once on the train split and `calibrate` once on
the cal split before evaluating on the test split (120 personas). If you
do not need either step, leave the flags `False` and the methods will be
no-ops.

### 1.3  Selective variant (optional)

If you also want a `+SKIP` variant, subclass your method and override
`predict_one` to emit `Prediction(answer=SKIP_SENTINEL, would_skip=True)`
when your selective rule fires. Pattern used by every existing method
that has a `+SKIP` companion (e.g. `SSBSelective`, `NBFSelective`).

### 1.4  Register the method

Edit [`src/survey2agent/methods/__init__.py`](src/survey2agent/methods/__init__.py)
and add your class to the imports + `__all__` list:

```python
from .your_method import YourMethod
# ...
__all__ = [
    # ... existing entries ...
    "YourMethod",
]
```

There is no central `REGISTRY` dict; downstream consumers (paper
artifacts, tests, leaderboard verifier) instantiate methods directly by
class. Registering in `__all__` is required when promoting the method
into the comparison matrix; for purely local experimentation you can
skip this step and instantiate your subclass directly from your own
script (see [`examples/03_programmatic_api/`](examples/03_programmatic_api/)).

### 1.5  Conformance test

Add `tests/methods/test_your_method.py`:

```python
from survey2agent.extraction.atoms import ExtractedAtom
from survey2agent.extraction.question_spec import QUESTIONS
from survey2agent.methods.base import Prediction
from survey2agent.methods import YourMethod


def test_returns_valid_prediction():
    m = YourMethod()
    atom = ExtractedAtom(
        persona="demo",
        extraction={qid: {} for qid in QUESTIONS},
    )
    for qid, q in QUESTIONS.items():
        pred = m.predict_one(atom, qid)
        assert isinstance(pred, Prediction)
        assert pred.answer in q["answer_space"] or pred.would_skip
```

Run only your test: `pytest tests/methods/test_your_method.py -v`.

### 1.6  Reproduce a paper-style table that includes your method

Pick the closest existing producer in
[`paper_artifacts/main/`](paper_artifacts/main/) or
[`paper_artifacts/appendix/`](paper_artifacts/appendix/) (e.g.
`tab_main_overall.py`), copy it, swap in `YourMethod()` next to the
existing methods, and call `write_outputs(...)` with a new `table_id`.

A minimal example end-to-end is shipped at
[`examples/01_minimal_method/`](examples/01_minimal_method/).

### 1.7 Reproduce a paper-locked appendix table

Every script under `paper_artifacts/appendix/` follows a fixed
paper-locked contract: hardcode the published cells as `PAPER_TAB_X` at
the top of the module, recompute them from frozen artifacts in
`main()`, and let `_common.emit_row(...)` tag every CSV row with
`paper_match = OK|FAIL` against `PAPER_TOLERANCE = 0.005`.

Full worked example:
[`paper_artifacts/appendix/source_ceiling_complement_table.py`](paper_artifacts/appendix/source_ceiling_complement_table.py)
— 19 (method × input) rows × 3 metrics = 57 paper-locked cells spanning
T0–T3, reusing `run_mixed_mode_across_seeds`,
`run_oracle_mode_across_seeds`, `run_struct_llm_across_seeds`, and
`run_llm_across_seeds` from `_common`.

Checklist when adding a new table (all five must land together):

1. **Script** `paper_artifacts/appendix/<name>.py`
   * `PAPER_TAB_<ID>: dict[key, float]` at module top, sourced from the
     paper source for the corresponding table.
   * `main()` computes each cell and passes
     `paper_point=PAPER_TAB_<ID>[key]` to `emit_row(...)`.
   * Write via `write_outputs(table_id, rows, md_table=...,
     subdir="appendix")`.
   * `main()` returns `1` if any `paper_match` starts with `"FAIL"`,
     else `0`.
2. **Registry** — add one entry to `_REGISTRY["appendix"]` in
   [`paper_artifacts/reproduce_paper.py`](paper_artifacts/reproduce_paper.py).
3. **Smoke test** — add `test_<name>_imports()` in
   [`tests/integration/test_paper_artifacts_smoke.py`](tests/integration/test_paper_artifacts_smoke.py);
   assert the module imports and `PAPER_TAB_<ID>` has the expected
   shape.
4. **Paper-lock slow test** — add `<name>` to `_APPENDIX_SCRIPTS` and a
   `test_<name>_paper_lock()` in
   [`tests/integration/test_paper_appendix_tier_b_reproduction.py`](tests/integration/test_paper_appendix_tier_b_reproduction.py);
   marked `pytest.mark.slow`, asserting `fail == 0`.
5. **Docs** — update the scripts tables in both
   [`paper_artifacts/README.md`](paper_artifacts/README.md) and
   [`README.md`](README.md), and add a manifest entry in
   [`paper_artifacts/MANIFEST.md`](paper_artifacts/MANIFEST.md)
   (caption / source / grid / reduction / tolerance).

Verify:

```bash
PYTHONPATH=src python3 -m paper_artifacts.reproduce_paper --names <name>
```

should print `<N>/<N> OK, 0 FAIL`.

---

## 2. Add a New Question Type

The benchmark currently has 18 questions × 8 reasoning families
(A-Arbitration, B-Identity, C-Plan-Reality, D-Trend, E-Factor,
F-Missing-Data, G-Annotation, Ctrl-Control). The single source of truth
is [`configs/questions.yaml`](configs/questions.yaml), mirrored at load
time by `survey2agent.extraction.question_spec`.

### 2.1  Add the question definition

Append a new entry to `configs/questions.yaml`:

```yaml
H1:
  question_text: "Your natural-language prompt for the extractor."
  type: H                       # new family letter
  topic: sleep                  # one of the existing topics
  answer_space:                 # full label list including any edge cases
    - "0_to_3"
    - "4_or_more"
    - "uncertain"
  answer_space_type: ordinal    # or "nominal"
  ordered_labels:               # BCF/ABF bias-shift rank order; required iff ordinal
    - "0_to_3"
    - "4_or_more"
  edge_options: ["uncertain"]   # treated as factual edge cases, not abstain
  time_window: 14
```

The loader auto-derives `ordinal_encoding = {label: i+1, ...}` from
`ordered_labels` and validates that no label collides with the reserved
`SKIP` sentinel. This order is the rank axis used by BCF/ABF bias shifts;
it may differ from the human-readable display order for derived-count
questions, so document the intended shift direction when adding a new
ordinal template.

### 2.2  Add the ground-truth derivation rule

All GT rules live in
[`src/survey2agent/data_generation/ground_truth.py`](src/survey2agent/data_generation/ground_truth.py)
and share one signature:

```python
def compute_h1(
    events: list[dict[str, Any]],
    persona: dict[str, Any],
    sources: dict[str, Any],
) -> dict[str, Any]:
    """Return {'answer': <label_in_answer_space>, ...optional debug fields}."""
    ...
```

`events` is the hidden-state event table (a list of dicts); `persona`
is the persona JSON; `sources` is the per-stream raw output. Look at
`compute_a1`, `compute_b2`, or `compute_d1` in the same file for
concrete patterns of how to filter events, read the persona profile,
and return a label. A complete, runnable example — spec file, GT rule
written against the real event schema, 30-day fixture, and test suite
— ships at
[`examples/02_custom_question/`](examples/02_custom_question/).

Then register the new rule in the `QUESTION_REGISTRY` dict at the
bottom of the same file:

```python
QUESTION_REGISTRY: dict[str, dict[str, Any]] = {
    ...
    "H1": {"compute": compute_h1, "type": "H", "topic": "sleep"},
}
```

The dispatcher `compute_all_ground_truths` iterates this dict and calls
each `compute_*` once per persona; new entries are picked up
automatically.

### 2.3  Implement the per-source direct readout `μ*(s, q)`

`μ*(s, q)` is the **deterministic answer that source `s` would give to
question `q` if you had perfect access to its raw fields**. It is what
the Source Reachability reference rows consume, and what the
source-reachability diagnostic table in
[`paper_artifacts/appendix/source_ceiling_complement_table.py`](paper_artifacts/appendix/source_ceiling_complement_table.py)
compares every learned method against. **Without a `μ*` readout, your
new question still works for answer-only methods that consume LLM
atoms, but Source Reachability / Struct-LLM / source-reachability rows will silently
report `None` for it.**

The readout shell lives in
[`src/survey2agent/extraction/_mu_shell.py`](src/survey2agent/extraction/_mu_shell.py).
Each of the existing 18 questions has one private function
`_mu_<qid>(sources) -> Mu` where `Mu = dict[str, str | None]` —
one entry per evidence stream, mapping to a label in `answer_space`
or `None` (the ⊥ "this stream cannot answer" case). Add yours next to
the existing ones:

```python
# src/survey2agent/extraction/_mu_shell.py

def _mu_h1(sources: dict) -> Mu:
    """H1 | Social — distinct activity-type count over the last 14 days."""
    mu = _empty_mu()

    sr = sources.get("daily_self_report", [])
    if sr:
        last_14 = sorted(sr, key=lambda r: int(r["day_index"]))[-14:]
        types = {a for r in last_14 for a in _safe(r, "social", "activities", default=[])}
        mu["daily_self_report"] = _classify_activity_count(len(types))

    # planner: optimistic — count the number of *planned* distinct activities
    pl = sources.get("planner", [])
    if pl:
        last_14 = sorted(pl, key=lambda r: int(r["day_index"]))[-14:]
        types = {a for r in last_14 for a in _safe(r, "social_plan", "activities", default=[])}
        mu["planner"] = _classify_activity_count(len(types))

    # profile_ltm: long-term baseline — read stated_profile.social.types_per_fortnight
    base = _safe(sources.get("profile_ltm", {}),
                 "stated_profile", "social", "types_per_fortnight")
    if base is not None:
        mu["profile_ltm"] = _classify_activity_count(int(base))

    # objective_log / device_log: ⊥ if this stream has no signal for the question
    return mu
```

Then register `_mu_h1` in the dispatcher `compute_all_mu` at the bottom
of `_mu_shell.py` so `oracle_extractor.build_oracle_atom(...)` picks it
up. Refer to `_mu_a1` (Sleep arbitration) and `_mu_d1` (trend) as
reference patterns for ordinal and trend questions respectively.

> **Bias direction sanity check.** The sign of the per-source error
> `μ*(s, q) − GT(q)` should match the bias direction `b` you declare
> for `(stream, topic)` in the bias matrix (§2.4). If `planner` is
> supposed to be optimistic (`b = +1`) but your readout returns labels
> lower than ground truth, your readout is misaligned with the
> distortion model and ABF / bias-aware fusion will mislearn `δ`.

### 2.4  (If your family is new) update bias direction matrix

If your topic is not already covered, add a row to the bias-direction
matrix in [`configs/questions.yaml`](configs/questions.yaml) (the
`bias_defaults` block) to declare the per-source bias direction
`b ∈ {-1, 0, +1}` for every evidence stream when the source is queried
about your new topic.

### 2.5  Re-generate the benchmark for one seed

The four pipeline stages take different arguments: stage L1 takes a numeric
`--seed` and writes to `--output-dir`; stages L2-L4 take the `--dataset-dir`
produced by the previous stage. The L5 NL renderer is a library, not a CLI;
the extractor invokes it inline.

```bash
SEED=20260321
DATASET_DIR=data/benchmark/seeds/s${SEED}

python3 -m survey2agent.data_generation.generate_personas \
    --seed ${SEED} --output-dir ${DATASET_DIR}
python3 -m survey2agent.data_generation.generate_events       --dataset-dir ${DATASET_DIR}
python3 -m survey2agent.data_generation.generate_sources      --dataset-dir ${DATASET_DIR}
python3 -m survey2agent.data_generation.generate_ground_truth --dataset-dir ${DATASET_DIR}
```

This rewrites `data/benchmark/seeds/s20260321/`. NL renders are produced
on-the-fly by `survey2agent.extraction.extractor.render_source` /
`render_full_memory` whenever an extraction or LLM-direct producer runs.

> **Tiny seed for fast iteration.** `generate_personas` accepts
> `--per-difficulty N` (default `160`, three classes → `480` personas).
> For an end-to-end generation smoke, use `--per-difficulty 10`
> (30 personas total). Very small values such as `1` or `2` are useful
> for local unit tests but can fail the diversity audit thresholds that
> protect the released benchmark distribution.
> All four pipeline stages are deterministic for a given
> `(seed, --per-difficulty)` pair (the canonical paper artifacts use
> `--per-difficulty 160`).

Smoke test:

```bash
make smoke
```

The conformance test will check that every persona has a ground-truth
entry for every question id including your new one.

### 2.6  Verification checklist

Before merging your new question into the matrix, the following six
checks should all pass on a small seed (`--per-difficulty 10` is the
recommended end-to-end smoke size):

| # | Check | How |
|---|---|---|
| 1 | GT label produced for every persona | grep your `qid` in `data/benchmark/seeds/<seed>/<persona_id>/ground_truth.json` |
| 2 | μ\* readout returns expected fields | `from survey2agent.extraction.oracle_extractor import build_oracle_atom` then `build_oracle_atom(persona_dir).extraction[qid]` is a dict with one key per stream, each in `answer_space ∪ {None}` |
| 3 | NL render and extraction run (or stub) | call `survey2agent.data_generation.nl_render.nl_memory_renderer.render_full_memory(sources)` and confirm a non-empty document; for the LLM extraction path use `python3 -m survey2agent run-extraction --provider openai --model gpt-5.4 --seed <seed>` (omit `--allow-api-call` to stay cache-only) |
| 4 | At least one fusion method evaluates the new question | run `MajorityVote`, `NBF`, or `DSNBF` over the oracle atoms; assert `predictions[qid]` is in `answer_space ∪ {SKIP}` |
| 5 | At least one LLM-path method evaluates the new question | run `LLMDirect` in cache-only mode (default — `--allow-api-call` is the explicit opt-in flag) with a stubbed cache entry; assert the prediction parses |
| 6 | Output artifacts include per-instance GT / atoms / predictions / aggregate | `paper_artifacts/<your-table>.py` reproduces an aggregate row that includes `qid`; `data/method_outputs/.../<seed>/.../predictions.jsonl` contains the row |

A complete, runnable demonstration of checks 1–2 ships at
[`examples/02_custom_question/`](examples/02_custom_question/); checks
3–6 require merging the new question into the main spec and running
the standard producer chain.

### 2.7  Where the new question shows up

Once §2.1 – §2.5 are in place, your new `qid` is **automatically** picked
up by the following surfaces (no further code change required):

| Surface | Pickup mechanism |
|---|---|
| Answer-only paper-locked pipeline | `paper_artifacts/main/forced_accuracy_main.py` iterates `QUESTIONS` |
| Selective-QA full table | `paper_artifacts/main/selective_qa_full.py` iterates `QUESTIONS` |
| Oracle / Struct-LLM rows in the source-reachability diagnostic table | `paper_artifacts/appendix/source_ceiling_complement_table.py` reads `compute_all_mu` output |
| Ground-truth conformance test | `tests/data_generation/test_smoke_byte_equivalence.py` checks every `qid` has a GT entry |
| Bootstrap CI / multi-seed aggregator | `evaluation/bootstrap.py` and `evaluation/multi_seed.py` are `qid`-agnostic |
| Every shipped method's `predict_one(atom, qid)` | methods read `QUESTIONS[qid]["answer_space"]`; no method hard-codes the qid list |

The following surfaces require a **manual touch**:

**Always required (LLM extraction pipeline)** — the LLM extraction
producer enumerates question ids and per-question source maps
explicitly rather than deriving them from `QUESTIONS`. To make your
new `qid` appear in LLM-extracted atoms (the `μ̂` path that T1, T2, and
LLM Schema-Aware methods consume), update **all three**:

* [`src/survey2agent/extraction/atoms.py`](src/survey2agent/extraction/atoms.py)
  — append your new `qid` to `EXPECTED_QUESTION_IDS: tuple[str, ...]`.
  This tuple is consumed by `_freeze_extraction` and the empty-atom
  factory in `extractor.py`.
* [`src/survey2agent/extraction/extractor.py`](src/survey2agent/extraction/extractor.py)
  — add a row for your `qid` to `SOURCE_QUESTION_MAP: dict[str, list[str]]`
  listing the evidence streams the LLM should extract from for this
  question.
* [`src/survey2agent/extraction/_pydantic_schemas.py`](src/survey2agent/extraction/_pydantic_schemas.py)
  — extend the per-question Pydantic schema so the LLM's structured
  output for this `qid` parses.

Without these three edits, the LLM extraction producer will skip your
new question and downstream T1/T2/LLM rows will see `None` atoms for
it — fusion methods will degrade to `SKIP` on every persona. The
deterministic Oracle path (`_mu_shell.compute_all_mu`) is unaffected
because §2.3 already wires it.

**Required only if your question introduces a new family letter**
(e.g. type "H"):

* [`paper_artifacts/main/per_type_accuracy.py`](paper_artifacts/main/per_type_accuracy.py) and
  [`paper_artifacts/appendix/t2_fusion_per_type_per_difficulty.py`](paper_artifacts/appendix/t2_fusion_per_type_per_difficulty.py)
  — both group by reasoning family, so a new family adds a new row /
  column. Update each `_TYPES` tuple plus the corresponding
  `PAPER_TAB_<X>` dict.
* [`README.md`](README.md) and [`ARCHITECTURE.md`](ARCHITECTURE.md)
  enumerations of the eight reasoning families.
* Any method that hard-codes per-family logic (currently zero — every
  shipped method is `qid`-agnostic and reads `answer_space` from
  `QUESTIONS[qid]`).

---

## 3. Add a New Persona Generator Seed

### 3.1  Pick a seed

Avoid `20260321..20260324` (reserved). Use any 8-digit integer; the
convention in this repo is `YYYYMMDD`.

### 3.2  Generate

```bash
SEED=20260401
DATASET_DIR=data/benchmark/seeds/s${SEED}

python3 -m survey2agent.data_generation.generate_personas \
    --seed ${SEED} --output-dir ${DATASET_DIR}
python3 -m survey2agent.data_generation.generate_events       --dataset-dir ${DATASET_DIR}
python3 -m survey2agent.data_generation.generate_sources      --dataset-dir ${DATASET_DIR}
python3 -m survey2agent.data_generation.generate_ground_truth --dataset-dir ${DATASET_DIR}
```

Output lands at `data/benchmark/seeds/s${SEED}/`. NL renders are
generated on-demand by the extractor; you do not need a separate render
step.

### 3.3  Register the seed

Add the entry to [`configs/seeds.yaml`](configs/seeds.yaml):

```yaml
seeds:
  # ... existing entries ...
  - id: 5
    role: extension          # or "dev" / "eval"
    numeric: 20260401
    persona_count: 480
```

Re-run any 4-seed aggregating producers (`reproduce_paper.py` will pick
up the new seed automatically through `configs/seeds.yaml`).

---

## 4. Add a New Evidence Stream

A stream is a per-persona projection from the hidden event table into a
noisy view ("source") that the reader (extractor or LLM) can see. The
benchmark currently has five: `profile_ltm`, `planner`,
`daily_self_report`, `objective_log`, `device_log`.

### 4.1  Define the projector

Add a new projector in
[`src/survey2agent/data_generation/source_projector.py`](src/survey2agent/data_generation/source_projector.py)
that maps an event table row to a (possibly distorted, possibly
dropped) view. The projector should write `<your_stream>.json` into
each persona's `structural_sources/` directory at generation time so
the rest of the pipeline can find it.

> **Source loader is hard-coded.** The deterministic Oracle path reads
> source JSONs through
> [`src/survey2agent/extraction/_source_loader.py::load_sources`](src/survey2agent/extraction/_source_loader.py),
> which **explicitly opens the five existing filenames** and returns a
> dict keyed by stream name. To make `μ*(your_stream, q)` actually read
> your new file, append a `read_text` call for `<your_stream>.json` and
> add the corresponding key to the returned dict. Without this edit
> the projector will write the JSON but the oracle readout will see an
> empty stream.

### 4.2  Declare bias direction `b` and dropout

Edit the QBD-2 block in `configs/questions.yaml` to add a column for
your stream with the per-topic bias direction. If your stream has field
dropout (like `device_log` at 50%), wire it into your projector.

### 4.3  Update the atom dataclass

Atom shapes are defined as Python dataclasses (not JSON Schema) in
[`src/survey2agent/extraction/atoms.py`](src/survey2agent/extraction/atoms.py).
Add a new field for your stream's predicates, and update any methods
that enumerate fields explicitly (most fusion methods iterate over
`SOURCE_NAMES` from `question_spec` so they pick up new streams
automatically).

### 4.4  Update `source_names`

```yaml
# configs/questions.yaml
source_names:
  - profile_ltm
  - planner
  - daily_self_report
  - objective_log
  - device_log
  - your_new_stream    # <-- add
```

### 4.5  Add the NL memory render template

Every stream contributes one section to the NL memory document that
extractors and LLM-Direct methods consume. Templates live in
[`src/survey2agent/data_generation/nl_render/nl_memory_renderer.py`](src/survey2agent/data_generation/nl_render/nl_memory_renderer.py).
The existing five streams use these renderer functions:
`render_profile` (profile_ltm), `render_planner` (planner),
`render_self_report` (daily_self_report), `render_objective_log`
(objective_log), `render_device_log` (device_log). Add a new
`render_<your_stream>(data)` next to them:

```python
# nl_memory_renderer.py

def render_your_stream(data: dict | list) -> str:
    """Render <your_stream>.json to NL paragraph."""
    lines = ["## <Section Heading For Your Stream>", ""]
    for record in data:
        date = record.get("date", "unknown date")
        lines.append(_date_heading(date))
        # Translate raw schema fields to natural-language phrases.
        # IMPORTANT: do NOT leak schema tokens (raw key names) into the
        # rendered text — extractors should infer fields from prose, not
        # from JSON-shaped strings.
        lines.append(f"On this day, the stream reported ...")
        lines.append("")
    return "\n".join(lines)
```

Then wire `render_your_stream` into `render_full_memory(...)` at the
bottom of the same module so it is invoked in the canonical section
order. The `render_full_memory` ordering is observable to the
extractor — keep your stream's section adjacent to the most similar
existing one (objective ↔ device for sensor data, planner ↔ self-report
for self-disclosed data).

### 4.6  Implement per-question direct readout `μ*(s, q)` for every supported question

For each question your new stream `s` should be able to answer, extend
the corresponding `_mu_<qid>(sources)` function in
[`src/survey2agent/extraction/_mu_shell.py`](src/survey2agent/extraction/_mu_shell.py)
to populate `mu["your_stream"]`. Streams that have **no signal** for a
given question should leave the entry as `None` (the ⊥ "this stream
cannot answer" case is first-class).

```python
# Inside _mu_a1, after the existing four blocks:

ys = sources.get("your_stream", [])
if ys:
    good = sum(1 for r in ys if float(_safe(r, "sleep_proxy_h", default=0)) >= 7.0)
    mu["your_stream"] = _classify_good_nights(good)
```

> **Distortion–readout consistency.** The bias direction `b` you
> declared in §4.2 is what ABF and the bias-aware fusion methods
> attempt to learn `δ` against. Your `μ*` readout for this stream
> should statistically lean in the same direction relative to `μ*` of
> `objective_log` (the `b=0` reference). Add a one-off pytest assertion
> that aggregates `sign(μ*(your_stream, q) − μ*(objective_log, q))`
> across personas and matches your declared `b ∈ {-1, 0, +1}`.

### 4.7  Update extraction prompt and atom schema (LLM path)

The LLM extraction prompt is parameterised over `source_names` from
`configs/questions.yaml`, so once §4.4 lands the prompt automatically
mentions your new stream. **However**, the canonical stream list
`EXPECTED_SOURCES` is enumerated explicitly in
[`src/survey2agent/extraction/atoms.py`](src/survey2agent/extraction/atoms.py)
and in the Pydantic schemas under
[`src/survey2agent/extraction/_pydantic_schemas.py`](src/survey2agent/extraction/_pydantic_schemas.py);
extend both with your new stream name and any per-question per-source
predicate slots that the LLM extractor should populate.

### 4.8  Re-generate one seed and run smoke

All 13 fusion / single-source / LLM methods consume the new stream
automatically — no method-level change is required, because they read
from the atom regardless of stream count.

```bash
python3 -m survey2agent.data_generation.generate_personas \
    --seed 20260401 --output-dir data/benchmark/seeds/s20260401 --per-difficulty 10
python3 -m survey2agent.data_generation.generate_events       --dataset-dir data/benchmark/seeds/s20260401
python3 -m survey2agent.data_generation.generate_sources      --dataset-dir data/benchmark/seeds/s20260401
python3 -m survey2agent.data_generation.generate_ground_truth --dataset-dir data/benchmark/seeds/s20260401
make smoke
```

### 4.9  Verification checklist

Before merging your new stream into the matrix, the following six
checks should all pass on a small seed (`--per-difficulty 10` is the
recommended end-to-end smoke size):

| # | Check | How |
|---|---|---|
| 1 | Source JSON / NDJSON written for every persona | `ls data/benchmark/seeds/<seed>/<persona_id>/<your_stream>.{json,ndjson}` |
| 2 | NL render section appears in the memory document | grep your section heading in the rendered NL memory |
| 3 | μ\* readout returns expected fields for every supported question | for each `qid`, `oracle_extractor.build_oracle_atom(persona_dir).extraction[qid]["your_stream"]` is in `answer_space ∪ {None}` |
| 4 | Bias direction matches declared `b` | sign of `μ*(your_stream, q) − μ*(objective_log, q)` aggregated across personas matches the `b ∈ {-1, 0, +1}` you declared |
| 5 | At least one fusion method consumes the new stream | run `NBF` or `DSNBF` over the oracle atoms; assert your stream appears in the per-source diagnostic dump |
| 6 | At least one LLM-path method handles the new stream | run `LLMDirect` in cache-only mode with a stubbed cache entry; verify the rendered prompt mentions your stream |

---

## 5. Plug in a New LLM Backend

The repo currently supports providers `openai`, `google`, `openrouter`.

### 5.1  Add the model

Append to [`configs/models.yaml`](configs/models.yaml):

```yaml
your-model-2026-04:
  paper_alias: "Your Model 2026.04"
  provider: openrouter           # or one of the supported providers
  api_model_id: "your-model-2026-04"
  api_endpoint: "https://openrouter.ai/api/v1/chat/completions"
  default_params:
    temperature: 0.0
  used_for:
    - llm_direct
    - schema_aware
```

If your provider is one of the three supported ones, you are done — the
existing client in `src/survey2agent/api_clients/` will pick it up.

### 5.2  (New provider) implement a client

Add a new module under `src/survey2agent/api_clients/` that implements
the same minimal interface as
[`src/survey2agent/api_clients/openai_batch.py`](src/survey2agent/api_clients/openai_batch.py)
and register it in
[`src/survey2agent/api_clients/__init__.py`](src/survey2agent/api_clients/__init__.py).

Cache keys are derived from `(provider, api_model_id, prompt SHA-256)`,
so adding a new model alias does not invalidate any existing cache.

### 5.3  Run the producer in cache-only mode first

```bash
python3 -m survey2agent run-llm-direct \
    --provider openrouter --model your-model-2026-04 \
    --seed s20260321 --personas all \
    --output-dir data/method_outputs/your-model-2026-04/s20260321/direct
```

Without `--allow-api-call`, the producer reads only from cache and
writes a stub when no cache entry is found — useful for verifying your
config end-to-end before incurring API spend.

When you are ready, add `--allow-api-call`. The runner will write
outputs to `data/method_outputs/<model>/<seed>/<variant>/` with the same
schema as the existing models.

---

## See also

* [README.md](README.md): one-command reproduction
* [REPRODUCIBILITY.md](REPRODUCIBILITY.md): ML reproducibility checklist
* [CACHE_POLICY.md](CACHE_POLICY.md): cache discipline
* [DATASHEET.md](DATASHEET.md): dataset documentation
* [ARCHITECTURE.md](ARCHITECTURE.md): system architecture
* [SUBMISSION_PROTOCOL.md](SUBMISSION_PROTOCOL.md): how to submit a new
  method's predictions to the comparison matrix
