# Example 01 — Minimal method (`AlwaysFirst`)

A 5-step walkthrough showing how to plug a brand-new method into the
benchmark in roughly 30 minutes. The method itself
([`always_first.py`](always_first.py)) is intentionally trivial — it
always picks the first label in each question's answer space — so the
focus stays on the integration contract rather than the algorithm.

If you want to add a serious method, follow the same 5 steps but
substitute your own logic in step 2. See
[../../EXTENDING.md §1](../../EXTENDING.md#1-add-a-new-method) for the
full reference.

## Prerequisites

```bash
make install      # installs the editable package + dev extras
make smoke        # confirm the baseline pipeline is green
```

## Step 1 — Read the contract

Open [`src/survey2agent/methods/base.py`](../../src/survey2agent/methods/base.py).
Three things to internalize:

1. `Method` is an `ABC`. The only mandatory override is
   `predict_one(atom, qid) -> Prediction`.
2. `Prediction(answer, would_skip, raw_answer=None)` enforces
   `(answer == "SKIP") == would_skip` in `__post_init__`. You cannot
   construct an inconsistent prediction.
3. Class attributes `name`, `requires_fit`, `requires_calibration` tell
   the runner whether to call `fit` / `calibrate` before evaluation.

## Step 2 — Write the method

[`always_first.py`](always_first.py) is the entire implementation:

```python
class AlwaysFirst(Method):
    name = "AlwaysFirst"
    requires_fit = False
    requires_calibration = False

    def predict_one(self, atom: ExtractedAtom, qid: str) -> Prediction:
        answer = QUESTIONS[qid]["answer_space"][0]
        return Prediction(answer=answer, would_skip=False)
```

Three things to notice:

- `QUESTIONS[qid]["answer_space"]` is the only piece of question
  metadata you need. Methods never hard-code question ids.
- The `atom` argument is unused here (a trivial baseline doesn't look
  at sources), but a real method would inspect `atom.profile_ltm`,
  `atom.daily_self_report`, etc.
- No reference to ground truth, no IO, no globals.

## Step 3 — Add a conformance test

[`test_always_first.py`](test_always_first.py) iterates all 18 question
ids and confirms each prediction is valid:

```bash
pytest examples/01_minimal_method/test_always_first.py -v
```

This 12-line test is the minimum every new method should ship with.

## Step 4 — Register the method (optional, only when promoting to the matrix)

For one-off experimentation you can keep the file under `examples/`. To
include the method in the comparison matrix and the paper-artifact
runner, register it once in
[`src/survey2agent/methods/__init__.py`](../../src/survey2agent/methods/__init__.py):

```python
from .always_first import AlwaysFirst
__all__ = [..., "AlwaysFirst"]
```

(There is no central registry dict — the package's `__all__` list is
the registry.)

## Step 5 — Evaluate

There is no per-method CLI runner; evaluation is driven by
`paper_artifacts/reproduce_paper.py` (which scores all registered
methods) and by the lightweight programmatic loop in
[example 03](../03_programmatic_api/). For a quick per-question
accuracy print using `AlwaysFirst`, run example 03's script:

```bash
python3 examples/03_programmatic_api/compute_accuracy.py
```

For an actual paper-style table including this method, register it in
`__all__` (Step 4 above) and run:

```bash
make reproduce-main
```

## Where to go next

- [`../02_custom_question/`](../02_custom_question/) — add a 19th
  question of your own.
- [`../03_programmatic_api/`](../03_programmatic_api/) — drive the
  evaluation from a notebook instead of the CLI.
- [`../../EXTENDING.md`](../../EXTENDING.md) — full reference for
  methods, questions, seeds, evidence streams, and LLM backends.
