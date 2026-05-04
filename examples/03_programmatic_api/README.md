# Example 03 — Programmatic API

Drive the benchmark from Python instead of the CLI runner. Useful for:

- Ad-hoc experimentation in a notebook
- Computing custom metrics not covered by the standard runner
- Smoke-testing a new method during development without writing
  prediction files to disk

[`compute_accuracy.py`](compute_accuracy.py) is a self-contained
~70-line script that loads ground truth from one seed, runs
`AlwaysFirst` (from example 01) over every (persona, qid) pair, and
prints per-question accuracy.

## Run

```bash
make fetch                     # only needed if you have not downloaded the benchmark yet
python3 examples/03_programmatic_api/compute_accuracy.py
```

Output (truncated):

```
qid       n   accuracy
A1      216      0.083
A2      216      0.139
...
```

Numbers vary slightly across seeds; the absolute values are not the
point. The point is that **roughly 70 lines is enough** to drive the
end-to-end loop: load GT → instantiate method → loop personas × qids
→ tally accuracy.

## Reusing the loop for your own method

Replace `AlwaysFirst` with your own subclass of `Method`, and (if your
method actually uses sources) load real `ExtractedAtom` objects from
`data/benchmark/seeds/<seed>/<persona>/extracted_atoms.json` instead of
the empty-atom shortcut. See
[`src/survey2agent/extraction/atoms.py`](../../src/survey2agent/extraction/atoms.py)
for the dataclass.

## Where this fits

This is the bottom layer: `predict_one` over `(atom, qid)`. The CLI
runner adds:

- Train / cal / test splitting (paper canon: 216/48/96/120)
- 4-seed bootstrap CIs (n=10000)
- Macro / Micro / Selective Macro / Coverage metrics
- Disk caching of predictions

If you need any of those, prefer
`make reproduce-main` (which scores every registered method end-to-end
and emits the paper tables under `paper_artifacts/output/main/`). If
you don't, the loop above is fine.
