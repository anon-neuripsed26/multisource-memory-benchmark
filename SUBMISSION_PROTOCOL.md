# Submission protocol — leaderboard

This document describes how an external contributor submits a method's
predictions for inclusion in the rolling leaderboard.

The protocol is intentionally narrow: predictions only, in a fixed
JSON shape, validated by a single script. Methods are not required to
upstream code (though doing so via [EXTENDING.md §1](EXTENDING.md#1-add-a-new-method)
is encouraged).

## What you submit

For each `(seed, persona)` pair on the test split of every seed listed
in [`configs/seeds.yaml`](configs/seeds.yaml), produce one JSON file:

```
<your_artifact_dir>/
    s20260321/
        bench_shift_001_drew_carter.json
        bench_shift_002_jules_carter.json
        ...
    s20260322/
        ...
    s20260323/
        ...
    s20260324/
        ...
```

The filename stem must equal the persona id (i.e. the persona
sub-directory name under `data/benchmark/seeds/<seed>/`).

Each file conforms to
[`schemas/method_prediction.schema.json`](schemas/method_prediction.schema.json)
and looks like:

```json
{
  "method_name": "MyMethod-2026",
  "seed": "s20260321",
  "persona_id": "bench_shift_001_drew_carter",
  "predictions": {
    "A1": {"answer": "10_to_19",  "would_skip": false},
    "A2": {"answer": "SKIP",      "would_skip": true},
    "A3": {"answer": "40_to_69",  "would_skip": false, "raw_answer": "40_to_69"},
    "...": "...",
    "Ctrl2": {"answer": "1_to_2",  "would_skip": false}
  },
  "metadata": {
    "commit": "abc1234",
    "wall_clock_seconds": 312,
    "notes": "L-BFGS-B fit on train, calibrated on cal split."
  }
}
```

Constraints (all enforced by `verify_external_submission.py`):

1. Exactly the 18 required qids (no more, no less): `A1, A2, A3, B2,
   B3, C2, C3, D1, D2, E1, E2, F1, F2, F3, G1, G2, Ctrl1, Ctrl2`.
2. `(answer == "SKIP") == would_skip` for every prediction.
3. `raw_answer != "SKIP"`. For non-skip predictions, `raw_answer` is
   either `null` or equal to `answer`.
4. The seed string matches the directory name and the file's
   `persona_id` matches the filename stem.

## What the maintainer scores

Submissions are scored on the same answer-only and selective-QA metrics used
in the paper:

| Metric | Description |
|---|---|
| Answer-only macro accuracy | Per-question forced accuracy averaged across the 18 questions; SKIP uses `raw_answer` when present and otherwise counts as wrong |
| Selective accuracy | Per-question accuracy over answered predictions only; questions with zero answered predictions are dropped |
| Coverage | Fraction of questions answered (1 - skip rate) |
| F0.5 | Cost-asymmetric selective score used for calibration and secondary reporting |

Each metric is reported with a 4-seed pool-then-bootstrap 95% percentile CI
(B=2000, seed=42, persona-clustered), matching the paper protocol
(see `paper_artifacts/MANIFEST.md` and `src/survey2agent/evaluation/bootstrap.py`).
Submissions that opt out of SKIP simply report Coverage = 1.0 and
Selective accuracy = Answer-only macro accuracy.

## How you submit

1. Generate predictions in the directory layout above.
2. Validate locally:

   ```bash
   python3 -m paper_artifacts.verify_external_submission \
       --predictions <your_dir> \
       --method-name MyMethod-2026 \
       --strict
   ```

   The `--strict` flag confirms every persona on the **test split**
   (120 personas per seed, listed in
   [`configs/splits.yaml`](configs/splits.yaml)) has a prediction file.
   Required for leaderboard inclusion. The verifier exits with status 1
   and lists the first missing personas if coverage is incomplete; if
   the local benchmark directory is missing it exits with status 1 and
   tells you to `make fetch` first.

3. Open a leaderboard submission issue against the project tracker.
   Use the title `[leaderboard] <method-name>` and include the following
   in the body: (a) method name and one-line description, (b) verifier
   output (the final OK line), (c) reproduction command, (d) link to
   your prediction artefacts.
4. Attach (a) a download link to the artifact directory (e.g. a release
   asset or HF dataset entry), (b) the exact reproduction command, and
   (c) the verifier output showing OK.

A maintainer will re-run the verifier, score the submission against
the locked metrics, append the row to `paper_artifacts/leaderboard.csv`,
and merge.

## What disqualifies a submission

- Use of ground-truth labels at inference time (verified by
  reproduction; if your reproduction command reads from
  `ground_truth.json`, the submission is rejected).
- Per-question or per-persona hard-coded routing (your method must run
  uniformly over all 18 questions).
- Predictions for fewer than all four registered seeds.
- Schema violations the verifier flags.

## Cadence

The leaderboard is updated quarterly during the active period
(2026-Q4 through 2028-Q4) and annually thereafter, subject to the
maintenance commitment in [DATASHEET.md](DATASHEET.md).

## Attribution

Each leaderboard row records: method name, submitter (group / handle),
4-metric numbers with CI, submission commit hash, and a link back to
the submission issue. During the double-blind review period
submissions are listed under their issue number only; identity is
revealed at camera-ready.
