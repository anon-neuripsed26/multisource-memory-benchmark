# Example 04 — Sample leaderboard submission

A runnable end-to-end example showing exactly what a leaderboard
submission looks like: it iterates the real persona directories under
`data/benchmark/seeds/<seed>/`, runs the trivial `Random` baseline,
and writes one schema-valid `<persona>.json` per persona.

The output is structurally identical to what an external contributor
must submit per [SUBMISSION_PROTOCOL.md](../../SUBMISSION_PROTOCOL.md).
Use it as a copy-and-modify template.

## Run

```bash
make fetch            # required: --strict reads real persona names from data/benchmark/seeds/
python3 examples/04_sample_submission/make_submission.py \
    --output-dir examples/04_sample_submission/out \
    --seed s20260321

python3 -m paper_artifacts.verify_external_submission \
    --predictions examples/04_sample_submission/out \
    --method-name SampleRandom-2026 \
    --strict
```

Expected output:

```
OK: 480 files validated across 1 seed(s) (480 (seed, persona) tuples).
```

(For all 4 seeds, run `make_submission.py` four times and verify
without `--seed`.)

## What this fixes vs. raw protocol-reading

A contributor reading `SUBMISSION_PROTOCOL.md` alone may stumble on
two things this example demonstrates by construction:

1. **Real persona names are not `persona_NNNNNN`** — they look like
   `bench_shift_001_drew_carter`, derived from the persona generator.
   The script reads them from the on-disk benchmark directory, so the
   filenames are guaranteed to match what `--strict` will look for.
2. **The verifier cross-checks `persona_id` against the filename
   stem.** This script writes both consistently; submissions that
   construct filenames separately from `persona_id` get rejected by
   the verifier.

If your method is non-trivial, replace `RandomMethod` in
`make_submission.py` with your own `Method` subclass and otherwise keep
the script as-is.
