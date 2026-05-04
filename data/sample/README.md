# `data/sample/` — Schema inspection and offline smoke tests

**This sample is for schema inspection and offline smoke tests only. It does not reproduce any number reported in the paper.**

Full paper reproduction requires the complete 4-seed × 480-persona benchmark, hosted on Hugging Face and materialized under `data/{benchmark,extracted_atoms,method_outputs}/` by the fetch script:

```bash
python data/fetch_benchmark.py
```

Then run `make reproduce` (~1-2 hours on a CPU-only laptop, zero API spend; see top-level `Makefile` for `make reproduce-main` / `make reproduce-appendix` subset targets).

## What is in this directory

The sample mirrors the release layout that lives under `data/{benchmark,extracted_atoms,method_outputs}/`, but contains only one (test-split) persona's full lifecycle plus one auxiliary atom file:

```
data/sample/
├── README.md                                  (this file)
├── personas_one.txt                           # one-persona CLI selector
├── questions_one.txt                          # one-question few-shot selector
├── benchmark/seeds/s20260321/
│   ├── bench_shift_121_avery_ellis/           # raw structural sources + GT
│   │   ├── event_table.json
│   │   ├── ground_truth.json
│   │   └── structural_sources/
│   │       ├── profile_ltm.json
│   │       ├── planner.json
│   │       ├── daily_self_report.json
│   │       ├── objective_log.json
│   │       ├── device_log.json
│   │       └── generation_metadata.json
│   └── nl_renders/
│       └── bench_shift_121_avery_ellis.md     # NL-rendered memory
├── extracted_atoms/s20260321/
│   ├── bench_shift_121_avery_ellis.json       # primary lifecycle persona
│   └── bench_shift_122_jordan_ellis.json      # second atom file (smoke test loads two)
└── method_outputs/gpt-5.4/s20260321/
    ├── direct/bench_shift_121_avery_ellis.json
    └── schema-aware/bench_shift_121_avery_ellis.json
```

`bench_shift_121_avery_ellis` is one full lifecycle example covering raw sources → NL render → extracted atoms → frozen LLM outputs.

`bench_shift_122_jordan_ellis` is included only as a second atom record so `tests/test_smoke_end_to_end.py` (which loads two atoms) can run against the sample alone.

## Smoke test

```bash
make smoke
```

The smoke test reads atoms from `data/sample/extracted_atoms/s20260321/` by default (with a fallback to the full `data/extracted_atoms/s20260321/` if the sample is unavailable). It exercises the `Random` and `MajorityClass` methods with synthetic ground truth — sample metrics are not paper metrics.

## Optional live-call smoke test

The sample also supports a one-persona, one-question live LLM call. This is **optional**, costs API spend, and is not needed for paper reproduction. The smallest synchronous path uses OpenRouter:

```bash
export OPENROUTER_API_KEY=...
PYTHONPATH=src python -m survey2agent run-few-shot \
  --provider openrouter --model deepseek-v3.2 \
  --seed data/sample/benchmark/seeds/s20260321 \
  --personas data/sample/personas_one.txt \
  --questions data/sample/questions_one.txt \
  --configs-root configs/few_shot \
  --output-dir /tmp/s2a_sample_few_shot_live \
  --allow-api-call
```

To verify the same path without touching the network, omit `--allow-api-call`; the command should fail with a cache-miss message after resolving the sample persona, question, and prompt bundle.

## Provenance

Every file under `data/sample/` is a byte-for-byte copy of the corresponding file under `data/{benchmark,extracted_atoms,method_outputs}/`. Do not hand-edit. To refresh after regenerating frozen artifacts, re-copy from the canonical locations.

## What this sample does not contain

- No train / cal / dev split data (only one test-split persona).
- No fitted method parameters (DSNBF / BCF / ABF require multi-persona training data).
- No bootstrap CI / per-type breakdown / multi-seed aggregation results.
- No data for seeds `s20260322` / `s20260323` / `s20260324`.

For any of the above, fetch the full benchmark.
