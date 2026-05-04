# Example 05: Adding a Custom Evidence Stream (runnable sandbox)

> **Status:** runnable sandbox. The 6 files in this directory
> demonstrate the **projector → renderer → μ\* readout → tests**
> pattern for adding a sixth evidence stream. They run offline against
> a tiny local fixture and do **not** modify any production code or
> the comparison matrix. Full production integration of a new stream
> additionally requires hand-edits to `_source_loader.load_sources`,
> `extractor.SOURCE_QUESTION_MAP`, the NL render hook, and the bias
> matrix — see [`EXTENDING.md` §4](../../EXTENDING.md#4-add-a-new-evidence-stream)
> for the full checklist.

---

## What this example demonstrates

We add a toy sixth stream `calendar_hint`: the persona's pre-committed
calendar of planned social/exercise events. Bias direction `b = +1`:
planned events systematically over-report what actually happens
(some commitments are skipped or rescheduled). The sandbox shows how
this stream would slot into the existing five-stream architecture
(`profile_ltm`, `planner`, `daily_self_report`, `objective_log`,
`device_log`).

We reuse question H1 from
[`examples/02_custom_question/`](../02_custom_question/) (Social —
distinct activity-type count over the last 14 days, answer space
`0_to_1` | `2_to_3` | `4_or_more`) as the demo question. examples/02
adds a question; examples/05 adds a source. Together they cover the
two extension axes.

---

## Files in this directory

| File | Role | Production analogue |
|---|---|---|
| [`fixture_events.json`](fixture_events.json) | Toy 14-day event table with `planned` / `happened` flags | The latent L2 event table per persona |
| [`calendar_hint_projector.py`](calendar_hint_projector.py) | `events → calendar_hint stream view` (only `planned == True` rows) | `src/survey2agent/data_generation/source_projector.py` |
| [`calendar_hint_renderer.py`](calendar_hint_renderer.py) | Stream view → NL paragraph for the LLM extractor to read | `src/survey2agent/data_generation/nl_render/` |
| [`compute_mu_calendar_hint.py`](compute_mu_calendar_hint.py) | μ\*(calendar_hint, H1) deterministic readout | `src/survey2agent/extraction/_mu_shell.py::compute_all_mu` |
| [`test_custom_stream.py`](test_custom_stream.py) | 6 pytest checks: schema, +1 bias direction, render shape, μ\* in answer space, μ\* fixture value, ⊥ on empty | `tests/data_generation/test_smoke_byte_equivalence.py` |
| `README.md` | This file | — |

---

## Run the sandbox

```bash
python3 -m pytest examples/05_custom_stream/ -v
```

Expected: **6 passed**. The tests verify:

1. The projector emits entries with the documented schema fields.
2. Distinct planned activity types ⊇ distinct happened types
   (the +1 bias direction). The fixture has 9 planned types vs 7
   happened types; the extras (`book_club`, `museum_trip`) are the
   "phantom" planned events.
3. The renderer produces non-empty NL text with one line per entry.
4. μ\*(calendar_hint, H1) returns a value in `H1_ANSWER_SPACE`.
5. μ\*(calendar_hint, H1) on the fixture returns `4_or_more` (9 distinct types).
6. μ\* returns `None` on an empty stream (canonical ⊥).

---

## What this sandbox does NOT do

The sandbox is intentionally scoped to the **per-persona, per-source**
side of the pipeline. To deploy `calendar_hint` as a real sixth source
that the comparison matrix evaluates, you would additionally need:

| Step | File | Why |
|---|---|---|
| Append `"calendar_hint"` to the source enumeration | [`src/survey2agent/extraction/atoms.py`](../../src/survey2agent/extraction/atoms.py) `EXPECTED_SOURCES` | Atom factories iterate this tuple |
| Read `calendar_hint.json` in the persona loader | [`src/survey2agent/extraction/_source_loader.py`](../../src/survey2agent/extraction/_source_loader.py) `load_sources` | Currently hard-codes the 5 existing filenames |
| Add `calendar_hint` to per-question source maps | [`src/survey2agent/extraction/extractor.py`](../../src/survey2agent/extraction/extractor.py) `SOURCE_QUESTION_MAP` | LLM extractor needs to know which sources to read for each `qid` |
| Fold the renderer into the full memory render | The NL render entry point (currently `nl_memory_renderer.render_full_memory`) | LLM atoms come from the rendered NL document |
| Register `μ*(calendar_hint, q)` for every relevant `q` | [`src/survey2agent/extraction/_mu_shell.py`](../../src/survey2agent/extraction/_mu_shell.py) `compute_all_mu` | Source Reachability reference rows would otherwise see `None` |
| Declare `b ∈ {-1, 0, +1}` per topic | The bias-direction table consumed by [`src/survey2agent/methods/_bias_model.py`](../../src/survey2agent/methods/_bias_model.py) and ABF in [`src/survey2agent/methods/abf.py`](../../src/survey2agent/methods/abf.py); see paper appendix “Per-source bias model” | ABF and other bias-aware fusion methods learn `δ_calendar_hint` against this prior |
| Re-run paper-locked regression | `pytest tests/integration/test_paper_appendix_tier_b_reproduction.py` | All 23 paper tables re-derive from the new 6-source corpus |

The reason none of these is included here is that any of them, done
half-correctly, would silently degrade the published method comparison.
The sandbox keeps the comparison matrix untouched; it documents the
pattern, not the deployment.
