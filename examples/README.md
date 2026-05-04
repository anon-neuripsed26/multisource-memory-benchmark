# Examples

Five extension-point examples (all runnable):

| Dir | What you'll learn | Wall-clock |
|---|---|---|
| [`01_minimal_method/`](01_minimal_method/) | Add a new method (subclass `Method`, return `Prediction`, write a conformance test). | ~30 min |
| [`02_custom_question/`](02_custom_question/) | Add a new question (yaml entry, GT rule, bias matrix, re-generate one seed). | ~1 h |
| [`03_programmatic_api/`](03_programmatic_api/) | Drive the eval loop from a Python script in ~70 lines, no CLI runner needed. | ~5 min |
| [`04_sample_submission/`](04_sample_submission/) | Generate a valid leaderboard submission and verify it locally. | ~10 min |
| [`05_custom_stream/`](05_custom_stream/) | Sandboxed projector + renderer + μ\* readout pattern for adding a sixth evidence stream (`calendar_hint` toy). Production integration checklist in EXTENDING §4. | ~10 min |

For the full cross-cutting reference (seeds, evidence streams, LLM
backends), see [../EXTENDING.md](../EXTENDING.md). For governance and
PR mechanics, see [../CONTRIBUTING.md](../CONTRIBUTING.md).
