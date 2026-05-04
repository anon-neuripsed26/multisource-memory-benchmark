"""Paper main tables reproduction gating tests.

Pins every cell of the four main paper tables (Forced-Accuracy Main Table, 2x2 Factorial Decomposition, Per-Type Diagnostic Analysis,
Full Selective QA Table) to within ±0.005 of the published numbers, by invoking each
script's ``main()`` and asserting the returned failure count is zero.
Each script's ``emit_row`` already attaches a ``paper_match`` marker per
cell using the same ±0.005 tolerance.

Sources
-------
* Forced-Accuracy Main Table / 2x2 Factorial Decomposition / Per-Type Diagnostic Analysis — Experiments section
* Full Selective QA Table — Appendix selective-QA section

No LLM API calls — every cell uses frozen artifacts under
``data/method_outputs/`` (including the new
``struct_llm/<mode>/<seed>/`` family).

The whole module is marked ``slow`` (combined runtime ~25-40 minutes).
"""

from __future__ import annotations

import pytest


pytestmark = [pytest.mark.slow, pytest.mark.needs_data]


def test_forced_accuracy_main_paper_lock() -> None:
    """Forced-Accuracy Main Table — every cell within ±0.005 of paper."""
    from paper_artifacts.main import forced_accuracy_main
    fail_count = forced_accuracy_main.main()
    assert fail_count == 0, (
        f"forced_accuracy_main: {fail_count} cells outside ±0.005 of paper "
        f"(see paper_artifacts/output/main/forced_accuracy.csv for FAIL rows)"
    )


def test_factorial_decomposition_paper_lock() -> None:
    """2x2 Factorial Decomposition — 4 cells + 4 decomposition values."""
    from paper_artifacts.main import factorial_decomposition
    fail_count = factorial_decomposition.main()
    assert fail_count == 0, (
        f"factorial_decomposition: {fail_count} cells/values outside tolerance "
        f"(see paper_artifacts/output/main/factorial.csv)"
    )


def test_per_type_accuracy_paper_lock() -> None:
    """Per-Type Diagnostic Analysis — per-type DSNBF, GPT-μ*, Δ, BestLLM, Source Reachability within ±0.005."""
    from paper_artifacts.main import per_type_accuracy
    fail_count = per_type_accuracy.main()
    assert fail_count == 0, (
        f"per_type_accuracy: {fail_count} cells outside ±0.005 of paper "
        f"(see paper_artifacts/output/main/per_type_accuracy.csv)"
    )


def test_selective_qa_full_paper_lock() -> None:
    """Full Selective QA Table — 17 rows × 2 metrics = 34 cells within ±0.005."""
    from paper_artifacts.main import selective_qa_full
    fail_count = selective_qa_full.main()
    assert fail_count == 0, (
        f"selective_qa_full: {fail_count} cells outside ±0.005 of paper "
        f"(see paper_artifacts/output/main/selective_qa.csv)"
    )
