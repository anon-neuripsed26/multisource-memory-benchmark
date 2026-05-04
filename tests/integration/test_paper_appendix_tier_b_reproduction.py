"""Slow paper-lock reproduction tests for paper_artifacts/appendix/ scripts.

Each test runs the script's ``main()`` end-to-end against the frozen LLM
artifacts and asserts every published paper cell reproduces to within
±0.005 absolute (the shared
``paper_artifacts._common.PAPER_TOLERANCE``). Coverage spans every
appendix script registered in
``paper_artifacts.reproduce_paper._REGISTRY`` — Tier-B per-type /
per-difficulty breakdowns (D1, E2a, E2b, E2a-skip, E2b-skip,
D3, D3b), Tier-C diagnostics (C1–C6b), the B-4 extraction faithfulness
audit and source-ceiling complement table, and the API Call Inventory API cache
inventory.

These tests are gated on ``pytest.mark.slow``. Total wall clock for all
runs is roughly 4–6 minutes on a workstation. Skip in default CI:

    pytest -m "not slow"

Run only paper-lock:

    pytest tests/integration/test_paper_appendix_tier_b_reproduction.py -m slow -v
"""

from __future__ import annotations

import pytest

import importlib


pytestmark = [pytest.mark.slow, pytest.mark.needs_data]


_APPENDIX_SCRIPTS = [
    "per_type_macro_accuracy_full",
    "t2_fusion_per_type_per_difficulty",
    "t3_llm_per_type_per_difficulty",
    "t2_fusion_selective_per_type_per_difficulty",
    "t3_llm_selective_per_type_per_difficulty",
    "prediction_distributions_e_causal",
    "prediction_distributions_c_pr_f_miss",
    "cross_seed_stability",
    "train_size_ablation",
    "cross_condition_gpt_vs_gemini",
    "dgp_perturbation",
    "noise_perturbation",
    "cross_extractor_robustness",
    "cross_bias_transfer",
    "per_question_extraction_accuracy",
    "atom_extraction_faithfulness",
    "source_ceiling_complement_table",
    "api_cache_inventory",
]


def _run(script_name: str) -> int:
    mod = importlib.import_module(f"paper_artifacts.appendix.{script_name}")
    return mod.main()


def test_per_type_macro_accuracy_full_paper_lock() -> None:
    fail = _run("per_type_macro_accuracy_full")
    assert fail == 0, f"per_type_macro_accuracy_full drifted: {fail} cells outside tolerance"


def test_t2_fusion_per_type_per_difficulty_paper_lock() -> None:
    fail = _run("t2_fusion_per_type_per_difficulty")
    assert fail == 0, f"t2_fusion_per_type_per_difficulty drifted: {fail} cells outside tolerance"


def test_t3_llm_per_type_per_difficulty_paper_lock() -> None:
    fail = _run("t3_llm_per_type_per_difficulty")
    assert fail == 0, f"t3_llm_per_type_per_difficulty drifted: {fail} cells outside tolerance"


def test_t2_fusion_selective_per_type_per_difficulty_paper_lock() -> None:
    fail = _run("t2_fusion_selective_per_type_per_difficulty")
    assert fail == 0, f"t2_fusion_selective_per_type_per_difficulty drifted: {fail} cells outside tolerance"


def test_t3_llm_selective_per_type_per_difficulty_paper_lock() -> None:
    fail = _run("t3_llm_selective_per_type_per_difficulty")
    assert fail == 0, f"t3_llm_selective_per_type_per_difficulty drifted: {fail} cells outside tolerance"


def test_prediction_distributions_e_causal_paper_lock() -> None:
    fail = _run("prediction_distributions_e_causal")
    assert fail == 0, f"prediction_distributions_e_causal drifted: {fail} cells outside tolerance"


def test_prediction_distributions_c_pr_f_miss_paper_lock() -> None:
    fail = _run("prediction_distributions_c_pr_f_miss")
    assert fail == 0, f"prediction_distributions_c_pr_f_miss drifted: {fail} cells outside tolerance"


# ---------------------------------------------------------------------------
# Tier-C diagnostics + B-4 faithfulness + API Call Inventory.
# ---------------------------------------------------------------------------


def test_cross_seed_stability_paper_lock() -> None:
    fail = _run("cross_seed_stability")
    assert fail == 0, f"cross_seed_stability drifted: {fail} cells outside tolerance"


def test_train_size_ablation_paper_lock() -> None:
    fail = _run("train_size_ablation")
    assert fail == 0, f"train_size_ablation drifted: {fail} cells outside tolerance"


def test_cross_condition_gpt_vs_gemini_paper_lock() -> None:
    fail = _run("cross_condition_gpt_vs_gemini")
    assert fail == 0, f"cross_condition_gpt_vs_gemini drifted: {fail} cells outside tolerance"


def test_dgp_perturbation_paper_lock() -> None:
    fail = _run("dgp_perturbation")
    assert fail == 0, f"dgp_perturbation drifted: {fail} cells outside tolerance"


def test_noise_perturbation_paper_lock() -> None:
    fail = _run("noise_perturbation")
    assert fail == 0, f"noise_perturbation drifted: {fail} cells outside tolerance"


def test_cross_extractor_robustness_paper_lock() -> None:
    fail = _run("cross_extractor_robustness")
    assert fail == 0, f"cross_extractor_robustness drifted: {fail} cells outside tolerance"


def test_cross_bias_transfer_paper_lock() -> None:
    fail = _run("cross_bias_transfer")
    assert fail == 0, f"cross_bias_transfer drifted: {fail} cells outside tolerance"


def test_per_question_extraction_accuracy_paper_lock() -> None:
    fail = _run("per_question_extraction_accuracy")
    assert fail == 0, f"per_question_extraction_accuracy drifted: {fail} cells outside tolerance"


def test_atom_extraction_faithfulness_paper_lock() -> None:
    fail = _run("atom_extraction_faithfulness")
    assert fail == 0, f"atom_extraction_faithfulness drifted: {fail} cells outside tolerance"


def test_source_ceiling_complement_table_paper_lock() -> None:
    fail = _run("source_ceiling_complement_table")
    assert fail == 0, f"source_ceiling_complement_table drifted: {fail} cells outside tolerance"


def test_api_cache_inventory_paper_lock() -> None:
    fail = _run("api_cache_inventory")
    assert fail == 0, f"api_cache_inventory drifted: {fail} cells outside tolerance"
