"""Smoke tests for paper_artifacts/ scripts.

Wiring-level checks only — they confirm imports succeed, the ``--help``
CLI works, and one fast subset of Full Selective QA Table produces non-empty output.
Full reproduction is exercised by the paper-lock test
``test_paper_main_tables_reproduction.py`` and (manually) by
``python -m paper_artifacts.reproduce_paper``.
"""

from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path

import pytest


def test_forced_accuracy_main_imports() -> None:
    mod = importlib.import_module("paper_artifacts.main.forced_accuracy_main")
    assert callable(mod.main)
    assert hasattr(mod, "PANEL_A_PAPER")
    assert hasattr(mod, "PANEL_B_PAPER")


def test_factorial_decomposition_imports() -> None:
    mod = importlib.import_module("paper_artifacts.main.factorial_decomposition")
    assert callable(mod.main)
    assert hasattr(mod, "PAPER_CELLS")
    assert set(mod.PAPER_CELLS) == {
        "fusion_oracle", "llm_oracle", "fusion_extracted", "llm_extracted"
    }


def test_per_type_accuracy_imports() -> None:
    mod = importlib.import_module("paper_artifacts.main.per_type_accuracy")
    assert callable(mod.main)
    assert hasattr(mod, "PAPER_TAB_PER_TYPE_ACCURACY")
    assert len(mod.PAPER_TAB_PER_TYPE_ACCURACY) == 8


def test_selective_qa_full_imports() -> None:
    mod = importlib.import_module("paper_artifacts.main.selective_qa_full")
    assert callable(mod.main)
    assert hasattr(mod, "TABLE_F1_SPEC")
    assert len(mod.TABLE_F1_SPEC) == 17


def test_per_type_macro_accuracy_full_imports() -> None:
    mod = importlib.import_module(
        "paper_artifacts.appendix.per_type_macro_accuracy_full"
    )
    assert callable(mod.main)


def test_t2_fusion_per_type_per_difficulty_imports() -> None:
    mod = importlib.import_module(
        "paper_artifacts.appendix.t2_fusion_per_type_per_difficulty"
    )
    assert callable(mod.main)
    assert hasattr(mod, "PAPER_TAB_T2_FUSION_PER_TYPE_PER_DIFFICULTY")
    assert hasattr(mod, "PAPER_FOOTER_DROP")


def test_t3_llm_per_type_per_difficulty_imports() -> None:
    mod = importlib.import_module(
        "paper_artifacts.appendix.t3_llm_per_type_per_difficulty"
    )
    assert callable(mod.main)
    assert hasattr(mod, "PAPER_TAB_T3_LLM_PER_TYPE_PER_DIFFICULTY")
    assert hasattr(mod, "PAPER_FOOTER_DROP")
    assert len(mod.LLM_COLS) == 8


def test_t2_fusion_selective_per_type_per_difficulty_imports() -> None:
    mod = importlib.import_module(
        "paper_artifacts.appendix.t2_fusion_selective_per_type_per_difficulty"
    )
    assert callable(mod.main)
    assert hasattr(mod, "PAPER_TAB_T2_FUSION_SELECTIVE_PER_TYPE_PER_DIFFICULTY")
    assert hasattr(mod, "PAPER_FOOTER_DROP")


def test_t3_llm_selective_per_type_per_difficulty_imports() -> None:
    mod = importlib.import_module(
        "paper_artifacts.appendix.t3_llm_selective_per_type_per_difficulty"
    )
    assert callable(mod.main)
    assert hasattr(mod, "PAPER_TAB_T3_LLM_SELECTIVE_PER_TYPE_PER_DIFFICULTY")
    assert hasattr(mod, "PAPER_FOOTER_DROP")
    assert len(mod.LLM_COLS) == 8


def test_prediction_distributions_e_causal_imports() -> None:
    mod = importlib.import_module(
        "paper_artifacts.appendix.prediction_distributions_e_causal"
    )
    assert callable(mod.main)
    assert hasattr(mod, "PAPER_TAB_PREDICTION_DISTRIBUTIONS_E_CAUSAL")
    assert set(mod.PAPER_TAB_PREDICTION_DISTRIBUTIONS_E_CAUSAL) == {"E1", "E2"}
    assert mod.SOURCE_ORDER == ("GT", "GPT", "DS", "Qwen3")


def test_prediction_distributions_c_pr_f_miss_imports() -> None:
    mod = importlib.import_module(
        "paper_artifacts.appendix.prediction_distributions_c_pr_f_miss"
    )
    assert callable(mod.main)
    assert hasattr(mod, "PAPER_TAB_PREDICTION_DISTRIBUTIONS_C_PR_F_MISS")
    assert set(mod.PAPER_TAB_PREDICTION_DISTRIBUTIONS_C_PR_F_MISS) == {"C2", "F3"}
    # F3 uses the canonical long-form label, displayed as "yes_worked".
    assert "yes_worked_despite_no_entry" in mod.ANSWER_ORDER["F3"]
    assert mod.DISPLAY_LABEL["yes_worked_despite_no_entry"] == "yes_worked"


def test_cross_seed_stability_imports() -> None:
    mod = importlib.import_module("paper_artifacts.appendix.cross_seed_stability")
    assert callable(mod.main)
    assert hasattr(mod, "PAPER_TAB_CROSS_SEED_STABILITY")


def test_train_size_ablation_imports() -> None:
    mod = importlib.import_module("paper_artifacts.appendix.train_size_ablation")
    assert callable(mod.main)
    assert hasattr(mod, "PAPER_TAB_TRAIN_SIZE_ABLATION")


def test_cross_condition_gpt_vs_gemini_imports() -> None:
    mod = importlib.import_module(
        "paper_artifacts.appendix.cross_condition_gpt_vs_gemini"
    )
    assert callable(mod.main)
    assert hasattr(mod, "PAPER_TAB_CROSS_CONDITION_GPT_VS_GEMINI")


def test_noise_perturbation_imports() -> None:
    mod = importlib.import_module("paper_artifacts.appendix.noise_perturbation")
    assert callable(mod.main)
    assert hasattr(mod, "PAPER_TAB_NOISE_PERTURBATION")


def test_dgp_perturbation_imports() -> None:
    mod = importlib.import_module("paper_artifacts.appendix.dgp_perturbation")
    assert callable(mod.main)
    assert hasattr(mod, "PAPER_TAB_DGP_PERTURBATION")
    assert hasattr(mod, "PAPER_NARRATIVE_DGP_PERTURBATION")
    assert len(mod.VARIANTS) == 9
    assert len(mod.COL_ORDER) == 7
    # SSB in DGP perturbation table must map to SSB-Global, NOT Single-Source-Best.
    assert mod.METHOD_MAP["SSB"] == "SSB-Global"


def test_cross_extractor_robustness_imports() -> None:
    mod = importlib.import_module(
        "paper_artifacts.appendix.cross_extractor_robustness"
    )
    assert callable(mod.main)
    assert hasattr(mod, "PAPER_TAB_CROSS_EXTRACTOR_ROBUSTNESS")


def test_cross_bias_transfer_imports() -> None:
    mod = importlib.import_module(
        "paper_artifacts.appendix.cross_bias_transfer"
    )
    assert callable(mod.main)
    assert hasattr(mod, "PAPER_TAB_CROSS_BIAS_TRANSFER")
    # 9 variants × 3 metrics (transfer / retrained / gap_pp) = 27 paper-lock cells.
    assert len(mod.PAPER_TAB_CROSS_BIAS_TRANSFER) == 27
    assert len(mod.VARIANTS) == 9
    # gap tolerance is in pp units (matching paper rounding precision),
    # not the standard ±0.005 fraction tolerance.
    assert mod.GAP_TOLERANCE_PP == 0.5


def test_per_question_extraction_accuracy_imports() -> None:
    mod = importlib.import_module(
        "paper_artifacts.appendix.per_question_extraction_accuracy"
    )
    assert callable(mod.main)
    assert hasattr(mod, "PAPER_TAB_PER_QUESTION_EXTRACTION_ACCURACY")
    assert len(mod.QUESTIONS) == 18


def test_atom_extraction_faithfulness_imports() -> None:
    mod = importlib.import_module(
        "paper_artifacts.appendix.atom_extraction_faithfulness"
    )
    assert callable(mod.main)
    assert hasattr(mod, "PAPER_TAB_ATOM_FAITHFULNESS")
    assert hasattr(mod, "PAPER_NARRATIVE_ATOM_FAITHFULNESS")
    # 5 per-source + Overall
    assert len(mod.PAPER_TAB_ATOM_FAITHFULNESS) == 6
    assert "Overall" in mod.PAPER_TAB_ATOM_FAITHFULNESS
    # 11 paper §B-4 narrative claims
    assert len(mod.PAPER_NARRATIVE_ATOM_FAITHFULNESS) == 11
    assert mod.PAPER_NARRATIVE_ATOM_FAITHFULNESS["overall_faithfulness_pct"] == 93.08
    assert mod.PAPER_NARRATIVE_ATOM_FAITHFULNESS["total_cells"] == 43200.0


def test_api_cache_inventory_imports() -> None:
    mod = importlib.import_module(
        "paper_artifacts.appendix.api_cache_inventory"
    )
    assert callable(mod.main)
    assert hasattr(mod, "PAPER_TAB_API_INVENTORY")
    assert hasattr(mod, "EXPECTED_CACHE_COUNTS")
    assert hasattr(mod, "PAPER_ROW_ORDER")
    # 8 paper rows
    assert len(mod.PAPER_TAB_API_INVENTORY) == 8
    assert len(mod.EXPECTED_CACHE_COUNTS) == 8
    assert set(mod.PAPER_TAB_API_INVENTORY) == set(mod.PAPER_ROW_ORDER)
    # Paper total = 10,800
    assert mod.PAPER_TOTAL_CALLS == 10800
    assert sum(mod.PAPER_TAB_API_INVENTORY.values()) == 10800


def test_source_ceiling_complement_table_imports() -> None:
    mod = importlib.import_module(
        "paper_artifacts.appendix.source_ceiling_complement_table"
    )
    assert callable(mod.main)
    assert hasattr(mod, "PAPER_TAB_SOURCE_CEILING_COMPLEMENT")
    # 19 (method, input) rows in the source-reachability complement diagnostic
    assert len(mod.PAPER_TAB_SOURCE_CEILING_COMPLEMENT) == 19
    # Each row has full / gt_present / gt_absent cells (57 total)
    for key, cells in mod.PAPER_TAB_SOURCE_CEILING_COMPLEMENT.items():
        assert set(cells) == {"full", "gt_present", "gt_absent"}, key
    # Every method appears in the group map
    for method, _ in mod.PAPER_TAB_SOURCE_CEILING_COMPLEMENT:
        assert method in mod._GROUP


def test_reproduce_paper_help() -> None:
    """``python -m paper_artifacts.reproduce_paper --help`` exits 0."""
    result = subprocess.run(
        [sys.executable, "-m", "paper_artifacts.reproduce_paper", "--help"],
        capture_output=True,
        text=True,
        timeout=30,
        cwd=str(Path(__file__).resolve().parents[2]),  # repo root
    )
    assert result.returncode == 0, (
        f"--help failed: stdout={result.stdout!r} stderr={result.stderr!r}"
    )
    assert "tier" in result.stdout.lower()


@pytest.mark.slow
@pytest.mark.needs_data
def test_selective_qa_full_runs() -> None:
    """Run a single fast row of Full Selective QA Table (SSB ext) end-to-end."""
    from paper_artifacts.main import selective_qa_full
    from paper_artifacts._common import OUTPUT_DIR

    fail_count = selective_qa_full.main(rows=["ssb_ext"])
    assert fail_count == 0, f"ssb_ext drifted from paper: {fail_count} failed cells"

    csv_path = OUTPUT_DIR / "main" / "selective_qa.csv"
    md_path = OUTPUT_DIR / "main" / "selective_qa.md"
    assert csv_path.exists()
    assert md_path.exists()

    csv_text = csv_path.read_text(encoding="utf-8")
    assert "ssb_ext" in csv_text
    assert "selective_accuracy" in csv_text
    assert "coverage" in csv_text
    assert "OK" in csv_text
