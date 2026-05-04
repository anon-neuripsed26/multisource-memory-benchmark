"""Reproduce paper Source-Reachability Complement.

Source: ``Appendix benchmark-details section`` (Table
``tab:B4-source-ceiling-complement`` at lines 124-155).

The table reports answer-only instance accuracy (%) on three slices
of the held-out test set, split by direct source reachability:

  Full       — all 8,640 (persona, qid) instances pooled across 4 seeds
  GT-present — 8,049 instances where at least one direct-readout source
               atom mu*_s exactly equals GT (the 93.16% Source Reachability)
  GT-absent  — 591 instances forming the 6.84% Source Reachability complement

Source Reachability itself is omitted because the GT-absent slice is defined
by its miss condition.

Per-cell paper-lock budget:
  19 rows x 3 metrics (Full / GT-present / GT-absent) = 57 paper-lock cells.

Method coverage (paper row order):
  T0  Majority Class                    (mu_hat)
  T1  SSB                               (mu_hat)
  T2  Majority Vote / ArgRAG-style, BCF, NBF, DSNBF, ABF on mu_hat
      NBF, DSNBF on mu*
  T3  GPT-5.4 Struct-LLM on mu_hat and mu*
      LLM Direct/Schema for GPT-5.4, Gemini 3.1, DeepSeek V3.2, Qwen3-235B
      (NL memory)
"""

from __future__ import annotations

import sys
import time
from typing import Callable

from .._common import (
    CANONICAL_SEEDS,
    PAPER_TOLERANCE,
    emit_row,
    load_shared_resources,
    run_llm_across_seeds,
    run_mixed_mode_across_seeds,
    run_oracle_mode_across_seeds,
    run_struct_llm_across_seeds,
    write_outputs,
    _md_fmt_pct,
)
from survey2agent.evaluation.runner import EvaluationResult
from survey2agent.methods import (
    ABF,
    BCF,
    DSNBF,
    LLMDirect,
    LLMSchemaAware,
    MajorityClass,
    MajorityVote,
    NBF,
    SSB,
    atom_to_mu_q,
)


Key = tuple[str, str, str]  # (seed, persona_id, qid)


# (method_label, input) -> {"full": pct, "gt_present": pct, "gt_absent": pct}
# Values are fractions (0-1). Source: paper Source-Reachability Complement hardcoded cells
# (appendix_b_benchmark_details.tex lines 134-153).
PAPER_TAB_SOURCE_CEILING_COMPLEMENT: dict[tuple[str, str], dict[str, float]] = {
    ("Majority Class",          "mu_hat"):    {"full": 0.571, "gt_present": 0.596, "gt_absent": 0.227},
    ("SSB",                     "mu_hat"):    {"full": 0.772, "gt_present": 0.819, "gt_absent": 0.132},
    ("Majority Vote / ArgRAG-style",  "mu_hat"):    {"full": 0.688, "gt_present": 0.734, "gt_absent": 0.058},
    ("BCF",                     "mu_hat"):    {"full": 0.692, "gt_present": 0.731, "gt_absent": 0.152},
    ("NBF",                     "mu_hat"):    {"full": 0.798, "gt_present": 0.826, "gt_absent": 0.421},
    ("DSNBF",                   "mu_hat"):    {"full": 0.803, "gt_present": 0.823, "gt_absent": 0.535},
    ("ABF",                     "mu_hat"):    {"full": 0.720, "gt_present": 0.765, "gt_absent": 0.107},
    ("NBF",                     "mu_star"):   {"full": 0.820, "gt_present": 0.860, "gt_absent": 0.277},
    ("DSNBF",                   "mu_star"):   {"full": 0.823, "gt_present": 0.857, "gt_absent": 0.360},
    ("GPT-5.4 Struct-LLM",      "mu_hat"):    {"full": 0.714, "gt_present": 0.762, "gt_absent": 0.059},
    ("GPT-5.4 Struct-LLM",      "mu_star"):   {"full": 0.735, "gt_present": 0.788, "gt_absent": 0.012},
    ("GPT-5.4 Direct",          "nl_memory"): {"full": 0.686, "gt_present": 0.724, "gt_absent": 0.173},
    ("GPT-5.4 Schema",          "nl_memory"): {"full": 0.697, "gt_present": 0.741, "gt_absent": 0.098},
    ("Gemini 3.1 Direct",       "nl_memory"): {"full": 0.678, "gt_present": 0.719, "gt_absent": 0.122},
    ("Gemini 3.1 Schema",       "nl_memory"): {"full": 0.700, "gt_present": 0.743, "gt_absent": 0.115},
    ("DeepSeek V3.2 Direct",    "nl_memory"): {"full": 0.561, "gt_present": 0.581, "gt_absent": 0.289},
    ("DeepSeek V3.2 Schema",    "nl_memory"): {"full": 0.569, "gt_present": 0.593, "gt_absent": 0.239},
    ("Qwen3 235B Direct",       "nl_memory"): {"full": 0.421, "gt_present": 0.418, "gt_absent": 0.464},
    ("Qwen3 235B Schema",       "nl_memory"): {"full": 0.480, "gt_present": 0.471, "gt_absent": 0.592},
}


# Paper "Group" column (T0/T1/T2/T3). Used only for md formatting.
_GROUP: dict[str, str] = {
    "Majority Class":          "T0",
    "SSB":                     "T1",
    "Majority Vote / ArgRAG-style":  "T2",
    "BCF":                     "T2",
    "NBF":                     "T2",
    "DSNBF":                   "T2",
    "ABF":                     "T2",
    "GPT-5.4 Struct-LLM":      "T3",
    "GPT-5.4 Direct":          "T3",
    "GPT-5.4 Schema":          "T3",
    "Gemini 3.1 Direct":       "T3",
    "Gemini 3.1 Schema":       "T3",
    "DeepSeek V3.2 Direct":    "T3",
    "DeepSeek V3.2 Schema":    "T3",
    "Qwen3 235B Direct":       "T3",
    "Qwen3 235B Schema":       "T3",
}


_INPUT_LABEL = {
    "mu_hat":    r"$\hat\mu$",
    "mu_star":   r"$\mu^\ast$",
    "nl_memory": "NL memory",
}


def _answer(r: EvaluationResult) -> str:
    """Forced-answer scoring: use raw_answer when selective wrappers set it."""
    return r.prediction.raw_answer if r.prediction.raw_answer is not None else r.prediction.answer


def _score(seed_results: dict[str, list[EvaluationResult]], keys: set[Key]) -> tuple[int, int, float]:
    total = 0
    correct = 0
    for seed, results in seed_results.items():
        for r in results:
            if (seed, r.persona_id, r.qid) not in keys:
                continue
            total += 1
            correct += int(_answer(r) == r.label)
    if total == 0:
        return 0, 0, float("nan")
    return correct, total, correct / total


def _build_masks(splits, gts, oracle_atoms) -> tuple[set[Key], set[Key], set[Key]]:
    all_keys: set[Key] = set()
    gt_present: set[Key] = set()
    gt_absent: set[Key] = set()
    for seed in CANONICAL_SEEDS:
        for persona_id in splits["test"]:
            atom = oracle_atoms[seed][persona_id]
            for qid, label in gts[seed][persona_id].items():
                key = (seed, persona_id, qid)
                all_keys.add(key)
                source_values = [v for v in atom_to_mu_q(atom, qid).values() if v is not None]
                if label in source_values:
                    gt_present.add(key)
                else:
                    gt_absent.add(key)
    return all_keys, gt_present, gt_absent


def _emit_method_rows(
    *,
    method_label: str,
    input_key: str,
    seed_results,
    masks: tuple[set[Key], set[Key], set[Key]],
    csv_rows: list[dict],
    md_acc: dict[tuple[str, str], dict[str, float]],
) -> None:
    full_keys, present_keys, absent_keys = masks
    paper_cells = PAPER_TAB_SOURCE_CEILING_COMPLEMENT[(method_label, input_key)]
    full_pt   = _score(seed_results, full_keys)[2]
    pres_pt   = _score(seed_results, present_keys)[2]
    abs_c, abs_n, abs_pt = _score(seed_results, absent_keys)
    md_acc[(method_label, input_key)] = {
        "full": full_pt, "gt_present": pres_pt, "gt_absent": abs_pt,
        "absent_correct": abs_c, "absent_total": abs_n,
    }
    row_base = f"{method_label}__{input_key}"
    csv_rows.append(emit_row(
        row_id=f"{row_base}__full",
        method_label=method_label,
        mode=input_key,
        metric="instance_acc_full",
        point=full_pt,
        n_personas=len(full_keys),
        paper_point=paper_cells["full"],
    ))
    csv_rows.append(emit_row(
        row_id=f"{row_base}__gt_present",
        method_label=method_label,
        mode=input_key,
        metric="instance_acc_gt_present",
        point=pres_pt,
        n_personas=len(present_keys),
        paper_point=paper_cells["gt_present"],
    ))
    csv_rows.append(emit_row(
        row_id=f"{row_base}__gt_absent",
        method_label=method_label,
        mode=input_key,
        metric="instance_acc_gt_absent",
        point=abs_pt,
        n_personas=len(absent_keys),
        paper_point=paper_cells["gt_absent"],
    ))
    print(
        f"  {_GROUP[method_label]} {method_label:24s} {input_key:10s} "
        f"full={_md_fmt_pct(full_pt)} pres={_md_fmt_pct(pres_pt)} "
        f"abs={_md_fmt_pct(abs_pt)} ({abs_c}/{abs_n})",
        flush=True,
    )


def _build_md_table(md_acc: dict[tuple[str, str], dict[str, float]]) -> str:
    # Paper row order
    order: list[tuple[str, str]] = [
        ("Majority Class",         "mu_hat"),
        ("SSB",                    "mu_hat"),
        ("Majority Vote / ArgRAG-style", "mu_hat"),
        ("BCF",                    "mu_hat"),
        ("NBF",                    "mu_hat"),
        ("DSNBF",                  "mu_hat"),
        ("ABF",                    "mu_hat"),
        ("NBF",                    "mu_star"),
        ("DSNBF",                  "mu_star"),
        ("GPT-5.4 Struct-LLM",     "mu_hat"),
        ("GPT-5.4 Struct-LLM",     "mu_star"),
        ("GPT-5.4 Direct",         "nl_memory"),
        ("GPT-5.4 Schema",         "nl_memory"),
        ("Gemini 3.1 Direct",      "nl_memory"),
        ("Gemini 3.1 Schema",      "nl_memory"),
        ("DeepSeek V3.2 Direct",   "nl_memory"),
        ("DeepSeek V3.2 Schema",   "nl_memory"),
        ("Qwen3 235B Direct",      "nl_memory"),
        ("Qwen3 235B Schema",      "nl_memory"),
    ]
    lines = [
        "| Group | Method | Input | Full | GT-present | GT-absent |",
        "|:---|:---|:---|---:|---:|---:|",
    ]
    last_group = ""
    for method, ikey in order:
        cells = md_acc.get((method, ikey))
        group = _GROUP[method] if _GROUP[method] != last_group else ""
        last_group = _GROUP[method]
        if cells is None:
            lines.append(f"| {group} | {method} | {_INPUT_LABEL[ikey]} | — | — | — |")
            continue
        method_disp = f"**{method}**" if method == "DSNBF" and ikey == "mu_hat" else method
        lines.append(
            f"| {group} | {method_disp} | {_INPUT_LABEL[ikey]} | "
            f"{_md_fmt_pct(cells['full'])} | "
            f"{_md_fmt_pct(cells['gt_present'])} | "
            f"{_md_fmt_pct(cells['gt_absent'])} |"
        )
    return "\n".join(lines)


def main() -> int:
    t0 = time.time()
    print("Loading frozen resources...", flush=True)
    splits, diff_idx, gts, oracle_atoms, llm_atoms = load_shared_resources()
    masks = _build_masks(splits, gts, oracle_atoms)
    full_keys, present_keys, absent_keys = masks
    print(
        f"Slice sizes: full={len(full_keys)}, gt_present={len(present_keys)}, "
        f"gt_absent={len(absent_keys)} ({len(absent_keys) / len(full_keys):.4%})",
        flush=True,
    )

    csv_rows: list[dict] = []
    md_acc: dict[tuple[str, str], dict[str, float]] = {}

    fusion_specs: list[tuple[str, Callable]] = [
        ("Majority Class",         MajorityClass),
        ("SSB",                    SSB),
        ("Majority Vote / ArgRAG-style", MajorityVote),
        ("BCF",                    BCF),
        ("NBF",                    NBF),
        ("DSNBF",                  DSNBF),
        ("ABF",                    ABF),
    ]
    for label, factory in fusion_specs:
        seed_results = run_mixed_mode_across_seeds(
            factory,
            oracle_atoms=oracle_atoms,
            llm_atoms=llm_atoms,
            gts=gts,
            splits=splits,
            diff_idx=diff_idx,
        )
        _emit_method_rows(
            method_label=label,
            input_key="mu_hat",
            seed_results=seed_results,
            masks=masks,
            csv_rows=csv_rows,
            md_acc=md_acc,
        )

    for label, factory in [("NBF", NBF), ("DSNBF", DSNBF)]:
        seed_results = run_oracle_mode_across_seeds(
            factory,
            oracle_atoms=oracle_atoms,
            gts=gts,
            splits=splits,
            diff_idx=diff_idx,
        )
        _emit_method_rows(
            method_label=label,
            input_key="mu_star",
            seed_results=seed_results,
            masks=masks,
            csv_rows=csv_rows,
            md_acc=md_acc,
        )

    for mode, ikey in [("extracted", "mu_hat"), ("oracle", "mu_star")]:
        seed_results = run_struct_llm_across_seeds(
            mode=mode,
            display_name=f"GPT-5.4 Struct {mode}",
            method_cls=LLMDirect,
            eval_atoms=llm_atoms if mode == "extracted" else oracle_atoms,
            gts=gts,
            splits=splits,
            diff_idx=diff_idx,
        )
        _emit_method_rows(
            method_label="GPT-5.4 Struct-LLM",
            input_key=ikey,
            seed_results=seed_results,
            masks=masks,
            csv_rows=csv_rows,
            md_acc=md_acc,
        )

    llm_specs: list[tuple[str, str, str, type]] = [
        ("GPT-5.4 Direct",        "gpt-5.4",                 "direct",       LLMDirect),
        ("GPT-5.4 Schema",        "gpt-5.4",                 "schema-aware", LLMSchemaAware),
        ("Gemini 3.1 Direct",     "gemini_p2",               "direct",       LLMDirect),
        ("Gemini 3.1 Schema",     "gemini_p2",               "schema-aware", LLMSchemaAware),
        ("DeepSeek V3.2 Direct",  "deepseek-v3.2",           "direct",       LLMDirect),
        ("DeepSeek V3.2 Schema",  "deepseek-v3.2",           "schema-aware", LLMSchemaAware),
        ("Qwen3 235B Direct",     "qwen3-235b-a22b-2507",    "direct",       LLMDirect),
        ("Qwen3 235B Schema",     "qwen3-235b-a22b-2507",    "schema-aware", LLMSchemaAware),
    ]
    for label, model, variant, cls in llm_specs:
        seed_results = run_llm_across_seeds(
            model=model,
            variant=variant,
            display_name=label,
            selective_cls=cls,
            llm_atoms=llm_atoms,
            gts=gts,
            splits=splits,
            diff_idx=diff_idx,
        )
        _emit_method_rows(
            method_label=label,
            input_key="nl_memory",
            seed_results=seed_results,
            masks=masks,
            csv_rows=csv_rows,
            md_acc=md_acc,
        )

    md_caption = (
        "Forced-answer instance accuracy (%) on three slices of the held-out "
        "test set, split by direct source reachability of GT. "
        f"Slice sizes: Full={len(full_keys)}, GT-present={len(present_keys)}, "
        f"GT-absent={len(absent_keys)} (the 6.84% Source Reachability complement)."
    )
    md_table = _build_md_table(md_acc)
    write_outputs(
        "source_ceiling_complement_table",
        csv_rows,
        md_table=md_table,
        script_name="paper_artifacts.appendix.source_ceiling_complement_table",
        md_caption=md_caption,
        md_footnotes=[
            "Paper-lock budget: 19 rows x 3 metrics = 57 cells "
            f"(tolerance ±{PAPER_TOLERANCE}).",
            "Source: paper Appendix benchmark-details section "
            "(tab:B4-source-ceiling-complement, lines 124-155).",
        ],
        subdir="appendix",
    )

    n_fail = sum(1 for r in csv_rows if str(r.get("paper_match", "")).startswith("FAIL"))
    elapsed = time.time() - t0
    print(
        f"\nWrote 57 paper-lock cells in {elapsed:.1f}s. "
        f"Match status: {len(csv_rows) - n_fail}/{len(csv_rows)} OK, {n_fail} FAIL.",
        flush=True,
    )
    if n_fail:
        print("FAIL rows:", flush=True)
        for r in csv_rows:
            if str(r.get("paper_match", "")).startswith("FAIL"):
                print(f"  {r['row_id']}: point={r['point']} paper={r['paper_value_point']} "
                      f"-> {r['paper_match']}", flush=True)
    return n_fail


if __name__ == "__main__":
    raise SystemExit(main())
