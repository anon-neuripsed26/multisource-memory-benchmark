"""Prompt builders for LLM-Direct, Schema-Aware, Struct-LLM, and Few-Shot producers.

Each builder returns a :class: that a runner can submit
through a ``SyncLLMClient`` or ``BatchLLMClient``. The builders are
deliberately pure:

* ``build_direct_request`` and ``build_schema_aware_request`` read the
  persona's structural sources via
  :func:`survey2agent.extraction.extractor.load_sources_raw`, render them to
  NL text, and attach a canned question list plus an output instruction.

* ``build_struct_llm_request`` reads a pre-computed extraction bundle
  (``{persona, extraction}``) from disk in place of the NL sources. Used by
  the factorial ablation arm that isolates the effect of structured input.

* ``build_few_shot_request`` provides an exemplar answer for the specific
  question being asked alongside the persona's NL memory.

The three variants (direct, schema_aware, struct_llm) differ only in the
embedded instruction block and (for struct-llm) the input payload. The
few-shot producer has a different request/response schema.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Literal

from survey2agent.api_clients import CompletionRequest
from survey2agent.data_generation.nl_render.nl_memory_renderer import render_full_memory

from ..extraction.atoms import EXPECTED_QUESTION_IDS
from ..extraction.extractor import load_sources_raw
from ..extraction.question_spec import QUESTIONS, QUESTION_TEXT

Variant = Literal["direct", "schema_aware", "struct_llm"]

FEW_SHOT_QIDS = [
    "A1", "A2", "A3", "B2", "B3", "C2", "C3", "Ctrl1", "Ctrl2",
    "D1", "D2", "E1", "E2", "F1", "F2", "F3", "G1", "G2"
]

_EXEMPLAR_META_RE = re.compile(
    r"^\*\(Persona: [^,]+, Difficulty: [^)]+\)\*$",
    re.MULTILINE,
)


def _redact_few_shot_exemplar_metadata(text: str) -> str:
    """Remove bookkeeping IDs/difficulty tags from few-shot exemplar prompts."""
    n = 0

    def _replace(_: re.Match[str]) -> str:
        nonlocal n
        n += 1
        return f"*(Example persona {n})*"

    return _EXEMPLAR_META_RE.sub(_replace, text)


# ---------------------------------------------------------------------------
# Shared instruction blocks
# ---------------------------------------------------------------------------

_SYSTEM_DIRECT = (
    "You are answering a survey about one person based on their personal "
    "memory (five source streams). For each question, pick the single best "
    "answer label from the listed options. Respond with a JSON object of "
    'the form {"answers": {qid: {"answer": "<label>", "would_skip": '
    "false}}}. Set would_skip=true only when the sources give you no "
    "reliable basis to answer.\n"
)

_SYSTEM_SCHEMA_AWARE = (
    _SYSTEM_DIRECT
    + "You also receive, per question, the list of sources typically "
    "informative for that question and bias patterns of each source. Use "
    "this schema awareness to weigh conflicting sources.\n"
)

_SYSTEM_STRUCT_LLM = (
    "You are answering a survey about one person based on a structured "
    "per-source extraction (one candidate label per source per question). "
    "For each question, combine the source labels into a single best "
    "answer. Respond with a JSON object of the form "
    '{"answers": {qid: {"answer": "<label>", "would_skip": false}}}.\n'
)


def _render_question_block(variant: Variant) -> str:
    """Render the survey question block for the prompt."""
    lines: list[str] = ["## Survey Questions", ""]
    for qid in EXPECTED_QUESTION_IDS:
        q = QUESTIONS[qid]
        options = ", ".join(f"`{o}`" for o in q["answer_space"])
        lines.append(f"### {qid} ({q['topic']}, {q['time_window']} days)")
        lines.append(QUESTION_TEXT[qid])
        lines.append(f"Options: {options}")
        if variant == "schema_aware":
            # Lightweight schema hint: which sources the benchmark design
            # considers informative for this qid. Full schema-aware prompt
            # content lives under methods/prompts/schema_aware.md for the
            # experiments that use that canonical wording.
            lines.append(
                "(Consider source bias patterns when sources disagree.)"
            )
        lines.append("")
    return "\n".join(lines)


def _output_instruction() -> str:
    return (
        "\n## Output\n"
        "Return a single JSON object with key 'answers' mapping each "
        "question id to an object with fields 'answer' (one label string) "
        "and 'would_skip' (boolean). Example:\n"
        '```json\n{"answers": {"A1": {"answer": "20_or_more", '
        '"would_skip": false}}}\n```\n'
    )


# ---------------------------------------------------------------------------
# Per-variant request builders
# ---------------------------------------------------------------------------


def build_direct_request(persona_dir: Path) -> CompletionRequest:
    """Build the prompt for the plain LLM-Direct variant."""
    sources_raw = load_sources_raw(Path(persona_dir))
    memory_nl = render_full_memory(sources_raw)
    user_prompt = "\n".join(
        [
            "## Persona Memory",
            memory_nl,
            "",
            _render_question_block("direct"),
            _output_instruction(),
        ]
    )
    return CompletionRequest(
        user_prompt=user_prompt,
        system_prompt=_SYSTEM_DIRECT,
    )


def build_schema_aware_request(persona_dir: Path) -> CompletionRequest:
    """Build the prompt for the Schema-Aware variant."""
    sources_raw = load_sources_raw(Path(persona_dir))
    memory_nl = render_full_memory(sources_raw)
    user_prompt = "\n".join(
        [
            "## Persona Memory",
            memory_nl,
            "",
            _render_question_block("schema_aware"),
            _output_instruction(),
        ]
    )
    return CompletionRequest(
        user_prompt=user_prompt,
        system_prompt=_SYSTEM_SCHEMA_AWARE,
    )


def build_struct_llm_request(
    persona_id: str,
    extraction_bundle_path: Path,
) -> CompletionRequest:
    """Build the prompt for Struct-LLM (atom-grid as input).

    Args:
        persona_id: Expected persona id; used only to verify the bundle's
            own ``persona`` field matches (guards against a misplaced file).
        extraction_bundle_path: Path to a ``{persona}.json`` emitted by the
            extraction producer. Must match the frozen extraction schema.
    """
    data = json.loads(Path(extraction_bundle_path).read_text(encoding="utf-8"))
    if data.get("persona") != persona_id:
        raise ValueError(
            f"extraction bundle persona={data.get('persona')!r} does not "
            f"match expected persona_id={persona_id!r}"
        )
    grid = data["extraction"]

    table_lines: list[str] = [
        "## Per-Source Extracted Labels",
        "",
        "For each question, five candidate labels (one per source, or null "
        "when the source has no relevant data):",
        "",
    ]
    for qid in EXPECTED_QUESTION_IDS:
        per_src = grid.get(qid, {})
        entries = ", ".join(
            f"{src}={per_src.get(src) if per_src.get(src) is not None else 'null'}"
            for src in (
                "profile_ltm",
                "planner",
                "daily_self_report",
                "objective_log",
                "device_log",
            )
        )
        q = QUESTIONS[qid]
        options = ", ".join(f"`{o}`" for o in q["answer_space"])
        table_lines.append(f"- **{qid}**: {QUESTION_TEXT[qid]}")
        table_lines.append(f"  Options: {options}")
        table_lines.append(f"  Sources: {entries}")

    user_prompt = "\n".join(
        table_lines
        + ["", _output_instruction()]
    )
    return CompletionRequest(
        user_prompt=user_prompt,
        system_prompt=_SYSTEM_STRUCT_LLM,
    )


def build_few_shot_request(
    persona_dir: Path,
    qid: str,
    configs_root: Path,
) -> CompletionRequest:
    """Build a few-shot request for a single persona/qid pair."""
    configs_root = Path(configs_root)
    persona_dir = Path(persona_dir)
    
    agent_msg = (configs_root / "AGENTS.md").read_text(encoding="utf-8")
    output_rules = (configs_root / "specs/output-rules.md").read_text(encoding="utf-8")
    system_prompt = f"{agent_msg}\n\n{output_rules}"
    
    exemplar = _redact_few_shot_exemplar_metadata(
        (configs_root / f"exemplars/{qid}.md").read_text(encoding="utf-8")
    )
    sources_raw = load_sources_raw(persona_dir)
    memory_nl = render_full_memory(sources_raw)
    
    q = QUESTIONS[qid]
    answer_space = ", ".join(f"`{o}`" for o in q["answer_space"])
    
    user_prompt = (
        f"## Exemplar\n\n{exemplar}\n\n"
        f"## Persona Memory\n\n{memory_nl}\n\n"
        f"## Task\n\n"
        f"Answer the following question for this persona:\n"
        f"**{qid}**: {QUESTION_TEXT[qid]}\n"
        f"Answer options: {answer_space}"
    )
    
    return CompletionRequest(
        user_prompt=user_prompt,
        system_prompt=system_prompt,
        custom_id=None,
    )


# ---------------------------------------------------------------------------
# Response parsing
# ---------------------------------------------------------------------------


def parse_answers_response(response_text: str) -> dict[str, dict[str, str | bool]]:
    """Parse an answers-bundle JSON response into ``{qid: {answer, would_skip}}``.

    Tolerates ```json`` fences and stray prose. Missing qids are filled with
    a defensive placeholder (``answer=""``, ``would_skip=True``) so the
    producer can write a schema-conformant bundle even on partial outputs.
    """
    import re

    text = response_text.strip()
    fenced = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # Try to pull the first top-level object.
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            return {qid: {"answer": "", "would_skip": True} for qid in EXPECTED_QUESTION_IDS}
        try:
            data = json.loads(m.group())
        except json.JSONDecodeError:
            return {qid: {"answer": "", "would_skip": True} for qid in EXPECTED_QUESTION_IDS}

    answers_obj = data.get("answers") if isinstance(data, dict) else None
    if not isinstance(answers_obj, dict):
        # Some models emit the qid map at the top level.
        answers_obj = data if isinstance(data, dict) else {}

    out: dict[str, dict[str, str | bool]] = {}
    for qid in EXPECTED_QUESTION_IDS:
        entry = answers_obj.get(qid) if isinstance(answers_obj, dict) else None
        if isinstance(entry, dict) and "answer" in entry:
            out[qid] = {
                "answer": str(entry.get("answer", "")),
                "would_skip": bool(entry.get("would_skip", False)),
            }
        elif isinstance(entry, str):
            out[qid] = {"answer": entry, "would_skip": False}
        else:
            out[qid] = {"answer": "", "would_skip": True}
    return out


__all__ = [
    "Variant",
    "FEW_SHOT_QIDS",
    "build_direct_request",
    "build_schema_aware_request",
    "build_struct_llm_request",
    "build_few_shot_request",
    "parse_answers_response",
]
