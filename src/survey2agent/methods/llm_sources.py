"""Concrete `LLMSource` implementations.

`FrozenBulkJSONSource` reads the per-persona 18-question bundles produced
by the frozen batch LLM Direct / Schema-Aware runs. Layout:

    {root}/{model}/{seed}/{variant}/{persona_id}.json
    {"persona": str, "answers": {qid: {"answer": str, "would_skip": bool}}}

`FrozenFewShotDirSource` reads per-(persona, qid) JSONs from the few-shot
results directory. Layout:

    {root}/{model}/{seed}/few-shot/{persona_id}__{qid}.json
    {"persona": str, "question": str, "answer": str, "would_skip": bool}

`LiveSource` is a stub: it raises `NotImplementedError` on every call.
Live LLM dispatch is intentionally out of scope for the frozen-only
validation phase.
"""

from __future__ import annotations

import json
from pathlib import Path

from survey2agent._paths import METHOD_OUTPUTS_ROOT

from .llm_base import LLMSource, RawLLMOutput

# Default root for frozen bulk and few-shot outputs. Tests and callers
# may override via the `root` kwarg. Resolves to
# `$S2A_DATA_ROOT/method_outputs/`.
_DEFAULT_BULK_ROOT = METHOD_OUTPUTS_ROOT
_DEFAULT_FEWSHOT_ROOT = _DEFAULT_BULK_ROOT
_FEWSHOT_VARIANT = "few-shot"

# Patch table for frozen records that violate Prediction
# invariants (e.g., the Qwen3 Ctrl2 singleton with answer="SKIP" +
# would_skip=False). See `frozen_artifact_quirks.json` for the schema and
# rationale. Loaded lazily and cached at module level so repeated
# instantiation of FrozenBulkJSONSource pays the I/O cost once.
_QUIRKS_FILENAME = "frozen_artifact_quirks.json"
_QUIRKS_CACHE: dict[tuple[str, str, str, str, str], dict[str, object]] | None = None


def _load_quirks(root: Path) -> dict[tuple[str, str, str, str, str], dict[str, object]]:
    """Return the quirks dict keyed by (model, seed, variant, persona, qid).

    Cached at module level after the first successful read against the
    canonical default root. A non-default ``root`` (e.g. tmp_path tests)
    re-reads from that root if a quirks file is present, otherwise returns
    an empty dict (no patches apply).
    """
    global _QUIRKS_CACHE
    if root == _DEFAULT_BULK_ROOT and _QUIRKS_CACHE is not None:
        return _QUIRKS_CACHE
    path = root / _QUIRKS_FILENAME
    if not path.exists():
        out: dict[tuple[str, str, str, str, str], dict[str, object]] = {}
    else:
        data = json.loads(path.read_text(encoding="utf-8"))
        out = {
            (
                str(p["model"]),
                str(p["seed"]),
                str(p["variant"]),
                str(p["persona"]),
                str(p["qid"]),
            ): dict(p["patch"])
            for p in data.get("patches", [])
        }
    if root == _DEFAULT_BULK_ROOT:
        _QUIRKS_CACHE = out
    return out


class FrozenBulkJSONSource(LLMSource):
    """Loads per-persona 18-question bundles from the bulk frozen layout."""

    def __init__(
        self,
        model: str,
        seed: str,
        variant: str,
        root: Path | None = None,
    ) -> None:
        base = Path(root) if root is not None else _DEFAULT_BULK_ROOT
        self.dir: Path = base / model / seed / variant
        self._cache: dict[str, dict[str, dict[str, str | bool]]] = {}
        self._model = model
        self._seed = seed
        self._variant = variant
        self._quirks = _load_quirks(base)

    def _load(self, persona_id: str) -> dict[str, dict[str, str | bool]]:
        if persona_id in self._cache:
            return self._cache[persona_id]
        path = self.dir / f"{persona_id}.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        answers = data["answers"]
        # Apply quirks patch BEFORE caching so all subsequent get() calls
        # see the patched values. Override is keyed by the full
        # (model, seed, variant, persona, qid) tuple to avoid collateral.
        for qid, entry in list(answers.items()):
            patch = self._quirks.get(
                (self._model, self._seed, self._variant, persona_id, qid)
            )
            if patch is not None:
                answers[qid] = {**entry, **patch}
        self._cache[persona_id] = answers
        return answers

    def get(self, persona_id: str, qid: str) -> RawLLMOutput:
        answers = self._load(persona_id)
        entry = answers[qid]
        return RawLLMOutput(answer=str(entry["answer"]), would_skip=bool(entry["would_skip"]))


class StructLLMSource(FrozenBulkJSONSource):
    """Loads structured-LLM frozen artifacts.

    Paper Tab:5 (factorial decomposition) and Tab:6 (per-type) use a
    GPT-5.4 agent that reads structured atom-as-prompt input (instead of
    NL text) and emits the same per-persona 18-question JSON bundle. The
    payload schema is byte-identical to ``FrozenBulkJSONSource``::

        {"persona": str, "answers": {qid: {"answer": str, "would_skip": bool}}}

    Layout differs from the bulk LLM artifacts: structured-LLM outputs
    are organized by ``(mode, seed)`` rather than ``(model, seed,
    variant)``, where ``mode ∈ {"oracle", "extracted"}`` indicates
    whether the agent read oracle μ or LLM-extracted μ atoms::

        {root}/struct_llm/{mode}/{seed}/{persona}.json

    The quirks lookup reuses the same patch table with
    ``model="struct_llm"`` and ``variant=mode``; no struct-llm patches
    are currently registered (forced-mode evaluation ignores
    ``would_skip``, so the structured-LLM agent's "guess + skip" records
    do not trigger any Prediction-invariant violation).
    """

    def __init__(
        self,
        mode: str,
        seed: str,
        root: Path | None = None,
    ) -> None:
        if mode not in ("oracle", "extracted"):
            raise ValueError(
                f"mode must be 'oracle' or 'extracted'; got {mode!r}"
            )
        base = Path(root) if root is not None else _DEFAULT_BULK_ROOT
        # Override the parent's path template; do NOT call super().__init__
        # because that would build the wrong directory.
        self.dir: Path = base / "struct_llm" / mode / seed
        self._cache: dict[str, dict[str, dict[str, str | bool]]] = {}
        self._model = "struct_llm"
        self._seed = seed
        self._variant = mode
        self._quirks = _load_quirks(base)


class FrozenFewShotDirSource(LLMSource):
    """Loads per-(persona, qid) JSONs from a few-shot results directory.

    Single-seed only (s20260321) — the few-shot pipeline did not run on
    the held-out seeds. The runner is responsible for restricting the
    eval split to that seed when this source is used.

    Two construction modes (mirrors `FrozenBulkJSONSource`):
    1. By `(model, seed)` against the canonical
       `data/method_outputs/{model}/{seed}/few-shot/` layout.
       `root=` overrides the data root.
    2. By an explicit `results_dir=` (keyword-only) pointing at any
       directory of `{persona}__{qid}.json` files. Used by tmp_path
       tests that build a fixture on the fly.
    """

    def __init__(
        self,
        model: str | None = None,
        seed: str = "s20260321",
        root: Path | None = None,
        *,
        results_dir: Path | None = None,
    ) -> None:
        if results_dir is not None:
            self.dir: Path = Path(results_dir)
            return
        if model is None:
            raise ValueError(
                "FrozenFewShotDirSource requires either `model` (with optional `seed`/`root`) "
                "or an explicit `results_dir=` keyword."
            )
        base = Path(root) if root is not None else _DEFAULT_FEWSHOT_ROOT
        self.dir = base / model / seed / _FEWSHOT_VARIANT

    def get(self, persona_id: str, qid: str) -> RawLLMOutput:
        path = self.dir / f"{persona_id}__{qid}.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        return RawLLMOutput(answer=str(data["answer"]), would_skip=bool(data["would_skip"]))


class LiveSource(LLMSource):
    """Stub for live LLM dispatch. Raises `NotImplementedError` on use."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs

    def get(self, persona_id: str, qid: str) -> RawLLMOutput:
        del persona_id, qid
        raise NotImplementedError(
            "LiveSource is a stub; live LLM dispatch is not implemented in this phase."
        )
