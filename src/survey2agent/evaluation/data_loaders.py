"""Data loaders for evaluation: atoms, ground truth, and split membership.

Path resolution
---------------
All paths derive from :data:`survey2agent._paths.DATA_ROOT` (overridable
via the ``S2A_DATA_ROOT`` environment variable):

- Atoms live in ``$DATA_ROOT/extracted_atoms/{seed}/{persona_id}.json``.
  Note: only the **test split** (120 personas) has been pre-extracted in
  the current frozen artifact set; ``load_atoms_for_seed`` returns
  whatever is on disk and does not assume 480.
- Ground truth lives in ``$DATA_ROOT/benchmark/seeds/{seed}/{persona_dir}/ground_truth.json``.
- Splits are derived from
  ``$DATA_ROOT/benchmark/seeds/s20260321/config/personas.json`` via
  :func:`survey2agent.data_generation.split_assigner.assign_splits`. The
  ``configs/splits.yaml`` file only records the canonical proportions
  (216/48/96/120) and does not enumerate persona IDs, so the yaml is
  unused for persona assignment. Splits are seed-stable: persona IDs are
  identical across the four seeds, so the canonical seed-1 assignment
  applies to every seed.

Naming
------
The split assigner uses ``"calibration"`` internally; we expose it as
``"cal"`` to match the public spec (``configs/splits.yaml`` and the rest
of the pipeline).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Mapping, Sequence

from survey2agent._paths import (
    EXTRACTED_ATOMS_ROOT,
    PROJECT_ROOT,
    SEEDS_ROOT,
    seed_dir,
)
from survey2agent.data_generation.split_assigner import assign_splits
from survey2agent.extraction.atoms import (
    EXPECTED_QUESTION_IDS,
    ExtractedAtom,
    load_atom,
    load_atoms_from_dir,
)

__all__ = [
    "TrainingRecord",
    "build_training_records",
    "load_atoms_for_seed",
    "load_ground_truths",
    "load_persona_difficulty_index",
    "load_splits",
    "PROJECT_ROOT",
]

_DEFAULT_SPLITS_YAML: Path = PROJECT_ROOT / "configs" / "splits.yaml"
_CANONICAL_SPLIT_SEED_DIR: Path = SEEDS_ROOT / "s20260321"

# Map split-assigner internal name → public split name.
_SPLIT_NAME_MAP: dict[str, str] = {
    "train": "train",
    "dev": "dev",
    "calibration": "cal",
    "test": "test",
}


# ── Public types ────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class TrainingRecord:
    """One ``(persona, qid, label)`` triple for fit / calibrate / evaluate.

    Distinct from :data:`survey2agent.methods.base.MethodTrainingRecord`
    (which is a per-persona method-facing record). The runner regroups these
    per-question records by persona before calling ``method.fit`` /
    ``method.calibrate``.

    The metadata fields ``reasoning_type``, ``topic``, and
    ``difficulty_class`` are populated by :func:`build_training_records`
    so downstream evaluation can group results without re-loading the
    question spec or persona spec.
    """

    atom: ExtractedAtom
    qid: str
    label: str
    reasoning_type: str
    topic: str
    difficulty_class: str


# ── Loaders ─────────────────────────────────────────────────────────────────


def load_splits(splits_yaml: Path | None = None) -> dict[str, list[str]]:
    """Return ``{split_name: [persona_id, ...]}`` for the canonical assignment.

    The ``splits_yaml`` argument is accepted for API stability but is
    **unused**: the on-disk yaml only encodes proportions (see module
    docstring), so the persona-level assignment is recomputed from
    ``personas.json`` via the deterministic stratified assigner.

    Keys: ``train`` / ``dev`` / ``cal`` / ``test`` (sizes 216 / 48 / 96 / 120).
    """
    _ = splits_yaml or _DEFAULT_SPLITS_YAML  # documented as unused
    personas_path = _CANONICAL_SPLIT_SEED_DIR / "config" / "personas.json"
    if not personas_path.is_file():
        raise FileNotFoundError(
            f"canonical personas.json not found at {personas_path}"
        )
    with personas_path.open("r", encoding="utf-8") as fh:
        personas = json.load(fh)["personas"]

    assignment = assign_splits(personas)
    mapping: dict[str, str] = assignment["mapping"]

    out: dict[str, list[str]] = {v: [] for v in _SPLIT_NAME_MAP.values()}
    for persona_id, raw_split in mapping.items():
        public_name = _SPLIT_NAME_MAP[raw_split]
        out[public_name].append(persona_id)
    for split in out:
        out[split].sort()
    return out


def load_atoms_for_seed(
    seed: str, *, mode: Literal["llm", "oracle"] = "llm"
) -> dict[str, ExtractedAtom]:
    """Load atoms for a seed.

    Parameters
    ----------
    seed : str
        Seed identifier, e.g. ``"s20260321"``.
    mode : ``"llm"`` or ``"oracle"``, keyword-only
        - ``"llm"`` (default): loads pre-computed LLM-extracted atoms from
          ``$DATA_ROOT/extracted_atoms/{seed}/``. In the current frozen
          artifact set only the test split (120 personas) has been
          pre-extracted.
        - ``"oracle"``: deterministically computes oracle atoms from raw
          sources at
          ``$DATA_ROOT/benchmark/seeds/{seed}/bench_*/structural_sources/``
          (480 personas, all splits). Runs ``compute_all_mu`` per persona
          via :func:`survey2agent.extraction.build_oracle_atoms_for_seed`.
          Enables ``Method.fit`` / ``Method.calibrate`` on full
          train/cal splits without LLM extraction artifacts.

    Returns ``{persona_id: ExtractedAtom}``.
    """
    if mode == "oracle":
        from survey2agent.extraction.oracle_extractor import (
            build_oracle_atoms_for_seed,
        )

        return build_oracle_atoms_for_seed(seed)
    if mode == "llm":
        atom_dir = EXTRACTED_ATOMS_ROOT / seed
        if not atom_dir.is_dir():
            raise FileNotFoundError(
                f"extracted-atoms directory not found: {atom_dir}"
            )
        return load_atoms_from_dir(atom_dir)
    raise ValueError(f"mode must be 'llm' or 'oracle', got {mode!r}")


def load_ground_truths(seed: str) -> dict[str, dict[str, str]]:
    """Load ground-truth labels for every persona under a seed.

    Reads ``$DATA_ROOT/benchmark/seeds/{seed}/{persona_dir}/ground_truth.json``
    for each persona directory. Returns ``{persona_id: {qid: label_str}}``.

    Each GT JSON has schema ``{qid: {"question_id": str, "answer": str, ...}}``;
    only the ``answer`` field is extracted. Per the locked rule "skip is
    never GT", entries whose ``answer`` equals the SKIP sentinel are
    excluded (in practice none exist).
    """
    seed_root = seed_dir(seed)
    if not seed_root.is_dir():
        raise FileNotFoundError(f"dataset seed directory not found: {seed_root}")

    # Lazy import to avoid a hard dependency cycle through methods.base.
    from survey2agent.methods.base import SKIP_SENTINEL

    out: dict[str, dict[str, str]] = {}
    for persona_dir in sorted(seed_root.iterdir()):
        if not persona_dir.is_dir() or not persona_dir.name.startswith("bench_"):
            continue
        gt_path = persona_dir / "ground_truth.json"
        if not gt_path.is_file():
            continue
        with gt_path.open("r", encoding="utf-8") as fh:
            raw = json.load(fh)
        per_qid: dict[str, str] = {}
        for qid, entry in raw.items():
            if not isinstance(entry, dict) or "answer" not in entry:
                continue
            answer = entry["answer"]
            if not isinstance(answer, str):
                continue
            if answer == SKIP_SENTINEL:
                continue
            per_qid[qid] = answer
        out[persona_dir.name] = per_qid
    return out


_PERSONA_DIFFICULTY_VALUES: frozenset[str] = frozenset(
    {"stable", "temporal_shift", "stated_vs_revealed"}
)


def load_persona_difficulty_index(seed: str = "s20260321") -> dict[str, str]:
    """Return ``{persona_id: difficulty_class}`` for one seed.

    Reads ``$DATA_ROOT/benchmark/seeds/{seed}/config/personas.json`` and
    extracts the ``difficulty_type`` field for each persona. Values are one
    of ``"stable"`` / ``"temporal_shift"`` / ``"stated_vs_revealed"`` (160
    each in the canonical 480-persona benchmark).

    Persona ids are seed-stable across the four seeds, so the canonical
    seed-1 file applies to every seed; the ``seed`` argument is exposed for
    explicit cross-seed verification.

    Raises:
        FileNotFoundError: if the seed directory or ``personas.json`` is
            missing.
        ValueError: if a persona entry lacks ``id`` / ``difficulty_type`` or
            carries an unknown difficulty value.
    """
    personas_path = seed_dir(seed) / "config" / "personas.json"
    if not personas_path.is_file():
        raise FileNotFoundError(
            f"personas.json not found for seed {seed!r}: {personas_path}"
        )
    with personas_path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    personas = payload.get("personas")
    if not isinstance(personas, list):
        raise ValueError(
            f"personas.json at {personas_path} missing top-level 'personas' list"
        )

    out: dict[str, str] = {}
    for entry in personas:
        if not isinstance(entry, dict):
            continue
        pid = entry.get("id")
        diff = entry.get("difficulty_type")
        if not isinstance(pid, str) or not isinstance(diff, str):
            raise ValueError(
                f"persona entry missing 'id' or 'difficulty_type' in "
                f"{personas_path}: {entry!r}"
            )
        if diff not in _PERSONA_DIFFICULTY_VALUES:
            raise ValueError(
                f"persona {pid!r} has unknown difficulty_type {diff!r} "
                f"(expected one of {sorted(_PERSONA_DIFFICULTY_VALUES)})"
            )
        out[pid] = diff
    return out


def build_training_records(
    atoms: Mapping[str, ExtractedAtom],
    ground_truths: Mapping[str, Mapping[str, str]],
    persona_ids: Sequence[str],
    qids: Sequence[str] | None = None,
    *,
    difficulty_index: Mapping[str, str] | None = None,
) -> list[TrainingRecord]:
    """Cartesian product of ``persona_ids × qids`` into ``TrainingRecord``s.

    - ``qids=None`` defaults to all 18 question IDs (``EXPECTED_QUESTION_IDS``).
    - ``(persona, qid)`` pairs are skipped silently if the persona has no
      atom, no GT, or the qid is missing from GT.
    - ``difficulty_index`` maps ``persona_id → difficulty_class`` (one of
      ``"stable"`` / ``"temporal_shift"`` / ``"stated_vs_revealed"``). Defaults
      to :func:`load_persona_difficulty_index` for the canonical seed
      (persona ids are seed-stable per the module docstring).
    - Each emitted ``TrainingRecord`` is enriched with ``reasoning_type``
      and ``topic`` from ``questions.yaml``.
    """
    # Lazy import: avoids a hard dependency from data_loaders to
    # extraction.question_spec at module import time.
    from survey2agent.extraction.question_spec import QUESTIONS

    selected_qids: tuple[str, ...] = (
        tuple(qids) if qids is not None else EXPECTED_QUESTION_IDS
    )
    diff_idx: Mapping[str, str] = (
        difficulty_index
        if difficulty_index is not None
        else load_persona_difficulty_index()
    )
    records: list[TrainingRecord] = []
    for pid in persona_ids:
        atom = atoms.get(pid)
        gt = ground_truths.get(pid)
        if atom is None or gt is None:
            continue
        difficulty_class = diff_idx.get(pid)
        if difficulty_class is None:
            raise KeyError(
                f"persona {pid!r} not found in difficulty_index "
                f"({len(diff_idx)} entries); "
                f"check that difficulty_index covers the seed in use"
            )
        for qid in selected_qids:
            label = gt.get(qid)
            if label is None:
                continue
            qspec = QUESTIONS.get(qid)
            if qspec is None:
                raise KeyError(
                    f"qid {qid!r} not found in questions.yaml"
                )
            records.append(
                TrainingRecord(
                    atom=atom,
                    qid=qid,
                    label=label,
                    reasoning_type=qspec["type"],
                    topic=qspec["topic"],
                    difficulty_class=difficulty_class,
                )
            )
    return records


# Re-export for convenience.
_ = load_atom  # keep imported symbol referenced for downstream callers
