"""Filesystem path constants for survey2agent.

All on-disk inputs (raw dataset, ground truth, NL renders, paper-artifact
result JSONs, pre-extracted atoms, method outputs) live under a single
configurable data root: ``S2A_DATA_ROOT``. The default is
``data/`` next to this source tree, which keeps the repository
self-contained for development.

Environment override
--------------------
Set ``S2A_DATA_ROOT`` to relocate the entire data tree, e.g. when storing
the published Hugging Face benchmark on a separate disk::

    export S2A_DATA_ROOT=/scratch/s2a_data

The override applies to **the whole tree** — ``benchmark/``,
``extracted_atoms/``, and ``method_outputs/`` are all expected to be
siblings under the chosen root. Mixing locations (e.g. moving only
``benchmark/`` while leaving the others in-repo) is unsupported.

Layout
------
::

    $S2A_DATA_ROOT/
    ├── extracted_atoms/{seed}/{persona_id}.json   # LLM-extracted atoms
    ├── method_outputs/...                         # cached method predictions
    └── benchmark/                                 # ★ HF-published dataset
        ├── seeds/{seed}/
        │   ├── config/personas.json
        │   ├── nl_renders/{persona_id}.md
        │   └── bench_*/{structural_sources/, ground_truth.json, ...}
        └── results/                               # paper-artifact source JSONs
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = [
    "PROJECT_ROOT",
    "DATA_ROOT",
    "EXTRACTED_ATOMS_ROOT",
    "METHOD_OUTPUTS_ROOT",
    "BENCHMARK_ROOT",
    "SEEDS_ROOT",
    "RESULTS_ROOT",
    "seed_dir",
    "persona_dir",
    "nl_renders_dir",
]


# ── Anchor: the release repository root ─────────────────────────────────────
# This file lives at: src/survey2agent/_paths.py
#   parents[0]=survey2agent, [1]=src, [2]=repo root
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]


# ── Configurable data root (environment-overridable) ────────────────────────
def _resolve_data_root() -> Path:
    override = os.environ.get("S2A_DATA_ROOT", "").strip()
    if override:
        return Path(override).expanduser().resolve()
    return PROJECT_ROOT / "data"


DATA_ROOT: Path = _resolve_data_root()

EXTRACTED_ATOMS_ROOT: Path = DATA_ROOT / "extracted_atoms"
METHOD_OUTPUTS_ROOT: Path = DATA_ROOT / "method_outputs"
BENCHMARK_ROOT: Path = DATA_ROOT / "benchmark"
SEEDS_ROOT: Path = BENCHMARK_ROOT / "seeds"
RESULTS_ROOT: Path = BENCHMARK_ROOT / "results"


# ── Per-seed convenience helpers ────────────────────────────────────────────
def seed_dir(seed: str) -> Path:
    """Return the per-seed directory: ``$DATA_ROOT/benchmark/seeds/{seed}``."""
    return SEEDS_ROOT / seed


def persona_dir(seed: str, persona_id: str) -> Path:
    """Return one persona directory under a seed."""
    return SEEDS_ROOT / seed / persona_id


def nl_renders_dir(seed: str) -> Path:
    """Return the natural-language render directory for a seed."""
    return SEEDS_ROOT / seed / "nl_renders"
