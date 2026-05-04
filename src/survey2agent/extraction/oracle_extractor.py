"""Deterministic oracle ExtractedAtom builder.

Builds an :class:`ExtractedAtom` for any persona by reading the raw source
JSONs and running the deterministic ``compute_all_mu`` extraction shell.
No LLM calls. Used to enable ``Method.fit`` / ``Method.calibrate`` on the
full 480-persona splits when LLM-extracted atoms are only available for
the 120-persona test split.
"""

from __future__ import annotations

from pathlib import Path

from .atoms import ExtractedAtom, _freeze_extraction
from ._mu_shell import compute_all_mu
from ._source_loader import load_sources


def build_oracle_atom(persona_dir: Path) -> ExtractedAtom:
    """Build a single oracle ``ExtractedAtom`` from a persona's source dir.

    Parameters
    ----------
    persona_dir : Path
        Path to e.g.
        ``$DATA_ROOT/benchmark/seeds/s20260321/bench_stable_001_*``.

    Returns
    -------
    ExtractedAtom
        ``persona`` = ``persona_dir.name``;
        ``extraction`` = output of ``compute_all_mu(load_sources(persona_dir))``,
        frozen via ``_freeze_extraction``.
    """
    persona_id = persona_dir.name
    sources = load_sources(persona_dir)
    mu_all = compute_all_mu(sources)
    return ExtractedAtom(persona=persona_id, extraction=_freeze_extraction(mu_all))


def build_oracle_atoms_for_seed(seed: str) -> dict[str, ExtractedAtom]:
    """Build oracle atoms for all 480 personas of a seed.

    Parameters
    ----------
    seed : str
        e.g. ``"s20260321"``.

    Returns
    -------
    dict[str, ExtractedAtom]
        ``{persona_id: ExtractedAtom}``, sorted by ``persona_id``.
    """
    # Local import to avoid a circular dependency at package import time.
    from survey2agent._paths import seed_dir as _seed_dir

    seed_root = _seed_dir(seed)
    if not seed_root.is_dir():
        raise FileNotFoundError(f"Seed dataset dir not found: {seed_root}")
    out: dict[str, ExtractedAtom] = {}
    for pd in sorted(seed_root.iterdir()):
        if pd.is_dir() and pd.name.startswith("bench_"):
            out[pd.name] = build_oracle_atom(pd)
    return out
