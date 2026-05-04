"""Survey2Agent V2 — data generation package.

Persona, source, ground-truth, and NL-render generators that produce the released benchmark.
Flat module layout preserved verbatim so all intra-package relative imports
(e.g. ``from .constants import ...``) continue to resolve unchanged. This is
load-bearing for byte-equivalent dataset reproduction against the V5.6
reference (``$S2A_DATA_ROOT/benchmark/seeds/s20260321/``).

Pipeline layers (run as ``python -m survey2agent.data_generation.<entry>``):
  * L1 personas          -> :mod:`generate_personas`
  * L2 event tables      -> :mod:`generate_events`
  * L3 source projection -> :mod:`generate_sources`
  * L4 ground truth      -> :mod:`generate_ground_truth`
  * L5 NL render         -> :mod:`nl_render.nl_memory_renderer`
"""

from . import (
    behavioral_params,
    constants,
    diversity_audit,
    event_generator,
    event_schema,
    generate_events,
    generate_ground_truth,
    generate_personas,
    generate_sources,
    ground_truth,
    persona_generator,
    persona_schema,
    semantic_conflicts,
    source_projector,
    split_assigner,
)

__all__ = [
    "behavioral_params",
    "constants",
    "diversity_audit",
    "event_generator",
    "event_schema",
    "generate_events",
    "generate_ground_truth",
    "generate_personas",
    "generate_sources",
    "ground_truth",
    "persona_generator",
    "persona_schema",
    "semantic_conflicts",
    "source_projector",
    "split_assigner",
]
