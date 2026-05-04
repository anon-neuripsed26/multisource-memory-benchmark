"""Core persona generation logic.

``generate_personas`` is the single entry point.  It produces *exactly* 480
personas (160 per difficulty class), retrying with seed offsets until
diversity constraints are satisfied (up to 25 attempts).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.random import Generator, PCG64

from .behavioral_params import (
    apply_shift,
    build_stated_profile,
    sample_actual_params_for_stated,
    sample_primary_exercise,
    sample_shift_scenario,
    sample_stable_params,
)
from .constants import (
    AGE_RANGE,
    CITIES,
    DIETARY_RESTRICTIONS,
    DIETARY_WEIGHTS,
    DIFFICULTY_TYPES,
    FIRST_NAMES,
    GENDERS,
    LAST_NAMES,
    OCCUPATIONS,
    PERSONAS_PER_DIFFICULTY,
    RELATIONSHIPS,
    SOCIAL_PREFERENCES,
    SOCIAL_PREF_WEIGHTS,
    TOTAL_PERSONAS,
)
from .diversity_audit import build_diversity_audit
from .persona_schema import (
    BehavioralParams,
    Persona,
    Profile,
    SparsityInjection,
    StableTraits,
    StatedProfile,
    StatedVsRevealed,
    TemporalShift,
    persona_to_dict,
)
from .semantic_conflicts import sample_semantic_conflict_profile


# ── Name generation ──────────────────────────────────────────────────────────

def _pick_name(index: int) -> str:
    """Deterministic name from *1-based* index.  Cycles through first×last."""
    zero = index - 1
    first = FIRST_NAMES[zero % len(FIRST_NAMES)]
    last = LAST_NAMES[(zero // len(FIRST_NAMES)) % len(LAST_NAMES)]
    return f"{first} {last}"


def _persona_id(prefix: str, idx: int, name: str) -> str:
    slug = name.lower().replace(" ", "_")
    return f"bench_{prefix}_{idx:03d}_{slug}"


# ── Profile & traits sampling ────────────────────────────────────────────────

def _sample_profile(rng: Generator) -> Profile:
    return Profile(
        age=int(rng.integers(AGE_RANGE[0], AGE_RANGE[1] + 1)),
        gender=GENDERS[int(rng.integers(len(GENDERS)))],
        occupation=OCCUPATIONS[int(rng.integers(len(OCCUPATIONS)))],
        relationship=RELATIONSHIPS[int(rng.integers(len(RELATIONSHIPS)))],
        city=CITIES[int(rng.integers(len(CITIES)))],
    )


def _sample_stable_traits(rng: Generator, primary_exercise: str) -> StableTraits:
    w_diet = list(DIETARY_WEIGHTS)
    w_diet_norm = [x / sum(w_diet) for x in w_diet]
    w_soc = list(SOCIAL_PREF_WEIGHTS)
    w_soc_norm = [x / sum(w_soc) for x in w_soc]
    return StableTraits(
        dietary_restriction=DIETARY_RESTRICTIONS[int(rng.choice(len(DIETARY_RESTRICTIONS), p=w_diet_norm))],
        social_preference=SOCIAL_PREFERENCES[int(rng.choice(len(SOCIAL_PREFERENCES), p=w_soc_norm))],
        close_friends=int(rng.integers(3, 11)),
        primary_exercise=primary_exercise,
    )


# ── Single-persona builders ─────────────────────────────────────────────────

def _build_stable_persona(rng: Generator, idx: int, global_name_idx: int) -> Persona:
    name = _pick_name(global_name_idx)
    exercise = sample_primary_exercise(rng)
    profile = _sample_profile(rng)
    traits = _sample_stable_traits(rng, exercise)
    behaviour = sample_stable_params(rng, exercise)

    scp = sample_semantic_conflict_profile(
        difficulty_type="stable",
        dietary_restriction=traits.dietary_restriction,
        occupation=profile.occupation,
        rng=rng,
    )

    return Persona(
        id=_persona_id("stable", idx, name),
        name=name,
        description="Stable benchmark persona",
        difficulty_type="stable",
        profile=profile,
        stable_traits=traits,
        behavioral_params=behaviour,
        semantic_conflict_profile=scp,
        temporal_shift=None,
        stated_vs_revealed=None,
    )


def _build_shift_persona(rng: Generator, idx: int, global_name_idx: int) -> Persona:
    name = _pick_name(global_name_idx)
    exercise = sample_primary_exercise(rng)
    profile = _sample_profile(rng)
    traits = _sample_stable_traits(rng, exercise)
    base_behaviour = sample_stable_params(rng, exercise)

    scenario, shift_day = sample_shift_scenario(rng)
    shifted = apply_shift(base_behaviour, scenario, rng)

    scp = sample_semantic_conflict_profile(
        difficulty_type="temporal_shift",
        dietary_restriction=traits.dietary_restriction,
        occupation=profile.occupation,
        rng=rng,
    )

    return Persona(
        id=_persona_id("shift", idx, name),
        name=name,
        description="Temporal-shift benchmark persona",
        difficulty_type="temporal_shift",
        profile=profile,
        stable_traits=traits,
        behavioral_params=base_behaviour,
        semantic_conflict_profile=scp,
        temporal_shift=TemporalShift(
            shift_day=shift_day,
            event_description=str(scenario["event_description"]),
            shifted_params=shifted,
            long_memory_reflects="pre_shift",
        ),
        stated_vs_revealed=None,
    )


def _build_stated_persona(rng: Generator, idx: int, global_name_idx: int) -> Persona:
    name = _pick_name(global_name_idx)
    exercise = sample_primary_exercise(rng)
    profile = _sample_profile(rng)
    traits = _sample_stable_traits(rng, exercise)
    actual = sample_actual_params_for_stated(rng, exercise)
    stated = build_stated_profile(actual, rng)

    scp = sample_semantic_conflict_profile(
        difficulty_type="stated_vs_revealed",
        dietary_restriction=traits.dietary_restriction,
        occupation=profile.occupation,
        rng=rng,
    )

    return Persona(
        id=_persona_id("stated", idx, name),
        name=name,
        description="Stated-vs-revealed benchmark persona",
        difficulty_type="stated_vs_revealed",
        profile=profile,
        stable_traits=traits,
        behavioral_params=actual,
        semantic_conflict_profile=scp,
        temporal_shift=None,
        stated_vs_revealed=StatedVsRevealed(
            stated_profile=StatedProfile(**stated),
            sparsity_injection=SparsityInjection(
                # IMPORTANT: missing_field_prob is sampled FIRST and is
                # currently unused downstream. It is retained for byte-level
                # reproducibility of the published dataset; see the note in
                # persona_schema.SparsityInjection. Do NOT
                # remove or reorder these two rng.uniform calls.
                missing_field_prob=round(float(rng.uniform(0.08, 0.18)), 2),
                missing_day_prob=round(float(rng.uniform(0.03, 0.08)), 2),
            ),
        ),
    )


# ── Public API ───────────────────────────────────────────────────────────────

_BUILDERS = {
    "stable": _build_stable_persona,
    "temporal_shift": _build_shift_persona,
    "stated_vs_revealed": _build_stated_persona,
}


def generate_personas(
    seed: int,
    *,
    per_difficulty: int = PERSONAS_PER_DIFFICULTY,
    max_attempts: int = 25,
) -> tuple[list[dict[str, Any]], dict[str, Any], int]:
    """Generate ``per_difficulty * 3`` personas, retrying until diversity passes.

    Parameters
    ----------
    seed : int
        Base RNG seed.
    per_difficulty : int
        Personas per difficulty class (default 160).
    max_attempts : int
        Max generation rounds before raising.

    Returns
    -------
    (personas_dicts, diversity_audit, attempt)
        *personas_dicts* is a list of plain dicts (JSON-serialisable).
    """
    personas_out: list[dict[str, Any]] = []
    audit: dict[str, Any] = {}
    winning_attempt = 0

    for attempt in range(1, max_attempts + 1):
        rng = Generator(PCG64(seed + attempt - 1))
        personas: list[Persona] = []
        global_name_idx = 1

        for dt in DIFFICULTY_TYPES:
            builder = _BUILDERS[dt]
            for idx in range(1, per_difficulty + 1):
                personas.append(builder(rng, idx, global_name_idx))
                global_name_idx += 1

        personas_dicts = [persona_to_dict(p) for p in personas]
        audit = build_diversity_audit(personas_dicts)
        audit["attempt"] = attempt

        if audit["passed"]:
            personas_out = personas_dicts
            winning_attempt = attempt
            break
    else:
        # Take last attempt even if not passed (raise below)
        personas_out = [persona_to_dict(p) for p in personas]

    if winning_attempt == 0:
        raise RuntimeError(
            f"Failed to generate a persona set passing diversity audit "
            f"after {max_attempts} attempts (seed={seed})."
        )

    return personas_out, audit, winning_attempt
