"""LLM atom extraction.

Public API:
    - `ExtractedAtom`, `MethodPrediction` : frozen dataclasses
    - `load_atom`, `load_atoms_from_dir`
    - `load_method_prediction`, `load_method_predictions_from_dir`
    - `EXPECTED_QUESTION_IDS`, `EXPECTED_SOURCES`
"""

from .atoms import (
    EXPECTED_QUESTION_IDS,
    EXPECTED_SOURCES,
    ExtractedAtom,
    MethodPrediction,
    load_atom,
    load_atoms_from_dir,
    load_method_prediction,
    load_method_predictions_from_dir,
)
from .extractor import (
    SOURCE_QUESTION_MAP,
    build_persona_extraction_request,
    extract_atom,
    load_sources_raw,
)
from .oracle_extractor import build_oracle_atom, build_oracle_atoms_for_seed
from .question_spec import QUESTION_TEXT, QUESTIONS, SOURCE_NAMES

__all__ = [
    "EXPECTED_QUESTION_IDS",
    "EXPECTED_SOURCES",
    "ExtractedAtom",
    "MethodPrediction",
    "QUESTIONS",
    "QUESTION_TEXT",
    "SOURCE_NAMES",
    "SOURCE_QUESTION_MAP",
    "build_persona_extraction_request",
    "build_oracle_atom",
    "build_oracle_atoms_for_seed",
    "extract_atom",
    "load_atom",
    "load_atoms_from_dir",
    "load_method_prediction",
    "load_method_predictions_from_dir",
    "load_sources_raw",
]
