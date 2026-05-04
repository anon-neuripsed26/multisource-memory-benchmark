"""13 core methods spanning T0 trivial, T1 single-source,
T2 fusion, T3 LLM, and T4 oracle. See ARCHITECTURE.md for the full inventory.

The public surface is the union of `__all__` declarations re-exported here;
there is no central registry — `__all__` IS the registry.
"""

from ._atom_adapter import atom_to_mu_q, iter_mu_q_per_question
from ._bias_model import bias_predict
from ._gt_loader import load_persona_gt
from .abf import ABF, ABFSelective
from .base import (
    SKIP_SENTINEL,
    GroundTruth,
    Method,
    MethodTrainingRecord,
    Prediction,
    TrainingRecord,
)
from .bcf import BCF
from .dsnbf import DSNBF, DSNBFSelective
from .llm_base import LLMSource, RawLLMOutput, normalize_to_prediction
from .llm_direct import LLMDirect, LLMDirectSelective
from .llm_few_shot import LLMFewShot, LLMFewShotSelective
from .llm_schema_aware import LLMSchemaAware, LLMSchemaAwareSelective
from .llm_sources import FrozenBulkJSONSource, FrozenFewShotDirSource, LiveSource, StructLLMSource
from .majority_class import MajorityClass
from .majority_vote import MajorityVote
from .nbf import NBF, NBFSelective
from .oracle import OracleExtraction
from .random_baseline import Random
from .ssb import SSB, SSBGlobal, SSBSelective

__all__ = [
    "atom_to_mu_q",
    "iter_mu_q_per_question",
    "bias_predict",
    "load_persona_gt",
    "SKIP_SENTINEL",
    "GroundTruth",
    "Method",
    "MethodTrainingRecord",
    "Prediction",
    "TrainingRecord",
    "Random",
    "MajorityClass",
    "SSB",
    "SSBGlobal",
    "SSBSelective",
    "MajorityVote",
    "BCF",
    "NBF",
    "NBFSelective",
    "DSNBF",
    "DSNBFSelective",
    "ABF",
    "ABFSelective",
    "OracleExtraction",
    "LLMSource",
    "RawLLMOutput",
    "normalize_to_prediction",
    "FrozenBulkJSONSource",
    "FrozenFewShotDirSource",
    "LiveSource",
    "StructLLMSource",
    "LLMDirect",
    "LLMDirectSelective",
    "LLMSchemaAware",
    "LLMSchemaAwareSelective",
    "LLMFewShot",
    "LLMFewShotSelective",
]
