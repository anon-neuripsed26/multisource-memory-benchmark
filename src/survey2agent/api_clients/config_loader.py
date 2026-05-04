"""Loader for `configs/models.yaml` → `ModelConfig` instances.

The YAML schema is documented in `configs/models.yaml`. Each top-level entry
under `models:` is keyed by an internal name (e.g. "gpt-5.4") that we use as
the lookup key. The `paper_alias` field is the display name used in the paper.

Public API:
    - load_model_config(yaml_path, key) -> ModelConfig
    - load_all_model_configs(yaml_path) -> dict[str, ModelConfig]
"""

from __future__ import annotations

import warnings
from pathlib import Path
from types import MappingProxyType
from typing import Any

import yaml

from .types import ModelConfig


_REQUIRED_FIELDS: tuple[str, ...] = ("paper_alias", "provider", "api_model_id")
_OPTIONAL_FIELDS: tuple[str, ...] = (
    "api_endpoint",
    "default_params",
    "used_for",
)
_RECOGNIZED_FIELDS: frozenset[str] = frozenset(_REQUIRED_FIELDS + _OPTIONAL_FIELDS)


def load_all_model_configs(yaml_path: str | Path) -> dict[str, ModelConfig]:
    """Parse the entire models.yaml file. Returns dict keyed by yaml entry name."""
    path = Path(yaml_path)
    if not path.exists():
        raise FileNotFoundError(f"models.yaml not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict) or "models" not in raw:
        raise ValueError(f"{path}: top-level must be a dict containing 'models:'")
    models_section = raw["models"]
    if not isinstance(models_section, dict):
        raise ValueError(f"{path}: 'models:' must be a dict, got {type(models_section).__name__}")

    out: dict[str, ModelConfig] = {}
    for key, entry in models_section.items():
        out[key] = _parse_entry(key, entry, source=str(path))
    return out


def load_model_config(yaml_path: str | Path, key: str) -> ModelConfig:
    """Parse models.yaml and return a single `ModelConfig` by yaml key.

    `key` is the yaml entry name (e.g. "gpt-5.4"), NOT the `paper_alias`.
    """
    all_configs = load_all_model_configs(yaml_path)
    if key not in all_configs:
        available = sorted(all_configs.keys())
        raise KeyError(
            f"model key {key!r} not found in {yaml_path}. Available keys: {available}"
        )
    return all_configs[key]


def _parse_entry(key: str, entry: Any, *, source: str) -> ModelConfig:
    if not isinstance(entry, dict):
        raise ValueError(f"{source}: entry {key!r} must be a dict, got {type(entry).__name__}")

    unknown = set(entry.keys()) - _RECOGNIZED_FIELDS
    for field_name in sorted(unknown):
        warnings.warn(
            f"{source}: entry {key!r}: ignoring unrecognized model config "
            f"field {field_name!r}",
            UserWarning,
            stacklevel=2,
        )

    for field_name in _REQUIRED_FIELDS:
        if field_name not in entry or entry[field_name] in (None, ""):
            raise ValueError(f"{source}: entry {key!r} missing required field {field_name!r}")

    paper_alias = str(entry["paper_alias"])
    provider = str(entry["provider"])
    api_model_id = str(entry["api_model_id"])
    api_endpoint = entry.get("api_endpoint")
    if api_endpoint is not None:
        api_endpoint = str(api_endpoint)

    default_params = entry.get("default_params") or {}
    if not isinstance(default_params, dict):
        raise ValueError(
            f"{source}: entry {key!r} default_params must be a dict, "
            f"got {type(default_params).__name__}"
        )

    return ModelConfig(
        paper_alias=paper_alias,
        provider=provider,
        api_model_id=api_model_id,
        api_endpoint=api_endpoint,
        default_params=MappingProxyType(dict(default_params)),
    )
