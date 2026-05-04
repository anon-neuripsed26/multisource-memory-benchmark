"""Tests for `api_clients.config_loader`."""

from __future__ import annotations

from pathlib import Path

import pytest

from survey2agent.api_clients.config_loader import (
    load_all_model_configs,
    load_model_config,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS_YAML = REPO_ROOT / "configs" / "models.yaml"


# ---------- happy path against real configs/models.yaml ----------

def test_load_real_models_yaml_has_four_entries() -> None:
    configs = load_all_model_configs(MODELS_YAML)
    assert set(configs.keys()) == {"gpt-5.4", "gemini-3.1-pro", "deepseek-v3.2", "qwen3-235b"}


def test_load_real_gpt_5_4() -> None:
    cfg = load_model_config(MODELS_YAML, "gpt-5.4")
    assert cfg.paper_alias == "GPT-5.4"
    assert cfg.provider == "openai"
    assert cfg.api_model_id == "gpt-5.4"
    assert cfg.default_params.get("reasoning_effort") == "xhigh"


def test_load_real_gemini() -> None:
    cfg = load_model_config(MODELS_YAML, "gemini-3.1-pro")
    assert cfg.provider == "google"
    assert cfg.api_model_id == "gemini-3.1-pro-preview"
    assert cfg.default_params.get("thinking_level") == "high"


def test_load_real_openrouter_models() -> None:
    deepseek = load_model_config(MODELS_YAML, "deepseek-v3.2")
    qwen = load_model_config(MODELS_YAML, "qwen3-235b")
    assert deepseek.provider == "openrouter"
    assert qwen.provider == "openrouter"
    assert deepseek.api_model_id == "deepseek-v3.2"
    assert qwen.api_model_id == "qwen3-235b-a22b-2507"


# ---------- error paths ----------

def test_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_all_model_configs(tmp_path / "nope.yaml")


def test_missing_key(tmp_path: Path) -> None:
    yaml = tmp_path / "m.yaml"
    yaml.write_text("models:\n  a:\n    paper_alias: A\n    provider: x\n    api_model_id: a\n")
    with pytest.raises(KeyError, match="not found"):
        load_model_config(yaml, "nonexistent")


def test_missing_required_field(tmp_path: Path) -> None:
    yaml = tmp_path / "m.yaml"
    yaml.write_text("models:\n  a:\n    provider: x\n    api_model_id: a\n")
    with pytest.raises(ValueError, match="paper_alias"):
        load_all_model_configs(yaml)


# ---------- forward-compatibility ----------

def test_unrecognized_field_emits_warning_and_loads(tmp_path: Path) -> None:
    """A YAML file with a stale `pricing_usd_per_million_tokens` block must
    load successfully and emit a `UserWarning` rather than raising."""
    yaml = tmp_path / "m.yaml"
    yaml.write_text(
        "models:\n  a:\n    paper_alias: A\n    provider: x\n    api_model_id: a\n"
        "    pricing_usd_per_million_tokens:\n      input: 1.0\n      output: 2.0\n"
    )
    with pytest.warns(UserWarning, match="pricing_usd_per_million_tokens"):
        cfg = load_model_config(yaml, "a")
    assert cfg.paper_alias == "A"
    assert cfg.api_model_id == "a"


# ---------- default_params is read-only ----------

def test_default_params_is_read_only(tmp_path: Path) -> None:
    yaml = tmp_path / "m.yaml"
    yaml.write_text(
        "models:\n  a:\n    paper_alias: A\n    provider: x\n    api_model_id: a\n"
        "    default_params:\n      temperature: 0.0\n      reasoning_effort: high\n"
    )
    cfg = load_model_config(yaml, "a")
    assert cfg.default_params["temperature"] == 0.0
    assert cfg.default_params["reasoning_effort"] == "high"
    with pytest.raises(TypeError):
        cfg.default_params["temperature"] = 0.7  # type: ignore[index]
