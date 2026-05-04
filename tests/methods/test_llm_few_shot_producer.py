import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from survey2agent.api_clients import CompletionRequest, SyncLLMClient, CompletionResult, BatchResultItem
from survey2agent.methods._llm_prompt_builders import FEW_SHOT_QIDS, build_few_shot_request
from survey2agent.methods.llm_batch_runner import run_few_shot_batch, LLMFewShotBatchReport

@pytest.fixture
def mock_configs(tmp_path):
    configs_root = tmp_path / "configs"
    configs_root.mkdir()
    (configs_root / "AGENTS.md").write_text("System Info", encoding="utf-8")
    (configs_root / "specs").mkdir()
    (configs_root / "specs/output-rules.md").write_text("Output Rules", encoding="utf-8")
    (configs_root / "exemplars").mkdir()
    for qid in FEW_SHOT_QIDS:
        (configs_root / f"exemplars/{qid}.md").write_text(f"Exemplar for {qid}", encoding="utf-8")
    return configs_root

@pytest.fixture
def mock_persona(tmp_path):
    persona_dir = tmp_path / "persona"
    persona_dir.mkdir()
    (persona_dir / "structural_sources").mkdir()
    return persona_dir

def test_few_shot_qids_count():
    assert len(FEW_SHOT_QIDS) == 18
    assert "A1" in FEW_SHOT_QIDS
    assert "B1" not in FEW_SHOT_QIDS

def test_build_few_shot_request_includes_exemplar_and_render(mock_persona, mock_configs):
    req = build_few_shot_request(mock_persona, "A1", mock_configs)
    assert isinstance(req, CompletionRequest)
    assert "System Info" in req.system_prompt
    assert "Output Rules" in req.system_prompt
    assert "Exemplar for A1" in req.user_prompt
    assert "A1" in req.user_prompt

def test_run_few_shot_batch_writes_one_json_per_persona_qid(tmp_path, mock_configs):
    output_dir = tmp_path / "output"
    persona_id = "p1"
    persona_dir = tmp_path / "persona_p1"
    persona_dir.mkdir()
    (persona_dir / "structural_sources").mkdir()
    
    mock_client = MagicMock(spec=SyncLLMClient)
    # Return a valid JSON response
    mock_client.complete.return_value = CompletionResult(
        text='{"answer": "label1", "would_skip": false}',
        finish_reason="stop",
        model_id="test-model",
        provider="test-provider",
        cache_hit=False
    )
    
    report = run_few_shot_batch(
        persona_ids=[persona_id],
        persona_dirs={persona_id: persona_dir},
        qids=["A1", "A2"],
        client=mock_client,
        output_dir=output_dir,
        configs_root=mock_configs,
        allow_api_call=True
    )
    
    assert isinstance(report, LLMFewShotBatchReport)
    assert report.n_total == 2
    assert report.n_success == 2
    
    files = list(output_dir.glob("*.json"))
    # persona_id__qid.json (2) and _failures.json (1)
    assert len(files) == 3 
    
    a1_file = output_dir / f"{persona_id}__A1.json"
    assert a1_file.exists()
    data = json.loads(a1_file.read_text())
    assert data["persona"] == persona_id
    assert data["question"] == "A1"
    assert data["answer"] == "label1"
    assert data["would_skip"] is False
