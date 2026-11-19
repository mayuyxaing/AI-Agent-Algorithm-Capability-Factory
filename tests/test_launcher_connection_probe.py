"""A responding model with bad JSON is not mislabeled as a TLS outage."""
import importlib.util
from pathlib import Path
import pytest
from forge.llm import LLMProviderError
from forge.util import read_json


@pytest.mark.parametrize("case", ["wrapped", "invalid_contract", "provider_error"])
def test_connection_probe_distinguishes_transport_and_contract(tmp_path, monkeypatch, case):
    script = Path(__file__).resolve().parents[1] / "scripts/run_glm.py"
    spec = importlib.util.spec_from_file_location("probe_launcher_test", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.chdir(tmp_path)
    class Fake:
        mode, model = "glm", "glm-4.7-flash"
        calls = []
        def call(self, role, context):
            if case == "provider_error":
                raise LLMProviderError("TLS failed")
            if case == "invalid_contract":
                return {"wrong": "shape"}
            return {"answer": {"task": "churn_classification", "target": "churn", "metrics": ["roc_auc"],
                               "constraints": [], "description": context["description"]}}
    monkeypatch.setattr("forge.llm.GLMLLM", Fake)
    code = module.probe_connection()
    result = read_json(tmp_path / "runtime/glm/connection-check.json")
    assert result["connected"] == (case != "provider_error")
    assert result["task_contract_passed"] == (case == "wrapped")
    assert code == {"wrapped": 0, "invalid_contract": 1, "provider_error": 2}[case]
    if case == "wrapped":
        assert result["answer_envelope_removed"] is True
