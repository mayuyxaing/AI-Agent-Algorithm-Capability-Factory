"""Meaningful contract, failure, repair, provenance and holdout tests."""
import json
from pathlib import Path
import subprocess
from urllib.error import URLError
import numpy as np
import pytest
from forge.data import generate, load_split
from forge.knowledge import KnowledgeStore
from forge.llm import MockLLM, OllamaLLM, template_code
from forge.safety import inspect_code
from forge.util import read_json, write_json
from forge.validation import evaluate, load_policy
from forge.workflow import run

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def setup(tmp_path):
    data = tmp_path / "churn.csv"
    generate(data, rows=600)
    store = KnowledgeStore(tmp_path / "knowledge.sqlite")
    store.ingest(ROOT / "materials")
    return data, store, load_policy(ROOT / "configs/churn.json")


def test_extract_idempotency_and_source_quotes(setup):
    _, store, _ = setup
    original = store.nodes()
    store.ingest(ROOT / "materials")
    assert store.nodes() == original
    functions = store.nodes("Function")
    assert any("sklearn.preprocessing" in n["dependencies"] for n in functions)
    for node in store.nodes("Knowledge"):
        assert node["evidence_quote"] in (ROOT / "materials" / node["source_path"]).read_text(encoding="utf-8")
    assert store.retrieve("客户流失预测未知类别")


def test_split_disjoint_and_target_validation(setup):
    data, _, _ = setup
    train, valid, test = load_split(data)
    ids = [set(zip(frame.tenure.fillna(-1), frame.monthly_charges.fillna(-1))) for frame in (train, valid, test)]
    assert not ids[0] & ids[1] and not ids[0] & ids[2] and not ids[1] & ids[2]
    import pandas as pd
    broken = pd.read_csv(data)
    broken.loc[0, "churn"] = np.nan
    broken.to_csv(data, index=False)
    with pytest.raises(ValueError, match="binary"):
        load_split(data)


@pytest.mark.parametrize("code", ["import os\ndef build_model(): pass\ndef predict(): pass",
                                      "def build_model(): return open('x')\ndef predict(): pass",
                                      "def build_model(): return eval('1')\ndef predict(): pass",
                                      "def build_model(): return x.__class__\ndef predict(): pass"])
def test_static_rejections(code):
    assert inspect_code(code)


def test_templates_and_injected_failure(setup, tmp_path):
    data, _, policy = setup
    train, valid, _ = load_split(data)
    bad = tmp_path / "bad" / "algorithm.py"
    bad.parent.mkdir()
    bad.write_text(template_code("logistic", broken=True), encoding="utf-8")
    failed = evaluate(bad, train, valid, policy, 42)
    assert not failed["passed"] and failed["checks"]["unknown_category"] is False
    good = tmp_path / "good" / "algorithm.py"
    good.parent.mkdir()
    good.write_text(template_code("logistic"), encoding="utf-8")
    passed = evaluate(good, train, valid, policy, 42)
    assert passed["passed"]
    payload = read_json(good.parent / "input.json")
    assert all("churn" not in row for row in payload["evaluation"])


def test_timeout_is_failure(setup, tmp_path, monkeypatch):
    data, _, policy = setup
    train, valid, _ = load_split(data)
    code = tmp_path / "algorithm.py"
    code.write_text(template_code("logistic"), encoding="utf-8")
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired("worker", 1)
    monkeypatch.setattr(subprocess, "run", timeout)
    result = evaluate(code, train, valid, policy, 42)
    assert not result["passed"] and result["failure_kind"] == "timeout"


def test_end_to_end_repair_and_reuse(setup, tmp_path):
    data, store, policy = setup
    folder, first = run("客户流失预测，比较模型，处理未知类别", data, store, MockLLM(), policy, tmp_path / "runs", inject_fault=True)
    assert first["status"] == "passed"
    logistic = next(c for c in first["candidates"] if c["algorithm"] == "logistic")
    assert len(logistic["attempts"]) == 2
    assert logistic["attempts"][0]["injected"] and logistic["attempts"][1]["passed"]
    assert first["test_result"]["passed"]
    graph = read_json(folder / "knowledge_graph.json")
    assert {"Source", "Capability", "AlgorithmVersion", "ValidationRun", "FailureCase", "Environment", "Metric"} <= {n["kind"] for n in graph["nodes"]}
    run_node = next(n for n in graph["nodes"] if n["kind"] == "ValidationRun")
    assert run_node["test_metrics"] == first["test_result"]["metrics"]
    assert all("validation_metrics" in n for n in graph["nodes"] if n["kind"] == "AlgorithmVersion")
    _, second = run("客户流失预测，处理未知类别", data, store, MockLLM(), policy, tmp_path / "runs")
    assert second["status"] == "passed" and second["experiences"]
    assert second["plan"]["experience_applied"]
    assert second["plan"]["experience_ids"]
    assert (folder / "selected/model.joblib").exists()


def test_repair_budget_and_no_test_for_failed_candidates(setup, tmp_path):
    class AlwaysBroken(MockLLM):
        def call(self, role, context):
            if role in {"generate", "repair"}:
                return {"code": "syntax error !", "explanation": "test provider"}
            return super().call(role, context)
    data, store, policy = setup
    folder, report = run("客户流失预测", data, store, AlwaysBroken(), policy, tmp_path / "runs")
    assert report["status"] == "no_valid_candidate" and "test_result" not in report
    assert all(len(c["attempts"]) == policy["max_repairs"] + 1 for c in report["candidates"])
    assert (folder / "report.md").exists()


def test_unrelated_task_has_failure_report(setup, tmp_path):
    data, store, policy = setup
    folder, report = run("生成图像", data, store, MockLLM(), policy, tmp_path / "runs")
    assert report["status"] == "failed"
    assert (folder / "report.json").exists()


def test_policy_rejects_unknown_metric(tmp_path):
    policy = read_json(ROOT / "configs/churn.json")
    policy["metrics"].append("imaginary")
    path = tmp_path / "policy.json"
    write_json(path, policy)
    with pytest.raises(ValueError):
        load_policy(path)


def test_ollama_wire_contract_and_no_silent_fallback(monkeypatch):
    class Response:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def read(self, limit):
            return json.dumps({"message": {"content": '{"task":"churn_classification"}'}}).encode()
    def fake(request, timeout):
        body = json.loads(request.data)
        assert request.full_url == "http://localhost:11434/api/chat"
        assert body["stream"] is False and body["format"] == "json"
        assert body["model"] == "test-model"
        return Response()
    monkeypatch.setattr("forge.llm.urlopen", fake)
    llm = OllamaLLM("test-model")
    assert llm.call("understand", {"description": "流失"})["task"] == "churn_classification"
    def fail(*args, **kwargs): raise URLError("unavailable")
    monkeypatch.setattr("forge.llm.urlopen", fail)
    with pytest.raises(RuntimeError, match="Ollama"):
        llm.call("understand", {})


def test_final_holdout_failure_does_not_trigger_repair_or_reselection(setup, tmp_path, monkeypatch):
    data, store, policy = setup
    calls = []
    def controlled(code, train, evaluation, policy, seed):
        calls.append(code.parent.name)
        if code.parent.name == "selected":
            return {"passed": False, "checks": {"minimum_roc_auc": False}, "error": "holdout failed"}
        return {"passed": True, "checks": {"contract": True}, "metrics": {"roc_auc": .8, "f1": .6, "accuracy": .7}}
    monkeypatch.setattr("forge.workflow.evaluate", controlled)
    folder, report = run("客户流失预测", data, store, MockLLM(), policy, tmp_path / "runs")
    assert report["status"] == "test_failed"
    assert calls == ["attempt_0", "attempt_0", "selected"]
    events = read_json(folder / "events.json")
    assert not any(event["role"] == "repair" for event in events)


def test_invalid_llm_quote_is_rejected(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "business.md").write_text("## 客户流失\n使用逻辑回归。", encoding="utf-8")
    class FabricatingLLM:
        mode = "ollama"
        def call(self, role, context):
            return {"facts": [{"title": "fabricated", "text": "false", "evidence_quote": "never in source"}]}
    store = KnowledgeStore(tmp_path / "kb.sqlite")
    with pytest.raises(ValueError, match="verifiable"):
        store.ingest(source, FabricatingLLM())
    assert not any(n["title"] == "fabricated" for n in store.nodes())


def test_invalid_probability_shape_is_rejected(setup, tmp_path):
    data, _, policy = setup
    train, valid, _ = load_split(data)
    code = tmp_path / "algorithm.py"
    code.write_text(template_code("logistic") + "\n\ndef predict(model, frame):\n    return [0.5]\n", encoding="utf-8")
    result = evaluate(code, train, valid, policy, 42)
    assert not result["passed"] and result["failure_kind"] == "execution"
    assert "probability vector" in result["error"]
