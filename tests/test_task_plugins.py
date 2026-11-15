"""Cross-task execution, extension hooks and infrastructure-fault attribution."""
from copy import deepcopy
from pathlib import Path
import json
import pytest
from forge.tasks import get_task, register_task, load_installed_tasks, TASKS
from forge.llm import MockLLM, plugins
from forge.knowledge import KnowledgeStore
from forge.validation import METRICS, register_metric, load_policy, evaluate
from forge.workflow import run
from forge.failures import analyze_failure
from forge.util import read_json

ROOT = Path(__file__).resolve().parents[1]


def test_text_pipeline_uses_same_workflow_and_persists_custom_transformer(tmp_path):
    task = get_task("text_classification")
    data = tmp_path / "tickets.csv"
    task.generate(data, rows=400)
    store = KnowledgeStore(tmp_path / "kb.sqlite")
    store.ingest(ROOT / "materials")
    folder, report = run("客服工单文本分类，比较模型", data, store, MockLLM(), load_policy(ROOT / task.config), tmp_path / "runs")
    assert report["status"] == "passed" and report["task_id"] == task.id
    assert {c["algorithm"] for c in report["candidates"]} == {"text_logistic", "text_nb"}
    assert report["test_result"]["checks"]["model_reload"]
    assert report["test_result"]["checks"]["unseen_vocabulary"]
    graph = read_json(folder / "knowledge_graph.json")
    capability = next(n for n in graph["nodes"] if n["id"] == "capability:text_classification")
    assert capability["input"]["required_columns"] == ["text"]


def test_task_registry_rejects_incomplete_plugins():
    class Broken:
        id = "incomplete"
    with pytest.raises(ValueError, match="missing"):
        register_task(Broken())


def test_algorithm_extension_loads_metadata_without_worker_edits(monkeypatch):
    descriptor = deepcopy(plugins("text_classification")["text_nb"])
    descriptor["id"] = "custom_nb"
    class Entry:
        def load(self): return lambda: descriptor
    monkeypatch.setattr("importlib.metadata.entry_points", lambda group: [Entry()] if group == "capability_forge.algorithms" else [])
    loaded = plugins("text_classification")
    assert loaded["custom_nb"]["estimator"] == "MultinomialNB"
    assert "custom_nb" in {p["id"] for p in get_task("text_classification").catalog(loaded)}


def test_metric_registration_is_validated_by_existing_configuration(tmp_path, monkeypatch):
    monkeypatch.setitem(METRICS, "precision_example", lambda y, p: .5)
    policy = read_json(ROOT / "configs/text.json")
    policy["metrics"].append("precision_example")
    path = tmp_path / "config.json"
    path.write_text(json.dumps(policy), encoding="utf-8")
    assert "precision_example" in load_policy(path)["metrics"]
    with pytest.raises(ValueError):
        register_metric("precision_example", lambda y, p: .5)


def test_validator_error_is_not_reused_as_algorithm_knowledge():
    item = {"error": "PicklingError: Can't pickle generated.FeatureValidator: No module named 'generated'"}
    result = analyze_failure(item)
    assert result["category"] == "validator_loader" and result["reusable"] is False
    assert analyze_failure({"error": "Found unknown categories"})["reusable"] is True


def test_no_wrong_task_history_leaks_into_text_retrieval(tmp_path):
    store = KnowledgeStore(tmp_path / "kb.sqlite")
    store.node("FailureCase", "churn issue", {"task": "churn_classification", "reusable": True})
    store.node("FailureCase", "text issue", {"task": "text_classification", "reusable": True})
    assert [n["title"] for n in store.experiences("text_classification")] == ["text issue"]


def test_installed_task_entrypoint_is_discovered(monkeypatch):
    import forge.tasks as registry
    class External(registry.TextTask):
        id = "external_text_task"
    class Entry:
        def load(self): return External
    monkeypatch.setattr(registry, "TASKS", {})
    monkeypatch.setattr("importlib.metadata.entry_points", lambda group: [Entry()] if group == "capability_forge.tasks" else [])
    registry.load_installed_tasks()
    assert registry.get_task("external_text_task").features == ["text"]


def test_installed_metric_entrypoint_is_discovered(monkeypatch):
    import forge.validation as registry
    class Entry:
        def load(self): return lambda: {"external_score": lambda y, p: .75}
    monkeypatch.setattr(registry, "METRICS", dict(METRICS))
    monkeypatch.setattr("importlib.metadata.entry_points", lambda group: [Entry()] if group == "capability_forge.metrics" else [])
    registry.load_installed_metrics()
    assert registry.METRICS["external_score"]([0], [.1]) == .75
