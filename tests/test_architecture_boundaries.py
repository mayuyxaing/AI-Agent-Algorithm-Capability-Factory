"""Failure injection at host/model, transaction and artifact boundaries."""
from copy import deepcopy
from pathlib import Path
import sqlite3
import pytest
from forge.contracts import EvaluationPolicy, ImplementationPlan, ValidationContractError, ValidationOutcome
from forge.data import generate
from forge.knowledge import KnowledgeStore
from forge.llm import MockLLM
from forge.session import RunSession
from forge.util import read_json, write_json
from forge.validation import METRICS, load_policy
from forge.workflow import run

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def configured(tmp_path):
    dataset = tmp_path / "data.csv"
    generate(dataset, rows=200)
    store = KnowledgeStore(tmp_path / "knowledge.sqlite")
    store.ingest(ROOT / "materials")
    return dataset, store, load_policy(ROOT / "configs/churn.json")


def passing_evaluator(*args):
    return {"passed": True, "checks": {"contract": True},
            "metrics": {"roc_auc": .8, "f1": .6, "accuracy": .7}}


@pytest.mark.parametrize("broken", [
    {"passed": True, "checks": {}, "metrics": {"roc_auc": .8}},
    {"passed": True, "checks": {"reload": False}, "metrics": {"roc_auc": .8}},
    {"passed": "true", "checks": {"reload": True}, "metrics": {"roc_auc": .8}},
    {"passed": True, "checks": {"reload": True}, "metrics": {}},
    {"passed": True, "checks": {"reload": True}, "metrics": {"roc_auc": float("nan")}},
])
def test_inconsistent_evaluator_result_never_enters_selection(broken):
    with pytest.raises(ValidationContractError):
        ValidationOutcome.from_dict(broken, ["roc_auc"])


@pytest.mark.parametrize("broken", [True, float("nan"), float("inf"), "0.5"])
def test_non_numeric_or_nonfinite_policy_gate_is_rejected(broken):
    policy = load_policy(ROOT / "configs/churn.json")
    policy["minimum"]["roc_auc"] = broken
    with pytest.raises(ValueError):
        EvaluationPolicy.from_dict(policy, METRICS)


def test_adapter_programming_error_is_not_repaired(configured, tmp_path, monkeypatch):
    dataset, store, policy = configured
    class DefectiveAdapter(MockLLM):
        def call(self, role, context):
            if role == "generate":
                raise TypeError("adapter implementation defect")
            if role == "repair":
                pytest.fail("Programming defects must not consume model repair calls")
            return super().call(role, context)
    monkeypatch.setattr("forge.workflow.evaluate", passing_evaluator)
    folder, report = run("客户流失预测", dataset, store, DefectiveAdapter(), policy, tmp_path / "runs")
    assert report["status"] == "failed" and report["failure_kind"] == "internal"
    assert report["error_stage"] == "candidates"
    assert not any(e["role"] == "repair_response" for e in read_json(folder / "events.json"))


def test_invalid_host_result_stops_without_model_repair(configured, tmp_path, monkeypatch):
    dataset, store, policy = configured
    def contradictory(*args):
        return {"passed": True, "checks": {"unknown_category": False}, "metrics": {}}
    monkeypatch.setattr("forge.workflow.evaluate", contradictory)
    folder, report = run("客户流失预测", dataset, store, MockLLM(), policy, tmp_path / "runs")
    assert report["failure_kind"] == "internal" and "ValidationContractError" in report["error"]
    assert not (folder / "selected").exists()
    assert not any(e["role"] == "repair_response" for e in read_json(folder / "events.json"))


def test_transaction_rolls_back_even_if_nested_error_was_caught(tmp_path):
    store = KnowledgeStore(tmp_path / "knowledge.sqlite")
    with pytest.raises(RuntimeError, match="aborted"):
        with store.transaction():
            store.node("Knowledge", "before", {})
            try:
                with store.transaction():
                    store.node("Knowledge", "partial", {})
                    raise RuntimeError("fault injection")
            except RuntimeError:
                pass
    assert store.nodes() == []
    with store.transaction():
        store.node("Knowledge", "later", {})
    assert len(store.nodes()) == 1


def test_graph_import_failure_leaves_empty_store(tmp_path):
    store = KnowledgeStore(tmp_path / "knowledge.sqlite")
    graph = {"nodes": [{"id": "a", "kind": "Knowledge", "title": "valid"}],
             "edges": [{"source": "a", "relation": "SUPPORTED_BY", "target": "absent"}]}
    with pytest.raises(sqlite3.IntegrityError):
        store.import_graph(graph)
    assert store.nodes() == []


def test_snapshot_publication_failure_keeps_old_facts_and_failed_audit(tmp_path, monkeypatch):
    materials = tmp_path / "materials"
    materials.mkdir()
    path = materials / "business.md"
    path.write_text("## 未知类别\n旧策略", encoding="utf-8")
    store = KnowledgeStore(tmp_path / "knowledge.sqlite")
    store.ingest(materials)
    old = store.retrieve("未知类别")
    path.write_text("## 未知类别\n新策略", encoding="utf-8")
    def failed_index():
        raise RuntimeError("index publication failed")
    monkeypatch.setattr(store, "index_strategies", failed_index)
    with pytest.raises(RuntimeError, match="publication failed"):
        store.ingest(materials)
    assert store.retrieve("未知类别") == old
    assert store.latest_ingestion()["status"] == "publication_error"


def test_run_persistence_failure_rolls_back_graph(configured, tmp_path, monkeypatch):
    dataset, store, policy = configured
    original = store.nodes()
    original_edge = store.edge
    def broken_edge(source, relation, target):
        if relation == "EXECUTED_IN":
            raise RuntimeError("failed halfway through saving run")
        original_edge(source, relation, target)
    monkeypatch.setattr(store, "edge", broken_edge)
    monkeypatch.setattr("forge.workflow.evaluate", passing_evaluator)
    folder, report = run("客户流失预测", dataset, store, MockLLM(), policy, tmp_path / "runs")
    assert report["status"] == "persistence_failed"
    assert read_json(folder / "report.json")["failure_kind"] == "persistence"
    assert store.nodes() == original
    with store.connect() as con:
        assert con.execute("SELECT count(*) FROM runs").fetchone()[0] == 0
    assert not (folder / "knowledge_graph.json").exists()


def test_event_history_and_request_cannot_be_mutated_by_provider(tmp_path):
    class MutatingAdapter(MockLLM):
        def call(self, role, context):
            context["nested"]["value"] = "modified by provider"
            return {"code": "example source"}
    session = RunSession(tmp_path, "id", MutatingAdapter())
    context = {"nested": {"value": "original"}}
    response = session.ask("generate", context)
    response["code"] = "later change"
    context["nested"]["value"] = "later host change"
    assert session.events[0]["input"]["context"]["nested"]["value"] == "original"
    assert session.events[0]["output"]["code"] == "example source"


def test_atomic_json_write_preserves_previous_value_on_failure(tmp_path, monkeypatch):
    path = tmp_path / "report.json"
    write_json(path, {"status": "previous"})
    def fail_replace(*args):
        raise OSError("replace failed")
    monkeypatch.setattr(Path, "replace", fail_replace)
    with pytest.raises(OSError):
        write_json(path, {"status": "new"})
    assert read_json(path) == {"status": "previous"}
    assert sorted(p.name for p in tmp_path.iterdir()) == ["report.json"]


def test_plan_values_are_detached_from_model_response():
    response = {"candidates": [{"algorithm": "logistic", "rationale": "simple", "hyperparameters": {"C": 1}}],
                "preprocessing": {"scale_numeric": True}, "evidence_ids": [], "experience_ids": [], "experience_applied": False}
    original = deepcopy(response)
    plan = ImplementationPlan.from_dict(response)
    response["candidates"][0]["hyperparameters"]["C"] = 999
    response["preprocessing"]["scale_numeric"] = False
    assert plan.to_dict() == original
    detached = plan.to_dict()
    detached["candidates"][0]["hyperparameters"]["C"] = 12
    assert plan.to_dict() == original
