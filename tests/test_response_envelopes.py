"""The same bounded response envelope policy applies before every structured stage."""
from pathlib import Path
import pytest
from forge.contracts import TaskSpecification
from forge.data import generate
from forge.knowledge import KnowledgeStore
from forge.llm import MockLLM
from forge.structured import checked_response, structured_payload
from forge.util import read_json
from forge.validation import load_policy
from forge.workflow import run

ROOT = Path(__file__).resolve().parents[1]
SPEC = {"task": "churn_classification", "target": "churn", "metrics": ["roc_auc", "f1"],
        "constraints": ["missing_values", "unknown_categories"], "description": "客户流失预测"}


@pytest.mark.parametrize("role", ["understand", "plan", "extract"])
def test_sole_answer_envelope_is_audited_without_extra_model_calls(role):
    events, calls = [], []
    raw = {"answer": SPEC}
    def ask(role, request):
        calls.append(request)
        return raw
    result = checked_response(role, {}, ask, TaskSpecification.from_dict,
                              lambda *event: events.append(event))
    assert result is SPEC and len(calls) == 1
    assert events[0][0] == role + "_normalized"
    assert events[0][2]["response"] is raw
    assert events[0][2]["fields_edited"] is False


@pytest.mark.parametrize("raw", [
    {"answer": {"answer": SPEC}}, {"answer": SPEC, "task": "conflicting"},
    {"answer": SPEC, "extra": "ambiguous"}, {"answer": None},
])
def test_ambiguous_or_recursive_envelopes_do_not_bypass_validation(raw):
    calls = []
    def ask(role, request):
        calls.append(request)
        return raw
    with pytest.raises(ValueError, match="after 2 responses"):
        checked_response("understand", {}, ask, TaskSpecification.from_dict, lambda *e: None)
    assert len(calls) == 2
    assert calls[1]["previous_response"] == raw


def test_unwrapping_does_not_invent_missing_task_fields():
    raw = {"answer": {"description": "customer churn"}}
    assert structured_payload(raw) == {"description": "customer churn"}
    with pytest.raises(ValueError, match="requires task"):
        TaskSpecification.from_dict(structured_payload(raw))


def test_wrapped_understanding_and_plan_complete_the_same_workflow(tmp_path, monkeypatch):
    data = tmp_path / "data.csv"
    generate(data, rows=200)
    store = KnowledgeStore(tmp_path / "knowledge.sqlite")
    store.ingest(ROOT / "materials")
    class WrappedResponses(MockLLM):
        def call(self, role, context):
            if role == "understand":
                assert "task" in context["output_schema"]["required"]
            result = super().call(role, context)
            return {"answer": result} if role in {"understand", "plan"} else result
    def evaluate(*args):
        return {"passed": True, "checks": {"contract": True}, "metrics": {"roc_auc": .8, "f1": .6, "accuracy": .7}}
    monkeypatch.setattr("forge.workflow.evaluate", evaluate)
    folder, report = run("客户流失预测，比较算法", data, store, WrappedResponses(),
                         load_policy(ROOT / "configs/churn.json"), tmp_path / "runs")
    assert report["status"] == "passed"
    assert report["structured_corrections"] == {"understand": 0, "plan": 0}
    events = read_json(folder / "events.json")
    assert {"understand_normalized", "plan_normalized"} <= {e["role"] for e in events}


def test_archived_pilot_understanding_is_readable_without_field_edits():
    from forge.tasks import get_task
    root = ROOT / "examples/real-benchmark/pilot-v3"
    manifest = read_json(root / "manifest.json")
    checked = 0
    for arm in ("direct", "knowledge"):
        path = root / manifest["results"]["baseline-42-" + arm]["report"]
        report = read_json(path)
        assert report["status"] == "failed"  # Preserve the historical outcome.
        for event in read_json(path.parent / "events.json"):
            if event["role"] == "understand_response":
                raw = event["output"]
                payload = structured_payload(raw)
                assert payload is raw["answer"]
                TaskSpecification.from_dict(payload)
                get_task().validate_spec(payload, report["description"])
                checked += 1
    assert checked == 4
