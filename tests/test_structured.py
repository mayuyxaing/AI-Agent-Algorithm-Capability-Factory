"""Malformed model plans get one audited correction, never a fabricated fallback."""
from pathlib import Path
import pytest
from forge.structured import checked_response
from forge.llm import MockLLM, plugins
from forge.planning import validate_plan, plan_schema, profile_training
from forge.data import generate, load_split
from forge.structured import generation_payload


def test_generation_answer_wrapper_preserves_source_exactly():
    source = '# 原始代码\ndef build_model(seed=42):\n    return None\n'
    response = {"answer": {"code": source, "explanation": "original"}}
    assert generation_payload(response) is response["answer"]
    assert generation_payload(response)["code"] == source
    assert generation_payload(response["answer"]) is response["answer"]
    assert generation_payload({"answer": {"code": source}, "explanation": "outer"}) == {"code": source, "explanation": "outer"}


@pytest.mark.parametrize("response", [{}, {"answer": {"code": ""}}, {"code": 123},
                                     {"answer": {"code": "x"}, "code": None},
                                     {"answer": {"answer": {"code": "x"}}}])
def test_generation_ambiguous_or_invalid_payload_is_rejected(response):
    with pytest.raises(ValueError, match="Expected"):
        generation_payload(response)


@pytest.mark.parametrize("bad", [{}, {"candidates": {}}, {"plan": {"candidates": []}}, {"candidates": ["logistic"]}])
def test_bad_shapes_raise_actionable_value_error(bad):
    with pytest.raises(ValueError):
        validate_plan(bad, plugins(), [], [])


def test_invalid_plan_is_preserved_then_corrected(tmp_path):
    data = tmp_path / "data.csv"
    generate(data, rows=200)
    context = {"data_profile": profile_training(load_split(data)[0]), "knowledge": [], "experiences": [], "output_schema": plan_schema()}
    valid = MockLLM().call("plan", context)
    calls, events = [], []
    bad = {"plan": {"candidates": []}}
    def ask(role, incoming):
        calls.append(incoming)
        if len(calls) == 1:
            return bad
        assert incoming["previous_response"] == bad
        assert "JSON array" in incoming["validation_error"]
        assert incoming["output_schema"] == plan_schema()
        return valid
    def record(role, incoming, outgoing): events.append((role, incoming, outgoing))
    result = checked_response("plan", context, ask, lambda value: validate_plan(value, plugins(), [], []), record)
    assert result == valid and len(calls) == 2
    assert events[0][0] == "plan_rejected" and events[0][2]["response"] == bad


def test_persistent_bad_plan_stops_after_one_correction():
    calls = []
    def ask(role, incoming):
        calls.append(incoming)
        return {}
    with pytest.raises(ValueError, match="after 2 responses"):
        checked_response("plan", {"output_schema": plan_schema()}, ask,
                         lambda value: validate_plan(value, plugins(), [], []), lambda *args: None)
    assert len(calls) == 2


def test_workflow_audits_bad_response_and_caches_only_corrected_plan(tmp_path, monkeypatch):
    from forge.workflow import run
    from forge.knowledge import KnowledgeStore
    from forge.validation import load_policy
    from forge.util import read_json
    root = Path(__file__).resolve().parents[1]
    data = tmp_path / "data.csv"
    generate(data, rows=200)
    store = KnowledgeStore(tmp_path / "kb.sqlite")
    store.ingest(root / "materials")
    class BadThenGood(MockLLM):
        plan_calls = 0
        def call(self, role, context):
            if role == "plan":
                self.plan_calls += 1
                if self.plan_calls == 1:
                    return {"wrong_root": "preserve this response"}
                assert context["previous_response"]["wrong_root"] == "preserve this response"
            return super().call(role, context)
    def evaluate(*args, **kwargs):
        return {"passed": True, "checks": {"test_stub": True}, "metrics": {"roc_auc": .8, "f1": .6, "accuracy": .7}}
    monkeypatch.setattr("forge.workflow.evaluate", evaluate)
    llm = BadThenGood()
    folder, report = run("客户流失预测，比较算法", data, store, llm, load_policy(root / "configs/churn.json"),
                         tmp_path / "runs", checkpoint_dir=tmp_path / "checkpoints")
    assert report["status"] == "passed" and llm.plan_calls == 2
    assert report["structured_corrections"]["plan"] == 1
    events = read_json(folder / "events.json")
    responses = [e for e in events if e["role"] == "plan_response"]
    assert len(responses) == 2 and responses[0]["output"]["wrong_root"] == "preserve this response"
    cached = [read_json(p) for p in (tmp_path / "checkpoints").glob("*.json")]
    plans = [item for item in cached if item["role"] == "plan"]
    assert len(plans) == 1 and isinstance(plans[0]["output"]["candidates"], list)


@pytest.mark.parametrize("malformed_json", [False, True])
def test_generation_contract_feedback_then_wrapped_code_is_evaluated(tmp_path, monkeypatch, malformed_json):
    from forge.workflow import run
    from forge.knowledge import KnowledgeStore
    from forge.validation import load_policy
    from forge.util import read_json
    root = Path(__file__).resolve().parents[1]
    data = tmp_path / "data.csv"
    generate(data, rows=200)
    store = KnowledgeStore(tmp_path / "kb.sqlite")
    store.ingest(root / "materials")

    class BadThenWrapped(MockLLM):
        def call(self, role, context):
            if role == "generate":
                assert context["output_schema"]["required"] == ["code", "explanation"]
                if malformed_json:
                    from forge.llm import LLMResponseFormatError
                    raise LLMResponseFormatError('{"code":"unterminated')
                return {"unexpected": "no source"}
            if role == "repair":
                assert context["feedback"]["checks"] == {"generation_contract": False}
                if malformed_json:
                    assert context["previous_response"]["raw_content"] == '{"code":"unterminated'
                else:
                    assert context["previous_response"] == {"unexpected": "no source"}
                return {"answer": super().call(role, context)}
            return super().call(role, context)

    evaluated = []
    def evaluate(path, *args):
        assert path.read_text(encoding="utf8").strip()
        evaluated.append(path)
        return {"passed": True, "checks": {"test_stub": True}, "metrics": {"roc_auc": .8, "f1": .6, "accuracy": .7}}

    monkeypatch.setattr("forge.workflow.evaluate", evaluate)
    folder, report = run("客户流失预测，比较算法", data, store, BadThenWrapped(),
                         load_policy(root / "configs/churn.json"), tmp_path / "runs")
    assert report["status"] == "passed" and len(evaluated) == 3
    assert all(c["attempts"][0]["checks"] == {"generation_contract": False} for c in report["candidates"])
    events = read_json(folder / "events.json")
    assert len([e for e in events if e["role"] == "generation_normalized"]) == 2
    assert all(e["output"]["source_edited"] is False for e in events if e["role"] == "generation_normalized")
