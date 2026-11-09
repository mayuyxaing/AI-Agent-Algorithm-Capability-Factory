"""Data-driven contracts, graph provenance and matched-budget experiment tests."""
from copy import deepcopy
from pathlib import Path
import pytest
from forge.data import generate, load_split
from forge.knowledge import KnowledgeStore
from forge.llm import MockLLM, plugins, template_code
from forge.planning import profile_training, validate_spec, validate_plan
from forge.util import digest, read_json, write_json
from forge.validation import evaluate, load_policy
from forge.workflow import run

ROOT = Path(__file__).resolve().parents[1]


def setup_data(tmp_path):
    path = tmp_path / "data.csv"
    generate(path, rows=600)
    store = KnowledgeStore(tmp_path / "knowledge.sqlite")
    store.ingest(ROOT / "materials")
    return path, store, load_policy(ROOT / "configs/churn.json")


def test_profile_is_training_only_and_synthetic_variants_change_conditions(tmp_path):
    default, changed = tmp_path / "base.csv", tmp_path / "changed.csv"
    generate(default)
    assert digest(default.read_bytes()) == digest((ROOT / "data/churn.csv").read_bytes())
    generate(changed, missing_rate=.3, logit_shift=-1.2)
    train, valid, test = load_split(changed)
    profile = profile_training(train)
    assert profile["scope"] == "training_only" and profile["rows"] == len(train)
    assert all(.2 < column["missing_rate"] < .4 for column in profile["columns"].values())
    assert profile["minority_rate"] < profile_training(load_split(default)[0])["minority_rate"]
    assert "class_imbalance" in profile["requirements"]


def test_placeholder_and_omitted_explicit_constraints_are_rejected():
    spec = MockLLM().call("understand", {"description": "客户流失预测"})
    validate_spec(spec, "客户流失预测，处理缺失值和类别不平衡")
    with pytest.raises(ValueError, match="placeholders"):
        validate_spec(dict(spec, constraints=["strings"]), "客户流失预测")
    with pytest.raises(ValueError, match="Explicit requirement"):
        validate_spec(dict(spec, constraints=[]), "客户流失预测，处理缺失值")


def test_plan_hyperparameters_are_bounded_and_data_driven(tmp_path):
    data, store, _ = setup_data(tmp_path)
    profile = profile_training(load_split(data)[0])
    context = {"data_profile": profile, "knowledge": store.retrieve("客户流失"), "experiences": []}
    plan = MockLLM().call("plan", context)
    validate_plan(plan, plugins(), context["knowledge"], [])
    forest = next(c for c in plan["candidates"] if c["algorithm"] == "forest")
    assert forest["hyperparameters"]["max_depth"] == 5
    invalid = deepcopy(plan)
    invalid["candidates"][0]["hyperparameters"]["n_estimators"] = 1_000_000
    with pytest.raises(ValueError, match="bounds"):
        validate_plan(invalid, plugins(), context["knowledge"], [])
    invalid = deepcopy(plan)
    invalid["candidates"][0]["hyperparameters"]["class_weight"] = None
    with pytest.raises(ValueError, match="class-imbalance"):
        validate_plan(invalid, plugins(), context["knowledge"], [], constraints=["class_imbalance"])


def test_graph_retrieval_has_verifiable_two_hop_paths(tmp_path):
    _, store, _ = setup_data(tmp_path)
    retrieved = store.retrieve_grounded("客户流失未知类别", ["unknown_categories"])
    assert any(item["retrieval_paths"] for item in retrieved)
    path = tmp_path / "graph.json"
    store.export(path)
    graph = read_json(path)
    edges = {(e["source"], e["relation"], e["target"]) for e in graph["edges"]}
    for item in retrieved:
        for requirement, strategy, evidence in item["retrieval_paths"]:
            assert (requirement, "ADDRESSED_BY", strategy) in edges
            assert (strategy, "SUPPORTED_BY", evidence) in edges
            assert evidence == item["id"]


def test_worker_rejects_code_that_ignores_planned_hyperparameters(tmp_path):
    data, _, policy = setup_data(tmp_path)
    train, valid, _ = load_split(data)
    plan = MockLLM().call("plan", {"data_profile": profile_training(train), "knowledge": [], "experiences": []})
    candidate = next(c for c in plan["candidates"] if c["algorithm"] == "logistic")
    wrong = dict(candidate["hyperparameters"], C=2.0)
    code = tmp_path / "algorithm.py"
    code.write_text(template_code("logistic", hyperparameters=wrong, preprocessing=plan["preprocessing"]), encoding="utf-8")
    policy = dict(policy, _expected_plan={"algorithm": "logistic", "hyperparameters": candidate["hyperparameters"],
                                        "preprocessing": plan["preprocessing"]})
    result = evaluate(code, train, valid, policy, 42)
    assert not result["passed"] and "C must equal" in result["error"]


def test_v2_prompt_has_no_complete_template_and_host_fault_is_audited(tmp_path):
    data, store, policy = setup_data(tmp_path)
    contexts = []
    class Recording(MockLLM):
        def call(self, role, context):
            if role in {"generate", "repair"}:
                contexts.append((role, context))
                assert "interface_reference" not in context
                assert "interface_contract" in context and "data_profile" in context
                assert "model_expression" not in str(context)
            return super().call(role, context)
    folder, report = run("客户流失预测，比较模型，处理缺失值和未知类别", data, store, Recording(), policy,
                         tmp_path / "runs", repair_demo=True)
    assert report["status"] == "passed" and report["complete_template_supplied"] is False
    logistic = next(c for c in report["candidates"] if c["algorithm"] == "logistic")
    assert logistic["attempts"][0]["fault_origin"] == "host_unknown_category"
    assert logistic["attempts"][0]["checks"]["unknown_category"] is False
    assert logistic["attempts"][1]["passed"]
    assert (folder / "logistic/attempt_0/model_response.py").exists()
    assert any(role == "repair" and context["feedback"]["checks"]["unknown_category"] is False for role, context in contexts)


@pytest.mark.parametrize("arm", ["direct", "knowledge", "knowledge_repair"])
def test_equal_budget_resampling_does_not_receive_feedback(tmp_path, monkeypatch, arm):
    data, store, policy = setup_data(tmp_path)
    calls = []
    class FirstBad(MockLLM):
        def call(self, role, context):
            if role in {"generate", "repair"}:
                calls.append((role, context))
                if len(calls) == 1:
                    return {"code": "broken source", "explanation": "test failure"}
            return super().call(role, context)
    def evaluator(code, train, evaluation, policy, seed):
        passed = code.read_text(encoding="utf-8") != "broken source"
        return {"passed": passed, "checks": {"test_stub": passed}, "error": "synthetic test feedback" if not passed else "",
                "metrics": {"roc_auc": .8, "f1": .6, "accuracy": .7}}
    monkeypatch.setattr("forge.workflow.evaluate", evaluator)
    _, report = run("客户流失预测，比较模型", data, store, FirstBad(), policy, tmp_path / "runs",
                    experiment_arm=arm, resample_budget=3)
    assert report["status"] == "passed" and report["generation_budget_per_candidate"] == 3
    role, context = calls[1]
    if arm == "knowledge_repair":
        assert role == "repair" and context["previous_code"] and context["feedback"]
    else:
        assert role == "generate" and context["previous_code"] is None and context["feedback"] is None
    if arm == "direct":
        assert all(not context["knowledge"] and not context["experiences"] for _, context in calls)


def test_benchmark_pairs_data_and_resumes_completed_cells(tmp_path, monkeypatch):
    from forge.benchmark import benchmark
    calls = []
    def workflow(description, dataset, store, llm, policy, output, **kwargs):
        calls.append((kwargs["experiment_arm"], digest(dataset.read_bytes()), kwargs["resample_budget"]))
        output.mkdir(parents=True)
        write_json(output / "events.json", [])
        return output, {"status": "passed", "dataset_sha256": digest(dataset.read_bytes()), "candidates": [], "llm_calls": []}
    monkeypatch.setattr("forge.benchmark.run", workflow)
    policy = load_policy(ROOT / "configs/churn.json")
    output = tmp_path / "campaign"
    result = benchmark(MockLLM, ROOT / "materials", policy, output, rows=200)
    assert result["status"] == "complete" and len(calls) == 3
    assert len({item[1] for item in calls}) == 1 and all(item[2] == 3 for item in calls)
    benchmark(MockLLM, ROOT / "materials", policy, output, rows=200)
    assert len(calls) == 3
