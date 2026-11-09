"""Checkpoints reduce calls without caching or bypassing validation."""
import importlib.util
from pathlib import Path
import pytest
from forge.checkpoints import ResponseCheckpoints
from forge.knowledge import KnowledgeStore
from forge.util import read_json, write_json
from forge.data import generate
from forge.llm import MockLLM, LLMProviderError
from forge.validation import load_policy
from forge.workflow import run

ROOT = Path(__file__).resolve().parents[1]


def test_cache_is_bound_to_model_data_policy_and_exact_context(tmp_path):
    context = {"description": "客户流失预测"}
    original = {"mode": "glm", "model": "glm-4.7-flash", "dataset": "A", "policy": "v1"}
    cache = ResponseCheckpoints(tmp_path, original)
    assert cache.lookup("understand", context) is None
    cache.save("understand", context, {"task": "churn_classification"}, "source_run")
    assert cache.lookup("understand", context)["task"] == "churn_classification"
    assert cache.hits[0]["source_run_id"] == "source_run"
    assert cache.lookup("understand", {"description": "different"}) is None
    for key, value in [("model", "other"), ("dataset", "B"), ("policy", "v2")]:
        assert ResponseCheckpoints(tmp_path, dict(original, **{key: value})).lookup("understand", context) is None


def test_corrupt_checkpoint_is_rejected(tmp_path):
    cache = ResponseCheckpoints(tmp_path, {"model": "test"})
    cache.save("understand", {}, {"task": "a"}, "run")
    path = cache.path("understand", {})
    item = read_json(path)
    item["output"]["task"] = "tampered"
    write_json(path, item)
    with pytest.raises(ValueError, match="corrupted"):
        cache.lookup("understand", {})


def test_material_change_invalidates_prepared_knowledge(tmp_path):
    spec = importlib.util.spec_from_file_location("run_glm", ROOT / "scripts/run_glm.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    materials = tmp_path / "materials"
    materials.mkdir()
    path = materials / "business.md"
    path.write_text("## 客户流失\n使用客户信息预测流失。", encoding="utf-8")
    class Extractor:
        mode = "glm"
        model = "glm-4.7-flash"
        def call(self, role, context):
            return {"facts": [{"title": "流失预测", "text": "客户流失预测", "source_chunk_id": "section_0"}]}
    db = tmp_path / "kb.sqlite"
    store = KnowledgeStore(db)
    store.ingest(materials, Extractor())
    assert module.reusable_materials(db, materials)["llm_accepted"] == 1
    path.write_text("## 客户流失\n业务规则已改变。", encoding="utf-8")
    assert module.reusable_materials(db, materials) is None


def test_resume_replays_completed_roles_but_always_revalidates(tmp_path, monkeypatch):
    data = tmp_path / "data.csv"
    generate(data, rows=200)
    store = KnowledgeStore(tmp_path / "kb.sqlite")
    store.ingest(ROOT / "materials")
    policy = load_policy(ROOT / "configs/churn.json")
    class FailsAtPlan(MockLLM):
        mode = "glm"
        model = "glm-4.7-flash"
        def call(self, role, context):
            if role == "plan":
                raise LLMProviderError("temporary overload", business_code="1305")
            return super().call(role, context)
    options = dict(description="客户流失预测", dataset=data, store=store, policy=policy,
                   output=tmp_path / "runs", checkpoint_dir=tmp_path / "checkpoints")
    _, first = run(llm=FailsAtPlan(), **options)
    assert first["status"] == "failed" and first["task_spec"]
    validations = []
    def evaluate(code, train, evaluation, policy, seed):
        validations.append(code)
        return {"passed": True, "checks": {"test_stub": True}, "metrics": {"roc_auc": .8, "f1": .6, "accuracy": .7}}
    monkeypatch.setattr("forge.workflow.evaluate", evaluate)
    class Continue(FailsAtPlan):
        def call(self, role, context):
            assert role != "understand", "completed role should be replayed"
            return MockLLM.call(self, role, context)
    _, second = run(llm=Continue(), **options)
    assert second["status"] == "passed" and len(validations) == 3
    assert second["checkpoint_hits"][0]["role"] == "understand"
    class ReplayOnly(FailsAtPlan):
        def call(self, role, context):
            raise AssertionError("All completed responses should already be saved")
    _, third = run(llm=ReplayOnly(), **options)
    assert third["status"] == "passed" and len(validations) == 6
    assert {entry["role"] for entry in third["checkpoint_hits"]} == {"understand", "plan", "generate"}
