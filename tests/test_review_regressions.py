"""Regressions for per-column validation, document freshness and campaign resume."""
from pathlib import Path
import pytest
from forge.data import generate, load_split
from forge.knowledge import KnowledgeStore
from forge.llm import MockLLM, template_code
from forge.tasks import get_task
from forge.util import digest, read_json, write_json
from forge.validation import evaluate, load_policy

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("field", ["contract", "payment_method"])
@pytest.mark.parametrize("mixed_only", [False, True])
def test_unknown_error_in_each_categorical_field_is_rejected(tmp_path, field, mixed_only):
    data = tmp_path / "data.csv"
    generate(data, rows=200)
    train, valid, _ = load_split(data)
    # A defective adapter handles all other input, but rejects this column's
    # unknown values (or only mixed known/unknown batches).
    guard = f"frame[{field!r}].fillna('').str.startswith('__previously_unseen_')"
    condition = "mask.any() and not mask.all()" if mixed_only else "mask.any()"
    source = template_code("logistic").replace(
        "    missing = set(FEATURES)",
        f"    mask = {guard}\n    if {condition}:\n        raise ValueError('Unknown category')\n    missing = set(FEATURES)")
    code = tmp_path / "algorithm.py"
    code.write_text(source, encoding="utf-8")
    # The injected guard must allow the ordinary evaluation frame.
    # Missing-column rejection is allowed to fail too if the guarded field is absent.
    result = evaluate(code, train, valid, load_policy(ROOT / "configs/churn.json"), 42)
    check = "unknown_category_" + ("mixed_" if mixed_only else "") + field
    assert result["executed"] and not result["passed"]
    assert result["checks"][check] is False
    other = "payment_method" if field == "contract" else "contract"
    assert result["checks"]["unknown_category_" + other] is True


def document(folder, text):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "business.md").write_text("## 未知类别策略\n" + text, encoding="utf-8")


def test_updated_and_deleted_documents_leave_history_but_never_rank(tmp_path):
    materials = tmp_path / "materials"
    document(materials, "未知类别旧策略 " * 40)
    store = KnowledgeStore(tmp_path / "kb.sqlite")
    store.ingest(materials)
    old = {n["id"] for n in store.current_facts()}
    document(materials, "未知类别新策略")
    store.ingest(materials)
    for results in (store.retrieve("未知类别旧策略"), store.retrieve_grounded("未知类别旧策略", ["unknown_categories"])):
        assert results and old.isdisjoint(n["id"] for n in results)
    assert old <= {n["id"] for n in store.nodes("Knowledge")}
    (materials / "business.md").unlink()
    store.ingest(materials)
    assert not store.retrieve("未知类别")
    assert not store.retrieve_grounded("未知类别", ["unknown_categories"])


def test_failed_ingestion_does_not_publish_partial_new_document(tmp_path):
    materials = tmp_path / "materials"
    document(materials, "未知类别原策略")
    store = KnowledgeStore(tmp_path / "kb.sqlite")
    store.ingest(materials)
    expected = store.retrieve_grounded("未知类别", ["unknown_categories"])
    # a.md is written before z.py fails parsing, testing partial-write isolation.
    (materials / "a.md").write_text("## 未知类别\n未知类别失败批次", encoding="utf-8")
    document(materials, "未知类别尚未发布的新策略")
    (materials / "z.py").write_text("def broken(", encoding="utf-8")
    with pytest.raises(SyntaxError):
        store.ingest(materials)
    assert store.latest_ingestion()["status"] == "extraction_error"
    assert store.retrieve_grounded("未知类别", ["unknown_categories"]) == expected


def test_same_relative_path_in_separate_corpora_stays_independent(tmp_path):
    store = KnowledgeStore(tmp_path / "kb.sqlite")
    for name in ("a", "b"):
        document(tmp_path / name, "未知类别 " + name)
        store.ingest(tmp_path / name)
    document(tmp_path / "a", "未知类别更新 a")
    store.ingest(tmp_path / "a")
    texts = {n["text"] for n in store.current_facts()}
    assert texts == {"## 未知类别策略\n未知类别更新 a", "## 未知类别策略\n未知类别 b"}


def test_export_import_preserves_current_snapshot(tmp_path):
    store = KnowledgeStore(tmp_path / "original.sqlite")
    materials = tmp_path / "materials"
    for version in ("old", "new"):
        document(materials, "未知类别 " + version)
        store.ingest(materials)
    exported = tmp_path / "graph.json"
    store.export(exported)
    restored = KnowledgeStore(tmp_path / "restored.sqlite")
    restored.import_graph(read_json(exported))
    assert restored.retrieve_grounded("未知类别", ["unknown_categories"]) == store.retrieve_grounded("未知类别", ["unknown_categories"])
    assert len(restored.nodes("Knowledge")) == 2


def test_legacy_graph_is_retired_after_successful_refresh(tmp_path):
    store = KnowledgeStore(tmp_path / "legacy.sqlite")
    sid = store.node("Source", "deleted.md", {"path": "deleted.md", "sha256": "legacy"})
    store.node("Knowledge", "旧知识", {"text": "未知类别旧策略", "source_id": sid})
    assert store.retrieve("未知类别")
    document(tmp_path / "materials", "未知类别新策略")
    store.ingest(tmp_path / "materials")
    assert all(n["title"] != "旧知识" for n in store.retrieve_grounded("未知类别", ["unknown_categories"]))


def fake_campaign(monkeypatch, *, interrupt=False):
    calls = []
    def workflow(description, dataset, store, llm, policy, output, **kwargs):
        assert not store.nodes("FailureCase") and not store.nodes("ValidationRun")
        assert not kwargs.get("checkpoint_dir")
        calls.append(store.path)
        # Deliberate poison: must never become the next attempt's experience.
        store.node("FailureCase", "poison", {"task": "churn_classification", "reusable": True})
        output.mkdir(parents=True)
        write_json(output / "events.json", [])
        report = {"status": "failed" if interrupt and len(calls) == 1 else "passed",
                  "failure_kind": "provider" if interrupt and len(calls) == 1 else None,
                  "dataset_sha256": digest(dataset.read_bytes()), "candidates": [], "llm_calls": []}
        write_json(output / "report.json", report)
        return output, report
    monkeypatch.setattr("forge.benchmark.run", workflow)
    return calls


def test_provider_restart_uses_new_database_and_no_old_experience(tmp_path, monkeypatch):
    from forge.benchmark import benchmark
    calls = fake_campaign(monkeypatch, interrupt=True)
    policy = load_policy(ROOT / "configs/churn.json")
    first = benchmark(MockLLM, ROOT / "materials", policy, tmp_path, rows=200)
    assert first["status"] == "provider_interrupted"
    final = benchmark(MockLLM, ROOT / "materials", policy, tmp_path, rows=200)
    assert final["status"] == "complete" and len(calls) == len(set(calls)) == 4
    assert final["results"]["baseline-42-direct"]["attempt_number"] == 2
    assert len(final["results"]["baseline-42-direct"]["prior_attempts"]) == 1
    benchmark(MockLLM, ROOT / "materials", policy, tmp_path, rows=200)
    assert len(calls) == 4


@pytest.mark.parametrize("tamper", ["dataset", "report", "implementation"])
def test_resume_refuses_changed_experiment_inputs_or_results(tmp_path, monkeypatch, tamper):
    from forge.benchmark import benchmark
    calls = fake_campaign(monkeypatch)
    policy = load_policy(ROOT / "configs/churn.json")
    result = benchmark(MockLLM, ROOT / "materials", policy, tmp_path, rows=200)
    if tamper == "implementation":
        monkeypatch.setattr("forge.benchmark.execution_identity", lambda: {"changed": True})
    else:
        path = tmp_path / ("datasets/baseline-42.csv" if tamper == "dataset" else result["results"]["baseline-42-direct"]["report"])
        path.write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError, match="differs|changed"):
        benchmark(MockLLM, ROOT / "materials", policy, tmp_path, rows=200)
    assert len(calls) == 3


def test_legacy_prepared_material_cache_requires_snapshot_refresh(tmp_path):
    import importlib.util
    import json
    spec = importlib.util.spec_from_file_location("review_run_glm", ROOT / "scripts/run_glm.py")
    script = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(script)
    materials = tmp_path / "materials"
    document(materials, "未知类别")
    db = tmp_path / "kb.sqlite"
    store = KnowledgeStore(db)
    class Extractor:
        mode, model = "glm", "glm-4.7-flash"
        def call(self, role, context):
            return {"facts": []}
    store.ingest(materials, Extractor())
    assert script.reusable_materials(db, materials)
    audit = store.latest_ingestion()
    audit.pop("retrieval_snapshot_version")
    with store.connect() as con:
        con.execute("UPDATE ingestions SET report=? WHERE id=?", (json.dumps(audit), audit["id"]))
    assert script.reusable_materials(db, materials) is None
