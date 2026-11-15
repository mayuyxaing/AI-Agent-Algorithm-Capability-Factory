"""Bad LLM facts are quarantined, while verified knowledge remains usable."""
from pathlib import Path
import pytest
from forge.cli import main
from forge.knowledge import KnowledgeStore
from forge.util import read_json


@pytest.fixture
def material(tmp_path):
    folder = tmp_path / "materials"
    folder.mkdir()
    (folder / "source.md").write_text("# 业务材料\n\n## 流失预测\n使用客户合同预测流失。\n\n## 类别处理\n未知类别应允许忽略。\n", encoding="utf-8")
    return folder


class MixedExtractor:
    mode = "glm"
    model = "fake-test-model"

    def call(self, role, context):
        assert role == "extract" and len(context["evidence_chunks"]) == 2
        return {"facts": [
            {"title": "有依据的类别策略", "text": "处理未知类别。", "source_chunk_id": "section_1"},
            {"title": "伪造来源", "text": "不存在。", "source_chunk_id": "section_999"},
            {"title": "兼容旧格式", "text": "合同预测。", "evidence_quote": "使用客户合同预测流失。"},
            {"title": "冲突的引文", "text": "不应信任。", "source_chunk_id": "section_0", "evidence_quote": "完全不同的文字"},
        ]}


def test_mixed_results_preserve_exact_source_and_quarantine_bad_facts(material, tmp_path):
    store = KnowledgeStore(tmp_path / "kb.sqlite")
    assert store.ingest(material, MixedExtractor()) == 4
    nodes = store.nodes("Knowledge")
    assert {"伪造来源", "冲突的引文"}.isdisjoint(n["title"] for n in nodes)
    accepted = next(n for n in nodes if n["title"] == "有依据的类别策略")
    source = (material / "source.md").read_text(encoding="utf-8")
    assert accepted["evidence_quote"] in source
    assert accepted["extraction"] == "llm_with_source_chunk"
    assert accepted["line"] == 6
    audit = store.latest_ingestion()
    assert audit["status"] == "partial_llm"
    assert audit["llm_accepted"] == audit["llm_rejected"] == 2
    assert len(audit["rejections"]) == 2


def test_all_bad_facts_leave_deterministic_knowledge_without_claiming_llm_success(material, tmp_path):
    class AllBad(MixedExtractor):
        def call(self, role, context):
            return {"facts": [{"title": "fabricated", "text": "false", "evidence_quote": "not in source"}]}
    store = KnowledgeStore(tmp_path / "kb.sqlite")
    assert store.ingest(material, AllBad()) == 2
    audit = store.latest_ingestion()
    assert audit["llm_accepted"] == 0 and audit["llm_rejected"] == 1
    assert audit["status"] == "partial_llm"
    assert all(n["extraction"] == "markdown_sections" for n in store.nodes("Knowledge"))


def test_strict_mode_still_stops_on_unverified_source(material, tmp_path):
    store = KnowledgeStore(tmp_path / "kb.sqlite")
    with pytest.raises(ValueError, match="verifiable"):
        store.ingest(material, MixedExtractor(), strict_llm=True)
    assert store.latest_ingestion()["status"] == "rejected_strict"
    assert not any(n["title"] == "伪造来源" for n in store.nodes())


def test_cli_writes_audit_and_completes_with_verified_subset(material, tmp_path, monkeypatch):
    monkeypatch.setattr("forge.cli.create_llm", lambda args: MixedExtractor())
    report = tmp_path / "ingestion.json"
    assert main(["--db", str(tmp_path / "kb.sqlite"), "ingest", str(material),
                 "--provider", "glm", "--report", str(report)]) == 0
    assert read_json(report)["status"] == "partial_llm"


def test_malformed_envelope_is_corrected_with_original_response_and_source(material, tmp_path):
    class Correcting(MixedExtractor):
        calls = 0

        def call(self, role, context):
            self.calls += 1
            if self.calls == 1:
                return {"capabilities": ["wrong envelope"]}
            assert context["previous_response"] == {"capabilities": ["wrong envelope"]}
            assert "facts" in context["validation_error"]
            assert context["output_schema"]["required"] == ["facts"]
            return {"facts": [{"title": "类别策略", "text": "忽略未知类别", "source_chunk_id": "section_1"}]}

    llm = Correcting()
    store = KnowledgeStore(tmp_path / "kb.sqlite")
    assert store.ingest(material, llm) == 3
    assert llm.calls == 2
    audit = store.latest_ingestion()
    assert audit["status"] == "passed" and audit["llm_accepted"] == 1
    assert [r["event"] for r in audit["responses"]] == ["extract_response", "extract_rejected", "extract_response"]
    assert audit["responses"][0]["response"] == {"capabilities": ["wrong envelope"]}


def test_persistent_invalid_response_is_bounded_and_failure_report_is_written(material, tmp_path, monkeypatch):
    class Invalid(MixedExtractor):
        calls = 0

        def call(self, role, context):
            self.calls += 1
            return ["not an object"]

    llm = Invalid()
    monkeypatch.setattr("forge.cli.create_llm", lambda args: llm)
    report = tmp_path / "failed.json"
    db = tmp_path / "kb.sqlite"
    assert main(["--db", str(db), "ingest", str(material), "--provider", "glm", "--report", str(report)]) == 2
    audit = read_json(report)
    assert llm.calls == 2
    assert audit["status"] == "invalid_response" and audit["llm_accepted"] == 0
    assert "after 2 responses" in audit["error"]
    assert len(audit["responses"]) == 4
    assert KnowledgeStore(db).nodes("Knowledge") == []


def test_extraction_prompt_is_shared_across_tasks():
    from forge.llm import role_prompt
    assert role_prompt("extract", {"task_id": "text_classification"}) == role_prompt("extract", {})
    assert "source_chunk_id" in role_prompt("extract", {"task_id": "text_classification"})


def test_provider_failure_is_reported_without_format_retry(material, tmp_path):
    from forge.llm import LLMProviderError

    class Unavailable(MixedExtractor):
        calls = 0

        def call(self, role, context):
            self.calls += 1
            raise LLMProviderError("service unavailable")

    llm = Unavailable()
    store = KnowledgeStore(tmp_path / "kb.sqlite")
    with pytest.raises(LLMProviderError, match="service unavailable"):
        store.ingest(material, llm)
    assert llm.calls == 1
    assert store.latest_ingestion()["status"] == "provider_error"
    assert store.latest_ingestion()["llm_accepted"] == 0


def test_malformed_json_text_is_audited_and_corrected(material, tmp_path):
    from forge.llm import LLMResponseFormatError

    class Correcting(MixedExtractor):
        calls = 0

        def call(self, role, context):
            self.calls += 1
            if self.calls == 1:
                raise LLMResponseFormatError('{"facts":')
            assert context["previous_response"]["raw_content"] == '{"facts":'
            return {"facts": [{"title": "策略", "text": "未知类别允许忽略", "source_chunk_id": "section_1"}]}

    llm = Correcting()
    store = KnowledgeStore(tmp_path / "kb.sqlite")
    assert store.ingest(material, llm) == 3
    assert llm.calls == 2
    assert store.latest_ingestion()["responses"][0]["response"]["raw_content"] == '{"facts":'
