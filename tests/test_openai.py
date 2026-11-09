"""HTTP contract tests with fake credentials/responses; never bill an account."""
import importlib.util
import json
import os
from pathlib import Path
from urllib.error import HTTPError
import pytest
from forge.cli import main
from forge.data import generate
from forge.knowledge import KnowledgeStore
from forge.llm import LLMProviderError, MockLLM, OpenAILLM
from forge.util import read_json
from forge.validation import load_policy
from forge.workflow import run

ROOT = Path(__file__).resolve().parents[1]


def test_missing_key_and_no_fallback(monkeypatch, capsys):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert main(["doctor", "--provider", "openai"]) == 2
    assert "OPENAI_API_KEY" in capsys.readouterr().out


def test_responses_contract_and_usage(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "fake-test-key")
    class Response:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def read(self, limit):
            return json.dumps({"id": "resp_test", "status": "completed", "usage": {"total_tokens": 15},
                "output": [{"type": "reasoning"}, {"type": "message", "content": [{"type": "output_text",
                "text": '{"task":"churn_classification","target":"churn"}'}]}]}).encode()
    def fake(request, timeout):
        assert request.full_url == "https://api.openai.com/v1/responses"
        assert request.get_header("Authorization") == "Bearer fake-test-key"
        body = json.loads(request.data)
        assert body["store"] is False and body["max_output_tokens"] == 6000
        assert "fake-test-key" not in request.data.decode()
        assert "JSON" in body["instructions"]
        return Response()
    monkeypatch.setattr("forge.llm.urlopen", fake)
    llm = OpenAILLM("test-model")
    assert llm.call("understand", {"description": "客户流失"})["target"] == "churn"
    assert llm.calls[0]["usage"]["total_tokens"] == 15
    assert "fake-test-key" not in json.dumps(llm.calls)


@pytest.mark.parametrize("status", [401, 429])
def test_http_errors_do_not_echo_body_or_secret(monkeypatch, status):
    monkeypatch.setenv("OPENAI_API_KEY", "fake-test-key")
    def fail(*args, **kwargs):
        raise HTTPError("https://api.openai.com", status, "fake-test-key", {}, None)
    monkeypatch.setattr("forge.llm.urlopen", fail)
    with pytest.raises(LLMProviderError) as error:
        OpenAILLM("test-model").call("understand", {})
    assert str(status) in str(error.value)
    assert "fake-test-key" not in str(error.value)


def test_incomplete_response_is_not_success(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "fake-test-key")
    class Response:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def read(self, limit): return b'{"status":"incomplete","output":[]}'
    monkeypatch.setattr("forge.llm.urlopen", lambda *a, **kw: Response())
    with pytest.raises(LLMProviderError, match="incomplete"):
        OpenAILLM("test-model").call("generate", {})


def test_service_error_does_not_trigger_code_repair(tmp_path):
    class Unavailable(MockLLM):
        mode = "openai"
        def call(self, role, context):
            if role == "generate":
                raise LLMProviderError("OpenAI HTTP 429: Check API credit balance and rate limits")
            assert role != "repair"
            return super().call(role, context)
    data = tmp_path / "data.csv"
    generate(data, rows=200)
    store = KnowledgeStore(tmp_path / "kb.sqlite")
    store.ingest(ROOT / "materials")
    folder, report = run("客户流失预测", data, store, Unavailable(), load_policy(ROOT / "configs/churn.json"), tmp_path / "runs")
    assert report["status"] == "failed" and "429" in report["error"]
    assert "429" in (folder / "report.md").read_text(encoding="utf-8")
    assert not any(e["role"] == "repair" for e in read_json(folder / "events.json"))


def test_interactive_key_is_removed_after_failure(monkeypatch):
    spec = importlib.util.spec_from_file_location("run_openai", ROOT / "scripts/run_openai.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr(module.getpass, "getpass", lambda prompt: "fake-test-key")
    def check(args):
        assert os.environ["OPENAI_API_KEY"] == "fake-test-key"
        return 2
    monkeypatch.setattr(module, "main", check)
    assert module.entry([]) == 2
    assert "OPENAI_API_KEY" not in os.environ
