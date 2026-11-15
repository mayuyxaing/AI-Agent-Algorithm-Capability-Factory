"""BigModel adapter checks use synthetic HTTP responses, never real credentials."""
import importlib.util
import json
import os
import socket
import ssl
from io import BytesIO
from pathlib import Path
from urllib.error import HTTPError, URLError
import pytest
from forge.cli import main
from forge.llm import GLMLLM, LLMProviderError
from forge.util import read_json

ROOT = Path(__file__).resolve().parents[1]


def response(body):
    class Response:
        def __enter__(self): return self
        def __exit__(self, *args): pass
        def read(self, limit): return json.dumps(body).encode()
    return Response()


def test_pinned_free_model_and_separate_key(monkeypatch):
    monkeypatch.delenv("ZHIPU_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "fake-openai-key")
    with pytest.raises(ValueError, match="ZHIPU_API_KEY"):
        GLMLLM()
    with pytest.raises(ValueError, match="no paid-model fallback"):
        GLMLLM("glm-4.7-flashx")


def test_glm_request_and_final_content(monkeypatch):
    monkeypatch.setenv("ZHIPU_API_KEY", "fake-glm-key")
    def fake(request, timeout):
        assert request.full_url == "https://open.bigmodel.cn/api/paas/v4/chat/completions"
        assert request.get_header("Authorization") == "Bearer fake-glm-key"
        payload = json.loads(request.data)
        assert payload["model"] == "glm-4.7-flash"
        assert payload["thinking"] == {"type": "disabled"}
        assert payload["response_format"] == {"type": "json_object"}
        assert payload["stream"] is False
        assert "fake-glm-key" not in request.data.decode()
        return response({"id": "glm_test", "model": "glm-4.7-flash", "usage": {"total_tokens": 10},
                         "choices": [{"finish_reason": "stop", "message": {
                             "reasoning_content": "not the final JSON", "content": '{"target":"churn"}'}}]})
    monkeypatch.setattr("forge.llm.urlopen", fake)
    llm = GLMLLM()
    assert llm.call("understand", {"description": "客户流失"}) == {"target": "churn"}
    assert llm.calls[0]["usage"]["total_tokens"] == 10
    assert "fake-glm-key" not in json.dumps(llm.calls)


@pytest.mark.parametrize("body", [
    {"choices": [{"finish_reason": "length", "message": {"content": "{}"}}]},
    {"choices": [{"finish_reason": "stop", "message": {"content": "not JSON"}}]},
    {"choices": []},
    {"error": {"code": "1302", "message": "fake-glm-key"}},
])
def test_bad_responses_are_not_success(monkeypatch, body):
    monkeypatch.setenv("ZHIPU_API_KEY", "fake-glm-key")
    monkeypatch.setattr("forge.llm.urlopen", lambda *a, **kw: response(body))
    with pytest.raises(LLMProviderError) as error:
        GLMLLM().call("generate", {})
    assert "fake-glm-key" not in str(error.value)


def test_rate_limit_does_not_fallback_or_retry(monkeypatch):
    monkeypatch.setenv("ZHIPU_API_KEY", "fake-glm-key")
    calls = []
    def fail(request, timeout):
        calls.append(request.full_url)
        raise HTTPError(request.full_url, 429, "fake-glm-key", {}, None)
    monkeypatch.setattr("forge.llm.urlopen", fail)
    with pytest.raises(LLMProviderError, match="429"):
        GLMLLM().call("understand", {})
    assert len(calls) == 1


@pytest.mark.parametrize("http_status,code", [(401, "1000"), (401, "1005"), (429, "1113"), (429, "1302")])
def test_business_codes_are_preserved_without_raw_messages(monkeypatch, http_status, code):
    monkeypatch.setenv("ZHIPU_API_KEY", "fake-glm-key")
    def fail(request, timeout):
        body = json.dumps({"error": {"code": code, "message": "fake-glm-key raw secret"}}).encode()
        raise HTTPError(request.full_url, http_status, "bad request", {}, BytesIO(body))
    monkeypatch.setattr("forge.llm.urlopen", fail)
    with pytest.raises(LLMProviderError) as error:
        GLMLLM().call("understand", {})
    assert f"business_code={code}" in str(error.value)
    assert "fake-glm-key" not in str(error.value) and "raw secret" not in str(error.value)


def test_doctor_missing_key_saves_safe_diagnostic(tmp_path, monkeypatch):
    monkeypatch.delenv("ZHIPU_API_KEY", raising=False)
    path = tmp_path / "connection.json"
    assert main(["doctor", "--provider", "glm", "--output", str(path)]) == 2
    assert "ZHIPU_API_KEY" in read_json(path)["error"]


def test_glm_launcher_uses_independent_db_and_clears_key(monkeypatch):
    spec = importlib.util.spec_from_file_location("run_glm", ROOT / "scripts/run_glm.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.delenv("ZHIPU_API_KEY", raising=False)
    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr(module.getpass, "getpass", lambda prompt: "fake-glm-key")
    monkeypatch.setattr(module, "reusable_materials", lambda *args: None)
    calls = []
    def fake_main(args):
        assert os.environ["ZHIPU_API_KEY"] == "fake-glm-key"
        calls.append(args)
        return 0
    monkeypatch.setattr(module, "main", fake_main)
    assert module.entry(["--run"]) == 0
    assert len(calls) == 3
    assert all(args[:2] == ["--db", "runtime/glm/knowledge.sqlite"] for args in calls)
    assert calls[1][-1] == "glm" and calls[2][-1] == "runtime/glm/runs"
    assert "ZHIPU_API_KEY" not in os.environ


def test_model_overload_retries_then_recovers(monkeypatch):
    monkeypatch.setenv("ZHIPU_API_KEY", "fake-glm-key")
    calls, delays = [], []
    def busy_then_ok(request, timeout):
        calls.append(request.full_url)
        if len(calls) <= 2:
            raise HTTPError(request.full_url, 429, "busy", {}, BytesIO(b'{"error":{"code":"1305"}}'))
        return response({"id": "ok", "choices": [{"finish_reason": "stop", "message": {"content": '{"target":"churn"}'}}]})
    monkeypatch.setattr("forge.llm.urlopen", busy_then_ok)
    monkeypatch.setattr("forge.llm.time.sleep", delays.append)
    llm = GLMLLM()
    assert llm.call("understand", {}) == {"target": "churn"}
    assert len(calls) == 3 and delays == [10, 20]
    assert len([c for c in llm.calls if c.get("status") == "retry"]) == 2


def test_persistent_overload_stops_after_three_attempts(monkeypatch):
    monkeypatch.setenv("ZHIPU_API_KEY", "fake-glm-key")
    calls, delays = [], []
    def busy(request, timeout):
        calls.append(request.full_url)
        raise HTTPError(request.full_url, 429, "busy", {}, BytesIO(b'{"error":{"code":"1305"}}'))
    monkeypatch.setattr("forge.llm.urlopen", busy)
    monkeypatch.setattr("forge.llm.time.sleep", delays.append)
    with pytest.raises(LLMProviderError, match="1305"):
        GLMLLM().call("understand", {})
    assert len(calls) == 3 and delays == [10, 20]


def test_network_timeout_retries_then_recovers(monkeypatch):
    monkeypatch.setenv("ZHIPU_API_KEY", "fake-glm-key")
    calls, delays = [], []

    def timeout_then_ok(request, timeout):
        calls.append(timeout)
        if len(calls) == 1:
            raise URLError(TimeoutError("fake-glm-key"))
        return response({"choices": [{"finish_reason": "stop", "message": {"content": '{"facts":[]}'}}]})

    monkeypatch.setattr("forge.llm.urlopen", timeout_then_ok)
    monkeypatch.setattr("forge.llm.time.sleep", delays.append)
    llm = GLMLLM()
    assert llm.call("extract", {}) == {"facts": []}
    assert len(calls) == 2 and delays == [10]
    assert llm.calls[0]["reason"] == "network"
    assert "fake-glm-key" not in json.dumps(llm.calls)


@pytest.mark.parametrize("failure,expected,count", [
    (TimeoutError("secret"), "timed out", 3),
    (socket.gaierror("secret"), "DNS", 3),
    (ssl.SSLCertVerificationError("secret"), "TLS", 1),
    (PermissionError("secret"), "access denied", 1),
])
def test_network_retry_is_bounded_and_diagnostics_are_safe(monkeypatch, failure, expected, count):
    monkeypatch.setenv("ZHIPU_API_KEY", "fake-glm-key")
    calls, delays = [], []

    def fail(request, timeout):
        calls.append(1)
        raise URLError(failure)

    monkeypatch.setattr("forge.llm.urlopen", fail)
    monkeypatch.setattr("forge.llm.time.sleep", delays.append)
    with pytest.raises(LLMProviderError, match=expected) as error:
        GLMLLM(max_attempts=6).call("understand", {})
    assert len(calls) == count
    assert len(delays) == count - 1
    assert "secret" not in str(error.value)
