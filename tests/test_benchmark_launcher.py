"""Private-key launcher forwards a fixed pilot or formal experiment matrix."""
import importlib.util
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]


def launcher(monkeypatch):
    spec = importlib.util.spec_from_file_location("benchmark_launch_test", ROOT / "scripts/run_glm.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.chdir(ROOT)
    monkeypatch.setenv("ZHIPU_API_KEY", "test-private-key")
    return module


@pytest.mark.parametrize("full", [False, True])
def test_benchmark_matrix_and_results_directory_are_forwarded(monkeypatch, full):
    module = launcher(monkeypatch)
    calls = []
    monkeypatch.setattr(module, "main", lambda args: calls.append(args) or 0)
    assert module.entry(["--full-benchmark" if full else "--benchmark"]) == 0
    assert len(calls) == 1
    command = calls[0]
    assert command[:3] == ["benchmark", "--provider", "glm"]
    assert "test-private-key" not in str(command)
    expected_output = "runtime/glm/benchmark-full-v3" if full else "runtime/glm/benchmark-v3"
    assert command[-1] == expected_output
    if full:
        assert command[command.index("--cases") + 1:command.index("--seeds")] == ["baseline", "missing_high", "minority"]
        assert command[command.index("--seeds") + 1:command.index("--rows")] == ["42", "43", "44"]


def test_full_benchmark_rejects_unsupported_text_task_before_key_entry(monkeypatch):
    module = launcher(monkeypatch)
    monkeypatch.delenv("ZHIPU_API_KEY")
    with pytest.raises(SystemExit) as exc:
        module.entry(["--full-benchmark", "--task", "text_classification"])
    assert exc.value.code == 2
