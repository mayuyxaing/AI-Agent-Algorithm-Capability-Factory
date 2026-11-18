"""Transient Windows file locks cannot truncate a previous event journal."""
from pathlib import Path
import pytest
from forge.util import read_json, write_json


def denied():
    error = PermissionError("injected Windows replacement lock")
    error.winerror = 5
    return error


def test_transient_windows_lock_recovers(monkeypatch, tmp_path):
    path = tmp_path / "events.json"
    write_json(path, {"old": True})
    original = Path.replace
    calls, sleeps = [], []
    def replace(source, destination):
        calls.append(destination)
        if len(calls) < 3:
            assert read_json(path) == {"old": True}
            raise denied()
        return original(source, destination)
    monkeypatch.setattr(Path, "replace", replace)
    monkeypatch.setattr("forge.util.time.sleep", sleeps.append)
    write_json(path, {"new": True})
    assert read_json(path) == {"new": True}
    assert sleeps == [.1, .2]


def test_persistent_lock_stops_and_preserves_previous_file(monkeypatch, tmp_path):
    path = tmp_path / "events.json"
    write_json(path, {"old": True})
    calls = []
    def replace(*args):
        calls.append(1)
        raise denied()
    monkeypatch.setattr(Path, "replace", replace)
    monkeypatch.setattr("forge.util.time.sleep", lambda seconds: None)
    with pytest.raises(PermissionError):
        write_json(path, {"new": True})
    assert len(calls) == 6
    assert read_json(path) == {"old": True}
    assert list(tmp_path.glob("*.tmp")) == []
