"""IO-only recovery cannot silently accept algorithm changes or rewrite results."""
import importlib.util
import json
from pathlib import Path
import pytest
from forge.util import digest, read_json, write_json


@pytest.mark.parametrize("other_changed", [False, True])
def test_recovery_preserves_reports_and_rejects_non_io_changes(tmp_path, monkeypatch, other_changed):
    source = Path(__file__).resolve().parents[1] / "scripts/recover_windows_io.py"
    spec = importlib.util.spec_from_file_location("io_recovery_test", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    root = tmp_path / "project"
    forge = root / "forge"
    forge.mkdir(parents=True)
    (forge / "util.py").write_text("new utility", encoding="utf-8")
    (forge / "algorithm.py").write_text("changed algorithm" if other_changed else "same algorithm", encoding="utf-8")
    old_files = {"util.py": digest("old utility"), "algorithm.py": digest("same algorithm")}
    old_hash = digest(json.dumps(old_files, sort_keys=True))
    monkeypatch.setattr(module, "ROOT", root)
    monkeypatch.setattr(module, "execution_identity", lambda: {"source_sha256": "new-source-hash"})
    monkeypatch.setattr(module.subprocess, "check_output", lambda *a, **kw: b"old utility")
    campaign = root / "runtime/campaign"
    report = campaign / "cells/done/report.json"
    write_json(report, {"status": "passed"})
    original_report = report.read_bytes()
    path = campaign / "manifest.json"
    write_json(path, {"protocol": {"execution": {"source_sha256": old_hash}, "policy": {"budget": 3}},
                      "results": {"done": {"report": "cells/done/report.json", "report_sha256": digest(original_report)}}})
    original_manifest = path.read_bytes()
    if other_changed:
        with pytest.raises(ValueError, match="exceed"):
            module.recover(campaign, "revision", apply=True)
        assert path.read_bytes() == original_manifest
        assert not (campaign / "manifest.before-windows-io-hotfix.json").exists()
    else:
        module.recover(campaign, "revision", apply=True)
        assert (campaign / "manifest.before-windows-io-hotfix.json").read_bytes() == original_manifest
        assert read_json(path)["protocol"]["policy"] == {"budget": 3}
        assert read_json(path)["protocol"]["execution"]["source_sha256"] == "new-source-hash"
    assert report.read_bytes() == original_report
