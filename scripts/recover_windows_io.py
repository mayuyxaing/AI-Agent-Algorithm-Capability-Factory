"""Audited compatibility migration for the bounded Windows replacement retry only."""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
from forge.provenance import execution_identity
from forge.util import digest, read_json, write_json

ROOT = Path(__file__).resolve().parents[1]


def recover(campaign, revision, apply=False):
    campaign = campaign.resolve()
    if not campaign.is_relative_to((ROOT / "runtime").resolve()):
        raise ValueError("Recovery is restricted to project runtime campaigns")
    path = campaign / "manifest.json"
    original = path.read_bytes()
    manifest = read_json(path)
    saved = manifest["protocol"]["execution"]["source_sha256"]
    current = execution_identity()["source_sha256"]
    if saved == current:
        print("Campaign already matches the current source; no migration needed")
        return
    git = ["git", "-c", f"safe.directory={ROOT.as_posix()}"]
    old_util = subprocess.check_output(git + ["show", revision + ":forge/util.py"], cwd=ROOT).decode("utf-8")
    old_util = old_util.replace("\r\n", "\n")
    files = {p.relative_to(ROOT / "forge").as_posix(): digest(p.read_text(encoding="utf-8"))
             for p in sorted((ROOT / "forge").rglob("*")) if p.suffix in {".py", ".json", ".txt"}}
    files["util.py"] = digest(old_util)
    if digest(json.dumps(files, sort_keys=True)) != saved:
        raise ValueError("Source changes exceed the utility hotfix; refuse to migrate")
    reports = {}
    for cell in manifest["results"].values():
        report = campaign / cell["report"]
        if not report.resolve().is_relative_to(campaign):
            raise ValueError("Invalid report path")
        if digest(report.read_bytes()) != cell["report_sha256"]:
            raise ValueError("Existing report integrity check failed")
        reports[cell["report"]] = cell["report_sha256"]
    datasets = {p.relative_to(campaign).as_posix(): digest(p.read_bytes()) for p in (campaign / "datasets").glob("*.csv")}
    unfinished = [p.relative_to(campaign).as_posix() for p in campaign.glob("cells/*/attempts/*/runs/*")
                  if p.is_dir() and not (p / "report.json").exists()]
    record = dict(created_at=datetime.now(timezone.utc).isoformat(), source_revision=revision,
                  reason="Bounded retry for Windows transient PermissionError during atomic JSON replacement",
                  changed_source_files=["forge/util.py"], prior_protocol=deepcopy(manifest["protocol"]),
                  previous_source_sha256=saved, current_source_sha256=current,
                  report_sha256=reports, dataset_sha256=datasets,
                  unfinished_attempts=unfinished,
                  limitation="Unfinished attempts may lack API usage; their unrecorded cost is unknown, not zero.")
    print(f"Verified {len(reports)} report hashes; only forge/util.py differs; unfinished attempts: {len(unfinished)}")
    if not apply:
        print("Dry run only")
        return
    if path.read_bytes() != original:
        raise ValueError("Manifest changed concurrently; retry only after the running process exits")
    backup = campaign / "manifest.before-windows-io-hotfix.json"
    if backup.exists():
        raise ValueError("Recovery backup already exists; refuse to overwrite")
    backup.write_bytes(original)
    write_json(campaign / "windows-io-hotfix.json", record)
    manifest["protocol"]["execution"]["source_sha256"] = current
    manifest.setdefault("io_compatibility_migrations", []).append("windows-io-hotfix.json")
    manifest["status"] = "local_io_interrupted"
    write_json(path, manifest)
    assert all(digest((campaign / p).read_bytes()) == h for p, h in reports.items())
    print("Applied audited IO-only migration; completed reports and previous protocol preserved")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    recover(args.campaign, args.source_revision, args.apply)
