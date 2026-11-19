"""Recompute descriptive results from immutable reports without model calls."""
import argparse
from collections import Counter
from pathlib import Path
from forge.util import digest, read_json, write_json


def summarize(root: Path):
    root = root.resolve()
    manifest = read_json(root / "manifest.json")
    protocol = manifest["protocol"]
    expected = {f"{case}-{seed}-{arm}" for case in protocol["cases"]
                for seed in protocol["seeds"] for arm in protocol["arms"]}
    if manifest["status"] != "complete" or set(manifest["results"]) != expected:
        raise ValueError("Campaign is not complete")
    groups, rows = {}, []
    for key, cell in manifest["results"].items():
        path = (root / cell["report"]).resolve()
        if not path.is_relative_to(root) or digest(path.read_bytes()) != cell["report_sha256"]:
            raise ValueError(f"Report integrity failed: {key}")
        report = read_json(path)
        dataset = root / "datasets" / f"{cell['case']}-{cell['seed']}.csv"
        if report["dataset_sha256"] != digest(dataset.read_bytes()):
            raise ValueError(f"Dataset identity failed: {key}")
        if report["experiment_arm"] != cell["arm"] or report["seed"] != cell["seed"]:
            raise ValueError(f"Cell identity failed: {key}")
        if report["fault_injection"] or report["mode"] != protocol["mode"] or report["model"] != protocol["model"]:
            raise ValueError(f"Provider or injection mismatch: {key}")
        if report.get("failure_kind") in {"provider", "internal", "persistence"}:
            raise ValueError(f"Unresolved infrastructure/host failure: {key}")
        if report["status"] == "passed":
            final = report["test_result"]
            if not final["passed"] or not final["checks"] or not all(final["checks"].values()):
                raise ValueError(f"Contradictory final result: {key}")
        candidates = report["candidates"]
        first = sum(bool(c["attempts"] and c["attempts"][0]["passed"]) for c in candidates)
        failure = "passed" if report["status"] == "passed" else "planning_contract" if report.get("error_stage") == "plan" else report["status"]
        events = read_json(path.parent / "events.json")
        calls = report.get("llm_calls", [])
        tokens = sum(c.get("usage", {}).get("total_tokens", 0) for c in calls)
        if tokens != cell["reported_tokens"]:
            raise ValueError(f"Usage mismatch: {key}")
        prior = cell.get("prior_attempts", [])
        row = dict(cell=key, case=cell["case"], seed=cell["seed"], arm=cell["arm"],
                   status=report["status"], failure_category=failure, candidate_count=len(candidates),
                   first_pass_candidates=first, repair_calls=sum(e["role"] == "repair" for e in events),
                   reported_tokens=tokens, reported_tokens_including_prior_attempts=tokens + sum(p["reported_tokens"] for p in prior),
                   prior_attempt_count=len(prior), test_metrics=report.get("test_result", {}).get("metrics", {}),
                   report=cell["report"])
        rows.append(row)
    for arm in protocol["arms"]:
        subset = [r for r in rows if r["arm"] == arm]
        count = sum(r["candidate_count"] for r in subset)
        groups[arm] = dict(tasks=len(subset), passed=sum(r["status"] == "passed" for r in subset),
            failure_breakdown=dict(Counter(r["failure_category"] for r in subset if r["status"] != "passed")),
            candidate_count=count, first_pass_candidates=sum(r["first_pass_candidates"] for r in subset),
            repair_calls=sum(r["repair_calls"] for r in subset),
            reported_tokens_including_prior_attempts=sum(r["reported_tokens_including_prior_attempts"] for r in subset),
            prior_attempt_count=sum(r["prior_attempt_count"] for r in subset))
        stored = manifest["statistics"][arm]
        if stored["reported_tokens_including_prior_attempts"] != groups[arm]["reported_tokens_including_prior_attempts"]:
            raise ValueError("Stored summary disagrees with reports")
        if stored["task_success_rate"] != groups[arm]["passed"] / len(subset):
            raise ValueError("Stored pass rate disagrees with reports")
    pairs = {}
    for left, right in (("direct", "knowledge"), ("knowledge", "knowledge_repair")):
        counter = Counter()
        for case in protocol["cases"]:
            for seed in protocol["seeds"]:
                a = next(r for r in rows if (r["case"], r["seed"], r["arm"]) == (case, seed, left))
                b = next(r for r in rows if (r["case"], r["seed"], r["arm"]) == (case, seed, right))
                x, y = a["status"] == "passed", b["status"] == "passed"
                counter["both_pass" if x and y else "left_only" if x else "right_only" if y else "both_fail"] += 1
        pairs[left + "_vs_" + right] = dict(counter)
    return dict(protocol_version=protocol["version"], model=protocol["model"], cells=len(rows),
                groups=groups, paired_task_outcomes=pairs, results=rows,
                io_migrations=manifest.get("io_compatibility_migrations", []),
                limits=["Nine synthetic datasets per arm; no repeated LLM sampling on each fixed task.",
                        "First-pass candidate rates exclude tasks rejected during planning.",
                        "API-reported tokens omit usage never returned or lost in incomplete attempts.",
                        "Observed completion rates do not establish stable causal gains or holdout-metric superiority."])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("campaign", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.campaign)
    write_json(args.output, result)
    print(f"Verified {result['cells']} cells; recomputed summaries without model calls")
