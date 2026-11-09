"""Explicit role handoffs with bounded repairs and independent evaluation."""
import importlib.metadata
import platform
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from .data import FEATURES, NUMERIC, CATEGORICAL, load_split
from .llm import plugins, template_code
from .reporting import report_markdown, graph_html
from .util import digest, write_json
from .validation import evaluate


def run(description, dataset: Path, store, llm, policy: dict, output: Path,
        inject_fault=False, seed=42):
    if inject_fault and llm.mode != "mock":
        raise ValueError("Demonstration fault injection is supported only in mock mode")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    folder = output / run_id
    folder.mkdir(parents=True)
    events = []

    def record(role, incoming, outgoing):
        event = {"step": len(events), "role": role, "input": incoming, "output": outgoing}
        events.append(event)
        write_json(folder / "events.json", events)

    report = {"run_id": run_id, "description": description, "status": "running",
              "mode": llm.mode, "model": llm.model, "fault_injection": inject_fault,
              "dataset_sha256": digest(dataset.read_bytes()), "seed": seed, "policy": policy,
              "environment": {"python": platform.python_version(), "platform": platform.platform(),
                              **{name: importlib.metadata.version(name) for name in ("numpy", "pandas", "scikit-learn", "joblib")}},
              "knowledge": [], "experiences": [], "candidates": []}
    try:
        train, valid, test = load_split(dataset, seed)
        report["split"] = {"train": len(train), "validation": len(valid), "test": len(test),
                           "train_positive_rate": float(train.churn.mean())}
        spec = llm.call("understand", {"description": description})
        if spec.get("task") != "churn_classification" or spec.get("target") != "churn":
            raise ValueError("This prototype only supports churn_classification with target churn")
        report["task_spec"] = spec
        record("understand", {"description": description}, spec)
        knowledge = store.retrieve(description)
        if not knowledge:
            raise ValueError("No relevant knowledge found; ingest materials first")
        experiences = store.experiences()
        report.update(knowledge=knowledge, experiences=experiences)
        record("retrieve", {"query": description}, {"knowledge": knowledge, "experiences": experiences})
        context = {"spec": spec, "knowledge": knowledge, "experiences": experiences, "plugins": list(plugins().values())}
        plan = llm.call("plan", context)
        candidates = plan.get("candidates", [])
        if not 1 <= len(candidates) <= 3:
            raise ValueError("Plan must contain 1 to 3 candidates")
        ids = [candidate["algorithm"] for candidate in candidates]
        if len(set(ids)) != len(ids) or not set(ids) <= set(plugins()):
            raise ValueError("Plan has duplicate or unregistered candidates")
        if not set(plan.get("evidence_ids", [])) <= {n["id"] for n in knowledge}:
            raise ValueError("Plan cites unknown evidence")
        if not set(plan.get("experience_ids", [])) <= {n["id"] for n in experiences}:
            raise ValueError("Plan cites unknown historical experience")
        report["plan"] = plan
        record("plan", context, plan)
        # A transparent constant-probability reference, computed by the trusted host.
        import numpy as np
        from .validation import METRICS
        constant = np.full(len(valid), float(train.churn.mean()))
        report["baseline"] = {name: METRICS[name](valid.churn, constant) for name in policy["metrics"]}
        for candidate in candidates:
            algorithm = candidate["algorithm"]
            entry = {"algorithm": algorithm, "rationale": str(candidate.get("rationale", "")),
                     "attempts": [], "passed": False}
            report["candidates"].append(entry)
            previous = feedback = None
            for attempt in range(policy["max_repairs"] + 1):
                role = "generate" if attempt == 0 else "repair"
                incoming = {"algorithm": algorithm, "spec": spec, "plan": plan,
                            "knowledge": knowledge, "experiences": experiences,
                            "features": {"numeric": NUMERIC, "categorical": CATEGORICAL},
                            "interface_reference": template_code(algorithm),
                            "inject_fault": inject_fault and algorithm == "logistic" and attempt == 0,
                            "previous_code": previous, "feedback": feedback}
                try:
                    generation = llm.call(role, incoming)
                    code = generation["code"]
                    if not isinstance(code, str) or not 1 <= len(code) <= 100_000:
                        raise ValueError("Generated code must be a string under 100KB")
                except Exception as exc:
                    generation = {"code": "", "explanation": f"Provider failure: {type(exc).__name__}: {exc}"}
                    code = ""
                record(role, incoming, generation)
                attempt_folder = folder / algorithm / f"attempt_{attempt}"
                attempt_folder.mkdir(parents=True)
                code_path = attempt_folder / "algorithm.py"
                code_path.write_text(code, encoding="utf-8")
                feedback = evaluate(code_path, train, valid, policy, seed)
                record("validate", {"code_sha256": digest(code), "split": "validation"}, feedback)
                entry["attempts"].append(dict(feedback, attempt=attempt, code_path=code_path.relative_to(folder).as_posix(),
                                              code_sha256=digest(code), injected=incoming["inject_fault"],
                                              explanation=generation.get("explanation", "")))
                previous = code
                if feedback["passed"]:
                    entry["passed"] = True
                    break
        passing = [candidate for candidate in report["candidates"] if candidate["passed"]]
        if passing:
            selected = max(passing, key=lambda candidate: candidate["attempts"][-1]["metrics"][policy["selection_metric"]])
            report["selected"] = selected["algorithm"]
            final_dir = folder / "selected"
            final_dir.mkdir()
            shutil.copy2(folder / selected["attempts"][-1]["code_path"], final_dir / "algorithm.py")
            # Never repair or reselect based on the final holdout.
            final = evaluate(final_dir / "algorithm.py", train, test, policy, seed)
            report["test_result"] = final
            report["status"] = "passed" if final["passed"] else "test_failed"
            record("final_evaluation", {"selected": report["selected"], "split": "test"}, final)
        else:
            report["status"] = "no_valid_candidate"
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        record("workflow_error", {}, {"error": report["error"]})
    write_json(folder / "report.json", report)
    (folder / "report.md").write_text(report_markdown(report), encoding="utf-8")
    store.save_run(report)
    store.export(folder / "knowledge_graph.json")
    graph_html(folder / "knowledge_graph.json", folder / "knowledge_graph.html")
    record("persist", {"status": report["status"]}, {"graph": "knowledge_graph.json"})
    return folder, report
