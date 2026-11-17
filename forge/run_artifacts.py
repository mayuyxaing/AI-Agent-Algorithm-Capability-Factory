"""Report metadata and artifact publication, separate from algorithm execution."""
from copy import deepcopy
import importlib.metadata
import platform
from pathlib import Path
from .contracts import JsonObject
from .reporting import report_markdown, graph_html
from .session import RunSession
from .util import write_json

WORKFLOW_VERSION = 3


def new_report(run_id, description, task, llm, policy, options, budget, data_hash, seed) -> JsonObject:
    report = dict(run_id=run_id, description=description, status="running", mode=llm.mode, model=llm.model,
        fault_injection=options.inject_fault or options.repair_demo, workflow_version=WORKFLOW_VERSION,
        experiment_arm=options.arm, complete_template_supplied=False, task_id=task.id,
        task_title=task.title, features=list(task.features), target=task.target,
        generation_budget_per_candidate=budget, dataset_sha256=data_hash, seed=seed, policy=deepcopy(policy),
        environment={"python": platform.python_version(), "platform": platform.platform(),
                     **{name: importlib.metadata.version(name) for name in ("numpy", "pandas", "scikit-learn", "joblib")}},
        knowledge=[], experiences=[], candidates=[])
    if hasattr(llm, "provenance"):
        report["response_provenance"] = deepcopy(llm.provenance)
    return report


def write_report(folder: Path, report: JsonObject) -> None:
    write_json(folder / "report.json", report)
    (folder / "report.md").write_text(report_markdown(report), encoding="utf-8")


def finish_run(store, report: JsonObject, session: RunSession) -> None:
    llm = session.llm
    if hasattr(llm, "calls"):
        report["llm_calls"] = deepcopy(llm.calls)
    if session.checkpoints:
        report["checkpoint_hits"] = deepcopy(session.checkpoints.hits)
    report["structured_corrections"] = {
        role: sum(e["role"] == role + "_rejected" for e in session.events) for role in ("understand", "plan")}
    try:
        # Graph writes roll back together if recording or graph rendering fails.
        # Filesystem publication and SQLite are not a distributed transaction.
        with store.transaction():
            store.save_run(report)
            store.export(session.folder / "knowledge_graph.json")
            graph_html(session.folder / "knowledge_graph.json", session.folder / "knowledge_graph.html")
    except Exception as exc:
        report.update(status="persistence_failed", failure_kind="persistence",
                      error=f"{type(exc).__name__}: {exc}")
        for name in ("knowledge_graph.json", "knowledge_graph.html"):
            (session.folder / name).unlink(missing_ok=True)
        session.record("persistence_error", {}, {"error": report["error"]})
    else:
        session.record("persist", {"status": report["status"]}, {"graph": "knowledge_graph.json"})
    write_report(session.folder, report)
