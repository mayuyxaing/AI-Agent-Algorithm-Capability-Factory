"""Persist validation runs and reusable failure experience as one graph transaction."""
import json


class RunHistory:
    def __init__(self, store):
        self.store = store

    def save_run(self, report: dict):
        with self.store.transaction():
            run_id = report["run_id"]
            with self.store.connect() as con:
                con.execute("INSERT OR REPLACE INTO runs VALUES(?,?)", (run_id, json.dumps(report, ensure_ascii=False)))
            task = report.get("task_id", "churn_classification")
            capability = self.store.node("Capability", report.get("task_title", "客户流失预测"), {
                "task": task, "input": {"format": "pandas.DataFrame", "required_columns":
                    report.get("features", ["tenure", "monthly_charges", "support_calls", "contract", "payment_method"])},
                "output": {"format": "1D probability array", "range": [0, 1], "positive_class": 1},
                "conditions": ["binary labels 0/1", "registered task feature contract", "train-only preprocessing"],
                "dependencies": ["numpy", "pandas", "scikit-learn", "joblib"]}, "capability:" + task)
            run_node = self.store.node("ValidationRun", run_id, {
                "status": report["status"], "mode": report["mode"], "model": report["model"],
                "selected": report.get("selected"), "dataset_sha256": report["dataset_sha256"],
                "seed": report["seed"], "test_metrics": report.get("test_result", {}).get("metrics", {}),
                "report_ref": f"runs/{run_id}/report.json"}, "run:" + run_id)
            environment = self.store.node("Environment", "Python execution environment", report["environment"])
            self.store.edge(run_node, "EXECUTED_IN", environment)
            for metric in report["policy"]["metrics"]:
                mid = self.store.node("Metric", metric, {"task": task}, "metric:" + task + ":" + metric)
                self.store.edge(capability, "EVALUATED_BY", mid)
            for evidence in report["knowledge"]:
                self.store.edge(run_node, "USED_KNOWLEDGE", evidence["id"])
            for experience in report["experiences"]:
                self.store.edge(run_node, "REUSED_EXPERIENCE", experience["id"])
            for candidate in report["candidates"]:
                for attempt in candidate["attempts"]:
                    version = self.store.node("AlgorithmVersion", candidate["algorithm"], {
                        "sha256": attempt["code_sha256"], "run_id": run_id,
                        "attempt": attempt["attempt"], "passed": attempt["passed"],
                        "path": attempt["code_path"], "validation_metrics": attempt.get("metrics", {}),
                        "hyperparameters": candidate.get("hyperparameters"), "preprocessing": candidate.get("preprocessing"),
                        "checks": attempt.get("checks", {})}, "version:" + run_id + ":" + candidate["algorithm"] + ":" + str(attempt["attempt"]))
                    self.store.edge(capability, "IMPLEMENTED_BY", version)
                    self.store.edge(version, "VALIDATED_IN", run_node)
                    if not attempt["passed"]:
                        from .failures import analyze_failure
                        analysis = analyze_failure(attempt)
                        failure = self.store.node("FailureCase", attempt.get("failure_kind", "validation_failure"), {
                            "task": task, "error": attempt.get("error", ""),
                            "checks": attempt.get("checks", {}), "injected": attempt["injected"],
                            "resolved_in_run": candidate["passed"],
                            **analysis})
                        self.store.edge(version, "EXPOSED", failure)
            return run_node
