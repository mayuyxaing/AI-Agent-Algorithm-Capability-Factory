"""Independent host-side metrics, gates and subprocess lifecycle."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from .data import FEATURES, TARGET
from .safety import inspect_code
from .util import read_json, write_json

METRICS = {
    "roc_auc": lambda y, p: float(roc_auc_score(y, p)),
    "f1": lambda y, p: float(f1_score(y, p >= .5, zero_division=0)),
    "accuracy": lambda y, p: float(accuracy_score(y, p >= .5)),
}


def evaluate(code_path: Path, train, evaluation, policy: dict, seed: int):
    started = time.perf_counter()
    issues = inspect_code(code_path.read_text(encoding="utf-8"))
    if issues:
        return dict(passed=False, failure_kind="static_check", error="; ".join(issues), checks={"static_check": False})
    folder = code_path.parent.resolve()
    (folder / "result.json").unlink(missing_ok=True)
    payload = {"train": json.loads(train[FEATURES + [TARGET]].to_json(orient="records")),
               "evaluation": json.loads(evaluation[FEATURES].to_json(orient="records")), "seed": seed,
               "expected_plan": policy.get("_expected_plan")}
    write_json(folder / "input.json", payload)
    env = {k: v for k, v in os.environ.items() if not any(s in k.upper() for s in ("TOKEN", "SECRET", "PASSWORD", "API_KEY"))}
    env.update(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONIOENCODING="utf-8")
    try:
        proc = subprocess.run([sys.executable, "-I", str(Path(__file__).with_name("worker.py").resolve()),
                               str(code_path.resolve()), str(folder / "input.json")],
                              cwd=folder, env=env, capture_output=True, timeout=policy["timeout_seconds"])
    except subprocess.TimeoutExpired:
        return dict(passed=False, failure_kind="timeout", error="Execution exceeded configured timeout", checks={"timeout": False})
    (folder / "stdout.log").write_bytes(proc.stdout)
    (folder / "stderr.log").write_bytes(proc.stderr)
    if proc.returncode or not (folder / "result.json").exists():
        return dict(passed=False, failure_kind="worker_exit", error=proc.stderr.decode("utf-8", errors="replace")[-3000:], checks={})
    try:
        result = read_json(folder / "result.json")
        result["wall_seconds"] = round(time.perf_counter() - started, 4)
        if not result["executed"]:
            return dict(result, passed=False, failure_kind="execution")
        import numpy as np
        probabilities = np.asarray(result.pop("probabilities"), dtype=float)
        if probabilities.shape != (len(evaluation),) or not np.isfinite(probabilities).all() or ((probabilities < 0) | (probabilities > 1)).any():
            raise ValueError("Invalid worker prediction output")
        scores = {name: METRICS[name](evaluation[TARGET], probabilities) for name in policy["metrics"]}
        result["metrics"] = scores
        for name, minimum in policy["minimum"].items():
            result["checks"]["minimum_" + name] = scores[name] >= minimum
        result["passed"] = all(result["checks"].values())
        if not result["passed"]:
            result.update(failure_kind="validation", error=json.dumps(result.get("check_errors", {}), ensure_ascii=False) +
                          " Failed checks: " + str([k for k, v in result["checks"].items() if not v]))
        return result
    except (ValueError, KeyError, TypeError) as exc:
        return dict(passed=False, failure_kind="invalid_worker_output", error=str(exc), checks={})


def load_policy(path: Path):
    policy = read_json(path)
    if policy.get("task") != "churn_classification":
        raise ValueError("Only churn_classification validator is implemented")
    if not policy.get("metrics") or any(m not in METRICS for m in policy["metrics"]):
        raise ValueError("Unknown or empty metric registry selection")
    if policy["selection_metric"] not in policy["metrics"]:
        raise ValueError("selection_metric must be evaluated")
    if not set(policy["minimum"]) <= set(policy["metrics"]):
        raise ValueError("All metric gates must be evaluated")
    if any(not 0 <= value <= 1 for value in policy["minimum"].values()):
        raise ValueError("Metric thresholds must be in [0,1]")
    if not 1 <= policy["timeout_seconds"] <= 600 or not 0 <= policy["max_repairs"] <= 5:
        raise ValueError("Invalid timeout or repair budget")
    return policy
