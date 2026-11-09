"""Trusted child runner. Labels for evaluation are never supplied to generated code.

Executed via ``python -I worker.py`` in a per-attempt directory. This is process
isolation and a timeout, NOT OS-enforced filesystem/network isolation.
"""
import importlib.util
import json
import sys
import time
import traceback
import tracemalloc
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

FEATURES = ["tenure", "monthly_charges", "support_calls", "contract", "payment_method"]


def vector(value, size):
    arr = np.asarray(value, dtype=float)
    if arr.shape != (size,) or not np.isfinite(arr).all() or (arr < 0).any() or (arr > 1).any():
        raise ValueError("predict must return finite probability vector shape=(n,), values in [0,1]")
    return arr


def execute(code_path: Path, input_path: Path):
    payload = json.loads(input_path.read_text(encoding="utf-8"))
    train = pd.DataFrame(payload["train"])
    evaluation = pd.DataFrame(payload["evaluation"], columns=FEATURES)
    spec = importlib.util.spec_from_file_location("generated", code_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.build_model(seed=payload["seed"])
    from sklearn.pipeline import Pipeline
    if not isinstance(model, Pipeline):
        raise TypeError("build_model must return sklearn.pipeline.Pipeline")
    model.fit(train[FEATURES], train["churn"])
    probabilities = vector(module.predict(model, evaluation.copy()), len(evaluation))
    checks, details = {"probability_contract": True}, {}

    def check(name, operation):
        try:
            operation()
            checks[name] = True
        except Exception as exc:
            checks[name] = False
            details[name] = f"{type(exc).__name__}: {exc}"[:2000]

    def missing_column():
        try:
            module.predict(model, evaluation.drop(columns=[FEATURES[0]]))
        except ValueError:
            return
        raise AssertionError("Missing required column must raise ValueError")

    def reordered():
        np.testing.assert_allclose(probabilities, vector(module.predict(model, evaluation[FEATURES[::-1]]), len(evaluation)), atol=1e-12)

    def unknown():
        sample = evaluation.head(5).copy()
        sample["contract"] = "__previously_unseen_contract__"
        vector(module.predict(model, sample), len(sample))

    def missing_values():
        sample = evaluation.head(5).copy()
        for column in FEATURES:
            sample[column] = np.nan
        vector(module.predict(model, sample), len(sample))

    def persistence():
        joblib.dump(model, "model.joblib")
        restored = joblib.load("model.joblib")
        np.testing.assert_allclose(probabilities, vector(module.predict(restored, evaluation), len(evaluation)), atol=1e-12)

    def repeat():
        np.testing.assert_allclose(probabilities, vector(module.predict(model, evaluation.copy()), len(evaluation)), atol=1e-12)

    for name, fn in [("missing_column_rejected", missing_column), ("column_order", reordered),
                     ("unknown_category", unknown), ("all_missing_values", missing_values),
                     ("repeat_prediction", repeat), ("model_reload", persistence)]:
        check(name, fn)
    return {"probabilities": probabilities.tolist(), "checks": checks, "check_errors": details}


if __name__ == "__main__":
    started = time.perf_counter()
    tracemalloc.start()
    try:
        result = execute(Path(sys.argv[1]), Path(sys.argv[2]))
        result["executed"] = True
    except Exception:
        result = {"executed": False, "error": traceback.format_exc()[-5000:], "checks": {}}
    result["worker_seconds"] = round(time.perf_counter() - started, 4)
    result["python_peak_mb"] = round(tracemalloc.get_traced_memory()[1] / 1024 ** 2, 3)
    Path("result.json").write_text(json.dumps(result, allow_nan=False), encoding="utf-8")
