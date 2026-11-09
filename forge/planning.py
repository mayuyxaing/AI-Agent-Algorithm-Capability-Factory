"""Typed task/plan contracts and bounded, train-data-driven planning."""
from .data import FEATURES, NUMERIC, CATEGORICAL, TARGET

CONSTRAINTS = {"train_only_preprocessing", "unknown_categories", "missing_values", "class_imbalance"}
PARAMETERS = {
    "logistic": {"C": (float, .01, 100), "max_iter": (int, 100, 3000),
                 "class_weight": [None, "balanced"], "solver": ["lbfgs", "liblinear"]},
    "forest": {"n_estimators": (int, 20, 200), "max_depth": (int, 2, 12),
               "min_samples_leaf": (int, 1, 20), "class_weight": [None, "balanced"]},
}
DEFAULTS = {
    "logistic": {"C": 1.0, "max_iter": 1000, "class_weight": "balanced", "solver": "lbfgs"},
    "forest": {"n_estimators": 100, "max_depth": 7, "min_samples_leaf": 5, "class_weight": "balanced"},
}
PREPROCESSING = {"numeric_imputer": ["median", "mean"],
                 "categorical_imputer": ["most_frequent", "constant"],
                 "scale_numeric": [True, False], "unknown_categories": ["ignore"]}
INTERFACE = {
    "build_model": "build_model(seed=42) returns an unfitted sklearn.pipeline.Pipeline",
    "predict": "predict(model, frame) returns finite 1D positive-class probabilities in input row order, values [0,1]",
    "missing_fields": "raise ValueError when a required feature is absent",
    "features": FEATURES, "numeric": NUMERIC, "categorical": CATEGORICAL,
    "training": "The host fits on training data only. Do not load files, train, or evaluate at module import.",
}


def profile_training(train):
    """Aggregate training data only; no evaluation rows/labels enter prompts."""
    columns = {}
    for column in FEATURES:
        series = train[column]
        item = {"kind": "numeric" if column in NUMERIC else "categorical",
                "missing_rate": round(float(series.isna().mean()), 6),
                "cardinality": int(series.nunique(dropna=True)), "constant": bool(series.nunique() <= 1)}
        if column in NUMERIC and series.notna().any():
            item["quantiles"] = {str(q): float(series.quantile(q)) for q in (.25, .5, .75)}
        columns[column] = item
    counts = train[TARGET].value_counts()
    minority_rate = float(counts.min() / len(train))
    requirements = ["train_only_preprocessing", "unknown_categories"]
    if any(item["missing_rate"] > 0 for item in columns.values()):
        requirements.append("missing_values")
    if minority_rate < .4:
        requirements.append("class_imbalance")
    return {"scope": "training_only", "rows": len(train), "columns": columns,
            "positive_rate": float(train[TARGET].mean()), "minority_rate": minority_rate,
            "requirements": requirements}


def validate_spec(spec, description):
    if spec.get("task") != "churn_classification" or spec.get("target") != "churn":
        raise ValueError("Only churn_classification with target churn is supported")
    if not isinstance(spec.get("metrics"), list) or not spec["metrics"] or not set(spec["metrics"]) <= {"roc_auc", "f1", "accuracy"}:
        raise ValueError("Task metrics must name supported metrics")
    constraints = spec.get("constraints")
    if not isinstance(constraints, list) or any(not isinstance(c, str) or c not in CONSTRAINTS for c in constraints):
        raise ValueError("Task constraints must be real enum values; placeholders such as 'strings' are invalid")
    cues = {"missing_values": ["缺失", "missing"], "unknown_categories": ["未知类别", "unknown categor", "unseen categor"],
            "class_imbalance": ["不平衡", "imbalan"]}
    for constraint, words in cues.items():
        if any(word in description.lower() for word in words) and constraint not in constraints:
            raise ValueError(f"Explicit requirement was not extracted: {constraint}")


def validate_plan(plan, plugins, knowledge, experiences, constraints=()):
    candidates = plan.get("candidates")
    if not isinstance(candidates, list) or not 1 <= len(candidates) <= 3:
        raise ValueError("Plan must contain 1 to 3 candidates")
    ids = [candidate.get("algorithm") for candidate in candidates]
    if len(set(ids)) != len(ids) or not set(ids) <= set(plugins):
        raise ValueError("Duplicate or unregistered candidate algorithms")
    preprocessing = plan.get("preprocessing", {})
    if set(preprocessing) != set(PREPROCESSING):
        raise ValueError("Plan must specify all preprocessing choices")
    for name, choices in PREPROCESSING.items():
        if preprocessing[name] not in choices or (name == "scale_numeric" and not isinstance(preprocessing[name], bool)):
            raise ValueError(f"Unsupported preprocessing choice: {name}")
    for candidate in candidates:
        if not isinstance(candidate.get("rationale"), str) or not candidate["rationale"].strip():
            raise ValueError("Each candidate needs a rationale")
        params = candidate.get("hyperparameters")
        rules = PARAMETERS.get(candidate["algorithm"], {})
        if not isinstance(params, dict) or set(params) != set(rules):
            raise ValueError("Candidate hyperparameters do not match the bounded search space")
        if "class_imbalance" in constraints and params.get("class_weight") != "balanced":
            raise ValueError("Explicit class-imbalance requirement needs class_weight='balanced' in this bounded implementation")
        for name, rule in rules.items():
            value = params[name]
            if isinstance(rule, tuple):
                kind, lower, upper = rule
                if isinstance(value, bool) or not isinstance(value, (int, float) if kind is float else int) or not lower <= value <= upper:
                    raise ValueError(f"Hyperparameter out of bounds: {name}")
            elif value not in rule:
                raise ValueError(f"Unsupported hyperparameter: {name}")
    for field, source in [("evidence_ids", knowledge), ("experience_ids", experiences)]:
        values = plan.get(field, [])
        if not isinstance(values, list) or not set(values) <= {item["id"] for item in source}:
            raise ValueError(f"Plan cites unknown {field}")


def planning_catalog(registered):
    """Expose bounded choices, not constructor expressions or complete templates."""
    return [{"id": key, "name": value["name"], "rationale": value["rationale"],
             "hyperparameter_defaults": DEFAULTS[key],
             "hyperparameter_space": {name: [rule[0].__name__, rule[1], rule[2]] if isinstance(rule, tuple) else rule
                                      for name, rule in PARAMETERS[key].items()}}
            for key, value in registered.items() if key in PARAMETERS]
