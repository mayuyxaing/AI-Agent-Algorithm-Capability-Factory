"""Task plugin interface shared by orchestration and isolated validation."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

TASKS = {}


def register_task(task):
    for field in ("id", "title", "target", "features", "numeric", "categorical", "interface", "config", "data"):
        if not hasattr(task, field):
            raise ValueError(f"Task plugin missing {field}")
    for method in ("generate", "load_split", "profile", "validate_spec", "validate_plan", "catalog", "schema", "check_preprocessing", "variants"):
        if not callable(getattr(task, method, None)):
            raise ValueError(f"Task plugin missing method {method}")
    if task.id in TASKS:
        raise ValueError(f"Task already registered: {task.id}")
    TASKS[task.id] = task


def get_task(name="churn_classification"):
    if name not in TASKS:
        raise ValueError(f"Unknown task plugin: {name}")
    return TASKS[name]


class ChurnTask:
    id, title, target = "churn_classification", "客户流失预测", "churn"
    features = ["tenure", "monthly_charges", "support_calls", "contract", "payment_method"]
    numeric, categorical = features[:3], features[3:]
    config, data = "configs/churn.json", "data/churn.csv"

    def generate(self, path):
        from .data import generate
        generate(path)

    def load_split(self, path, seed):
        from .data import load_split
        return load_split(path, seed)

    def profile(self, train):
        from .planning import profile_training
        return profile_training(train)

    def validate_spec(self, spec, description):
        from .planning import validate_spec
        return validate_spec(spec, description)

    def validate_plan(self, *args, **kwargs):
        from .planning import validate_plan
        return validate_plan(*args, **kwargs)

    def catalog(self, registered):
        from .planning import planning_catalog
        return planning_catalog(registered)

    def schema(self):
        from .planning import plan_schema
        return plan_schema()

    def check_preprocessing(self, model, expected):
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer
        from sklearn.preprocessing import StandardScaler
        transformers = [v for v in model.get_params(deep=True).values() if isinstance(v, ColumnTransformer)]
        for names, strategy in [(self.numeric, expected["numeric_imputer"]), (self.categorical, expected["categorical_imputer"])]:
            branch = next((transformer for ct in transformers for _, transformer, columns in ct.transformers
                           if isinstance(columns, (list, tuple)) and set(columns) == set(names)), None)
            if branch is None or not hasattr(branch, "get_params"):
                raise ValueError(f"Plan mismatch: ColumnTransformer must explicitly group {names}")
            objects = [branch, *branch.get_params(deep=True).values()]
            if not any(isinstance(obj, SimpleImputer) and obj.strategy == strategy for obj in objects):
                raise ValueError(f"Plan mismatch: imputer for {names} must use {strategy}")
            if names == self.numeric and any(isinstance(obj, StandardScaler) for obj in objects) != expected["scale_numeric"]:
                raise ValueError("Plan mismatch: numeric scaling differs from plan")

    def variants(self, evaluation):
        unknown, missing = evaluation.head(5).copy(), evaluation.head(5).copy()
        unknown["contract"] = "__previously_unseen_contract__"
        for name in self.features:
            missing[name] = np.nan
        return {"unknown_category": unknown, "all_missing_values": missing}

    @property
    def interface(self):
        from .planning import INTERFACE
        return INTERFACE


class TextTask:
    id, title, target = "text_classification", "客服工单紧急程度分类", "label"
    features, numeric, categorical = ["text"], [], ["text"]
    config, data = "configs/text.json", "data/tickets.csv"
    interface = {"build_model": "build_model(seed=42) returns an unfitted sklearn Pipeline",
                 "predict": "predict(model, frame) returns 1D finite class-1 probabilities in row order",
                 "features": ["text"], "target": "label", "classes": {"0": "ordinary", "1": "urgent"},
                 "empty": "Treat missing/empty text as empty string, reject absent text column with ValueError",
                 "training_call": "host calls model.fit(train[['text']], y): X is a DataFrame of shape (n,1), not a Series; the first transformer must select X['text'] and return n strings",
                 "prediction_call": "predict(model, frame) receives the same DataFrame schema; keep training and prediction preprocessing consistent",
                 "preprocessing": "Fit character-level TF-IDF on training data only. Handle unseen vocabulary."}

    def generate(self, path, rows=800, seed=42):
        rng = np.random.default_rng(seed)
        urgent = ["账户被盗资金异常", "系统宕机业务停止", "紧急无法登录", "支付失败重复扣款"]
        normal = ["咨询产品使用说明", "查询会员积分", "修改头像操作指南", "了解营业时间"]
        records = []
        for i in range(rows):
            topic = int(rng.integers(0, 2))
            label = topic if rng.random() > .12 else 1 - topic
            text = f"{rng.choice(urgent if topic else normal)} {rng.choice(['请协助处理','需要帮助','麻烦回复','谢谢'])} 工单编号{i}"
            records.append({"text": text if rng.random() > .03 else None, "label": label})
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(records).to_csv(path, index=False)

    def load_split(self, path, seed):
        frame = pd.read_csv(path)
        if not set(self.features + [self.target]) <= set(frame.columns):
            raise ValueError("Text task requires text and label columns")
        if frame.label.isna().any() or set(frame.label.unique()) != {0, 1} or frame.label.value_counts().min() < 10:
            raise ValueError("Text labels must contain at least 10 examples of both binary classes")
        if frame.text.dropna().map(lambda v: not isinstance(v, str)).any():
            raise ValueError("Non-missing text values must be strings")
        train, rest = train_test_split(frame, test_size=.4, stratify=frame.label, random_state=seed)
        valid, test = train_test_split(rest, test_size=.5, stratify=rest.label, random_state=seed)
        return tuple(f.reset_index(drop=True) for f in (train, valid, test))

    def profile(self, train):
        lengths = train.text.fillna("").str.len()
        return {"scope": "training_only", "rows": len(train), "positive_rate": float(train.label.mean()),
                "minority_rate": float(train.label.value_counts().min() / len(train)),
                "missing_rate": float(train.text.isna().mean()), "mean_length": float(lengths.mean()),
                "requirements": ["train_only_preprocessing", "missing_values", "unseen_vocabulary"]}

    def validate_spec(self, spec, description):
        if spec.get("task") != self.id or spec.get("target") != self.target:
            raise ValueError("Text task must use task=text_classification and target=label")
        allowed = {"train_only_preprocessing", "missing_values", "unseen_vocabulary"}
        if not isinstance(spec.get("constraints"), list) or any(not isinstance(c, str) for c in spec["constraints"]) or not set(spec["constraints"]) <= allowed:
            raise ValueError("Invalid text-task constraints")
        from .validation import METRICS
        if not isinstance(spec.get("metrics"), list) or not spec["metrics"] or any(not isinstance(m, str) for m in spec["metrics"]) or not set(spec["metrics"]) <= set(METRICS):
            raise ValueError("Invalid text-task metrics")

    def catalog(self, registered):
        return [{k: p[k] for k in ("id", "name", "rationale", "defaults", "parameter_space")} for p in registered.values()]

    def validate_plan(self, plan, registered, knowledge, experiences, constraints=()):
        candidates = plan.get("candidates")
        if not isinstance(candidates, list) or not 1 <= len(candidates) <= 3:
            raise ValueError("Text plan needs a candidates array")
        ids = [c.get("algorithm") for c in candidates if isinstance(c, dict)]
        if len(ids) != len(candidates) or any(not isinstance(v, str) for v in ids) or len(set(ids)) != len(ids) or not set(ids) <= set(registered):
            raise ValueError("Unregistered or duplicate text algorithm")
        if not isinstance(plan.get("experience_applied"), bool):
            raise ValueError("experience_applied must be boolean")
        for candidate in candidates:
            validate_parameters(candidate.get("hyperparameters"), registered[candidate["algorithm"]]["parameter_space"])
            if not isinstance(candidate.get("rationale"), str) or not candidate["rationale"]:
                raise ValueError("Candidate needs a rationale")
        prep = plan.get("preprocessing", {})
        validate_parameters(prep, {"ngram_max": ["int", 1, 2], "max_features": ["int", 500, 5000]})
        for key, source in [("evidence_ids", knowledge), ("experience_ids", experiences)]:
            values = plan.get(key)
            if not isinstance(values, list) or any(not isinstance(v, str) for v in values) or not set(values) <= {v["id"] for v in source}:
                raise ValueError("Unknown text-plan evidence")

    def schema(self):
        from .llm import plugins
        variants = []
        for descriptor in plugins(self.id).values():
            properties = {}
            for name, rule in descriptor["parameter_space"].items():
                properties[name] = {"type": "integer" if rule[0] == "int" else "number", "minimum": rule[1], "maximum": rule[2]} if rule and rule[0] in ("int", "float") else {"enum": rule}
            variants.append({"type": "object", "required": ["algorithm", "rationale", "hyperparameters"], "properties": {
                "algorithm": {"const": descriptor["id"]}, "rationale": {"type": "string", "minLength": 1},
                "hyperparameters": {"type": "object", "required": list(properties), "properties": properties}}})
        return {"type": "object", "required": ["candidates", "preprocessing", "evidence_ids", "experience_ids", "experience_applied"],
                "properties": {"candidates": {"type": "array", "minItems": 1, "maxItems": 3, "items": {"oneOf": variants}},
                               "preprocessing": {"type": "object", "required": ["ngram_max", "max_features"], "properties": {
                                   "ngram_max": {"type": "integer", "minimum": 1, "maximum": 2},
                                   "max_features": {"type": "integer", "minimum": 500, "maximum": 5000}}},
                               "evidence_ids": {"type": "array", "items": {"type": "string"}},
                               "experience_ids": {"type": "array", "items": {"type": "string"}},
                               "experience_applied": {"type": "boolean"}}}

    def prompt(self, role):
        if role == "understand":
            return 'Return JSON: task="text_classification",target="label",metrics=["roc_auc","f1"],constraints chosen from ["train_only_preprocessing","missing_values","unseen_vocabulary"],description. Class 1 is urgent customer ticket, class 0 ordinary.'
        if role == "plan":
            return 'Return JSON matching output_schema. Compare text_logistic and text_nb. Each candidate has algorithm,rationale,hyperparameters matching plugin defaults/parameter_space. preprocessing has ngram_max integer1..2,max_features integer500..5000. Also provide evidence_ids,experience_ids,experience_applied. Use training data_profile. Cite only supplied IDs. On correction fix previous_response using validation_error.'
        return ('Return exactly {"code":"complete Python source","explanation":"design reasoning"}, matching output_schema. '
                'Put code at the top level, without an answer wrapper. Implement interface_contract with sklearn Pipeline: '
                'a module-level text cleaning transformer, character-level TfidfVectorizer(analyzer="char",ngram_range=(1,ngram_max),max_features=max_features), '
                'and candidate_contract.estimator with the active candidate hyperparameters. Implement only algorithm, not the other candidate in plan. '
                'Use random_state=seed only when candidate_contract.seeded is true. '
                'The host calls model.fit(DataFrame({"text":["example",None]}), y). The cleaner must select X["text"] before fillna and return one string per row. '
                'Never iterate a DataFrame as documents or construct pd.Series from a DataFrame. Define BOTH module-level build_model and predict functions. '
                'predict must return 1D class-1 probabilities; keep the pipeline input shape consistent with training. '
                'Absent text column raises ValueError. Only sklearn,numpy,pandas imports; no files/network/dunder access. '
                'Do not fit at module import. For repair use previous_code, feedback and repair_guidance, do not relax tests.')

    def mock_call(self, role, context):
        from .llm import plugins
        catalog = plugins(self.id)
        if role == "understand":
            if not any(w in context["description"].lower() for w in ("文本", "工单", "text", "ticket")):
                raise ValueError("Describe a text/ticket classification task")
            return {"task": self.id, "target": self.target, "metrics": ["roc_auc", "f1"],
                    "constraints": ["train_only_preprocessing", "missing_values", "unseen_vocabulary"], "description": context["description"]}
        if role == "plan":
            return {"candidates": [{"algorithm": p["id"], "rationale": p["rationale"], "hyperparameters": p["defaults"]} for p in catalog.values()],
                    "preprocessing": {"ngram_max": 2, "max_features": 3000},
                    "evidence_ids": [n["id"] for n in context["knowledge"]], "experience_ids": [], "experience_applied": False}
        if role in {"generate", "repair"}:
            from string import Template
            candidate = next(c for c in context["plan"]["candidates"] if c["algorithm"] == context["algorithm"])
            plugin = catalog[candidate["algorithm"]]
            params = ",".join(f"{k}={v!r}" for k, v in candidate["hyperparameters"].items())
            if plugin.get("seeded"):
                params += ",random_state=seed"
            code = Template((Path(__file__).parent / "templates/text.py.txt").read_text(encoding="utf-8")).substitute(
                model_import=plugin["import"], expression=f"{plugin['estimator']}({params})", **context["plan"]["preprocessing"])
            return {"code": code, "explanation": "Offline text template; not real LLM evidence"}
        raise ValueError("Unsupported text role")

    def check_preprocessing(self, model, expected):
        from sklearn.feature_extraction.text import TfidfVectorizer
        vectors = [v for v in model.get_params(deep=True).values() if isinstance(v, TfidfVectorizer)]
        if len(vectors) != 1:
            raise ValueError("Text plan requires one TfidfVectorizer")
        vector = vectors[0]
        if vector.analyzer != "char" or vector.ngram_range != (1, expected["ngram_max"]) or vector.max_features != expected["max_features"]:
            raise ValueError("TF-IDF preprocessing differs from text plan")

    def variants(self, evaluation):
        unknown, missing = evaluation.head(5).copy(), evaluation.head(5).copy()
        unknown["text"], missing["text"] = "全新外星词汇 zzzqwx", np.nan
        return {"unseen_vocabulary": unknown, "all_missing_values": missing}


def validate_parameters(params, space):
    if not isinstance(params, dict) or set(params) != set(space):
        raise ValueError("Parameter keys differ from registered plugin")
    for name, rule in space.items():
        value = params[name]
        if rule and rule[0] in ("int", "float"):
            kind = int if rule[0] == "int" else (int, float)
            if isinstance(value, bool) or not isinstance(value, kind) or not rule[1] <= value <= rule[2]:
                raise ValueError(f"Parameter out of bounds: {name}")
        elif value not in rule:
            raise ValueError(f"Unsupported parameter: {name}")


register_task(ChurnTask())
register_task(TextTask())


def load_installed_tasks():
    """Installed trusted distributions can expose capability_forge.tasks entry points."""
    from importlib.metadata import entry_points
    for entry in entry_points(group="capability_forge.tasks"):
        register_task(entry.load()())


load_installed_tasks()
