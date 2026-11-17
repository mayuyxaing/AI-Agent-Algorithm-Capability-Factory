"""Validated boundaries between model responses and host-controlled execution.

JSON remains the audit format. Internal stages use frozen records with detached
input/output dictionaries, so invalid responses cannot silently become executable
plans or passing results.
"""
from copy import deepcopy
from dataclasses import dataclass
from math import isfinite
import re
from typing import Any, Protocol

JsonObject = dict[str, Any]


class LanguageModel(Protocol):
    mode: str
    model: str

    def call(self, role: str, context: JsonObject) -> JsonObject: ...


class TaskPlugin(Protocol):
    id: str
    title: str
    target: str
    features: list[str]
    numeric: list[str]
    categorical: list[str]
    interface: JsonObject

    def load_split(self, path, seed): ...
    def profile(self, train) -> JsonObject: ...
    def validate_spec(self, spec, description) -> None: ...
    def validate_plan(self, plan, registered, knowledge, experiences, constraints=()) -> None: ...
    def catalog(self, registered) -> list[JsonObject]: ...
    def schema(self) -> JsonObject: ...


@dataclass(frozen=True)
class TaskSpecification:
    task: str
    target: str
    metrics: tuple[str, ...]
    constraints: tuple[str, ...]
    description: str

    @classmethod
    def from_dict(cls, value: JsonObject) -> "TaskSpecification":
        if not isinstance(value, dict):
            raise ValueError("Task specification must be an object")
        for name in ("task", "target"):
            if not isinstance(value.get(name), str) or not value[name]:
                raise ValueError(f"Task specification requires {name}")
        for name in ("metrics", "constraints"):
            if not isinstance(value.get(name), list) or any(not isinstance(v, str) for v in value[name]):
                raise ValueError(f"Task specification {name} must be a string array")
        if not value["metrics"]:
            raise ValueError("Task specification needs at least one metric")
        description = value.get("description", "")
        if not isinstance(description, str):
            raise ValueError("Task description must be a string")
        return cls(value["task"], value["target"], tuple(value["metrics"]), tuple(value["constraints"]), description)

    def to_dict(self) -> JsonObject:
        return dict(task=self.task, target=self.target, metrics=list(self.metrics),
                    constraints=list(self.constraints), description=self.description)


@dataclass(frozen=True)
class CandidatePlan:
    algorithm: str
    rationale: str
    _parameters: JsonObject

    @classmethod
    def from_dict(cls, value: JsonObject) -> "CandidatePlan":
        if not isinstance(value, dict):
            raise ValueError("Candidate must be an object")
        algorithm = value.get("algorithm")
        if not isinstance(algorithm, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,63}", algorithm):
            raise ValueError("Algorithm ID must be a safe directory component")
        if not isinstance(value.get("rationale"), str) or not value["rationale"].strip():
            raise ValueError("Candidate needs a nonempty rationale")
        if not isinstance(value.get("hyperparameters"), dict):
            raise ValueError("Candidate hyperparameters must be an object")
        return cls(algorithm, value["rationale"], deepcopy(value["hyperparameters"]))

    @property
    def hyperparameters(self) -> JsonObject:
        return deepcopy(self._parameters)

    def to_dict(self) -> JsonObject:
        return dict(algorithm=self.algorithm, rationale=self.rationale, hyperparameters=self.hyperparameters)


@dataclass(frozen=True)
class ImplementationPlan:
    candidates: tuple[CandidatePlan, ...]
    _preprocessing: JsonObject
    evidence_ids: tuple[str, ...]
    experience_ids: tuple[str, ...]
    experience_applied: bool

    @classmethod
    def from_dict(cls, value: JsonObject) -> "ImplementationPlan":
        if not isinstance(value, dict):
            raise ValueError("Plan must be an object")
        candidates = value.get("candidates")
        if not isinstance(candidates, list) or not 1 <= len(candidates) <= 3:
            raise ValueError("Plan must contain 1 to 3 candidates")
        parsed = tuple(CandidatePlan.from_dict(c) for c in candidates)
        if len({c.algorithm for c in parsed}) != len(parsed):
            raise ValueError("Plan contains duplicate algorithms")
        if not isinstance(value.get("preprocessing"), dict):
            raise ValueError("Plan preprocessing must be an object")
        for key in ("evidence_ids", "experience_ids"):
            if not isinstance(value.get(key), list) or any(not isinstance(v, str) for v in value[key]):
                raise ValueError(f"Plan {key} must be a string array")
        if type(value.get("experience_applied")) is not bool:
            raise ValueError("experience_applied must be boolean")
        return cls(parsed, deepcopy(value["preprocessing"]), tuple(value["evidence_ids"]),
                   tuple(value["experience_ids"]), value["experience_applied"])

    @property
    def preprocessing(self) -> JsonObject:
        return deepcopy(self._preprocessing)

    def to_dict(self) -> JsonObject:
        return dict(candidates=[c.to_dict() for c in self.candidates], preprocessing=self.preprocessing,
                    evidence_ids=list(self.evidence_ids), experience_ids=list(self.experience_ids),
                    experience_applied=self.experience_applied)


@dataclass(frozen=True)
class RunOptions:
    arm: str = "knowledge_repair"
    inject_fault: bool = False
    repair_demo: bool = False
    resample_budget: int = 1

    def generation_budget(self, policy: JsonObject, task_id: str, mode: str) -> int:
        if self.arm not in {"direct", "knowledge", "knowledge_repair"}:
            raise ValueError("Unknown experiment arm")
        if type(self.resample_budget) is not int or not 1 <= self.resample_budget <= 6:
            raise ValueError("Resample budget must be between 1 and 6")
        repairs = policy.get("max_repairs")
        if type(repairs) is not int or not 0 <= repairs <= 5:
            raise ValueError("Repair budget must be between 0 and 5")
        if task_id != "churn_classification" and (self.repair_demo or self.inject_fault):
            raise ValueError("Unknown-category fault demo is specific to the churn task")
        if self.inject_fault and mode != "mock":
            raise ValueError("Demonstration fault injection is supported only in mock mode")
        if self.repair_demo and (not repairs or self.arm != "knowledge_repair"):
            raise ValueError("Repair demo requires a repair budget")
        return repairs + 1 if self.arm == "knowledge_repair" else self.resample_budget


class ValidationContractError(RuntimeError):
    """A host evaluator violated its interface; this is not a model repair case."""


@dataclass(frozen=True)
class EvaluationPolicy:
    """One validation path for CLI configuration and programmatic run() calls."""
    task: str
    metrics: tuple[str, ...]
    selection_metric: str
    timeout_seconds: int
    max_repairs: int
    _payload: JsonObject

    @classmethod
    def from_dict(cls, value: JsonObject, available_metrics) -> "EvaluationPolicy":
        if not isinstance(value, dict) or not isinstance(value.get("task"), str):
            raise ValueError("Policy requires a task ID")
        metrics = value.get("metrics")
        if not isinstance(metrics, list) or not metrics or any(not isinstance(m, str) or m not in available_metrics for m in metrics):
            raise ValueError("Unknown or empty metric registry selection")
        selection = value.get("selection_metric")
        if not isinstance(selection, str) or selection not in metrics:
            raise ValueError("selection_metric must be evaluated")
        minimum = value.get("minimum")
        if not isinstance(minimum, dict) or not set(minimum) <= set(metrics):
            raise ValueError("All metric gates must be evaluated")
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not isfinite(v) or not 0 <= v <= 1 for v in minimum.values()):
            raise ValueError("Metric thresholds must be finite numbers in [0,1]")
        timeout, repairs = value.get("timeout_seconds"), value.get("max_repairs")
        if type(timeout) is not int or not 1 <= timeout <= 600 or type(repairs) is not int or not 0 <= repairs <= 5:
            raise ValueError("Invalid timeout or repair budget")
        return cls(value["task"], tuple(metrics), selection, timeout, repairs, deepcopy(value))

    def to_dict(self) -> JsonObject:
        return deepcopy(self._payload)


@dataclass(frozen=True)
class ValidationOutcome:
    passed: bool
    _payload: JsonObject

    @classmethod
    def from_dict(cls, value: JsonObject, required_metrics=()) -> "ValidationOutcome":
        if not isinstance(value, dict) or type(value.get("passed")) is not bool:
            raise ValidationContractError("Evaluator must return a boolean passed field")
        checks = value.get("checks")
        if not isinstance(checks, dict) or any(not isinstance(k, str) or type(v) is not bool for k, v in checks.items()):
            raise ValidationContractError("Evaluator checks must be a boolean mapping")
        if value["passed"]:
            if not checks or not all(checks.values()):
                raise ValidationContractError("Passing evaluation needs nonempty, all-passing checks")
            metrics = value.get("metrics", {})
            if not isinstance(metrics, dict):
                raise ValidationContractError("Evaluator metrics must be an object")
            for name in required_metrics:
                score = metrics.get(name)
                if isinstance(score, bool) or not isinstance(score, (int, float)) or not isfinite(score):
                    raise ValidationContractError(f"Passing evaluation lacks finite metric: {name}")
        return cls(value["passed"], deepcopy(value))

    def to_dict(self) -> JsonObject:
        return deepcopy(self._payload)
