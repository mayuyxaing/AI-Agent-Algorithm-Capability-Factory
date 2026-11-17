"""Task understanding, grounded planning and final holdout evaluation."""
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
import shutil
import pandas as pd
from .contracts import ImplementationPlan, JsonObject, RunOptions, TaskSpecification, ValidationOutcome
from .structured import checked_response
from .session import RunSession
from .validation import METRICS


@dataclass
class PreparedTask:
    train: pd.DataFrame
    valid: pd.DataFrame
    test: pd.DataFrame
    profile: JsonObject
    specification: TaskSpecification


@dataclass
class RetrievedContext:
    knowledge: list[JsonObject]
    experiences: list[JsonObject]


def prepare_task(task, description: str, dataset: Path, seed: int, session: RunSession) -> PreparedTask:
    train, valid, test = task.load_split(dataset, seed)
    profile = task.profile(train)
    session.record("profile", {"scope": "training_only"}, profile)
    context = {"description": description, "task_id": task.id, "metrics_available": list(METRICS),
               "output_schema": {"type": "object", "required": ["task", "target", "metrics", "constraints", "description"],
                   "properties": {"task": {"type": "string"}, "target": {"type": "string"},
                       "metrics": {"type": "array", "minItems": 1, "items": {"enum": list(METRICS)}},
                       "constraints": {"type": "array", "items": {"type": "string"}},
                       "description": {"type": "string"}}},
               "output_instruction": "Return the fields required by output_schema at the JSON root."}

    def validate(value):
        TaskSpecification.from_dict(value)
        task.validate_spec(value, description)

    raw = checked_response("understand", context, session.ask, validate, session.record)
    specification = TaskSpecification.from_dict(raw)
    session.record("understand", context, specification.to_dict())
    return PreparedTask(train, valid, test, profile, specification)


def retrieve_context(store, description: str, prepared: PreparedTask,
                     options: RunOptions, session: RunSession) -> RetrievedContext:
    requirements = prepared.profile["requirements"] + list(prepared.specification.constraints)
    knowledge = store.retrieve_grounded(description, requirements) if options.arm != "direct" else []
    if not knowledge and options.arm != "direct":
        raise ValueError("No relevant knowledge found; ingest materials first")
    experiences = store.experiences(prepared.specification.task) if options.arm != "direct" else []
    session.record("retrieve", {"query": description}, {"knowledge": knowledge, "experiences": experiences})
    return RetrievedContext(knowledge, experiences)


def plan_candidates(task, description: str, prepared: PreparedTask, retrieved: RetrievedContext,
                    catalog: JsonObject, options: RunOptions, session: RunSession) -> ImplementationPlan:
    context = dict(task_id=task.id, spec=prepared.specification.to_dict(), data_profile=prepared.profile,
                   knowledge=retrieved.knowledge, experiences=retrieved.experiences,
                   plugins=task.catalog(catalog), output_schema=task.schema())

    def validate(value):
        plan = ImplementationPlan.from_dict(value)
        task.validate_plan(value, catalog, retrieved.knowledge, retrieved.experiences,
                           constraints=prepared.specification.constraints)
        if any(w in description.lower() for w in ("比较", "compare")) and len(plan.candidates) < 2:
            raise ValueError("Comparison requested: include at least two separate candidate objects")
        if options.repair_demo and not any(c.algorithm == "logistic" for c in plan.candidates):
            raise ValueError("Repair demonstration requires a logistic candidate")

    raw = checked_response("plan", context, session.ask, validate, session.record)
    plan = ImplementationPlan.from_dict(raw)
    session.record("plan", context, plan.to_dict())
    return plan


def final_evaluation(candidates: list[JsonObject], plan: ImplementationPlan, prepared: PreparedTask,
                     policy: JsonObject, seed: int, session: RunSession, evaluate: Callable) -> JsonObject:
    passing = [c for c in candidates if c["passed"]]
    if not passing:
        return {"status": "no_valid_candidate"}
    selected = max(passing, key=lambda c: c["attempts"][-1]["metrics"][policy["selection_metric"]])
    folder = session.folder / "selected"
    folder.mkdir()
    shutil.copy2(session.folder / selected["attempts"][-1]["code_path"], folder / "algorithm.py")
    candidate = next(c for c in plan.candidates if c.algorithm == selected["algorithm"])
    final_policy = dict(policy, _expected_plan={"algorithm": candidate.algorithm,
                        "hyperparameters": candidate.hyperparameters, "preprocessing": plan.preprocessing})
    # Holdout results are terminal; they never enter generation or candidate selection.
    result = evaluate(folder / "algorithm.py", prepared.train, prepared.test, final_policy, seed)
    final = ValidationOutcome.from_dict(result, policy["metrics"])
    session.record("final_evaluation", {"selected": candidate.algorithm, "split": "test"}, final.to_dict())
    return {"selected": candidate.algorithm, "test_result": final.to_dict(),
            "status": "passed" if final.passed else "test_failed"}
