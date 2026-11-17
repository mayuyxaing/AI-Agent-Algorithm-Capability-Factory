"""Bounded candidate generation, feedback and validation; no holdout access."""
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
import pandas as pd
from .contracts import CandidatePlan, ImplementationPlan, JsonObject, RunOptions, TaskPlugin, TaskSpecification, ValidationOutcome
from .failures import analyze_failure
from .session import RunSession
from .structured import generation_payload
from .util import digest


@dataclass
class CandidateExecutor:
    session: RunSession
    task: TaskPlugin
    catalog: JsonObject
    policy: JsonObject
    options: RunOptions
    budget: int
    seed: int
    evaluate: Callable

    @staticmethod
    def new_result(candidate: CandidatePlan, plan: ImplementationPlan) -> JsonObject:
        return dict(algorithm=candidate.algorithm, rationale=candidate.rationale,
                    hyperparameters=candidate.hyperparameters, preprocessing=plan.preprocessing,
                    attempts=[], passed=False)

    def execute(self, candidate: CandidatePlan, plan: ImplementationPlan,
                spec: TaskSpecification, profile: JsonObject, knowledge: list,
                experiences: list, train: pd.DataFrame, valid: pd.DataFrame,
                entry: JsonObject | None = None) -> JsonObject:
        entry = self.new_result(candidate, plan) if entry is None else entry
        previous = feedback = previous_response = None
        for attempt in range(self.budget):
            role = "repair" if attempt and self.options.arm == "knowledge_repair" else "generate"
            incoming = self._context(candidate, plan, spec, profile, knowledge, experiences,
                                     attempt, role, previous, feedback, previous_response)
            # Adapter/programming exceptions propagate to the workflow boundary.
            # Only an invalid returned payload consumes the model repair budget.
            response = self.session.ask(role, incoming)
            generation = self._generation(response, role)
            self.session.record(role, incoming, generation)
            code, code_path, injected = self._save_source(candidate.algorithm, attempt, generation["code"])
            feedback = self._validate(code, code_path, candidate, plan, generation, train, valid)
            self.session.record("validate", {"code_sha256": digest(code), "split": "validation"}, feedback)
            entry["attempts"].append(dict(feedback, attempt=attempt,
                code_path=code_path.relative_to(self.session.folder).as_posix(), code_sha256=digest(code),
                injected=incoming["inject_fault"] or injected,
                fault_origin="host_unknown_category" if injected else "mock_template" if incoming["inject_fault"] else None,
                model_response_sha256=digest(generation["code"]), explanation=generation.get("explanation", "")))
            previous, previous_response = code, response
            if feedback["passed"]:
                entry["passed"] = True
                break
        return entry

    def _context(self, candidate, plan, spec, profile, knowledge, experiences,
                 attempt, role, previous, feedback, previous_response):
        return dict(task_id=self.task.id, algorithm=candidate.algorithm, spec=spec.to_dict(), plan=plan.to_dict(),
            candidate_contract={k: self.catalog[candidate.algorithm][k] for k in ("id", "estimator", "seeded")},
            active_hyperparameters=candidate.hyperparameters, knowledge=knowledge, experiences=experiences,
            data_profile=profile, features={"numeric": self.task.numeric, "categorical": self.task.categorical},
            interface_contract=self.task.interface,
            output_schema={"type": "object", "required": ["code", "explanation"], "properties": {
                "code": {"type": "string", "minLength": 1, "maxLength": 100000}, "explanation": {"type": "string"}}},
            inject_fault=self.options.inject_fault and candidate.algorithm == "logistic" and attempt == 0,
            previous_code=previous if role == "repair" else None,
            feedback=feedback if role == "repair" else None,
            repair_guidance=analyze_failure(feedback) if role == "repair" and feedback else None,
            previous_response=previous_response if role == "repair" else None,
            independent_sample=attempt if self.options.arm != "knowledge_repair" else None)

    def _generation(self, response: JsonObject, role: str) -> JsonObject:
        try:
            generation = generation_payload(response)
        except ValueError as exc:
            return {"code": "", "explanation": f"Generation contract failure: {type(exc).__name__}: {exc}"}
        if generation is not response:
            self.session.record("generation_normalized", {"role": role, "envelope": "answer"},
                                {"code_sha256": digest(generation["code"]), "source_edited": False})
        return generation

    def _save_source(self, algorithm: str, attempt: int, source: str) -> tuple[str, Path, bool]:
        folder = self.session.folder / algorithm / f"attempt_{attempt}"
        folder.mkdir(parents=True)
        injected = self.options.repair_demo and algorithm == "logistic" and attempt == 0
        code = source
        if injected:
            from .faults import inject_unknown_category_fault
            (folder / "model_response.py").write_text(source, encoding="utf-8")
            code = inject_unknown_category_fault(source)
        path = folder / "algorithm.py"
        path.write_text(code, encoding="utf-8")
        return code, path, injected

    def _validate(self, code, path, candidate, plan, generation, train, valid):
        if not code:
            return dict(passed=False, failure_kind="generation_contract", checks={"generation_contract": False},
                        metrics={}, error=generation["explanation"])
        policy = dict(self.policy, _expected_plan={"algorithm": candidate.algorithm,
                      "hyperparameters": candidate.hyperparameters, "preprocessing": plan.preprocessing})
        result = self.evaluate(path, train, valid, policy, self.seed)
        return ValidationOutcome.from_dict(result, self.policy["metrics"]).to_dict()
