"""Explicit role handoffs with bounded repairs and independent evaluation."""
import importlib.metadata
import platform
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from .data import FEATURES, NUMERIC, CATEGORICAL, load_split
from .llm import LLMProviderError, plugins, structured_call
from .reporting import report_markdown, graph_html
from .util import digest, write_json
from .validation import evaluate
from .checkpoints import ResponseCheckpoints
from .planning import INTERFACE, planning_catalog, profile_training, validate_spec, validate_plan, plan_schema
from .structured import checked_response, generation_payload


def run(description, dataset: Path, store, llm, policy: dict, output: Path,
        inject_fault=False, seed=42, checkpoint_dir=None, experiment_arm="knowledge_repair", repair_demo=False,
        resample_budget=1):
    from .tasks import get_task
    from .llm import plugins as registered_plugins
    task = get_task(policy.get("task", "churn_classification"))
    FEATURES, NUMERIC, CATEGORICAL, INTERFACE = task.features, task.numeric, task.categorical, task.interface
    load_split, profile_training = task.load_split, task.profile
    validate_spec, validate_plan = task.validate_spec, task.validate_plan
    planning_catalog, plan_schema = task.catalog, task.schema
    plugins = lambda: registered_plugins(task.id)
    if task.id != "churn_classification" and (repair_demo or inject_fault):
        raise ValueError("Unknown-category fault demo is specific to the churn task")
    if experiment_arm not in {"direct", "knowledge", "knowledge_repair"}:
        raise ValueError("Unknown experiment arm")
    if not 1 <= resample_budget <= 6:
        raise ValueError("Resample budget must be between 1 and 6")
    policy = dict(policy)
    if experiment_arm != "knowledge_repair":
        policy["max_repairs"] = 0
    generation_budget = policy["max_repairs"] + 1 if experiment_arm == "knowledge_repair" else resample_budget
    if repair_demo and not policy["max_repairs"]:
        raise ValueError("Repair demo requires a repair budget")
    if inject_fault and llm.mode != "mock":
        raise ValueError("Demonstration fault injection is supported only in mock mode")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    folder = output / run_id
    folder.mkdir(parents=True)
    events = []
    checkpoints = ResponseCheckpoints(checkpoint_dir, {"mode": llm.mode, "model": llm.model,
        "dataset_sha256": digest(dataset.read_bytes()), "seed": seed, "policy": policy,
        "fault_injection": inject_fault, "repair_demo": repair_demo,
        "experiment_arm": experiment_arm, "generation_budget": generation_budget, "workflow_version": 2,
        "task_id": task.id, "task_contract": task.interface}) if checkpoint_dir else None

    def ask(role, context):
        cached = checkpoints.lookup(role, context) if checkpoints else None
        if cached is not None:
            print(f"复用已保存的 {role} 模型响应；后续验证仍会执行。", flush=True)
            record(role + "_response", {"context": context, "source": "checkpoint"}, cached)
            return cached
        print(f"正在执行 {role}……", flush=True)
        result = structured_call(llm, role, context)
        record(role + "_response", {"context": context, "source": getattr(llm, "response_source", "live")}, result)
        return result

    def record(role, incoming, outgoing):
        event = {"step": len(events), "role": role, "input": incoming, "output": outgoing}
        events.append(event)
        write_json(folder / "events.json", events)
        if checkpoints and role in {"understand", "plan", "generate", "repair"}:
            if role not in {"generate", "repair"} or outgoing.get("code"):
                checkpoints.save(role, incoming, outgoing, run_id)

    report = {"run_id": run_id, "description": description, "status": "running",
              "mode": llm.mode, "model": llm.model, "fault_injection": inject_fault or repair_demo,
              "workflow_version": 2, "experiment_arm": experiment_arm, "complete_template_supplied": False,
              "task_id": task.id, "task_title": task.title, "features": FEATURES, "target": task.target,
              "generation_budget_per_candidate": generation_budget,
              "dataset_sha256": digest(dataset.read_bytes()), "seed": seed, "policy": policy,
              "environment": {"python": platform.python_version(), "platform": platform.platform(),
                              **{name: importlib.metadata.version(name) for name in ("numpy", "pandas", "scikit-learn", "joblib")}},
              "knowledge": [], "experiences": [], "candidates": []}
    if hasattr(llm, "provenance"):
        report["response_provenance"] = llm.provenance
    try:
        train, valid, test = load_split(dataset, seed)
        profile = profile_training(train)
        report["data_profile"] = profile
        record("profile", {"scope": "training_only"}, profile)
        report["split"] = {"train": len(train), "validation": len(valid), "test": len(test),
                           "train_positive_rate": float(train[task.target].mean())}
        from .validation import METRICS
        understand_context = {"description": description, "task_id": task.id, "metrics_available": list(METRICS)}
        spec = checked_response("understand", understand_context, ask,
                                lambda value: validate_spec(value, description), record)
        report["task_spec"] = spec
        record("understand", understand_context, spec)
        knowledge = store.retrieve_grounded(description, profile["requirements"] + spec["constraints"]) if experiment_arm != "direct" else []
        if not knowledge and experiment_arm != "direct":
            raise ValueError("No relevant knowledge found; ingest materials first")
        experiences = store.experiences(task.id) if experiment_arm != "direct" else []
        report.update(knowledge=knowledge, experiences=experiences)
        report["knowledge_ingestion"] = store.latest_ingestion()
        record("retrieve", {"query": description}, {"knowledge": knowledge, "experiences": experiences})
        context = {"task_id": task.id, "spec": spec, "data_profile": profile, "knowledge": knowledge,
                   "experiences": experiences, "plugins": planning_catalog(plugins()), "output_schema": plan_schema()}
        def check_plan(value):
            validate_plan(value, plugins(), knowledge, experiences, constraints=spec["constraints"])
            candidates = value["candidates"]
            if any(word in description.lower() for word in ("比较", "compare")) and len(candidates) < 2:
                raise ValueError("Comparison requested: include logistic and forest as separate candidate objects")
            if repair_demo and not any(c["algorithm"] == "logistic" for c in candidates):
                raise ValueError("Repair demonstration requires a logistic candidate")
        plan = checked_response("plan", context, ask, check_plan, record)
        candidates = plan.get("candidates", [])
        if any(word in description.lower() for word in ("比较", "compare")) and len(candidates) < 2:
            raise ValueError("Comparison was requested; plan must contain at least two candidates")
        if repair_demo and not any(c.get("algorithm") == "logistic" for c in candidates):
            raise ValueError("Repair demonstration requires a logistic candidate")
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
        constant = np.full(len(valid), float(train[task.target].mean()))
        report["baseline"] = {name: METRICS[name](valid[task.target], constant) for name in policy["metrics"]}
        for candidate in candidates:
            algorithm = candidate["algorithm"]
            entry = {"algorithm": algorithm, "rationale": str(candidate.get("rationale", "")),
                     "hyperparameters": candidate["hyperparameters"], "preprocessing": plan["preprocessing"],
                     "attempts": [], "passed": False}
            report["candidates"].append(entry)
            previous = feedback = previous_response = None
            for attempt in range(generation_budget):
                from .failures import analyze_failure
                role = "repair" if attempt > 0 and experiment_arm == "knowledge_repair" else "generate"
                incoming = {"task_id": task.id, "algorithm": algorithm, "spec": spec, "plan": plan,
                            "candidate_contract": {key: plugins()[algorithm][key] for key in ("id", "estimator", "seeded")},
                            "active_hyperparameters": candidate["hyperparameters"],
                            "knowledge": knowledge, "experiences": experiences,
                            "data_profile": profile,
                            "features": {"numeric": NUMERIC, "categorical": CATEGORICAL},
                            "interface_contract": INTERFACE,
                            "output_schema": {"type": "object", "required": ["code", "explanation"],
                                              "properties": {"code": {"type": "string", "minLength": 1, "maxLength": 100000},
                                                             "explanation": {"type": "string"}}},
                            "inject_fault": inject_fault and algorithm == "logistic" and attempt == 0,
                            "previous_code": previous if role == "repair" else None,
                            "feedback": feedback if role == "repair" else None,
                            "repair_guidance": analyze_failure(feedback) if role == "repair" and feedback else None,
                            "previous_response": previous_response if role == "repair" else None,
                            "independent_sample": attempt if experiment_arm != "knowledge_repair" else None}
                try:
                    response = ask(role, incoming)
                    previous_response = response
                    generation = generation_payload(response)
                    if generation is not response:
                        record("generation_normalized", {"role": role, "envelope": "answer"},
                               {"code_sha256": digest(generation["code"]), "source_edited": False})
                    code = generation["code"]
                except LLMProviderError:
                    raise
                except Exception as exc:
                    generation = {"code": "", "explanation": f"Generation contract failure: {type(exc).__name__}: {exc}"}
                    code = ""
                record(role, incoming, generation)
                attempt_folder = folder / algorithm / f"attempt_{attempt}"
                attempt_folder.mkdir(parents=True)
                code_path = attempt_folder / "algorithm.py"
                source_code_sha256 = digest(code)
                host_injected = repair_demo and algorithm == "logistic" and attempt == 0
                if host_injected:
                    from .faults import inject_unknown_category_fault
                    (attempt_folder / "model_response.py").write_text(code, encoding="utf-8")
                    code = inject_unknown_category_fault(code)
                code_path.write_text(code, encoding="utf-8")
                candidate_policy = dict(policy, _expected_plan={"algorithm": algorithm,
                                        "hyperparameters": candidate["hyperparameters"], "preprocessing": plan["preprocessing"]})
                feedback = (evaluate(code_path, train, valid, candidate_policy, seed) if code else
                            {"passed": False, "failure_kind": "generation_contract", "checks": {"generation_contract": False},
                             "metrics": {}, "error": generation["explanation"]})
                record("validate", {"code_sha256": digest(code), "split": "validation"}, feedback)
                entry["attempts"].append(dict(feedback, attempt=attempt, code_path=code_path.relative_to(folder).as_posix(),
                                              code_sha256=digest(code), injected=incoming["inject_fault"] or host_injected,
                                              fault_origin="host_unknown_category" if host_injected else "mock_template" if incoming["inject_fault"] else None,
                                              model_response_sha256=source_code_sha256,
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
            selected_plan = next(c for c in candidates if c["algorithm"] == selected["algorithm"])
            final_policy = dict(policy, _expected_plan={"algorithm": selected["algorithm"],
                                "hyperparameters": selected_plan["hyperparameters"], "preprocessing": plan["preprocessing"]})
            final = evaluate(final_dir / "algorithm.py", train, test, final_policy, seed)
            report["test_result"] = final
            report["status"] = "passed" if final["passed"] else "test_failed"
            record("final_evaluation", {"selected": report["selected"], "split": "test"}, final)
        else:
            report["status"] = "no_valid_candidate"
    except Exception as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        report["failure_kind"] = "provider" if isinstance(exc, LLMProviderError) else "contract_or_workflow"
        record("workflow_error", {}, {"error": report["error"]})
    if hasattr(llm, "calls"):
        report["llm_calls"] = list(llm.calls)
    if checkpoints:
        report["checkpoint_hits"] = checkpoints.hits
    report["structured_corrections"] = {role: sum(e["role"] == role + "_rejected" for e in events)
                                        for role in ("understand", "plan")}
    write_json(folder / "report.json", report)
    (folder / "report.md").write_text(report_markdown(report), encoding="utf-8")
    store.save_run(report)
    store.export(folder / "knowledge_graph.json")
    graph_html(folder / "knowledge_graph.json", folder / "knowledge_graph.html")
    record("persist", {"status": report["status"]}, {"graph": "knowledge_graph.json"})
    return folder, report
