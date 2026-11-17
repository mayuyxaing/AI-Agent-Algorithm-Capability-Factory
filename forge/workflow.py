"""Public orchestration entry point; stages own their contracts and side effects."""
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import uuid
import numpy as np
from .candidate_execution import CandidateExecutor
from .checkpoints import ResponseCheckpoints
from .contracts import EvaluationPolicy, RunOptions
from .llm import LLMProviderError, plugins
from .provenance import execution_identity, provider_identity
from .run_artifacts import WORKFLOW_VERSION, finish_run, new_report
from .session import RunSession
from .tasks import get_task
from .util import digest
from .validation import METRICS, evaluate
from .workflow_stages import prepare_task, retrieve_context, plan_candidates, final_evaluation


def run(description, dataset: Path, store, llm, policy: dict, output: Path,
        inject_fault=False, seed=42, checkpoint_dir=None, experiment_arm="knowledge_repair", repair_demo=False,
        resample_budget=1):
    """Run one task; preserved CLI/API signature and JSON report format.

    Model contract errors can use the configured correction/repair budget.
    Provider failures stop the run. Host programming/evaluator errors stop the
    run as internal errors and must never be offered to the model as repairs.
    """
    config = EvaluationPolicy.from_dict(policy, METRICS)
    task = get_task(config.task)
    catalog = deepcopy(plugins(task.id))
    options = RunOptions(experiment_arm, inject_fault, repair_demo, resample_budget)
    budget = options.generation_budget(policy, task.id, llm.mode)
    policy = config.to_dict()
    if options.arm != "knowledge_repair":
        policy["max_repairs"] = 0
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    folder = output / run_id
    folder.mkdir(parents=True)
    data_hash = digest(dataset.read_bytes())
    identity = dict(mode=llm.mode, model=llm.model, dataset_sha256=data_hash, seed=seed, policy=policy,
                    fault_injection=inject_fault, repair_demo=repair_demo, experiment_arm=experiment_arm,
                    generation_budget=budget, workflow_version=WORKFLOW_VERSION,
                    task_id=task.id, task_contract=task.interface, execution=execution_identity(),
                    provider=provider_identity(llm), plugins=catalog)
    checkpoints = ResponseCheckpoints(checkpoint_dir, identity) if checkpoint_dir else None
    session = RunSession(folder, run_id, llm, checkpoints)
    report = new_report(run_id, description, task, llm, policy, options, budget, data_hash, seed)
    stage = "prepare"
    try:
        prepared = prepare_task(task, description, dataset, seed, session)
        report.update(data_profile=prepared.profile, task_spec=prepared.specification.to_dict(), split={
            "train": len(prepared.train), "validation": len(prepared.valid), "test": len(prepared.test),
            "train_positive_rate": float(prepared.train[task.target].mean())})
        stage = "retrieve"
        retrieved = retrieve_context(store, description, prepared, options, session)
        report.update(knowledge=retrieved.knowledge, experiences=retrieved.experiences,
                      knowledge_ingestion=store.latest_ingestion())
        stage = "plan"
        plan = plan_candidates(task, description, prepared, retrieved, catalog, options, session)
        report["plan"] = plan.to_dict()
        constant = np.full(len(prepared.valid), float(prepared.train[task.target].mean()))
        report["baseline"] = {name: METRICS[name](prepared.valid[task.target], constant) for name in policy["metrics"]}
        stage = "candidates"
        executor = CandidateExecutor(session, task, catalog, policy, options, budget, seed, evaluate)
        for candidate in plan.candidates:
            entry = executor.new_result(candidate, plan)
            report["candidates"].append(entry)
            executor.execute(candidate, plan, prepared.specification, prepared.profile,
                             retrieved.knowledge, retrieved.experiences, prepared.train, prepared.valid, entry)
        stage = "final_evaluation"
        report.update(final_evaluation(report["candidates"], plan, prepared, policy, seed, session, evaluate))
    except Exception as exc:
        if isinstance(exc, LLMProviderError):
            kind = "provider"
        elif isinstance(exc, ValueError):
            kind = "contract_or_workflow"
        else:
            kind = "internal"
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}", failure_kind=kind, error_stage=stage)
        session.record("workflow_error", {"stage": stage}, {"error": report["error"], "failure_kind": kind})
    finish_run(store, report, session)
    return folder, report
