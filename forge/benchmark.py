"""Matched-data ablations. Mock campaigns verify plumbing, not LLM benefit."""
import json
import importlib.metadata
import platform
from pathlib import Path
from .data import generate
from .knowledge import KnowledgeStore
from .util import digest, read_json, write_json
from .workflow import run
from .llm import OllamaLLM, plugins

ARMS = ["direct", "knowledge", "knowledge_repair"]
CASES = {
    "baseline": {"missing_rate": .035},
    "missing_high": {"missing_rate": .25},
    "minority": {"missing_rate": .08, "logit_shift": -1.2, "interaction_strength": .5},
}
DESCRIPTION = "客户流失预测，处理缺失值、未知类别和类别不平衡，比较逻辑回归和随机森林，使用 ROC-AUC 和 F1 评价。"


def execution_identity():
    """Reject resume when validators, templates, plugins or dependencies changed."""
    root = Path(__file__).parent
    files = {p.relative_to(root).as_posix(): digest(p.read_text(encoding="utf-8"))
             for p in sorted(root.rglob("*")) if p.suffix in {".py", ".json", ".txt"}}
    return {"source_sha256": digest(json.dumps(files, sort_keys=True)), "plugins": plugins(),
            "python": platform.python_version(),
            "distributions": sorted([d.metadata["Name"], d.version] for d in importlib.metadata.distributions())}


def provider_identity(llm):
    result = {k: getattr(llm, k, None) for k in ("mode", "model", "timeout", "max_output_tokens", "max_attempts")}
    # Endpoints can contain credentials; bind their digest, never their raw URL.
    result["endpoint_sha256"] = digest(str(getattr(llm, "endpoint", getattr(llm, "base_url", ""))))
    return result


def benchmark(factory, materials: Path, policy, output: Path, cases=("baseline",), seeds=(42,), rows=600):
    if rows < 200 or rows > 10000 or not cases or not seeds or not set(cases) <= set(CASES):
        raise ValueError("Invalid benchmark cases, seeds or rows (200..10000)")
    first = factory()
    corpus = {p.relative_to(materials).as_posix(): digest(p.read_text(encoding="utf-8")) for p in sorted(materials.rglob("*"))
              if p.is_file() and p.suffix in {".md", ".py"}}
    protocol = {"version": 3, "execution": execution_identity(), "provider": provider_identity(first), "mode": first.mode, "model": first.model, "cases": list(cases), "seeds": list(seeds),
                "prompt_sha256": digest(json.dumps(OllamaLLM.prompts(), sort_keys=True, ensure_ascii=False)),
                "structured_corrections_per_role": 1,
                "rows": rows, "arms": ARMS, "policy": policy, "corpus": corpus,
                "knowledge_source": "same frozen deterministic Markdown/AST corpus, rule-indexed graph",
                "generation_budget_per_candidate": policy["max_repairs"] + 1,
                "budget_note": "Same generation ceiling per candidate per attempt. Provider interruption restarts that cell from a pristine corpus without response reuse; prior attempts and their cost remain reported. Direct/knowledge resample without feedback; knowledge_repair uses feedback. Stop at first pass.",
                "resume_policy": "skip verified completed cells; restart provider-interrupted cells in a new isolated attempt directory",
                "description": DESCRIPTION}
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "manifest.json"
    manifest = read_json(manifest_path) if manifest_path.exists() else {"protocol": protocol, "results": {}}
    if manifest["protocol"] != protocol:
        raise ValueError("Existing campaign protocol differs; use a new output directory")
    write_json(manifest_path, manifest)
    paused = False
    for case in cases:
        for seed in seeds:
            dataset = output / "datasets" / f"{case}-{seed}.csv"
            expected = output / "datasets" / f".expected-{case}-{seed}.csv"
            generate(expected, rows=rows, seed=seed, **CASES[case])
            expected_bytes = expected.read_bytes()
            expected.unlink()
            if dataset.exists() and dataset.read_bytes() != expected_bytes:
                raise ValueError("Campaign dataset changed; use a new output directory")
            if not dataset.exists():
                dataset.write_bytes(expected_bytes)
            for arm in ARMS:
                key = f"{case}-{seed}-{arm}"
                old = manifest["results"].get(key)
                if old and old["failure_kind"] != "provider":
                    saved = (output / old["report"]).resolve()
                    if not saved.is_relative_to(output.resolve()) or not saved.is_file() or digest(saved.read_bytes()) != old.get("report_sha256"):
                        raise ValueError("Completed campaign report is missing or changed")
                    continue
                print(f"对照实验：{key}", flush=True)
                cell = output / "cells" / key
                attempts = cell / "attempts"
                number = 1
                while (attempts / f"{number:04d}").exists():
                    number += 1
                attempt = attempts / f"{number:04d}"
                store = KnowledgeStore(attempt / "knowledge.sqlite")
                store.ingest(materials)
                llm = factory()
                if provider_identity(llm) != protocol["provider"]:
                    raise ValueError("Provider identity changed within campaign")
                folder, report = run(DESCRIPTION, dataset, store, llm, policy, attempt / "runs", seed=seed,
                                     experiment_arm=arm,
                                     resample_budget=policy["max_repairs"] + 1)
                calls = report.get("llm_calls", [])
                result = {"case": case, "seed": seed, "arm": arm, "status": report["status"],
                          "failure_kind": report.get("failure_kind"), "dataset_sha256": report["dataset_sha256"],
                          "attempt_number": number,
                          "report": (folder / "report.json").relative_to(output).as_posix(),
                          "report_sha256": digest((folder / "report.json").read_bytes()),
                          "first_pass_candidates": sum(bool(c["attempts"] and c["attempts"][0]["passed"]) for c in report["candidates"]),
                          "candidate_count": len(report["candidates"]),
                          "repair_calls": sum(e["role"] == "repair" for e in read_json(folder / "events.json")),
                          "successful_inferences": sum("response_id" in c for c in calls),
                          "reported_tokens": sum(c.get("usage", {}).get("total_tokens", 0) for c in calls),
                          "test_metrics": report.get("test_result", {}).get("metrics", {})}
                result["prior_attempts"] = (old.get("prior_attempts", []) + [{k: old[k] for k in
                    ("status", "reported_tokens", "successful_inferences", "report")}]) if old else []
                manifest["results"][key] = result
                write_json(manifest_path, manifest)
                if result["failure_kind"] == "provider":
                    paused = True
                    break
            if paused:
                break
        if paused:
            break
    manifest["status"] = "provider_interrupted" if paused else "complete"
    statistics = {}
    for arm in ARMS:
        records = [r for r in manifest["results"].values() if r["arm"] == arm]
        evaluated = [r for r in records if r["failure_kind"] != "provider"]
        candidates = sum(r["candidate_count"] for r in evaluated)
        statistics[arm] = {
            "evaluated_tasks": len(evaluated), "provider_interrupted_tasks": len(records) - len(evaluated),
            "task_success_rate": sum(r["status"] == "passed" for r in evaluated) / len(evaluated) if evaluated else None,
            "candidate_first_pass_rate": sum(r["first_pass_candidates"] for r in evaluated) / candidates if candidates else None,
            "repair_calls": sum(r["repair_calls"] for r in records),
            "reported_tokens_including_prior_attempts": sum(r["reported_tokens"] + sum(p["reported_tokens"] for p in r.get("prior_attempts", [])) for r in records)}
    manifest["statistics"] = statistics
    write_json(manifest_path, manifest)
    table = ["# 对照实验结果", "", f"模式：{first.mode} / {first.model}；状态：{manifest['status']}", "",
             "Mock 只验证实验流程，不能据此证明真实 LLM 的知识增强效果。各组每候选生成次数上限相同；直接生成/知识增强组失败后独立重采样，不接收代码或验证反馈；修复组接收反馈。首次通过即停止，实际调用和 tokens 仍可能不同。",
             "语料固定为规则/AST 抽取，不把 LLM 抽取的差异混入本次对照。服务错误单独标注，未运行的单元不计为算法失败。", "",
             "| 条件 | 组别 | 状态 | 首次通过候选/总候选 | 修复调用 | 已报告 tokens | 测试 AUC |",
             "|---|---|---|---|---|---|---|"]
    for result in manifest["results"].values():
        table.append(f"| {result['case']} / {result['seed']} | {result['arm']} | {result['status']} | {result['first_pass_candidates']}/{result['candidate_count']} | {result['repair_calls']} | {result['reported_tokens']} | {result['test_metrics'].get('roc_auc', 'N/A')} |")
    table += ["", "相同 case/seed 使用相同数据与划分；不同组数据库隔离，不共享失败经验。",
              "每个单元仅对选中版本评估保留测试集；不根据组间测试结果修改方案。",
              "续跑先核对代码、插件、依赖、数据及已完成报告。服务中断单元从新数据库完整重启，不继承响应缓存和失败经验；原尝试保留并计入成本。生成预算按每次尝试计算，重启不等于原单元内部断点续接。独立重复实验使用新输出目录。", ""]
    table += ["## 按组汇总", "", "```json", json.dumps(statistics, ensure_ascii=False, indent=2), "```", "",
              "成功率分母排除服务中断，包含模型结构化契约失败和算法验证失败；单案例/单种子不能支持统计显著性结论。", ""]
    (output / "summary.md").write_text("\n".join(table), encoding="utf-8")
    return manifest
