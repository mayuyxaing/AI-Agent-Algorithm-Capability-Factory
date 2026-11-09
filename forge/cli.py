"""User-facing CLI; all commands work without a server or cloud account."""
import argparse
import json
from pathlib import Path
from .data import generate
from .knowledge import KnowledgeStore
from .llm import MockLLM, OllamaLLM
from .reporting import graph_html
from .util import read_json
from .validation import load_policy
from .workflow import run


def main(argv=None):
    parser = argparse.ArgumentParser(description="算法能力工厂：理解→检索→规划→生成→验证→修复→沉淀")
    parser.add_argument("--db", type=Path, default=Path("runtime/knowledge.sqlite"))
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("init", help="生成模拟数据并抽取业务材料")
    init.add_argument("--data", type=Path, default=Path("data/churn.csv"))
    init.add_argument("--materials", type=Path, default=Path("materials"))
    ingest = sub.add_parser("ingest", help="从 Markdown 与 Python 抽取知识")
    ingest.add_argument("folder", type=Path)
    ingest.add_argument("--provider", choices=["mock", "ollama"], default="mock")
    ingest.add_argument("--model", default="")
    ingest.add_argument("--base-url", default="http://localhost:11434")
    submit = sub.add_parser("run", help="提交能力描述，生成代码与验证报告")
    group = submit.add_mutually_exclusive_group(required=True)
    group.add_argument("--description")
    group.add_argument("--task-file", type=Path)
    submit.add_argument("--data", type=Path, default=Path("data/churn.csv"))
    submit.add_argument("--config", type=Path, default=Path("configs/churn.json"))
    submit.add_argument("--provider", choices=["mock", "ollama"], default="mock")
    submit.add_argument("--model", default="")
    submit.add_argument("--base-url", default="http://localhost:11434")
    submit.add_argument("--output", type=Path, default=Path("runtime/runs"))
    submit.add_argument("--inject-fault", action="store_true", help="仅 mock：初版逻辑回归未知类别处理故障")
    submit.add_argument("--seed", type=int, default=42)
    graph = sub.add_parser("graph", help="导出图谱 JSON 与离线 HTML")
    graph.add_argument("--output", type=Path, default=Path("runtime/knowledge_graph.json"))
    inspect = sub.add_parser("inspect", help="读取运行报告")
    inspect.add_argument("report", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "inspect":
            print(json.dumps(read_json(args.report), ensure_ascii=False, indent=2))
            return 0
        store = KnowledgeStore(args.db)
        if args.command == "init":
            if not args.materials.is_dir():
                raise ValueError(f"Materials directory not found: {args.materials}")
            if not args.data.exists():
                generate(args.data)
            count = store.ingest(args.materials)
            print(json.dumps({"data": str(args.data), "extracted": count, "db": str(args.db)}, ensure_ascii=False))
        elif args.command == "ingest":
            if not args.folder.is_dir():
                raise ValueError(f"Materials directory not found: {args.folder}")
            llm = MockLLM() if args.provider == "mock" else OllamaLLM(args.model, args.base_url)
            print(json.dumps({"extracted": store.ingest(args.folder, llm)}, ensure_ascii=False))
        elif args.command == "graph":
            store.export(args.output)
            graph_html(args.output, args.output.with_suffix(".html"))
            print(args.output)
        elif args.command == "run":
            description = args.description or args.task_file.read_text(encoding="utf-8").strip()
            if not description.strip():
                raise ValueError("Description cannot be empty")
            llm = MockLLM() if args.provider == "mock" else OllamaLLM(args.model, args.base_url)
            folder, report = run(description, args.data, store, llm, load_policy(args.config),
                                 args.output, args.inject_fault, args.seed)
            print(json.dumps({"run_id": report["run_id"], "status": report["status"],
                              "selected": report.get("selected"), "report": str(folder / "report.md"),
                              "error": report.get("error")}, ensure_ascii=False, indent=2))
            return 0 if report["status"] == "passed" else 1
        return 0
    except (ValueError, OSError, RuntimeError, KeyError) as exc:
        print(json.dumps({"error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False))
        return 2
