"""Private local BigModel key entry; only glm-4.7-flash is requested."""
import argparse
import getpass
import os
from pathlib import Path
import sys
from forge.cli import main
from forge.knowledge import KnowledgeStore
from forge.util import digest
from forge.data import generate


def reusable_materials(db: Path, materials: Path):
    if not db.exists():
        return None
    store = KnowledgeStore(db)
    audit = store.latest_ingestion()
    if not audit or audit.get("mode") != "glm" or audit.get("model") != "glm-4.7-flash" or audit.get("status") not in {"passed", "partial_llm"}:
        return None
    expected = {p.relative_to(materials).as_posix(): digest(p.read_text(encoding="utf-8"))
                for p in sorted(materials.rglob("*")) if p.is_file() and p.suffix in {".md", ".py"}}
    if not expected:
        return None
    if "source_hashes" in audit:
        return audit if audit["source_hashes"] == expected else None
    # Migrate earlier successful ingestions that predate manifest recording.
    sources = store.nodes("Source")
    if {n["path"] for n in sources} != set(expected):
        return None
    return audit if all(any(n["path"] == name and n["sha256"] == sha for n in sources)
                        for name, sha in expected.items()) else None


def entry(argv=None):
    parser = argparse.ArgumentParser(description="智谱 GLM-4.7-Flash：隐藏输入密钥，检查连接或运行闭环")
    parser.add_argument("--run", action="store_true", help="完整流程；默认只做一次连接检查")
    parser.add_argument("--refresh-knowledge", action="store_true", help="强制重新抽取材料")
    parser.add_argument("--repair-demo", action="store_true", help="显式注入代码缺陷并由真实模型修复")
    parser.add_argument("--benchmark", action="store_true", help="真实模型三组对照实验；比单次流程需要更多调用")
    parser.add_argument("--task", default="churn_classification", choices=["churn_classification", "text_classification"])
    args = parser.parse_args(argv)
    if args.task != "churn_classification" and (args.benchmark or args.repair_demo):
        parser.error("目前对照与故障注入演示针对客户流失；文本任务使用 --run --task text_classification")
    os.chdir(Path(__file__).resolve().parents[1])
    previous = os.environ.get("ZHIPU_API_KEY")
    if not previous:
        if not sys.stdin.isatty():
            print("请双击 check-glm.cmd，在自己的终端隐藏输入智谱密钥，不要发到聊天里。")
            return 2
        key = getpass.getpass("粘贴智谱 BigModel API Key 后回车（不显示、不保存）：").strip()
        if not key:
            print("未输入密钥，已退出。")
            return 2
        os.environ["ZHIPU_API_KEY"] = key
    try:
        print("使用智谱官方 glm-4.7-flash；不会自动切换收费模型。", flush=True)
        if args.benchmark:
            return main(["benchmark", "--provider", "glm", "--llm-attempts", "6",
                         "--output", "runtime/glm/benchmark-v2"])
        if not args.run:
            print("正在检查连接，请等待……", flush=True)
            return main(["doctor", "--provider", "glm", "--probe", "--output", "runtime/glm/connection-check.json"])
        from forge.tasks import get_task
        task = get_task(args.task)
        db = "runtime/glm/knowledge.sqlite" if task.id == "churn_classification" else "runtime/glm/text_classification/knowledge.sqlite"
        prepared = reusable_materials(Path(db), Path("materials")) if not args.refresh_knowledge else None
        if prepared:
            print(f"复用已核对且材料未变的知识，LLM 接受 {prepared['llm_accepted']} 条；跳过重复抽取。", flush=True)
            if not Path(task.data).exists():
                task.generate(Path(task.data))
        else:
            code = main(["--db", db, "init", "--task", task.id])
            if code:
                return code
            print("正在从模拟业务材料抽取知识……", flush=True)
            audit_path = str(Path(db).with_name("ingestion-report.json"))
            code = main(["--db", db, "ingest", "materials", "--report", audit_path,
                         "--llm-attempts", "6", "--provider", "glm"])
            if code:
                return code
        print("正在规划、生成并验证代码；每一步轨迹写入 runtime/glm/runs/。", flush=True)
        command = ["--db", db, "run", "--task", task.id, "--task-file", "examples/task.txt" if task.id == "churn_classification" else "examples/text_task.txt", "--provider", "glm",
                   "--llm-attempts", "6", "--checkpoints", "runtime/glm/checkpoints"]
        if args.repair_demo:
            command.append("--repair-demo")
        return main(command + ["--output", "runtime/glm/runs"])
    finally:
        if previous is None:
            os.environ.pop("ZHIPU_API_KEY", None)


if __name__ == "__main__":
    raise SystemExit(entry())
