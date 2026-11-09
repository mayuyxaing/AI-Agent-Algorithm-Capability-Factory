"""Private local BigModel key entry; only glm-4.7-flash is requested."""
import argparse
import getpass
import os
from pathlib import Path
import sys
from forge.cli import main


def entry(argv=None):
    parser = argparse.ArgumentParser(description="智谱 GLM-4.7-Flash：隐藏输入密钥，检查连接或运行闭环")
    parser.add_argument("--run", action="store_true", help="完整流程；默认只做一次连接检查")
    args = parser.parse_args(argv)
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
        if not args.run:
            print("正在检查连接，请等待……", flush=True)
            return main(["doctor", "--provider", "glm", "--probe", "--output", "runtime/glm/connection-check.json"])
        db = "runtime/glm/knowledge.sqlite"
        code = main(["--db", db, "init"])
        if code:
            return code
        print("正在从模拟业务材料抽取知识……", flush=True)
        code = main(["--db", db, "ingest", "materials", "--provider", "glm"])
        if code:
            return code
        print("正在规划、生成并验证代码；每一步轨迹写入 runtime/glm/runs/。", flush=True)
        return main(["--db", db, "run", "--task-file", "examples/task.txt", "--provider", "glm",
                     "--output", "runtime/glm/runs"])
    finally:
        if previous is None:
            os.environ.pop("ZHIPU_API_KEY", None)


if __name__ == "__main__":
    raise SystemExit(entry())
