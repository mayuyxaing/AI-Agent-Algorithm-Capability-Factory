"""Interactive local key entry. Never saves the key to a file or command line."""
import argparse
import getpass
import os
from pathlib import Path
import sys
from forge.cli import main


def entry(argv=None):
    parser = argparse.ArgumentParser(description="隐藏输入 API Key，检查连接或运行真实模型流程")
    parser.add_argument("--run", action="store_true", help="执行完整流程；不加此参数则只发一次连接检查请求")
    parser.add_argument("--model", default="gpt-4.1-mini")
    args = parser.parse_args(argv)
    os.chdir(Path(__file__).resolve().parents[1])
    previous = os.environ.get("OPENAI_API_KEY")
    if not previous:
        if not sys.stdin.isatty():
            print("请在你自己的交互终端运行此脚本，隐藏输入密钥；不要把密钥发到聊天。")
            return 2
        key = getpass.getpass("粘贴 OpenAI API Key 后按回车（输入不会显示，不保存到文件）：").strip()
        if not key:
            print("未输入密钥，已退出。")
            return 2
        os.environ["OPENAI_API_KEY"] = key
    try:
        if not args.run:
            return main(["doctor", "--provider", "openai", "--model", args.model, "--probe"])
        # Keep real-model evidence separate from offline demonstration experience.
        db = "runtime/openai/knowledge.sqlite"
        code = main(["--db", db, "init"])
        if code:
            return code
        code = main(["--db", db, "ingest", "materials", "--provider", "openai", "--model", args.model])
        if code:
            return code
        return main(["--db", db, "run", "--task-file", "examples/task.txt", "--provider", "openai",
                     "--model", args.model, "--output", "runtime/openai/runs"])
    finally:
        if previous is None:
            os.environ.pop("OPENAI_API_KEY", None)


if __name__ == "__main__":
    raise SystemExit(entry())
