"""Independent revalidation of a generated source file using a fresh split."""
import argparse
from pathlib import Path
import shutil
import tempfile
from forge.data import load_split
from forge.validation import evaluate, load_policy
from forge.util import write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--code", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("data/churn.csv"))
    parser.add_argument("--config", type=Path, default=Path("configs/churn.json"))
    parser.add_argument("--output", type=Path, default=Path("runtime/revalidation.json"))
    args = parser.parse_args()
    train, valid, _ = load_split(args.data)
    with tempfile.TemporaryDirectory() as scratch:
        code = Path(scratch) / "algorithm.py"
        shutil.copy2(args.code, code)
        result = evaluate(code, train, valid, load_policy(args.config), 42)
    write_json(args.output, result)
    print(args.output)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
