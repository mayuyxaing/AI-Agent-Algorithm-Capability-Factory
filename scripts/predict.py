"""Run a trusted generated classifier and its locally created model artifact."""
import argparse
import importlib.util
import sys
from pathlib import Path
import joblib
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="仅加载自己生成并信任的代码与模型")
    parser.add_argument("--artifact", type=Path, required=True, help="运行结果中的 selected 目录")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", default="churn_classification")
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("generated", args.artifact / "algorithm.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    model = joblib.load(args.artifact / "model.joblib")
    frame = pd.read_csv(args.input)
    from forge.tasks import get_task
    task = get_task(args.task)
    column = "churn_probability" if task.id == "churn_classification" else "urgent_probability"
    frame[column] = module.predict(model, frame)
    frame["predicted_" + task.target] = (frame[column] >= .5).astype(int)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    print(f"Predicted {len(frame)} rows: {args.output}")


if __name__ == "__main__":
    main()
