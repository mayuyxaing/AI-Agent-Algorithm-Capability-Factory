"""Reproduce an explicitly assisted repair of recorded GLM text code, without API calls.

Only the cleaner's DataFrame-to-Series boundary is patched. Original algorithms,
hyperparameters and failed reports remain intact. This is not a new GLM repair.
"""
import argparse
import ast
import difflib
import shutil
from pathlib import Path

from forge.knowledge import KnowledgeStore
from forge.util import digest, read_json, write_json
from forge.validation import load_policy
from forge.workflow import run


def patch_cleaner(source):
    """Insert a guarded text column selection in one module-level cleaner."""
    tree = ast.parse(source)
    transforms = [method for node in tree.body if isinstance(node, ast.ClassDef)
                  for method in node.body if isinstance(method, ast.FunctionDef) and method.name == "transform"]
    if len(transforms) != 1 or "import pandas as pd" not in source:
        raise ValueError("This targeted example requires one cleaner and pandas imported as pd")
    method = transforms[0]
    if len(method.args.args) < 2 or method.args.args[1].arg != "X":
        raise ValueError("Expected transform(self, X)")
    first = method.body[0]
    if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
        position = first.end_lineno
    else:
        position = first.lineno - 1
    indent = " " * (method.col_offset + 4)
    addition = (f'{indent}if isinstance(X, pd.DataFrame):\n'
                f'{indent}    if "text" not in X.columns:\n'
                f'{indent}        raise ValueError("Missing text column")\n'
                f'{indent}    X = X["text"]\n')
    lines = source.splitlines(keepends=True)
    lines.insert(position, addition)
    patched = "".join(lines)
    ast.parse(patched)
    return patched


class AssistedResponses:
    mode = "assisted_glm"
    response_source = "recorded_glm_with_explicit_assistant_patch"

    def __init__(self, original, codes, provenance):
        self.model = original["model"]
        self.original, self.codes, self.provenance = original, codes, provenance
        self.calls = []

    def call(self, role, context):
        if role == "understand":
            return self.original["task_spec"]
        if role == "plan":
            return self.original["plan"]
        if role == "generate":
            return {"code": self.codes[context["algorithm"]],
                    "explanation": "Recorded GLM source with an explicit coding-assistant DataFrame boundary patch; no new API call or autonomous GLM repair."}
        raise ValueError("Assisted reproduction has no further model repair responses")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--knowledge-db", type=Path, help="Optional source database; otherwise restore the archived property graph")
    parser.add_argument("--data", type=Path, default=Path("data/tickets.csv"))
    parser.add_argument("--output", type=Path, default=Path("examples/v3/text-assisted-repair"))
    args = parser.parse_args()
    original = read_json(args.source_run / "report.json")
    if original["task_id"] != "text_classification" or original["mode"] != "glm":
        raise ValueError("Expected a recorded real GLM text-classification run")
    if digest(args.data.read_bytes()) != original["dataset_sha256"]:
        raise ValueError("Source run and input dataset hashes differ")
    if args.knowledge_db and not args.knowledge_db.is_file():
        raise ValueError("Source knowledge database does not exist")
    if not args.knowledge_db and not (args.source_run / "knowledge_graph.json").is_file():
        raise ValueError("Source knowledge graph is required when no database is supplied")
    if args.output.exists():
        raise ValueError("Use a fresh output directory to preserve previous evidence")
    args.output.mkdir(parents=True)
    codes, changes = {}, []
    for candidate in original["candidates"]:
        attempt = candidate["attempts"][-1]
        source = (args.source_run / attempt["code_path"]).read_text(encoding="utf-8")
        patched = patch_cleaner(source)
        algorithm = candidate["algorithm"]
        codes[algorithm] = patched
        folder = args.output / "sources" / algorithm
        folder.mkdir(parents=True)
        (folder / "original.py").write_text(source, encoding="utf-8")
        (folder / "interface.patch").write_text("".join(difflib.unified_diff(
            source.splitlines(True), patched.splitlines(True), fromfile="original.py", tofile="algorithm.py")), encoding="utf-8")
        changes.append({"algorithm": algorithm, "source_code_path": attempt["code_path"],
                        "source_sha256": digest(source), "patched_sha256": digest(patched)})
    provenance = {"kind": "assisted_reproduction", "source_run_id": original["run_id"],
                  "original_status": original["status"], "new_api_calls": 0,
                  "modification_author": "coding assistant, outside the GLM workflow",
                  "change": "Select the text column when the cleaner receives a DataFrame; no estimator/parameter/metric changes",
                  "changes": changes}
    write_json(args.output / "provenance.json", provenance)
    shutil.copy2(args.source_run / "report.json", args.output / "original-report.json")
    store = KnowledgeStore(args.output / "knowledge.sqlite")
    if args.knowledge_db:
        with KnowledgeStore(args.knowledge_db).connect() as source_db, store.connect() as destination:
            source_db.backup(destination)
    else:
        graph = read_json(args.source_run / "knowledge_graph.json")
        if not store.nodes():
            store.import_graph(graph)
        audit = original.get("knowledge_ingestion")
        if audit:
            import json
            with store.connect() as con:
                con.execute("INSERT INTO ingestions VALUES(?,?)", (audit["id"], json.dumps(audit, ensure_ascii=False)))
    policy = dict(load_policy(Path("configs/text.json")), max_repairs=0)
    folder, report = run(original["description"], args.data, store,
                         AssistedResponses(original, codes, provenance), policy,
                         args.output / "runs", seed=original["seed"])
    print({"status": report["status"], "mode": report["mode"], "report": str(folder / "report.md")})
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
