"""SQLite property graph with source-grounded Markdown/Python extraction."""
import ast
import json
import re
import sqlite3
from pathlib import Path
from .util import digest, write_json


class KnowledgeStore:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        with self.connect() as con:
            con.executescript('''
            CREATE TABLE IF NOT EXISTS nodes(
              id TEXT PRIMARY KEY, kind TEXT NOT NULL, title TEXT NOT NULL,
              properties TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS edges(
              source TEXT NOT NULL REFERENCES nodes(id), relation TEXT NOT NULL,
              target TEXT NOT NULL REFERENCES nodes(id),
              PRIMARY KEY(source, relation, target));
            CREATE TABLE IF NOT EXISTS runs(
              id TEXT PRIMARY KEY, report TEXT NOT NULL);
            ''')

    def connect(self):
        con = sqlite3.connect(self.path, timeout=30)
        con.execute("PRAGMA foreign_keys=ON")
        return con

    def node(self, kind: str, title: str, properties: dict, key: str | None = None):
        identifier = key or f"{kind}:{digest(title + json.dumps(properties, sort_keys=True, ensure_ascii=False))[:20]}"
        with self.connect() as con:
            con.execute("INSERT OR REPLACE INTO nodes VALUES(?,?,?,?)",
                        (identifier, kind, title, json.dumps(properties, ensure_ascii=False)))
        return identifier

    def edge(self, source: str, relation: str, target: str):
        with self.connect() as con:
            con.execute("INSERT OR IGNORE INTO edges VALUES(?,?,?)", (source, relation, target))

    def nodes(self, kind: str | None = None):
        with self.connect() as con:
            rows = con.execute("SELECT id,kind,title,properties FROM nodes" +
                               (" WHERE kind=?" if kind else "") + " ORDER BY id",
                               (kind,) if kind else ()).fetchall()
        return [dict(id=i, kind=k, title=t, **json.loads(p)) for i, k, t, p in rows]

    def ingest(self, folder: Path, llm=None):
        count = 0
        for path in sorted(folder.rglob("*")):
            if not path.is_file() or path.suffix not in {".md", ".py"}:
                continue
            source = path.read_text(encoding="utf-8")
            relative = path.relative_to(folder).as_posix()
            sid = self.node("Source", relative, {"path": relative, "sha256": digest(source)})
            facts = []
            if path.suffix == ".py":
                tree = ast.parse(source)
                dependencies = sorted({n.names[0].name if isinstance(n, ast.Import) else n.module or ""
                                       for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))})
                for n in tree.body:
                    if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        facts.append(dict(title=n.name, text=ast.get_docstring(n) or n.name,
                                          line=n.lineno, evidence_quote=ast.get_source_segment(source, n),
                                          dependencies=dependencies, extraction="python_ast"))
            else:
                lines = source.splitlines()
                for index, line in enumerate(lines):
                    if line.startswith("## "):
                        end = next((j for j in range(index + 1, len(lines)) if lines[j].startswith("## ")), len(lines))
                        quote = "\n".join(lines[index:end]).strip()
                        facts.append(dict(title=line[3:], text=quote, line=index + 1,
                                          evidence_quote=quote, extraction="markdown_sections"))
                if llm and llm.mode != "mock":
                    result = llm.call("extract", {"source": source[:16000]})
                    if not isinstance(result.get("facts"), list):
                        raise ValueError("Invalid extraction response")
                    for fact in result["facts"][:12]:
                        quote = fact.get("evidence_quote", "")
                        if not isinstance(quote, str) or not quote or quote not in source:
                            raise ValueError("LLM extraction has no verifiable exact source quote")
                        facts.append(dict(title=str(fact["title"]), text=str(fact["text"]),
                                          evidence_quote=quote, line=source[:source.index(quote)].count("\n") + 1,
                                          extraction="llm_with_quote"))
            for fact in facts:
                kind = "Function" if path.suffix == ".py" else "Knowledge"
                nid = self.node(kind, fact.pop("title"), dict(**fact, source_id=sid, source_path=relative))
                self.edge(nid, "SUPPORTED_BY", sid)
                count += 1
        return count

    @staticmethod
    def tokens(text):
        # Chinese bigrams plus ASCII words; no online embedding dependency.
        words = re.findall(r"[a-zA-Z_]+", text.lower())
        for sequence in re.findall(r"[\u4e00-\u9fff]+", text):
            words.extend(sequence[i:i + 2] for i in range(max(1, len(sequence) - 1)))
        return set(words)

    def retrieve(self, query: str, limit: int = 6):
        tokens = self.tokens(query)
        matches = []
        for node in self.nodes():
            if node["kind"] not in {"Knowledge", "Function"}:
                continue
            score = len(tokens & self.tokens(node["title"] + " " + node.get("text", "")))
            if score:
                matches.append(dict(node, score=score))
        return sorted(matches, key=lambda x: (-x["score"], x["id"]))[:limit]

    def experiences(self):
        return [n for n in self.nodes("FailureCase") if n.get("task") == "churn_classification"][-6:]

    def save_run(self, report: dict):
        run_id = report["run_id"]
        with self.connect() as con:
            con.execute("INSERT OR REPLACE INTO runs VALUES(?,?)", (run_id, json.dumps(report, ensure_ascii=False)))
        capability = self.node("Capability", "客户流失预测", {
            "task": "churn_classification", "input": {"format": "pandas.DataFrame", "required_columns":
                ["tenure", "monthly_charges", "support_calls", "contract", "payment_method"]},
            "output": {"format": "1D probability array", "range": [0, 1], "positive_class": 1},
            "conditions": ["binary labels 0/1", "tabular customer data", "train-only preprocessing"],
            "dependencies": ["numpy", "pandas", "scikit-learn", "joblib"]}, "capability:churn")
        run_node = self.node("ValidationRun", run_id, {
            "status": report["status"], "mode": report["mode"], "model": report["model"],
            "selected": report.get("selected"), "dataset_sha256": report["dataset_sha256"],
            "seed": report["seed"], "test_metrics": report.get("test_result", {}).get("metrics", {}),
            "report_ref": f"runs/{run_id}/report.json"}, "run:" + run_id)
        environment = self.node("Environment", "Python execution environment", report["environment"])
        self.edge(run_node, "EXECUTED_IN", environment)
        for metric in report["policy"]["metrics"]:
            mid = self.node("Metric", metric, {"task": "churn_classification"}, "metric:" + metric)
            self.edge(capability, "EVALUATED_BY", mid)
        for evidence in report["knowledge"]:
            self.edge(run_node, "USED_KNOWLEDGE", evidence["id"])
        for experience in report["experiences"]:
            self.edge(run_node, "REUSED_EXPERIENCE", experience["id"])
        for candidate in report["candidates"]:
            for attempt in candidate["attempts"]:
                version = self.node("AlgorithmVersion", candidate["algorithm"], {
                    "sha256": attempt["code_sha256"], "run_id": run_id,
                    "attempt": attempt["attempt"], "passed": attempt["passed"],
                    "path": attempt["code_path"], "validation_metrics": attempt.get("metrics", {}),
                    "checks": attempt.get("checks", {})}, "version:" + run_id + ":" + candidate["algorithm"] + ":" + str(attempt["attempt"]))
                self.edge(capability, "IMPLEMENTED_BY", version)
                self.edge(version, "VALIDATED_IN", run_node)
                if not attempt["passed"]:
                    failure = self.node("FailureCase", attempt.get("failure_kind", "validation_failure"), {
                        "task": "churn_classification", "error": attempt.get("error", ""),
                        "checks": attempt.get("checks", {}), "injected": attempt["injected"],
                        "resolved_in_run": candidate["passed"],
                        "remedy": "Check failed interface tests; use train-only imputation and OneHotEncoder(handle_unknown='ignore')."})
                    self.edge(version, "EXPOSED", failure)
        return run_node

    def export(self, path: Path):
        with self.connect() as con:
            edges = [dict(source=s, relation=r, target=t) for s, r, t in con.execute("SELECT * FROM edges ORDER BY source,relation,target")]
        write_json(path, {"schema_version": "1.0", "nodes": self.nodes(), "edges": edges})
