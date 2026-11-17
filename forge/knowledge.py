"""SQLite property graph with source-grounded Markdown/Python extraction."""
import ast
import json
import re
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from .util import digest, write_json
from .structured import checked_response


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
            CREATE TABLE IF NOT EXISTS ingestions(
              id TEXT PRIMARY KEY, report TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS source_versions(
              source_id TEXT PRIMARY KEY, corpus TEXT NOT NULL, path TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS active_facts(
              corpus TEXT NOT NULL, fact_id TEXT NOT NULL,
              PRIMARY KEY(corpus, fact_id));
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

    def latest_ingestion(self):
        with self.connect() as con:
            row = con.execute("SELECT report FROM ingestions ORDER BY rowid DESC LIMIT 1").fetchone()
        return json.loads(row[0]) if row else None

    def current_facts(self):
        """Default retrieval sees committed corpus snapshots; nodes() retains history."""
        with self.connect() as con:
            managed = {r[0] for r in con.execute("SELECT source_id FROM source_versions")}
            active = {r[0] for r in con.execute("SELECT fact_id FROM active_facts")}
        return [n for n in self.nodes() if n["kind"] in {"Knowledge", "Function"}
                and (n.get("source_id") not in managed or n["id"] in active)]

    def ingest(self, folder: Path, llm=None, strict_llm=False):
        count = 0
        corpus = folder.resolve().as_posix()
        snapshot = []
        documents = [(p, p.read_text(encoding="utf-8")) for p in sorted(folder.rglob("*"))
                     if p.is_file() and p.suffix in {".md", ".py"}]
        audit = {"id": uuid.uuid4().hex, "created_at": datetime.now(timezone.utc).isoformat(),
                 "mode": getattr(llm, "mode", "deterministic"), "model": getattr(llm, "model", None),
                 "llm_accepted": 0, "llm_rejected": 0, "rejections": [],
                 "responses": [], "status": "running", "retrieval_snapshot_version": 1}
        audit["source_hashes"] = {p.relative_to(folder).as_posix(): digest(source) for p, source in documents}

        def save_audit(status):
            audit.update(status=status, extracted=count)
            with self.connect() as con:
                con.execute("INSERT OR REPLACE INTO ingestions VALUES(?,?)", (audit["id"], json.dumps(audit, ensure_ascii=False)))

        try:
            for path, source in documents:
                relative = path.relative_to(folder).as_posix()
                sid = self.node("Source", relative, {"path": relative, "sha256": digest(source), "corpus": corpus})
                with self.connect() as con:
                    con.execute("INSERT OR IGNORE INTO source_versions VALUES(?,?,?)", (sid, corpus, relative))
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
                        # Trusted source spans avoid asking a model to reproduce punctuation exactly.
                        chunks = {f"section_{i}": fact["evidence_quote"] for i, fact in enumerate(facts)
                                  if fact["evidence_quote"] in source[:16000]}
                        if not chunks:
                            chunks = {"document_0": source[:16000]}
                        context = {"source": source[:16000], "evidence_chunks": [
                            {"id": key, "quote": quote} for key, quote in chunks.items()],
                            "output_schema": {"type": "object", "required": ["facts"], "properties": {
                                "facts": {"type": "array", "maxItems": 12, "items": {
                                    "type": "object", "required": ["title", "text", "source_chunk_id"],
                                    "properties": {key: {"type": "string"} for key in
                                                   ("title", "text", "source_chunk_id")}}}}}}

                        def ask(role, request):
                            from .llm import structured_call
                            response = structured_call(llm, role, request)
                            record(role + "_response", {}, {"response": response})
                            return response

                        def record(event, request, response):
                            audit["responses"].append(dict(source_path=relative, event=event,
                                                           **request, **response))
                            save_audit("running")

                        def validate_envelope(response):
                            if not isinstance(response, dict) or not isinstance(response.get("facts"), list):
                                raise ValueError('Expected a JSON object with a "facts" array; each fact needs title, text and source_chunk_id')
                            if len(response["facts"]) > 12:
                                raise ValueError("facts must contain at most 12 entries")

                        try:
                            result = checked_response("extract", context, ask, validate_envelope, record)
                        except Exception as exc:
                            audit["error"] = str(exc)
                            save_audit("invalid_response" if isinstance(exc, ValueError) else "provider_error")
                            raise
                        for index, fact in enumerate(result["facts"][:12]):
                            reason = None
                            if not isinstance(fact, dict) or not all(isinstance(fact.get(k), str) and fact[k].strip() for k in ("title", "text")):
                                reason, quote = "invalid_fact_fields", ""
                            elif "source_chunk_id" in fact:
                                chunk_id = fact["source_chunk_id"]
                                quote = chunks.get(chunk_id, "") if isinstance(chunk_id, str) else ""
                                if not quote:
                                    reason = "unknown_source_chunk_id"
                                elif "evidence_quote" in fact and (not isinstance(fact["evidence_quote"], str) or
                                        not fact["evidence_quote"] or fact["evidence_quote"] not in quote):
                                    reason = "quote_conflicts_with_source_chunk"
                            else:
                                # Backward compatibility: exact legacy quotes remain verifiable.
                                quote = fact.get("evidence_quote", "")
                                if not isinstance(quote, str) or not quote or quote not in source:
                                    reason = "no_verifiable_exact_source_quote"
                            if reason:
                                audit["llm_rejected"] += 1
                                audit["rejections"].append({"source_path": relative, "fact_index": index,
                                                           "reason": reason, "fact": fact})
                                if strict_llm:
                                    save_audit("rejected_strict")
                                    raise ValueError("LLM extraction has no verifiable exact source quote or valid source reference")
                                continue
                            audit["llm_accepted"] += 1
                            facts.append(dict(title=fact["title"], text=fact["text"],
                                              evidence_quote=quote, line=source[:source.index(quote)].count("\n") + 1,
                                              extraction="llm_with_source_chunk" if "source_chunk_id" in fact else "llm_with_quote"))
                for fact in facts:
                    kind = "Function" if path.suffix == ".py" else "Knowledge"
                    nid = self.node(kind, fact.pop("title"), dict(**fact, source_id=sid, source_path=relative))
                    self.edge(nid, "SUPPORTED_BY", sid)
                    snapshot.append(nid)
                    count += 1
        except Exception as exc:
            if audit['status'] == 'running':
                audit['error'] = f'{type(exc).__name__}: {exc}'
                save_audit('extraction_error')
            raise
        # Publish only after the whole folder succeeds. Old nodes and edges remain
        # auditable, but changed/deleted documents cannot compete in retrieval.
        legacy = [n for n in self.nodes("Source") if "corpus" not in n]
        with self.connect() as con:
            # Legacy exports had no corpus identity. Bind unowned legacy sources
            # to the first successfully reingested corpus during migration.
            con.executemany("INSERT OR IGNORE INTO source_versions VALUES(?,?,?)",
                            [(n["id"], corpus, n.get("path", n["title"])) for n in legacy])
            con.execute("DELETE FROM active_facts WHERE corpus=?", (corpus,))
            con.executemany("INSERT OR IGNORE INTO active_facts VALUES(?,?)", [(corpus, n) for n in snapshot])
        save_audit("partial_llm" if audit["llm_rejected"] else "passed")
        self.index_strategies()
        return count

    def index_strategies(self):
        """Rule-grounded relation extraction; support edges keep original evidence."""
        rules = {
            "missing_values": ("缺失值处理", "imputation", "训练集填补", ["缺失", "填补", "imput"]),
            "unknown_categories": ("未知类别", "unknown_safe_encoding", "允许未知类别的编码", ["未知类别", "handle_unknown", "unknown categor"]),
            "class_imbalance": ("类别不平衡", "class_weighting", "类别权重处理不平衡", ["不平衡", "class_weight", "imbalanc"]),
            "train_only_preprocessing": ("训练集内拟合", "train_only_pipeline", "训练集拟合预处理 Pipeline", ["训练集", "pipeline", "泄漏"]),
            "unseen_vocabulary": ("未见词表", "train_fitted_tfidf", "训练集拟合的文本词表", ["未知词", "词表", "tf-idf", "tfidf", "未见"]),
        }
        for fact in self.current_facts():
            text = (fact["title"] + " " + fact.get("text", "") + " " + fact.get("evidence_quote", "")).lower()
            for key, (requirement, strategy, title, cues) in rules.items():
                if not any(cue.lower() in text for cue in cues):
                    continue
                rid = self.node("Requirement", requirement, {"task": "shared_binary_tasks", "key": key,
                                "extraction": "rule_index"}, "requirement:" + key)
                sid = self.node("Strategy", title, {"task": "shared_binary_tasks", "key": strategy,
                                "extraction": "rule_index"}, "strategy:" + strategy)
                self.edge(rid, "ADDRESSED_BY", sid)
                self.edge(sid, "SUPPORTED_BY", fact["id"])

    def retrieve_grounded(self, query, requirements, limit=6):
        self.index_strategies()  # Also upgrades previously populated local databases.
        lexical = {n["id"]: n for n in self.retrieve(query, limit=1000)}
        paths = {}
        with self.connect() as con:
            for requirement in sorted(set(requirements)):
                rid = "requirement:" + requirement
                rows = con.execute("SELECT a.target,b.target FROM edges a JOIN edges b ON a.target=b.source "
                                   "WHERE a.source=? AND a.relation='ADDRESSED_BY' AND b.relation='SUPPORTED_BY' "
                                   "ORDER BY a.target,b.target", (rid,)).fetchall()
                for strategy, evidence in rows:
                    paths.setdefault(evidence, []).append([rid, strategy, evidence])
        ranked = []
        for fact in self.current_facts():
            graph_paths = paths.get(fact["id"], [])
            score = lexical.get(fact["id"], {}).get("score", 0) + 20 * len(graph_paths)
            if score:
                ranked.append(dict(fact, score=score, retrieval_paths=graph_paths))
        return sorted(ranked, key=lambda n: (-n["score"], not n.get("extraction", "").startswith("llm"), n["id"]))[:limit]

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
        for node in self.current_facts():
            score = len(tokens & self.tokens(node["title"] + " " + node.get("text", "")))
            if score:
                matches.append(dict(node, score=score))
        return sorted(matches, key=lambda x: (-x["score"], x["id"]))[:limit]

    def experiences(self, task="churn_classification"):
        return [n for n in self.nodes("FailureCase") if n.get("task") == task and n.get("reusable", False)][-6:]

    def save_run(self, report: dict):
        run_id = report["run_id"]
        with self.connect() as con:
            con.execute("INSERT OR REPLACE INTO runs VALUES(?,?)", (run_id, json.dumps(report, ensure_ascii=False)))
        task = report.get("task_id", "churn_classification")
        capability = self.node("Capability", report.get("task_title", "客户流失预测"), {
            "task": task, "input": {"format": "pandas.DataFrame", "required_columns":
                report.get("features", ["tenure", "monthly_charges", "support_calls", "contract", "payment_method"])},
            "output": {"format": "1D probability array", "range": [0, 1], "positive_class": 1},
            "conditions": ["binary labels 0/1", "registered task feature contract", "train-only preprocessing"],
            "dependencies": ["numpy", "pandas", "scikit-learn", "joblib"]}, "capability:" + task)
        run_node = self.node("ValidationRun", run_id, {
            "status": report["status"], "mode": report["mode"], "model": report["model"],
            "selected": report.get("selected"), "dataset_sha256": report["dataset_sha256"],
            "seed": report["seed"], "test_metrics": report.get("test_result", {}).get("metrics", {}),
            "report_ref": f"runs/{run_id}/report.json"}, "run:" + run_id)
        environment = self.node("Environment", "Python execution environment", report["environment"])
        self.edge(run_node, "EXECUTED_IN", environment)
        for metric in report["policy"]["metrics"]:
            mid = self.node("Metric", metric, {"task": task}, "metric:" + task + ":" + metric)
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
                    "hyperparameters": candidate.get("hyperparameters"), "preprocessing": candidate.get("preprocessing"),
                    "checks": attempt.get("checks", {})}, "version:" + run_id + ":" + candidate["algorithm"] + ":" + str(attempt["attempt"]))
                self.edge(capability, "IMPLEMENTED_BY", version)
                self.edge(version, "VALIDATED_IN", run_node)
                if not attempt["passed"]:
                    from .failures import analyze_failure
                    analysis = analyze_failure(attempt)
                    failure = self.node("FailureCase", attempt.get("failure_kind", "validation_failure"), {
                        "task": task, "error": attempt.get("error", ""),
                        "checks": attempt.get("checks", {}), "injected": attempt["injected"],
                        "resolved_in_run": candidate["passed"],
                        **analysis})
                    self.edge(version, "EXPOSED", failure)
        return run_node

    def export(self, path: Path):
        with self.connect() as con:
            edges = [dict(source=s, relation=r, target=t) for s, r, t in con.execute("SELECT * FROM edges ORDER BY source,relation,target")]
            versions = [list(r) for r in con.execute("SELECT * FROM source_versions ORDER BY source_id")]
            active = [list(r) for r in con.execute("SELECT * FROM active_facts ORDER BY corpus,fact_id")]
        write_json(path, {"schema_version": "1.1", "nodes": self.nodes(), "edges": edges,
                          "retrieval_state": {"source_versions": versions, "active_facts": active}})

    def import_graph(self, graph):
        """Restore an export into an empty store, including its active snapshot."""
        if self.nodes():
            raise ValueError("Import requires an empty knowledge store")
        for item in graph["nodes"]:
            properties = {k: v for k, v in item.items() if k not in {"id", "kind", "title"}}
            self.node(item["kind"], item["title"], properties, item["id"])
        for edge in graph["edges"]:
            self.edge(edge["source"], edge["relation"], edge["target"])
        state = graph.get("retrieval_state", {})
        with self.connect() as con:
            con.executemany("INSERT INTO source_versions VALUES(?,?,?)", state.get("source_versions", []))
            con.executemany("INSERT INTO active_facts VALUES(?,?)", state.get("active_facts", []))
