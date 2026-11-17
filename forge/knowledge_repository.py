"""SQLite persistence and transaction boundaries, independent of extraction and ranking."""
import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from .util import digest, write_json


class GraphRepository:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._transaction_connection = None
        self._rollback_only = False
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

    @contextmanager
    def connect(self):
        """Reuse an active transaction without committing its nested operations."""
        if self._transaction_connection is not None:
            yield self._transaction_connection
            return
        con = sqlite3.connect(self.path, timeout=30)
        con.execute("PRAGMA foreign_keys=ON")
        try:
            with con:
                yield con
        finally:
            con.close()

    @contextmanager
    def transaction(self):
        """Commit a complete graph mutation, or roll it back on any exception.

        Repository instances are scoped to one workflow, not shared across threads.
        Nested calls join the outer transaction; nested savepoints are not exposed.
        """
        if self._transaction_connection is not None:
            try:
                yield self
            except Exception:
                self._rollback_only = True
                raise
            return
        with self.connect() as con:
            con.execute("BEGIN IMMEDIATE")
            self._transaction_connection = con
            self._rollback_only = False
            try:
                yield self
                if self._rollback_only:
                    raise RuntimeError("Transaction aborted by an earlier nested operation")
            finally:
                self._transaction_connection = None
                self._rollback_only = False

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

    def export(self, path: Path):
        with self.connect() as con:
            edges = [dict(source=s, relation=r, target=t) for s, r, t in con.execute("SELECT * FROM edges ORDER BY source,relation,target")]
            versions = [list(r) for r in con.execute("SELECT * FROM source_versions ORDER BY source_id")]
            active = [list(r) for r in con.execute("SELECT * FROM active_facts ORDER BY corpus,fact_id")]
        write_json(path, {"schema_version": "1.1", "nodes": self.nodes(), "edges": edges,
                          "retrieval_state": {"source_versions": versions, "active_facts": active}})

    def import_graph(self, graph):
        """Restore an export into an empty store, including its active snapshot."""
        with self.transaction():
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
