"""Source extraction and audited publication of complete material snapshots."""
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from .util import digest
from .structured import checked_response
from .source_documents import parse_source


class MaterialIngestor:
    def __init__(self, store):
        self.store = store

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
            with self.store.connect() as con:
                con.execute("INSERT OR REPLACE INTO ingestions VALUES(?,?)", (audit["id"], json.dumps(audit, ensure_ascii=False)))

        try:
            for path, source in documents:
                relative = path.relative_to(folder).as_posix()
                sid = self.store.node("Source", relative, {"path": relative, "sha256": digest(source), "corpus": corpus})
                with self.store.connect() as con:
                    con.execute("INSERT OR IGNORE INTO source_versions VALUES(?,?,?)", (sid, corpus, relative))
                facts = parse_source(path.suffix, source)
                if path.suffix == ".md" and llm and llm.mode != "mock":
                    facts += extract_verified_facts(relative, source, facts, llm, strict_llm, audit, save_audit)
                for fact in facts:
                    kind = "Function" if path.suffix == ".py" else "Knowledge"
                    nid = self.store.node(kind, fact.pop("title"), dict(**fact, source_id=sid, source_path=relative))
                    self.store.edge(nid, "SUPPORTED_BY", sid)
                    snapshot.append(nid)
                    count += 1
        except Exception as exc:
            if audit['status'] == 'running':
                audit['error'] = f'{type(exc).__name__}: {exc}'
                save_audit('extraction_error')
            raise
        # Publish only after the whole folder succeeds. Old nodes and edges remain
        # auditable, but changed/deleted documents cannot compete in retrieval.
        legacy = [n for n in self.store.nodes("Source") if "corpus" not in n]
        try:
            with self.store.transaction():
                with self.store.connect() as con:
                    # Legacy exports had no corpus identity. Bind unowned sources
                    # to the first successfully reingested corpus during migration.
                    con.executemany("INSERT OR IGNORE INTO source_versions VALUES(?,?,?)",
                                    [(n["id"], corpus, n.get("path", n["title"])) for n in legacy])
                    con.execute("DELETE FROM active_facts WHERE corpus=?", (corpus,))
                    con.executemany("INSERT OR IGNORE INTO active_facts VALUES(?,?)", [(corpus, n) for n in snapshot])
                save_audit("partial_llm" if audit["llm_rejected"] else "passed")
                self.store.index_strategies()
        except Exception as exc:
            audit["error"] = f"{type(exc).__name__}: {exc}"
            save_audit("publication_error")
            raise
        return count


def extract_verified_facts(relative, source, facts, llm, strict_llm, audit, save_audit):
    accepted = []
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
        reason, quote = verify_source_reference(fact, chunks, source)
        if reason:
            audit["llm_rejected"] += 1
            audit["rejections"].append({"source_path": relative, "fact_index": index,
                                       "reason": reason, "fact": fact})
            if strict_llm:
                save_audit("rejected_strict")
                raise ValueError("LLM extraction has no verifiable exact source quote or valid source reference")
            continue
        audit["llm_accepted"] += 1
        accepted.append(dict(title=fact["title"], text=fact["text"],
                          evidence_quote=quote, line=source[:source.index(quote)].count("\n") + 1,
                          extraction="llm_with_source_chunk" if "source_chunk_id" in fact else "llm_with_quote"))
    return accepted


def verify_source_reference(fact, chunks, source):
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
    return reason, quote
