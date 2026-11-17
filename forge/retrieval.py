"""Rule-grounded indexing and ranking over the repository's current snapshot."""
import re


class KnowledgeRetriever:
    def __init__(self, store):
        self.store = store

    def index_strategies(self):
        """Rule-grounded relation extraction; support edges keep original evidence."""
        rules = {
            "missing_values": ("缺失值处理", "imputation", "训练集填补", ["缺失", "填补", "imput"]),
            "unknown_categories": ("未知类别", "unknown_safe_encoding", "允许未知类别的编码", ["未知类别", "handle_unknown", "unknown categor"]),
            "class_imbalance": ("类别不平衡", "class_weighting", "类别权重处理不平衡", ["不平衡", "class_weight", "imbalanc"]),
            "train_only_preprocessing": ("训练集内拟合", "train_only_pipeline", "训练集拟合预处理 Pipeline", ["训练集", "pipeline", "泄漏"]),
            "unseen_vocabulary": ("未见词表", "train_fitted_tfidf", "训练集拟合的文本词表", ["未知词", "词表", "tf-idf", "tfidf", "未见"]),
        }
        for fact in self.store.current_facts():
            text = (fact["title"] + " " + fact.get("text", "") + " " + fact.get("evidence_quote", "")).lower()
            for key, (requirement, strategy, title, cues) in rules.items():
                if not any(cue.lower() in text for cue in cues):
                    continue
                rid = self.store.node("Requirement", requirement, {"task": "shared_binary_tasks", "key": key,
                                "extraction": "rule_index"}, "requirement:" + key)
                sid = self.store.node("Strategy", title, {"task": "shared_binary_tasks", "key": strategy,
                                "extraction": "rule_index"}, "strategy:" + strategy)
                self.store.edge(rid, "ADDRESSED_BY", sid)
                self.store.edge(sid, "SUPPORTED_BY", fact["id"])

    def retrieve_grounded(self, query, requirements, limit=6):
        self.store.index_strategies()  # Also upgrades previously populated local databases.
        lexical = {n["id"]: n for n in self.retrieve(query, limit=1000)}
        paths = {}
        with self.store.connect() as con:
            for requirement in sorted(set(requirements)):
                rid = "requirement:" + requirement
                rows = con.execute("SELECT a.target,b.target FROM edges a JOIN edges b ON a.target=b.source "
                                   "WHERE a.source=? AND a.relation='ADDRESSED_BY' AND b.relation='SUPPORTED_BY' "
                                   "ORDER BY a.target,b.target", (rid,)).fetchall()
                for strategy, evidence in rows:
                    paths.setdefault(evidence, []).append([rid, strategy, evidence])
        ranked = []
        for fact in self.store.current_facts():
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
        for node in self.store.current_facts():
            score = len(tokens & self.tokens(node["title"] + " " + node.get("text", "")))
            if score:
                matches.append(dict(node, score=score))
        return sorted(matches, key=lambda x: (-x["score"], x["id"]))[:limit]

    def experiences(self, task="churn_classification"):
        return [n for n in self.store.nodes("FailureCase") if n.get("task") == task and n.get("reusable", False)][-6:]


