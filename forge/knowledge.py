"""Stable public knowledge API; extraction, retrieval and persistence have separate owners."""
from pathlib import Path
from .knowledge_repository import GraphRepository
from .extraction import MaterialIngestor
from .retrieval import KnowledgeRetriever
from .run_history import RunHistory


class KnowledgeStore(GraphRepository):
    def ingest(self, folder: Path, llm=None, strict_llm=False):
        return MaterialIngestor(self).ingest(folder, llm, strict_llm)

    def index_strategies(self):
        return KnowledgeRetriever(self).index_strategies()

    def retrieve(self, query: str, limit: int = 6):
        return KnowledgeRetriever(self).retrieve(query, limit)

    def retrieve_grounded(self, query, requirements, limit=6):
        return KnowledgeRetriever(self).retrieve_grounded(query, requirements, limit)

    def experiences(self, task="churn_classification"):
        return KnowledgeRetriever(self).experiences(task)

    def save_run(self, report: dict):
        return RunHistory(self).save_run(report)

    tokens = staticmethod(KnowledgeRetriever.tokens)
