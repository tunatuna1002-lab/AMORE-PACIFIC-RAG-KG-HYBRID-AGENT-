"""
Shared fakes/fixtures for characterization tests.

Rules followed here:
- Collaborators are injected through constructors (fakes/stubs), never by
  patching module paths.
- No network, no ChromaDB on disk: the document retriever is a hand-written
  fake exposing exactly the methods HybridRetriever calls.
- KnowledgeGraph always gets an explicit tmp persist_path + auto_load=False.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from src.domain.entities.relations import Relation, RelationType
from src.ontology.business_rules import register_all_rules
from src.ontology.knowledge_graph import KnowledgeGraph
from src.ontology.reasoner import OntologyReasoner

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Two canned RAG chunks (with ids) returned by the fake dense search.
CANNED_CHUNKS: list[dict[str, Any]] = [
    {
        "id": "c1",
        "content": "SoS(Share of Shelf)는 점유율 지표입니다.",
        "score": 0.9,
        "metadata": {"title": "SoS 정의", "doc_type": "metric_guide"},
    },
    {
        "id": "c2",
        "content": "HHI는 시장 집중도입니다.",
        "score": 0.7,
        "metadata": {"title": "HHI", "doc_type": "playbook"},
    },
]

# Minimal dashboard metrics in the shape HybridRetriever._build_inference_context reads.
CURRENT_METRICS: dict[str, Any] = {
    "brand_metrics": [
        {
            "brand_name": "LANEIGE",
            "is_laneige": True,
            "share_of_shelf": 0.12,
            "avg_rank": 8.5,
            "product_count": 3,
            "category_id": "lip_care",
        }
    ],
    "market_metrics": [{"category_id": "lip_care", "hhi": 0.15, "cpi": 110.0}],
    "alerts": [],
}


class FakeDocRetriever:
    """Stand-in for src.rag.retriever.DocumentRetriever.

    HybridRetriever calls: ``initialize()``, ``search(query, top_k=,
    doc_type_filter=)`` and — only if the attributes exist — ``search_bm25``
    and ``reciprocal_rank_fusion``. Those two are attached per-instance so a
    single class can model both the dense-only and the hybrid-RRF retriever.
    """

    def __init__(
        self,
        chunks: list[dict[str, Any]] | None = None,
        bm25_results: list[dict[str, Any]] | None = None,
        with_rrf: bool = False,
    ) -> None:
        self.chunks = list(chunks if chunks is not None else CANNED_CHUNKS)
        self.calls: list[tuple[str, int, list[str] | None]] = []
        self.initialized = False
        if bm25_results is not None:
            self.search_bm25 = lambda query, top_k=10: [dict(r) for r in bm25_results]
        if with_rrf:
            # Naive concatenation "fusion" - keeps every item as-is (ids included/omitted).
            self.reciprocal_rank_fusion = lambda *lists, k=60, top_k=10: [
                d for lst in lists for d in lst
            ][:top_k]

    async def initialize(self) -> bool:
        self.initialized = True
        return True

    async def search(
        self,
        query: str,
        top_k: int = 5,
        doc_filter: str | None = None,
        doc_type_filter: list[str] | None = None,
        **_: Any,
    ) -> list[dict[str, Any]]:
        self.calls.append((query, top_k, doc_type_filter))
        return [dict(c) for c in self.chunks]


def seed_kg(kg: KnowledgeGraph) -> KnowledgeGraph:
    """Seed the four triples every retriever/chatbot test relies on."""
    kg.add_relation(Relation("LANEIGE", RelationType.OWNED_BY_GROUP, "AMOREPACIFIC"))
    kg.add_relation(
        Relation(
            "LANEIGE",
            RelationType.HAS_PRODUCT,
            "B0LANE1",
            properties={
                "title": "LANEIGE Lip Sleeping Mask",
                "product_name": "LANEIGE Lip Sleeping Mask",
                "rank": 1,
                "category": "lip_care",
            },
        )
    )
    kg.add_relation(Relation("B0LANE1", RelationType.BELONGS_TO_CATEGORY, "lip_care"))
    kg.add_relation(Relation("COSRX", RelationType.COMPETES_WITH, "LANEIGE"))
    return kg


@pytest.fixture
def kg(tmp_path: Path) -> KnowledgeGraph:
    graph = KnowledgeGraph(
        persist_path=str(tmp_path / "kg.json"), auto_load=False, auto_save=False
    )
    return seed_kg(graph)


@pytest.fixture
def reasoner(kg: KnowledgeGraph) -> OntologyReasoner:
    r = OntologyReasoner(kg)
    register_all_rules(r)
    return r


@pytest.fixture
def fake_doc_retriever() -> FakeDocRetriever:
    return FakeDocRetriever()
