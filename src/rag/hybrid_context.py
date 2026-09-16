"""HybridContext + EntityExtractor
================================

The retriever's data surface: the result object ``HybridRetriever.retrieve``
returns, and the thin entity-extraction wrapper it feeds from.

Both names are re-exported from :mod:`src.rag.hybrid_retriever`, which is where
the rest of the codebase (and ``src.rag.__init__``) imports them from.

Moved verbatim out of ``hybrid_retriever.py`` (F3 split).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.domain.entities.relations import InferenceResult


@dataclass
class HybridContext:
    """
    하이브리드 검색 결과

    Attributes:
        query: 원본 쿼리
        entities: 추출된 엔티티
        ontology_facts: 지식 그래프에서 조회한 사실
        inferences: 온톨로지 추론 결과
        rag_chunks: RAG 검색 결과 청크
        combined_context: 통합된 컨텍스트 (LLM 프롬프트용)
        metadata: 추가 메타데이터
    """

    query: str
    entities: dict[str, list[str]] = field(default_factory=dict)
    ontology_facts: list[dict[str, Any]] = field(default_factory=list)
    inferences: list[InferenceResult] = field(default_factory=list)
    rag_chunks: list[dict[str, Any]] = field(default_factory=list)
    combined_context: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """딕셔너리 변환"""
        return {
            "query": self.query,
            "entities": self.entities,
            "ontology_facts": self.ontology_facts,
            "inferences": [inf.to_dict() for inf in self.inferences],
            "rag_chunks": self.rag_chunks,
            "combined_context": self.combined_context,
            "metadata": self.metadata,
        }


class EntityExtractor:
    """
    쿼리에서 엔티티 추출 (thin wrapper around EntityLinker).

    All extraction is delegated to EntityLinker.extract_entities().
    """

    def __init__(self) -> None:
        from src.rag.entity_linker import EntityLinker

        self._linker = EntityLinker(use_spacy=False)

    @classmethod
    def get_known_brands(cls) -> list:
        """EntityLinker에서 브랜드 목록 가져오기 (이름 + 별칭 평탄화)"""
        from src.rag.entity_linker import EntityLinker

        return list(EntityLinker(use_spacy=False)._get_merged_brands().keys())

    @classmethod
    def get_brand_normalization_map(cls) -> dict:
        """EntityLinker에서 별칭 → 정규화된 브랜드명 매핑"""
        from src.rag.entity_linker import EntityLinker

        return EntityLinker(use_spacy=False)._get_merged_brands()

    def extract(self, query: str, knowledge_graph=None) -> dict[str, list[str]]:
        """
        쿼리에서 엔티티 추출. Delegates to EntityLinker.extract_entities().

        Args:
            query: 사용자 쿼리
            knowledge_graph: 지식 그래프 (제품 검색용, optional)

        Returns:
            {
                "brands": [...],
                "categories": [...],
                "indicators": [...],
                "time_range": [...],
                "products": [...],
                "sentiments": [...],
                "sentiment_clusters": [...],
                "concepts": [...]
            }
        """
        entities = self._linker.extract_entities(query, knowledge_graph=knowledge_graph)
        try:
            entities["concepts"] = self._linker.extract_concepts(query)
        except Exception:
            entities["concepts"] = []
        return entities
