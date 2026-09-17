"""검색 오류 가시화(F3) 회귀 테스트.

배경 (2026-09-17): HybridRetriever.retrieve()의 바깥 except Exception이 모든 실패를
삼키고 metadata["error"]만 남긴 채 빈 컨텍스트로 반환했다. 평가 하니스는 이를 검색
실패가 아니라 그냥 "근거 없는 답변"으로 채점했다 — Chroma가 "Error finding id"로
검색에 실패한 68·70번 문항이 평가에서 조용히 0점 대신 부정확한 점수를 받았다.

이 테스트는 실제 HybridRetriever(+ 실제 KnowledgeGraph, 실제 OntologyReasoner)에
예외를 던지는 가짜 doc_retriever / metric_facts_provider만 주입해, 두 갈래가
올바르게 분류되는지 확인한다:
  - 핵심 검색 실패(dense 검색) → metadata["retrieval_error"]
  - 선택 기능 실패(DB 지표 조회) → metadata["degraded"], 성공 경로는 계속 완주

가짜로 두는 것은 LLM 호출·임베딩·색인 I/O(doc_retriever)와 SQLite 접근
(metric_facts_provider)뿐이다. KnowledgeGraph·OntologyReasoner는 실제 객체다.
"""

from __future__ import annotations

from typing import Any

import pytest

from src.ontology.knowledge_graph import KnowledgeGraph
from src.ontology.reasoner import OntologyReasoner
from src.rag.hybrid_retriever import HybridRetriever

QUERY = "LANEIGE 립케어 경쟁력 분석"


class _RaisingDocRetriever:
    """dense 검색(핵심 검색 경로)에서 예외를 던지는 가짜 문서 검색기."""

    async def initialize(self) -> None:
        return None

    async def search(
        self, query: str, top_k: int = 5, doc_type_filter: list[str] | None = None
    ) -> list[dict[str, Any]]:
        raise RuntimeError("Error finding id: chroma index corrupted")


class _EmptyDocRetriever:
    """dense 검색은 성공하지만 항상 빈 결과를 돌려주는 가짜 문서 검색기."""

    async def initialize(self) -> None:
        return None

    async def search(
        self, query: str, top_k: int = 5, doc_type_filter: list[str] | None = None
    ) -> list[dict[str, Any]]:
        return []


class _RaisingMetricFactsProvider:
    """DB(SQLite) 지표 사실 조회에서 예외를 던지는 가짜 제공자 — 선택 기능."""

    async def collect(self, entities: dict[str, list[str]]) -> list[dict[str, Any]]:
        raise RuntimeError("database is locked")


def _real_kg(tmp_path) -> KnowledgeGraph:
    """실제 KnowledgeGraph — 빈 임시 파일이라 네트워크·기존 데이터 의존이 없다."""
    return KnowledgeGraph(persist_path=str(tmp_path / "kg.json"), auto_save=False, auto_load=False)


@pytest.mark.asyncio
class TestCoreRetrievalFailureVisibility:
    """핵심 검색 실패 → metadata['retrieval_error'] (F3)."""

    async def test_dense_search_failure_sets_retrieval_error(self, tmp_path):
        kg = _real_kg(tmp_path)
        reasoner = OntologyReasoner(kg)
        retriever = HybridRetriever(
            knowledge_graph=kg,
            reasoner=reasoner,
            doc_retriever=_RaisingDocRetriever(),
            auto_init_rules=True,
        )

        context = await retriever.retrieve(QUERY)

        assert "retrieval_error" in context.metadata
        assert "RuntimeError" in context.metadata["retrieval_error"]
        assert "Error finding id" in context.metadata["retrieval_error"]
        # 기존 소비자 호환용 키도 유지한다
        assert context.metadata.get("error") == "Error finding id: chroma index corrupted"
        # 서비스 동작(빈 컨텍스트로 반환)은 바뀌지 않는다
        assert context.rag_chunks == []
        assert context.combined_context == ""

    async def test_retrieve_unified_legacy_path_also_surfaces_retrieval_error(self, tmp_path):
        """retrieve_unified()의 legacy 변환 경로도 같은 metadata를 그대로 옮긴다."""
        kg = _real_kg(tmp_path)
        reasoner = OntologyReasoner(kg)
        retriever = HybridRetriever(
            knowledge_graph=kg,
            reasoner=reasoner,
            doc_retriever=_RaisingDocRetriever(),
            auto_init_rules=True,
        )

        result = await retriever.retrieve_unified(QUERY)

        assert "retrieval_error" in result.metadata
        assert "RuntimeError" in result.metadata["retrieval_error"]


@pytest.mark.asyncio
class TestOptionalFeatureFailureVisibility:
    """선택 기능(비핵심) 실패 → metadata['degraded'], 채점 경로는 계속 완주 (F3)."""

    async def test_db_metric_facts_failure_is_degraded_not_retrieval_error(self, tmp_path):
        kg = _real_kg(tmp_path)
        reasoner = OntologyReasoner(kg)
        retriever = HybridRetriever(
            knowledge_graph=kg,
            reasoner=reasoner,
            doc_retriever=_EmptyDocRetriever(),
            metric_facts_provider=_RaisingMetricFactsProvider(),
            auto_init_rules=True,
        )

        context = await retriever.retrieve(QUERY)

        # 핵심 실패로 분류되지 않는다
        assert "retrieval_error" not in context.metadata
        assert "error" not in context.metadata

        degraded = context.metadata.get("degraded", [])
        assert any(d["component"] == "db_metric_facts" for d in degraded)
        assert any("database is locked" in d["error"] for d in degraded)

        # 성공 경로를 끝까지 완주했다는 증거 — 이 키들은 정상 종료해야만 채워진다
        assert "retrieval_time_ms" in context.metadata
        assert "search_method" in context.metadata
        assert context.metric_facts == []

    async def test_degraded_list_survives_final_metadata_reassignment(self, tmp_path):
        """retrieve() 끝의 context.metadata = {...} 재할당이 누적된 degraded를 지우지 않는다."""
        kg = _real_kg(tmp_path)
        reasoner = OntologyReasoner(kg)
        retriever = HybridRetriever(
            knowledge_graph=kg,
            reasoner=reasoner,
            doc_retriever=_EmptyDocRetriever(),
            metric_facts_provider=_RaisingMetricFactsProvider(),
            auto_init_rules=True,
        )

        context = await retriever.retrieve(QUERY)

        assert isinstance(context.metadata.get("degraded"), list)
        assert len(context.metadata["degraded"]) >= 1

    async def test_no_failures_means_empty_degraded_list(self, tmp_path):
        """실패가 전혀 없으면 degraded는 빈 리스트로 존재한다 (키 자체는 항상 있음)."""
        kg = _real_kg(tmp_path)
        reasoner = OntologyReasoner(kg)
        retriever = HybridRetriever(
            knowledge_graph=kg,
            reasoner=reasoner,
            doc_retriever=_EmptyDocRetriever(),
            auto_init_rules=True,
        )

        context = await retriever.retrieve(QUERY)

        assert "retrieval_error" not in context.metadata
        assert context.metadata.get("degraded") == []


@pytest.mark.asyncio
class TestRerankerDisabledIsNotDegraded:
    """reranker 플래그 OFF(기본값)는 실패가 아니라 정상 분기 — degraded에 넣지 않는다."""

    async def test_reranker_disabled_by_default_does_not_appear_in_degraded(self, tmp_path):
        kg = _real_kg(tmp_path)
        reasoner = OntologyReasoner(kg)
        retriever = HybridRetriever(
            knowledge_graph=kg,
            reasoner=reasoner,
            doc_retriever=_EmptyDocRetriever(),
            auto_init_rules=True,
        )

        context = await retriever.retrieve(QUERY)

        components = {d["component"] for d in context.metadata.get("degraded", [])}
        assert "relevance_grading" not in components
