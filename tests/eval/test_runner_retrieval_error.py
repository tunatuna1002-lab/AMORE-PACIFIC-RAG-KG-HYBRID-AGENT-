"""검색 오류 가시화(F3)가 평가 러너·리포트까지 이어지는지 확인하는 회귀 테스트.

배경: HybridRetriever.retrieve()가 검색 실패를 삼키고 빈 컨텍스트로 계속 진행하면,
에이전트는 예외를 던지지 않은 채 응답을 생성한다. 러너가 hybrid_context.metadata의
retrieval_error를 보지 않으면 이 문항은 "근거 없는 정상 응답"으로 채점되어 지표를
오염시킨다. 이 테스트는 실제 HybridRetriever(+ 실제 KnowledgeGraph/OntologyReasoner)를
쓰는 최소 가짜 에이전트(답변 생성만 가짜)로 러너·리포트 전체 경로를 검증한다.

가짜로 두는 것: doc_retriever(색인 I/O), metric_facts_provider(SQLite), 그리고
답변 생성 자체(LLM 호출 없음). judge는 기본 StubJudge(use_judge=False)라 LLM을
호출하지 않는다.
"""

from __future__ import annotations

from typing import Any

import pytest

from eval.report import ReportGenerator
from eval.runner import EvalRunner
from eval.schemas import EvalConfig, EvalItem, GoldEvidence
from src.ontology.knowledge_graph import KnowledgeGraph
from src.ontology.reasoner import OntologyReasoner
from src.rag.hybrid_retriever import HybridRetriever

QUESTION = "LANEIGE 립케어 경쟁력 분석"


class _RaisingDocRetriever:
    async def initialize(self) -> None:
        return None

    async def search(
        self, query: str, top_k: int = 5, doc_type_filter: list[str] | None = None
    ) -> list[dict[str, Any]]:
        raise RuntimeError("Error finding id: chroma index corrupted")


class _EmptyDocRetriever:
    async def initialize(self) -> None:
        return None

    async def search(
        self, query: str, top_k: int = 5, doc_type_filter: list[str] | None = None
    ) -> list[dict[str, Any]]:
        return []


class _RaisingMetricFactsProvider:
    async def collect(self, entities: dict[str, list[str]]) -> list[dict[str, Any]]:
        raise RuntimeError("database is locked")


class _RetrieverBackedAgent:
    """실제 HybridRetriever를 쓰되 답변 생성만 가짜인 최소 에이전트 (LLM 호출 없음)."""

    model = "gpt-4.1-mini"

    def __init__(self, retriever: HybridRetriever, answer: str = "테스트 답변입니다.") -> None:
        self.retriever = retriever
        self.answer = answer

    async def chat(self, question: str) -> dict[str, Any]:
        context = await self.retriever.retrieve(question)
        return {
            "response": self.answer,
            "sources": [],
            "citations": [],
            "hybrid_context": context,
        }


def _real_retriever(tmp_path, doc_retriever, metric_facts_provider=None) -> HybridRetriever:
    kg = KnowledgeGraph(persist_path=str(tmp_path / "kg.json"), auto_save=False, auto_load=False)
    reasoner = OntologyReasoner(kg)
    return HybridRetriever(
        knowledge_graph=kg,
        reasoner=reasoner,
        doc_retriever=doc_retriever,
        metric_facts_provider=metric_facts_provider,
        auto_init_rules=True,
    )


def _item(item_id: str = "t001") -> EvalItem:
    return EvalItem(
        id=item_id,
        question=QUESTION,
        gold=GoldEvidence(answer="LANEIGE의 Lip Care SoS는 2.0%입니다."),
    )


@pytest.mark.asyncio
class TestRunnerClassifiesRetrievalErrorAsInfrastructureFailure:
    async def test_core_retrieval_failure_is_excluded_from_scoring(self, tmp_path):
        retriever = _real_retriever(tmp_path, _RaisingDocRetriever())
        runner = EvalRunner(agent=_RetrieverBackedAgent(retriever), config=EvalConfig())

        result = await runner.run_item(_item())

        assert result.trace is not None
        assert result.trace.error is not None
        assert result.trace.error.startswith("retrieval_error:")
        assert "chroma index corrupted" in result.trace.error
        assert result.passed is False
        # 인프라 실패는 모델 실패 사유로 태깅하지 않는다 (기존 관례와 동일)
        assert result.fail_reason_tags == []

    async def test_report_counts_it_as_errored_not_scored(self, tmp_path):
        retriever = _real_retriever(tmp_path, _RaisingDocRetriever())
        runner = EvalRunner(agent=_RetrieverBackedAgent(retriever), config=EvalConfig())

        result = await runner.run_item(_item("bad001"))
        report = ReportGenerator().generate_report([result], tmp_path / "out")

        assert report.aggregates.total == 0  # 채점된 문항 없음
        assert report.aggregates.errored == 1
        assert report.aggregates.error_item_ids == ["bad001"]


@pytest.mark.asyncio
class TestRunnerLeavesDegradedItemsScored:
    async def test_optional_feature_failure_does_not_block_scoring(self, tmp_path):
        retriever = _real_retriever(
            tmp_path, _EmptyDocRetriever(), metric_facts_provider=_RaisingMetricFactsProvider()
        )
        runner = EvalRunner(agent=_RetrieverBackedAgent(retriever), config=EvalConfig())

        result = await runner.run_item(_item("degraded001"))

        # 인프라 실패로 빠지지 않는다 — 채점이 계속 진행됐다
        assert result.trace.error is None
        assert result.trace.degraded
        assert any(d["component"] == "db_metric_facts" for d in result.trace.degraded)

    async def test_report_counts_degraded_items_while_still_scoring(self, tmp_path):
        retriever = _real_retriever(
            tmp_path, _EmptyDocRetriever(), metric_facts_provider=_RaisingMetricFactsProvider()
        )
        runner = EvalRunner(agent=_RetrieverBackedAgent(retriever), config=EvalConfig())

        result = await runner.run_item(_item("degraded002"))
        report = ReportGenerator().generate_report([result], tmp_path / "out")

        assert report.aggregates.total == 1  # 채점됐다
        assert report.aggregates.errored == 0
        assert report.aggregates.degraded_items == 1
