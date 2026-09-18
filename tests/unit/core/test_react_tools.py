"""
ReAct 도구 실행기 어댑터 검증 (트랙 4-A)

ReAct는 별도 도구 3종(query_data·query_knowledge_graph·calculate_metrics)을 갖고 있었지만
이제 DecisionMaker와 같은 레지스트리(src/core/tool_registry.py)를 쓴다. 여기서는 어댑터가
같은 도구 목록을 내고, 관찰이 카드 렌더링 문자열이 되는지 본다.

실제 객체: SQLite·KG(임시 경로)·HybridRetriever·어댑터. 가짜는 문서 검색기뿐이다.
"""

from __future__ import annotations

import pytest

from src.core.react_tools import ReActToolExecutor, build_react_tool_executor
from src.core.tool_registry import TOOL_NAMES, ToolRegistry, tool_evidence
from src.domain.entities.evidence import EvidenceKind
from src.infrastructure.feature_flags import FeatureFlags
from src.rag.hybrid_retriever import HybridRetriever
from src.rag.metric_facts import MetricFactsProvider
from tests.unit.rag.evidence_pipeline_fixtures import (
    AS_OF,
    FakeDocRetriever,
    make_kg,
    make_metrics_db,
)


@pytest.fixture(autouse=True)
def isolated_flags():
    FeatureFlags.reset_instance()
    yield
    FeatureFlags.reset_instance()


@pytest.fixture
def registry(tmp_path) -> ToolRegistry:
    retriever = HybridRetriever(
        knowledge_graph=make_kg(tmp_path),
        doc_retriever=FakeDocRetriever(),
        metric_facts_provider=MetricFactsProvider(make_metrics_db(tmp_path), as_of=AS_OF),
    )
    return ToolRegistry(retriever)


@pytest.fixture
def executor(registry) -> ReActToolExecutor:
    return build_react_tool_executor(registry)


def test_react_executor_exposes_the_registry_tools(executor, registry):
    assert executor.get_available_tools() == list(TOOL_NAMES)
    assert executor.registry is registry


@pytest.mark.asyncio
async def test_observation_is_the_card_rendering_with_ids(executor):
    result = await executor.execute("get_metrics", {"brand": "LANEIGE", "category": "lip_care"})

    assert result.success, result.error
    observation = str(result.data)  # ReAct 루프가 관찰로 쓰는 문자열
    assert "[DB 수치]" in observation
    cards = tool_evidence(result)
    assert cards and {c.kind for c in cards} == {EvidenceKind.METRIC}
    for card in cards:
        assert f"[{card.id}]" in observation
    assert AS_OF in observation
    # 구조화된 데이터도 그대로 남는다 (data는 dict이다)
    assert result.data["as_of"] == AS_OF


@pytest.mark.asyncio
async def test_kg_observation_excludes_undated_numeric_edges(executor):
    result = await executor.execute("kg_neighbors", {"entity": "LANEIGE"})

    observation = str(result.data)
    assert "ownedBy" in observation
    assert "hasSoS" not in observation and "hasHHI" not in observation


@pytest.mark.asyncio
async def test_failure_is_returned_not_raised(executor):
    result = await executor.execute("get_metrics", {"unknown_param": 1})

    assert result.success is False
    assert "unknown_param" in result.error


@pytest.mark.asyncio
async def test_unbound_registry_reports_no_tools():
    executor = build_react_tool_executor(ToolRegistry())

    assert executor.get_available_tools() == []
    result = await executor.execute("search_docs", {"query": "HHI"})
    assert result.success is False
