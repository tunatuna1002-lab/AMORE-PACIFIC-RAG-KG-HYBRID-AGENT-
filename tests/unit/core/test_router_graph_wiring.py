"""
홉 수 라우터 × QueryGraph 배선 + ReAct 섀도 모드 (트랙 5-C)
============================================================

검증 대상
- MEDIUM/LOW 신뢰도에서 경로를 홉 수가 정한다 (1홉 → decide, 2홉 이상 → react)
- route_trace에 hops·router_stages·router_reason·router_basis·router_route가 남는다
- 섀도 모드: 답변은 파이프라인 것을 그대로 돌려주고 ReAct 결과는 metadata에만 적힌다
- 섀도 실패는 응답을 깨뜨리지 않는다

실제 객체: QueryGraph, HopRouter, ConfidenceAssessor, ResponseCache
가짜(LLM/검색 I/O 없음): ContextGatherer, DecisionMaker, ToolCoordinator, ResponsePipeline,
ReActAgent
"""

from __future__ import annotations

import pytest

from src.core.cache import ResponseCache
from src.core.confidence import ConfidenceAssessor
from src.core.graph_state import QueryState
from src.core.models import Context, Decision, Response, ToolResult
from src.core.query_graph import REACT_MODE_SHADOW, QueryGraph
from src.core.react_agent import ReActResult, ReActStep
from src.core.router import HopRouter

PIPELINE_TEXT = "파이프라인이 만든 답변"
SHADOW_TEXT = "섀도 ReAct가 만든 답변"

# 1홉(수치 조회 한 번)과 3홉(엔티티 → 관계 → 수치) 질문
Q_ONE_HOP = "LANEIGE Lip Care SoS는 얼마인가요?"
Q_MULTI_HOP = "LANEIGE 제품이 속한 카테고리의 HHI는?"


class FakeGatherer:
    def __init__(self, context: Context) -> None:
        self._context = context

    async def gather(self, query: str, current_metrics=None):
        return self._context


class FakeDecisionMaker:
    def __init__(self) -> None:
        self.calls = 0

    async def decide(self, query, context, system_state, confidence_level="medium"):
        self.calls += 1
        return Decision(tool="direct_answer", tool_params={}, confidence=0.5, reason="fake")


class FakeToolCoordinator:
    async def execute(self, tool_name: str, params: dict):
        return ToolResult(tool_name=tool_name, success=True, data={})


class FakePipeline:
    async def generate(self, query, context, decision, tool_result):
        return Response(text=PIPELINE_TEXT, confidence_score=0.5, sources=["sqlite:brand_metrics"])


class FakeReActAgent:
    """미리 정해진 결과(또는 예외)를 돌려주는 가짜 ReAct."""

    def __init__(self, result: ReActResult | None = None, error: Exception | None = None) -> None:
        self._result = result
        self._error = error
        self.calls: list[str] = []

    async def run(self, query: str, context: str):
        self.calls.append(query)
        if self._error is not None:
            raise self._error
        return self._result


def _shadow_result() -> ReActResult:
    return ReActResult(
        final_answer=SHADOW_TEXT,
        steps=[
            ReActStep(thought="관계부터", action="kg_neighbors", observation="[K-1] ownedBy"),
            ReActStep(thought="수치", action="get_metrics", observation="[M-1] SoS 2.0%"),
        ],
        iterations=2,
        confidence=0.8,
        hop_count=1,
    )


def _thin_context(query: str) -> Context:
    """MEDIUM/LOW 신뢰도가 나오는 얇은 컨텍스트 (HIGH면 라우터까지 가지 않는다)."""
    return Context(query=query, entities={"brands": ["laneige"]}, summary="컨텍스트 요약")


def make_graph(query: str, *, react_agent=None, react_mode=None) -> QueryGraph:
    return QueryGraph(
        cache=ResponseCache(),
        context_gatherer=FakeGatherer(_thin_context(query)),
        confidence_assessor=ConfidenceAssessor(),
        decision_maker=FakeDecisionMaker(),
        tool_coordinator=FakeToolCoordinator(),
        response_pipeline=FakePipeline(),
        react_agent=react_agent,
        router=HopRouter(llm_fallback=False),
        react_mode=react_mode,
    )


# =============================================================================
# 경로 판정
# =============================================================================


@pytest.mark.asyncio
async def test_one_hop_question_goes_to_decide_even_with_react_available():
    agent = FakeReActAgent(_shadow_result())
    graph = make_graph(Q_ONE_HOP, react_agent=agent)

    state = await graph.run(QueryState(query=Q_ONE_HOP))

    trace = state.metadata["route_trace"]
    assert trace["route"] == "decide"
    assert trace["hops"] == 1
    assert trace["router_route"] == "pipeline"
    assert agent.calls == []


@pytest.mark.asyncio
async def test_multi_hop_question_goes_to_react():
    agent = FakeReActAgent(_shadow_result())
    graph = make_graph(Q_MULTI_HOP, react_agent=agent)

    state = await graph.run(QueryState(query=Q_MULTI_HOP))

    trace = state.metadata["route_trace"]
    assert trace["route"] == "react"
    assert trace["hops"] == 3
    assert trace["is_complex"] is True
    assert agent.calls == [Q_MULTI_HOP]


@pytest.mark.asyncio
async def test_route_trace_carries_router_fields():
    graph = make_graph(Q_MULTI_HOP)

    state = await graph.run(QueryState(query=Q_MULTI_HOP))

    trace = state.metadata["route_trace"]
    assert trace["hops"] == 3
    assert trace["router_stages"] == ["entity_resolution", "relation", "metric"]
    assert trace["router_basis"] == "rules"
    assert trace["router_route"] == "react"
    assert "entity_resolution" in trace["router_reason"]
    # 응답에도 같은 dict가 실린다
    assert state.response.metadata["route_trace"] == trace


@pytest.mark.asyncio
async def test_react_unavailable_falls_back_to_decide_but_still_records_hops():
    """ReAct가 없으면 2홉 질문도 파이프라인으로 간다 — 판정 자체는 기록된다."""
    graph = make_graph(Q_MULTI_HOP)

    state = await graph.run(QueryState(query=Q_MULTI_HOP))

    trace = state.metadata["route_trace"]
    assert trace["route"] == "decide"
    assert trace["hops"] == 3
    assert trace["is_complex"] is None  # ReAct가 없으면 관측 필드를 채우지 않는다


# =============================================================================
# 섀도 모드
# =============================================================================


@pytest.mark.asyncio
async def test_shadow_mode_returns_pipeline_answer_and_records_react():
    agent = FakeReActAgent(_shadow_result())
    graph = make_graph(Q_MULTI_HOP, react_agent=agent, react_mode=REACT_MODE_SHADOW)

    state = await graph.run(QueryState(query=Q_MULTI_HOP))

    # 반환 텍스트는 파이프라인 것 그대로다
    assert state.response.text == PIPELINE_TEXT
    assert state.metadata["route_trace"]["route"] == "decide"

    shadow = state.response.metadata["react_shadow"]
    assert shadow["ran"] is True
    assert shadow["error"] is None
    assert shadow["answer"] == SHADOW_TEXT
    assert shadow["tools"] == ["kg_neighbors", "get_metrics"]
    assert shadow["iterations"] == 2
    assert len(shadow["steps"]) == 2
    assert isinstance(shadow["token_usage"], dict)
    assert shadow["elapsed_ms"] >= 0
    # route_trace에도 같은 기록이 실린다 (평가에서 집계한다)
    assert state.metadata["route_trace"]["react_shadow"] == shadow
    assert agent.calls == [Q_MULTI_HOP]


@pytest.mark.asyncio
async def test_shadow_mode_skips_one_hop_questions():
    agent = FakeReActAgent(_shadow_result())
    graph = make_graph(Q_ONE_HOP, react_agent=agent, react_mode=REACT_MODE_SHADOW)

    state = await graph.run(QueryState(query=Q_ONE_HOP))

    assert state.response.text == PIPELINE_TEXT
    assert agent.calls == []
    assert "react_shadow" not in state.metadata["route_trace"]


@pytest.mark.asyncio
async def test_shadow_failure_does_not_break_the_response():
    agent = FakeReActAgent(error=RuntimeError("도구 폭발"))
    graph = make_graph(Q_MULTI_HOP, react_agent=agent, react_mode=REACT_MODE_SHADOW)

    state = await graph.run(QueryState(query=Q_MULTI_HOP))

    assert state.response.text == PIPELINE_TEXT
    shadow = state.response.metadata["react_shadow"]
    assert shadow["ran"] is False
    assert "도구 폭발" in shadow["error"]


@pytest.mark.asyncio
async def test_shadow_mode_never_routes_to_react():
    """섀도는 관측일 뿐이다 — 답변 경로가 react로 바뀌면 안 된다."""
    agent = FakeReActAgent(_shadow_result())
    graph = make_graph(Q_MULTI_HOP, react_agent=agent, react_mode=REACT_MODE_SHADOW)

    state = await graph.run(QueryState(query=Q_MULTI_HOP))

    assert state.metadata["route_trace"]["route"] != "react"
    assert state.response.text != SHADOW_TEXT
