"""
ReAct 네이티브 function calling (트랙 5-C)
==========================================

무엇을 검증하나
- 도구 선택은 ``tools=``/``tool_choice``로 한다. 프롬프트에서 JSON 문자열을 파싱하지 않는다.
- 모델에 넘기는 스키마는 단일 레지스트리(``tool_registry``)가 만든 것 + 루프 제어 2종
  (final_answer·refine_search)이다.
- 관찰은 증거 카드 렌더링 그대로이고, 도구 결과는 ``role="tool"`` 메시지로 되돌아간다.
- 반복 한도와 **질문당 토큰 예산**을 지킨다.
- ``ReActResult.sources``는 ``list[str]``이라 ``/api/v4/chat`` 응답 모델에 그대로 실린다.

LLM 호출(acompletion)만 가짜다. 도구 실행기는 실제 레지스트리를 어댑터로 감싼 것이다.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from src.api.models import BrainChatResponse
from src.core.react_agent import ReActAgent, ReActResult
from src.core.react_tools import build_react_tool_executor
from src.core.tool_registry import TOOL_NAMES, ToolRegistry
from src.rag.hybrid_retriever import HybridRetriever
from src.rag.metric_facts import MetricFactsProvider
from tests.unit.core.react_fc_fixtures import json_reply, text_reply, tool_call_reply
from tests.unit.rag.evidence_pipeline_fixtures import (
    AS_OF,
    FakeDocRetriever,
    make_kg,
    make_metrics_db,
)

REFLECTION = json_reply({"quality_score": 0.8, "needs_improvement": False})


def _bound_registry(tmp_path) -> ToolRegistry:
    retriever = HybridRetriever(
        knowledge_graph=make_kg(tmp_path),
        doc_retriever=FakeDocRetriever(),
        metric_facts_provider=MetricFactsProvider(make_metrics_db(tmp_path), as_of=AS_OF),
    )
    return ToolRegistry(retriever)


def _agent(tmp_path, **kwargs) -> ReActAgent:
    agent = ReActAgent(ircot_enabled=False, **kwargs)
    agent.set_tool_executor(build_react_tool_executor(_bound_registry(tmp_path)))
    return agent


# =============================================================================
# 스키마: 레지스트리 정의 + 루프 제어 2종
# =============================================================================


def test_tool_schemas_are_registry_schemas_plus_loop_controls(tmp_path):
    schemas = _agent(tmp_path).tool_schemas()
    names = [s["function"]["name"] for s in schemas]

    assert set(TOOL_NAMES).issubset(names)
    assert "final_answer" in names
    assert "refine_search" in names
    for schema in schemas:
        assert schema["type"] == "function"
        assert schema["function"]["parameters"]["type"] == "object"


@pytest.mark.asyncio
async def test_loop_passes_tools_and_tool_choice_to_litellm(tmp_path):
    agent = _agent(tmp_path)
    llm = AsyncMock(
        side_effect=[
            tool_call_reply("최종", "final_answer", {"answer": "답", "confidence": 0.9}),
            REFLECTION,
        ]
    )
    with patch("src.core.react_agent.acompletion", llm):
        await agent.run("질문", "컨텍스트")

    first_kwargs = llm.await_args_list[0].kwargs
    assert first_kwargs["tool_choice"] == "auto"
    assert [s["function"]["name"] for s in first_kwargs["tools"]]
    assert first_kwargs["messages"][0]["role"] == "system"


# =============================================================================
# 도구 실행 · 관찰
# =============================================================================


@pytest.mark.asyncio
async def test_tool_call_runs_the_registry_tool_and_keeps_evidence_ids(tmp_path):
    agent = _agent(tmp_path)
    llm = AsyncMock(
        side_effect=[
            tool_call_reply("관계 확인", "kg_neighbors", {"entity": "laneige"}),
            tool_call_reply(
                "정리", "final_answer", {"answer": "LANEIGE는 아모레 소속", "confidence": 0.8}
            ),
            REFLECTION,
        ]
    )
    with patch("src.core.react_agent.acompletion", llm):
        result = await agent.run("LANEIGE 모회사는?", "컨텍스트")

    assert result.steps[0].action == "kg_neighbors"
    assert "ownedBy" in result.steps[0].observation
    assert result.final_answer == "LANEIGE는 아모레 소속"

    # 두 번째 호출의 메시지에 tool 역할 관찰이 실제로 들어간다 (모델이 관찰을 본다)
    second_messages = llm.await_args_list[1].kwargs["messages"]
    roles = [m["role"] for m in second_messages]
    assert "assistant" in roles and "tool" in roles
    tool_message = next(m for m in second_messages if m["role"] == "tool")
    assert tool_message["name"] == "kg_neighbors"
    assert "ownedBy" in tool_message["content"]


@pytest.mark.asyncio
async def test_disallowed_tool_name_is_blocked_before_the_registry(tmp_path):
    agent = _agent(tmp_path)
    llm = AsyncMock(
        side_effect=[
            tool_call_reply("크롤 시도", "crawl_amazon", {}),
            tool_call_reply("정리", "final_answer", {"answer": "답"}),
            REFLECTION,
        ]
    )
    with (
        patch("src.core.react_agent.acompletion", llm),
        patch.object(ToolRegistry, "execute", AsyncMock()) as execute,
    ):
        result = await agent.run("질문", "컨텍스트")

    assert result.steps[0].observation.startswith("Security Error")
    execute.assert_not_awaited()
    assert result.final_answer == "답"


@pytest.mark.asyncio
async def test_content_without_tool_calls_becomes_the_final_answer(tmp_path):
    agent = _agent(tmp_path)
    llm = AsyncMock(side_effect=[text_reply("도구 없이 바로 답한다"), REFLECTION])
    with patch("src.core.react_agent.acompletion", llm):
        result = await agent.run("질문", "컨텍스트")

    assert result.final_answer == "도구 없이 바로 답한다"
    assert result.iterations == 1


# =============================================================================
# 한도: 반복 · 토큰 예산
# =============================================================================


@pytest.mark.asyncio
async def test_step_limit_is_respected(tmp_path):
    agent = _agent(tmp_path, max_iterations=2)
    step = tool_call_reply("계속 조회", "kg_neighbors", {"entity": "laneige"})
    llm = AsyncMock(side_effect=[step, step, text_reply("관찰 기반 요약"), REFLECTION])
    with patch("src.core.react_agent.acompletion", llm):
        result = await agent.run("질문", "컨텍스트")

    assert result.iterations == 2
    assert len(result.steps) == 2
    assert result.final_answer == "관찰 기반 요약"
    assert result.budget_exceeded is False


@pytest.mark.asyncio
async def test_token_budget_stops_the_loop_before_the_step_limit(tmp_path):
    agent = _agent(tmp_path, max_iterations=5, max_total_tokens=50)
    step = tool_call_reply("계속 조회", "kg_neighbors", {"entity": "laneige"}, total_tokens=40)
    # 스텝 응답을 5개나 준비해도 예산이 2개에서 끊는다 (남은 건 소비되지 않는다)
    llm = AsyncMock(side_effect=[step, step, text_reply("예산 소진 요약"), REFLECTION])
    with patch("src.core.react_agent.acompletion", llm):
        result = await agent.run("질문", "컨텍스트")

    assert result.budget_exceeded is True
    assert result.iterations == 2  # 40 → 80 (>= 50)에서 멈춘다
    assert result.token_usage["total_tokens"] >= 80
    assert result.final_answer == "예산 소진 요약"


@pytest.mark.asyncio
async def test_token_usage_is_recorded_even_without_usage_field(tmp_path):
    """usage를 주지 않는 응답에서도 집계가 깨지지 않는다 (0으로 센다)."""
    agent = _agent(tmp_path)
    llm = AsyncMock(
        side_effect=[tool_call_reply("끝", "final_answer", {"answer": "답"}), REFLECTION]
    )
    with patch("src.core.react_agent.acompletion", llm):
        result = await agent.run("질문", "컨텍스트")

    assert result.token_usage == {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


# =============================================================================
# sources: API 응답 모델이 받는 모양이어야 한다
# =============================================================================


@pytest.mark.asyncio
async def test_sources_are_strings_accepted_by_the_chat_response_model(tmp_path):
    agent = _agent(tmp_path)
    llm = AsyncMock(
        side_effect=[
            tool_call_reply(
                "수치 조회", "get_metrics", {"brand": "LANEIGE", "category": "Lip Care"}
            ),
            tool_call_reply("정리", "final_answer", {"answer": "SoS는 …"}),
            REFLECTION,
        ]
    )
    with patch("src.core.react_agent.acompletion", llm):
        result = await agent.run("LANEIGE Lip Care SoS는?", "컨텍스트")

    assert isinstance(result.sources, list)
    assert result.sources, "도구가 돌려준 근거 카드에서 출처 라벨이 나와야 한다"
    assert all(isinstance(s, str) for s in result.sources)

    # /api/v4/chat 응답 모델(sources: list[str])이 그대로 받아야 한다
    payload = BrainChatResponse(
        text=result.final_answer,
        confidence=result.confidence,
        sources=result.sources,
        tools_used=[s.action for s in result.steps if s.action],
        processing_time_ms=1.0,
        from_cache=False,
        brain_mode="autonomous",
    )
    assert payload.sources == result.sources


def test_react_result_defaults_are_api_safe():
    result = ReActResult(final_answer="답")
    assert result.sources == []
    assert result.token_usage == {}
    assert result.budget_exceeded is False


# =============================================================================
# 그래프 경로: ReAct 응답이 API 응답 모델을 통과해야 한다
# =============================================================================


@pytest.mark.asyncio
async def test_graph_react_node_puts_string_sources_on_the_response(tmp_path):
    """QueryGraph._node_react가 만드는 Response.sources가 list[str]이어야 한다.

    예전에는 context.rag_docs(dict 목록)를 그대로 실어 /api/v4/chat 응답 모델
    (sources: list[str])에서 검증 오류가 났다.
    """
    from src.core.graph_state import QueryState
    from src.core.models import Context
    from src.core.query_graph import QueryGraph

    agent = _agent(tmp_path)
    llm = AsyncMock(
        side_effect=[
            tool_call_reply("수치", "get_metrics", {"brand": "LANEIGE", "category": "Lip Care"}),
            tool_call_reply("정리", "final_answer", {"answer": "SoS는 …"}),
            REFLECTION,
        ]
    )

    graph = QueryGraph(
        cache=None,
        context_gatherer=None,
        confidence_assessor=None,
        decision_maker=None,
        tool_coordinator=None,
        response_pipeline=None,
        react_agent=agent,
    )
    context = Context(query="LANEIGE Lip Care SoS는?", summary="요약")
    # 옛 버그 재현 조건: rag_docs가 dict 목록으로 채워져 있다
    context.rag_docs = [{"content": "doc", "title": "문서"}]
    state = QueryState(query=context.query)
    state.context = context

    with patch("src.core.react_agent.acompletion", llm):
        await graph._node_react(state)

    assert all(isinstance(s, str) for s in state.response.sources)
    BrainChatResponse(
        text=state.response.text,
        confidence=state.response.confidence_score,
        sources=state.response.sources,
        tools_used=state.response.tools_called,
        processing_time_ms=1.0,
        from_cache=False,
        brain_mode="autonomous",
    )
