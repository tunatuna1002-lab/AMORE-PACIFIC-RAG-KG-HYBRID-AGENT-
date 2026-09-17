"""
ReActAgent 루프 결함 수리 검증 (R1, 2026-09)

- final_answer가 채워지지 않는 `step.observation`에서 답을 읽어 항상 빈 문자열이던 문제
- 반복 한도에 걸리면 최종 답이 빈 문자열로 끝나던 문제
- 허용됐지만 백엔드가 없는 도구, 허용되지 않은 도구를 골랐을 때의 동작

LLM 호출만 가짜로 두고 도구 실행기는 실제 객체를 쓴다 — 단일 레지스트리
(``src/core/tool_registry.py``)를 ReAct 어댑터로 감싼 것이다 (트랙 4-A).
문서 검색기만 가짜다.
"""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from src.core.react_agent import ReActAgent
from src.core.react_tools import build_react_tool_executor
from src.core.tool_registry import ToolRegistry
from src.rag.hybrid_retriever import HybridRetriever
from src.rag.metric_facts import MetricFactsProvider
from tests.unit.rag.evidence_pipeline_fixtures import (
    AS_OF,
    FakeDocRetriever,
    make_kg,
    make_metrics_db,
)


def _reply(payload: dict) -> SimpleNamespace:
    content = json.dumps(payload, ensure_ascii=False)
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


REFLECTION = _reply({"quality_score": 0.7, "needs_improvement": False})


def _bound_registry(tmp_path) -> ToolRegistry:
    retriever = HybridRetriever(
        knowledge_graph=make_kg(tmp_path),
        doc_retriever=FakeDocRetriever(),
        metric_facts_provider=MetricFactsProvider(make_metrics_db(tmp_path), as_of=AS_OF),
    )
    return ToolRegistry(retriever)


@pytest.mark.asyncio
async def test_final_answer_comes_from_action_input(tmp_path):
    agent = ReActAgent()
    agent.set_tool_executor(build_react_tool_executor(_bound_registry(tmp_path)))
    llm = AsyncMock(
        side_effect=[
            _reply({"thought": "t", "action": "final_answer", "action_input": {"answer": "답"}}),
            REFLECTION,
        ]
    )
    with patch("src.core.react_agent.acompletion", llm):
        result = await agent.run("질문", "컨텍스트")

    assert result.final_answer == "답"


@pytest.mark.asyncio
async def test_exhausted_iterations_still_produce_nonempty_answer(tmp_path):
    agent = ReActAgent(max_iterations=2, ircot_enabled=False)
    agent.set_tool_executor(build_react_tool_executor(_bound_registry(tmp_path)))
    step = {"thought": "조회", "action": "kg_neighbors", "action_input": {"entity": "laneige"}}
    llm = AsyncMock(
        side_effect=[
            _reply(step),
            _reply(step),
            SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="관찰 기반 요약 답변"))]
            ),
            REFLECTION,
        ]
    )
    with patch("src.core.react_agent.acompletion", llm):
        result = await agent.run("질문", "컨텍스트")

    assert result.final_answer == "관찰 기반 요약 답변"
    assert result.iterations == 2
    assert "ownedBy" in result.steps[0].observation  # 실제 KG 관계 카드가 관찰에 실린다


@pytest.mark.asyncio
async def test_allowed_tool_without_backend_records_error_and_continues():
    agent = ReActAgent(ircot_enabled=False)
    agent.set_tool_executor(build_react_tool_executor(ToolRegistry()))  # 백엔드 미연결
    llm = AsyncMock(
        side_effect=[
            _reply({"thought": "t", "action": "apply_rules", "action_input": {}}),
            _reply({"thought": "t", "action": "final_answer", "action_input": {"answer": "답"}}),
            REFLECTION,
        ]
    )
    with patch("src.core.react_agent.acompletion", llm):
        result = await agent.run("질문", "컨텍스트")

    assert "연결되지 않았습니다" in result.steps[0].observation
    assert result.final_answer == "답"


@pytest.mark.asyncio
async def test_disallowed_tool_is_blocked_and_loop_continues(tmp_path):
    agent = ReActAgent(ircot_enabled=False)
    registry = _bound_registry(tmp_path)
    agent.set_tool_executor(build_react_tool_executor(registry))
    llm = AsyncMock(
        side_effect=[
            _reply({"thought": "t", "action": "crawl_amazon", "action_input": {}}),
            _reply({"thought": "t", "action": "final_answer", "action_input": {"answer": "답"}}),
            REFLECTION,
        ]
    )
    with (
        patch("src.core.react_agent.acompletion", llm),
        patch.object(ToolRegistry, "execute", AsyncMock()) as execute,
    ):
        result = await agent.run("질문", "컨텍스트")

    assert result.steps[0].observation.startswith("Security Error")
    execute.assert_not_awaited()  # 허용 목록에서 막혀 레지스트리까지 가지 않는다
    assert result.final_answer == "답"


@pytest.mark.asyncio
async def test_missing_tool_executor_records_error():
    agent = ReActAgent(ircot_enabled=False)
    llm = AsyncMock(
        side_effect=[
            _reply({"thought": "t", "action": "get_metrics", "action_input": {}}),
            _reply({"thought": "t", "action": "final_answer", "action_input": {"answer": "답"}}),
            REFLECTION,
        ]
    )
    with patch("src.core.react_agent.acompletion", llm):
        result = await agent.run("질문", "컨텍스트")

    assert "실행기" in result.steps[0].observation
