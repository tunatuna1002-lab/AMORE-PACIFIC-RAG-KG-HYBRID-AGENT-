"""
ReActAgent 루프 결함 수리 검증 (R1, 2026-09)

- final_answer가 채워지지 않는 `step.observation`에서 답을 읽어 항상 빈 문자열이던 문제
- 반복 한도에 걸리면 최종 답이 빈 문자열로 끝나던 문제
- 허용됐지만 실행기가 없는 도구, 허용되지 않은 도구를 골랐을 때의 동작

LLM 호출만 가짜로 두고 ToolExecutor는 실제 객체를 쓴다.
"""

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from src.core.react_agent import ReActAgent
from src.core.tools import ToolExecutor


def _reply(payload: dict) -> SimpleNamespace:
    content = json.dumps(payload, ensure_ascii=False)
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


REFLECTION = _reply({"quality_score": 0.7, "needs_improvement": False})


@pytest.mark.asyncio
async def test_final_answer_comes_from_action_input():
    agent = ReActAgent()
    agent.set_tool_executor(ToolExecutor())
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
async def test_exhausted_iterations_still_produce_nonempty_answer():
    agent = ReActAgent(max_iterations=2, ircot_enabled=False)
    executor = ToolExecutor()

    async def _kg(**kwargs):
        return {"competitors": ["aquaphor"]}

    executor.register_executor("query_knowledge_graph", _kg)
    agent.set_tool_executor(executor)
    step = {"thought": "조회", "action": "query_knowledge_graph", "action_input": {"entity": "x"}}
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


@pytest.mark.asyncio
async def test_allowed_but_unregistered_tool_records_error_and_continues():
    agent = ReActAgent(ircot_enabled=False)
    agent.set_tool_executor(ToolExecutor())  # 실행기 없음
    llm = AsyncMock(
        side_effect=[
            _reply({"thought": "t", "action": "calculate_metrics", "action_input": {}}),
            _reply({"thought": "t", "action": "final_answer", "action_input": {"answer": "답"}}),
            REFLECTION,
        ]
    )
    with patch("src.core.react_agent.acompletion", llm):
        result = await agent.run("질문", "컨텍스트")

    assert "실행기가 없습니다" in result.steps[0].observation
    assert result.final_answer == "답"


@pytest.mark.asyncio
async def test_disallowed_tool_is_blocked_and_loop_continues():
    agent = ReActAgent(ircot_enabled=False)
    executor = ToolExecutor()
    crawl = AsyncMock(return_value={"ok": True})
    executor.register_executor("crawl_amazon", crawl)
    agent.set_tool_executor(executor)
    llm = AsyncMock(
        side_effect=[
            _reply({"thought": "t", "action": "crawl_amazon", "action_input": {}}),
            _reply({"thought": "t", "action": "final_answer", "action_input": {"answer": "답"}}),
            REFLECTION,
        ]
    )
    with patch("src.core.react_agent.acompletion", llm):
        result = await agent.run("질문", "컨텍스트")

    assert result.steps[0].observation.startswith("Security Error")
    crawl.assert_not_awaited()
    assert result.final_answer == "답"


@pytest.mark.asyncio
async def test_missing_tool_executor_records_error():
    agent = ReActAgent(ircot_enabled=False)
    llm = AsyncMock(
        side_effect=[
            _reply({"thought": "t", "action": "query_data", "action_input": {}}),
            _reply({"thought": "t", "action": "final_answer", "action_input": {"answer": "답"}}),
            REFLECTION,
        ]
    )
    with patch("src.core.react_agent.acompletion", llm):
        result = await agent.run("질문", "컨텍스트")

    assert "실행기" in result.steps[0].observation
