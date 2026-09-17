"""
ReAct Agent 단위 테스트
"""

import json
from unittest.mock import MagicMock

import pytest

from src.core.models import ToolResult
from src.core.react_agent import (
    ALLOWED_ACTIONS,
    ReActAgent,
    ReActResult,
    ReActStep,
    validate_action,
)


def _make_llm_response(thought: str, action: str | None = None, action_input: dict | None = None):
    """LLM 응답 Mock 생성 헬퍼 (실제 네트워크 호출 없이 _parse_step의 JSON 파싱 흐름을 그대로 검증)"""
    payload: dict = {"thought": thought}
    if action is not None:
        payload["action"] = action
    if action_input is not None:
        payload["action_input"] = action_input
    content = f"```json\n{json.dumps(payload, ensure_ascii=False)}\n```"
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = content
    return response


def _make_reflection_response(quality_score: float = 0.85, needs_improvement: bool = False):
    """Self-Reflection 응답 Mock 생성 헬퍼"""
    payload = {
        "quality_score": quality_score,
        "missing_info": [],
        "needs_improvement": needs_improvement,
        "improvement_suggestion": "",
    }
    content = f"```json\n{json.dumps(payload)}\n```"
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = content
    return response


# ============================================================
# Security Tests (P0)
# ============================================================


def test_allowed_actions_is_frozen():
    """ALLOWED_ACTIONS가 불변 frozenset인지 확인"""
    assert isinstance(ALLOWED_ACTIONS, frozenset)
    assert "query_data" in ALLOWED_ACTIONS
    assert "query_knowledge_graph" in ALLOWED_ACTIONS
    assert "calculate_metrics" in ALLOWED_ACTIONS
    assert "final_answer" in ALLOWED_ACTIONS


def test_validate_action_allowed():
    """허용된 액션 검증"""
    is_valid, error = validate_action("query_data", {"brand": "LANEIGE"})
    assert is_valid is True
    assert error == ""


def test_validate_action_not_allowed():
    """허용되지 않은 액션 검증"""
    is_valid, error = validate_action("dangerous_action", {})
    assert is_valid is False
    assert "not allowed" in error
    assert "dangerous_action" in error


def test_validate_action_input_type_check():
    """파라미터 타입 검증"""
    # limit은 int여야 함
    is_valid, error = validate_action("query_data", {"limit": "not_a_number"})
    assert is_valid is False
    assert "must be int" in error


def test_validate_action_none_input():
    """action_input이 None인 경우"""
    is_valid, error = validate_action("final_answer", None)
    assert is_valid is True
    assert error == ""


def test_validate_action_empty_input():
    """action_input이 빈 dict인 경우"""
    is_valid, error = validate_action("query_data", {})
    assert is_valid is True
    assert error == ""


class MockToolExecutor:
    """테스트용 도구 실행기"""

    async def execute(self, tool_name: str, params: dict) -> ToolResult:
        """Mock 도구 실행"""
        if tool_name == "query_data":
            return ToolResult(
                tool_name=tool_name, success=True, data={"brand": "LANEIGE", "rank": 5}
            )
        elif tool_name == "final_answer":
            return ToolResult(
                tool_name=tool_name, success=True, data={"answer": "LANEIGE는 5위입니다."}
            )
        return ToolResult(tool_name=tool_name, success=False, error="Unknown tool")


@pytest.fixture
def react_agent():
    """ReAct Agent 픽스처"""
    agent = ReActAgent(max_iterations=2)
    agent.set_tool_executor(MockToolExecutor())
    return agent


@pytest.mark.asyncio
async def test_react_step_parsing():
    """ReAct Step 파싱 테스트"""
    agent = ReActAgent()

    # Valid JSON
    content = """```json
{
    "thought": "현재 상황을 분석합니다",
    "action": "query_data",
    "action_input": {"brand": "LANEIGE"}
}
```"""

    step = agent._parse_step(content)
    assert step.thought == "현재 상황을 분석합니다"
    assert step.action == "query_data"
    assert step.action_input == {"brand": "LANEIGE"}


@pytest.mark.asyncio
async def test_react_step_invalid_json():
    """잘못된 JSON 처리 테스트"""
    agent = ReActAgent()

    content = "Just plain text without JSON"
    step = agent._parse_step(content)
    assert step.thought == content
    assert step.action is None


@pytest.mark.asyncio
async def test_format_steps():
    """Step 포맷팅 테스트"""
    agent = ReActAgent()

    steps = [
        ReActStep(thought="첫 번째 생각", action="query_data", observation="결과: LANEIGE 5위"),
        ReActStep(thought="두 번째 생각", action="final_answer"),
    ]

    formatted = agent._format_steps(steps)
    assert "Step 1" in formatted
    assert "Step 2" in formatted
    assert "첫 번째 생각" in formatted
    assert "query_data" in formatted


@pytest.mark.asyncio
async def test_react_run(react_agent, monkeypatch):
    """ReAct 실행 테스트 (통합)

    F15: acompletion(LLM 호출)만 가짜로 두고, ReAct 루프 자체(파싱 -> 액션 검증 ->
    도구 실행 -> 종료 조건 -> self-reflection)는 실제 코드 경로로 검증한다.
    이 테스트는 외부로 나가는 네트워크 호출을 만들지 않는다.
    """
    call_count = 0

    async def mock_acompletion(**kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # 1st step: 실제 도구 실행(query_data)을 유도
            return _make_llm_response(
                "LANEIGE 순위 데이터를 조회합니다",
                "query_data",
                {"brand": "LANEIGE"},
            )
        elif call_count == 2:
            # 2nd step: 루프 종료 조건(final_answer, action_input.answer)을 유도
            return _make_llm_response(
                "충분한 정보를 얻었으므로 최종 답변합니다",
                "final_answer",
                {"answer": "LANEIGE는 5위입니다", "confidence": 0.9},
            )
        # 3rd call: self-reflection (final_answer가 채워졌으므로 _force_final_answer는
        # 호출되지 않아야 함)
        return _make_reflection_response(quality_score=0.85, needs_improvement=False)

    monkeypatch.setattr("src.core.react_agent.acompletion", mock_acompletion)

    result = await react_agent.run(query="LANEIGE 순위는?", context="최근 데이터: 없음")

    assert isinstance(result, ReActResult)
    assert result.iterations <= 2
    assert len(result.steps) > 0
    assert isinstance(result.confidence, float)
    assert 0.0 <= result.confidence <= 1.0

    # LLM 호출 횟수 검증: step 1회 + step 2회 + reflection 1회 = 3회 (네트워크 호출은 전부 가짜)
    assert call_count == 3

    # 실제 ReAct 루프 로직 검증: 1단계는 query_data 액션이 검증(validate_action)을 통과해
    # MockToolExecutor를 거쳐 실행되고, 그 결과가 observation에 실제로 반영되어야 한다.
    assert result.steps[0].action == "query_data"
    assert result.steps[0].observation is not None
    assert "LANEIGE" in result.steps[0].observation

    # 2단계는 final_answer로 루프를 종료시켜야 하고, 최종 답은 action_input.answer에서
    # 와야 한다 (run()이 결과를 observation이 아닌 action_input에서 읽도록 고쳐졌음).
    assert result.steps[1].action == "final_answer"
    assert result.final_answer == "LANEIGE는 5위입니다"

    # self-reflection 결과(quality_score)가 실제로 confidence에 반영되어야 한다.
    assert result.confidence == 0.85
    assert result.needs_improvement is False


def test_react_result_dataclass():
    """ReActResult 데이터클래스 테스트"""
    result = ReActResult(
        final_answer="답변입니다",
        steps=[ReActStep(thought="생각")],
        iterations=1,
        confidence=0.8,
        needs_improvement=False,
    )

    assert result.final_answer == "답변입니다"
    assert len(result.steps) == 1
    assert result.iterations == 1
    assert result.confidence == 0.8
    assert not result.needs_improvement
