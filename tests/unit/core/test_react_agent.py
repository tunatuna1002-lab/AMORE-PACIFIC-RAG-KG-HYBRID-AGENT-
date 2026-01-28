"""
ReAct Agent 단위 테스트
"""

import pytest

from src.core.models import ToolResult
from src.core.react_agent import (
    ALLOWED_ACTIONS,
    ReActAgent,
    ReActResult,
    ReActStep,
    validate_action,
)

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
async def test_react_run(react_agent):
    """ReAct 실행 테스트 (통합)"""
    result = await react_agent.run(query="LANEIGE 순위는?", context="최근 데이터: 없음")

    assert isinstance(result, ReActResult)
    assert result.iterations <= 2
    assert len(result.steps) > 0
    assert isinstance(result.confidence, float)
    assert 0.0 <= result.confidence <= 1.0


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
