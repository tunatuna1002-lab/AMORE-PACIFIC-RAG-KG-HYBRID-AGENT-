"""
ReAct Agent Multi-hop & IRCoT 단위 테스트
"""

import json
from unittest.mock import MagicMock

import pytest

from src.core.models import ToolResult
from src.core.react_agent import (
    ReActAgent,
    ReActResult,
)

# ============================================================
# Fixtures
# ============================================================


def _make_llm_response(thought, action, action_input=None):
    """LLM 응답 Mock 생성 헬퍼"""
    payload = {"thought": thought, "action": action}
    if action_input is not None:
        payload["action_input"] = action_input
    content = f"```json\n{json.dumps(payload, ensure_ascii=False)}\n```"
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = content
    return response


def _make_reflection_response(quality_score=0.8, needs_improvement=False):
    """Reflection 응답 Mock 생성 헬퍼"""
    payload = {
        "quality_score": quality_score,
        "needs_improvement": needs_improvement,
        "missing_info": [],
        "improvement_suggestion": "",
    }
    content = f"```json\n{json.dumps(payload)}\n```"
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].message.content = content
    return response


class MockToolExecutor:
    """Multi-hop 테스트용 도구 실행기"""

    def __init__(self, responses=None):
        self.call_log = []
        self.responses = responses or {}

    async def execute(self, tool_name: str, params: dict) -> ToolResult:
        self.call_log.append((tool_name, params))
        if tool_name in self.responses:
            return self.responses[tool_name]
        return ToolResult(
            tool_name=tool_name,
            success=True,
            data={"result": f"mock data for {tool_name}"},
        )


@pytest.fixture
def agent():
    """기본 ReActAgent"""
    return ReActAgent(max_iterations=5, max_hops=2, ircot_enabled=True)


@pytest.fixture
def agent_no_ircot():
    """IRCoT 비활성화된 Agent"""
    return ReActAgent(max_iterations=5, max_hops=2, ircot_enabled=False)


# ============================================================
# C-1: Multi-hop refine_search Tests
# ============================================================


@pytest.mark.asyncio
async def test_multihop_refine_search_basic(agent, monkeypatch):
    """refine_search action이 2차 검색을 트리거한다"""
    executor = MockToolExecutor(
        responses={
            "query_data": ToolResult(
                tool_name="query_data",
                success=True,
                data={"brand": "LANEIGE", "rank": 3},
            ),
        }
    )
    agent.set_tool_executor(executor)

    call_count = 0

    async def mock_acompletion(**kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return _make_llm_response(
                "LANEIGE 정보를 검색합니다",
                "refine_search",
                {
                    "refined_query": "LANEIGE lip care ranking",
                    "reason": "초기 검색 정제",
                    "focus_entities": ["LANEIGE"],
                },
            )
        elif call_count == 2:
            return _make_llm_response(
                "정보를 충분히 수집했습니다",
                "final_answer",
                {"answer": "LANEIGE는 3위입니다", "confidence": 0.9},
            )
        return _make_reflection_response()

    monkeypatch.setattr("src.core.react_agent.acompletion", mock_acompletion)

    result = await agent.run("LANEIGE 순위는?", "test context")
    assert result.hop_count == 1
    assert len(executor.call_log) >= 1
    # refine_search는 query_data로 변환되어 실행
    assert executor.call_log[0][0] == "query_data"


@pytest.mark.asyncio
async def test_multihop_max_hops_limit(agent, monkeypatch):
    """max_hops=2 제한이 작동한다"""
    executor = MockToolExecutor()
    agent.set_tool_executor(executor)
    agent.max_hops = 2

    call_count = 0

    async def mock_acompletion(**kwargs):
        nonlocal call_count
        call_count += 1
        if call_count <= 3:
            return _make_llm_response(
                f"검색 {call_count}",
                "refine_search",
                {
                    "refined_query": f"query {call_count}",
                    "reason": "추가 검색",
                    "focus_entities": [],
                },
            )
        elif call_count == 4:
            return _make_llm_response(
                "완료",
                "final_answer",
                {"answer": "결과", "confidence": 0.8},
            )
        return _make_reflection_response()

    monkeypatch.setattr("src.core.react_agent.acompletion", mock_acompletion)

    result = await agent.run("복잡한 질문", "context")
    assert result.hop_count <= 2


@pytest.mark.asyncio
async def test_multihop_combined_observations(agent, monkeypatch):
    """두 hop 결과가 결합된다"""
    call_idx = 0
    executor = MockToolExecutor(
        responses={
            "query_data": ToolResult(
                tool_name="query_data",
                success=True,
                data={"brand": "LANEIGE", "info": "lip care"},
            ),
        }
    )
    agent.set_tool_executor(executor)

    async def mock_acompletion(**kwargs):
        nonlocal call_idx
        call_idx += 1
        if call_idx == 1:
            return _make_llm_response(
                "먼저 기본 데이터를 수집",
                "query_data",
                {"brand": "LANEIGE"},
            )
        elif call_idx == 2:
            return _make_llm_response(
                "추가 정보 필요",
                "refine_search",
                {
                    "refined_query": "LANEIGE competitors",
                    "reason": "경쟁사 분석",
                    "focus_entities": ["LANEIGE", "Dior"],
                },
            )
        elif call_idx == 3:
            return _make_llm_response(
                "완료",
                "final_answer",
                {"answer": "결과 종합", "confidence": 0.9},
            )
        return _make_reflection_response()

    monkeypatch.setattr("src.core.react_agent.acompletion", mock_acompletion)

    result = await agent.run("LANEIGE vs 경쟁사", "context")
    # refine_search step의 observation에 이전 결과가 포함됨
    refine_step = None
    for step in result.steps:
        if step.action == "refine_search":
            refine_step = step
            break
    assert refine_step is not None
    assert refine_step.observation is not None
    assert "Hop" in refine_step.observation


# ============================================================
# C-2: IRCoT Tests
# ============================================================


def test_ircot_needs_retrieval_korean(agent):
    """한국어 키워드가 감지된다"""
    assert agent._needs_retrieval("정보 부족합니다. 추가 데이터가 필요합니다.")
    assert agent._needs_retrieval("이 부분은 확인 필요합니다")
    assert agent._needs_retrieval("추가 검색을 해야 합니다")
    assert agent._needs_retrieval("이 브랜드에 대해 더 알아봐야 합니다")


def test_ircot_needs_retrieval_english(agent):
    """영어 키워드가 감지된다"""
    assert agent._needs_retrieval("I need more information about this brand")
    assert agent._needs_retrieval("The data is insufficient for analysis")
    assert agent._needs_retrieval("This metric is unknown in the current context")


def test_ircot_no_retrieval_needed(agent):
    """일반 thought는 검색을 트리거하지 않는다"""
    assert not agent._needs_retrieval("LANEIGE는 Lip Care 카테고리에서 5위입니다")
    assert not agent._needs_retrieval("데이터 분석을 완료했습니다")
    assert not agent._needs_retrieval("최종 답변을 작성하겠습니다")


@pytest.mark.asyncio
async def test_ircot_auto_inject_search(agent, monkeypatch):
    """IRCoT가 자동으로 refine_search를 주입한다"""
    executor = MockToolExecutor()
    agent.set_tool_executor(executor)

    call_count = 0

    async def mock_acompletion(**kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            # thought에 '정보 부족' 포함, action은 query_data
            return _make_llm_response(
                "정보 부족 - LANEIGE 데이터가 필요합니다",
                "query_data",
                {"brand": "LANEIGE"},
            )
        elif call_count == 2:
            return _make_llm_response(
                "이제 충분한 정보가 있습니다",
                "final_answer",
                {"answer": "LANEIGE 분석 완료", "confidence": 0.9},
            )
        return _make_reflection_response()

    monkeypatch.setattr("src.core.react_agent.acompletion", mock_acompletion)

    result = await agent.run("LANEIGE 분석", "context")
    # IRCoT가 query_data를 refine_search로 바꿨으므로 hop_count >= 1
    assert result.hop_count >= 1
    # 첫 step의 action이 refine_search로 변경되었는지 확인
    assert result.steps[0].action == "refine_search"


@pytest.mark.asyncio
async def test_ircot_disabled(agent_no_ircot, monkeypatch):
    """ircot_enabled=False이면 자동 주입하지 않는다"""
    executor = MockToolExecutor()
    agent_no_ircot.set_tool_executor(executor)

    call_count = 0

    async def mock_acompletion(**kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return _make_llm_response(
                "정보 부족 - 데이터 필요",
                "query_data",
                {"brand": "LANEIGE"},
            )
        elif call_count == 2:
            return _make_llm_response(
                "완료",
                "final_answer",
                {"answer": "결과", "confidence": 0.8},
            )
        return _make_reflection_response()

    monkeypatch.setattr("src.core.react_agent.acompletion", mock_acompletion)

    result = await agent_no_ircot.run("LANEIGE 분석", "context")
    # IRCoT 비활성이므로 hop_count == 0, action 변경 없음
    assert result.hop_count == 0
    assert result.steps[0].action == "query_data"


@pytest.mark.asyncio
async def test_hop_count_in_result(agent, monkeypatch):
    """ReActResult.hop_count가 정확히 추적된다"""
    executor = MockToolExecutor()
    agent.set_tool_executor(executor)

    call_count = 0

    async def mock_acompletion(**kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return _make_llm_response(
                "1차 검색",
                "refine_search",
                {"refined_query": "query1", "reason": "r", "focus_entities": []},
            )
        elif call_count == 2:
            return _make_llm_response(
                "2차 검색",
                "refine_search",
                {"refined_query": "query2", "reason": "r", "focus_entities": []},
            )
        elif call_count == 3:
            return _make_llm_response(
                "완료",
                "final_answer",
                {"answer": "done", "confidence": 0.9},
            )
        return _make_reflection_response()

    monkeypatch.setattr("src.core.react_agent.acompletion", mock_acompletion)

    result = await agent.run("복잡한 질문", "context")
    assert result.hop_count == 2


def test_refine_search_action_validation():
    """refine_search 파라미터가 올바르게 검증된다"""
    from src.core.react_agent import validate_action

    # 유효한 파라미터
    is_valid, err = validate_action(
        "refine_search",
        {"refined_query": "test", "reason": "reason", "focus_entities": ["A"]},
    )
    assert is_valid is True

    # 잘못된 타입
    is_valid, err = validate_action(
        "refine_search",
        {"refined_query": 123},  # str이어야 함
    )
    assert is_valid is False
    assert "must be str" in err


@pytest.mark.asyncio
async def test_multihop_with_tool_executor(agent, monkeypatch):
    """Mock executor와 multi-hop 통합 테스트"""
    responses = {
        "query_data": ToolResult(
            tool_name="query_data",
            success=True,
            data={"products": ["A", "B", "C"]},
        ),
    }
    executor = MockToolExecutor(responses=responses)
    agent.set_tool_executor(executor)

    call_count = 0

    async def mock_acompletion(**kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return _make_llm_response(
                "첫 검색",
                "refine_search",
                {
                    "refined_query": "lip care products",
                    "reason": "초기 검색",
                    "focus_entities": ["LANEIGE"],
                },
            )
        elif call_count == 2:
            return _make_llm_response(
                "완료",
                "final_answer",
                {"answer": "A, B, C 제품", "confidence": 0.9},
            )
        return _make_reflection_response()

    monkeypatch.setattr("src.core.react_agent.acompletion", mock_acompletion)

    result = await agent.run("제품 목록", "context")
    assert result.hop_count == 1
    # executor가 query_data로 호출됨
    assert len(executor.call_log) == 1
    assert executor.call_log[0][0] == "query_data"
    assert "LANEIGE" in executor.call_log[0][1].get("brand", "")


@pytest.mark.asyncio
async def test_ircot_with_final_answer(agent, monkeypatch):
    """final_answer는 IRCoT 키워드가 있어도 중단한다"""
    executor = MockToolExecutor()
    agent.set_tool_executor(executor)

    call_count = 0

    async def mock_acompletion(**kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return _make_llm_response(
                "정보 부족하지만 final answer",
                "final_answer",
                {"answer": "답변", "confidence": 0.7},
            )
        return _make_reflection_response()

    monkeypatch.setattr("src.core.react_agent.acompletion", mock_acompletion)

    result = await agent.run("질문", "context")
    # final_answer이므로 IRCoT가 작동하지 않음 (final_answer 체크가 먼저)
    assert result.hop_count == 0
    assert result.iterations == 1


@pytest.mark.asyncio
async def test_multihop_empty_observation(agent, monkeypatch):
    """빈 검색 결과 처리"""
    executor = MockToolExecutor(
        responses={
            "query_data": ToolResult(
                tool_name="query_data",
                success=True,
                data={},
            ),
        }
    )
    agent.set_tool_executor(executor)

    call_count = 0

    async def mock_acompletion(**kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return _make_llm_response(
                "검색",
                "refine_search",
                {"refined_query": "empty query", "reason": "r", "focus_entities": []},
            )
        elif call_count == 2:
            return _make_llm_response(
                "완료",
                "final_answer",
                {"answer": "데이터 없음", "confidence": 0.5},
            )
        return _make_reflection_response()

    monkeypatch.setattr("src.core.react_agent.acompletion", mock_acompletion)

    result = await agent.run("없는 데이터", "context")
    assert result.hop_count == 1
    # 빈 결과여도 에러 없이 처리
    assert result.steps[0].observation is not None


@pytest.mark.asyncio
async def test_max_iterations_with_multihop(monkeypatch):
    """max_iterations가 multi-hop에서도 존중된다"""
    agent = ReActAgent(max_iterations=2, max_hops=5)
    executor = MockToolExecutor()
    agent.set_tool_executor(executor)

    call_count = 0

    async def mock_acompletion(**kwargs):
        nonlocal call_count
        call_count += 1
        # 항상 refine_search만 반환 (final_answer 안 함)
        return _make_llm_response(
            f"검색 {call_count}",
            "refine_search",
            {"refined_query": f"q{call_count}", "reason": "r", "focus_entities": []},
        )

    monkeypatch.setattr("src.core.react_agent.acompletion", mock_acompletion)

    result = await agent.run("무한 루프 방지", "context")
    # max_iterations=2이므로 2번만 실행
    assert result.iterations == 2
    assert result.hop_count <= 2


@pytest.mark.asyncio
async def test_refine_search_focus_entities(agent, monkeypatch):
    """focus_entities가 올바르게 전달된다"""
    executor = MockToolExecutor()
    agent.set_tool_executor(executor)

    call_count = 0

    async def mock_acompletion(**kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return _make_llm_response(
                "엔티티 기반 검색",
                "refine_search",
                {
                    "refined_query": "lip care comparison",
                    "reason": "엔티티 기반 정제",
                    "focus_entities": ["LANEIGE", "Dior", "Chanel"],
                },
            )
        elif call_count == 2:
            return _make_llm_response(
                "완료",
                "final_answer",
                {"answer": "비교 결과", "confidence": 0.9},
            )
        return _make_reflection_response()

    monkeypatch.setattr("src.core.react_agent.acompletion", mock_acompletion)

    result = await agent.run("브랜드 비교", "context")
    # executor에 brand 파라미터가 전달됨
    assert len(executor.call_log) == 1
    params = executor.call_log[0][1]
    assert "brand" in params
    assert "LANEIGE" in params["brand"]
    assert "Dior" in params["brand"]
    assert "Chanel" in params["brand"]


def test_react_result_hop_count_default():
    """ReActResult.hop_count 기본값은 0"""
    result = ReActResult(final_answer="test")
    assert result.hop_count == 0


def test_react_agent_init_defaults():
    """ReActAgent 초기화 기본값 확인"""
    agent = ReActAgent()
    assert agent.max_hops == 2
    assert agent.ircot_enabled is True
    assert agent.max_iterations == 5


def test_react_agent_init_custom():
    """ReActAgent 커스텀 초기화"""
    agent = ReActAgent(max_hops=3, ircot_enabled=False, max_iterations=10)
    assert agent.max_hops == 3
    assert agent.ircot_enabled is False
    assert agent.max_iterations == 10
