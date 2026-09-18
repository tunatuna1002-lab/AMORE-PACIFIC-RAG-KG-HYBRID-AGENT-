"""
DecisionMaker 단위 테스트

도구 선택은 네이티브 function calling이다 (트랙 4-A). 예전 테스트는 LLM이 낸 JSON 문자열을
파싱하는 동작(_parse_decision)을 검증했는데 그 경로 자체가 사라졌다 — 스키마·인자는 API가
주고, 파싱 실패 폴백은 tests/unit/core/test_decision_function_calling.py가 본다.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.core.decision_maker import DecisionMaker
from src.core.models import Context, Decision, SystemState


@pytest.fixture
def decision_maker():
    return DecisionMaker(model="gpt-4o-mini", temperature=0.1, max_tokens=500)


@pytest.fixture
def mock_context():
    return Context(
        query="LANEIGE 현재 순위 알려줘",
        entities={"brands": ["LANEIGE"], "categories": ["lip_care"]},
        rag_docs=[],
        kg_facts=[],
        kg_inferences=[],
        system_state=SystemState(
            data_freshness="fresh",
            kg_initialized=True,
            kg_triple_count=100,
        ),
        summary="LANEIGE는 Lip Care 카테고리에서 활동하는 K-Beauty 브랜드입니다.",
    )


@pytest.fixture
def mock_system_state():
    return {
        "data_status": "fresh",
        "mode": "autonomous",
        "available_tools": ["get_metrics", "kg_neighbors", "search_docs"],
        "failed_tools": [],
    }


class TestDecisionMaker:
    """DecisionMaker 클래스 테스트"""

    def test_init(self, decision_maker):
        """초기화 테스트"""
        assert decision_maker.model == "gpt-4o-mini"
        assert decision_maker.temperature == 0.1
        assert decision_maker.max_tokens == 500
        assert decision_maker._decision_count == 0

    @pytest.mark.asyncio
    async def test_decide_success(self, decision_maker, mock_context, mock_system_state):
        """도구 호출이 없으면 direct_answer"""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "컨텍스트 충분\n- LANEIGE 순위 정보 존재"
        mock_response.choices[0].message.tool_calls = None

        with patch("src.core.decision_maker.acompletion", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = mock_response
            result = await decision_maker.decide(
                "LANEIGE 순위", mock_context, mock_system_state, "high"
            )

        assert isinstance(result, Decision)
        assert result.tool == "direct_answer"
        assert result.key_points == ["LANEIGE 순위 정보 존재"]
        assert decision_maker._decision_count == 1

    @pytest.mark.asyncio
    async def test_decide_with_tool(self, decision_maker, mock_context, mock_system_state):
        """도구 선택 의사결정 (function calling)"""
        call = MagicMock()
        call.function.name = "get_metrics"
        call.function.arguments = '{"brand": "LANEIGE", "category": "lip_care"}'
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "DB 수치가 필요하다"
        mock_response.choices[0].message.tool_calls = [call]

        with patch("src.core.decision_maker.acompletion", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = mock_response
            result = await decision_maker.decide(
                "LANEIGE 점유율", mock_context, mock_system_state, "low"
            )

        assert result.tool == "get_metrics"
        assert result.tool_params == {"brand": "LANEIGE", "category": "lip_care"}
        assert result.requires_tool() is True

    @pytest.mark.asyncio
    async def test_decide_llm_failure_fallback(
        self, decision_maker, mock_context, mock_system_state
    ):
        """LLM 실패 시 폴백"""
        with patch("src.core.decision_maker.acompletion", new_callable=AsyncMock) as mock_llm:
            mock_llm.side_effect = Exception("API Error")
            result = await decision_maker.decide("테스트", mock_context, mock_system_state)

        assert result.tool == "direct_answer"
        assert result.confidence == 0.3
        assert "LLM 오류" in result.reason

    def test_fallback_decision(self, decision_maker):
        """폴백은 direct_answer + 사유 기록"""
        result = decision_maker._fallback_decision("파싱 실패")
        assert result.tool == "direct_answer"
        assert result.confidence == 0.3
        assert "파싱 실패" in result.reason

    def test_format_system_state(self, decision_maker, mock_system_state):
        """시스템 상태 포맷"""
        formatted = decision_maker._format_system_state(mock_system_state)
        assert "fresh" in formatted
        assert "get_metrics" in formatted

    def test_get_stats(self, decision_maker):
        """통계"""
        stats = decision_maker.get_stats()
        assert stats["decision_count"] == 0
        assert stats["model"] == "gpt-4o-mini"

    def test_mode_prompts(self, decision_maker):
        """모드별 프롬프트 존재 확인"""
        assert "high" in DecisionMaker.MODE_PROMPTS
        assert "medium" in DecisionMaker.MODE_PROMPTS
        assert "low" in DecisionMaker.MODE_PROMPTS
        assert "unknown" in DecisionMaker.MODE_PROMPTS
