"""
E2E Type Flow Verification Tests
=================================
Verifies that the Decision/Context/Response type contracts
are maintained throughout the processing pipeline.

Phase 1-C: These tests catch the V3→V4 migration bugs where
dict was used instead of Decision dataclass.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.core.models import Context, Decision, Response


class TestDecisionMakerTypeContract:
    """DecisionMaker가 항상 Decision 객체를 반환하는지 검증"""

    def test_decision_dataclass_attributes(self):
        """Decision 객체가 필수 속성을 가지는지"""
        d = Decision(
            tool="direct_answer",
            tool_params={},
            reason="test",
            key_points=["a"],
            confidence=0.8,
        )
        assert isinstance(d.tool, str)
        assert isinstance(d.tool_params, dict)
        assert isinstance(d.reason, str)
        assert isinstance(d.key_points, list)
        assert isinstance(d.confidence, float)

    def test_decision_requires_tool(self):
        """requires_tool() 메서드 동작 검증"""
        d1 = Decision(tool="direct_answer")
        assert not d1.requires_tool()

        d2 = Decision(tool="calculate_metrics")
        assert d2.requires_tool()

        d3 = Decision(tool=None)
        assert not d3.requires_tool()

    def test_decision_default_values(self):
        """Decision 기본값 검증"""
        d = Decision()
        assert d.tool is None
        assert d.tool_params == {}
        assert d.reason == ""
        assert d.key_points == []
        assert d.confidence == 0.0


class TestDecisionMakerReturnsDecision:
    """DecisionMaker.decide()가 Decision을 반환하는지 검증"""

    @pytest.mark.asyncio
    async def test_decide_returns_decision_object(self):
        """decide()의 반환 타입이 Decision인지"""
        from src.core.decision_maker import DecisionMaker

        dm = DecisionMaker()

        # Mock LLM response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[
            0
        ].message.content = '{"tool": "direct_answer", "reason": "test", "confidence": 0.8}'

        with patch(
            "src.core.decision_maker.acompletion",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            context = Context(query="test query")
            system_state = {"mode": "normal", "data_status": "fresh", "available_tools": []}

            result = await dm.decide("test query", context, system_state)

            assert isinstance(result, Decision), f"Expected Decision, got {type(result)}"
            assert hasattr(result, "tool")
            assert hasattr(result, "confidence")
            assert hasattr(result, "key_points")

    @pytest.mark.asyncio
    async def test_fallback_returns_decision(self):
        """LLM 실패 시에도 Decision을 반환하는지"""
        from src.core.decision_maker import DecisionMaker

        dm = DecisionMaker()

        with patch(
            "src.core.decision_maker.acompletion",
            new_callable=AsyncMock,
            side_effect=Exception("API Error"),
        ):
            context = Context(query="test")
            system_state = {"mode": "normal", "data_status": "fresh", "available_tools": []}

            result = await dm.decide("test", context, system_state)

            assert isinstance(
                result, Decision
            ), f"Fallback should return Decision, got {type(result)}"
            assert result.tool == "direct_answer"
            assert result.confidence < 0.5  # Low confidence for fallback

    @pytest.mark.asyncio
    async def test_parse_failure_returns_decision(self):
        """JSON 파싱 실패 시에도 Decision을 반환하는지"""
        from src.core.decision_maker import DecisionMaker

        dm = DecisionMaker()

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "This is not valid JSON at all"

        with patch(
            "src.core.decision_maker.acompletion",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            context = Context(query="test")
            system_state = {"mode": "normal", "data_status": "fresh", "available_tools": []}

            result = await dm.decide("test", context, system_state)

            assert isinstance(result, Decision)
            assert result.tool == "direct_answer"


class TestContextTypeContract:
    """Context 객체 타입 계약 검증"""

    def test_context_base_attributes(self):
        """Context가 필수 속성을 가지는지"""
        ctx = Context(query="test query", entities={"brands": ["LANEIGE"]})
        assert isinstance(ctx.query, str)
        assert isinstance(ctx.entities, dict)
        assert isinstance(ctx.rag_docs, list)
        assert isinstance(ctx.kg_facts, list)
        assert isinstance(ctx.kg_inferences, list)

    def test_context_summary_type(self):
        """summary는 str 또는 None"""
        ctx = Context(query="test")
        assert ctx.summary is None or isinstance(ctx.summary, str)


class TestResponseTypeContract:
    """Response 객체 타입 계약 검증"""

    def test_response_has_text_not_content(self):
        """Response는 .text를 사용 (.content가 아님)"""
        r = Response(text="Hello")
        assert hasattr(r, "text")
        assert r.text == "Hello"
        # content는 없어야 함 (V3 호환성 문제)
        # Note: content가 있어도 text가 primary

    def test_response_has_confidence_score(self):
        """Response는 .confidence_score를 사용 (.confidence가 아님)"""
        r = Response(text="test", confidence_score=0.9)
        assert hasattr(r, "confidence_score")
        assert r.confidence_score == 0.9

    def test_response_has_tools_called(self):
        """Response는 .tools_called를 사용"""
        r = Response(text="test", tools_called=["crawl"])
        assert hasattr(r, "tools_called")
        assert r.tools_called == ["crawl"]

    def test_response_fallback_method(self):
        """Response.fallback() 클래스 메서드 검증"""
        r = Response.fallback("Error message")
        assert isinstance(r, Response)
        assert r.is_fallback is True
        assert "Error" in r.text


class TestBrainDecisionFlow:
    """Brain의 decision 흐름에서 속성 접근 패턴 검증"""

    def test_decision_attribute_access_pattern(self):
        """Brain이 사용하는 decision.tool, decision.tool_params 패턴 검증"""
        d = Decision(
            tool="calculate_metrics",
            tool_params={"metric": "sos", "brand": "LANEIGE"},
            reason="SoS 분석 필요",
            key_points=["LANEIGE SoS 확인"],
            confidence=0.85,
        )

        # Brain의 process_query에서 사용하는 패턴들
        assert d.tool == "calculate_metrics"
        assert d.tool != "direct_answer"
        assert d.tool_params["metric"] == "sos"
        assert d.confidence == 0.85

    def test_decision_for_direct_answer(self):
        """direct_answer Decision 패턴"""
        d = Decision(tool="direct_answer", confidence=0.9)
        assert d.tool == "direct_answer"
        assert not d.requires_tool()


class TestDashboardApiFieldMapping:
    """dashboard_api.py의 Response→BrainChatResponse 매핑 검증"""

    def test_response_to_dashboard_mapping(self):
        """Response 필드가 dashboard API에서 올바르게 매핑되는지"""
        r = Response(
            text="분석 결과입니다",
            query_type="market_analysis",
            confidence_score=0.85,
            sources=["data_source_1"],
            tools_called=["calculate_metrics"],
            suggestions=["LANEIGE SoS 추이를 확인해보세요"],
        )

        # dashboard_api.py에서 사용하는 필드 매핑
        assert r.text is not None  # NOT r.content
        assert r.confidence_score is not None  # NOT r.confidence
        assert isinstance(r.sources, list)
        assert r.query_type is not None  # reasoning으로 매핑됨
        assert isinstance(r.tools_called, list)
        assert isinstance(r.suggestions, list)
