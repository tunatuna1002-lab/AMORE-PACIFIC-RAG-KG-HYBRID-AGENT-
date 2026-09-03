"""
Core 모듈 단위 테스트
====================
LLM Orchestrator 핵심 컴포넌트 테스트
"""

import pytest

from src.core.cache import ResponseCache
from src.core.confidence import ConfidenceAssessor

# 테스트 대상 모듈
from src.core.models import (
    ConfidenceLevel,
    Context,
    Decision,
    KGFact,
    Response,
    SystemState,
    ToolResult,
)
from src.core.state import OrchestratorState
from src.core.tools import AGENT_TOOLS, ToolExecutor, get_all_tool_schemas


class TestModels:
    """데이터 모델 테스트"""

    def test_confidence_level(self):
        """신뢰도 레벨 테스트"""
        assert ConfidenceLevel.HIGH.value == "high"
        assert ConfidenceLevel.MEDIUM.value == "medium"
        assert ConfidenceLevel.LOW.value == "low"
        assert ConfidenceLevel.UNKNOWN.value == "unknown"

    def test_context_creation(self):
        """Context 생성 테스트"""
        context = Context(query="라네즈 현재 순위 알려줘", entities={"brands": ["laneige"]})

        assert context.query == "라네즈 현재 순위 알려줘"
        assert "laneige" in context.entities["brands"]
        assert context.gathered_at is not None

    def test_context_has_sufficient_context(self):
        """컨텍스트 충분성 확인"""
        empty = Context(query="test")
        assert not empty.has_sufficient_context()

        with_docs = Context(query="test", rag_docs=[{"content": "test"}])
        assert with_docs.has_sufficient_context()

    def test_response_factory_methods(self):
        """Response 팩토리 메서드 테스트"""
        clarification = Response.clarification(
            "더 구체적으로 질문해주세요", ["SoS가 뭔가요?", "현재 순위 알려주세요"]
        )
        assert clarification.is_clarification
        assert clarification.query_type == "clarification"

        fallback = Response.fallback("오류 발생")
        assert fallback.is_fallback
        assert "다시 질문" in fallback.suggestions[0]

    def test_decision_requires_tool(self):
        """Decision 도구 필요 여부"""
        direct = Decision(tool="direct_answer")
        assert not direct.requires_tool()

        crawl = Decision(tool="crawl_amazon")
        assert crawl.requires_tool()

    def test_tool_result_summary(self):
        """ToolResult 요약 생성"""
        result = ToolResult(
            tool_name="crawl_amazon", success=True, data={"total_products": 100, "laneige_count": 5}
        )
        summary = result.to_summary()
        assert "100" in summary
        assert "LANEIGE" in summary


class TestConfidenceAssessor:
    """신뢰도 평가 테스트"""

    def test_high_confidence(self):
        """높은 신뢰도 테스트"""
        assessor = ConfidenceAssessor()

        route_result = {"query_type": "definition", "confidence": 0.9}
        context = Context(
            query="SoS가 뭐야?",
            rag_docs=[{"content": "1"}, {"content": "2"}, {"content": "3"}],
            kg_inferences=[{"insight": "test"}],
        )

        level = assessor.assess(route_result, context)
        assert level == ConfidenceLevel.HIGH

    def test_unknown_confidence(self):
        """낮은 신뢰도 테스트"""
        assessor = ConfidenceAssessor()

        route_result = {"query_type": "unknown", "confidence": 0.1}
        context = Context(query="???")

        level = assessor.assess(route_result, context)
        assert level == ConfidenceLevel.UNKNOWN

    def test_should_skip_llm(self):
        """LLM 스킵 조건 테스트"""
        assessor = ConfidenceAssessor()
        assert assessor.should_skip_llm_decision(ConfidenceLevel.HIGH)
        assert not assessor.should_skip_llm_decision(ConfidenceLevel.MEDIUM)


class TestResponseCache:
    """캐시 테스트"""

    def test_cache_set_get(self):
        """캐시 저장/조회"""
        cache = ResponseCache()

        response = Response(text="테스트 응답")
        cache.set("test query", response, "query")

        cached = cache.get("test query", "query")
        assert cached is not None
        assert cached.text == "테스트 응답"

    def test_cache_miss(self):
        """캐시 미스"""
        cache = ResponseCache()
        result = cache.get("nonexistent", "query")
        assert result is None

    def test_cache_invalidation(self):
        """캐시 무효화"""
        cache = ResponseCache()

        response = Response(text="test")
        cache.set("key1", response, "query")
        cache.set("key2", {"data": 1}, "kg")

        # 특정 타입 무효화
        cache.invalidate(cache_type="query")

        assert cache.get("key1", "query") is None
        assert cache.get("key2", "kg") is not None


class TestOrchestratorState:
    """상태 관리 테스트"""

    def test_crawl_needed(self, tmp_path):
        """크롤링 필요 여부"""
        # 임시 경로로 격리하여 기존 상태 파일 영향 제거
        state = OrchestratorState(_persist_path=tmp_path / "state.json")

        # 초기 상태: 크롤링 필요
        assert state.is_crawl_needed()

        # 크롤링 완료 표시
        state.mark_crawled(100)
        assert not state.is_crawl_needed()
        assert state.data_freshness == "fresh"

    def test_tool_tracking(self, tmp_path):
        """도구 실행 추적"""
        state = OrchestratorState(_persist_path=tmp_path / "state.json")

        state.start_tool("crawl_amazon")
        assert state.is_tool_running("crawl_amazon")
        assert state.has_active_tools()

        state.end_tool("crawl_amazon")
        assert not state.is_tool_running("crawl_amazon")
        assert not state.has_active_tools()

    def test_context_summary(self, tmp_path):
        """컨텍스트 요약"""
        state = OrchestratorState(_persist_path=tmp_path / "state.json")
        state.mark_crawled(50)
        state.mark_kg_initialized(1000)

        summary = state.to_context_summary()
        assert "fresh" in summary
        assert "1000" in summary


class TestTools:
    """도구 정의 테스트"""

    def test_tool_schema(self):
        """OpenAI 스키마 변환"""
        tool = AGENT_TOOLS["crawl_amazon"]
        schema = tool.to_openai_schema()

        assert schema["type"] == "function"
        assert schema["function"]["name"] == "crawl_amazon"
        assert "parameters" in schema["function"]

    def test_all_tools_have_schema(self):
        """모든 도구 스키마 테스트"""
        schemas = get_all_tool_schemas()
        assert len(schemas) >= 4  # 최소 4개 도구

        for schema in schemas:
            assert "function" in schema
            assert "name" in schema["function"]


class TestToolExecutor:
    """도구 실행기 테스트"""

    @pytest.mark.asyncio
    async def test_direct_answer(self):
        """직접 응답 도구"""
        executor = ToolExecutor()

        result = await executor.execute("direct_answer", {"reason": "test"})
        assert result.success
        assert result.tool_name == "direct_answer"

    @pytest.mark.asyncio
    async def test_unknown_tool(self):
        """미등록 도구"""
        executor = ToolExecutor()

        result = await executor.execute("unknown_tool", {})
        assert not result.success
        assert "연결된 실행기가 없습니다" in result.error

    def test_tool_availability(self):
        """도구 사용 가능 여부"""
        executor = ToolExecutor()

        # direct_answer는 항상 사용 가능
        assert executor.is_tool_available("direct_answer")

        # 미등록 도구는 불가
        assert not executor.is_tool_available("unregistered")


# 통합 테스트
class TestIntegration:
    """통합 테스트"""

    def test_full_flow_simulation(self, tmp_path):
        """전체 흐름 시뮬레이션"""
        # 1. 상태 초기화 (임시 경로로 격리)
        state = OrchestratorState(_persist_path=tmp_path / "state.json")
        assert state.is_crawl_needed()

        # 2. 크롤링 완료
        state.mark_crawled(100)
        assert state.data_freshness == "fresh"

        # 3. KG 초기화
        state.mark_kg_initialized(500)
        assert state.kg_initialized

        # 4. 컨텍스트 생성
        context = Context(
            query="라네즈 분석해줘",
            entities={"brands": ["laneige"]},
            kg_facts=[KGFact(fact_type="brand_info", entity="laneige", data={"sos": 0.15})],
            system_state=SystemState(
                data_freshness="fresh", kg_initialized=True, kg_triple_count=500
            ),
        )

        # 5. 신뢰도 평가
        assessor = ConfidenceAssessor()
        level = assessor.assess({"query_type": "analysis", "confidence": 0.7}, context)

        # 컨텍스트 있으므로 최소 MEDIUM
        assert level in [ConfidenceLevel.HIGH, ConfidenceLevel.MEDIUM]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
