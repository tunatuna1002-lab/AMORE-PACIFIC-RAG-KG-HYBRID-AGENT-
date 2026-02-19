"""
LLMOrchestrator 단위 테스트
============================
Coverage Target: 17.41% → 50%+

테스트 범위:
- __init__ (초기화)
- set_client, set_router, set_retriever (설정)
- initialize (비동기 초기화)
- process (메인 처리 메서드)
- _rule_based_routing (Rule 기반 라우팅)
- _route_by_confidence (신뢰도 기반 분기)
- _llm_decision_flow (LLM 판단 흐름)
- _get_llm_decision (LLM 판단 요청)
- _format_tools_description (도구 설명 포맷팅)
- _parse_decision_response (LLM 응답 파싱)
- _default_decision (기본 판단)
- _generate_direct_response (직접 응답 생성)
- _generate_rag_based_response (RAG 기반 응답 생성)
- _infer_query_type_from_query (질문 유형 추론)
- _generate_clarification (명확화 요청)
- get_stats, reset_stats, get_state_summary (상태/통계)
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.core.llm_orchestrator import LLMOrchestrator
from src.core.models import ConfidenceLevel, Context, Decision, Response, SystemState

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_openai_client():
    """Mock OpenAI 클라이언트"""
    client = MagicMock()
    client.chat = MagicMock()
    client.chat.completions = MagicMock()
    client.chat.completions.create = AsyncMock()
    return client


@pytest.fixture
def mock_rag_router():
    """Mock RAGRouter"""
    router = MagicMock()
    router.route = MagicMock(
        return_value={
            "query_type": "brand_metrics",
            "confidence": 0.8,
            "requires_data": True,
            "requires_rag": True,
        }
    )
    router.extract_entities = MagicMock(
        return_value={
            "brands": ["LANEIGE"],
            "categories": ["lip_care"],
        }
    )
    return router


@pytest.fixture
def mock_context_gatherer():
    """Mock ContextGatherer"""
    gatherer = MagicMock()
    gatherer.initialize = AsyncMock()
    gatherer.gather = AsyncMock()
    gatherer.set_retriever = MagicMock()
    return gatherer


@pytest.fixture
def mock_tool_executor():
    """Mock ToolExecutor"""
    executor = MagicMock()
    executor.execute = AsyncMock()
    return executor


@pytest.fixture
def mock_response_pipeline():
    """Mock ResponsePipeline"""
    pipeline = MagicMock()
    pipeline.generate = AsyncMock()
    pipeline.generate_with_tool_result = AsyncMock()
    pipeline.set_client = MagicMock()
    return pipeline


@pytest.fixture
def mock_cache():
    """Mock ResponseCache"""
    cache = MagicMock()
    cache.get = MagicMock(return_value=None)
    cache.set = MagicMock()
    cache.get_stats = MagicMock(return_value={"hits": 0, "misses": 0})
    return cache


@pytest.fixture
def mock_state():
    """Mock OrchestratorState"""
    state = MagicMock()
    state.set_session = MagicMock()
    state.to_dict = MagicMock(return_value={"session_id": "test"})
    state.to_context_summary = MagicMock(return_value="State: active")
    return state


@pytest.fixture
def mock_context():
    """Mock Context"""
    return Context(
        query="LANEIGE 순위 알려줘",
        entities={"brands": ["LANEIGE"], "categories": ["lip_care"]},
        rag_docs=[
            {
                "content": "LANEIGE Lip Sleeping Mask는 1위입니다",
                "metadata": {"title": "Lip Care Report"},
            }
        ],
        kg_facts=[],
        kg_inferences=[],
        system_state=SystemState(
            data_freshness="fresh",
            kg_initialized=True,
            kg_triple_count=50,
        ),
        summary="LANEIGE 정보",
    )


@pytest.fixture
def orchestrator(
    mock_openai_client,
    mock_rag_router,
    mock_context_gatherer,
    mock_tool_executor,
    mock_response_pipeline,
    mock_cache,
    mock_state,
):
    """기본 Orchestrator fixture"""
    return LLMOrchestrator(
        openai_client=mock_openai_client,
        rag_router=mock_rag_router,
        context_gatherer=mock_context_gatherer,
        tool_executor=mock_tool_executor,
        response_pipeline=mock_response_pipeline,
        cache=mock_cache,
        state=mock_state,
        model="gpt-4o-mini",
    )


# =============================================================================
# 초기화 및 설정 테스트
# =============================================================================


class TestInitialization:
    """초기화 및 설정 테스트"""

    def test_init_with_all_params(self, orchestrator):
        """모든 파라미터로 초기화"""
        assert orchestrator.client is not None
        assert orchestrator.router is not None
        assert orchestrator.context_gatherer is not None
        assert orchestrator.tool_executor is not None
        assert orchestrator.response_pipeline is not None
        assert orchestrator.cache is not None
        assert orchestrator.state is not None
        assert orchestrator.model == "gpt-4o-mini"
        assert orchestrator.confidence_assessor is not None

    def test_init_with_defaults(self):
        """기본값으로 초기화"""
        orch = LLMOrchestrator()
        assert orch.client is None
        assert orch.router is None
        assert orch.context_gatherer is not None  # ContextGatherer()
        assert orch.tool_executor is not None  # ToolExecutor()
        assert orch.response_pipeline is not None  # ResponsePipeline()
        assert orch.cache is not None  # ResponseCache()
        assert orch.state is not None  # OrchestratorState()

    def test_init_stats(self, orchestrator):
        """통계 초기화"""
        stats = orchestrator.get_stats()
        assert stats["total_queries"] == 0
        assert stats["cache_hits"] == 0
        assert stats["rule_handled"] == 0
        assert stats["llm_decisions"] == 0
        assert stats["tools_called"] == 0
        assert stats["clarifications"] == 0

    def test_set_client(self, orchestrator, mock_response_pipeline):
        """클라이언트 설정"""
        new_client = MagicMock()
        orchestrator.set_client(new_client)
        assert orchestrator.client == new_client
        mock_response_pipeline.set_client.assert_called_once_with(new_client)

    def test_set_router(self, orchestrator):
        """라우터 설정"""
        new_router = MagicMock()
        orchestrator.set_router(new_router)
        assert orchestrator.router == new_router

    def test_set_retriever(self, orchestrator, mock_context_gatherer):
        """리트리버 설정"""
        new_retriever = MagicMock()
        orchestrator.set_retriever(new_retriever)
        mock_context_gatherer.set_retriever.assert_called_once_with(new_retriever)

    @pytest.mark.asyncio
    async def test_initialize(self, orchestrator, mock_context_gatherer):
        """비동기 초기화"""
        await orchestrator.initialize()
        mock_context_gatherer.initialize.assert_called_once()


# =============================================================================
# process() 메인 메서드 테스트
# =============================================================================


class TestProcess:
    """process() 메인 메서드 테스트"""

    @pytest.mark.asyncio
    async def test_process_cache_hit(self, orchestrator, mock_cache, mock_context_gatherer):
        """캐시 히트"""
        cached_response = Response(
            text="Cached response",
            query_type="brand_metrics",
            confidence_level=ConfidenceLevel.HIGH,
            confidence_score=5.0,
        )
        mock_cache.get.return_value = cached_response

        result = await orchestrator.process("Test query")

        assert result == cached_response
        assert orchestrator.get_stats()["cache_hits"] == 1
        mock_context_gatherer.gather.assert_not_called()

    @pytest.mark.asyncio
    async def test_process_skip_cache(
        self, orchestrator, mock_cache, mock_context_gatherer, mock_context
    ):
        """캐시 스킵"""
        mock_cache.get.return_value = Response(text="Cached")
        mock_context_gatherer.gather.return_value = mock_context

        with patch.object(
            orchestrator, "_route_by_confidence", new_callable=AsyncMock
        ) as mock_route:
            mock_route.return_value = Response(text="Fresh")
            result = await orchestrator.process("Test query", skip_cache=True)

        assert result.text == "Fresh"
        mock_cache.get.assert_not_called()

    @pytest.mark.asyncio
    async def test_process_with_session_id(
        self, orchestrator, mock_state, mock_context_gatherer, mock_context
    ):
        """세션 ID 설정"""
        mock_context_gatherer.gather.return_value = mock_context

        with patch.object(
            orchestrator, "_route_by_confidence", new_callable=AsyncMock
        ) as mock_route:
            mock_route.return_value = Response(text="Test")
            await orchestrator.process("Test query", session_id="session123")

        mock_state.set_session.assert_called_once_with("session123")

    @pytest.mark.asyncio
    async def test_process_high_confidence(
        self,
        orchestrator,
        mock_context_gatherer,
        mock_context,
        mock_response_pipeline,
    ):
        """HIGH 신뢰도 - 직접 응답"""
        mock_context_gatherer.gather.return_value = mock_context

        with patch.object(orchestrator.confidence_assessor, "assess") as mock_assess:
            mock_assess.return_value = ConfidenceLevel.HIGH
            result = await orchestrator.process("LANEIGE 순위")

        # RAG docs가 있으므로 _generate_rag_based_response가 호출됨
        assert "1위" in result.text
        assert orchestrator.get_stats()["rule_handled"] == 1
        assert orchestrator.get_stats()["total_queries"] == 1

    @pytest.mark.asyncio
    async def test_process_unknown_confidence(
        self, orchestrator, mock_context_gatherer, mock_context
    ):
        """UNKNOWN 신뢰도 - 명확화 요청"""
        mock_context_gatherer.gather.return_value = mock_context

        with patch.object(orchestrator.confidence_assessor, "assess") as mock_assess:
            mock_assess.return_value = ConfidenceLevel.UNKNOWN
            result = await orchestrator.process("알려줘")

        assert result.is_clarification
        assert orchestrator.get_stats()["clarifications"] == 1

    @pytest.mark.asyncio
    async def test_process_exception_fallback(self, orchestrator, mock_context_gatherer):
        """예외 발생 시 폴백"""
        mock_context_gatherer.gather.side_effect = Exception("Test error")

        result = await orchestrator.process("Test query")

        assert result.is_fallback
        assert "오류가 발생했습니다" in result.text

    @pytest.mark.asyncio
    async def test_process_processing_time(self, orchestrator, mock_context_gatherer, mock_context):
        """처리 시간 기록"""
        mock_context_gatherer.gather.return_value = mock_context

        with patch.object(
            orchestrator, "_route_by_confidence", new_callable=AsyncMock
        ) as mock_route:
            mock_route.return_value = Response(text="Test")
            result = await orchestrator.process("Test query")

        assert result.processing_time_ms > 0

    @pytest.mark.asyncio
    async def test_process_cache_storage(
        self, orchestrator, mock_cache, mock_context_gatherer, mock_context
    ):
        """캐시 저장"""
        mock_context_gatherer.gather.return_value = mock_context
        mock_cache.get.return_value = None

        with patch.object(
            orchestrator, "_route_by_confidence", new_callable=AsyncMock
        ) as mock_route:
            response = Response(text="Test", is_fallback=False)
            mock_route.return_value = response
            await orchestrator.process("Test query")

        mock_cache.set.assert_called_once()


# =============================================================================
# 라우팅 테스트
# =============================================================================


class TestRouting:
    """라우팅 메서드 테스트"""

    def test_rule_based_routing_with_router(self, orchestrator, mock_rag_router):
        """라우터 있을 때"""
        route_result, entities = orchestrator._rule_based_routing("LANEIGE 순위")

        assert route_result["query_type"] == "brand_metrics"
        assert entities["brands"] == ["LANEIGE"]
        mock_rag_router.route.assert_called_once()
        mock_rag_router.extract_entities.assert_called_once()

    def test_rule_based_routing_without_router(self):
        """라우터 없을 때"""
        orch = LLMOrchestrator()
        route_result, entities = orch._rule_based_routing("Test query")

        assert route_result["query_type"] == "unknown"
        assert route_result["confidence"] == 0.0
        assert route_result["requires_rag"] is True
        assert entities == {}

    @pytest.mark.asyncio
    async def test_route_by_confidence_high(self, orchestrator, mock_context):
        """HIGH 신뢰도 라우팅"""
        route_result = {"query_type": "brand_metrics", "confidence": 0.9}

        with patch.object(
            orchestrator, "_generate_direct_response", new_callable=AsyncMock
        ) as mock_gen:
            mock_gen.return_value = Response(text="Direct")
            result = await orchestrator._route_by_confidence(
                query="Test",
                context=mock_context,
                route_result=route_result,
                confidence=ConfidenceLevel.HIGH,
                current_metrics=None,
            )

        assert result.text == "Direct"
        mock_gen.assert_called_once()

    @pytest.mark.asyncio
    async def test_route_by_confidence_unknown(self, orchestrator, mock_context):
        """UNKNOWN 신뢰도 라우팅"""
        route_result = {"query_type": "unknown", "confidence": 0.0}

        result = await orchestrator._route_by_confidence(
            query="Test",
            context=mock_context,
            route_result=route_result,
            confidence=ConfidenceLevel.UNKNOWN,
            current_metrics=None,
        )

        assert result.is_clarification

    @pytest.mark.asyncio
    async def test_route_by_confidence_medium(self, orchestrator, mock_context):
        """MEDIUM 신뢰도 - LLM 판단"""
        route_result = {"query_type": "general", "confidence": 0.5}

        with patch.object(orchestrator, "_llm_decision_flow", new_callable=AsyncMock) as mock_flow:
            mock_flow.return_value = Response(text="LLM result")
            result = await orchestrator._route_by_confidence(
                query="Test",
                context=mock_context,
                route_result=route_result,
                confidence=ConfidenceLevel.MEDIUM,
                current_metrics=None,
            )

        assert result.text == "LLM result"
        mock_flow.assert_called_once()


# =============================================================================
# LLM 판단 흐름 테스트
# =============================================================================


class TestLLMDecisionFlow:
    """LLM 판단 흐름 테스트"""

    @pytest.mark.asyncio
    async def test_llm_decision_flow_with_tool(
        self,
        orchestrator,
        mock_context,
        mock_tool_executor,
        mock_response_pipeline,
    ):
        """도구 실행 필요"""
        decision = Decision(
            tool="crawl_amazon",
            tool_params={"category": "lip_care"},
            reason="크롤링 필요",
            confidence=0.9,
        )

        with patch.object(
            orchestrator, "_get_llm_decision", new_callable=AsyncMock
        ) as mock_decision:
            mock_decision.return_value = decision
            mock_tool_executor.execute.return_value = {"status": "success"}
            mock_response_pipeline.generate_with_tool_result.return_value = Response(
                text="Tool result"
            )

            result = await orchestrator._llm_decision_flow(
                query="크롤링해줘",
                context=mock_context,
                route_result={},
                confidence=ConfidenceLevel.MEDIUM,
                current_metrics=None,
            )

        assert result.text == "Tool result"
        assert orchestrator.get_stats()["tools_called"] == 1
        mock_tool_executor.execute.assert_called_once()

    @pytest.mark.asyncio
    async def test_llm_decision_flow_direct_answer(
        self,
        orchestrator,
        mock_context,
        mock_response_pipeline,
    ):
        """직접 응답"""
        decision = Decision(
            tool="direct_answer",
            reason="컨텍스트로 응답 가능",
            confidence=0.8,
        )

        with patch.object(
            orchestrator, "_get_llm_decision", new_callable=AsyncMock
        ) as mock_decision:
            mock_decision.return_value = decision
            mock_response_pipeline.generate.return_value = Response(text="Direct answer")

            result = await orchestrator._llm_decision_flow(
                query="Test",
                context=mock_context,
                route_result={},
                confidence=ConfidenceLevel.MEDIUM,
                current_metrics=None,
            )

        assert result.text == "Direct answer"
        assert orchestrator.get_stats()["llm_decisions"] == 1

    @pytest.mark.asyncio
    async def test_get_llm_decision_success(self, orchestrator, mock_openai_client, mock_context):
        """LLM 판단 성공"""
        llm_response = MagicMock()
        llm_response.choices = [MagicMock()]
        llm_response.choices[0].message.content = json.dumps(
            {
                "tool": "crawl_amazon",
                "tool_params": {"category": "lip_care"},
                "reason": "크롤링 필요",
                "key_points": ["데이터 수집"],
            }
        )
        mock_openai_client.chat.completions.create.return_value = llm_response

        decision = await orchestrator._get_llm_decision(
            query="크롤링해줘",
            context=mock_context,
            route_result={},
        )

        assert decision.tool == "crawl_amazon"
        assert decision.reason == "크롤링 필요"
        mock_openai_client.chat.completions.create.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_llm_decision_no_client(self, mock_context):
        """클라이언트 없을 때"""
        orch = LLMOrchestrator(openai_client=None)

        decision = await orch._get_llm_decision(
            query="크롤링해줘",
            context=mock_context,
            route_result={},
        )

        assert decision.tool == "direct_answer"  # _default_decision: 크롤링 차단

    @pytest.mark.asyncio
    async def test_get_llm_decision_exception(self, orchestrator, mock_openai_client, mock_context):
        """LLM 호출 실패"""
        mock_openai_client.chat.completions.create.side_effect = Exception("API error")

        decision = await orchestrator._get_llm_decision(
            query="Test",
            context=mock_context,
            route_result={},
        )

        assert decision.tool is not None  # _default_decision fallback

    def test_format_tools_description(self, orchestrator):
        """도구 설명 포맷팅"""
        with patch(
            "src.core.llm_orchestrator.AGENT_TOOLS",
            {
                "crawl_amazon": MagicMock(description="Amazon 크롤링"),
                "calculate_metrics": MagicMock(description="지표 계산"),
            },
        ):
            desc = orchestrator._format_tools_description()

        assert "crawl_amazon" in desc
        assert "Amazon 크롤링" in desc

    def test_parse_decision_response_valid_json(self, orchestrator):
        """유효한 JSON 파싱"""
        response_text = """
        다음과 같이 판단합니다:
        ```json
        {
            "tool": "crawl_amazon",
            "tool_params": {"category": "lip_care"},
            "reason": "크롤링 필요",
            "key_points": ["데이터 수집"]
        }
        ```
        """
        decision = orchestrator._parse_decision_response(response_text)

        assert decision.tool == "crawl_amazon"
        assert decision.reason == "크롤링 필요"
        assert decision.confidence == 0.8

    def test_parse_decision_response_invalid_json(self, orchestrator):
        """잘못된 JSON 파싱"""
        response_text = "This is not JSON"
        decision = orchestrator._parse_decision_response(response_text)

        assert decision.tool == "direct_answer"
        assert "파싱 실패" in decision.reason

    def test_default_decision_crawling(self, orchestrator):
        """크롤링 요청 → direct_answer로 차단"""
        decision = orchestrator._default_decision("크롤링해줘", {})
        assert decision.tool == "direct_answer"
        assert decision.confidence == 0.8

    def test_default_decision_metrics(self, orchestrator):
        """지표 계산 요청 감지"""
        decision = orchestrator._default_decision("지표 계산해줘", {})
        assert decision.tool == "calculate_metrics"
        assert decision.confidence == 0.8

    def test_default_decision_data_query(self, orchestrator):
        """데이터 조회 요청"""
        route_result = {"requires_data": True}
        decision = orchestrator._default_decision("순위 알려줘", route_result)
        assert decision.tool == "query_data"

    def test_default_decision_direct_answer(self, orchestrator):
        """직접 응답 가능"""
        decision = orchestrator._default_decision("SoS가 뭐야?", {})
        assert decision.tool == "direct_answer"


# =============================================================================
# 응답 생성 테스트
# =============================================================================


class TestResponseGeneration:
    """응답 생성 테스트"""

    @pytest.mark.asyncio
    async def test_generate_direct_response_with_rag_docs(self, orchestrator, mock_context):
        """RAG 문서 있을 때 직접 응답"""
        result = await orchestrator._generate_direct_response("Test", mock_context)

        assert isinstance(result, Response)
        assert "1위" in result.text

    @pytest.mark.asyncio
    async def test_generate_direct_response_no_rag_docs(self, orchestrator, mock_response_pipeline):
        """RAG 문서 없을 때"""
        context = Context(
            query="Test",
            entities={},
            rag_docs=[],
            kg_facts=[],
            kg_inferences=[],
            system_state=SystemState(),
            summary="",
        )
        mock_response_pipeline.generate.return_value = Response(text="Pipeline response")

        result = await orchestrator._generate_direct_response("Test", context)

        assert result.text == "Pipeline response"
        mock_response_pipeline.generate.assert_called_once()

    def test_generate_rag_based_response(self, orchestrator, mock_context):
        """RAG 기반 응답 생성"""
        result = orchestrator._generate_rag_based_response("Test", mock_context)

        assert isinstance(result, Response)
        assert "1위" in result.text
        assert result.confidence_level == ConfidenceLevel.HIGH
        assert "Lip Care Report" in result.sources

    def test_generate_rag_based_response_no_content(self, orchestrator):
        """RAG 문서에 내용 없을 때"""
        context = Context(
            query="Test",
            entities={},
            rag_docs=[{"content": "", "metadata": {}}],
            kg_facts=[],
            kg_inferences=[],
            system_state=SystemState(),
            summary="",
        )

        result = orchestrator._generate_rag_based_response("Test", context)

        assert "찾을 수 없습니다" in result.text

    def test_infer_query_type_definition(self, orchestrator):
        """정의 질문"""
        query_type = orchestrator._infer_query_type_from_query("SoS가 뭐야?")
        assert query_type == "definition"

    def test_infer_query_type_interpretation(self, orchestrator):
        """해석 질문"""
        query_type = orchestrator._infer_query_type_from_query("SoS가 높으면 어때?")
        assert query_type == "interpretation"

    def test_infer_query_type_data_query(self, orchestrator):
        """데이터 조회 질문"""
        query_type = orchestrator._infer_query_type_from_query("현재 순위는?")
        assert query_type == "data_query"

    def test_infer_query_type_analysis(self, orchestrator):
        """분석 질문"""
        query_type = orchestrator._infer_query_type_from_query("LANEIGE 분석해줘")
        assert query_type == "analysis"

    def test_infer_query_type_general(self, orchestrator):
        """일반 질문"""
        query_type = orchestrator._infer_query_type_from_query("알려줘")
        assert query_type == "general"

    def test_generate_clarification(self, orchestrator, mock_context):
        """명확화 요청 생성"""
        route_result = {"fallback_message": "질문을 이해하지 못했습니다"}

        result = orchestrator._generate_clarification("Test", mock_context, route_result)

        assert result.is_clarification
        assert "질문을 이해하지 못했습니다" in result.text
        assert len(result.suggestions) > 0

    def test_generate_clarification_default_message(self, orchestrator, mock_context):
        """기본 명확화 메시지"""
        result = orchestrator._generate_clarification("Test", mock_context, {})

        assert result.is_clarification
        assert "파악하지 못했습니다" in result.text


# =============================================================================
# 상태 및 통계 테스트
# =============================================================================


class TestStatsAndState:
    """상태 및 통계 테스트"""

    def test_get_stats(self, orchestrator, mock_cache, mock_state):
        """통계 조회"""
        stats = orchestrator.get_stats()

        assert "total_queries" in stats
        assert "cache_hits" in stats
        assert "cache_stats" in stats
        assert "state" in stats

    def test_reset_stats(self, orchestrator):
        """통계 초기화"""
        orchestrator._stats["total_queries"] = 10
        orchestrator.reset_stats()

        stats = orchestrator.get_stats()
        assert stats["total_queries"] == 0
        assert stats["cache_hits"] == 0

    def test_get_state_summary(self, orchestrator, mock_state):
        """상태 요약"""
        summary = orchestrator.get_state_summary()

        assert summary == "State: active"
        mock_state.to_context_summary.assert_called_once()


# =============================================================================
# 통합 시나리오 테스트
# =============================================================================


class TestIntegrationScenarios:
    """통합 시나리오 테스트"""

    @pytest.mark.asyncio
    async def test_full_flow_high_confidence(
        self,
        orchestrator,
        mock_context_gatherer,
        mock_context,
        mock_response_pipeline,
    ):
        """전체 흐름: HIGH 신뢰도"""
        mock_context_gatherer.gather.return_value = mock_context

        with patch.object(orchestrator.confidence_assessor, "assess") as mock_assess:
            mock_assess.return_value = ConfidenceLevel.HIGH
            result = await orchestrator.process("LANEIGE 순위")

        # RAG docs가 있으므로 _generate_rag_based_response가 호출됨
        assert "1위" in result.text
        assert orchestrator.get_stats()["total_queries"] == 1
        assert orchestrator.get_stats()["rule_handled"] == 1

    @pytest.mark.asyncio
    async def test_full_flow_medium_confidence_with_tool(
        self,
        orchestrator,
        mock_context_gatherer,
        mock_context,
        mock_openai_client,
        mock_tool_executor,
        mock_response_pipeline,
    ):
        """전체 흐름: MEDIUM 신뢰도 + 도구 실행"""
        mock_context_gatherer.gather.return_value = mock_context

        llm_response = MagicMock()
        llm_response.choices = [MagicMock()]
        llm_response.choices[0].message.content = json.dumps(
            {
                "tool": "calculate_metrics",
                "tool_params": {},
                "reason": "지표 계산 필요",
            }
        )
        mock_openai_client.chat.completions.create.return_value = llm_response
        mock_tool_executor.execute.return_value = {"sos": 15.5}
        mock_response_pipeline.generate_with_tool_result.return_value = Response(
            text="지표 계산 완료"
        )

        with patch.object(orchestrator.confidence_assessor, "assess") as mock_assess:
            mock_assess.return_value = ConfidenceLevel.MEDIUM
            result = await orchestrator.process("지표 계산해줘")

        assert result.text == "지표 계산 완료"
        assert orchestrator.get_stats()["llm_decisions"] == 1
        assert orchestrator.get_stats()["tools_called"] == 1

    @pytest.mark.asyncio
    async def test_stats_accumulation(self, orchestrator, mock_context_gatherer, mock_context):
        """통계 누적"""
        mock_context_gatherer.gather.return_value = mock_context

        # HIGH confidence 2번
        with patch.object(orchestrator.confidence_assessor, "assess") as mock_assess:
            mock_assess.return_value = ConfidenceLevel.HIGH
            await orchestrator.process("Query 1")
            await orchestrator.process("Query 2")

        # UNKNOWN confidence 1번
        with patch.object(orchestrator.confidence_assessor, "assess") as mock_assess:
            mock_assess.return_value = ConfidenceLevel.UNKNOWN
            await orchestrator.process("Query 3")

        stats = orchestrator.get_stats()
        assert stats["total_queries"] == 3
        assert stats["rule_handled"] == 2
        assert stats["clarifications"] == 1
