"""
ContextGatherer 단위 테스트
============================
Coverage Target: 50%+ (from 12.60%)

Critical test areas:
- Initialization with/without retriever
- gather() with retrieve_unified() path
- gather() with legacy retrieve() fallback
- gather() without retriever
- gather() with/without entities
- gather() with include_system_state flag
- gather_for_decision() lightweight context
- Error handling in gather()
- _convert_kg_facts() with max_kg_facts limit
- _get_system_state() from OrchestratorState
- _build_summary() formatting
- _build_decision_summary() formatting
- _format_system_state() variations
- _format_kg_fact() for all fact types
- get_stats() with/without retriever
"""

from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.core.context_gatherer import ContextGatherer
from src.core.models import Context, KGFact, SystemState
from src.core.state import OrchestratorState
from src.domain.value_objects.retrieval_result import UnifiedRetrievalResult

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_orchestrator_state():
    """OrchestratorState mock"""
    state = OrchestratorState()
    state.last_crawl_time = datetime.now() - timedelta(hours=2)
    state.data_freshness = "fresh"
    state.kg_initialized = True
    state.kg_triple_count = 1000
    return state


@pytest.fixture
def mock_unified_result():
    """UnifiedRetrievalResult - actual instance, not mock"""
    return UnifiedRetrievalResult(
        query="LANEIGE SoS?",
        entities={"brands": ["LANEIGE"], "categories": ["Lip Care"]},
        rag_chunks=[
            {"content": "RAG doc 1", "metadata": {"title": "Doc 1"}},
            {"content": "RAG doc 2", "metadata": {"title": "Doc 2"}},
        ],
        ontology_facts=[
            {"type": "brand_info", "entity": "LANEIGE", "data": {"sos": 0.15, "avg_rank": 5.2}},
            {"type": "competitors", "entity": "LANEIGE", "data": [{"brand": "Burt's Bees"}]},
        ],
        inferences=[
            {"insight": "LANEIGE dominates Lip Care", "recommendation": "Maintain leadership"}
        ],
        combined_context="[KG Facts] LANEIGE SoS 15%\n[RAG] Doc 1 content",
        confidence=0.85,
        entity_links=["LANEIGE"],
        retriever_type="unified",
    )


@pytest.fixture
def mock_hybrid_context():
    """Legacy HybridContext mock"""
    context = MagicMock()
    context.entities = {"brands": ["LANEIGE"]}
    context.rag_chunks = [{"content": "Legacy RAG doc", "metadata": {}}]
    context.ontology_facts = [{"type": "brand_info", "entity": "LANEIGE", "data": {"sos": 0.12}}]
    context.inferences = [
        MagicMock(to_dict=lambda: {"insight": "Legacy inference", "recommendation": "Test rec"})
    ]
    return context


@pytest.fixture
def mock_retriever_unified(mock_unified_result):
    """Retriever with retrieve_unified() method"""
    retriever = MagicMock()
    retriever.initialize = AsyncMock()
    retriever.retrieve_unified = AsyncMock(return_value=mock_unified_result)
    return retriever


@pytest.fixture
def mock_retriever_legacy(mock_hybrid_context):
    """Retriever with legacy retrieve() method"""
    retriever = MagicMock()
    retriever.initialize = AsyncMock()
    retriever.retrieve = AsyncMock(return_value=mock_hybrid_context)
    # No retrieve_unified attribute
    delattr(retriever, "retrieve_unified") if hasattr(retriever, "retrieve_unified") else None
    return retriever


@pytest.fixture
def mock_retriever_with_kg():
    """Retriever with KG for gather_for_decision()"""
    retriever = MagicMock()
    kg = MagicMock()
    kg.get_entity_metadata = MagicMock(
        return_value={"sos": 0.15, "avg_rank": 5.2, "product_count": 3}
    )
    retriever.kg = kg
    retriever.get_stats = MagicMock(return_value={"kg_facts": 100, "rag_docs": 50})
    return retriever


# =============================================================================
# Initialization Tests
# =============================================================================


class TestInitialization:
    """초기화 테스트"""

    def test_init_minimal(self):
        """최소 파라미터로 초기화"""
        gatherer = ContextGatherer()
        assert gatherer.retriever is None
        assert gatherer.state is not None
        assert gatherer.max_rag_docs == 5
        assert gatherer.max_kg_facts == 10
        assert not gatherer._initialized

    def test_init_with_retriever(self, mock_retriever_unified, mock_orchestrator_state):
        """Retriever와 state를 제공하여 초기화"""
        gatherer = ContextGatherer(
            hybrid_retriever=mock_retriever_unified,
            orchestrator_state=mock_orchestrator_state,
            max_rag_docs=3,
            max_kg_facts=7,
        )
        assert gatherer.retriever == mock_retriever_unified
        assert gatherer.state == mock_orchestrator_state
        assert gatherer.max_rag_docs == 3
        assert gatherer.max_kg_facts == 7

    def test_init_creates_default_state(self):
        """State가 없으면 기본 생성"""
        gatherer = ContextGatherer()
        assert isinstance(gatherer.state, OrchestratorState)

    @pytest.mark.asyncio
    async def test_initialize_idempotent(self, mock_retriever_unified):
        """initialize() 중복 호출 시 스킵"""
        gatherer = ContextGatherer(hybrid_retriever=mock_retriever_unified)
        await gatherer.initialize()
        assert gatherer._initialized
        assert mock_retriever_unified.initialize.call_count == 1

        # 다시 호출
        await gatherer.initialize()
        # 여전히 1회만 호출
        assert mock_retriever_unified.initialize.call_count == 1

    @pytest.mark.asyncio
    async def test_initialize_calls_retriever(self, mock_retriever_unified):
        """initialize() 시 retriever.initialize() 호출"""
        gatherer = ContextGatherer(hybrid_retriever=mock_retriever_unified)
        await gatherer.initialize()
        mock_retriever_unified.initialize.assert_called_once()

    @pytest.mark.asyncio
    async def test_initialize_without_retriever(self):
        """Retriever 없이 initialize() 호출"""
        gatherer = ContextGatherer()
        await gatherer.initialize()
        assert gatherer._initialized

    def test_set_retriever(self, mock_retriever_unified):
        """set_retriever()로 지연 주입"""
        gatherer = ContextGatherer()
        gatherer._initialized = True
        gatherer.set_retriever(mock_retriever_unified)
        assert gatherer.retriever == mock_retriever_unified
        assert not gatherer._initialized  # 초기화 플래그 리셋


# =============================================================================
# gather() Tests - Main Method
# =============================================================================


class TestGather:
    """gather() 메서드 테스트 (메인 컨텍스트 수집)"""

    @pytest.mark.asyncio
    async def test_gather_unified_path(self, mock_retriever_unified, mock_orchestrator_state):
        """retrieve_unified() 경로 - 정상 케이스"""
        gatherer = ContextGatherer(
            hybrid_retriever=mock_retriever_unified, orchestrator_state=mock_orchestrator_state
        )
        result = await gatherer.gather(query="LANEIGE SoS?", entities=None)

        # 기본 검증
        assert isinstance(result, Context)
        assert result.query == "LANEIGE SoS?"
        # entities가 retriever에서 추출됨 (UnifiedRetrievalResult에서)
        assert result.entities
        assert "LANEIGE" in result.entities.get("brands", [])

        # RAG docs
        assert len(result.rag_docs) == 2
        assert result.rag_docs[0]["content"] == "RAG doc 1"

        # KG facts
        assert len(result.kg_facts) == 2
        assert result.kg_facts[0].fact_type == "brand_info"
        assert result.kg_facts[0].entity == "LANEIGE"

        # Inferences
        assert len(result.kg_inferences) == 1
        assert "dominates" in result.kg_inferences[0]["insight"]

        # Summary from combined_context
        assert result.summary
        assert "[KG Facts]" in result.summary or "[시스템 상태]" in result.summary

        # System state
        assert result.system_state is not None
        assert result.gathered_at is not None

    @pytest.mark.asyncio
    async def test_gather_legacy_path(self, mock_retriever_legacy, mock_orchestrator_state):
        """retrieve() fallback 경로 - legacy 지원"""
        gatherer = ContextGatherer(
            hybrid_retriever=mock_retriever_legacy, orchestrator_state=mock_orchestrator_state
        )
        result = await gatherer.gather(query="Test query")

        assert isinstance(result, Context)
        assert result.query == "Test query"
        assert len(result.rag_docs) == 1
        assert len(result.kg_facts) == 1
        assert len(result.kg_inferences) == 1

    @pytest.mark.asyncio
    async def test_gather_without_retriever(self, mock_orchestrator_state):
        """Retriever 없이 gather() - 시스템 상태만"""
        gatherer = ContextGatherer(orchestrator_state=mock_orchestrator_state)
        result = await gatherer.gather(query="Test")

        assert isinstance(result, Context)
        assert result.query == "Test"
        assert len(result.rag_docs) == 0
        assert len(result.kg_facts) == 0
        assert result.system_state is not None

    @pytest.mark.asyncio
    async def test_gather_with_entities_provided(
        self, mock_retriever_unified, mock_orchestrator_state
    ):
        """entities를 명시적으로 제공"""
        gatherer = ContextGatherer(
            hybrid_retriever=mock_retriever_unified, orchestrator_state=mock_orchestrator_state
        )
        entities = {"brands": ["TEST_BRAND"], "categories": ["TEST_CAT"]}
        result = await gatherer.gather(query="Test", entities=entities)

        # 제공된 entities 유지 (retriever가 추출한 entities로 덮어쓰지 않음)
        assert result.entities == entities

    @pytest.mark.asyncio
    async def test_gather_without_system_state(
        self, mock_retriever_unified, mock_orchestrator_state
    ):
        """include_system_state=False"""
        gatherer = ContextGatherer(
            hybrid_retriever=mock_retriever_unified, orchestrator_state=mock_orchestrator_state
        )
        result = await gatherer.gather(query="Test", include_system_state=False)

        assert result.system_state is None

    @pytest.mark.asyncio
    async def test_gather_max_rag_docs_limit(self, mock_retriever_unified, mock_orchestrator_state):
        """max_rag_docs 제한 적용"""
        gatherer = ContextGatherer(
            hybrid_retriever=mock_retriever_unified,
            orchestrator_state=mock_orchestrator_state,
            max_rag_docs=1,
        )
        result = await gatherer.gather(query="Test")

        # max_rag_docs=1이므로 1개만
        assert len(result.rag_docs) <= 1

    @pytest.mark.asyncio
    async def test_gather_auto_initialize(self, mock_retriever_unified):
        """초기화되지 않은 상태에서 gather() 호출 시 자동 초기화"""
        gatherer = ContextGatherer(hybrid_retriever=mock_retriever_unified)
        assert not gatherer._initialized

        await gatherer.gather(query="Test")
        assert gatherer._initialized
        mock_retriever_unified.initialize.assert_called_once()

    @pytest.mark.asyncio
    async def test_gather_exception_handling(self, mock_orchestrator_state):
        """Retriever 에러 발생 시 예외 처리"""
        retriever = MagicMock()
        retriever.initialize = AsyncMock()
        retriever.retrieve_unified = AsyncMock(side_effect=Exception("Retriever failed"))

        gatherer = ContextGatherer(
            hybrid_retriever=retriever, orchestrator_state=mock_orchestrator_state
        )
        result = await gatherer.gather(query="Test")

        # 에러가 발생해도 Context 반환
        assert isinstance(result, Context)
        assert "오류" in result.summary
        assert "Retriever failed" in result.summary

    @pytest.mark.asyncio
    async def test_gather_summary_with_existing_combined_context(
        self, mock_retriever_unified, mock_orchestrator_state
    ):
        """retrieve_unified()가 combined_context를 제공한 경우 summary 유지"""
        gatherer = ContextGatherer(
            hybrid_retriever=mock_retriever_unified, orchestrator_state=mock_orchestrator_state
        )
        result = await gatherer.gather(query="Test")

        # combined_context가 있으므로 그대로 사용 (시스템 상태 추가)
        assert result.summary
        assert "[KG Facts]" in result.summary or "[시스템 상태]" in result.summary

    @pytest.mark.asyncio
    async def test_gather_summary_without_combined_context(
        self, mock_retriever_legacy, mock_orchestrator_state
    ):
        """combined_context가 없으면 _build_summary() 사용"""
        gatherer = ContextGatherer(
            hybrid_retriever=mock_retriever_legacy, orchestrator_state=mock_orchestrator_state
        )
        result = await gatherer.gather(query="Test")

        assert result.summary
        # _build_summary()로 생성된 요약
        assert len(result.summary) > 0


# =============================================================================
# gather_for_decision() Tests
# =============================================================================


class TestGatherForDecision:
    """gather_for_decision() 경량 컨텍스트 테스트"""

    @pytest.mark.asyncio
    async def test_gather_for_decision_with_kg(
        self, mock_retriever_with_kg, mock_orchestrator_state
    ):
        """KG 데이터를 활용한 경량 컨텍스트"""
        gatherer = ContextGatherer(
            hybrid_retriever=mock_retriever_with_kg, orchestrator_state=mock_orchestrator_state
        )
        entities = {"brands": ["LANEIGE", "Burt's Bees"]}
        result = await gatherer.gather_for_decision(query="Test", entities=entities)

        assert isinstance(result, Context)
        assert result.query == "Test"
        assert result.entities == entities

        # KG facts 수집 (최대 5개)
        assert len(result.kg_facts) <= 5
        assert result.kg_facts[0].fact_type == "brand_info"
        assert result.kg_facts[0].entity in ["LANEIGE", "Burt's Bees"]

        # System state
        assert result.system_state is not None

        # Decision summary
        assert result.summary
        assert "데이터:" in result.summary
        assert "KG:" in result.summary

    @pytest.mark.asyncio
    async def test_gather_for_decision_without_kg(self, mock_orchestrator_state):
        """KG 없이 경량 컨텍스트"""
        retriever = MagicMock()
        # kg 속성 없음
        delattr(retriever, "kg") if hasattr(retriever, "kg") else None

        gatherer = ContextGatherer(
            hybrid_retriever=retriever, orchestrator_state=mock_orchestrator_state
        )
        result = await gatherer.gather_for_decision(query="Test", entities={"brands": []})

        assert isinstance(result, Context)
        assert len(result.kg_facts) == 0
        assert result.system_state is not None

    @pytest.mark.asyncio
    async def test_gather_for_decision_no_retriever(self, mock_orchestrator_state):
        """Retriever 없이 경량 컨텍스트"""
        gatherer = ContextGatherer(orchestrator_state=mock_orchestrator_state)
        result = await gatherer.gather_for_decision(query="Test", entities={})

        assert isinstance(result, Context)
        assert len(result.kg_facts) == 0
        assert result.system_state is not None


# =============================================================================
# Helper Method Tests
# =============================================================================


class TestHelperMethods:
    """헬퍼 메서드 테스트"""

    def test_convert_kg_facts(self):
        """_convert_kg_facts() 변환 로직"""
        gatherer = ContextGatherer(max_kg_facts=2)
        ontology_facts = [
            {"type": "brand_info", "entity": "LANEIGE", "data": {"sos": 0.15}},
            {"type": "competitors", "entity": "LANEIGE", "data": []},
            {"type": "extra", "entity": "TEST", "data": {}},  # max_kg_facts=2이므로 무시
        ]

        result = gatherer._convert_kg_facts(ontology_facts)

        assert len(result) == 2  # max_kg_facts 적용
        assert isinstance(result[0], KGFact)
        assert result[0].fact_type == "brand_info"
        assert result[0].entity == "LANEIGE"
        assert result[0].data == {"sos": 0.15}

    def test_convert_kg_facts_empty(self):
        """빈 ontology_facts"""
        gatherer = ContextGatherer()
        result = gatherer._convert_kg_facts([])
        assert result == []

    def test_convert_kg_facts_missing_fields(self):
        """필드 누락 시 기본값"""
        gatherer = ContextGatherer()
        ontology_facts = [{}]  # 모든 필드 누락
        result = gatherer._convert_kg_facts(ontology_facts)

        assert len(result) == 1
        assert result[0].fact_type == "unknown"
        assert result[0].entity == ""
        assert result[0].data == {}

    def test_get_system_state(self, mock_orchestrator_state):
        """_get_system_state() - OrchestratorState에서 SystemState 생성"""
        gatherer = ContextGatherer(orchestrator_state=mock_orchestrator_state)
        result = gatherer._get_system_state()

        assert isinstance(result, SystemState)
        assert result.last_crawl_time == mock_orchestrator_state.last_crawl_time
        assert result.data_freshness == "fresh"
        assert result.kg_triple_count == 1000
        assert result.kg_initialized is True

    def test_format_system_state_with_crawl_time(self):
        """_format_system_state() - 크롤링 기록 있음"""
        gatherer = ContextGatherer()
        state = SystemState(
            last_crawl_time=datetime.now() - timedelta(hours=3),
            data_freshness="fresh",
            kg_initialized=True,
            kg_triple_count=500,
        )

        result = gatherer._format_system_state(state)

        assert "마지막 크롤링:" in result
        assert "3.0시간 전" in result
        assert "데이터 상태: fresh" in result
        assert "KG: 500 트리플" in result

    def test_format_system_state_no_crawl_time(self):
        """_format_system_state() - 크롤링 기록 없음"""
        gatherer = ContextGatherer()
        state = SystemState(
            last_crawl_time=None, data_freshness="unknown", kg_initialized=False, kg_triple_count=0
        )

        result = gatherer._format_system_state(state)

        assert "크롤링 기록 없음" in result
        assert "데이터 상태: unknown" in result
        assert "KG:" not in result  # kg_initialized=False이므로 표시 안함

    def test_format_kg_fact_brand_info(self):
        """_format_kg_fact() - brand_info 타입"""
        gatherer = ContextGatherer()
        fact = KGFact(fact_type="brand_info", entity="LANEIGE", data={"sos": 0.15, "avg_rank": 5.2})

        result = gatherer._format_kg_fact(fact)

        assert "LANEIGE" in result
        assert "SoS 15.0%" in result
        assert "평균순위 5.2" in result

    def test_format_kg_fact_brand_products(self):
        """_format_kg_fact() - brand_products 타입"""
        gatherer = ContextGatherer()
        fact = KGFact(fact_type="brand_products", entity="LANEIGE", data={"product_count": 12})

        result = gatherer._format_kg_fact(fact)

        assert "LANEIGE 제품 12개" in result

    def test_format_kg_fact_competitors(self):
        """_format_kg_fact() - competitors 타입"""
        gatherer = ContextGatherer()
        fact = KGFact(
            fact_type="competitors",
            entity="LANEIGE",
            data=[{"brand": "Burt's Bees"}, {"brand": "Neutrogena"}],
        )

        result = gatherer._format_kg_fact(fact)

        assert "LANEIGE 경쟁사:" in result
        assert "Burt's Bees" in result

    def test_format_kg_fact_category_brands(self):
        """_format_kg_fact() - category_brands 타입"""
        gatherer = ContextGatherer()
        fact = KGFact(
            fact_type="category_brands",
            entity="Lip Care",
            data={"top_brands": [{"brand": "LANEIGE"}, {"brand": "Burt's Bees"}]},
        )

        result = gatherer._format_kg_fact(fact)

        assert "Lip Care Top 브랜드:" in result
        assert "LANEIGE" in result

    def test_format_kg_fact_unknown_type(self):
        """_format_kg_fact() - 알 수 없는 타입"""
        gatherer = ContextGatherer()
        fact = KGFact(fact_type="unknown_type", entity="TEST", data={})

        result = gatherer._format_kg_fact(fact)

        assert result == ""

    def test_build_summary(self, mock_orchestrator_state):
        """_build_summary() - 전체 요약 생성"""
        gatherer = ContextGatherer(orchestrator_state=mock_orchestrator_state)
        context = Context(
            query="Test",
            system_state=SystemState(
                last_crawl_time=datetime.now() - timedelta(hours=1),
                data_freshness="fresh",
                kg_initialized=True,
                kg_triple_count=100,
            ),
            kg_inferences=[
                {"insight": "Insight 1", "recommendation": "Rec 1"},
                {"insight": "Insight 2", "recommendation": "Rec 2"},
            ],
            kg_facts=[
                KGFact(fact_type="brand_info", entity="LANEIGE", data={"sos": 0.15}),
            ],
            rag_docs=[
                {"content": "RAG content 1", "metadata": {"title": "Doc 1"}},
                {"content": "RAG content 2", "metadata": {}},
            ],
        )

        result = gatherer._build_summary(context)

        assert "[시스템 상태]" in result
        assert "[분석 인사이트]" in result
        assert "Insight 1" in result
        assert "Rec 1" in result
        assert "[관련 정보]" in result
        assert "LANEIGE" in result
        assert "[참조 문서]" in result
        assert "Doc 1" in result

    def test_build_summary_empty_context(self):
        """_build_summary() - 빈 컨텍스트"""
        gatherer = ContextGatherer()
        context = Context(query="Test")

        result = gatherer._build_summary(context)

        # 빈 문자열이거나 최소한의 내용
        assert isinstance(result, str)

    def test_build_decision_summary_fresh_data(self, mock_orchestrator_state):
        """_build_decision_summary() - 최신 데이터"""
        gatherer = ContextGatherer(orchestrator_state=mock_orchestrator_state)
        context = Context(
            query="Test",
            system_state=SystemState(
                last_crawl_time=datetime.now() - timedelta(minutes=10),
                data_freshness="fresh",
                kg_initialized=True,
                kg_triple_count=100,
            ),
            kg_facts=[
                KGFact(fact_type="brand_info", entity="LANEIGE", data={"sos": 0.15}),
            ],
        )

        result = gatherer._build_decision_summary(context)

        assert "데이터: 최신" in result
        assert "KG: 100 트리플" in result
        assert "LANEIGE SoS: 15.0%" in result

    def test_build_decision_summary_no_crawl(self):
        """_build_decision_summary() - 크롤링 기록 없음"""
        gatherer = ContextGatherer()
        context = Context(
            query="Test",
            system_state=SystemState(
                last_crawl_time=None,
                data_freshness="unknown",
                kg_initialized=False,
                kg_triple_count=0,
            ),
        )

        result = gatherer._build_decision_summary(context)

        assert "데이터: 없음 (크롤링 필요)" in result
        assert "KG: 미초기화" in result


# =============================================================================
# Utility Tests
# =============================================================================


class TestUtilities:
    """유틸리티 메서드 테스트"""

    def test_get_stats_without_retriever(self):
        """get_stats() - Retriever 없음"""
        gatherer = ContextGatherer(max_rag_docs=7, max_kg_facts=12)
        stats = gatherer.get_stats()

        assert stats["initialized"] is False
        assert stats["max_rag_docs"] == 7
        assert stats["max_kg_facts"] == 12
        assert stats["has_retriever"] is False
        assert "retriever_stats" not in stats

    def test_get_stats_with_retriever(self, mock_retriever_with_kg):
        """get_stats() - Retriever 있음"""
        gatherer = ContextGatherer(hybrid_retriever=mock_retriever_with_kg)
        stats = gatherer.get_stats()

        assert stats["has_retriever"] is True
        assert "retriever_stats" in stats
        assert stats["retriever_stats"]["kg_facts"] == 100

    def test_get_stats_after_initialization(self, mock_retriever_unified):
        """get_stats() - 초기화 후"""
        gatherer = ContextGatherer(hybrid_retriever=mock_retriever_unified)
        gatherer._initialized = True
        stats = gatherer.get_stats()

        assert stats["initialized"] is True


# =============================================================================
# Edge Cases
# =============================================================================


class TestEdgeCases:
    """엣지 케이스 및 경계 조건 테스트"""

    @pytest.mark.asyncio
    async def test_gather_with_non_unified_result_type(self, mock_orchestrator_state):
        """retrieve_unified()가 UnifiedRetrievalResult가 아닌 타입 반환"""
        retriever = MagicMock()
        retriever.initialize = AsyncMock()
        retriever.retrieve_unified = AsyncMock(return_value={"wrong": "type"})  # dict 반환

        gatherer = ContextGatherer(
            hybrid_retriever=retriever, orchestrator_state=mock_orchestrator_state
        )
        result = await gatherer.gather(query="Test")

        # 에러 없이 처리되어야 함
        assert isinstance(result, Context)
        assert len(result.rag_docs) == 0

    @pytest.mark.asyncio
    async def test_gather_with_current_metrics(self, mock_retriever_unified):
        """current_metrics 파라미터 전달"""
        gatherer = ContextGatherer(hybrid_retriever=mock_retriever_unified)
        metrics = {"sos": {"LANEIGE": 0.15}}
        await gatherer.gather(query="Test", current_metrics=metrics)

        # retrieve_unified()에 current_metrics 전달 확인
        mock_retriever_unified.retrieve_unified.assert_called_once()
        call_args = mock_retriever_unified.retrieve_unified.call_args
        assert call_args.kwargs["current_metrics"] == metrics

    def test_convert_kg_facts_respects_max_limit(self):
        """_convert_kg_facts()가 max_kg_facts를 정확히 지키는지"""
        gatherer = ContextGatherer(max_kg_facts=3)
        ontology_facts = [
            {"type": f"type_{i}", "entity": f"entity_{i}", "data": {}} for i in range(10)
        ]

        result = gatherer._convert_kg_facts(ontology_facts)

        assert len(result) == 3  # 정확히 3개

    def test_format_kg_fact_brand_info_partial_data(self):
        """_format_kg_fact() - 일부 데이터만 있는 brand_info"""
        gatherer = ContextGatherer()
        fact = KGFact(fact_type="brand_info", entity="LANEIGE", data={"sos": 0.15})  # avg_rank 없음

        result = gatherer._format_kg_fact(fact)

        assert "LANEIGE" in result
        assert "SoS 15.0%" in result
        assert "평균순위" not in result  # avg_rank가 없으므로 표시 안함

    def test_format_kg_fact_competitors_empty_list(self):
        """_format_kg_fact() - 경쟁사 목록 비어있음"""
        gatherer = ContextGatherer()
        fact = KGFact(fact_type="competitors", entity="LANEIGE", data=[])

        result = gatherer._format_kg_fact(fact)

        assert result == ""  # 빈 경쟁사 목록은 표시 안함

    @pytest.mark.asyncio
    async def test_gather_summary_appends_system_state(
        self, mock_retriever_unified, mock_orchestrator_state
    ):
        """combined_context가 있지만 시스템 상태가 없는 경우 앞에 추가"""
        # combined_context에 시스템 상태가 없는 경우
        unified_result = MagicMock()
        unified_result.entities = {}
        unified_result.rag_chunks = []
        unified_result.ontology_facts = []
        unified_result.inferences = []
        unified_result.combined_context = "[KG Facts] Some facts"  # 시스템 상태 없음
        unified_result.confidence = 0.8
        unified_result.entity_links = {}
        unified_result.retriever_type = "unified"

        retriever = MagicMock()
        retriever.initialize = AsyncMock()
        retriever.retrieve_unified = AsyncMock(return_value=unified_result)

        gatherer = ContextGatherer(
            hybrid_retriever=retriever, orchestrator_state=mock_orchestrator_state
        )
        result = await gatherer.gather(query="Test")

        # 시스템 상태가 앞에 추가되어야 함
        assert "[시스템 상태]" in result.summary
        assert result.summary.startswith("[시스템 상태]")

    def test_build_summary_limits_inferences_to_3(self, mock_orchestrator_state):
        """_build_summary()는 최대 3개의 인사이트만 포함"""
        gatherer = ContextGatherer(orchestrator_state=mock_orchestrator_state)
        context = Context(
            query="Test",
            kg_inferences=[
                {"insight": f"Insight {i}", "recommendation": f"Rec {i}"} for i in range(10)
            ],
        )

        result = gatherer._build_summary(context)

        # 1, 2, 3은 있어야 함
        assert "Insight 0" in result
        assert "Insight 1" in result
        assert "Insight 2" in result
        # 4번째 이상은 없어야 함
        assert "Insight 3" not in result
