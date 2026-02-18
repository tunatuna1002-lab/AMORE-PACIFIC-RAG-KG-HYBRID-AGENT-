"""
HybridRetriever 단위 테스트
==========================
src/rag/hybrid_retriever.py의 하이브리드 검색기 테스트
"""

import json
from unittest.mock import AsyncMock, MagicMock, mock_open, patch

import pytest

from src.domain.entities.relations import (
    InferenceResult,
    InsightType,
)
from src.rag.hybrid_retriever import (
    INTENT_DOC_TYPE_PRIORITY,
    EntityExtractor,
    HybridContext,
    HybridRetriever,
    QueryIntent,
    classify_intent,
    get_doc_type_filter,
)

# =============================================================================
# QueryIntent Tests
# =============================================================================


class TestQueryIntent:
    """QueryIntent enum과 분류 함수 테스트"""

    def test_query_intent_enum_values(self):
        """QueryIntent enum이 모든 필수 값을 가지는지 테스트"""
        expected_intents = {"diagnosis", "trend", "crisis", "metric", "general"}
        actual_intents = {intent.value for intent in QueryIntent}
        assert actual_intents == expected_intents

    def test_intent_doc_type_priority_mapping(self):
        """INTENT_DOC_TYPE_PRIORITY 매핑이 모든 Intent를 커버하는지 테스트"""
        for intent in QueryIntent:
            assert intent in INTENT_DOC_TYPE_PRIORITY

    def test_classify_intent_trend(self):
        """트렌드 키워드 감지 테스트"""
        queries = [
            "최근 트렌드가 뭐야",
            "요즘 인기 있는 제품",
            "바이럴된 키워드 알려줘",
            "펩타이드 트렌드 분석",
        ]
        for query in queries:
            assert classify_intent(query) == QueryIntent.TREND

    def test_classify_intent_crisis(self):
        """위기 대응 의도 감지 테스트"""
        queries = [
            "부정 리뷰가 많은데 어떻게 해",
            "문제 대응 방법",
            "이슈 발생 시 대응",
            "인플루언서 마케팅 필요해",
        ]
        for query in queries:
            assert classify_intent(query) == QueryIntent.CRISIS

    def test_classify_intent_diagnosis(self):
        """원인 분석 의도 감지 테스트"""
        queries = [
            "왜 순위가 떨어졌어",
            "원인 분석해줘",
            "갑자기 변동이 생긴 이유",
            "급변한 원인 확인",
        ]
        for query in queries:
            assert classify_intent(query) == QueryIntent.DIAGNOSIS

    def test_classify_intent_metric(self):
        """지표 해석 의도 감지 테스트"""
        queries = [
            "SoS가 뭐야",
            "HHI 지표 해석",
            "CPI 의미 알려줘",
            "점유율 정의",
        ]
        for query in queries:
            assert classify_intent(query) == QueryIntent.METRIC

    def test_classify_intent_general(self):
        """일반 의도 fallback 테스트"""
        queries = [
            "LANEIGE 제품 보여줘",
            "브랜드 정보",
            "카테고리 현황",
        ]
        for query in queries:
            assert classify_intent(query) == QueryIntent.GENERAL

    def test_classify_intent_priority_trend_over_diagnosis(self):
        """트렌드 키워드가 분석 키워드보다 우선하는지 테스트"""
        query = "최근 트렌드 분석해줘"
        assert classify_intent(query) == QueryIntent.TREND

    def test_get_doc_type_filter_diagnosis(self):
        """DIAGNOSIS 의도의 문서 필터 테스트"""
        doc_types = get_doc_type_filter(QueryIntent.DIAGNOSIS)
        assert "playbook" in doc_types
        assert "metric_guide" in doc_types
        assert isinstance(doc_types, list)

    def test_get_doc_type_filter_trend(self):
        """TREND 의도의 문서 필터 테스트"""
        doc_types = get_doc_type_filter(QueryIntent.TREND)
        assert "intelligence" in doc_types
        assert "knowledge_base" in doc_types

    def test_get_doc_type_filter_general(self):
        """GENERAL 의도는 필터가 None인지 테스트"""
        doc_types = get_doc_type_filter(QueryIntent.GENERAL)
        assert doc_types is None


# =============================================================================
# EntityExtractor Tests
# =============================================================================


class TestEntityExtractor:
    """EntityExtractor 테스트"""

    def test_entity_extractor_default_config(self):
        """기본 설정이 로드되는지 테스트"""
        brands = EntityExtractor.get_known_brands()
        assert "laneige" in brands
        assert "cosrx" in brands
        # category and indicator extraction verified via extract()
        extractor = EntityExtractor()
        entities = extractor.extract("lip care sos")
        assert "lip_care" in entities["categories"]
        assert "sos" in entities["indicators"]

    def test_entity_extractor_brand_extraction(self):
        """브랜드 추출 테스트"""
        extractor = EntityExtractor()
        query = "LANEIGE와 COSRX 비교해줘"
        entities = extractor.extract(query)

        assert "laneige" in entities["brands"]
        assert "cosrx" in entities["brands"]
        assert len(entities["brands"]) == 2

    def test_entity_extractor_brand_normalization(self):
        """브랜드명 정규화 테스트"""
        extractor = EntityExtractor()
        query = "라네즈 제품 보여줘"  # 별칭
        entities = extractor.extract(query)

        # 별칭이 정규화된 브랜드명으로 변환되어야 함
        assert "laneige" in entities["brands"]

    def test_entity_extractor_category_extraction(self):
        """카테고리 추출 테스트"""
        extractor = EntityExtractor()
        query = "립케어 카테고리 분석"
        entities = extractor.extract(query)

        assert "lip_care" in entities["categories"]

    def test_entity_extractor_indicator_extraction(self):
        """지표 추출 테스트"""
        extractor = EntityExtractor()
        query = "SoS와 HHI 지표 보여줘"
        entities = extractor.extract(query)

        assert "sos" in entities["indicators"]
        assert "hhi" in entities["indicators"]

    def test_entity_extractor_asin_pattern(self):
        """ASIN 패턴 추출 테스트"""
        extractor = EntityExtractor()
        query = "B08XYZ1234 제품 정보"
        entities = extractor.extract(query)

        assert "B08XYZ1234" in entities["products"]

    def test_entity_extractor_sentiment_extraction(self):
        """감성 키워드 추출 테스트"""
        extractor = EntityExtractor()
        query = "보습 효과가 좋은 제품"
        entities = extractor.extract(query)

        assert "보습" in entities["sentiments"]
        assert "Hydration" in entities["sentiment_clusters"]

    def test_entity_extractor_time_range_extraction(self):
        """시간 범위 추출 테스트"""
        extractor = EntityExtractor()
        query = "최근 7일 동안의 데이터"
        entities = extractor.extract(query)

        assert "7days" in entities["time_range"]

    def test_get_known_brands_returns_list(self):
        """get_known_brands()가 브랜드 목록을 반환하는지 테스트"""
        brands = EntityExtractor.get_known_brands()
        assert isinstance(brands, list)
        assert len(brands) > 0
        assert "laneige" in brands

    def test_get_brand_normalization_map_returns_dict(self):
        """get_brand_normalization_map()이 정규화 맵을 반환하는지 테스트"""
        mapping = EntityExtractor.get_brand_normalization_map()
        assert isinstance(mapping, dict)
        assert len(mapping) > 0
        # 모든 값이 문자열이어야 함
        assert all(isinstance(v, str) for v in mapping.values())


# =============================================================================
# HybridContext Tests
# =============================================================================


class TestHybridContext:
    """HybridContext dataclass 테스트"""

    def test_hybrid_context_creation(self):
        """HybridContext 생성 테스트"""
        context = HybridContext(
            query="test query",
            entities={"brands": ["laneige"]},
            ontology_facts=[{"type": "brand_info", "data": {}}],
            inferences=[],
            rag_chunks=[],
            combined_context="test context",
        )

        assert context.query == "test query"
        assert "laneige" in context.entities["brands"]
        assert len(context.ontology_facts) == 1

    def test_hybrid_context_default_fields(self):
        """HybridContext 기본값 테스트"""
        context = HybridContext(query="test")

        assert context.entities == {}
        assert context.ontology_facts == []
        assert context.inferences == []
        assert context.rag_chunks == []
        assert context.combined_context == ""
        assert context.metadata == {}

    def test_hybrid_context_to_dict(self):
        """HybridContext 직렬화 테스트"""
        inference = InferenceResult(
            rule_name="test_rule",
            insight_type=InsightType.MARKET_POSITION,
            insight="test insight",
            confidence=0.9,
        )

        context = HybridContext(
            query="test query", entities={"brands": ["laneige"]}, inferences=[inference]
        )

        result = context.to_dict()

        assert result["query"] == "test query"
        assert result["entities"]["brands"] == ["laneige"]
        assert len(result["inferences"]) == 1
        assert result["inferences"][0]["insight_type"] == "market_position"

    def test_hybrid_context_serialization_roundtrip(self):
        """HybridContext 직렬화/역직렬화 테스트"""
        inference = InferenceResult(
            rule_name="test_rule",
            insight_type=InsightType.COMPETITIVE_THREAT,
            insight="threat detected",
            confidence=0.85,
            recommendation="take action",
        )

        context = HybridContext(
            query="threat analysis",
            entities={"brands": ["laneige", "cosrx"]},
            inferences=[inference],
            metadata={"retrieval_time_ms": 123.45},
        )

        serialized = context.to_dict()

        # JSON 직렬화 가능한지 확인
        json_str = json.dumps(serialized)
        deserialized = json.loads(json_str)

        assert deserialized["query"] == "threat analysis"
        assert len(deserialized["inferences"]) == 1
        assert deserialized["metadata"]["retrieval_time_ms"] == 123.45


# =============================================================================
# HybridRetriever Tests
# =============================================================================


@pytest.fixture
def mock_knowledge_graph():
    """KnowledgeGraph mock"""
    mock = MagicMock()
    mock.get_entity_metadata = MagicMock(return_value={"sos": 0.05})
    mock.get_brand_products = MagicMock(return_value=[])
    mock.get_competitors = MagicMock(return_value=[])
    mock.get_category_brands = MagicMock(return_value=[])
    mock.get_category_hierarchy = MagicMock(return_value={})
    mock.get_product_sentiments = MagicMock(return_value={})
    mock.get_brand_sentiment_profile = MagicMock(return_value={})
    mock.find_products_by_sentiment = MagicMock(return_value=[])
    mock.query = MagicMock(return_value=[])
    mock.load_category_hierarchy = MagicMock(return_value=0)
    mock.get_stats = MagicMock(return_value={})
    return mock


@pytest.fixture
def mock_reasoner():
    """OntologyReasoner mock"""
    mock = MagicMock()
    mock.infer = MagicMock(return_value=[])
    mock.get_inference_stats = MagicMock(return_value={})
    mock.rules = []
    return mock


@pytest.fixture
def mock_doc_retriever():
    """DocumentRetriever mock"""
    mock = MagicMock()
    mock.initialize = AsyncMock()
    mock.search = AsyncMock(return_value=[])
    # Default BM25 to empty so _hybrid_search uses dense_only path
    mock.search_bm25 = MagicMock(return_value=[])
    return mock


class TestHybridRetriever:
    """HybridRetriever 테스트"""

    def test_retriever_initialization_default(self):
        """기본 초기화 테스트"""
        with (
            patch("src.rag.hybrid_retriever.KnowledgeGraph") as MockKG,
            patch("src.rag.hybrid_retriever.OntologyReasoner") as MockReasoner,
            patch("src.rag.hybrid_retriever.DocumentRetriever") as MockDocRetriever,
            patch("src.rag.hybrid_retriever.register_all_rules"),
        ):
            MockKG.return_value = MagicMock()
            MockReasoner.return_value = MagicMock(rules=[])
            MockDocRetriever.return_value = MagicMock()

            retriever = HybridRetriever(auto_init_rules=False)

            assert retriever.kg is not None
            assert retriever.reasoner is not None
            assert retriever.doc_retriever is not None
            assert isinstance(retriever.entity_extractor, EntityExtractor)
            assert not retriever._initialized

    def test_retriever_initialization_with_components(
        self, mock_knowledge_graph, mock_reasoner, mock_doc_retriever
    ):
        """컴포넌트를 주입한 초기화 테스트"""
        retriever = HybridRetriever(
            knowledge_graph=mock_knowledge_graph,
            reasoner=mock_reasoner,
            doc_retriever=mock_doc_retriever,
            auto_init_rules=False,
        )

        assert retriever.kg is mock_knowledge_graph
        assert retriever.reasoner is mock_reasoner
        assert retriever.doc_retriever is mock_doc_retriever

    def test_retriever_auto_init_rules(self):
        """비즈니스 규칙 자동 등록 테스트"""
        mock_reasoner = MagicMock(rules=[])

        with (
            patch("src.rag.hybrid_retriever.KnowledgeGraph"),
            patch("src.rag.hybrid_retriever.DocumentRetriever"),
            patch("src.rag.hybrid_retriever.register_all_rules") as mock_register,
        ):
            HybridRetriever(reasoner=mock_reasoner, auto_init_rules=True)

            # register_all_rules가 호출되었는지 확인
            mock_register.assert_called_once()

    @pytest.mark.asyncio
    async def test_retriever_initialize(
        self, mock_knowledge_graph, mock_reasoner, mock_doc_retriever
    ):
        """비동기 초기화 테스트"""
        retriever = HybridRetriever(
            knowledge_graph=mock_knowledge_graph,
            reasoner=mock_reasoner,
            doc_retriever=mock_doc_retriever,
            auto_init_rules=False,
        )

        await retriever.initialize()

        # doc_retriever.initialize가 호출되었는지 확인
        mock_doc_retriever.initialize.assert_called_once()

        # 카테고리 계층 로드가 시도되었는지 확인
        mock_knowledge_graph.load_category_hierarchy.assert_called_once()

        assert retriever._initialized

    @pytest.mark.asyncio
    async def test_retriever_expand_query_with_inferences(self):
        """추론 결과 기반 쿼리 확장 테스트"""
        mock_kg = MagicMock()
        mock_reasoner = MagicMock()
        mock_doc_retriever = MagicMock()

        retriever = HybridRetriever(
            knowledge_graph=mock_kg,
            reasoner=mock_reasoner,
            doc_retriever=mock_doc_retriever,
            auto_init_rules=False,
        )

        # 추론 결과 시뮬레이션
        inferences = [
            InferenceResult(
                rule_name="market_position",
                insight_type=InsightType.MARKET_POSITION,
                insight="dominant position",
            ),
            InferenceResult(
                rule_name="risk", insight_type=InsightType.RISK_ALERT, insight="risk detected"
            ),
        ]

        entities = {"indicators": ["sos"]}

        expanded_query = retriever._expand_query("LANEIGE 분석", inferences, entities)

        # 확장된 쿼리에 관련 키워드가 포함되었는지 확인
        assert "시장 포지션" in expanded_query or "위험 신호" in expanded_query
        assert "SoS" in expanded_query

    @pytest.mark.asyncio
    async def test_retriever_query_knowledge_graph(self, mock_knowledge_graph):
        """지식 그래프 쿼리 테스트"""
        retriever = HybridRetriever(
            knowledge_graph=mock_knowledge_graph,
            reasoner=MagicMock(),
            doc_retriever=MagicMock(),
            auto_init_rules=False,
        )

        entities = {"brands": ["laneige"], "categories": ["lip_care"]}

        # 브랜드 정보 mock 설정
        mock_knowledge_graph.get_entity_metadata.return_value = {"sos": 0.08}
        mock_knowledge_graph.get_brand_products.return_value = [
            {"asin": "B08XYZ", "name": "Lip Mask"}
        ]
        mock_knowledge_graph.get_competitors.return_value = [{"brand": "cosrx", "sos": 0.05}]

        facts = retriever._query_knowledge_graph(entities)

        # 브랜드 정보가 조회되었는지 확인
        assert any(f["type"] == "brand_info" for f in facts)
        assert any(f["type"] == "brand_products" for f in facts)
        assert any(f["type"] == "competitors" for f in facts)

    @pytest.mark.asyncio
    async def test_retriever_build_inference_context(self):
        """추론 컨텍스트 구성 테스트"""
        mock_kg = MagicMock()
        mock_kg.get_competitors.return_value = [{"brand": "cosrx", "sos": 0.05}]
        mock_kg.query.return_value = []

        retriever = HybridRetriever(
            knowledge_graph=mock_kg,
            reasoner=MagicMock(),
            doc_retriever=MagicMock(),
            auto_init_rules=False,
        )

        entities = {"brands": ["laneige"], "categories": ["lip_care"], "indicators": ["sos"]}

        current_metrics = {
            "summary": {"laneige_sos_by_category": {"lip_care": 0.08}},
            "brand_metrics": [
                {"is_laneige": True, "share_of_shelf": 0.08, "avg_rank": 15.5, "product_count": 3}
            ],
        }

        context = retriever._build_inference_context(entities, current_metrics)

        assert context["brand"] == "laneige"
        assert context["is_target"] is True
        assert context["category"] == "lip_care"
        assert context["sos"] == 0.08
        assert context["competitor_count"] == 1

    def test_retriever_get_stats(self, mock_knowledge_graph, mock_reasoner, mock_doc_retriever):
        """통계 조회 테스트"""
        retriever = HybridRetriever(
            knowledge_graph=mock_knowledge_graph,
            reasoner=mock_reasoner,
            doc_retriever=mock_doc_retriever,
            auto_init_rules=False,
        )

        stats = retriever.get_stats()

        assert "knowledge_graph" in stats
        assert "reasoner" in stats
        assert "rules_count" in stats
        assert "initialized" in stats
        assert stats["initialized"] is False

    @pytest.mark.asyncio
    async def test_retriever_retrieve_full_flow(
        self, mock_knowledge_graph, mock_reasoner, mock_doc_retriever
    ):
        """전체 검색 플로우 통합 테스트"""
        # Mock 설정
        mock_knowledge_graph.get_entity_metadata.return_value = {"sos": 0.08}
        mock_knowledge_graph.get_brand_products.return_value = []
        mock_knowledge_graph.get_competitors.return_value = []
        mock_knowledge_graph.get_category_brands.return_value = []
        mock_knowledge_graph.query.return_value = []

        mock_reasoner.infer.return_value = [
            InferenceResult(
                rule_name="test_rule",
                insight_type=InsightType.MARKET_POSITION,
                insight="test insight",
                confidence=0.9,
            )
        ]

        mock_doc_retriever.search.return_value = [
            {"id": "doc1", "content": "test content", "metadata": {"title": "Test Doc"}}
        ]

        retriever = HybridRetriever(
            knowledge_graph=mock_knowledge_graph,
            reasoner=mock_reasoner,
            doc_retriever=mock_doc_retriever,
            auto_init_rules=False,
        )

        context = await retriever.retrieve(query="LANEIGE Lip Care 분석", current_metrics={})

        # 결과 검증
        assert context.query == "LANEIGE Lip Care 분석"
        assert len(context.entities) > 0
        assert len(context.inferences) == 1
        assert len(context.rag_chunks) == 1
        assert context.combined_context != ""
        assert "retrieval_time_ms" in context.metadata


# =============================================================================
# Self-RAG Tests
# =============================================================================


class TestSelfRAG:
    """Self-RAG 검색 필요성 판단 테스트"""

    def _make_retriever(self):
        return HybridRetriever(
            knowledge_graph=MagicMock(),
            reasoner=MagicMock(),
            doc_retriever=MagicMock(),
            auto_init_rules=False,
        )

    def test_should_retrieve_greeting_false(self):
        """인사말 → 검색 불필요"""
        retriever = self._make_retriever()
        should, reason, _conf = retriever.should_retrieve("안녕하세요")
        assert should is False
        assert reason == "greeting_or_command"

    def test_should_retrieve_hello_false(self):
        """영어 인사말 → 검색 불필요"""
        retriever = self._make_retriever()
        should, reason, _conf = retriever.should_retrieve("hello")
        assert should is False
        assert reason == "greeting_or_command"

    def test_should_retrieve_thanks_false(self):
        """감사 → 검색 불필요"""
        retriever = self._make_retriever()
        should, reason, _conf = retriever.should_retrieve("고마워요")
        assert should is False
        assert reason == "greeting_or_command"

    def test_should_retrieve_help_false(self):
        """도움말 → 검색 불필요"""
        retriever = self._make_retriever()
        should, reason, _conf = retriever.should_retrieve("도움말")
        assert should is False
        assert reason == "greeting_or_command"

    def test_should_retrieve_brand_query_true(self):
        """브랜드 질문 → 검색 필요"""
        retriever = self._make_retriever()
        should, reason, _conf = retriever.should_retrieve("LANEIGE의 시장 점유율은?")
        assert should is True
        assert reason == "domain_query_detected"

    def test_should_retrieve_metric_query_true(self):
        """지표 질문 → 검색 필요"""
        retriever = self._make_retriever()
        should, reason, _conf = retriever.should_retrieve("SoS 지표 분석해줘")
        assert should is True
        assert reason == "domain_query_detected"

    def test_should_retrieve_analysis_keyword_true(self):
        """분석 키워드 → 검색 필요"""
        retriever = self._make_retriever()
        should, reason, _conf = retriever.should_retrieve("시장 경쟁 비교")
        assert should is True
        assert reason == "domain_query_detected"

    def test_should_retrieve_question_true(self):
        """질문형 → 검색 필요"""
        retriever = self._make_retriever()
        should, reason, _conf = retriever.should_retrieve("왜 순위가 떨어졌어?")
        assert should is True
        assert reason == "domain_query_detected"

    def test_should_retrieve_short_query_false(self):
        """매우 짧은 쿼리 → 검색 불필요"""
        retriever = self._make_retriever()
        should, reason, _conf = retriever.should_retrieve("ㅎ")
        assert should is False
        assert reason == "query_too_short"

    def test_should_retrieve_empty_query_false(self):
        """빈 쿼리 → 검색 불필요"""
        retriever = self._make_retriever()
        should, reason, _conf = retriever.should_retrieve("")
        assert should is False
        assert reason == "query_too_short"

    def test_should_retrieve_none_query_false(self):
        """None 쿼리 → 검색 불필요"""
        retriever = self._make_retriever()
        should, reason, _conf = retriever.should_retrieve(None)
        assert should is False
        assert reason == "query_too_short"

    def test_should_retrieve_cosrx_true(self):
        """COSRX 브랜드 → 검색 필요"""
        retriever = self._make_retriever()
        should, reason, _conf = retriever.should_retrieve("COSRX 최신 데이터")
        assert should is True
        assert reason == "domain_query_detected"

    def test_should_retrieve_default_long_query_true(self):
        """도메인 키워드 없어도 긴 쿼리는 기본 검색"""
        retriever = self._make_retriever()
        should, reason, _conf = retriever.should_retrieve("이것저것 알려주세요")
        assert should is True
        assert reason == "default_retrieve"

    def test_should_retrieve_short_non_domain_false(self):
        """짧고 도메인 관련 없는 쿼리 → 검색 불필요"""
        retriever = self._make_retriever()
        should, reason, _conf = retriever.should_retrieve("ㅋㅋㅋ")
        assert should is False
        assert reason == "short_non_domain_query"

    @pytest.mark.asyncio
    async def test_self_rag_gate_skips_retrieval(self):
        """should_retrieve=False 시 검색 스킵 통합 테스트"""
        mock_kg = MagicMock()
        mock_kg.load_category_hierarchy.return_value = 0

        mock_doc_retriever = MagicMock()
        mock_doc_retriever.initialize = AsyncMock()
        mock_doc_retriever.search = AsyncMock(return_value=[])

        retriever = HybridRetriever(
            knowledge_graph=mock_kg,
            reasoner=MagicMock(),
            doc_retriever=mock_doc_retriever,
            auto_init_rules=False,
        )

        context = await retriever.retrieve("안녕하세요", current_metrics={})

        # 검색이 스킵되어야 함
        assert context.metadata.get("self_rag_skip") is True
        assert context.metadata.get("skip_reason") == "greeting_or_command"
        assert context.rag_chunks == []
        assert context.inferences == []
        # doc_retriever.search가 호출되지 않아야 함
        mock_doc_retriever.search.assert_not_called()

    @pytest.mark.asyncio
    async def test_self_rag_gate_allows_domain_query(self):
        """도메인 쿼리는 정상 검색 수행"""
        mock_kg = MagicMock()
        mock_kg.get_entity_metadata.return_value = {}
        mock_kg.get_brand_products.return_value = []
        mock_kg.get_competitors.return_value = []
        mock_kg.query.return_value = []
        mock_kg.load_category_hierarchy.return_value = 0

        mock_doc_retriever = MagicMock()
        mock_doc_retriever.initialize = AsyncMock()
        mock_doc_retriever.search = AsyncMock(return_value=[])

        retriever = HybridRetriever(
            knowledge_graph=mock_kg,
            reasoner=MagicMock(),
            doc_retriever=mock_doc_retriever,
            auto_init_rules=False,
        )

        context = await retriever.retrieve("LANEIGE 점유율 분석", current_metrics={})

        # 정상 검색이 수행되어야 함
        assert context.metadata.get("self_rag_skip") is not True
        assert "retrieval_time_ms" in context.metadata


# =============================================================================
# Entity Extractor Rank Pattern Tests
# =============================================================================


class TestEntityExtractorRankPattern:
    """순위 패턴 기반 제품 추출 테스트"""

    def test_extract_rank_pattern_korean(self):
        """한국어 순위 패턴 추출 테스트"""
        mock_kg = MagicMock()
        mock_relation = MagicMock()
        mock_relation.subject = "B08TEST123"
        mock_relation.properties = {"rank": 3}
        mock_kg.query.return_value = [mock_relation]

        extractor = EntityExtractor()
        query = "립케어 3위 제품 보여줘"
        entities = extractor.extract(query, knowledge_graph=mock_kg)

        # 카테고리와 순위 패턴 모두 추출되어야 함
        assert "lip_care" in entities["categories"]
        # KG 쿼리가 호출되어 제품이 추출되어야 함
        assert "B08TEST123" in entities["products"]

    def test_extract_rank_pattern_english(self):
        """영어 순위 패턴 추출 테스트"""
        mock_kg = MagicMock()
        mock_relation = MagicMock()
        mock_relation.subject = "B08RANK001"
        mock_relation.properties = {"rank": 5}
        mock_kg.query.return_value = [mock_relation]

        extractor = EntityExtractor()
        query = "show me rank 5 product in lip care"
        entities = extractor.extract(query, knowledge_graph=mock_kg)

        assert "lip_care" in entities["categories"]
        assert "B08RANK001" in entities["products"]

    def test_extract_rank_pattern_no_kg(self):
        """지식 그래프 없을 때 순위 패턴 무시 테스트"""
        extractor = EntityExtractor()
        query = "립케어 3위 제품"
        entities = extractor.extract(query, knowledge_graph=None)

        # 카테고리는 추출되지만 제품은 추출되지 않음
        assert "lip_care" in entities["categories"]
        assert len(entities["products"]) == 0


# =============================================================================
# Query Knowledge Graph Extended Tests
# =============================================================================


class TestQueryKnowledgeGraphExtended:
    """지식 그래프 쿼리 확장 테스트"""

    def test_query_kg_trend_keywords(self, mock_knowledge_graph):
        """트렌드 키워드 사실 조회 테스트"""
        # 브랜드별 트렌드 설정
        mock_relation = MagicMock()
        mock_relation.object = "peptide"
        mock_knowledge_graph.query.return_value = [mock_relation]

        retriever = HybridRetriever(
            knowledge_graph=mock_knowledge_graph,
            reasoner=MagicMock(),
            doc_retriever=MagicMock(),
            auto_init_rules=False,
        )

        entities = {"brands": ["laneige"]}
        facts = retriever._query_knowledge_graph(entities)

        # 트렌드 키워드 사실이 포함되어야 함
        trend_facts = [f for f in facts if f["type"] == "trend_keywords"]
        assert len(trend_facts) > 0
        assert "peptide" in trend_facts[0]["data"]["keywords"]

    def test_query_kg_category_hierarchy(self, mock_knowledge_graph):
        """카테고리 계층 사실 조회 테스트"""
        mock_knowledge_graph.get_category_hierarchy.return_value = {
            "name": "Lip Care",
            "level": 2,
            "path": ["Beauty", "Skin Care", "Lip Care"],
            "ancestors": [{"name": "Skin Care", "level": 1}],
            "descendants": [],
        }

        retriever = HybridRetriever(
            knowledge_graph=mock_knowledge_graph,
            reasoner=MagicMock(),
            doc_retriever=MagicMock(),
            auto_init_rules=False,
        )

        entities = {"categories": ["lip_care"]}
        facts = retriever._query_knowledge_graph(entities)

        # 계층 정보가 포함되어야 함
        hierarchy_facts = [f for f in facts if f["type"] == "category_hierarchy"]
        assert len(hierarchy_facts) > 0
        assert hierarchy_facts[0]["data"]["level"] == 2
        assert "Lip Care" in hierarchy_facts[0]["data"]["path"]

    def test_query_kg_sentiment_products(self, mock_knowledge_graph):
        """감성 기반 제품 검색 테스트"""
        mock_knowledge_graph.find_products_by_sentiment.return_value = [
            {"asin": "B08MOIST01", "name": "Hydrating Cream"}
        ]

        retriever = HybridRetriever(
            knowledge_graph=mock_knowledge_graph,
            reasoner=MagicMock(),
            doc_retriever=MagicMock(),
            auto_init_rules=False,
        )

        entities = {"sentiment_clusters": ["Hydration"]}
        facts = retriever._query_knowledge_graph(entities)

        # 감성 제품 사실이 포함되어야 함
        sentiment_facts = [f for f in facts if f["type"] == "sentiment_products"]
        assert len(sentiment_facts) > 0

    def test_query_kg_brand_sentiment_profile(self, mock_knowledge_graph):
        """브랜드 감성 프로필 조회 테스트"""
        mock_knowledge_graph.get_brand_sentiment_profile.return_value = {
            "all_tags": ["moisturizing", "hydrating"],
            "clusters": {"Hydration": 25, "Effectiveness": 15},
            "dominant_sentiment": "Hydration",
        }

        retriever = HybridRetriever(
            knowledge_graph=mock_knowledge_graph,
            reasoner=MagicMock(),
            doc_retriever=MagicMock(),
            auto_init_rules=False,
        )

        entities = {"brands": ["laneige"], "sentiments": ["보습"]}
        facts = retriever._query_knowledge_graph(entities)

        # 브랜드 감성 사실이 포함되어야 함
        brand_sentiment_facts = [f for f in facts if f["type"] == "brand_sentiment"]
        assert len(brand_sentiment_facts) > 0
        assert brand_sentiment_facts[0]["data"]["dominant_sentiment"] == "Hydration"

    def test_query_kg_product_sentiments(self, mock_knowledge_graph):
        """제품 감성 조회 테스트"""
        mock_knowledge_graph.get_product_sentiments.return_value = {
            "sentiment_tags": ["hydrating", "smooth"],
            "ai_summary": "Great moisturizing product",
        }

        retriever = HybridRetriever(
            knowledge_graph=mock_knowledge_graph,
            reasoner=MagicMock(),
            doc_retriever=MagicMock(),
            auto_init_rules=False,
        )

        entities = {"products": ["B08TEST123"], "sentiments": ["보습"]}
        facts = retriever._query_knowledge_graph(entities)

        # 제품 감성 사실이 포함되어야 함
        product_sentiment_facts = [f for f in facts if f["type"] == "product_sentiment"]
        assert len(product_sentiment_facts) > 0

    def test_query_kg_competitor_network_exception(self, mock_knowledge_graph):
        """경쟁사 네트워크 조회 예외 처리 테스트"""
        mock_knowledge_graph.get_neighbors.side_effect = Exception("Network error")

        retriever = HybridRetriever(
            knowledge_graph=mock_knowledge_graph,
            reasoner=MagicMock(),
            doc_retriever=MagicMock(),
            auto_init_rules=False,
        )

        entities = {"brands": ["laneige"]}
        # 예외가 발생해도 다른 사실은 조회되어야 함
        facts = retriever._query_knowledge_graph(entities)

        # competitor_network 사실은 없어야 함
        network_facts = [f for f in facts if f["type"] == "competitor_network"]
        assert len(network_facts) == 0


# =============================================================================
# Build Inference Context Sentiment Tests
# =============================================================================


class TestBuildInferenceContextSentiment:
    """추론 컨텍스트 감성 데이터 테스트"""

    def test_build_context_with_trend_keywords(self):
        """트렌드 키워드 컨텍스트 구성 테스트"""
        mock_kg = MagicMock()
        mock_relation = MagicMock()
        mock_relation.object = "glass_skin"
        mock_kg.query.return_value = [mock_relation]
        mock_kg.get_competitors.return_value = []

        retriever = HybridRetriever(
            knowledge_graph=mock_kg,
            reasoner=MagicMock(),
            doc_retriever=MagicMock(),
            auto_init_rules=False,
        )

        entities = {"brands": ["laneige"]}
        context = retriever._build_inference_context(entities, {})

        assert "trend_keywords" in context
        assert "glass_skin" in context["trend_keywords"]

    def test_build_context_with_brand_sentiment(self):
        """브랜드 감성 프로필 컨텍스트 구성 테스트"""
        mock_kg = MagicMock()
        mock_kg.get_competitors.return_value = []
        mock_kg.query.return_value = []
        mock_kg.get_brand_sentiment_profile.return_value = {
            "all_tags": ["hydrating", "soothing"],
            "clusters": {"Hydration": 30, "Effectiveness": 20},
            "dominant_sentiment": "Hydration",
        }

        retriever = HybridRetriever(
            knowledge_graph=mock_kg,
            reasoner=MagicMock(),
            doc_retriever=MagicMock(),
            auto_init_rules=False,
        )

        entities = {"brands": ["laneige"], "sentiments": ["보습"]}
        context = retriever._build_inference_context(entities, {})

        assert "sentiment_tags" in context
        assert "hydrating" in context["sentiment_tags"]
        assert context["dominant_sentiment"] == "Hydration"

    def test_build_context_with_product_sentiment(self):
        """제품 감성 컨텍스트 구성 테스트"""
        mock_kg = MagicMock()
        mock_kg.get_competitors.return_value = []
        mock_kg.query.return_value = []
        mock_kg.get_brand_sentiment_profile.return_value = {}
        mock_kg.get_product_sentiments.return_value = {
            "sentiment_tags": ["moisturizing", "gentle"],
            "ai_summary": "Excellent hydration",
            "sentiment_clusters": {"Hydration": 5},
        }

        retriever = HybridRetriever(
            knowledge_graph=mock_kg,
            reasoner=MagicMock(),
            doc_retriever=MagicMock(),
            auto_init_rules=False,
        )

        entities = {"brands": ["laneige"], "sentiments": ["보습"]}
        current_metrics = {
            "product_metrics": [{"asin": "B08TEST123", "current_rank": 10}],
        }
        context = retriever._build_inference_context(entities, current_metrics)

        assert "ai_summary" in context
        assert context["ai_summary"] == "Excellent hydration"

    def test_build_context_with_competitor_sentiment(self):
        """경쟁사 감성 비교 컨텍스트 구성 테스트"""
        mock_kg = MagicMock()
        mock_kg.get_competitors.return_value = [
            {"brand": "cosrx"},
            {"brand": "tirtir"},
        ]
        mock_kg.query.return_value = []
        mock_kg.get_brand_sentiment_profile.side_effect = [
            {  # laneige
                "all_tags": ["hydrating"],
                "clusters": {"Hydration": 30},
            },
            {  # cosrx
                "all_tags": ["affordable", "gentle"],
                "clusters": {"Pricing": 25, "Skin_Compatibility": 15},
            },
            {  # tirtir
                "all_tags": ["trendy"],
                "clusters": {"Effectiveness": 20},
            },
        ]

        retriever = HybridRetriever(
            knowledge_graph=mock_kg,
            reasoner=MagicMock(),
            doc_retriever=MagicMock(),
            auto_init_rules=False,
        )

        entities = {"brands": ["laneige"], "sentiments": ["보습"]}
        context = retriever._build_inference_context(entities, {})

        assert "competitor_sentiment_tags" in context
        assert "affordable" in context["competitor_sentiment_tags"]
        assert "competitor_sentiment_clusters" in context
        assert context["competitor_sentiment_clusters"]["Pricing"] == 25


# =============================================================================
# Rewrite For Relevance Tests
# =============================================================================


class TestRewriteForRelevance:
    """관련성 개선을 위한 쿼리 재작성 테스트"""

    def test_rewrite_with_indicators(self):
        """지표 추가 재작성 테스트"""
        retriever = HybridRetriever(
            knowledge_graph=MagicMock(),
            reasoner=MagicMock(),
            doc_retriever=MagicMock(),
            auto_init_rules=False,
        )

        query = "LANEIGE 분석"
        entities = {"brands": ["laneige"], "indicators": ["sos", "hhi"]}

        rewritten = retriever._rewrite_for_relevance(query, entities)

        # 지표 전체 이름이 추가되어야 함
        assert "Share of Shelf" in rewritten or "점유율" in rewritten
        assert "HHI" in rewritten or "시장집중도" in rewritten

    def test_rewrite_with_categories(self):
        """카테고리 추가 재작성 테스트"""
        retriever = HybridRetriever(
            knowledge_graph=MagicMock(),
            reasoner=MagicMock(),
            doc_retriever=MagicMock(),
            auto_init_rules=False,
        )

        query = "브랜드 경쟁력"
        entities = {"categories": ["lip_care"]}

        rewritten = retriever._rewrite_for_relevance(query, entities)

        # 카테고리 전체 이름이 추가되어야 함
        assert "Lip Care" in rewritten or "립케어" in rewritten

    def test_rewrite_no_change_if_already_present(self):
        """이미 포함된 용어는 추가하지 않음 테스트"""
        retriever = HybridRetriever(
            knowledge_graph=MagicMock(),
            reasoner=MagicMock(),
            doc_retriever=MagicMock(),
            auto_init_rules=False,
        )

        query = "LANEIGE Share of Shelf Lip Care 분석"
        entities = {"brands": ["laneige"], "indicators": ["sos"], "categories": ["lip_care"]}

        rewritten = retriever._rewrite_for_relevance(query, entities)

        # 변경되지 않아야 함 (이미 모든 용어 포함)
        assert rewritten == query or rewritten.count("LANEIGE") == 1


# =============================================================================
# Weighted Merge Tests
# =============================================================================


class TestWeightedMerge:
    """가중치 기반 병합 테스트"""

    def test_weighted_merge_ontology_facts_scoring(self):
        """온톨로지 사실 점수 할당 테스트"""
        retriever = HybridRetriever(
            knowledge_graph=MagicMock(),
            reasoner=MagicMock(),
            doc_retriever=MagicMock(),
            auto_init_rules=False,
        )

        context = HybridContext(
            query="test",
            ontology_facts=[
                {"type": "brand_info", "entity": "laneige"},
                {"type": "category_brands", "entity": "lip_care"},
                {"type": "trend_keywords", "entity": "laneige"},
            ],
        )

        merged = retriever._weighted_merge(context)

        # 점수가 할당되어야 함
        assert all("_weighted_score" in fact for fact in merged.ontology_facts)
        # brand_info > category_brands > trend_keywords 순으로 점수가 높아야 함
        scores = [f["_weighted_score"] for f in merged.ontology_facts]
        assert scores[0] >= scores[1] >= scores[2]

    def test_weighted_merge_rag_chunks_scoring(self):
        """RAG 청크 점수 할당 테스트"""
        retriever = HybridRetriever(
            knowledge_graph=MagicMock(),
            reasoner=MagicMock(),
            doc_retriever=MagicMock(),
            auto_init_rules=False,
        )

        context = HybridContext(
            query="test",
            rag_chunks=[
                {"id": "1", "score": 0.9, "metadata": {"doc_type": "intelligence"}},
                {"id": "2", "score": 0.8, "metadata": {"doc_type": "playbook"}},
                {"id": "3", "score": 0.7, "metadata": {"doc_type": "metric_guide"}},
            ],
        )

        merged = retriever._weighted_merge(context)

        # 점수가 할당되어야 함
        assert all("_weighted_score" in chunk for chunk in merged.rag_chunks)
        # intelligence (weekly) > playbook (quarterly) freshness 반영
        assert merged.rag_chunks[0]["metadata"]["doc_type"] == "intelligence"

    def test_weighted_merge_truncation(self):
        """최대 항목 수 제한 테스트"""
        retriever = HybridRetriever(
            knowledge_graph=MagicMock(),
            reasoner=MagicMock(),
            doc_retriever=MagicMock(),
            auto_init_rules=False,
        )

        # 많은 사실 생성 (max_items 초과)
        many_facts = [{"type": "brand_info", "entity": f"brand_{i}"} for i in range(10)]

        context = HybridContext(query="test", ontology_facts=many_facts)

        merged = retriever._weighted_merge(context)

        # 최대 5개로 제한되어야 함
        assert len(merged.ontology_facts) <= 5


# =============================================================================
# Combine Contexts Tests
# =============================================================================


class TestCombineContexts:
    """컨텍스트 통합 포매팅 테스트"""

    def test_combine_contexts_with_inferences(self):
        """추론 결과 포매팅 테스트"""
        retriever = HybridRetriever(
            knowledge_graph=MagicMock(),
            reasoner=MagicMock(),
            doc_retriever=MagicMock(),
            auto_init_rules=False,
        )

        inference = InferenceResult(
            rule_name="test_rule",
            insight_type=InsightType.MARKET_POSITION,
            insight="Dominant position",
            confidence=0.9,
            recommendation="Maintain leadership",
            evidence={"satisfied_conditions": ["sos > 0.15"]},
        )

        context = HybridContext(query="test", inferences=[inference])

        combined = retriever._combine_contexts(context, include_explanations=True)

        assert "## 분석 결과" in combined
        assert "Dominant position" in combined
        assert "Maintain leadership" in combined
        assert "90%" in combined
        assert "sos > 0.15" in combined

    def test_combine_contexts_brand_info_facts(self):
        """브랜드 정보 사실 포매팅 테스트"""
        retriever = HybridRetriever(
            knowledge_graph=MagicMock(),
            reasoner=MagicMock(),
            doc_retriever=MagicMock(),
            auto_init_rules=False,
        )

        context = HybridContext(
            query="test",
            ontology_facts=[
                {
                    "type": "brand_info",
                    "entity": "laneige",
                    "data": {"sos": 0.08, "avg_rank": 12.5},
                }
            ],
        )

        combined = retriever._combine_contexts(context)

        assert "## 관련 정보" in combined
        assert "laneige" in combined
        assert "8.0%" in combined
        assert "12.5" in combined

    def test_combine_contexts_brand_products_facts(self):
        """브랜드 제품 사실 포매팅 테스트"""
        retriever = HybridRetriever(
            knowledge_graph=MagicMock(),
            reasoner=MagicMock(),
            doc_retriever=MagicMock(),
            auto_init_rules=False,
        )

        context = HybridContext(
            query="test",
            ontology_facts=[
                {"type": "brand_products", "entity": "laneige", "data": {"product_count": 15}}
            ],
        )

        combined = retriever._combine_contexts(context)

        assert "15개" in combined

    def test_combine_contexts_competitors_facts(self):
        """경쟁사 사실 포매팅 테스트"""
        retriever = HybridRetriever(
            knowledge_graph=MagicMock(),
            reasoner=MagicMock(),
            doc_retriever=MagicMock(),
            auto_init_rules=False,
        )

        context = HybridContext(
            query="test",
            ontology_facts=[
                {
                    "type": "competitors",
                    "entity": "laneige",
                    "data": [{"brand": "cosrx"}, {"brand": "tirtir"}],
                }
            ],
        )

        combined = retriever._combine_contexts(context)

        assert "경쟁사" in combined
        assert "cosrx" in combined

    def test_combine_contexts_category_hierarchy_facts(self):
        """카테고리 계층 사실 포매팅 테스트"""
        retriever = HybridRetriever(
            knowledge_graph=MagicMock(),
            reasoner=MagicMock(),
            doc_retriever=MagicMock(),
            auto_init_rules=False,
        )

        context = HybridContext(
            query="test",
            ontology_facts=[
                {
                    "type": "category_hierarchy",
                    "entity": "lip_care",
                    "data": {
                        "name": "Lip Care",
                        "level": 2,
                        "path": [
                            {"name": "Beauty", "id": "beauty"},
                            {"name": "Skin Care", "id": "skin_care"},
                            {"name": "Lip Care", "id": "lip_care"},
                        ],
                        "ancestors": [{"name": "Skin Care"}],
                    },
                }
            ],
        )

        combined = retriever._combine_contexts(context)

        assert "계층" in combined
        assert "Beauty > Skin Care > Lip Care" in combined
        assert "Level 2" in combined

    def test_combine_contexts_rag_chunks(self):
        """RAG 청크 포매팅 테스트"""
        retriever = HybridRetriever(
            knowledge_graph=MagicMock(),
            reasoner=MagicMock(),
            doc_retriever=MagicMock(),
            auto_init_rules=False,
        )

        context = HybridContext(
            query="test",
            rag_chunks=[
                {
                    "metadata": {"title": "SoS 해석 가이드"},
                    "content": "SoS는 시장 점유율을 나타내는 지표입니다.",
                }
            ],
        )

        combined = retriever._combine_contexts(context)

        assert "## 참고 가이드라인" in combined
        assert "SoS 해석 가이드" in combined
        assert "시장 점유율" in combined

    def test_combine_contexts_long_content_truncation(self):
        """긴 내용 잘림 테스트"""
        retriever = HybridRetriever(
            knowledge_graph=MagicMock(),
            reasoner=MagicMock(),
            doc_retriever=MagicMock(),
            auto_init_rules=False,
        )

        long_content = "X" * 600  # 500자 초과

        context = HybridContext(
            query="test",
            rag_chunks=[{"metadata": {"title": "Test"}, "content": long_content}],
        )

        combined = retriever._combine_contexts(context)

        # 500자로 잘리고 "..."이 붙어야 함
        assert "..." in combined
        assert combined.count("X") <= 510  # 약간의 여유


# =============================================================================
# Retrieve For Entity Tests
# =============================================================================


class TestRetrieveForEntity:
    """엔티티별 검색 테스트"""

    @pytest.mark.asyncio
    async def test_retrieve_for_brand_entity(self):
        """브랜드 엔티티 검색 테스트"""
        mock_kg = MagicMock()
        mock_kg.get_entity_metadata.return_value = {}
        mock_kg.get_brand_products.return_value = []
        mock_kg.get_competitors.return_value = []
        mock_kg.get_category_brands.return_value = []
        mock_kg.query.return_value = []

        mock_doc_retriever = MagicMock()
        mock_doc_retriever.initialize = AsyncMock()
        mock_doc_retriever.search = AsyncMock(return_value=[])

        retriever = HybridRetriever(
            knowledge_graph=mock_kg,
            reasoner=MagicMock(),
            doc_retriever=mock_doc_retriever,
            auto_init_rules=False,
        )

        context = await retriever.retrieve_for_entity("LANEIGE", entity_type="brand")

        assert context.query == "LANEIGE 브랜드 분석"
        assert "laneige" in context.entities.get("brands", [])

    @pytest.mark.asyncio
    async def test_retrieve_for_product_entity(self):
        """제품 엔티티 검색 테스트"""
        mock_kg = MagicMock()
        mock_kg.get_entity_metadata.return_value = {}
        mock_kg.get_brand_products.return_value = []
        mock_kg.get_competitors.return_value = []
        mock_kg.query.return_value = []

        mock_doc_retriever = MagicMock()
        mock_doc_retriever.initialize = AsyncMock()
        mock_doc_retriever.search = AsyncMock(return_value=[])

        retriever = HybridRetriever(
            knowledge_graph=mock_kg,
            reasoner=MagicMock(),
            doc_retriever=mock_doc_retriever,
            auto_init_rules=False,
        )

        context = await retriever.retrieve_for_entity("B08XYZ1234", entity_type="product")

        assert context.query == "B08XYZ1234 제품 분석"
        assert "B08XYZ1234" in context.entities.get("products", [])

    @pytest.mark.asyncio
    async def test_retrieve_for_category_entity(self):
        """카테고리 엔티티 검색 테스트"""
        mock_kg = MagicMock()
        mock_kg.get_entity_metadata.return_value = {}
        mock_kg.get_category_brands.return_value = []
        mock_kg.query.return_value = []

        mock_doc_retriever = MagicMock()
        mock_doc_retriever.initialize = AsyncMock()
        mock_doc_retriever.search = AsyncMock(return_value=[])

        retriever = HybridRetriever(
            knowledge_graph=mock_kg,
            reasoner=MagicMock(),
            doc_retriever=mock_doc_retriever,
            auto_init_rules=False,
        )

        context = await retriever.retrieve_for_entity("lip_care", entity_type="category")

        assert context.query == "lip_care 카테고리 분석"
        assert "lip_care" in context.entities.get("categories", [])

    @pytest.mark.asyncio
    async def test_retrieve_for_unknown_entity_type(self):
        """알 수 없는 엔티티 타입 테스트"""
        mock_kg = MagicMock()
        mock_kg.get_entity_metadata.return_value = {}
        mock_kg.query.return_value = []

        mock_doc_retriever = MagicMock()
        mock_doc_retriever.initialize = AsyncMock()
        mock_doc_retriever.search = AsyncMock(return_value=[])

        retriever = HybridRetriever(
            knowledge_graph=mock_kg,
            reasoner=MagicMock(),
            doc_retriever=mock_doc_retriever,
            auto_init_rules=False,
        )

        context = await retriever.retrieve_for_entity("unknown", entity_type="unknown_type")

        assert context.query == "unknown 분석"
        assert context.entities.get("brands", []) == []


# =============================================================================
# Retrieve Edge Cases Tests
# =============================================================================


class TestRetrieveEdgeCases:
    """검색 엣지 케이스 테스트"""

    @pytest.mark.asyncio
    async def test_retrieve_additional_search_when_filtered_results_low(self):
        """필터링된 결과가 부족할 때 추가 검색 테스트"""
        mock_kg = MagicMock()
        mock_kg.get_entity_metadata.return_value = {}
        mock_kg.get_brand_products.return_value = []
        mock_kg.get_competitors.return_value = []
        mock_kg.query.return_value = []

        mock_doc_retriever = MagicMock()
        mock_doc_retriever.initialize = AsyncMock()
        # 첫 검색에서 2개만 반환, 두 번째 검색에서 3개 추가
        mock_doc_retriever.search = AsyncMock(
            side_effect=[
                [{"id": "1", "content": "test1"}, {"id": "2", "content": "test2"}],  # 필터링된 검색
                [
                    {"id": "3", "content": "test3"},
                    {"id": "4", "content": "test4"},
                    {"id": "5", "content": "test5"},
                ],  # 전체 문서 검색
            ]
        )
        # BM25 returns empty so _hybrid_search uses dense_only path
        mock_doc_retriever.search_bm25 = MagicMock(return_value=[])

        retriever = HybridRetriever(
            knowledge_graph=mock_kg,
            reasoner=MagicMock(),
            doc_retriever=mock_doc_retriever,
            auto_init_rules=False,
        )

        # 관련성 판정을 우회하도록 설정
        retriever.relevance_grader.grade_documents = AsyncMock(
            return_value=(
                [{"id": "1", "content": "test1"}, {"id": "2", "content": "test2"}],  # relevant
                [],  # irrelevant
            )
        )
        retriever.relevance_grader.needs_rewrite = MagicMock(return_value=False)

        context = await retriever.retrieve("왜 순위가 떨어졌어", current_metrics={})

        # 총 5개 문서가 반환되어야 함
        assert len(context.rag_chunks) >= 2

    @pytest.mark.asyncio
    async def test_retrieve_relevance_grading_failure_fallback(self):
        """관련성 판정 실패 시 폴백 테스트"""
        mock_kg = MagicMock()
        mock_kg.get_entity_metadata.return_value = {}
        mock_kg.get_brand_products.return_value = []
        mock_kg.get_competitors.return_value = []
        mock_kg.query.return_value = []

        mock_doc_retriever = MagicMock()
        mock_doc_retriever.initialize = AsyncMock()
        mock_doc_retriever.search = AsyncMock(return_value=[{"id": "1", "content": "test"}])
        # BM25 returns empty so _hybrid_search uses dense_only path
        mock_doc_retriever.search_bm25 = MagicMock(return_value=[])

        retriever = HybridRetriever(
            knowledge_graph=mock_kg,
            reasoner=MagicMock(),
            doc_retriever=mock_doc_retriever,
            auto_init_rules=False,
        )

        # 관련성 판정기를 예외 발생하도록 설정
        retriever.relevance_grader.grade_documents = AsyncMock(
            side_effect=Exception("Grading failed")
        )

        context = await retriever.retrieve("test query", current_metrics={})

        # 원본 결과가 유지되어야 함
        assert len(context.rag_chunks) == 1

    @pytest.mark.asyncio
    async def test_retrieve_rag_metrics_recording_failure(self):
        """RAG 메트릭 기록 실패 테스트"""
        mock_kg = MagicMock()
        mock_kg.get_entity_metadata.return_value = {}
        mock_kg.get_brand_products.return_value = []
        mock_kg.get_competitors.return_value = []
        mock_kg.query.return_value = []

        mock_doc_retriever = MagicMock()
        mock_doc_retriever.initialize = AsyncMock()
        mock_doc_retriever.search = AsyncMock(return_value=[])

        retriever = HybridRetriever(
            knowledge_graph=mock_kg,
            reasoner=MagicMock(),
            doc_retriever=mock_doc_retriever,
            auto_init_rules=False,
        )

        # 메트릭 기록을 실패하도록 설정
        retriever.rag_metrics.record_retrieval = MagicMock(side_effect=Exception("Metrics failed"))

        # 예외가 발생해도 검색은 성공해야 함
        context = await retriever.retrieve("test query", current_metrics={})

        assert context.query == "test query"

    @pytest.mark.asyncio
    async def test_retrieve_general_exception_handling(self):
        """일반 예외 처리 테스트"""
        mock_kg = MagicMock()
        mock_kg.get_entity_metadata.side_effect = Exception("KG failure")

        mock_doc_retriever = MagicMock()
        mock_doc_retriever.initialize = AsyncMock()

        retriever = HybridRetriever(
            knowledge_graph=mock_kg,
            reasoner=MagicMock(),
            doc_retriever=mock_doc_retriever,
            auto_init_rules=False,
        )

        context = await retriever.retrieve("test query", current_metrics={})

        # 에러가 메타데이터에 기록되어야 함
        assert "error" in context.metadata

    @pytest.mark.asyncio
    async def test_retrieve_unified_with_dict_inferences(self):
        """딕셔너리 추론 결과 처리 테스트"""
        mock_kg = MagicMock()
        mock_kg.get_entity_metadata.return_value = {}
        mock_kg.get_brand_products.return_value = []
        mock_kg.get_competitors.return_value = []
        mock_kg.query.return_value = []

        mock_doc_retriever = MagicMock()
        mock_doc_retriever.initialize = AsyncMock()
        mock_doc_retriever.search = AsyncMock(return_value=[])

        # 딕셔너리를 반환하는 reasoner
        mock_reasoner = MagicMock()
        mock_reasoner.infer.return_value = [
            {"insight": "test insight", "confidence": 0.9}  # dict 형태
        ]

        retriever = HybridRetriever(
            knowledge_graph=mock_kg,
            reasoner=mock_reasoner,
            doc_retriever=mock_doc_retriever,
            auto_init_rules=False,
        )

        result = await retriever.retrieve_unified("test query", current_metrics={})

        # 딕셔너리가 그대로 포함되어야 함
        assert len(result.inferences) == 1
        assert isinstance(result.inferences[0], dict)


# =============================================================================
# Initialize Edge Cases Tests
# =============================================================================


class TestInitializeEdgeCases:
    """초기화 엣지 케이스 테스트"""

    @pytest.mark.asyncio
    async def test_initialize_category_hierarchy_load_exception(self):
        """카테고리 계층 로드 예외 테스트"""
        mock_kg = MagicMock()
        mock_kg.load_category_hierarchy.side_effect = Exception("Hierarchy load failed")

        mock_doc_retriever = MagicMock()
        mock_doc_retriever.initialize = AsyncMock()

        retriever = HybridRetriever(
            knowledge_graph=mock_kg,
            reasoner=MagicMock(),
            doc_retriever=mock_doc_retriever,
            auto_init_rules=False,
        )

        # 예외가 발생해도 초기화는 성공해야 함
        await retriever.initialize()

        assert retriever._initialized is True


# =============================================================================
# Update Knowledge Graph Tests
# =============================================================================


class TestUpdateKnowledgeGraph:
    """지식 그래프 업데이트 테스트"""

    def test_update_kg_with_crawl_data(self):
        """크롤링 데이터로 KG 업데이트 테스트"""
        mock_kg = MagicMock()
        mock_kg.load_from_crawl_data.return_value = 50

        retriever = HybridRetriever(
            knowledge_graph=mock_kg,
            reasoner=MagicMock(),
            doc_retriever=MagicMock(),
            auto_init_rules=False,
        )

        crawl_data = {"products": [{"asin": "B08TEST", "brand": "LANEIGE"}]}
        stats = retriever.update_knowledge_graph(crawl_data=crawl_data)

        assert stats["crawl_relations"] == 50
        mock_kg.load_from_crawl_data.assert_called_once_with(crawl_data)

    def test_update_kg_with_metrics_data(self):
        """메트릭 데이터로 KG 업데이트 테스트"""
        mock_kg = MagicMock()
        mock_kg.load_from_metrics_data.return_value = 30

        retriever = HybridRetriever(
            knowledge_graph=mock_kg,
            reasoner=MagicMock(),
            doc_retriever=MagicMock(),
            auto_init_rules=False,
        )

        metrics_data = {"brand_metrics": [{"brand": "laneige", "sos": 0.08}]}
        stats = retriever.update_knowledge_graph(metrics_data=metrics_data)

        assert stats["metrics_relations"] == 30
        mock_kg.load_from_metrics_data.assert_called_once_with(metrics_data)

    def test_update_kg_with_both_data(self):
        """크롤링과 메트릭 데이터 모두로 KG 업데이트 테스트"""
        mock_kg = MagicMock()
        mock_kg.load_from_crawl_data.return_value = 50
        mock_kg.load_from_metrics_data.return_value = 30

        retriever = HybridRetriever(
            knowledge_graph=mock_kg,
            reasoner=MagicMock(),
            doc_retriever=MagicMock(),
            auto_init_rules=False,
        )

        crawl_data = {"products": []}
        metrics_data = {"brand_metrics": []}
        stats = retriever.update_knowledge_graph(crawl_data=crawl_data, metrics_data=metrics_data)

        assert stats["crawl_relations"] == 50
        assert stats["metrics_relations"] == 30


# =============================================================================
# OWL Strategy Tests
# =============================================================================


class TestOWLStrategy:
    """OWL 전략 통합 테스트"""

    @pytest.mark.asyncio
    async def test_retrieve_unified_with_owl_strategy(self):
        """OWL strategy가 설정된 경우 retrieve_unified 위임 테스트"""
        from src.domain.value_objects.retrieval_result import UnifiedRetrievalResult

        mock_owl_strategy = MagicMock()
        mock_result = UnifiedRetrievalResult(
            query="test",
            entities={},
            ontology_facts=[],
            inferences=[],
            rag_chunks=[],
            combined_context="OWL result",
            confidence=0.95,
            entity_links=[],
            metadata={},
            retriever_type="owl",
        )
        mock_owl_strategy.retrieve = AsyncMock(return_value=mock_result)

        retriever = HybridRetriever(
            knowledge_graph=MagicMock(),
            reasoner=MagicMock(),
            doc_retriever=MagicMock(),
            auto_init_rules=False,
            owl_strategy=mock_owl_strategy,
        )

        result = await retriever.retrieve_unified("test query", current_metrics={}, top_k=5)

        assert result.combined_context == "OWL result"
        assert result.retriever_type == "owl"
        mock_owl_strategy.retrieve.assert_called_once()

    @pytest.mark.asyncio
    async def test_search_with_owl_strategy(self):
        """OWL strategy가 설정된 경우 search 위임 테스트"""
        mock_owl_strategy = MagicMock()
        mock_owl_strategy.search = AsyncMock(return_value=[{"id": "owl_doc", "content": "test"}])

        retriever = HybridRetriever(
            knowledge_graph=MagicMock(),
            reasoner=MagicMock(),
            doc_retriever=MagicMock(),
            auto_init_rules=False,
            owl_strategy=mock_owl_strategy,
        )

        results = await retriever.search("test query", top_k=5, doc_filter="test")

        assert len(results) == 1
        assert results[0]["id"] == "owl_doc"
        mock_owl_strategy.search.assert_called_once_with(
            query="test query", top_k=5, doc_filter="test"
        )

    @pytest.mark.asyncio
    async def test_search_without_owl_strategy(self):
        """OWL strategy 없이 search 호출 시 doc_retriever 사용 테스트"""
        mock_doc_retriever = MagicMock()
        mock_doc_retriever.search = AsyncMock(return_value=[{"id": "doc1", "content": "test"}])

        retriever = HybridRetriever(
            knowledge_graph=MagicMock(),
            reasoner=MagicMock(),
            doc_retriever=mock_doc_retriever,
            auto_init_rules=False,
            owl_strategy=None,
        )

        results = await retriever.search("test query", top_k=3)

        assert len(results) == 1
        mock_doc_retriever.search.assert_called_once()


# =============================================================================
# Relevance Grading and Rewrite Tests
# =============================================================================


class TestRelevanceGradingRewrite:
    """관련성 판정 및 쿼리 재작성 통합 테스트"""

    @pytest.mark.asyncio
    async def test_retrieve_with_relevance_rewrite_triggered(self):
        """관련성 부족 시 쿼리 재작성 트리거 테스트"""
        mock_kg = MagicMock()
        mock_kg.get_entity_metadata.return_value = {}
        mock_kg.get_brand_products.return_value = []
        mock_kg.get_competitors.return_value = []
        mock_kg.query.return_value = []

        mock_doc_retriever = MagicMock()
        mock_doc_retriever.initialize = AsyncMock()
        # 첫 검색: 3개 문서, 두 번째 검색(재작성 후): 2개 추가
        mock_doc_retriever.search = AsyncMock(
            side_effect=[
                [
                    {"id": "1", "content": "test1"},
                    {"id": "2", "content": "test2"},
                    {"id": "3", "content": "test3"},
                ],
                [{"id": "4", "content": "test4"}, {"id": "5", "content": "test5"}],
            ]
        )

        retriever = HybridRetriever(
            knowledge_graph=mock_kg,
            reasoner=MagicMock(),
            doc_retriever=mock_doc_retriever,
            auto_init_rules=False,
        )

        # 관련성 판정: 1개만 관련 → 재작성 필요
        retriever.relevance_grader.grade_documents = AsyncMock(
            return_value=([{"id": "1", "content": "test1"}], [{"id": "2"}, {"id": "3"}])
        )
        retriever.relevance_grader.needs_rewrite = MagicMock(return_value=True)

        # _rewrite_for_relevance를 패치하여 다른 쿼리 반환
        with patch.object(
            retriever, "_rewrite_for_relevance", return_value="LANEIGE Share of Shelf 점유율 분석"
        ):
            context = await retriever.retrieve("LANEIGE 분석", current_metrics={})

            # 재작성 후 문서가 추가되어야 함
            assert len(context.rag_chunks) >= 1
            # 두 번째 search 호출 확인 (재작성된 쿼리)
            assert mock_doc_retriever.search.call_count == 2

    @pytest.mark.asyncio
    async def test_retrieve_with_sufficient_relevant_docs_no_rewrite(self):
        """충분한 관련 문서가 있을 때 재작성 스킵 테스트"""
        mock_kg = MagicMock()
        mock_kg.get_entity_metadata.return_value = {}
        mock_kg.get_brand_products.return_value = []
        mock_kg.get_competitors.return_value = []
        mock_kg.query.return_value = []

        mock_doc_retriever = MagicMock()
        mock_doc_retriever.initialize = AsyncMock()
        mock_doc_retriever.search = AsyncMock(
            return_value=[
                {"id": "1", "content": "test1"},
                {"id": "2", "content": "test2"},
                {"id": "3", "content": "test3"},
            ]
        )

        retriever = HybridRetriever(
            knowledge_graph=mock_kg,
            reasoner=MagicMock(),
            doc_retriever=mock_doc_retriever,
            auto_init_rules=False,
        )

        # 3개 모두 관련 문서 → 재작성 불필요
        retriever.relevance_grader.grade_documents = AsyncMock(
            return_value=(
                [
                    {"id": "1", "content": "test1"},
                    {"id": "2", "content": "test2"},
                    {"id": "3", "content": "test3"},
                ],
                [],
            )
        )
        retriever.relevance_grader.needs_rewrite = MagicMock(return_value=False)

        context = await retriever.retrieve("LANEIGE 분석", current_metrics={})

        # 재작성이 트리거되지 않으므로 첫 번째 검색만 수행
        assert mock_doc_retriever.search.call_count == 1
        assert len(context.rag_chunks) == 3


# =============================================================================
# Build Inference Context Edge Cases
# =============================================================================


class TestBuildInferenceContextEdgeCases:
    """추론 컨텍스트 구성 엣지 케이스 테스트"""

    def test_build_context_without_categories_hhi(self):
        """카테고리 없이 HHI 할당 테스트"""
        mock_kg = MagicMock()
        mock_kg.get_competitors.return_value = []
        mock_kg.query.return_value = []

        retriever = HybridRetriever(
            knowledge_graph=mock_kg,
            reasoner=MagicMock(),
            doc_retriever=MagicMock(),
            auto_init_rules=False,
        )

        entities = {"brands": ["laneige"]}
        current_metrics = {
            "market_metrics": [
                {"category_id": "lip_care", "hhi": 0.15, "cpi": 105, "churn_rate_7d": 0.12}
            ]
        }

        context = retriever._build_inference_context(entities, current_metrics)

        # 카테고리 필터 없이 첫 번째 market_metric 적용
        assert context.get("hhi") == 0.15
        assert context.get("cpi") == 105

    def test_build_context_sos_fallback(self):
        """SoS 폴백 로직 테스트"""
        mock_kg = MagicMock()
        mock_kg.get_competitors.return_value = []
        mock_kg.query.return_value = []

        retriever = HybridRetriever(
            knowledge_graph=mock_kg,
            reasoner=MagicMock(),
            doc_retriever=MagicMock(),
            auto_init_rules=False,
        )

        entities = {"brands": ["laneige"], "categories": ["face_powder"]}
        current_metrics = {
            "summary": {"laneige_sos_by_category": {"lip_care": 0.08, "face_powder": 0.05}}
        }

        context = retriever._build_inference_context(entities, current_metrics)

        # face_powder 카테고리의 SoS가 할당되어야 함
        assert context.get("sos") == 0.05

    def test_build_context_brand_metrics_matching(self):
        """브랜드 메트릭 매칭 테스트"""
        mock_kg = MagicMock()
        mock_kg.get_competitors.return_value = []
        mock_kg.query.return_value = []

        retriever = HybridRetriever(
            knowledge_graph=mock_kg,
            reasoner=MagicMock(),
            doc_retriever=MagicMock(),
            auto_init_rules=False,
        )

        entities = {"brands": ["cosrx"]}
        current_metrics = {
            "summary": {"laneige_sos_by_category": {}},
            "brand_metrics": [
                {"brand_name": "LANEIGE", "share_of_shelf": 0.08, "avg_rank": 15},
                {"brand_name": "COSRX", "share_of_shelf": 0.05, "avg_rank": 22},
            ],
        }

        context = retriever._build_inference_context(entities, current_metrics)

        # cosrx 브랜드 메트릭이 매칭되어야 함
        assert context.get("sos") == 0.05
        assert context.get("avg_rank") == 22

    def test_build_context_sentiment_exception_handling(self):
        """감성 조회 예외 처리 테스트"""
        mock_kg = MagicMock()
        mock_kg.get_competitors.return_value = [{"brand": "cosrx"}]
        mock_kg.query.return_value = []
        mock_kg.get_brand_sentiment_profile.side_effect = [
            {"all_tags": ["hydrating"]},  # laneige
            Exception("Sentiment fetch failed"),  # cosrx (예외)
        ]

        retriever = HybridRetriever(
            knowledge_graph=mock_kg,
            reasoner=MagicMock(),
            doc_retriever=MagicMock(),
            auto_init_rules=False,
        )

        entities = {"brands": ["laneige"], "sentiments": ["보습"]}

        # 예외가 발생해도 컨텍스트 구성은 성공해야 함
        context = retriever._build_inference_context(entities, {})

        assert "sentiment_tags" in context
        # 경쟁사 감성은 빈 리스트여야 함 (예외로 인해)
        assert context.get("competitor_sentiment_tags", []) == []


# =============================================================================
# Category Hierarchy Facts Tests
# =============================================================================


class TestCategoryHierarchyFacts:
    """카테고리 계층 사실 조회 상세 테스트"""

    def test_query_kg_category_hierarchy_with_string_path(self):
        """문자열 경로를 가진 계층 정보 테스트"""
        mock_kg = MagicMock()
        mock_kg.get_category_hierarchy.return_value = {
            "name": "Lip Care",
            "level": 2,
            "path": ["Beauty", "Skin Care", "Lip Care"],  # 문자열 리스트
            "ancestors": [],
            "descendants": [],
        }

        retriever = HybridRetriever(
            knowledge_graph=mock_kg,
            reasoner=MagicMock(),
            doc_retriever=MagicMock(),
            auto_init_rules=False,
        )

        entities = {"categories": ["lip_care"]}
        facts = retriever._query_knowledge_graph(entities)

        hierarchy_facts = [f for f in facts if f["type"] == "category_hierarchy"]
        assert len(hierarchy_facts) > 0
        assert hierarchy_facts[0]["data"]["path"] == ["Beauty", "Skin Care", "Lip Care"]

    def test_query_kg_category_hierarchy_error(self):
        """카테고리 계층 조회 에러 처리 테스트"""
        mock_kg = MagicMock()
        mock_kg.get_category_hierarchy.return_value = {"error": "Not found"}

        retriever = HybridRetriever(
            knowledge_graph=mock_kg,
            reasoner=MagicMock(),
            doc_retriever=MagicMock(),
            auto_init_rules=False,
        )

        entities = {"categories": ["unknown_category"]}
        facts = retriever._query_knowledge_graph(entities)

        # 에러 응답은 사실로 추가되지 않아야 함
        hierarchy_facts = [f for f in facts if f["type"] == "category_hierarchy"]
        assert len(hierarchy_facts) == 0


# =============================================================================
# Initialize Already Initialized Tests
# =============================================================================


class TestInitializeIdempotency:
    """초기화 중복 호출 테스트"""

    @pytest.mark.asyncio
    async def test_initialize_already_initialized_skip(self):
        """이미 초기화된 경우 재초기화 스킵 테스트"""
        mock_doc_retriever = MagicMock()
        mock_doc_retriever.initialize = AsyncMock()

        mock_kg = MagicMock()
        mock_kg.load_category_hierarchy.return_value = 10

        retriever = HybridRetriever(
            knowledge_graph=mock_kg,
            reasoner=MagicMock(),
            doc_retriever=mock_doc_retriever,
            auto_init_rules=False,
        )

        # 첫 초기화
        await retriever.initialize()
        assert retriever._initialized is True

        # 초기화 호출 횟수 확인
        first_call_count = mock_doc_retriever.initialize.call_count

        # 재초기화 시도
        await retriever.initialize()

        # 재초기화가 스킵되어야 함
        assert mock_doc_retriever.initialize.call_count == first_call_count


# =============================================================================
# Expand Query Edge Cases
# =============================================================================


class TestExpandQueryEdgeCases:
    """쿼리 확장 엣지 케이스 테스트"""

    def test_expand_query_with_multiple_insight_types(self):
        """여러 인사이트 유형으로 쿼리 확장 테스트"""
        retriever = HybridRetriever(
            knowledge_graph=MagicMock(),
            reasoner=MagicMock(),
            doc_retriever=MagicMock(),
            auto_init_rules=False,
        )

        inferences = [
            InferenceResult(
                rule_name="r1",
                insight_type=InsightType.GROWTH_OPPORTUNITY,
                insight="opportunity",
            ),
            InferenceResult(
                rule_name="r2",
                insight_type=InsightType.PRICE_QUALITY_GAP,
                insight="price gap",
            ),
        ]

        entities = {"indicators": ["hhi", "cpi"]}

        expanded = retriever._expand_query("test query", inferences, entities)

        # 인사이트 및 지표 관련 확장어가 포함되어야 함
        assert "성장" in expanded or "기회" in expanded or "가격" in expanded
        assert "HHI" in expanded or "CPI" in expanded

    def test_expand_query_no_expansion(self):
        """확장 조건이 없을 때 원본 유지 테스트"""
        retriever = HybridRetriever(
            knowledge_graph=MagicMock(),
            reasoner=MagicMock(),
            doc_retriever=MagicMock(),
            auto_init_rules=False,
        )

        # 확장 조건 없음
        inferences = []
        entities = {}

        expanded = retriever._expand_query("test query", inferences, entities)

        # 원본 쿼리와 동일해야 함
        assert expanded == "test query"


# =============================================================================
# Combine Contexts Without Explanations
# =============================================================================


class TestCombineContextsNoExplanations:
    """설명 제외 컨텍스트 통합 테스트"""

    def test_combine_contexts_without_explanations(self):
        """추론 설명 제외 포매팅 테스트"""
        retriever = HybridRetriever(
            knowledge_graph=MagicMock(),
            reasoner=MagicMock(),
            doc_retriever=MagicMock(),
            auto_init_rules=False,
        )

        inference = InferenceResult(
            rule_name="test_rule",
            insight_type=InsightType.COMPETITIVE_THREAT,
            insight="Threat detected",
            confidence=0.85,
            evidence={"satisfied_conditions": ["condition1", "condition2"]},
        )

        context = HybridContext(query="test", inferences=[inference])

        combined = retriever._combine_contexts(context, include_explanations=False)

        # 인사이트는 포함되어야 하지만 근거 조건은 제외되어야 함
        assert "Threat detected" in combined
        assert "condition1" not in combined
        assert "satisfied_conditions" not in combined


# =============================================================================
# Load Retrieval Weights Edge Cases
# =============================================================================


class TestLoadRetrievalWeights:
    """검색 가중치 로딩 테스트"""

    def test_load_retrieval_weights_file_not_exists(self):
        """설정 파일 없을 때 기본값 사용 테스트"""
        with patch("pathlib.Path.exists", return_value=False):
            retriever = HybridRetriever(
                knowledge_graph=MagicMock(),
                reasoner=MagicMock(),
                doc_retriever=MagicMock(),
                auto_init_rules=False,
            )

            weights = retriever._retrieval_weights

            # 기본값이 로드되어야 함
            assert "weights" in weights
            assert weights["weights"]["kg"] == 0.4
            assert weights["weights"]["rag"] == 0.4
            assert weights["weights"]["inference"] == 0.2

    def test_load_retrieval_weights_custom_config(self):
        """커스텀 가중치 설정 로드 테스트"""
        custom_config = {
            "weights": {"kg": 0.5, "rag": 0.3, "inference": 0.2},
            "freshness": {"weekly": 1.0, "quarterly": 0.8, "static": 0.6},
            "max_context_items": {"ontology_facts": 10, "inferences": 8, "rag_chunks": 5},
        }

        with (
            patch("pathlib.Path.exists", return_value=True),
            patch("builtins.open", mock_open(read_data=json.dumps(custom_config))),
        ):
            retriever = HybridRetriever(
                knowledge_graph=MagicMock(),
                reasoner=MagicMock(),
                doc_retriever=MagicMock(),
                auto_init_rules=False,
            )

            weights = retriever._retrieval_weights

            # 커스텀 값이 로드되어야 함
            assert weights["weights"]["kg"] == 0.5
            assert weights["max_context_items"]["ontology_facts"] == 10


# =============================================================================
# Weighted Merge Inference Scoring
# =============================================================================


class TestWeightedMergeInferences:
    """가중치 기반 추론 병합 테스트"""

    def test_weighted_merge_inferences_scoring(self):
        """추론 점수 할당 및 정렬 테스트"""
        retriever = HybridRetriever(
            knowledge_graph=MagicMock(),
            reasoner=MagicMock(),
            doc_retriever=MagicMock(),
            auto_init_rules=False,
        )

        inferences = [
            InferenceResult(
                rule_name="r1",
                insight_type=InsightType.MARKET_POSITION,
                insight="insight1",
                confidence=0.9,
            ),
            InferenceResult(
                rule_name="r2",
                insight_type=InsightType.RISK_ALERT,
                insight="insight2",
                confidence=0.7,
            ),
            InferenceResult(
                rule_name="r3",
                insight_type=InsightType.GROWTH_OPPORTUNITY,
                insight="insight3",
                confidence=0.5,
            ),
        ]

        context = HybridContext(query="test", inferences=inferences)

        merged = retriever._weighted_merge(context)

        # 신뢰도 순으로 정렬되어야 함 (0.9 > 0.7 > 0.5)
        assert len(merged.inferences) <= 5  # max_items 제한
        assert getattr(merged.inferences[0], "confidence", 0) >= getattr(
            merged.inferences[-1], "confidence", 0
        )

    def test_weighted_merge_metadata_stored(self):
        """가중치 점수가 메타데이터에 저장되는지 테스트"""
        retriever = HybridRetriever(
            knowledge_graph=MagicMock(),
            reasoner=MagicMock(),
            doc_retriever=MagicMock(),
            auto_init_rules=False,
        )

        context = HybridContext(
            query="test",
            ontology_facts=[{"type": "brand_info", "entity": "test"}],
            rag_chunks=[{"id": "1", "score": 0.8, "metadata": {}}],
            inferences=[
                InferenceResult(
                    rule_name="r1",
                    insight_type=InsightType.MARKET_POSITION,
                    insight="test",
                    confidence=0.9,
                )
            ],
        )

        merged = retriever._weighted_merge(context)

        # 메타데이터에 점수가 저장되어야 함
        assert "weighted_scores" in merged.metadata
        assert "ontology_facts" in merged.metadata["weighted_scores"]
        assert "rag_chunks" in merged.metadata["weighted_scores"]
        assert "inferences" in merged.metadata["weighted_scores"]
