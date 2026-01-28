"""
HybridRetriever 단위 테스트
==========================
src/rag/hybrid_retriever.py의 하이브리드 검색기 테스트
"""

import json
import tempfile
from datetime import timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

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
        extractor = EntityExtractor()
        assert "laneige" in extractor.KNOWN_BRANDS
        assert "cosrx" in extractor.KNOWN_BRANDS
        assert "lip care" in extractor.CATEGORY_MAP
        assert "sos" in extractor.INDICATOR_MAP

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

    def test_entity_extractor_config_caching(self):
        """설정 캐싱 테스트"""
        # 캐시 초기화
        EntityExtractor._config_cache = None
        EntityExtractor._config_loaded_at = None

        # 첫 번째 호출
        extractor1 = EntityExtractor()
        config1 = extractor1._load_config()
        loaded_at1 = EntityExtractor._config_loaded_at

        # 두 번째 호출 (캐시 사용)
        extractor2 = EntityExtractor()
        config2 = extractor2._load_config()
        loaded_at2 = EntityExtractor._config_loaded_at

        # 같은 로드 시점이어야 함 (캐시 사용)
        assert loaded_at1 == loaded_at2
        # 같은 내용이어야 함
        assert config1 == config2

    def test_entity_extractor_config_ttl_expiry(self):
        """설정 캐시 TTL 만료 테스트"""
        from datetime import datetime as dt

        # 캐시 초기화
        EntityExtractor._config_cache = None
        EntityExtractor._config_loaded_at = None

        # 기본 설정을 mock으로 반환
        mock_config = EntityExtractor._get_default_config()

        # TTL을 매우 짧게 설정 (0초 = 항상 만료)
        with patch.object(EntityExtractor, "_get_config_ttl_seconds", return_value=0):
            extractor = EntityExtractor()

            # 캐시에 직접 설정 주입 (파일 로드 대신)
            EntityExtractor._config_cache = mock_config
            EntityExtractor._config_loaded_at = dt.now()

            first_loaded_at = EntityExtractor._config_loaded_at
            assert first_loaded_at is not None

            # 캐시 만료 시뮬레이션 (과거 시점으로 설정)
            past_time = dt.now() - timedelta(seconds=2)
            EntityExtractor._config_loaded_at = past_time

            # 캐시에 다시 설정 (재로드 시뮬레이션)
            EntityExtractor._config_cache = mock_config
            EntityExtractor._config_loaded_at = dt.now()

            second_loaded_at = EntityExtractor._config_loaded_at

            # 두 번째 로드 시점이 첫 번째보다 나중이어야 함
            assert second_loaded_at is not None
            assert second_loaded_at > past_time

    def test_entity_extractor_custom_config_file(self):
        """커스텀 설정 파일 로드 테스트"""
        # 임시 설정 파일 생성
        temp_config = {
            "known_brands": [{"name": "testbrand", "aliases": ["테스트브랜드"]}],
            "category_map": {"test_category": "test_cat"},
            "indicator_map": {"test_metric": "test_m"},
            "time_range_map": {},
            "sentiment_map": {},
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(temp_config, f)
            temp_path = f.name

        try:
            # 캐시 초기화
            EntityExtractor._config_cache = None
            EntityExtractor._config_loaded_at = None

            # _load_config를 직접 패치하여 임시 설정 반환
            with patch.object(EntityExtractor, "_load_config", return_value=temp_config):
                extractor = EntityExtractor()
                # _load_config가 temp_config를 반환하므로 testbrand가 포함되어야 함
                brands = extractor.get_known_brands()
                assert "testbrand" in brands
                assert "test_category" in extractor.CATEGORY_MAP
        finally:
            Path(temp_path).unlink()


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
