"""
RAGRouter 단위 테스트
====================
질의 분류, 문서 라우팅, 엔티티 추출, Fallback 처리 검증
"""

import pytest

from src.rag.router import QueryType, RAGRouter

# ---------------------------------------------------------------------------
# QueryType enum
# ---------------------------------------------------------------------------


class TestQueryType:
    """QueryType enum 기본 테스트"""

    def test_all_types_exist(self):
        expected = {
            "definition",
            "interpretation",
            "combination",
            "insight_rule",
            "data_query",
            "analysis",
            "unknown",
        }
        assert {qt.value for qt in QueryType} == expected

    def test_string_enum(self):
        assert QueryType.DEFINITION == "definition"
        assert isinstance(QueryType.DEFINITION, str)


# ---------------------------------------------------------------------------
# classify_query
# ---------------------------------------------------------------------------


class TestClassifyQuery:
    """질의 분류 테스트"""

    @pytest.fixture
    def router(self):
        return RAGRouter()

    # --- 정의 질의 ---
    @pytest.mark.parametrize(
        "query",
        [
            "SoS가 뭐야?",
            "HHI 정의 알려줘",
            "CPI 산출식 설명해줘",
            "어떻게 계산하는 거야 SoS",
        ],
    )
    def test_definition_queries(self, router, query):
        qt, confidence = router.classify_query(query)
        assert qt == QueryType.DEFINITION

    # --- 해석 질의 ---
    @pytest.mark.parametrize(
        "query",
        [
            "HHI가 높으면 어떤 의미야?",
            "SoS 해석 방법",
            "CPI 낮으면 뜻이 뭐야",
        ],
    )
    def test_interpretation_queries(self, router, query):
        qt, confidence = router.classify_query(query)
        assert qt == QueryType.INTERPRETATION

    # --- 조합 질의 ---
    @pytest.mark.parametrize(
        "query",
        [
            "SoS 상승하고 HHI 하락 시나리오",
            "지표 조합 시나리오 분석",
        ],
    )
    def test_combination_queries(self, router, query):
        qt, confidence = router.classify_query(query)
        assert qt == QueryType.COMBINATION

    # --- 데이터 조회 ---
    @pytest.mark.parametrize(
        "query",
        [
            "라네즈 현재 순위",
            "오늘 랭킹 보여줘",
            "최근 제품 랭킹",
        ],
    )
    def test_data_queries(self, router, query):
        qt, confidence = router.classify_query(query)
        assert qt == QueryType.DATA_QUERY

    # --- 분석 질의 ---
    @pytest.mark.parametrize(
        "query",
        [
            "라네즈 경쟁사 비교 분석",
            "3개월 트렌드 분석 리포트",
        ],
    )
    def test_analysis_queries(self, router, query):
        qt, confidence = router.classify_query(query)
        assert qt == QueryType.ANALYSIS

    # --- Unknown ---
    def test_unknown_query(self, router):
        qt, confidence = router.classify_query("안녕하세요")
        assert qt == QueryType.UNKNOWN
        assert confidence == 0.0


# ---------------------------------------------------------------------------
# classify_query_detailed
# ---------------------------------------------------------------------------


class TestClassifyQueryDetailed:
    """상세 분류 테스트"""

    @pytest.fixture
    def router(self):
        return RAGRouter()

    def test_returns_four_values(self, router):
        result = router.classify_query_detailed("SoS 정의")
        assert len(result) == 4

    def test_matched_keywords_populated(self, router):
        _, _, _, keywords = router.classify_query_detailed("SoS가 뭐야?")
        assert len(keywords) > 0
        assert any(kw in ["뭐야", "sos"] for kw in keywords)

    def test_max_score_threshold(self, router):
        """최소 점수 미달 → UNKNOWN"""
        qt, confidence, max_score, _ = router.classify_query_detailed("hello world")
        assert qt == QueryType.UNKNOWN
        assert max_score < 1.5


# ---------------------------------------------------------------------------
# route
# ---------------------------------------------------------------------------


class TestRoute:
    """라우팅 결과 테스트"""

    @pytest.fixture
    def router(self):
        return RAGRouter()

    def test_route_returns_dict(self, router):
        result = router.route("SoS 정의")
        assert isinstance(result, dict)
        expected_keys = {
            "query_type",
            "confidence",
            "max_score",
            "matched_keywords",
            "target_doc",
            "requires_data",
            "requires_rag",
            "fallback_message",
        }
        assert expected_keys == set(result.keys())

    def test_definition_route(self, router):
        result = router.route("SoS가 뭐야?")
        assert result["query_type"] == QueryType.DEFINITION
        assert result["target_doc"] == "strategic_indicators"
        assert result["requires_rag"] is True
        assert result["requires_data"] is False

    def test_data_query_route(self, router):
        result = router.route("라네즈 현재 순위")
        assert result["requires_data"] is True

    def test_unknown_fallback(self, router):
        result = router.route("asdfghjkl")
        assert result["query_type"] == QueryType.UNKNOWN
        assert result["fallback_message"] is not None


# ---------------------------------------------------------------------------
# get_target_document
# ---------------------------------------------------------------------------


class TestGetTargetDocument:
    """문서 매핑 테스트"""

    @pytest.fixture
    def router(self):
        return RAGRouter()

    def test_definition_doc(self, router):
        assert router.get_target_document(QueryType.DEFINITION) == "strategic_indicators"

    def test_interpretation_doc(self, router):
        assert router.get_target_document(QueryType.INTERPRETATION) == "metric_interpretation"

    def test_combination_doc(self, router):
        assert router.get_target_document(QueryType.COMBINATION) == "indicator_combination"

    def test_unknown_returns_none(self, router):
        assert router.get_target_document(QueryType.UNKNOWN) is None

    def test_data_query_returns_none(self, router):
        assert router.get_target_document(QueryType.DATA_QUERY) is None


# ---------------------------------------------------------------------------
# extract_entities
# ---------------------------------------------------------------------------


class TestExtractEntities:
    """엔티티 추출 테스트"""

    @pytest.fixture
    def router(self):
        return RAGRouter()

    def test_extract_brand_laneige(self, router):
        entities = router.extract_entities("라네즈 분석")
        assert "laneige" in entities["brands"]

    def test_extract_brand_alias(self, router):
        """동의어 인식"""
        entities = router.extract_entities("라네이지 립 마스크")
        assert "laneige" in entities["brands"]

    def test_extract_brand_english(self, router):
        entities = router.extract_entities("LANEIGE product")
        assert "laneige" in entities["brands"]

    def test_extract_multiple_brands(self, router):
        entities = router.extract_entities("COSRX vs LANEIGE 비교")
        assert "cosrx" in entities["brands"]
        assert "laneige" in entities["brands"]

    def test_extract_category(self, router):
        entities = router.extract_entities("Lip Care 카테고리 분석")
        assert "lip_care" in entities["categories"]

    def test_extract_indicator(self, router):
        entities = router.extract_entities("SoS 지표 해석")
        assert "sos" in entities["indicators"]

    def test_extract_time_range(self, router):
        entities = router.extract_entities("최근 7일 데이터")
        assert entities["time_range"] == "7days"

    def test_extract_time_today(self, router):
        entities = router.extract_entities("오늘 순위")
        assert entities["time_range"] == "today"

    def test_no_entities(self, router):
        entities = router.extract_entities("hello world")
        assert entities["brands"] == []
        assert entities["categories"] == []
        assert entities["indicators"] == []
        assert entities["time_range"] is None


# ---------------------------------------------------------------------------
# needs_clarification
# ---------------------------------------------------------------------------


class TestNeedsClarification:
    """명확화 요청 테스트"""

    @pytest.fixture
    def router(self):
        return RAGRouter()

    def test_data_query_no_brand(self, router):
        route_result = {"query_type": QueryType.DATA_QUERY}
        entities = {
            "brands": [],
            "products": [],
            "categories": [],
            "indicators": [],
            "time_range": None,
        }
        msg = router.needs_clarification(route_result, entities)
        assert msg is not None
        assert "브랜드" in msg

    def test_data_query_with_brand(self, router):
        route_result = {"query_type": QueryType.DATA_QUERY}
        entities = {
            "brands": ["laneige"],
            "products": [],
            "categories": [],
            "indicators": [],
            "time_range": None,
        }
        msg = router.needs_clarification(route_result, entities)
        assert msg is None

    def test_analysis_no_time(self, router):
        route_result = {"query_type": QueryType.ANALYSIS}
        entities = {
            "brands": ["laneige"],
            "products": [],
            "categories": [],
            "indicators": [],
            "time_range": None,
        }
        msg = router.needs_clarification(route_result, entities)
        assert msg is not None
        assert "기간" in msg

    def test_definition_no_clarification(self, router):
        route_result = {"query_type": QueryType.DEFINITION}
        entities = {
            "brands": [],
            "products": [],
            "categories": [],
            "indicators": [],
            "time_range": None,
        }
        msg = router.needs_clarification(route_result, entities)
        assert msg is None


# ---------------------------------------------------------------------------
# get_fallback_response
# ---------------------------------------------------------------------------


class TestGetFallbackResponse:
    """Fallback 응답 테스트"""

    @pytest.fixture
    def router(self):
        return RAGRouter()

    def test_unknown_fallback(self, router):
        msg = router.get_fallback_response("unknown")
        assert "질문의 의도" in msg

    def test_no_data_fallback(self, router):
        msg = router.get_fallback_response("no_data")
        assert "데이터를 찾을 수 없습니다" in msg

    def test_clarification_fallback(self, router):
        msg = router.get_fallback_response("clarification")
        assert "추가 정보" in msg

    def test_invalid_reason_defaults(self, router):
        msg = router.get_fallback_response("nonexistent_reason")
        assert "질문의 의도" in msg  # UNKNOWN fallback
