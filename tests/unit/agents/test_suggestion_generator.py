"""
SuggestionGenerator 단위 테스트
"""

from unittest.mock import MagicMock

from src.agents.suggestion_generator import SuggestionGenerator
from src.domain.entities import InferenceResult, InsightType
from src.rag.router import QueryType


class TestSuggestionGeneratorInit:
    """SuggestionGenerator 초기화 테스트"""

    def test_suggestion_generator_without_kg(self):
        """KG 없이 생성 가능"""
        generator = SuggestionGenerator()
        assert generator._kg is None

    def test_suggestion_generator_with_kg(self):
        """KG와 함께 생성"""
        mock_kg = MagicMock()
        generator = SuggestionGenerator(knowledge_graph=mock_kg)
        assert generator._kg is mock_kg


class TestSuggestionGeneratorGenerate:
    """SuggestionGenerator.generate 테스트"""

    def test_generate_returns_max_3_suggestions(self):
        """최대 3개 제안 반환"""
        generator = SuggestionGenerator()
        result = generator.generate(
            query_type=QueryType.ANALYSIS,
            entities={"brands": ["LANEIGE", "COSRX"], "categories": ["Lip Care"]},
            inferences=[],
            response="",
        )
        assert len(result) <= 3

    def test_generate_with_response_keywords(self):
        """응답 키워드 기반 제안"""
        generator = SuggestionGenerator()
        result = generator.generate(
            query_type=QueryType.ANALYSIS,
            entities={},
            inferences=[],
            response="LANEIGE 순위가 하락했습니다. 경쟁사 대비 SoS가 낮습니다.",
        )
        # 키워드 기반 제안이 포함되어야 함
        assert any("하락" in s or "경쟁" in s or "점유율" in s for s in result)

    def test_generate_with_entities(self):
        """엔티티 기반 제안"""
        generator = SuggestionGenerator()
        result = generator.generate(
            query_type=QueryType.ANALYSIS,
            entities={"brands": ["LANEIGE"], "categories": [], "indicators": []},
            inferences=[],
            response="",
        )
        assert any("LANEIGE" in s for s in result)


class TestSuggestionGeneratorExtractKeywords:
    """키워드 추출 테스트"""

    def test_extract_rank_drop_keyword(self):
        """순위 하락 키워드 추출"""
        generator = SuggestionGenerator()
        keywords = generator._extract_response_keywords("순위가 급락했습니다.")
        assert any("하락" in k or "원인" in k for k in keywords)

    def test_extract_trend_keyword(self):
        """트렌드 키워드 추출"""
        generator = SuggestionGenerator()
        keywords = generator._extract_response_keywords("최근 트렌드가 변화하고 있습니다.")
        assert any("트렌드" in k for k in keywords)

    def test_extract_no_keywords(self):
        """키워드 없음"""
        generator = SuggestionGenerator()
        keywords = generator._extract_response_keywords("일반적인 응답입니다.")
        assert keywords == []


class TestSuggestionGeneratorEntitySuggestions:
    """엔티티 기반 제안 테스트"""

    def test_brand_suggestions(self):
        """브랜드 기반 제안"""
        generator = SuggestionGenerator()
        suggestions = generator._generate_entity_suggestions(
            {"brands": ["LANEIGE"], "categories": [], "indicators": []}
        )
        assert any("LANEIGE" in s for s in suggestions)

    def test_category_suggestions(self):
        """카테고리 기반 제안"""
        generator = SuggestionGenerator()
        suggestions = generator._generate_entity_suggestions(
            {"brands": [], "categories": ["Lip Care"], "indicators": []}
        )
        assert any("Lip Care" in s for s in suggestions)

    def test_indicator_suggestions(self):
        """지표 기반 제안"""
        generator = SuggestionGenerator()
        suggestions = generator._generate_entity_suggestions(
            {"brands": [], "categories": [], "indicators": ["sos"]}
        )
        assert any("SOS" in s for s in suggestions)

    def test_with_kg_competitor(self):
        """KG 경쟁사 조회 연동"""
        mock_kg = MagicMock()
        mock_kg.get_related_brands.return_value = ["Summer Fridays"]

        generator = SuggestionGenerator(knowledge_graph=mock_kg)
        suggestions = generator._generate_entity_suggestions(
            {"brands": ["LANEIGE"], "categories": [], "indicators": []}
        )

        # KG 호출 확인
        mock_kg.get_related_brands.assert_called_once_with("LANEIGE", limit=2)
        # 경쟁사 비교 제안 포함
        assert any("Summer Fridays" in s or "비교" in s for s in suggestions)


class TestSuggestionGeneratorInferenceSuggestions:
    """추론 기반 제안 테스트"""

    def test_competitive_inference(self):
        """경쟁 관련 추론"""
        inference = InferenceResult(
            rule_name="competition_rule",
            insight="경쟁이 심화되고 있습니다.",
            insight_type=InsightType.COMPETITIVE_THREAT,
            confidence=0.8,
        )
        generator = SuggestionGenerator()
        suggestions = generator._generate_inference_suggestions([inference])
        assert any("경쟁" in s for s in suggestions)

    def test_recommendation_based_suggestion(self):
        """권장 액션 기반 제안"""
        inference = InferenceResult(
            rule_name="growth_rule",
            insight="성장 가능성이 있습니다.",
            insight_type=InsightType.GROWTH_OPPORTUNITY,
            confidence=0.7,
            recommendation="가격 경쟁력 강화",
        )
        generator = SuggestionGenerator()
        suggestions = generator._generate_inference_suggestions([inference])
        assert any("가격 경쟁력" in s or "실행 방법" in s for s in suggestions)


class TestSuggestionGeneratorTypeSuggestions:
    """쿼리 유형 기반 제안 테스트"""

    def test_definition_type_suggestions(self):
        """정의 질문 제안"""
        generator = SuggestionGenerator()
        suggestions = generator._generate_type_suggestions(
            QueryType.DEFINITION, {"indicators": ["sos"]}
        )
        assert any("SOS" in s or "의미" in s for s in suggestions)

    def test_analysis_type_suggestions(self):
        """분석 질문 제안"""
        generator = SuggestionGenerator()
        suggestions = generator._generate_type_suggestions(QueryType.ANALYSIS, {})
        assert any("트렌드" in s or "경쟁" in s for s in suggestions)

    def test_unknown_type_fallback(self):
        """알 수 없는 유형 폴백"""
        generator = SuggestionGenerator()
        suggestions = generator._generate_type_suggestions(QueryType.UNKNOWN, {})
        assert len(suggestions) > 0


class TestSuggestionGeneratorFallback:
    """폴백 제안 테스트"""

    def test_fallback_suggestions(self):
        """폴백 제안 반환"""
        generator = SuggestionGenerator()
        fallback = generator.get_fallback_suggestions()
        assert len(fallback) == 3
        assert any("SoS" in s for s in fallback)
