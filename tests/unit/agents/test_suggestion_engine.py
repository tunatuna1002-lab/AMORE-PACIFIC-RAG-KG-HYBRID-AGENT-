"""
Unit tests for SuggestionEngine
"""

import pytest

from src.agents.suggestion_engine import SuggestionEngine
from src.domain.entities.relations import InferenceResult, InsightType
from src.rag.router import QueryType


class TestSuggestionEngine:
    """Test SuggestionEngine functionality"""

    @pytest.fixture
    def engine(self):
        """Create SuggestionEngine instance"""
        return SuggestionEngine()

    @pytest.fixture
    def mock_kg(self):
        """Mock knowledge graph with get_related_brands method"""

        class MockKG:
            def get_related_brands(self, brand: str, limit: int = 2):
                if brand == "LANEIGE":
                    return ["Innisfree", "Etude House"]
                return []

        return MockKG()

    def test_generate_with_empty_context(self, engine):
        """Test suggestion generation with empty context"""
        suggestions = engine.generate(
            query_type=QueryType.UNKNOWN,
            entities={},
            inferences=[],
            response="",
        )

        assert isinstance(suggestions, list)
        assert len(suggestions) == 3  # Always returns 3 suggestions
        assert all(isinstance(s, str) for s in suggestions)

    def test_generate_with_rich_context(self, mock_kg):
        """Test suggestion generation with rich context (entities + inferences)"""
        engine = SuggestionEngine(knowledge_graph=mock_kg)

        entities = {
            "brands": ["LANEIGE"],
            "categories": ["Lip Care"],
            "indicators": ["SoS"],
        }

        inferences = [
            InferenceResult(
                insight="LANEIGE 순위 하락",
                insight_type=InsightType.COMPETITIVE_THREAT,
                rule_name="rank_drop",
                confidence=0.9,
                evidence=["rank dropped"],
                recommendation="개선 필요",
            )
        ]

        response = "LANEIGE가 Lip Care에서 순위가 하락했습니다. 경쟁사 분석이 필요합니다."

        suggestions = engine.generate(
            query_type=QueryType.ANALYSIS,
            entities=entities,
            inferences=inferences,
            response=response,
        )

        assert len(suggestions) == 3
        # Should include keyword-based (순위 하락), entity-based (vs competitors), inference-based
        assert any("하락" in s or "경쟁" in s for s in suggestions)

    def test_extract_response_keywords(self, engine):
        """Test keyword extraction from response"""
        response = "LANEIGE 순위가 급락했습니다. 경쟁사 분석이 필요합니다."
        keywords = engine._extract_response_keywords(response)

        assert isinstance(keywords, list)
        assert len(keywords) <= 2  # Max 2 keywords
        assert any("하락" in k or "경쟁" in k for k in keywords)

    def test_generate_entity_suggestions(self, mock_kg):
        """Test entity-based suggestion generation"""
        engine = SuggestionEngine(knowledge_graph=mock_kg)

        entities = {
            "brands": ["LANEIGE", "Innisfree"],
            "categories": ["Lip Care"],
            "indicators": ["SoS", "HHI"],
        }

        suggestions = engine._generate_entity_suggestions(entities)

        assert isinstance(suggestions, list)
        # Should include brand comparison, category trends, indicator improvement
        assert any("LANEIGE" in s for s in suggestions)
        assert any("Lip Care" in s or "SoS" in s for s in suggestions)

    def test_generate_inference_suggestions(self, engine):
        """Test inference-based suggestion generation"""
        inferences = [
            InferenceResult(
                insight="가격 경쟁력 낮음",
                insight_type=InsightType.PRICE_POSITION,
                rule_name="price_rule",
                confidence=0.85,
                evidence=["high price"],
                recommendation="가격 조정",
            ),
            InferenceResult(
                insight="성장 기회 존재",
                insight_type=InsightType.GROWTH_OPPORTUNITY,
                rule_name="growth_rule",
                confidence=0.75,
                evidence=["growing market"],
                recommendation="시장 확대",
            ),
        ]

        suggestions = engine._generate_inference_suggestions(inferences)

        assert isinstance(suggestions, list)
        # Should include price strategy, growth opportunity, recommendation-based
        assert any("가격" in s or "성장" in s for s in suggestions)

    def test_generate_type_suggestions(self, engine):
        """Test query-type-based fallback suggestions"""
        entities = {"indicators": ["SoS"]}

        # Test DEFINITION type
        suggestions = engine._generate_type_suggestions(QueryType.DEFINITION, entities)
        assert isinstance(suggestions, list)
        assert len(suggestions) > 0
        # Should have indicator-related or general suggestions
        assert any("SoS" in s or "지표" in s or "데이터" in s for s in suggestions)

        # Test ANALYSIS type
        suggestions = engine._generate_type_suggestions(QueryType.ANALYSIS, {})
        assert len(suggestions) > 0
        assert any("트렌드" in s or "경쟁사" in s or "분석" in s for s in suggestions)

    def test_deduplication(self, engine):
        """Test that duplicate suggestions are removed"""
        entities = {
            "brands": ["LANEIGE"],
            "categories": ["Lip Care"],
        }

        # Generate multiple times - should deduplicate
        suggestions = engine.generate(
            query_type=QueryType.ANALYSIS,
            entities=entities,
            inferences=[],
            response="",
        )

        # Check no exact duplicates
        assert len(suggestions) == len(set(suggestions))

    def test_get_fallback_suggestions(self, engine):
        """Test fallback suggestions"""
        suggestions = engine.get_fallback_suggestions()

        assert isinstance(suggestions, list)
        assert len(suggestions) == 3
        assert all(isinstance(s, str) for s in suggestions)
        assert any("SoS" in s or "LANEIGE" in s or "인사이트" in s for s in suggestions)

    def test_priority_order(self, mock_kg):
        """Test that 4-tier priority is respected"""
        engine = SuggestionEngine(knowledge_graph=mock_kg)

        entities = {"brands": ["LANEIGE"]}
        inferences = [
            InferenceResult(
                insight="Test insight",
                insight_type=InsightType.COMPETITIVE_THREAT,
                rule_name="test",
                confidence=0.9,
                evidence=[],
            )
        ]
        response = "순위가 하락했습니다"

        suggestions = engine.generate(
            query_type=QueryType.ANALYSIS,
            entities=entities,
            inferences=inferences,
            response=response,
        )

        # First should be keyword-based (from response)
        assert len(suggestions) >= 1
        # Should not exceed 3
        assert len(suggestions) == 3
