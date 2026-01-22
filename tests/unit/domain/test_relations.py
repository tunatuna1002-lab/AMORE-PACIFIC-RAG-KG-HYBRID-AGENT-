"""
Domain Relations Tests (TDD - RED Phase)
=========================================
온톨로지 관계 타입 및 추론 결과에 대한 테스트
"""

import pytest
from datetime import datetime


class TestRelationType:
    """RelationType Enum 테스트"""

    def test_entity_relations(self):
        """엔티티 관계 유형 검증"""
        from src.domain.entities.relations import RelationType

        assert RelationType.HAS_PRODUCT.value == "hasProduct"
        assert RelationType.BELONGS_TO_CATEGORY.value == "belongsToCategory"
        assert RelationType.OWNED_BY.value == "ownedBy"

    def test_competition_relations(self):
        """경쟁 관계 유형 검증"""
        from src.domain.entities.relations import RelationType

        assert RelationType.COMPETES_WITH.value == "competesWith"
        assert RelationType.DIRECT_COMPETITOR.value == "directCompetitor"
        assert RelationType.INDIRECT_COMPETITOR.value == "indirectCompetitor"

    def test_metric_relations(self):
        """지표 관계 유형 검증"""
        from src.domain.entities.relations import RelationType

        assert RelationType.INDICATES.value == "indicates"
        assert RelationType.CORRELATES_WITH.value == "correlatesWith"
        assert RelationType.INFLUENCES.value == "influences"


class TestRelation:
    """Relation (Triple) 테스트"""

    def test_relation_creation(self):
        """Relation 생성 검증"""
        from src.domain.entities.relations import Relation, RelationType

        relation = Relation(
            subject="LANEIGE",
            predicate=RelationType.HAS_PRODUCT,
            object="B08XYZ1234"
        )

        assert relation.subject == "LANEIGE"
        assert relation.predicate == RelationType.HAS_PRODUCT
        assert relation.object == "B08XYZ1234"

    def test_relation_with_properties(self):
        """Relation의 추가 속성 검증"""
        from src.domain.entities.relations import Relation, RelationType

        relation = Relation(
            subject="LANEIGE",
            predicate=RelationType.HAS_PRODUCT,
            object="B08XYZ1234",
            properties={"product_name": "Lip Sleeping Mask", "category": "lip_care"},
            confidence=0.95,
            source="crawl"
        )

        assert relation.properties["product_name"] == "Lip Sleeping Mask"
        assert relation.confidence == 0.95
        assert relation.source == "crawl"

    def test_relation_validation(self):
        """Relation 유효성 검증"""
        from src.domain.entities.relations import Relation, RelationType

        # subject가 비어있으면 에러
        with pytest.raises(ValueError):
            Relation(
                subject="",
                predicate=RelationType.HAS_PRODUCT,
                object="B08XYZ1234"
            )

        # object가 비어있으면 에러
        with pytest.raises(ValueError):
            Relation(
                subject="LANEIGE",
                predicate=RelationType.HAS_PRODUCT,
                object=""
            )

        # confidence가 범위를 벗어나면 에러
        with pytest.raises(ValueError):
            Relation(
                subject="LANEIGE",
                predicate=RelationType.HAS_PRODUCT,
                object="B08XYZ1234",
                confidence=1.5
            )

    def test_relation_to_dict(self):
        """Relation의 딕셔너리 변환 검증"""
        from src.domain.entities.relations import Relation, RelationType

        relation = Relation(
            subject="LANEIGE",
            predicate=RelationType.HAS_PRODUCT,
            object="B08XYZ1234"
        )

        data = relation.to_dict()

        assert data["subject"] == "LANEIGE"
        assert data["predicate"] == "hasProduct"
        assert data["object"] == "B08XYZ1234"

    def test_relation_from_dict(self):
        """딕셔너리에서 Relation 생성 검증"""
        from src.domain.entities.relations import Relation, RelationType

        data = {
            "subject": "LANEIGE",
            "predicate": "hasProduct",
            "object": "B08XYZ1234",
            "confidence": 0.9
        }

        relation = Relation.from_dict(data)

        assert relation.subject == "LANEIGE"
        assert relation.predicate == RelationType.HAS_PRODUCT
        assert relation.confidence == 0.9

    def test_relation_equality(self):
        """Relation 동등성 비교 검증"""
        from src.domain.entities.relations import Relation, RelationType

        r1 = Relation(
            subject="LANEIGE",
            predicate=RelationType.HAS_PRODUCT,
            object="B08XYZ1234"
        )

        r2 = Relation(
            subject="LANEIGE",
            predicate=RelationType.HAS_PRODUCT,
            object="B08XYZ1234",
            confidence=0.5  # confidence가 달라도 같은 relation
        )

        r3 = Relation(
            subject="LANEIGE",
            predicate=RelationType.HAS_PRODUCT,
            object="B08XYZ5678"  # 다른 object
        )

        assert r1 == r2
        assert r1 != r3

    def test_relation_hash(self):
        """Relation 해시 검증 (중복 제거용)"""
        from src.domain.entities.relations import Relation, RelationType

        r1 = Relation(
            subject="LANEIGE",
            predicate=RelationType.HAS_PRODUCT,
            object="B08XYZ1234"
        )

        r2 = Relation(
            subject="LANEIGE",
            predicate=RelationType.HAS_PRODUCT,
            object="B08XYZ1234"
        )

        # 같은 해시 -> set에서 중복 제거됨
        relations_set = {r1, r2}
        assert len(relations_set) == 1


class TestInsightType:
    """InsightType Enum 테스트"""

    def test_market_position_insights(self):
        """시장 포지션 인사이트 유형 검증"""
        from src.domain.entities.relations import InsightType

        assert InsightType.MARKET_POSITION.value == "market_position"
        assert InsightType.MARKET_DOMINANCE.value == "market_dominance"
        assert InsightType.MARKET_SHARE.value == "market_share"

    def test_risk_insights(self):
        """위험 인사이트 유형 검증"""
        from src.domain.entities.relations import InsightType

        assert InsightType.RISK_ALERT.value == "risk_alert"
        assert InsightType.RANK_SHOCK.value == "rank_shock"
        assert InsightType.RATING_DECLINE.value == "rating_decline"


class TestInferenceResult:
    """InferenceResult 테스트"""

    def test_inference_result_creation(self):
        """InferenceResult 생성 검증"""
        from src.domain.entities.relations import InferenceResult, InsightType

        result = InferenceResult(
            rule_name="market_dominance_rule",
            insight_type=InsightType.MARKET_DOMINANCE,
            insight="LANEIGE가 Lip Care 카테고리에서 시장 지배력을 가짐",
            confidence=0.85,
            evidence={"sos": 15.5, "rank": 1}
        )

        assert result.rule_name == "market_dominance_rule"
        assert result.insight_type == InsightType.MARKET_DOMINANCE
        assert result.confidence == 0.85
        assert result.evidence["sos"] == 15.5

    def test_inference_result_to_dict(self):
        """InferenceResult 딕셔너리 변환 검증"""
        from src.domain.entities.relations import InferenceResult, InsightType

        result = InferenceResult(
            rule_name="test_rule",
            insight_type=InsightType.MARKET_POSITION,
            insight="Test insight",
            recommendation="Take action"
        )

        data = result.to_dict()

        assert data["rule_name"] == "test_rule"
        assert data["insight_type"] == "market_position"
        assert data["recommendation"] == "Take action"


class TestMarketPosition:
    """MarketPosition Enum 테스트"""

    def test_dominant_positions(self):
        """지배적 포지션 유형 검증"""
        from src.domain.entities.relations import MarketPosition

        assert MarketPosition.DOMINANT.value == "dominant"
        assert MarketPosition.DOMINANT_IN_FRAGMENTED.value == "dominant_in_fragmented"

    def test_challenger_positions(self):
        """도전자 포지션 유형 검증"""
        from src.domain.entities.relations import MarketPosition

        assert MarketPosition.CHALLENGER.value == "challenger"
        assert MarketPosition.STRONG_CHALLENGER.value == "strong_challenger"

    def test_follower_positions(self):
        """추종자 포지션 유형 검증"""
        from src.domain.entities.relations import MarketPosition

        assert MarketPosition.FOLLOWER.value == "follower"
        assert MarketPosition.NICHE.value == "niche"


class TestRelationHelpers:
    """관계 생성 헬퍼 함수 테스트"""

    def test_create_brand_product_relation(self):
        """브랜드-제품 관계 생성 헬퍼 검증"""
        from src.domain.entities.relations import (
            create_brand_product_relation,
            RelationType
        )

        relation = create_brand_product_relation(
            brand="LANEIGE",
            product_asin="B08XYZ1234",
            product_name="Lip Sleeping Mask",
            category="lip_care"
        )

        assert relation.subject == "LANEIGE"
        assert relation.predicate == RelationType.HAS_PRODUCT
        assert relation.object == "B08XYZ1234"
        assert relation.properties["product_name"] == "Lip Sleeping Mask"

    def test_create_competition_relation(self):
        """경쟁 관계 생성 헬퍼 검증"""
        from src.domain.entities.relations import (
            create_competition_relation,
            RelationType
        )

        # 직접 경쟁자
        direct = create_competition_relation(
            brand1="LANEIGE",
            brand2="Summer Fridays",
            category="lip_care",
            competition_type="direct"
        )

        assert direct.predicate == RelationType.DIRECT_COMPETITOR

        # 간접 경쟁자
        indirect = create_competition_relation(
            brand1="LANEIGE",
            brand2="Burt's Bees",
            category="lip_care",
            competition_type="indirect"
        )

        assert indirect.predicate == RelationType.INDIRECT_COMPETITOR
