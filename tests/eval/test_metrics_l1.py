"""Tests for L1 query metrics."""

import pytest

from eval.metrics.l1_query import (
    L1QueryMetrics,
    concept_map_f1,
    constraint_extraction_f1,
    entity_link_f1,
)
from eval.schemas import EntityLinkingTrace, GoldEvidence, OntologyReasoningTrace


class TestL1QueryMetrics:
    """Tests for L1 query metrics."""

    @pytest.fixture
    def calculator(self):
        """Create L1 metrics calculator."""
        return L1QueryMetrics()

    def test_entity_link_f1_exact_match(self, calculator):
        """Test entity linking F1 with exact match."""
        trace = EntityLinkingTrace(
            extracted_brands=["laneige", "cosrx"],
            extracted_categories=[],
            extracted_indicators=[],
            extracted_products=[],
        )
        gold = GoldEvidence(kg_entities=["laneige", "cosrx"])

        f1 = calculator._compute_entity_link_f1(trace, gold)
        assert f1 == 1.0

    def test_entity_link_f1_partial_match(self, calculator):
        """Test entity linking F1 with partial match."""
        trace = EntityLinkingTrace(
            extracted_brands=["laneige"],
            extracted_categories=[],
            extracted_indicators=[],
            extracted_products=[],
        )
        gold = GoldEvidence(kg_entities=["laneige", "cosrx"])

        f1 = calculator._compute_entity_link_f1(trace, gold)
        # Precision: 1/1 = 1.0, Recall: 1/2 = 0.5
        # F1 = 2 * (1.0 * 0.5) / (1.0 + 0.5) = 0.667
        assert 0.66 < f1 < 0.67

    def test_entity_link_f1_no_match(self, calculator):
        """Test entity linking F1 with no match."""
        trace = EntityLinkingTrace(
            extracted_brands=["innisfree"],
            extracted_categories=[],
            extracted_indicators=[],
            extracted_products=[],
        )
        gold = GoldEvidence(kg_entities=["laneige", "cosrx"])

        f1 = calculator._compute_entity_link_f1(trace, gold)
        assert f1 == 0.0

    def test_entity_link_f1_empty_gold(self, calculator):
        """Test entity linking F1 with empty gold."""
        trace = EntityLinkingTrace(
            extracted_brands=["laneige"],
            extracted_categories=[],
            extracted_indicators=[],
            extracted_products=[],
        )
        gold = GoldEvidence(kg_entities=[])

        f1 = calculator._compute_entity_link_f1(trace, gold)
        assert f1 == 0.0  # Predicted but no gold

    def test_entity_link_f1_both_empty(self, calculator):
        """Test entity linking F1 with both empty."""
        trace = EntityLinkingTrace(
            extracted_brands=[],
            extracted_categories=[],
            extracted_indicators=[],
            extracted_products=[],
        )
        gold = GoldEvidence(kg_entities=[])

        f1 = calculator._compute_entity_link_f1(trace, gold)
        assert f1 == 1.0  # Both empty = perfect match

    def test_entity_link_f1_case_insensitive(self, calculator):
        """Test entity linking F1 is case insensitive."""
        trace = EntityLinkingTrace(
            extracted_brands=["LANEIGE"],
            extracted_categories=[],
            extracted_indicators=[],
            extracted_products=[],
        )
        gold = GoldEvidence(kg_entities=["laneige"])

        f1 = calculator._compute_entity_link_f1(trace, gold)
        assert f1 == 1.0

    def test_concept_map_f1_exact_match(self, calculator):
        """Test concept mapping F1 with exact match."""
        trace = EntityLinkingTrace(
            extracted_brands=[],
            extracted_categories=["lip_care", "skin_care"],
            extracted_indicators=[],
            extracted_products=[],
        )
        gold = GoldEvidence(concepts=["lip_care", "skin_care"])

        f1 = calculator._compute_concept_map_f1(trace, gold)
        assert f1 == 1.0

    def test_constraint_extraction_f1(self, calculator):
        """Test constraint extraction F1."""
        ontology_trace = OntologyReasoningTrace(
            inferences=[],
            applied_rules=["market_dominance", "competitor_threat"],
            constraint_violations=[],
        )
        gold = GoldEvidence(constraints=["market_dominance", "competitor_threat"])

        f1 = calculator._compute_constraint_extraction_f1(ontology_trace, gold)
        assert f1 == 1.0

    def test_compute_full_metrics(self, calculator):
        """Test full L1 metrics computation."""
        entity_trace = EntityLinkingTrace(
            extracted_brands=["laneige"],
            extracted_categories=["lip_care"],
            extracted_indicators=[],  # Empty to match gold
            extracted_products=[],
        )
        ontology_trace = OntologyReasoningTrace(
            inferences=[],
            applied_rules=["market_dominance"],
            constraint_violations=[],
        )
        gold = GoldEvidence(
            # 골드 kg_entities는 브랜드·제품·지표뿐 아니라 카테고리 ID도 열거하므로
            # entity 집합에 추출 카테고리가 포함된다 (사이클 4 필드 매핑 수정)
            kg_entities=["laneige", "lip_care"],
            concepts=["lip_care"],
            constraints=["market_dominance"],
        )

        metrics = calculator.compute(entity_trace, ontology_trace, gold)

        assert metrics.entity_link_f1 == 1.0
        assert metrics.concept_map_f1 == 1.0
        assert metrics.constraint_extraction_f1 == 1.0

    def test_entity_link_f1_counts_categories(self, calculator):
        """골드 kg_entities의 카테고리 ID가 추출 카테고리와 매칭돼야 한다."""
        entity_trace = EntityLinkingTrace(
            extracted_brands=[],
            extracted_categories=["face_powder"],
            extracted_indicators=[],
            extracted_products=[],
        )
        gold = GoldEvidence(kg_entities=["face_powder"])

        assert calculator._compute_entity_link_f1(entity_trace, gold) == 1.0

    def test_dict_constraints_do_not_crash(self, calculator):
        """dict 형태 골드 constraint가 채점을 크래시시키지 않아야 한다.

        회귀 방지: v4.1 실행에서 dict constraint 7문항이 set() TypeError로
        문항 전체 채점을 실패시켜 응답조차 기록되지 않았다.
        """
        ontology_trace = OntologyReasoningTrace(applied_rules=["period:=:last_3_months"])
        gold = GoldEvidence(
            constraints=[{"field": "period", "operator": "=", "value": "last_3_months"}]
        )

        assert calculator._compute_constraint_extraction_f1(ontology_trace, gold) == 1.0

    def test_dict_constraints_no_match(self, calculator):
        """dict constraint가 정규화돼도 매칭되지 않으면 0.0."""
        ontology_trace = OntologyReasoningTrace(applied_rules=["market_dominance"])
        gold = GoldEvidence(constraints=[{"field": "rating", "operator": ">=", "value": 4.5}])

        assert calculator._compute_constraint_extraction_f1(ontology_trace, gold) == 0.0


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_entity_link_f1_function(self):
        """Test entity_link_f1 convenience function."""
        trace = EntityLinkingTrace(
            extracted_brands=["laneige"],
            extracted_categories=[],
            extracted_indicators=[],
            extracted_products=[],
        )
        gold = GoldEvidence(kg_entities=["laneige"])

        f1 = entity_link_f1(trace, gold)
        assert f1 == 1.0

    def test_concept_map_f1_function(self):
        """Test concept_map_f1 convenience function."""
        trace = EntityLinkingTrace(
            extracted_brands=[],
            extracted_categories=["lip_care"],
            extracted_indicators=[],
            extracted_products=[],
        )
        gold = GoldEvidence(concepts=["lip_care"])

        f1 = concept_map_f1(trace, gold)
        assert f1 == 1.0

    def test_constraint_extraction_f1_function(self):
        """Test constraint_extraction_f1 convenience function."""
        trace = OntologyReasoningTrace(
            inferences=[],
            applied_rules=["rule1"],
            constraint_violations=[],
        )
        gold = GoldEvidence(constraints=["rule1"])

        f1 = constraint_extraction_f1(trace, gold)
        assert f1 == 1.0
