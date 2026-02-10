"""Tests for L4 ontology metrics."""

import pytest

from eval.metrics.l4_ontology import (
    L4OntologyMetrics,
    constraint_violation_rate,
    type_consistency_rate,
    validate_inference_quality,
)
from eval.schemas import GoldEvidence, KGQueryTrace, OntologyReasoningTrace
from eval.validators.ontology_validator import OntologyValidator


class TestL4OntologyMetrics:
    """Tests for L4 ontology metrics."""

    @pytest.fixture
    def calculator(self):
        """Create L4 metrics calculator."""
        return L4OntologyMetrics()

    @pytest.fixture
    def validator(self):
        """Create ontology validator."""
        return OntologyValidator()

    def test_constraint_violation_rate_no_violations(self, calculator):
        """Test constraint violation rate with no violations."""
        trace = OntologyReasoningTrace(
            inferences=[
                {
                    "rule_name": "market_dominance",
                    "insight_type": "market_position",
                    "insight": "test insight",
                    "confidence": 0.9,
                }
            ],
            applied_rules=["market_dominance"],
            constraint_violations=[],
        )

        rate = calculator._compute_constraint_violation_rate(trace)
        assert rate == 0.0  # No violations

    def test_constraint_violation_rate_with_violations(self, calculator):
        """Test constraint violation rate with violations."""
        trace = OntologyReasoningTrace(
            inferences=[
                {"rule_name": "rule1", "insight_type": "market_position", "insight": "i1"},
                {"rule_name": "rule2", "insight_type": "invalid_type", "insight": "i2"},
            ],
            applied_rules=["rule1", "rule2"],
            constraint_violations=["violation1"],
        )

        rate = calculator._compute_constraint_violation_rate(trace)
        # Has explicit violation + invalid insight_type
        assert rate > 0.0

    def test_constraint_violation_rate_empty_inferences(self, calculator):
        """Test constraint violation rate with empty inferences."""
        trace = OntologyReasoningTrace(
            inferences=[],
            applied_rules=[],
            constraint_violations=[],
        )

        rate = calculator._compute_constraint_violation_rate(trace)
        assert rate == 0.0  # No inferences = no violations

    def test_type_consistency_rate_all_consistent(self, calculator):
        """Test type consistency rate with all consistent types."""
        trace = KGQueryTrace(
            kg_entities_found=["B08XYZ123"],  # ASIN pattern = product type
            kg_edges_found=[],
            ontology_facts=[],
            competitor_network=[],
        )

        rate = calculator._compute_type_consistency_rate(trace)
        assert rate == 1.0

    def test_type_consistency_rate_empty_entities(self, calculator):
        """Test type consistency rate with empty entities."""
        trace = KGQueryTrace(
            kg_entities_found=[],
            kg_edges_found=[],
            ontology_facts=[],
            competitor_network=[],
        )

        rate = calculator._compute_type_consistency_rate(trace)
        assert rate == 1.0  # Empty = perfect consistency

    def test_compute_full_metrics(self, calculator):
        """Test full L4 metrics computation."""
        ontology_trace = OntologyReasoningTrace(
            inferences=[
                {
                    "rule_name": "market_position",
                    "insight_type": "market_position",
                    "insight": "test",
                }
            ],
            applied_rules=["market_position"],
            constraint_violations=[],
        )
        kg_trace = KGQueryTrace(
            kg_entities_found=["laneige", "cosrx"],
            kg_edges_found=[],
            ontology_facts=[],
            competitor_network=[],
        )
        gold = GoldEvidence()

        metrics = calculator.compute(ontology_trace, kg_trace, gold)

        assert 0.0 <= metrics.constraint_violation_rate <= 1.0
        assert 0.0 <= metrics.type_consistency_rate <= 1.0


class TestConvenienceFunctions:
    """Tests for convenience functions."""

    def test_constraint_violation_rate_function(self):
        """Test constraint_violation_rate convenience function."""
        trace = OntologyReasoningTrace(
            inferences=[
                {
                    "rule_name": "rule1",
                    "insight_type": "market_position",
                    "insight": "test",
                }
            ],
            applied_rules=["rule1"],
            constraint_violations=[],
        )

        rate = constraint_violation_rate(trace)
        assert rate == 0.0

    def test_type_consistency_rate_function(self):
        """Test type_consistency_rate convenience function."""
        trace = KGQueryTrace(
            kg_entities_found=["entity1"],
            kg_edges_found=[],
            ontology_facts=[],
            competitor_network=[],
        )

        rate = type_consistency_rate(trace)
        assert rate == 1.0

    def test_validate_inference_quality_function(self):
        """Test validate_inference_quality convenience function."""
        trace = OntologyReasoningTrace(
            inferences=[
                {
                    "rule_name": "rule1",
                    "insight_type": "market_position",
                    "insight": "test",
                }
            ],
            applied_rules=["rule1"],
            constraint_violations=[],
        )

        score, errors = validate_inference_quality(trace)
        assert score == 1.0
        assert len(errors) == 0

    def test_validate_inference_quality_with_errors(self):
        """Test validate_inference_quality with errors."""
        trace = OntologyReasoningTrace(
            inferences=[
                {
                    "rule_name": "rule1",
                    # Missing insight_type and insight
                }
            ],
            applied_rules=["rule1"],
            constraint_violations=[],
        )

        score, errors = validate_inference_quality(trace)
        assert score < 1.0
        assert len(errors) > 0


class TestOntologyValidator:
    """Tests for OntologyValidator."""

    @pytest.fixture
    def validator(self):
        """Create validator instance."""
        return OntologyValidator()

    def test_validate_valid_triple(self, validator):
        """Test validating a valid triple."""
        is_valid, error = validator.validate_triple(
            subject="laneige",
            predicate="hasProduct",
            obj="B08XYZ123",
            subject_type="brand",
            object_type="product",
        )
        assert is_valid
        assert error == ""

    def test_validate_invalid_triple(self, validator):
        """Test validating an invalid triple."""
        is_valid, error = validator.validate_triple(
            subject="laneige",
            predicate="invalidRelation",
            obj="something",
            subject_type="brand",
            object_type="product",
        )
        assert not is_valid
        assert "Invalid relation" in error

    def test_validate_inference_valid(self, validator):
        """Test validating a valid inference."""
        inference = {
            "rule_name": "market_dominance",
            "insight_type": "market_position",
            "insight": "LANEIGE dominates the market",
            "confidence": 0.9,
        }

        errors = validator.validate_inference(inference)
        assert len(errors) == 0

    def test_validate_inference_missing_fields(self, validator):
        """Test validating inference with missing fields."""
        inference = {
            "rule_name": "rule1",
            # Missing insight_type and insight
        }

        errors = validator.validate_inference(inference)
        assert len(errors) > 0

    def test_validate_inference_invalid_confidence(self, validator):
        """Test validating inference with invalid confidence."""
        inference = {
            "rule_name": "rule1",
            "insight_type": "market_position",
            "insight": "test",
            "confidence": 1.5,  # Invalid: > 1.0
        }

        errors = validator.validate_inference(inference)
        assert len(errors) > 0
        assert any("confidence" in e.lower() for e in errors)

    def test_validate_inference_invalid_insight_type(self, validator):
        """Test validating inference with invalid insight type."""
        inference = {
            "rule_name": "rule1",
            "insight_type": "invalid_type",
            "insight": "test",
        }

        errors = validator.validate_inference(inference)
        assert len(errors) > 0

    def test_infer_entity_type_brand(self, validator):
        """Test inferring brand entity type."""
        entity_type = validator.infer_entity_type("laneige")
        assert entity_type == "brand"

    def test_infer_entity_type_product(self, validator):
        """Test inferring product entity type."""
        # ASIN format: B0 + 8 alphanumeric chars
        entity_type = validator.infer_entity_type("B0XYZ12345")
        assert entity_type == "product"

    def test_infer_entity_type_category(self, validator):
        """Test inferring category entity type."""
        entity_type = validator.infer_entity_type("lip_care")
        assert entity_type == "category"

    def test_infer_entity_type_metric(self, validator):
        """Test inferring metric entity type."""
        entity_type = validator.infer_entity_type("sos")
        assert entity_type == "metric"
