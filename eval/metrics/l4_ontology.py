"""
L4 Ontology Metrics
===================
Ontology constraint compliance metrics.

Measures how well the system adheres to ontology rules:
- Constraint Violation Rate: Proportion of inferences that violate constraints
- Type Consistency Rate: Proportion of entities with consistent types
"""

from eval.metrics.base import MetricCalculator
from eval.schemas import GoldEvidence, KGQueryTrace, L4Metrics, OntologyReasoningTrace
from eval.validators.ontology_validator import OntologyValidator


class L4OntologyMetrics(MetricCalculator):
    """
    L4 metrics for ontology compliance.

    Uses OntologyValidator to check constraints and type consistency.
    """

    def __init__(self, validator: OntologyValidator | None = None):
        """
        Initialize L4 metrics calculator.

        Args:
            validator: OntologyValidator instance (creates new if not provided)
        """
        self.validator = validator or OntologyValidator()

    def compute(
        self,
        ontology_trace: OntologyReasoningTrace,
        kg_trace: KGQueryTrace,
        gold: GoldEvidence,
    ) -> L4Metrics:
        """
        Compute L4 metrics.

        Args:
            ontology_trace: Ontology reasoning trace
            kg_trace: KG query trace (for type consistency)
            gold: Gold standard evidence (unused but kept for interface consistency)

        Returns:
            L4Metrics with constraint_violation_rate and type_consistency_rate
        """
        violation_rate = self._compute_constraint_violation_rate(ontology_trace)
        consistency_rate = self._compute_type_consistency_rate(kg_trace)

        return L4Metrics(
            constraint_violation_rate=violation_rate,
            type_consistency_rate=consistency_rate,
        )

    def _compute_constraint_violation_rate(self, trace: OntologyReasoningTrace) -> float:
        """
        Compute constraint violation rate.

        Validates all inferences and returns proportion of violations.
        """
        if not trace.inferences:
            return 0.0  # No inferences = no violations

        # Also count explicit violations in trace
        explicit_violations = len(trace.constraint_violations)

        # Validate inferences
        validation_errors = self.validator.validate_inferences(trace.inferences)

        total_checks = len(trace.inferences) + max(1, explicit_violations)
        total_violations = len(validation_errors) + explicit_violations

        return min(1.0, total_violations / total_checks)

    def _compute_type_consistency_rate(self, trace: KGQueryTrace) -> float:
        """
        Compute type consistency rate.

        Checks if all entities have consistent types based on ontology rules.
        """
        all_entities = list(trace.kg_entities_found)

        if not all_entities:
            return 1.0  # No entities = perfect consistency

        consistency_rate, _ = self.validator.check_type_consistency(all_entities)
        return consistency_rate


def constraint_violation_rate(
    trace: OntologyReasoningTrace,
    validator: OntologyValidator | None = None,
) -> float:
    """
    Convenience function for constraint violation rate.

    Args:
        trace: Ontology reasoning trace
        validator: Optional validator instance

    Returns:
        Proportion of constraint violations (0.0-1.0)
    """
    calc = L4OntologyMetrics(validator=validator)
    return calc._compute_constraint_violation_rate(trace)


def type_consistency_rate(
    trace: KGQueryTrace,
    validator: OntologyValidator | None = None,
) -> float:
    """
    Convenience function for type consistency rate.

    Args:
        trace: KG query trace
        validator: Optional validator instance

    Returns:
        Proportion of consistent entity types (0.0-1.0)
    """
    calc = L4OntologyMetrics(validator=validator)
    return calc._compute_type_consistency_rate(trace)


def validate_inference_quality(
    trace: OntologyReasoningTrace,
    validator: OntologyValidator | None = None,
) -> tuple[float, list[str]]:
    """
    Validate inference quality and return detailed errors.

    Args:
        trace: Ontology reasoning trace
        validator: Optional validator instance

    Returns:
        Tuple of (quality_score, list of error messages)
    """
    validator = validator or OntologyValidator()

    errors = validator.validate_inferences(trace.inferences)

    if not trace.inferences:
        return 1.0, []

    quality_score = 1.0 - (len(errors) / len(trace.inferences))
    return max(0.0, quality_score), errors


def validate_kg_facts_quality(
    ontology_facts: list[dict],
    validator: OntologyValidator | None = None,
) -> tuple[float, list[str]]:
    """
    Validate KG facts quality and return detailed errors.

    Args:
        ontology_facts: List of ontology fact dicts
        validator: Optional validator instance

    Returns:
        Tuple of (quality_score, list of error messages)
    """
    validator = validator or OntologyValidator()

    errors = validator.validate_kg_facts(ontology_facts)

    if not ontology_facts:
        return 1.0, []

    quality_score = 1.0 - (len(errors) / len(ontology_facts))
    return max(0.0, quality_score), errors
