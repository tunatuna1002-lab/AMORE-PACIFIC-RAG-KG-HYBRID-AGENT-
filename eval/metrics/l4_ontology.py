"""
L4 Ontology Metrics
===================
Ontology constraint compliance metrics.

Measures how well the system adheres to ontology rules:
- Constraint Violation Rate: Proportion of inferences that violate constraints
- Type Consistency Rate: Proportion of entities with consistent types
"""

from collections import Counter
from typing import Any

from eval.metrics.base import MetricCalculator
from eval.schemas import GoldEvidence, KGQueryTrace, L4Metrics, OntologyReasoningTrace
from eval.validators.ontology_validator import (
    EntityTypeRegistry,
    OntologyValidator,
    gold_edge_types,
)


class L4OntologyMetrics(MetricCalculator):
    """
    L4 metrics for ontology compliance.

    Uses OntologyValidator to check constraints and type consistency.
    """

    def __init__(
        self,
        validator: OntologyValidator | None = None,
        registry: EntityTypeRegistry | None = None,
    ):
        """
        Initialize L4 metrics calculator.

        Args:
            validator: OntologyValidator instance (creates new if not provided)
            registry: 기대 타입 등록부 (없으면 config 기반 기본 등록부 — 온톨로지 로더가
                있으면 그쪽이 우선한다)
        """
        self.validator = validator or OntologyValidator()
        self.registry = registry

    def compute(
        self,
        ontology_trace: OntologyReasoningTrace,
        kg_trace: KGQueryTrace,
        gold: GoldEvidence,
        rule_evaluation: dict[str, Any] | None = None,
    ) -> L4Metrics:
        """
        Compute L4 metrics.

        Args:
            ontology_trace: Ontology reasoning trace
            kg_trace: KG query trace (for type consistency)
            gold: Gold standard evidence — [2026-09 사후] 기대 타입의 원천
                (kg_entity_types, 없으면 kg_edges 시그니처)
            rule_evaluation: trace.rule_evaluation (발화 규칙 목록 ``fired``). 없으면
                inferences 전체를 검사한다.

        Returns:
            L4Metrics. 레거시 두 필드(constraint_violation_rate·type_consistency_rate)는
            과거 비교를 위해 계산식을 바꾸지 않았다 — 233문항 전부 0.0/1.0으로 고정되던
            그 값이다. 새 지표는 rule_constraint_violation_rate·typed_consistency_rate.
        """
        violation_rate = self._compute_constraint_violation_rate(ontology_trace)
        consistency_rate = self._compute_type_consistency_rate(kg_trace)

        return L4Metrics(
            constraint_violation_rate=violation_rate,
            type_consistency_rate=consistency_rate,
            **self.compute_rule_constraints(ontology_trace, rule_evaluation),
            **self.compute_typed_consistency(kg_trace, gold),
        )

    def compute_rule_constraints(
        self,
        ontology_trace: OntologyReasoningTrace,
        rule_evaluation: dict[str, Any] | None,
    ) -> dict[str, Any]:
        """[2026-09 사후] O0-A: 발화 규칙 추론의 제약 위반 비율.

        rate = 위반이 하나 이상인 추론 수 / 검사한 추론 수. 검사한 추론이 0이면 None
        (규칙이 발화하지 않은 문항은 "위반 0"이 아니라 "판정 없음"이다). 무엇을 위반으로
        세는지는 ``OntologyValidator.check_rule_inferences`` docstring에 정의돼 있다.
        """
        result = self.validator.check_rule_inferences(
            ontology_trace.inferences, rule_evaluation, registry=self.registry
        )
        checked = result["checked"]
        return {
            "rule_constraint_violation_rate": (result["violating"] / checked) if checked else None,
            "rule_checked_inferences": checked,
            "rule_violating_inferences": result["violating"],
            "rule_fired_unchecked": result["fired_unchecked"],
            "rule_violation_kinds": result["kinds"],
        }

    def compute_typed_consistency(
        self, kg_trace: KGQueryTrace, gold: GoldEvidence
    ) -> dict[str, Any]:
        """[2026-09 사후] O0-A: 기대 타입 기준 타입 일관성.

        기대 타입 = 골드 명시 타입(``gold.kg_entity_types``) > 골드 엣지 시그니처
        (``gold_edge_types``) > 온톨로지 로더 > config 등록부 > ASIN 모양. 골드에 명시 타입이
        없으면 ``types_registry_derived=True``로 표시한다(현재 골든셋 전부 해당).
        검사 단위와 위반 정의는 ``OntologyValidator.check_trace_types`` 참조.
        """
        gold_types = gold_edge_types(gold.kg_edges)
        gold_types.update(gold.kg_entity_types or {})
        result = self.validator.check_trace_types(
            kg_trace.kg_edges_found,
            kg_trace.ontology_facts,
            gold_types=gold_types,
            registry=self.registry,
        )
        checks = result["checks"]
        sources = sorted(result["sources"])
        return {
            "typed_consistency_rate": (result["consistent"] / checks) if checks else None,
            "type_checks": checks,
            "type_violations": result["violations"],
            "type_untyped": result["untyped"],
            "type_source": "+".join(sources) if sources else "none",
            "types_registry_derived": not bool(gold.kg_entity_types),
            "type_violation_kinds": result["violation_kinds"],
        }

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


def aggregate_l4_extended(metrics: list[L4Metrics]) -> dict[str, Any]:
    """[2026-09 사후] O0-A: 문항별 L4Metrics의 새 지표를 리포트 요약용으로 집계한다.

    - ``rule_constraint_violation_rate``: 추론을 1건 이상 검사한 문항의 문항 평균(macro)
    - ``rule_violation_micro``: 위반 추론 수 / 검사 추론 수
    - ``typed_consistency_rate``: 타입 검사가 1건 이상인 문항의 문항 평균(macro)
    - ``typed_consistency_micro``: 일치 건수 / 검사 건수
    - ``type_source_counts``: 문항별 type_source 분포, ``types_registry_derived_items``
    """
    rule_rates = [
        m.rule_constraint_violation_rate
        for m in metrics
        if m.rule_constraint_violation_rate is not None
    ]
    checked = sum(m.rule_checked_inferences for m in metrics)
    violating = sum(m.rule_violating_inferences for m in metrics)
    rule_kinds: Counter[str] = Counter()
    type_kinds: Counter[str] = Counter()
    for m in metrics:
        rule_kinds.update(m.rule_violation_kinds)
        type_kinds.update(m.type_violation_kinds)
    type_rates = [m.typed_consistency_rate for m in metrics if m.typed_consistency_rate is not None]
    type_checks = sum(m.type_checks for m in metrics)
    type_violations = sum(m.type_violations for m in metrics)
    return {
        "items": len(metrics),
        "constraint_violation_rate_legacy": (
            sum(m.constraint_violation_rate for m in metrics) / len(metrics) if metrics else None
        ),
        "type_consistency_rate_legacy": (
            sum(m.type_consistency_rate for m in metrics) / len(metrics) if metrics else None
        ),
        "rule_checked_items": len(rule_rates),
        "rule_constraint_violation_rate": (sum(rule_rates) / len(rule_rates))
        if rule_rates
        else None,
        "rule_checked_inferences": checked,
        "rule_violating_inferences": violating,
        "rule_violation_micro": (violating / checked) if checked else None,
        "rule_fired_unchecked": sum(m.rule_fired_unchecked for m in metrics),
        "rule_violation_kinds": dict(rule_kinds.most_common()),
        "type_checked_items": len(type_rates),
        "typed_consistency_rate": (sum(type_rates) / len(type_rates)) if type_rates else None,
        "type_checks": type_checks,
        "type_violations": type_violations,
        "typed_consistency_micro": ((type_checks - type_violations) / type_checks)
        if type_checks
        else None,
        "type_untyped": sum(m.type_untyped for m in metrics),
        "type_source_counts": dict(Counter(m.type_source for m in metrics).most_common()),
        "types_registry_derived_items": sum(1 for m in metrics if m.types_registry_derived),
        "type_violation_kinds": dict(type_kinds.most_common()),
    }
