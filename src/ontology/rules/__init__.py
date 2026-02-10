"""
Business Rules - Modular Organization
Rules are categorized by domain for better maintainability.
"""

from ..reasoner import InferenceRule
from .alert_rules import ALERT_RULES
from .growth_rules import GROWTH_RULES
from .ir_rules import IR_CROSS_ANALYSIS_RULES

# Import categorized rules
from .market_rules import MARKET_RULES

# Import helper functions (used externally)
from .price_rules import (
    PRICE_RULES,
    calculate_discount_dependency_score,
    calculate_premium_defense_index,
    count_period_overlap,
    rule_bestseller_badge_effect,
    rule_discount_dependent,
    rule_viral_effect,
)
from .sentiment_rules import SENTIMENT_RULES

# Assemble all rules
ALL_BUSINESS_RULES: list[InferenceRule] = [
    *MARKET_RULES,
    *ALERT_RULES,
    *GROWTH_RULES,
    *PRICE_RULES,
    *SENTIMENT_RULES,
    *IR_CROSS_ANALYSIS_RULES,
]


def register_all_rules(reasoner) -> int:
    """
    모든 비즈니스 규칙을 추론 엔진에 등록

    Args:
        reasoner: OntologyReasoner 인스턴스

    Returns:
        등록된 규칙 수
    """
    reasoner.register_rules(ALL_BUSINESS_RULES)
    return len(ALL_BUSINESS_RULES)


def get_rules_by_category(category: str) -> list[InferenceRule]:
    """
    카테고리별 규칙 필터링

    Args:
        category: "market", "risk", "growth", "price" 등

    Returns:
        해당 카테고리의 규칙 리스트
    """
    return [r for r in ALL_BUSINESS_RULES if category in r.tags]


def get_high_priority_rules(min_priority: int = 7) -> list[InferenceRule]:
    """높은 우선순위 규칙만 필터링"""
    return [r for r in ALL_BUSINESS_RULES if r.priority >= min_priority]


def get_ir_rules() -> list[InferenceRule]:
    """IR 크로스 분석 규칙만 반환"""
    return IR_CROSS_ANALYSIS_RULES


__all__ = [
    "ALL_BUSINESS_RULES",
    "MARKET_RULES",
    "GROWTH_RULES",
    "PRICE_RULES",
    "ALERT_RULES",
    "SENTIMENT_RULES",
    "IR_CROSS_ANALYSIS_RULES",
    "register_all_rules",
    "get_rules_by_category",
    "get_high_priority_rules",
    "get_ir_rules",
    "count_period_overlap",
    "calculate_discount_dependency_score",
    "calculate_premium_defense_index",
    "rule_discount_dependent",
    "rule_viral_effect",
    "rule_bestseller_badge_effect",
]
