"""
Business Rules - backward compatibility layer.
Rules are now organized in src/ontology/rules/ subdirectory.

규칙 구성:
1. 시장 포지션 규칙 (Market Position) - rules/market_rules.py
2. 경쟁 위협 규칙 (Competitive Threat) - rules/alert_rules.py
3. 성장 기회 규칙 (Growth Opportunity) - rules/growth_rules.py
4. 위험 경고 규칙 (Risk Alert) - rules/alert_rules.py
5. 가격 포지션 규칙 (Price Position) - rules/price_rules.py
6. 순위-할인 인과관계 규칙 (Rank-Discount Causality) - rules/price_rules.py
7. 감성 분석 규칙 (Sentiment Analysis) - rules/sentiment_rules.py
8. IR 크로스 분석 규칙 (IR Cross-Analysis) - rules/ir_rules.py
"""

from .rules import (
    ALERT_RULES,
    ALL_BUSINESS_RULES,
    GROWTH_RULES,
    IR_CROSS_ANALYSIS_RULES,
    MARKET_RULES,
    PRICE_RULES,
    SENTIMENT_RULES,
    calculate_discount_dependency_score,
    calculate_premium_defense_index,
    count_period_overlap,
    get_high_priority_rules,
    get_ir_rules,
    get_rules_by_category,
    register_all_rules,
    rule_bestseller_badge_effect,
    rule_discount_dependent,
    rule_viral_effect,
)

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
