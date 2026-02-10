"""
Risk and Alert Rules
위험 및 경고 관련 규칙
"""

from ..reasoner import InferenceRule, RuleCondition, StandardConditions
from ..relations import InsightType

# =========================================================================
# 규칙: 가격-품질 불일치 (Price-Quality Mismatch)
# =========================================================================

RULE_PRICE_QUALITY_MISMATCH = InferenceRule(
    name="price_quality_mismatch",
    description="높은 CPI와 낮은 평점 갭은 가격 대비 품질 인식 문제 가능성을 나타냄",
    conditions=[
        StandardConditions.cpi_above(110),  # CPI > 110 (프리미엄)
        StandardConditions.rating_gap_negative(),  # 평점 경쟁 열위
    ],
    conclusion=lambda ctx: {
        "insight": f"프리미엄 가격 포지션(CPI: {ctx.get('cpi', 100):.0f}) 대비 "
        f"평점 경쟁력이 열위하여 가격-품질 인식 간 불일치 가능성이 있습니다.",
        "risk": "price_perception",
        "recommendation": "제품 품질 개선 또는 가격 전략 재검토 고려. "
        "고객 리뷰 분석을 통한 불만 요인 파악 권장.",
        "related_entities": [ctx.get("brand", ""), ctx.get("asin", "")],
        "metadata": {"cpi": ctx.get("cpi"), "rating_gap": ctx.get("rating_gap")},
        "confidence_modifier": 0.95,
    },
    insight_type=InsightType.PRICE_QUALITY_GAP,
    priority=9,
    confidence=0.85,
    tags=["price", "quality", "risk"],
)


# =========================================================================
# 규칙: 시장 구조 변화 (Market Disruption)
# =========================================================================

RULE_MARKET_DISRUPTION = InferenceRule(
    name="market_disruption",
    description="순위 급변과 높은 Churn Rate는 시장 구조 변화 신호",
    conditions=[
        StandardConditions.has_rank_shock(),
        StandardConditions.churn_rate_high(0.2),
    ],
    conclusion=lambda ctx: {
        "insight": f"순위 급변과 높은 시장 변동성(Churn Rate: {ctx.get('churn_rate', 0)*100:.1f}%)이 "
        f"동시에 관측되어 시장 구조 변화 가능성이 있습니다.",
        "signal": "market_disruption",
        "recommendation": "신규 진입자/이탈자 분석 및 경쟁 구도 변화 모니터링 필요. "
        "단기 노이즈 가능성도 있어 추이 확인 권장.",
        "related_entities": ctx.get("products", [])[:5] if ctx.get("products") else [],
        "metadata": {"churn_rate": ctx.get("churn_rate"), "has_rank_shock": True},
    },
    insight_type=InsightType.COMPETITIVE_THREAT,
    priority=9,
    confidence=0.8,
    tags=["market", "disruption", "alert"],
)


# =========================================================================
# 규칙: 순위 하락 경고 (Rank Decline Alert)
# =========================================================================

RULE_RANK_DECLINE = InferenceRule(
    name="rank_decline_alert",
    description="순위 하락 추세와 높은 변동성은 포지션 약화 신호",
    conditions=[
        StandardConditions.rank_declining(),
        RuleCondition(
            name="high_volatility",
            check=lambda ctx: ctx.get("rank_volatility", 0) > 5,
            description="순위 변동성 > 5",
        ),
    ],
    conclusion=lambda ctx: {
        "insight": f"순위 하락 추세(7일 변동: +{ctx.get('rank_change_7d', 0)})와 "
        f"높은 변동성(표준편차: {ctx.get('rank_volatility', 0):.1f})이 관측되어 "
        f"포지션 약화 가능성이 있습니다.",
        "risk": "position_weakening",
        "recommendation": "원인 분석 필요: 경쟁사 활동, 리뷰 변화, 가격 경쟁력 점검. "
        "단기 노이즈일 수 있어 추이 모니터링 병행.",
        "related_entities": [ctx.get("asin", "")],
        "metadata": {
            "rank_change_7d": ctx.get("rank_change_7d"),
            "rank_volatility": ctx.get("rank_volatility"),
        },
    },
    insight_type=InsightType.RISK_ALERT,
    priority=8,
    confidence=0.8,
    tags=["rank", "decline", "alert"],
)


ALERT_RULES: list[InferenceRule] = [
    RULE_PRICE_QUALITY_MISMATCH,
    RULE_MARKET_DISRUPTION,
    RULE_RANK_DECLINE,
]
