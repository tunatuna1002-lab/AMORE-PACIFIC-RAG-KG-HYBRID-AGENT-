"""
IR Cross-Analysis Rules
IR 크로스 분석 규칙 (2026-01-26 추가)
"""

from ..reasoner import InferenceRule, RuleCondition
from ..relations import InsightType

RULE_IR_PRIME_DAY_IMPACT = InferenceRule(
    name="ir_prime_day_impact",
    description="IR Prime Day 성과 언급 + Amazon 해당 기간 순위 급등 = Prime Day 영향",
    conditions=[
        RuleCondition(
            name="ir_mentions_prime_day",
            check=lambda ctx: ctx.get("ir_mentions_prime_day", False),
            description="IR에서 Prime Day 언급",
        ),
        RuleCondition(
            name="amazon_rank_surge",
            check=lambda ctx: ctx.get("rank_change_during_event", 0) < -10,
            description="이벤트 기간 순위 10+ 상승",
        ),
    ],
    conclusion=lambda ctx: {
        "insight": f"IR 보고서에서 Prime Day 성과가 언급되었고, "
        f"실제 Amazon에서 해당 기간 순위가 {abs(ctx.get('rank_change_during_event', 0))}단계 상승했습니다. "
        f"Prime Day 프로모션이 순위 상승의 주요 원인으로 판단됩니다.",
        "causality": "prime_day_impact",
        "confidence": 0.85,
        "recommendation": "다음 Prime Day를 위한 재고 확보 및 프로모션 전략 수립 필요.",
        "related_entities": [ctx.get("brand", ""), ctx.get("category", "")],
        "metadata": {
            "ir_evidence": "IR 2025 Q3",
            "event": "Prime Day",
            "rank_change": ctx.get("rank_change_during_event"),
        },
    },
    insight_type=InsightType.GROWTH_MOMENTUM,
    priority=9,
    confidence=0.85,
    tags=["ir", "causality", "prime_day", "positive"],
)


RULE_IR_AMERICAS_CORRELATION = InferenceRule(
    name="ir_americas_revenue_correlation",
    description="IR Americas 매출 증가 + LANEIGE SoS 상승 = IR 실적이 Amazon에 반영",
    conditions=[
        RuleCondition(
            name="ir_americas_growth",
            check=lambda ctx: ctx.get("ir_americas_yoy", 0) > 0,
            description="IR Americas 매출 YoY 증가",
        ),
        RuleCondition(
            name="sos_increase",
            check=lambda ctx: ctx.get("sos_change", 0) > 0,
            description="SoS 상승",
        ),
        RuleCondition(
            name="is_target_brand",
            check=lambda ctx: ctx.get("is_target", False),
            description="타겟 브랜드",
        ),
    ],
    conclusion=lambda ctx: {
        "insight": f"아모레퍼시픽 IR Americas 매출이 전년 대비 {ctx.get('ir_americas_yoy', 0):+.1f}% 성장했고, "
        f"LANEIGE Amazon SoS도 {ctx.get('sos_change', 0)*100:+.1f}%p 상승했습니다. "
        f"IR 실적 성장이 Amazon 점유율에 반영된 것으로 판단됩니다.",
        "correlation": "ir_amazon_positive",
        "confidence": 0.80,
        "recommendation": "현재 성장 모멘텀 유지. 지역별 전략 강화.",
        "related_entities": [ctx.get("brand", "LANEIGE"), "AMOREPACIFIC"],
        "metadata": {
            "ir_americas_yoy": ctx.get("ir_americas_yoy"),
            "sos_change": ctx.get("sos_change"),
            "ir_source": "AP_3Q25_EN.md",
        },
    },
    insight_type=InsightType.MARKET_POSITION,
    priority=8,
    confidence=0.80,
    tags=["ir", "correlation", "americas", "positive"],
)


RULE_IR_GROWTH_MOMENTUM = InferenceRule(
    name="ir_growth_momentum",
    description="IR 2분기 이상 연속 성장 = Amazon 지속 성장 가능성 높음",
    conditions=[
        RuleCondition(
            name="ir_consecutive_growth",
            check=lambda ctx: ctx.get("ir_consecutive_growth_quarters", 0) >= 2,
            description="IR 2분기 이상 연속 성장",
        ),
    ],
    conclusion=lambda ctx: {
        "insight": f"아모레퍼시픽 IR에서 {ctx.get('ir_consecutive_growth_quarters', 0)}분기 연속 "
        f"성장이 확인되어 Amazon 성과도 지속 성장 가능성이 높습니다.",
        "momentum": "sustained_growth",
        "confidence": 0.70,
        "recommendation": "장기 성장 전략 수립. 신제품 출시 적기 검토.",
        "related_entities": [ctx.get("brand", ""), "AMOREPACIFIC"],
        "metadata": {
            "consecutive_quarters": ctx.get("ir_consecutive_growth_quarters"),
            "ir_sources": ["AP_1Q25_EN.md", "AP_2Q25_EN.md", "AP_3Q25_EN.md"],
        },
    },
    insight_type=InsightType.GROWTH_MOMENTUM,
    priority=7,
    confidence=0.70,
    tags=["ir", "momentum", "growth", "positive"],
)


RULE_IR_GROWTH_SLOWDOWN = InferenceRule(
    name="ir_growth_slowdown_warning",
    description="IR 성장률 전분기 대비 50% 이상 감소 = Amazon 점유율 하락 리스크",
    conditions=[
        RuleCondition(
            name="ir_growth_slowdown",
            check=lambda ctx: (
                ctx.get("ir_current_qtr_growth", 0) < ctx.get("ir_prev_qtr_growth", 0) * 0.5
            )
            if ctx.get("ir_prev_qtr_growth", 0) > 0
            else False,
            description="IR 성장률 50%+ 감소",
        ),
    ],
    conclusion=lambda ctx: {
        "insight": f"IR 성장률이 전분기 {ctx.get('ir_prev_qtr_growth', 0):.1f}%에서 "
        f"현재 {ctx.get('ir_current_qtr_growth', 0):.1f}%로 급감했습니다. "
        f"Amazon 점유율 하락 리스크를 모니터링해야 합니다.",
        "risk": "growth_slowdown",
        "confidence": 0.65,
        "recommendation": "원인 분석 및 대응 전략 수립. 경쟁사 동향 모니터링 강화.",
        "related_entities": [ctx.get("brand", ""), "AMOREPACIFIC"],
        "metadata": {
            "prev_growth": ctx.get("ir_prev_qtr_growth"),
            "current_growth": ctx.get("ir_current_qtr_growth"),
            "decline_ratio": (
                ctx.get("ir_prev_qtr_growth", 1) - ctx.get("ir_current_qtr_growth", 0)
            )
            / ctx.get("ir_prev_qtr_growth", 1)
            * 100
            if ctx.get("ir_prev_qtr_growth", 0) > 0
            else 0,
        },
    },
    insight_type=InsightType.RISK_ALERT,
    priority=8,
    confidence=0.65,
    tags=["ir", "risk", "slowdown", "alert"],
)


RULE_IR_BRAND_CAMPAIGN_EFFECT = InferenceRule(
    name="ir_brand_campaign_effect",
    description="IR 캠페인 언급 + Amazon 해당 제품 순위 상승 = 캠페인 효과 확인",
    conditions=[
        RuleCondition(
            name="ir_mentions_campaign",
            check=lambda ctx: ctx.get("ir_campaign_mentioned", False),
            description="IR에서 마케팅 캠페인 언급",
        ),
        RuleCondition(
            name="product_rank_improved",
            check=lambda ctx: ctx.get("rank_change_7d", 0) < 0,
            description="해당 제품 순위 상승",
        ),
    ],
    conclusion=lambda ctx: {
        "insight": f"IR에서 언급된 '{ctx.get('campaign_name', '마케팅 캠페인')}'이 "
        f"Amazon에서 순위 상승({abs(ctx.get('rank_change_7d', 0))}단계)으로 연결되었습니다. "
        f"마케팅 캠페인 효과가 Amazon 성과에 반영된 것으로 판단됩니다.",
        "causality": "campaign_effect",
        "confidence": 0.75,
        "recommendation": "효과적인 캠페인 전략 지속 및 확대 검토.",
        "related_entities": [ctx.get("brand", ""), ctx.get("asin", "")],
        "metadata": {
            "campaign": ctx.get("campaign_name"),
            "rank_change": ctx.get("rank_change_7d"),
            "ir_source": ctx.get("ir_source"),
        },
    },
    insight_type=InsightType.GROWTH_MOMENTUM,
    priority=7,
    confidence=0.75,
    tags=["ir", "campaign", "causality", "positive"],
)


RULE_BRAND_OWNERSHIP_VERIFICATION = InferenceRule(
    name="brand_ownership_verification",
    description="브랜드 소유권 검증 (COSRX = 한국 브랜드, 아모레퍼시픽 소속)",
    conditions=[
        RuleCondition(
            name="brand_in_amorepacific",
            check=lambda ctx: ctx.get("parent_group") == "AMOREPACIFIC",
            description="아모레퍼시픽 그룹 소속",
        ),
    ],
    conclusion=lambda ctx: {
        "insight": f"{ctx.get('brand', '해당 브랜드')}는 아모레퍼시픽 그룹 소속 브랜드입니다. "
        f"원산지: {ctx.get('country_of_origin', 'Korea')}. "
        f"{'인수 연도: ' + str(ctx.get('acquired')) if ctx.get('acquired') else ''}",
        "verification": "brand_ownership",
        "confidence": 1.0,
        "recommendation": "그룹 시너지 활용 및 자매 브랜드와의 협력 전략 검토.",
        "related_entities": [ctx.get("brand", ""), "AMOREPACIFIC"],
        "metadata": {
            "parent_group": "AMOREPACIFIC",
            "country_of_origin": ctx.get("country_of_origin", "Korea"),
            "acquired": ctx.get("acquired"),
            "segment": ctx.get("segment"),
            "evidence": ctx.get("evidence", ["config/brands.json", "IR Reports"]),
        },
    },
    insight_type=InsightType.MARKET_POSITION,
    priority=5,
    confidence=1.0,
    tags=["brand", "ownership", "verification"],
)


IR_CROSS_ANALYSIS_RULES: list[InferenceRule] = [
    RULE_IR_PRIME_DAY_IMPACT,
    RULE_IR_AMERICAS_CORRELATION,
    RULE_IR_GROWTH_MOMENTUM,
    RULE_IR_GROWTH_SLOWDOWN,
    RULE_IR_BRAND_CAMPAIGN_EFFECT,
    RULE_BRAND_OWNERSHIP_VERIFICATION,
]
