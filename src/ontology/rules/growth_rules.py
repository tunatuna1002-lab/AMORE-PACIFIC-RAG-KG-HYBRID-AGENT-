"""
Growth, Opportunity, and Stability Rules
성장, 기회 및 안정성 관련 규칙
"""

from ..reasoner import InferenceRule, RuleCondition, StandardConditions
from ..relations import InsightType, MarketPosition

# =========================================================================
# 규칙: 안정적 성장 (Stable Growth)
# =========================================================================

RULE_STABLE_GROWTH = InferenceRule(
    name="stable_growth",
    description="장기 Top N 체류와 순위 상승 추세는 안정적 성장을 나타냄",
    conditions=[
        StandardConditions.streak_days_above(30),  # 30일 이상 연속 체류
        StandardConditions.rank_improving(),  # 순위 상승 중
    ],
    conclusion=lambda ctx: {
        "insight": f"Top N 연속 {ctx.get('streak_days', 0)}일 체류 중이며 "
        f"순위 상승 추세(7일 변동: {ctx.get('rank_change_7d', 0)})를 보여 "
        f"안정적인 성장 모멘텀을 유지하고 있습니다.",
        "position": "stable_growth",
        "recommendation": "현재 전략 유지 권장. 경쟁사 대응 동향 모니터링.",
        "related_entities": [ctx.get("asin", ""), ctx.get("brand", "")],
        "metadata": {
            "streak_days": ctx.get("streak_days"),
            "rank_change_7d": ctx.get("rank_change_7d"),
        },
    },
    insight_type=InsightType.GROWTH_MOMENTUM,
    priority=7,
    confidence=0.9,
    tags=["growth", "stability", "positive"],
)


# =========================================================================
# 규칙: 트렌드 정렬 기회 (Trend Alignment Opportunity)
# =========================================================================

RULE_TREND_ALIGNMENT = InferenceRule(
    name="trend_alignment_opportunity",
    description="외부/문서 트렌드 키워드와의 정렬이 확인되면 성장 기회 신호",
    conditions=[
        RuleCondition(
            name="has_trend_keywords",
            check=lambda ctx: len(ctx.get("trend_keywords", [])) >= 2,
            description="트렌드 키워드 2개 이상",
        ),
        StandardConditions.is_target_brand(),
    ],
    conclusion=lambda ctx: {
        "insight": f"현재 트렌드 키워드({', '.join(ctx.get('trend_keywords', [])[:3])})와 "
        f"{ctx.get('brand', '브랜드')}가 정렬될 가능성이 있습니다.",
        "opportunity": "trend_alignment",
        "recommendation": "해당 키워드 중심 콘텐츠/광고 테스트 및 해시태그 전략 강화를 검토하세요.",
        "related_entities": [ctx.get("brand", "")],
        "metadata": {"trend_keywords": ctx.get("trend_keywords", [])[:5]},
    },
    insight_type=InsightType.GROWTH_OPPORTUNITY,
    priority=7,
    confidence=0.75,
    tags=["trend", "opportunity", "positive"],
)


# =========================================================================
# 규칙: Top 10 안정 포지션 (Stable Top 10)
# =========================================================================

RULE_TOP10_STABILITY = InferenceRule(
    name="top10_stability",
    description="Top 10 장기 체류는 안정적인 시장 포지션을 나타냄",
    conditions=[
        StandardConditions.in_top_n(10),
        StandardConditions.streak_days_above(14),  # 2주 이상 연속
        RuleCondition(
            name="low_volatility",
            check=lambda ctx: ctx.get("rank_volatility", 10) < 3,
            description="순위 변동성 < 3 (안정적)",
        ),
    ],
    conclusion=lambda ctx: {
        "insight": f"현재 {ctx.get('current_rank', 0)}위로 Top 10을 "
        f"{ctx.get('streak_days', 0)}일 연속 유지하며 "
        f"안정적인 포지션을 확보하고 있습니다.",
        "position": MarketPosition.DOMINANT.value,
        "recommendation": "현재 포지션 유지 전략 지속. 경쟁사 진입 모니터링.",
        "related_entities": [ctx.get("asin", ""), ctx.get("brand", "")],
        "metadata": {
            "current_rank": ctx.get("current_rank"),
            "streak_days": ctx.get("streak_days"),
            "rank_volatility": ctx.get("rank_volatility"),
        },
    },
    insight_type=InsightType.STABILITY,
    priority=6,
    confidence=0.9,
    tags=["rank", "stability", "positive"],
)


# =========================================================================
# 규칙: 카테고리 진입 기회 (Category Entry Opportunity)
# =========================================================================

RULE_CATEGORY_OPPORTUNITY = InferenceRule(
    name="category_entry_opportunity",
    description="분산 시장에서 타겟 브랜드 부재는 진입 기회를 나타냄",
    conditions=[
        StandardConditions.hhi_below(0.15),  # 분산 시장
        RuleCondition(
            name="low_target_presence",
            check=lambda ctx: ctx.get("sos", 0) < 0.03,  # SoS < 3%
            description="타겟 브랜드 점유율 < 3%",
        ),
        RuleCondition(
            name="is_target",
            check=lambda ctx: ctx.get("is_target", False),
            description="타겟 브랜드 분석",
        ),
    ],
    conclusion=lambda ctx: {
        "insight": f"{ctx.get('category', '해당 카테고리')}는 분산된 시장 구조(HHI: {ctx.get('hhi', 0):.3f})로 "
        f"현재 {ctx.get('brand', '타겟 브랜드')}의 존재감이 낮아({ctx.get('sos', 0)*100:.1f}%) "
        f"진입 확대 기회가 있을 수 있습니다.",
        "opportunity": "category_entry",
        "recommendation": "카테고리 트렌드 및 소비자 니즈 분석 후 제품 라인업 확장 검토. "
        "기존 강점 활용 가능한 인접 카테고리 우선 고려.",
        "related_entities": [ctx.get("category", "")],
        "metadata": {
            "category": ctx.get("category"),
            "hhi": ctx.get("hhi"),
            "current_sos": ctx.get("sos"),
        },
    },
    insight_type=InsightType.ENTRY_OPPORTUNITY,
    priority=6,
    confidence=0.7,
    tags=["opportunity", "expansion", "positive"],
)


# =========================================================================
# 규칙: 평점 모멘텀 (Rating Momentum)
# =========================================================================

RULE_RATING_MOMENTUM = InferenceRule(
    name="rating_momentum_positive",
    description="평점 상승 추세는 제품 평판 개선 신호",
    conditions=[
        RuleCondition(
            name="rating_trend_positive",
            check=lambda ctx: ctx.get("rating_trend", 0) > 0.05,
            description="평점 추세 양수 (상승)",
        ),
        RuleCondition(
            name="has_reviews",
            check=lambda ctx: ctx.get("review_count", 0) > 100,
            description="리뷰 100개 이상 (신뢰성)",
        ),
    ],
    conclusion=lambda ctx: {
        "insight": f"평점이 상승 추세(추세 기울기: {ctx.get('rating_trend', 0):+.3f})를 보이며 "
        f"{ctx.get('review_count', 0):,}개의 리뷰 기반으로 "
        f"제품 평판이 개선되고 있을 수 있습니다.",
        "momentum": "rating_positive",
        "recommendation": "긍정적 리뷰 요인 분석 및 강점 마케팅 활용 검토.",
        "related_entities": [ctx.get("asin", "")],
        "metadata": {
            "rating_trend": ctx.get("rating_trend"),
            "review_count": ctx.get("review_count"),
        },
    },
    insight_type=InsightType.GROWTH_MOMENTUM,
    priority=5,
    confidence=0.8,
    tags=["rating", "momentum", "positive"],
)


# =========================================================================
# 규칙: Top 3 달성 (Top 3 Achievement)
# =========================================================================

RULE_TOP3_ACHIEVEMENT = InferenceRule(
    name="top3_achievement",
    description="Top 3 순위 달성은 카테고리 내 강한 경쟁력을 나타냄",
    conditions=[
        StandardConditions.in_top_n(3),  # Top 3 이내
        RuleCondition(
            name="is_target_brand",
            check=lambda ctx: ctx.get("is_target", False)
            or str(ctx.get("brand", "")).upper() == "LANEIGE",
            description="타겟 브랜드",
        ),
    ],
    conclusion=lambda ctx: {
        "insight": f"{ctx.get('brand', '해당 브랜드')}가 {ctx.get('category', '카테고리')}에서 "
        f"{ctx.get('current_rank', 0)}위를 기록하여 Top 3 포지션을 확보했습니다.",
        "achievement": "top3",
        "recommendation": "현재 Top 3 포지션 유지 전략 강화. 1위 도약 기회 모색.",
        "related_entities": [ctx.get("brand", ""), ctx.get("asin", "")],
        "metadata": {"current_rank": ctx.get("current_rank"), "category": ctx.get("category")},
    },
    insight_type=InsightType.COMPETITIVE_ADVANTAGE,
    priority=9,
    confidence=0.95,
    tags=["rank", "achievement", "positive"],
)


# =========================================================================
# 규칙: 높은 평점 경쟁력 (Strong Rating Position)
# =========================================================================

RULE_STRONG_RATING = InferenceRule(
    name="strong_rating_position",
    description="경쟁사 대비 높은 평점은 제품 품질 우위를 나타냄",
    conditions=[
        RuleCondition(
            name="rating_advantage",
            check=lambda ctx: ctx.get("rating_gap", 0) > 0.05,
            description="평점 갭 > 0.05 (경쟁 우위)",
        ),
        RuleCondition(
            name="is_target",
            check=lambda ctx: ctx.get("is_target", False),
            description="타겟 브랜드",
        ),
    ],
    conclusion=lambda ctx: {
        "insight": f"경쟁사 대비 평점 우위(갭: +{ctx.get('rating_gap', 0):.2f})를 보이고 있어 "
        f"제품 품질 및 고객 만족도에서 경쟁 우위를 확보하고 있습니다.",
        "advantage": "rating_superiority",
        "recommendation": "평점 우위를 마케팅에 활용. 리뷰 확보 전략 강화.",
        "related_entities": [ctx.get("brand", "")],
        "metadata": {"rating_gap": ctx.get("rating_gap")},
    },
    insight_type=InsightType.COMPETITIVE_ADVANTAGE,
    priority=6,
    confidence=0.85,
    tags=["rating", "quality", "positive"],
)


GROWTH_RULES: list[InferenceRule] = [
    RULE_STABLE_GROWTH,
    RULE_TREND_ALIGNMENT,
    RULE_TOP10_STABILITY,
    RULE_CATEGORY_OPPORTUNITY,
    RULE_RATING_MOMENTUM,
    RULE_TOP3_ACHIEVEMENT,
    RULE_STRONG_RATING,
]
