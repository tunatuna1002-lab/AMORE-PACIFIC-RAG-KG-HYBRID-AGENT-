"""
Market Position and Competition Rules
시장 포지션 및 경쟁 관련 규칙
"""

from ..reasoner import InferenceRule, RuleCondition, StandardConditions
from ..relations import InsightType, MarketPosition

# =========================================================================
# 규칙 1: 분산 시장 지배자 (Dominant in Fragmented Market)
# =========================================================================

RULE_MARKET_DOMINANCE = InferenceRule(
    name="market_dominance_fragmented",
    description="높은 SoS와 낮은 HHI는 분산된 시장에서의 지배적 포지션을 나타냄",
    conditions=[
        StandardConditions.sos_above(0.15),  # SoS >= 15%
        StandardConditions.hhi_below(0.15),  # HHI < 0.15 (분산 시장)
    ],
    conclusion=lambda ctx: {
        "insight": f"{ctx.get('brand', '해당 브랜드')}는 분산된 시장 구조에서 "
        f"{ctx.get('sos', 0) * 100:.1f}%의 점유율로 강한 존재감을 보유하고 있습니다.",
        "position": MarketPosition.DOMINANT_IN_FRAGMENTED.value,
        "recommendation": "현재 포지션 유지 및 점유율 확대 기회 모색. "
        "신규 진입자 동향 모니터링 권장.",
        "related_entities": [ctx.get("brand", "")],
        "metadata": {"sos": ctx.get("sos"), "hhi": ctx.get("hhi"), "market_type": "fragmented"},
    },
    insight_type=InsightType.MARKET_DOMINANCE,
    priority=10,
    confidence=0.9,
    tags=["market", "position", "positive"],
)


# =========================================================================
# 규칙 2: 집중 시장 지배자 (Dominant in Concentrated Market)
# =========================================================================

RULE_DOMINANT_CONCENTRATED = InferenceRule(
    name="market_dominance_concentrated",
    description="높은 SoS와 높은 HHI는 집중된 시장에서의 지배적 포지션을 나타냄",
    conditions=[
        StandardConditions.sos_above(0.20),  # SoS >= 20%
        StandardConditions.hhi_above(0.25),  # HHI >= 0.25 (집중 시장)
    ],
    conclusion=lambda ctx: {
        "insight": f"{ctx.get('brand', '해당 브랜드')}는 집중된 시장에서 "
        f"{ctx.get('sos', 0) * 100:.1f}%의 높은 점유율을 보유한 주요 플레이어입니다.",
        "position": MarketPosition.DOMINANT.value,
        "recommendation": "현재 지배적 포지션 유지. 시장 구조 변화 및 규제 리스크 모니터링.",
        "related_entities": [ctx.get("brand", "")],
        "metadata": {"sos": ctx.get("sos"), "hhi": ctx.get("hhi"), "market_type": "concentrated"},
    },
    insight_type=InsightType.MARKET_DOMINANCE,
    priority=10,
    confidence=0.9,
    tags=["market", "position", "positive"],
)


# =========================================================================
# 규칙 3: 도전자 포지션 (Challenger Position)
# =========================================================================

RULE_CHALLENGER_POSITION = InferenceRule(
    name="challenger_position",
    description="집중 시장에서 중간 수준의 SoS는 도전자 포지션을 나타냄",
    conditions=[
        StandardConditions.hhi_above(0.25),  # 집중 시장
        RuleCondition(
            name="mid_sos",
            check=lambda ctx: 0.05 <= ctx.get("sos", 0) < 0.15,
            description="SoS 5~15% (중간 수준)",
        ),
    ],
    conclusion=lambda ctx: {
        "insight": f"{ctx.get('brand', '해당 브랜드')}는 집중된 시장에서 "
        f"{ctx.get('sos', 0) * 100:.1f}%의 점유율로 도전자 위치에 있습니다.",
        "position": MarketPosition.CHALLENGER.value,
        "recommendation": "니치 전략 또는 차별화 강화를 통한 점유율 확대 검토. "
        "선두 브랜드 대비 경쟁 우위 요소 발굴 필요.",
        "related_entities": [ctx.get("brand", "")],
        "metadata": {"sos": ctx.get("sos"), "hhi": ctx.get("hhi"), "market_type": "concentrated"},
    },
    insight_type=InsightType.MARKET_POSITION,
    priority=8,
    confidence=0.85,
    tags=["market", "position", "neutral"],
)


# =========================================================================
# 규칙: 분산 시장 경쟁 (Fragmented Market Competition)
# =========================================================================

RULE_FRAGMENTED_COMPETITION = InferenceRule(
    name="fragmented_market_competition",
    description="분산된 시장에서의 다수 경쟁자 존재는 성장 기회와 위협이 공존",
    conditions=[
        StandardConditions.hhi_below(0.15),  # 분산 시장
        StandardConditions.has_competitors(5),  # 5개 이상 경쟁사
    ],
    conclusion=lambda ctx: {
        "insight": f"시장이 분산되어 있고(HHI: {ctx.get('hhi', 0):.3f}) "
        f"{ctx.get('competitor_count', 0)}개의 경쟁 브랜드가 활동 중입니다. "
        f"시장 점유율 확대 기회가 있으나 경쟁도 치열합니다.",
        "market_structure": "fragmented_competitive",
        "recommendation": "차별화 전략 강화 및 타겟 고객층 명확화. "
        "경쟁사 대비 강점(가격/품질/브랜드) 부각 필요.",
        "related_entities": [comp.get("brand") for comp in ctx.get("competitors", [])[:5]],
        "metadata": {
            "hhi": ctx.get("hhi"),
            "competitor_count": ctx.get("competitor_count"),
            "market_type": "fragmented",
        },
    },
    insight_type=InsightType.MARKET_POSITION,
    priority=7,
    confidence=0.85,
    tags=["market", "competition", "neutral"],
)


# =========================================================================
# 규칙: 평균 순위 우위 (Strong Average Rank)
# =========================================================================

RULE_STRONG_AVG_RANK = InferenceRule(
    name="strong_avg_rank",
    description="경쟁사 대비 낮은 평균 순위는 전반적 경쟁력 우위를 나타냄",
    conditions=[
        RuleCondition(
            name="low_avg_rank",
            check=lambda ctx: ctx.get("avg_rank", 100) < 20,
            description="평균 순위 < 20",
        ),
        RuleCondition(
            name="is_target",
            check=lambda ctx: ctx.get("is_target", False),
            description="타겟 브랜드",
        ),
    ],
    conclusion=lambda ctx: {
        "insight": f"{ctx.get('brand', '해당 브랜드')}의 평균 순위 {ctx.get('avg_rank', 0):.1f}위는 "
        f"경쟁사 대비 우수한 전반적 포지션을 나타냅니다.",
        "position": "rank_leader",
        "recommendation": "전체 제품 포트폴리오의 순위 유지/개선 전략 지속.",
        "related_entities": [ctx.get("brand", "")],
        "metadata": {"avg_rank": ctx.get("avg_rank")},
    },
    insight_type=InsightType.MARKET_POSITION,
    priority=6,
    confidence=0.85,
    tags=["rank", "position", "positive"],
)


# =========================================================================
# 규칙: 경쟁 압박 (Competitive Pressure)
# =========================================================================

RULE_COMPETITIVE_PRESSURE = InferenceRule(
    name="competitive_pressure",
    description="경쟁사 점유율 상승과 자사 점유율 하락은 경쟁 압박 신호",
    conditions=[
        RuleCondition(
            name="sos_declining",
            check=lambda ctx: ctx.get("sos_change", 0) < -0.02,  # SoS 2%p 이상 하락
            description="SoS 2%p 이상 하락",
        ),
        StandardConditions.has_competitors(3),  # 경쟁사 3개 이상
    ],
    conclusion=lambda ctx: {
        "insight": f"점유율이 하락 추세({ctx.get('sos_change', 0) * 100:+.1f}%p)이며 "
        f"{ctx.get('competitor_count', 0)}개 경쟁 브랜드의 활동이 활발하여 "
        f"경쟁 압박이 증가하고 있을 수 있습니다.",
        "threat": "competitive_pressure",
        "recommendation": "경쟁사 전략 분석 및 차별화 포인트 강화 검토. "
        "가격/프로모션/리뷰 관리 전략 점검.",
        "related_entities": [comp.get("brand") for comp in ctx.get("competitors", [])[:3]],
        "metadata": {
            "sos_change": ctx.get("sos_change"),
            "competitor_count": ctx.get("competitor_count"),
        },
    },
    insight_type=InsightType.COMPETITIVE_THREAT,
    priority=8,
    confidence=0.75,
    tags=["competition", "threat", "alert"],
)


MARKET_RULES: list[InferenceRule] = [
    RULE_MARKET_DOMINANCE,
    RULE_DOMINANT_CONCENTRATED,
    RULE_CHALLENGER_POSITION,
    RULE_FRAGMENTED_COMPETITION,
    RULE_STRONG_AVG_RANK,
    RULE_COMPETITIVE_PRESSURE,
]
