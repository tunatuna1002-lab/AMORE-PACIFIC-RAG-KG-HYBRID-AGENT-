"""
Business Rules
Amazon 베스트셀러 분석을 위한 비즈니스 추론 규칙 정의

규칙 구성:
1. 시장 포지션 규칙 (Market Position)
2. 경쟁 위협 규칙 (Competitive Threat)
3. 성장 기회 규칙 (Growth Opportunity)
4. 위험 경고 규칙 (Risk Alert)
5. 가격 포지션 규칙 (Price Position)
"""

from typing import Dict, Any, List

from .relations import InsightType, MarketPosition
from .reasoner import InferenceRule, RuleCondition, StandardConditions


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
                  f"{ctx.get('sos', 0)*100:.1f}%의 점유율로 강한 존재감을 보유하고 있습니다.",
        "position": MarketPosition.DOMINANT_IN_FRAGMENTED.value,
        "recommendation": "현재 포지션 유지 및 점유율 확대 기회 모색. "
                         "신규 진입자 동향 모니터링 권장.",
        "related_entities": [ctx.get("brand", "")],
        "metadata": {
            "sos": ctx.get("sos"),
            "hhi": ctx.get("hhi"),
            "market_type": "fragmented"
        }
    },
    insight_type=InsightType.MARKET_DOMINANCE,
    priority=10,
    confidence=0.9,
    tags=["market", "position", "positive"]
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
                  f"{ctx.get('sos', 0)*100:.1f}%의 높은 점유율을 보유한 주요 플레이어입니다.",
        "position": MarketPosition.DOMINANT.value,
        "recommendation": "현재 지배적 포지션 유지. 시장 구조 변화 및 규제 리스크 모니터링.",
        "related_entities": [ctx.get("brand", "")],
        "metadata": {
            "sos": ctx.get("sos"),
            "hhi": ctx.get("hhi"),
            "market_type": "concentrated"
        }
    },
    insight_type=InsightType.MARKET_DOMINANCE,
    priority=10,
    confidence=0.9,
    tags=["market", "position", "positive"]
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
            description="SoS 5~15% (중간 수준)"
        ),
    ],
    conclusion=lambda ctx: {
        "insight": f"{ctx.get('brand', '해당 브랜드')}는 집중된 시장에서 "
                  f"{ctx.get('sos', 0)*100:.1f}%의 점유율로 도전자 위치에 있습니다.",
        "position": MarketPosition.CHALLENGER.value,
        "recommendation": "니치 전략 또는 차별화 강화를 통한 점유율 확대 검토. "
                         "선두 브랜드 대비 경쟁 우위 요소 발굴 필요.",
        "related_entities": [ctx.get("brand", "")],
        "metadata": {
            "sos": ctx.get("sos"),
            "hhi": ctx.get("hhi"),
            "market_type": "concentrated"
        }
    },
    insight_type=InsightType.MARKET_POSITION,
    priority=8,
    confidence=0.85,
    tags=["market", "position", "neutral"]
)


# =========================================================================
# 규칙 4: 가격-품질 불일치 (Price-Quality Mismatch)
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
        "metadata": {
            "cpi": ctx.get("cpi"),
            "rating_gap": ctx.get("rating_gap")
        },
        "confidence_modifier": 0.95
    },
    insight_type=InsightType.PRICE_QUALITY_GAP,
    priority=9,
    confidence=0.85,
    tags=["price", "quality", "risk"]
)


# =========================================================================
# 규칙 5: 시장 구조 변화 (Market Disruption)
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
        "metadata": {
            "churn_rate": ctx.get("churn_rate"),
            "has_rank_shock": True
        }
    },
    insight_type=InsightType.COMPETITIVE_THREAT,
    priority=9,
    confidence=0.8,
    tags=["market", "disruption", "alert"]
)


# =========================================================================
# 규칙 6: 안정적 성장 (Stable Growth)
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
            "rank_change_7d": ctx.get("rank_change_7d")
        }
    },
    insight_type=InsightType.GROWTH_MOMENTUM,
    priority=7,
    confidence=0.9,
    tags=["growth", "stability", "positive"]
)


# =========================================================================
# 규칙 7: 순위 하락 경고 (Rank Decline Alert)
# =========================================================================

RULE_RANK_DECLINE = InferenceRule(
    name="rank_decline_alert",
    description="순위 하락 추세와 높은 변동성은 포지션 약화 신호",
    conditions=[
        StandardConditions.rank_declining(),
        RuleCondition(
            name="high_volatility",
            check=lambda ctx: ctx.get("rank_volatility", 0) > 5,
            description="순위 변동성 > 5"
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
            "rank_volatility": ctx.get("rank_volatility")
        }
    },
    insight_type=InsightType.RISK_ALERT,
    priority=8,
    confidence=0.8,
    tags=["rank", "decline", "alert"]
)


# =========================================================================
# 규칙 8: Top 10 안정 포지션 (Stable Top 10)
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
            description="순위 변동성 < 3 (안정적)"
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
            "rank_volatility": ctx.get("rank_volatility")
        }
    },
    insight_type=InsightType.STABILITY,
    priority=6,
    confidence=0.9,
    tags=["rank", "stability", "positive"]
)


# =========================================================================
# 규칙 9: 경쟁 압박 (Competitive Pressure)
# =========================================================================

RULE_COMPETITIVE_PRESSURE = InferenceRule(
    name="competitive_pressure",
    description="경쟁사 점유율 상승과 자사 점유율 하락은 경쟁 압박 신호",
    conditions=[
        RuleCondition(
            name="sos_declining",
            check=lambda ctx: ctx.get("sos_change", 0) < -0.02,  # SoS 2%p 이상 하락
            description="SoS 2%p 이상 하락"
        ),
        StandardConditions.has_competitors(3),  # 경쟁사 3개 이상
    ],
    conclusion=lambda ctx: {
        "insight": f"점유율이 하락 추세({ctx.get('sos_change', 0)*100:+.1f}%p)이며 "
                  f"{ctx.get('competitor_count', 0)}개 경쟁 브랜드의 활동이 활발하여 "
                  f"경쟁 압박이 증가하고 있을 수 있습니다.",
        "threat": "competitive_pressure",
        "recommendation": "경쟁사 전략 분석 및 차별화 포인트 강화 검토. "
                         "가격/프로모션/리뷰 관리 전략 점검.",
        "related_entities": [
            comp.get("brand") for comp in ctx.get("competitors", [])[:3]
        ],
        "metadata": {
            "sos_change": ctx.get("sos_change"),
            "competitor_count": ctx.get("competitor_count")
        }
    },
    insight_type=InsightType.COMPETITIVE_THREAT,
    priority=8,
    confidence=0.75,
    tags=["competition", "threat", "alert"]
)


# =========================================================================
# 규칙 10: 카테고리 진입 기회 (Category Entry Opportunity)
# =========================================================================

RULE_CATEGORY_OPPORTUNITY = InferenceRule(
    name="category_entry_opportunity",
    description="분산 시장에서 타겟 브랜드 부재는 진입 기회를 나타냄",
    conditions=[
        StandardConditions.hhi_below(0.15),  # 분산 시장
        RuleCondition(
            name="low_target_presence",
            check=lambda ctx: ctx.get("sos", 0) < 0.03,  # SoS < 3%
            description="타겟 브랜드 점유율 < 3%"
        ),
        RuleCondition(
            name="is_target",
            check=lambda ctx: ctx.get("is_target", False),
            description="타겟 브랜드 분석"
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
            "current_sos": ctx.get("sos")
        }
    },
    insight_type=InsightType.ENTRY_OPPORTUNITY,
    priority=6,
    confidence=0.7,
    tags=["opportunity", "expansion", "positive"]
)


# =========================================================================
# 추가 규칙: 평점 모멘텀 (Rating Momentum)
# =========================================================================

RULE_RATING_MOMENTUM = InferenceRule(
    name="rating_momentum_positive",
    description="평점 상승 추세는 제품 평판 개선 신호",
    conditions=[
        RuleCondition(
            name="rating_trend_positive",
            check=lambda ctx: ctx.get("rating_trend", 0) > 0.05,
            description="평점 추세 양수 (상승)"
        ),
        RuleCondition(
            name="has_reviews",
            check=lambda ctx: ctx.get("review_count", 0) > 100,
            description="리뷰 100개 이상 (신뢰성)"
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
            "review_count": ctx.get("review_count")
        }
    },
    insight_type=InsightType.GROWTH_MOMENTUM,
    priority=5,
    confidence=0.8,
    tags=["rating", "momentum", "positive"]
)


# =========================================================================
# 추가 규칙: 가성비 포지션 (Value Position)
# =========================================================================

RULE_VALUE_POSITION = InferenceRule(
    name="value_position",
    description="낮은 CPI와 높은 평점은 강한 가성비 포지션을 나타냄",
    conditions=[
        StandardConditions.cpi_below(90),  # CPI < 90 (가성비)
        StandardConditions.rating_gap_positive(),  # 평점 우위
    ],
    conclusion=lambda ctx: {
        "insight": f"카테고리 평균 대비 저렴한 가격(CPI: {ctx.get('cpi', 100):.0f})과 "
                  f"높은 평점 경쟁력으로 강한 가성비 포지션을 확보하고 있습니다.",
        "position": "value_leader",
        "recommendation": "가성비 마케팅 강화 및 리뷰 확보 전략 지속.",
        "related_entities": [ctx.get("asin", ""), ctx.get("brand", "")],
        "metadata": {
            "cpi": ctx.get("cpi"),
            "rating_gap": ctx.get("rating_gap")
        }
    },
    insight_type=InsightType.PRICE_POSITION,
    priority=6,
    confidence=0.85,
    tags=["price", "value", "positive"]
)


# =========================================================================
# 추가 규칙: Top 3 달성 (Top 3 Achievement)
# =========================================================================

RULE_TOP3_ACHIEVEMENT = InferenceRule(
    name="top3_achievement",
    description="Top 3 순위 달성은 카테고리 내 강한 경쟁력을 나타냄",
    conditions=[
        StandardConditions.in_top_n(3),  # Top 3 이내
        RuleCondition(
            name="is_target_brand",
            check=lambda ctx: ctx.get("is_target", False) or
                             str(ctx.get("brand", "")).upper() == "LANEIGE",
            description="타겟 브랜드"
        ),
    ],
    conclusion=lambda ctx: {
        "insight": f"{ctx.get('brand', '해당 브랜드')}가 {ctx.get('category', '카테고리')}에서 "
                  f"{ctx.get('current_rank', 0)}위를 기록하여 Top 3 포지션을 확보했습니다.",
        "achievement": "top3",
        "recommendation": "현재 Top 3 포지션 유지 전략 강화. 1위 도약 기회 모색.",
        "related_entities": [ctx.get("brand", ""), ctx.get("asin", "")],
        "metadata": {
            "current_rank": ctx.get("current_rank"),
            "category": ctx.get("category")
        }
    },
    insight_type=InsightType.COMPETITIVE_ADVANTAGE,
    priority=9,
    confidence=0.95,
    tags=["rank", "achievement", "positive"]
)


# =========================================================================
# 추가 규칙: 분산 시장 경쟁 (Fragmented Market Competition)
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
            "market_type": "fragmented"
        }
    },
    insight_type=InsightType.MARKET_POSITION,
    priority=7,
    confidence=0.85,
    tags=["market", "competition", "neutral"]
)


# =========================================================================
# 추가 규칙: 프리미엄 가격 포지션 (Premium Price Position)
# =========================================================================

RULE_PREMIUM_POSITION = InferenceRule(
    name="premium_price_position",
    description="높은 CPI와 높은 평점은 성공적인 프리미엄 포지셔닝을 나타냄",
    conditions=[
        StandardConditions.cpi_above(150),  # CPI > 150 (프리미엄)
        RuleCondition(
            name="good_rating",
            check=lambda ctx: ctx.get("rating_gap", 0) >= 0,
            description="평점 동등 이상"
        ),
    ],
    conclusion=lambda ctx: {
        "insight": f"CPI {ctx.get('cpi', 0):.0f}의 프리미엄 가격 포지션에서 "
                  f"평점 경쟁력을 유지하고 있어 성공적인 프리미엄 전략을 수행 중입니다.",
        "position": "premium_success",
        "recommendation": "프리미엄 브랜드 가치 강화 지속. 신규 프리미엄 라인 확장 검토.",
        "related_entities": [ctx.get("brand", "")],
        "metadata": {
            "cpi": ctx.get("cpi"),
            "rating_gap": ctx.get("rating_gap")
        }
    },
    insight_type=InsightType.PRICE_POSITION,
    priority=7,
    confidence=0.85,
    tags=["price", "premium", "positive"]
)


# =========================================================================
# 추가 규칙: 높은 평점 경쟁력 (Strong Rating Position)
# =========================================================================

RULE_STRONG_RATING = InferenceRule(
    name="strong_rating_position",
    description="경쟁사 대비 높은 평점은 제품 품질 우위를 나타냄",
    conditions=[
        RuleCondition(
            name="rating_advantage",
            check=lambda ctx: ctx.get("rating_gap", 0) > 0.05,
            description="평점 갭 > 0.05 (경쟁 우위)"
        ),
        RuleCondition(
            name="is_target",
            check=lambda ctx: ctx.get("is_target", False),
            description="타겟 브랜드"
        ),
    ],
    conclusion=lambda ctx: {
        "insight": f"경쟁사 대비 평점 우위(갭: +{ctx.get('rating_gap', 0):.2f})를 보이고 있어 "
                  f"제품 품질 및 고객 만족도에서 경쟁 우위를 확보하고 있습니다.",
        "advantage": "rating_superiority",
        "recommendation": "평점 우위를 마케팅에 활용. 리뷰 확보 전략 강화.",
        "related_entities": [ctx.get("brand", "")],
        "metadata": {
            "rating_gap": ctx.get("rating_gap")
        }
    },
    insight_type=InsightType.COMPETITIVE_ADVANTAGE,
    priority=6,
    confidence=0.85,
    tags=["rating", "quality", "positive"]
)


# =========================================================================
# 추가 규칙: 평균 순위 우위 (Strong Average Rank)
# =========================================================================

RULE_STRONG_AVG_RANK = InferenceRule(
    name="strong_avg_rank",
    description="경쟁사 대비 낮은 평균 순위는 전반적 경쟁력 우위를 나타냄",
    conditions=[
        RuleCondition(
            name="low_avg_rank",
            check=lambda ctx: ctx.get("avg_rank", 100) < 20,
            description="평균 순위 < 20"
        ),
        RuleCondition(
            name="is_target",
            check=lambda ctx: ctx.get("is_target", False),
            description="타겟 브랜드"
        ),
    ],
    conclusion=lambda ctx: {
        "insight": f"{ctx.get('brand', '해당 브랜드')}의 평균 순위 {ctx.get('avg_rank', 0):.1f}위는 "
                  f"경쟁사 대비 우수한 전반적 포지션을 나타냅니다.",
        "position": "rank_leader",
        "recommendation": "전체 제품 포트폴리오의 순위 유지/개선 전략 지속.",
        "related_entities": [ctx.get("brand", "")],
        "metadata": {
            "avg_rank": ctx.get("avg_rank")
        }
    },
    insight_type=InsightType.MARKET_POSITION,
    priority=6,
    confidence=0.85,
    tags=["rank", "position", "positive"]
)


# =========================================================================
# 전체 규칙 리스트
# =========================================================================

ALL_BUSINESS_RULES: List[InferenceRule] = [
    # 시장 포지션 규칙
    RULE_MARKET_DOMINANCE,
    RULE_DOMINANT_CONCENTRATED,
    RULE_CHALLENGER_POSITION,
    RULE_FRAGMENTED_COMPETITION,
    RULE_STRONG_AVG_RANK,

    # 위험/경고 규칙
    RULE_PRICE_QUALITY_MISMATCH,
    RULE_MARKET_DISRUPTION,
    RULE_RANK_DECLINE,
    RULE_COMPETITIVE_PRESSURE,

    # 성장/기회 규칙
    RULE_STABLE_GROWTH,
    RULE_TOP10_STABILITY,
    RULE_CATEGORY_OPPORTUNITY,
    RULE_RATING_MOMENTUM,
    RULE_TOP3_ACHIEVEMENT,

    # 가격/품질 규칙
    RULE_VALUE_POSITION,
    RULE_PREMIUM_POSITION,
    RULE_STRONG_RATING,
]


def get_rules_by_category(category: str) -> List[InferenceRule]:
    """
    카테고리별 규칙 필터링

    Args:
        category: "market", "risk", "growth", "price" 등

    Returns:
        해당 카테고리의 규칙 리스트
    """
    return [r for r in ALL_BUSINESS_RULES if category in r.tags]


def get_high_priority_rules(min_priority: int = 7) -> List[InferenceRule]:
    """높은 우선순위 규칙만 필터링"""
    return [r for r in ALL_BUSINESS_RULES if r.priority >= min_priority]


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
