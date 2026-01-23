"""
Business Rules
Amazon 베스트셀러 분석을 위한 비즈니스 추론 규칙 정의

규칙 구성:
1. 시장 포지션 규칙 (Market Position)
2. 경쟁 위협 규칙 (Competitive Threat)
3. 성장 기회 규칙 (Growth Opportunity)
4. 위험 경고 규칙 (Risk Alert)
5. 가격 포지션 규칙 (Price Position)
6. 순위-할인 인과관계 규칙 (Rank-Discount Causality)
"""

from typing import Dict, Any, List, Optional

from .relations import InsightType, MarketPosition
from .reasoner import InferenceRule, RuleCondition, StandardConditions


# =========================================================================
# 헬퍼 함수: 순위-할인 상관관계 분석
# =========================================================================

def count_period_overlap(periods_a: List[Dict], periods_b: List[Dict]) -> int:
    """
    두 기간 리스트의 겹치는 횟수 계산

    Args:
        periods_a: [{"start": "2024-01-01", "end": "2024-01-07"}, ...]
        periods_b: [{"start": "2024-01-02", "end": "2024-01-05"}, ...]

    Returns:
        겹치는 기간의 수
    """
    from datetime import datetime

    overlap_count = 0

    for period_b in periods_b:
        b_start_str = period_b.get("start")
        b_end_str = period_b.get("end")

        if not b_start_str:
            continue

        try:
            b_start = datetime.fromisoformat(b_start_str.replace("Z", "+00:00"))
            b_end = datetime.fromisoformat(b_end_str.replace("Z", "+00:00")) if b_end_str else b_start
        except (ValueError, TypeError):
            continue

        for period_a in periods_a:
            a_start_str = period_a.get("start")
            a_end_str = period_a.get("end")

            if not a_start_str:
                continue

            try:
                a_start = datetime.fromisoformat(a_start_str.replace("Z", "+00:00"))
                a_end = datetime.fromisoformat(a_end_str.replace("Z", "+00:00")) if a_end_str else a_start
            except (ValueError, TypeError):
                continue

            # 겹침 조건: 한 기간의 시작이 다른 기간의 끝 이전이고,
            # 한 기간의 끝이 다른 기간의 시작 이후
            if b_start <= a_end and a_start <= b_end:
                overlap_count += 1
                break  # 이 period_b는 이미 겹침으로 카운트됨

    return overlap_count


def calculate_discount_dependency_score(product_history: List[Dict]) -> float:
    """
    할인 의존도 점수 계산 (0~100)

    점수 해석:
    - 0-30: 낮음 (브랜드력 주도)
    - 31-60: 중간
    - 61-100: 높음 (할인 의존)

    Args:
        product_history: 제품 이력 데이터 리스트
            [{"rank": 10, "discount_percent": 20, "date": "2024-01-01"}, ...]

    Returns:
        할인 의존도 점수 (0~100)
    """
    if not product_history or len(product_history) < 2:
        return 0.0

    discount_rank_correlation = 0
    total_periods = len(product_history)

    for i in range(1, total_periods):
        prev = product_history[i-1]
        curr = product_history[i]

        # 할인율 증가 + 순위 상승 = 상관관계
        prev_discount = prev.get("discount_percent", 0) or 0
        curr_discount = curr.get("discount_percent", 0) or 0
        prev_rank = prev.get("rank", 100) or 100
        curr_rank = curr.get("rank", 100) or 100

        discount_increased = curr_discount > prev_discount
        rank_improved = curr_rank < prev_rank  # 순위가 낮아지면 개선

        if discount_increased and rank_improved:
            discount_rank_correlation += 1

    dependency_score = (discount_rank_correlation / (total_periods - 1)) * 100 if total_periods > 1 else 0
    return round(dependency_score, 1)


def calculate_premium_defense_index(
    product_data: Dict,
    category_avg_price: float
) -> Dict[str, Any]:
    """
    프리미엄 방어 지수 계산

    높은 가격대에서도 점유율을 유지하는지 측정

    Args:
        product_data: 제품 데이터 (price, rank 포함)
        category_avg_price: 카테고리 평균 가격

    Returns:
        {
            "index": 0~100 점수,
            "interpretation": 해석,
            "insight": 인사이트
        }
    """
    price = product_data.get("price", 0)
    rank = product_data.get("rank", 100)

    if not price or not category_avg_price or category_avg_price == 0:
        return {
            "index": 0,
            "interpretation": "데이터 부족",
            "insight": "가격 정보가 충분하지 않습니다."
        }

    price_premium = (price - category_avg_price) / category_avg_price * 100

    # 프리미엄 가격인데 좋은 순위면 높은 점수
    if price_premium > 20:  # 평균보다 20% 이상 비쌈
        if rank <= 10:
            return {
                "index": 90,
                "interpretation": "강력한 프리미엄 방어",
                "insight": f"평균보다 {price_premium:.0f}% 높은 가격에도 Top 10 유지"
            }
        elif rank <= 30:
            return {
                "index": 70,
                "interpretation": "양호한 프리미엄 방어",
                "insight": "프리미엄 가격에서 상위권 유지"
            }
        else:
            return {
                "index": 40,
                "interpretation": "프리미엄 방어 취약",
                "insight": "높은 가격이 순위에 부정적 영향"
            }
    else:
        return {
            "index": 50,
            "interpretation": "일반 가격대",
            "insight": "카테고리 평균 수준의 가격"
        }


def rule_discount_dependent(product_data: Dict) -> Optional[Dict]:
    """
    할인 의존형 제품 감지

    - 할인 기간과 순위 상승이 80% 이상 일치하면 '할인 의존형'
    - 권장: 할인 없이도 경쟁력 확보 전략 필요

    Args:
        product_data: 제품 데이터
            {
                "discount_periods": [{"start": "...", "end": "..."}],
                "rank_improvements": [{"start": "...", "end": "..."}]
            }

    Returns:
        인사이트 딕셔너리 또는 None
    """
    discount_periods = product_data.get("discount_periods", [])
    rank_improvements = product_data.get("rank_improvements", [])

    if not discount_periods or not rank_improvements:
        return None

    # 일치율 계산
    overlap_count = count_period_overlap(discount_periods, rank_improvements)
    overlap_ratio = overlap_count / len(rank_improvements) if rank_improvements else 0

    if overlap_ratio >= 0.8:
        return {
            "rule_name": "discount_dependent",
            "insight_type": "price_dependency",
            "tag": "할인 의존형",
            "tag_color": "red",
            "confidence": overlap_ratio,
            "insight": f"순위 상승의 {overlap_ratio*100:.0f}%가 할인 기간과 일치합니다. 할인 의존도가 높습니다.",
            "recommendation": "브랜드 인지도 강화 및 비가격 경쟁력 확보 전략이 필요합니다.",
            "related_entities": [product_data.get("asin", ""), product_data.get("brand", "")]
        }
    return None


def rule_viral_effect(product_data: Dict) -> Optional[Dict]:
    """
    바이럴/브랜드력 효과 감지

    - 가격 변동 없이 순위 상승하면 '바이럴 효과'
    - 긍정적 신호: 브랜드 파워 증가

    Args:
        product_data: 제품 데이터
            {
                "price_stable": bool,
                "rank_change_7d": int (음수면 순위 상승)
            }

    Returns:
        인사이트 딕셔너리 또는 None
    """
    price_stable = product_data.get("price_stable", False)
    rank_improved = product_data.get("rank_change_7d", 0) < 0  # 음수면 상승

    if price_stable and rank_improved:
        rank_change = abs(product_data.get("rank_change_7d", 0))
        return {
            "rule_name": "viral_effect",
            "insight_type": "brand_strength",
            "tag": "바이럴 효과",
            "tag_color": "green",
            "confidence": min(1.0, rank_change / 20),
            "insight": f"가격 변동 없이 순위가 {rank_change}단계 상승했습니다. 브랜드력 또는 바이럴 효과로 추정됩니다.",
            "recommendation": "현재 마케팅 전략을 유지하고 고객 리뷰/SNS 반응을 모니터링하세요.",
            "related_entities": [product_data.get("asin", ""), product_data.get("brand", "")]
        }
    return None


def rule_bestseller_badge_effect(product_data: Dict) -> Optional[Dict]:
    """
    베스트셀러 배지 획득 후 순위 유지 분석

    Args:
        product_data: 제품 데이터
            {
                "badge": str,
                "rank_change_7d": int
            }

    Returns:
        인사이트 딕셔너리 또는 None
    """
    has_badge = product_data.get("badge") == "Best Seller"
    rank_stable = abs(product_data.get("rank_change_7d", 0)) <= 3

    if has_badge and rank_stable:
        return {
            "rule_name": "bestseller_badge_effect",
            "insight_type": "market_position",
            "tag": "베스트셀러 유지",
            "tag_color": "blue",
            "confidence": 0.9,
            "insight": "베스트셀러 배지를 유지하며 순위가 안정적입니다.",
            "recommendation": "현 포지션 유지를 위해 재고 관리와 리뷰 관리에 집중하세요.",
            "related_entities": [product_data.get("asin", ""), product_data.get("brand", "")]
        }
    return None


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
# 추가 규칙: 트렌드 정렬 기회 (Trend Alignment Opportunity)
# =========================================================================

RULE_TREND_ALIGNMENT = InferenceRule(
    name="trend_alignment_opportunity",
    description="외부/문서 트렌드 키워드와의 정렬이 확인되면 성장 기회 신호",
    conditions=[
        RuleCondition(
            name="has_trend_keywords",
            check=lambda ctx: len(ctx.get("trend_keywords", [])) >= 2,
            description="트렌드 키워드 2개 이상"
        ),
        StandardConditions.is_target_brand()
    ],
    conclusion=lambda ctx: {
        "insight": f"현재 트렌드 키워드({', '.join(ctx.get('trend_keywords', [])[:3])})와 "
                  f"{ctx.get('brand', '브랜드')}가 정렬될 가능성이 있습니다.",
        "opportunity": "trend_alignment",
        "recommendation": "해당 키워드 중심 콘텐츠/광고 테스트 및 해시태그 전략 강화를 검토하세요.",
        "related_entities": [ctx.get("brand", "")],
        "metadata": {
            "trend_keywords": ctx.get("trend_keywords", [])[:5]
        }
    },
    insight_type=InsightType.GROWTH_OPPORTUNITY,
    priority=7,
    confidence=0.75,
    tags=["trend", "opportunity", "positive"]
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
# 순위-할인 인과관계 규칙: 할인 의존형 (Discount Dependent)
# =========================================================================

RULE_DISCOUNT_DEPENDENT = InferenceRule(
    name="discount_dependent",
    description="순위 상승이 할인 기간과 높은 일치율을 보이면 할인 의존형",
    conditions=[
        RuleCondition(
            name="has_discount_periods",
            check=lambda ctx: bool(ctx.get("discount_periods")),
            description="할인 기간 데이터 존재"
        ),
        RuleCondition(
            name="has_rank_improvements",
            check=lambda ctx: bool(ctx.get("rank_improvements")),
            description="순위 상승 기간 데이터 존재"
        ),
        RuleCondition(
            name="high_overlap_ratio",
            check=lambda ctx: (
                count_period_overlap(
                    ctx.get("discount_periods", []),
                    ctx.get("rank_improvements", [])
                ) / len(ctx.get("rank_improvements", [1])) >= 0.8
            ) if ctx.get("rank_improvements") else False,
            description="할인-순위상승 일치율 >= 80%"
        ),
    ],
    conclusion=lambda ctx: {
        "insight": rule_discount_dependent(ctx)["insight"],
        "position": "discount_dependent",
        "recommendation": "브랜드 인지도 강화 및 비가격 경쟁력 확보 전략이 필요합니다.",
        "tag": "할인 의존형",
        "tag_color": "red",
        "related_entities": [ctx.get("asin", ""), ctx.get("brand", "")],
        "metadata": {
            "discount_periods_count": len(ctx.get("discount_periods", [])),
            "rank_improvements_count": len(ctx.get("rank_improvements", [])),
            "overlap_ratio": count_period_overlap(
                ctx.get("discount_periods", []),
                ctx.get("rank_improvements", [])
            ) / len(ctx.get("rank_improvements", [1])) if ctx.get("rank_improvements") else 0
        }
    },
    insight_type=InsightType.PRICE_DEPENDENCY,
    priority=8,
    confidence=0.85,
    tags=["price", "discount", "dependency", "alert"]
)


# =========================================================================
# 순위-할인 인과관계 규칙: 바이럴 효과형 (Viral Effect)
# =========================================================================

RULE_VIRAL_EFFECT = InferenceRule(
    name="viral_effect",
    description="가격 변동 없이 순위 상승은 브랜드력/바이럴 효과",
    conditions=[
        RuleCondition(
            name="price_stable",
            check=lambda ctx: ctx.get("price_stable", False),
            description="가격 안정"
        ),
        RuleCondition(
            name="rank_improved",
            check=lambda ctx: ctx.get("rank_change_7d", 0) < 0,
            description="순위 상승 (음수)"
        ),
    ],
    conclusion=lambda ctx: {
        "insight": f"가격 변동 없이 순위가 {abs(ctx.get('rank_change_7d', 0))}단계 상승했습니다. "
                  f"브랜드력 또는 바이럴 효과로 추정됩니다.",
        "position": "viral_growth",
        "recommendation": "현재 마케팅 전략을 유지하고 고객 리뷰/SNS 반응을 모니터링하세요.",
        "tag": "바이럴 효과",
        "tag_color": "green",
        "related_entities": [ctx.get("asin", ""), ctx.get("brand", "")],
        "metadata": {
            "rank_change_7d": ctx.get("rank_change_7d"),
            "price_stable": True
        }
    },
    insight_type=InsightType.BRAND_STRENGTH,
    priority=8,
    confidence=0.8,
    tags=["brand", "viral", "growth", "positive"]
)


# =========================================================================
# 순위-할인 인과관계 규칙: 베스트셀러 배지 효과
# =========================================================================

RULE_BESTSELLER_BADGE_EFFECT = InferenceRule(
    name="bestseller_badge_effect",
    description="베스트셀러 배지 유지 중 순위 안정은 브랜드 신뢰도 확보",
    conditions=[
        RuleCondition(
            name="has_bestseller_badge",
            check=lambda ctx: ctx.get("badge") == "Best Seller",
            description="베스트셀러 배지 보유"
        ),
        RuleCondition(
            name="rank_stable",
            check=lambda ctx: abs(ctx.get("rank_change_7d", 0)) <= 3,
            description="순위 변동 ±3 이내 (안정)"
        ),
    ],
    conclusion=lambda ctx: {
        "insight": "베스트셀러 배지를 유지하며 순위가 안정적입니다.",
        "position": "bestseller_stable",
        "recommendation": "현 포지션 유지를 위해 재고 관리와 리뷰 관리에 집중하세요.",
        "tag": "베스트셀러 유지",
        "tag_color": "blue",
        "related_entities": [ctx.get("asin", ""), ctx.get("brand", "")],
        "metadata": {
            "badge": ctx.get("badge"),
            "rank_change_7d": ctx.get("rank_change_7d")
        }
    },
    insight_type=InsightType.MARKET_POSITION,
    priority=7,
    confidence=0.9,
    tags=["badge", "stability", "positive"]
)


# =========================================================================
# 순위-할인 인과관계 규칙: 높은 할인 의존도 점수
# =========================================================================

RULE_HIGH_DISCOUNT_DEPENDENCY_SCORE = InferenceRule(
    name="high_discount_dependency_score",
    description="할인 의존도 점수 61점 이상은 할인 중심 전략 신호",
    conditions=[
        RuleCondition(
            name="has_product_history",
            check=lambda ctx: bool(ctx.get("product_history")),
            description="제품 이력 데이터 존재"
        ),
        RuleCondition(
            name="high_dependency",
            check=lambda ctx: calculate_discount_dependency_score(
                ctx.get("product_history", [])
            ) >= 61,
            description="할인 의존도 점수 >= 61"
        ),
    ],
    conclusion=lambda ctx: {
        "insight": f"할인 의존도 점수 {calculate_discount_dependency_score(ctx.get('product_history', [])):.1f}점으로 "
                  f"할인 중심 전략에 크게 의존하고 있습니다.",
        "risk": "high_discount_dependency",
        "recommendation": "브랜드 가치 강화 및 프리미엄 포지셔닝 전략 검토가 필요합니다.",
        "tag": "할인 고의존",
        "tag_color": "orange",
        "related_entities": [ctx.get("asin", ""), ctx.get("brand", "")],
        "metadata": {
            "dependency_score": calculate_discount_dependency_score(ctx.get("product_history", []))
        }
    },
    insight_type=InsightType.PRICE_DEPENDENCY,
    priority=7,
    confidence=0.85,
    tags=["price", "discount", "dependency", "risk"]
)


# =========================================================================
# 순위-할인 인과관계 규칙: 프리미엄 방어 성공
# =========================================================================

RULE_PREMIUM_DEFENSE_SUCCESS = InferenceRule(
    name="premium_defense_success",
    description="높은 가격에서도 상위권 순위 유지는 강한 브랜드 파워",
    conditions=[
        RuleCondition(
            name="has_price_data",
            check=lambda ctx: ctx.get("price") and ctx.get("category_avg_price"),
            description="가격 데이터 존재"
        ),
        RuleCondition(
            name="premium_price",
            check=lambda ctx: (
                (ctx.get("price", 0) - ctx.get("category_avg_price", 0)) /
                ctx.get("category_avg_price", 1) * 100 > 20
            ) if ctx.get("category_avg_price") else False,
            description="평균 대비 20% 이상 높은 가격"
        ),
        RuleCondition(
            name="good_rank",
            check=lambda ctx: ctx.get("rank", 100) <= 10,
            description="Top 10 순위"
        ),
    ],
    conclusion=lambda ctx: {
        "insight": calculate_premium_defense_index(
            ctx, ctx.get("category_avg_price", 0)
        )["insight"],
        "position": "premium_defender",
        "recommendation": "프리미엄 브랜드 가치를 마케팅에 적극 활용하세요.",
        "tag": "프리미엄 방어",
        "tag_color": "gold",
        "related_entities": [ctx.get("asin", ""), ctx.get("brand", "")],
        "metadata": calculate_premium_defense_index(ctx, ctx.get("category_avg_price", 0))
    },
    insight_type=InsightType.PRICE_POSITION,
    priority=8,
    confidence=0.9,
    tags=["price", "premium", "defense", "positive"]
)


# =========================================================================
# 감성 분석 기반 규칙 (Sentiment-Based Rules)
# =========================================================================

RULE_SENTIMENT_STRENGTH_HYDRATION = InferenceRule(
    name="sentiment_strength_hydration",
    description="Hydration 클러스터에서 강한 감성 태그는 보습력 강점을 나타냄",
    conditions=[
        RuleCondition(
            name="has_hydration_sentiment",
            check=lambda ctx: any(
                cluster == "Hydration"
                for cluster in ctx.get("sentiment_clusters", {}).keys()
            ),
            description="Hydration 클러스터 감성 보유"
        ),
        RuleCondition(
            name="hydration_tags_count",
            check=lambda ctx: len(ctx.get("sentiment_clusters", {}).get("Hydration", [])) >= 2,
            description="Hydration 감성 태그 2개 이상"
        ),
    ],
    conclusion=lambda ctx: {
        "insight": f"고객들이 보습력({', '.join(ctx.get('sentiment_clusters', {}).get('Hydration', []))})을 "
                  f"주요 강점으로 인식하고 있습니다.",
        "strength": "hydration",
        "recommendation": "보습력을 핵심 마케팅 메시지로 강조하세요. "
                         "관련 키워드를 제품 설명과 광고에 활용하세요.",
        "related_entities": [ctx.get("asin", ""), ctx.get("brand", "")],
        "metadata": {
            "sentiment_cluster": "Hydration",
            "tags": ctx.get("sentiment_clusters", {}).get("Hydration", [])
        }
    },
    insight_type=InsightType.SENTIMENT_STRENGTH,
    priority=7,
    confidence=0.85,
    tags=["sentiment", "strength", "hydration", "positive"]
)


RULE_SENTIMENT_VALUE_ADVANTAGE = InferenceRule(
    name="sentiment_value_advantage",
    description="가성비 감성이 있고 경쟁사에 없으면 가성비 경쟁 우위",
    conditions=[
        RuleCondition(
            name="has_value_sentiment",
            check=lambda ctx: any(
                tag.lower() in ["value for money", "good value", "affordable", "worth the price"]
                for tag in ctx.get("sentiment_tags", [])
            ),
            description="가성비 관련 감성 태그 보유"
        ),
        RuleCondition(
            name="competitor_lacks_value",
            check=lambda ctx: not any(
                tag.lower() in ["value for money", "good value", "affordable"]
                for tag in ctx.get("competitor_sentiment_tags", [])
            ),
            description="경쟁사에 가성비 감성 없음"
        ),
    ],
    conclusion=lambda ctx: {
        "insight": "경쟁사 대비 '가성비' 인식에서 우위를 보이고 있습니다. "
                  "고객들이 가격 대비 만족도를 높게 평가합니다.",
        "advantage": "value_perception",
        "recommendation": "가성비 메시지를 마케팅에 적극 활용하세요. "
                         "비교 광고나 '가격 대비 효과' 강조 콘텐츠가 효과적일 수 있습니다.",
        "related_entities": [ctx.get("asin", ""), ctx.get("brand", "")],
        "metadata": {
            "sentiment_type": "value",
            "has_competitor_value": False
        }
    },
    insight_type=InsightType.SENTIMENT_ADVANTAGE,
    priority=8,
    confidence=0.8,
    tags=["sentiment", "value", "competitive", "positive"]
)


RULE_SENTIMENT_WEAKNESS_PACKAGING = InferenceRule(
    name="sentiment_weakness_packaging",
    description="패키징 관련 부정 감성이 있으면 패키징 개선 필요",
    conditions=[
        RuleCondition(
            name="no_packaging_positive",
            check=lambda ctx: not any(
                cluster == "Packaging"
                for cluster in ctx.get("sentiment_clusters", {}).keys()
            ),
            description="패키징 긍정 감성 없음"
        ),
        RuleCondition(
            name="competitor_has_packaging",
            check=lambda ctx: any(
                cluster == "Packaging"
                for cluster in ctx.get("competitor_sentiment_clusters", {}).keys()
            ),
            description="경쟁사에 패키징 긍정 감성 있음"
        ),
    ],
    conclusion=lambda ctx: {
        "insight": "경쟁사 대비 패키징 관련 고객 인식이 약합니다. "
                  "패키징 개선 또는 패키징 장점 커뮤니케이션이 필요할 수 있습니다.",
        "weakness": "packaging",
        "recommendation": "패키징 디자인/기능 개선을 검토하거나, "
                         "기존 패키징의 장점을 고객에게 더 잘 전달하세요.",
        "related_entities": [ctx.get("asin", ""), ctx.get("brand", "")],
        "metadata": {
            "sentiment_gap": "packaging",
            "competitor_has": True
        }
    },
    insight_type=InsightType.SENTIMENT_WEAKNESS,
    priority=6,
    confidence=0.75,
    tags=["sentiment", "weakness", "packaging", "risk"]
)


RULE_SENTIMENT_USABILITY_STRENGTH = InferenceRule(
    name="sentiment_usability_strength",
    description="사용 편의성 감성이 강하면 사용성 강점",
    conditions=[
        RuleCondition(
            name="has_usability_sentiment",
            check=lambda ctx: any(
                cluster == "Usability"
                for cluster in ctx.get("sentiment_clusters", {}).keys()
            ),
            description="Usability 클러스터 감성 보유"
        ),
    ],
    conclusion=lambda ctx: {
        "insight": f"고객들이 사용 편의성({', '.join(ctx.get('sentiment_clusters', {}).get('Usability', []))})을 "
                  f"긍정적으로 평가하고 있습니다.",
        "strength": "usability",
        "recommendation": "'간편한 사용법', '휴대성' 등을 마케팅 포인트로 활용하세요.",
        "related_entities": [ctx.get("asin", ""), ctx.get("brand", "")],
        "metadata": {
            "sentiment_cluster": "Usability",
            "tags": ctx.get("sentiment_clusters", {}).get("Usability", [])
        }
    },
    insight_type=InsightType.SENTIMENT_STRENGTH,
    priority=6,
    confidence=0.8,
    tags=["sentiment", "strength", "usability", "positive"]
)


RULE_SENTIMENT_EFFECTIVENESS = InferenceRule(
    name="sentiment_effectiveness_strong",
    description="효과성 관련 감성이 강하면 제품 효과 강점",
    conditions=[
        RuleCondition(
            name="has_effectiveness_sentiment",
            check=lambda ctx: any(
                cluster == "Effectiveness"
                for cluster in ctx.get("sentiment_clusters", {}).keys()
            ),
            description="Effectiveness 클러스터 감성 보유"
        ),
        RuleCondition(
            name="effectiveness_tags_count",
            check=lambda ctx: len(ctx.get("sentiment_clusters", {}).get("Effectiveness", [])) >= 1,
            description="효과성 감성 태그 1개 이상"
        ),
    ],
    conclusion=lambda ctx: {
        "insight": f"고객들이 제품 효과({', '.join(ctx.get('sentiment_clusters', {}).get('Effectiveness', []))})를 "
                  f"높게 평가하고 있습니다. 실제 효과가 구매 결정에 중요한 요소입니다.",
        "strength": "effectiveness",
        "recommendation": "Before/After 콘텐츠, 고객 리뷰 기반 효과 증명 마케팅을 강화하세요.",
        "related_entities": [ctx.get("asin", ""), ctx.get("brand", "")],
        "metadata": {
            "sentiment_cluster": "Effectiveness",
            "tags": ctx.get("sentiment_clusters", {}).get("Effectiveness", [])
        }
    },
    insight_type=InsightType.SENTIMENT_STRENGTH,
    priority=7,
    confidence=0.85,
    tags=["sentiment", "strength", "effectiveness", "positive"]
)


RULE_SENTIMENT_GAP_SENSORY = InferenceRule(
    name="sentiment_gap_sensory",
    description="경쟁사에 Sensory 감성이 있고 자사에 없으면 감각적 경험 격차",
    conditions=[
        RuleCondition(
            name="no_sensory_sentiment",
            check=lambda ctx: not any(
                cluster == "Sensory"
                for cluster in ctx.get("sentiment_clusters", {}).keys()
            ),
            description="Sensory 감성 없음"
        ),
        RuleCondition(
            name="competitor_has_sensory",
            check=lambda ctx: any(
                cluster == "Sensory"
                for cluster in ctx.get("competitor_sentiment_clusters", {}).keys()
            ),
            description="경쟁사에 Sensory 감성 있음"
        ),
    ],
    conclusion=lambda ctx: {
        "insight": "경쟁사 대비 '향', '텍스처' 등 감각적 경험에 대한 고객 인식이 부족합니다.",
        "gap": "sensory_experience",
        "recommendation": "제품의 향, 질감 등 감각적 특성을 마케팅에서 강조하거나 "
                         "제품 포뮬러 개선을 검토하세요.",
        "related_entities": [ctx.get("asin", ""), ctx.get("brand", "")],
        "metadata": {
            "sentiment_gap": "sensory",
            "competitor_has": True
        }
    },
    insight_type=InsightType.SENTIMENT_GAP,
    priority=6,
    confidence=0.75,
    tags=["sentiment", "gap", "sensory", "risk"]
)


RULE_CUSTOMER_PERCEPTION_POSITIVE = InferenceRule(
    name="customer_perception_positive",
    description="AI 요약에 긍정 키워드가 있으면 전반적 고객 인식 긍정",
    conditions=[
        RuleCondition(
            name="has_ai_summary",
            check=lambda ctx: bool(ctx.get("ai_summary")),
            description="AI 요약 데이터 존재"
        ),
        RuleCondition(
            name="positive_keywords_in_summary",
            check=lambda ctx: any(
                keyword in (ctx.get("ai_summary", "") or "").lower()
                for keyword in ["love", "great", "excellent", "amazing", "best", "recommend", "perfect"]
            ),
            description="AI 요약에 긍정 키워드 포함"
        ),
    ],
    conclusion=lambda ctx: {
        "insight": "Amazon AI 리뷰 요약에서 전반적으로 긍정적인 고객 인식이 확인됩니다.",
        "perception": "positive",
        "recommendation": "긍정적 리뷰를 마케팅에 활용하고, 리뷰 수를 늘려 신뢰도를 강화하세요.",
        "related_entities": [ctx.get("asin", ""), ctx.get("brand", "")],
        "metadata": {
            "has_ai_summary": True,
            "perception_type": "positive"
        }
    },
    insight_type=InsightType.CUSTOMER_PERCEPTION,
    priority=7,
    confidence=0.8,
    tags=["sentiment", "perception", "positive"]
)


RULE_CUSTOMER_PERCEPTION_MIXED = InferenceRule(
    name="customer_perception_mixed",
    description="AI 요약에 긍정과 부정이 섞여 있으면 혼합된 인식",
    conditions=[
        RuleCondition(
            name="has_ai_summary",
            check=lambda ctx: bool(ctx.get("ai_summary")),
            description="AI 요약 데이터 존재"
        ),
        RuleCondition(
            name="mixed_keywords_in_summary",
            check=lambda ctx: (
                any(pos in (ctx.get("ai_summary", "") or "").lower()
                    for pos in ["like", "good", "nice"]) and
                any(neg in (ctx.get("ai_summary", "") or "").lower()
                    for neg in ["but", "however", "although", "wish", "could be better"])
            ),
            description="AI 요약에 긍정+부정 키워드 혼재"
        ),
    ],
    conclusion=lambda ctx: {
        "insight": "고객 인식이 긍정과 부정이 혼재되어 있습니다. "
                  "특정 측면에서 개선 기회가 있을 수 있습니다.",
        "perception": "mixed",
        "recommendation": "부정적 피드백의 원인을 분석하고 해당 영역의 개선을 검토하세요.",
        "related_entities": [ctx.get("asin", ""), ctx.get("brand", "")],
        "metadata": {
            "has_ai_summary": True,
            "perception_type": "mixed"
        }
    },
    insight_type=InsightType.CUSTOMER_PERCEPTION,
    priority=6,
    confidence=0.75,
    tags=["sentiment", "perception", "mixed", "neutral"]
)


# 감성 규칙 리스트
SENTIMENT_RULES: List[InferenceRule] = [
    RULE_SENTIMENT_STRENGTH_HYDRATION,
    RULE_SENTIMENT_VALUE_ADVANTAGE,
    RULE_SENTIMENT_WEAKNESS_PACKAGING,
    RULE_SENTIMENT_USABILITY_STRENGTH,
    RULE_SENTIMENT_EFFECTIVENESS,
    RULE_SENTIMENT_GAP_SENSORY,
    RULE_CUSTOMER_PERCEPTION_POSITIVE,
    RULE_CUSTOMER_PERCEPTION_MIXED,
]


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
    RULE_TREND_ALIGNMENT,
    RULE_TOP10_STABILITY,
    RULE_CATEGORY_OPPORTUNITY,
    RULE_RATING_MOMENTUM,
    RULE_TOP3_ACHIEVEMENT,

    # 가격/품질 규칙
    RULE_VALUE_POSITION,
    RULE_PREMIUM_POSITION,
    RULE_STRONG_RATING,

    # 순위-할인 인과관계 규칙
    RULE_DISCOUNT_DEPENDENT,
    RULE_VIRAL_EFFECT,
    RULE_BESTSELLER_BADGE_EFFECT,
    RULE_HIGH_DISCOUNT_DEPENDENCY_SCORE,
    RULE_PREMIUM_DEFENSE_SUCCESS,

    # 감성 분석 규칙
    *SENTIMENT_RULES,
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
