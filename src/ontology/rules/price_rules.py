"""
Price, Discount, and Causality Rules
가격, 할인 및 인과관계 관련 규칙
"""

from typing import Any

from ..reasoner import InferenceRule, RuleCondition, StandardConditions
from ..relations import InsightType

# =========================================================================
# 헬퍼 함수: 순위-할인 상관관계 분석
# =========================================================================


def count_period_overlap(periods_a: list[dict], periods_b: list[dict]) -> int:
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
            b_end = (
                datetime.fromisoformat(b_end_str.replace("Z", "+00:00")) if b_end_str else b_start
            )
        except (ValueError, TypeError):
            continue

        for period_a in periods_a:
            a_start_str = period_a.get("start")
            a_end_str = period_a.get("end")

            if not a_start_str:
                continue

            try:
                a_start = datetime.fromisoformat(a_start_str.replace("Z", "+00:00"))
                a_end = (
                    datetime.fromisoformat(a_end_str.replace("Z", "+00:00"))
                    if a_end_str
                    else a_start
                )
            except (ValueError, TypeError):
                continue

            # 겹침 조건: 한 기간의 시작이 다른 기간의 끝 이전이고,
            # 한 기간의 끝이 다른 기간의 시작 이후
            if b_start <= a_end and a_start <= b_end:
                overlap_count += 1
                break  # 이 period_b는 이미 겹침으로 카운트됨

    return overlap_count


def calculate_discount_dependency_score(product_history: list[dict]) -> float:
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
        prev = product_history[i - 1]
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

    dependency_score = (
        (discount_rank_correlation / (total_periods - 1)) * 100 if total_periods > 1 else 0
    )
    return round(dependency_score, 1)


def calculate_premium_defense_index(
    product_data: dict, category_avg_price: float
) -> dict[str, Any]:
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
            "insight": "가격 정보가 충분하지 않습니다.",
        }

    price_premium = (price - category_avg_price) / category_avg_price * 100

    # 프리미엄 가격인데 좋은 순위면 높은 점수
    if price_premium > 20:  # 평균보다 20% 이상 비쌈
        if rank <= 10:
            return {
                "index": 90,
                "interpretation": "강력한 프리미엄 방어",
                "insight": f"평균보다 {price_premium:.0f}% 높은 가격에도 Top 10 유지",
            }
        elif rank <= 30:
            return {
                "index": 70,
                "interpretation": "양호한 프리미엄 방어",
                "insight": "프리미엄 가격에서 상위권 유지",
            }
        else:
            return {
                "index": 40,
                "interpretation": "프리미엄 방어 취약",
                "insight": "높은 가격이 순위에 부정적 영향",
            }
    else:
        return {
            "index": 50,
            "interpretation": "일반 가격대",
            "insight": "카테고리 평균 수준의 가격",
        }


def rule_discount_dependent(product_data: dict) -> dict | None:
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
            "related_entities": [product_data.get("asin", ""), product_data.get("brand", "")],
        }
    return None


def rule_viral_effect(product_data: dict) -> dict | None:
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
            "related_entities": [product_data.get("asin", ""), product_data.get("brand", "")],
        }
    return None


def rule_bestseller_badge_effect(product_data: dict) -> dict | None:
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
            "related_entities": [product_data.get("asin", ""), product_data.get("brand", "")],
        }
    return None


# =========================================================================
# 규칙: 가성비 포지션 (Value Position)
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
        "metadata": {"cpi": ctx.get("cpi"), "rating_gap": ctx.get("rating_gap")},
    },
    insight_type=InsightType.PRICE_POSITION,
    priority=6,
    confidence=0.85,
    tags=["price", "value", "positive"],
)


# =========================================================================
# 규칙: 프리미엄 가격 포지션 (Premium Price Position)
# =========================================================================

RULE_PREMIUM_POSITION = InferenceRule(
    name="premium_price_position",
    description="높은 CPI와 높은 평점은 성공적인 프리미엄 포지셔닝을 나타냄",
    conditions=[
        StandardConditions.cpi_above(150),  # CPI > 150 (프리미엄)
        RuleCondition(
            name="good_rating",
            check=lambda ctx: ctx.get("rating_gap", 0) >= 0,
            description="평점 동등 이상",
        ),
    ],
    conclusion=lambda ctx: {
        "insight": f"CPI {ctx.get('cpi', 0):.0f}의 프리미엄 가격 포지션에서 "
        f"평점 경쟁력을 유지하고 있어 성공적인 프리미엄 전략을 수행 중입니다.",
        "position": "premium_success",
        "recommendation": "프리미엄 브랜드 가치 강화 지속. 신규 프리미엄 라인 확장 검토.",
        "related_entities": [ctx.get("brand", "")],
        "metadata": {"cpi": ctx.get("cpi"), "rating_gap": ctx.get("rating_gap")},
    },
    insight_type=InsightType.PRICE_POSITION,
    priority=7,
    confidence=0.85,
    tags=["price", "premium", "positive"],
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
            description="할인 기간 데이터 존재",
        ),
        RuleCondition(
            name="has_rank_improvements",
            check=lambda ctx: bool(ctx.get("rank_improvements")),
            description="순위 상승 기간 데이터 존재",
        ),
        RuleCondition(
            name="high_overlap_ratio",
            check=lambda ctx: (
                count_period_overlap(
                    ctx.get("discount_periods", []), ctx.get("rank_improvements", [])
                )
                / len(ctx.get("rank_improvements", [1]))
                >= 0.8
            )
            if ctx.get("rank_improvements")
            else False,
            description="할인-순위상승 일치율 >= 80%",
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
                ctx.get("discount_periods", []), ctx.get("rank_improvements", [])
            )
            / len(ctx.get("rank_improvements", [1]))
            if ctx.get("rank_improvements")
            else 0,
        },
    },
    insight_type=InsightType.PRICE_DEPENDENCY,
    priority=8,
    confidence=0.85,
    tags=["price", "discount", "dependency", "alert"],
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
            description="가격 안정",
        ),
        RuleCondition(
            name="rank_improved",
            check=lambda ctx: ctx.get("rank_change_7d", 0) < 0,
            description="순위 상승 (음수)",
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
        "metadata": {"rank_change_7d": ctx.get("rank_change_7d"), "price_stable": True},
    },
    insight_type=InsightType.BRAND_STRENGTH,
    priority=8,
    confidence=0.8,
    tags=["brand", "viral", "growth", "positive"],
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
            description="베스트셀러 배지 보유",
        ),
        RuleCondition(
            name="rank_stable",
            check=lambda ctx: abs(ctx.get("rank_change_7d", 0)) <= 3,
            description="순위 변동 ±3 이내 (안정)",
        ),
    ],
    conclusion=lambda ctx: {
        "insight": "베스트셀러 배지를 유지하며 순위가 안정적입니다.",
        "position": "bestseller_stable",
        "recommendation": "현 포지션 유지를 위해 재고 관리와 리뷰 관리에 집중하세요.",
        "tag": "베스트셀러 유지",
        "tag_color": "blue",
        "related_entities": [ctx.get("asin", ""), ctx.get("brand", "")],
        "metadata": {"badge": ctx.get("badge"), "rank_change_7d": ctx.get("rank_change_7d")},
    },
    insight_type=InsightType.MARKET_POSITION,
    priority=7,
    confidence=0.9,
    tags=["badge", "stability", "positive"],
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
            description="제품 이력 데이터 존재",
        ),
        RuleCondition(
            name="high_dependency",
            check=lambda ctx: calculate_discount_dependency_score(ctx.get("product_history", []))
            >= 61,
            description="할인 의존도 점수 >= 61",
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
        },
    },
    insight_type=InsightType.PRICE_DEPENDENCY,
    priority=7,
    confidence=0.85,
    tags=["price", "discount", "dependency", "risk"],
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
            description="가격 데이터 존재",
        ),
        RuleCondition(
            name="premium_price",
            check=lambda ctx: (
                (ctx.get("price", 0) - ctx.get("category_avg_price", 0))
                / ctx.get("category_avg_price", 1)
                * 100
                > 20
            )
            if ctx.get("category_avg_price")
            else False,
            description="평균 대비 20% 이상 높은 가격",
        ),
        RuleCondition(
            name="good_rank",
            check=lambda ctx: ctx.get("rank", 100) <= 10,
            description="Top 10 순위",
        ),
    ],
    conclusion=lambda ctx: {
        "insight": calculate_premium_defense_index(ctx, ctx.get("category_avg_price", 0))[
            "insight"
        ],
        "position": "premium_defender",
        "recommendation": "프리미엄 브랜드 가치를 마케팅에 적극 활용하세요.",
        "tag": "프리미엄 방어",
        "tag_color": "gold",
        "related_entities": [ctx.get("asin", ""), ctx.get("brand", "")],
        "metadata": calculate_premium_defense_index(ctx, ctx.get("category_avg_price", 0)),
    },
    insight_type=InsightType.PRICE_POSITION,
    priority=8,
    confidence=0.9,
    tags=["price", "premium", "defense", "positive"],
)


PRICE_RULES: list[InferenceRule] = [
    RULE_VALUE_POSITION,
    RULE_PREMIUM_POSITION,
    RULE_DISCOUNT_DEPENDENT,
    RULE_VIRAL_EFFECT,
    RULE_BESTSELLER_BADGE_EFFECT,
    RULE_HIGH_DISCOUNT_DEPENDENCY_SCORE,
    RULE_PREMIUM_DEFENSE_SUCCESS,
]
