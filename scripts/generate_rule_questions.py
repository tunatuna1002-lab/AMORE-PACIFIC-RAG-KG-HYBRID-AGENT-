#!/usr/bin/env python3
"""③규칙 판단 유형 문항을 규칙 엔진 코드와 2026-08-31 DB에서 **기계적으로** 생성한다.

배경 (docs/plans/evidence-react-ontology-kickoff-prompt-2026-09-17.md 1-B)
--------------------------------------------------------------------------
골든셋에는 규칙 판단 전용 gold가 없다. 규칙 엔진(`src/ontology/rules/*.py`, 37개)의 조건과
임계값을 코드에서 읽고, 입력값은 `raw_data`에서 정본 공식으로 독립 계산해(scripts/
golden_snapshot_facts.py) 정답 결론을 만든다. LLM·유료 API를 쓰지 않는다.

정답 결론을 정하는 방법
----------------------
1. 입력값: raw_data 독립 계산 (golden_snapshot_facts.category_facts).
2. 결론: 규칙이 **의도한 단위**로 문맥을 만든 뒤 실제 규칙 객체의 `evaluate_conditions`
   (조건 하나만 묻는 문항은 해당 `RuleCondition.evaluate`)로 판정한다. 같은 판정을 문항
   정의의 임계값 산술로도 따로 계산해 둘이 다르면 생성을 중단한다(해석 오류 방지).
   - SoS: DB는 0~100(%), 규칙 임계는 0~1(`sos_above(0.15)`의 설명이 "SoS >= 15%").
     그래서 문맥에는 sos/100을 넣는다. DB 값을 그대로 넣었을 때의 결론도
     `conclusion_if_sos_unconverted`로 남긴다(스케일 불일치의 영향 기록 — 고치지 않는다).
   - HHI: DB·규칙 모두 0~1. CPI: DB·규칙 모두 100 기준. 평점 갭: 평점 차(점).
3. 교차 확인: 같은 입력을 지표 테이블(brand_metrics·market_metrics)에서 읽어 결론을 다시
   내고, `MetricFactsProvider`(챗봇이 실제로 받는 DB 사실)가 무엇을 주는지 기록한다.
   **두 출처의 결론이 다르거나 기대 수치가 10%를 넘게 어긋나면 그 후보는 싣지 않고**
   generation_log_rule.json의 excluded에 이유와 함께 남긴다 — 어느 출처를 믿느냐에 따라
   정답이 달라지는 문항은 분모로 쓸 수 없다.

규칙별 입력 가용성
------------------
RULE_INPUTS에 37개 규칙의 입력과 출처 상태를 선언했다. 생성은 모든 입력이
`db_snapshot`·`kg_curated`·`target_flag`인 규칙만 한다. 나머지는 이유를 로그에 남긴다
(규칙 미발화 원인 분석용). tests/eval/test_classify_golden_types.py가 표가
ALL_BUSINESS_RULES와 정확히 일치하는지 확인한다.

사용법:
    python3 scripts/generate_rule_questions.py --db <amore_data.db> --kg <knowledge_graph.json>
    python3 scripts/generate_rule_questions.py --db ... --kg ... --check
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import golden_snapshot_facts as gf  # noqa: E402

GENERATOR = "scripts/generate_rule_questions.py"

# ---------------------------------------------------------------------------
# 규칙 입력 가용성 (2026-08-31 기준 DB·KG 실측, 2026-09-17 확인)
# ---------------------------------------------------------------------------
DB = "db_snapshot"  # as_of 날짜의 지표 테이블 또는 raw_data 컬럼 + 정본 공식으로 바로 계산
KG = "kg_curated"  # KG 큐레이션 트리플(config/brands.json)
TARGET = "target_flag"  # 브랜드명이 LANEIGE인지로 정해지는 플래그
HISTORY = "history_not_stored"  # 여러 날짜 raw_data와 코드 공식이 있으나 as_of 저장값 없음
UNDEFINED = "no_derivation_defined"  # DB에서 만드는 코드 정의 자체가 없음
EMPTY = "db_column_empty"  # 컬럼은 있으나 as_of 날짜 값이 전부 NULL
KG_UNVERSIONED = "kg_unversioned"  # KG에 있으나 날짜 버전 없음·표기 이중화로 값이 조회 방식에 좌우
ABSENT = "absent"  # DB·KG 어디에도 없음(감성·IR 구조화 값·외부 트렌드)
GENERATABLE = {DB, KG, TARGET}

RULE_INPUTS: dict[str, dict[str, Any]] = {
    # --- market_rules.py ---
    "market_dominance_fragmented": {
        "family": "market",
        "inputs": {"sos": DB, "hhi": DB},
    },
    "market_dominance_concentrated": {
        "family": "market",
        "inputs": {"sos": DB, "hhi": DB},
    },
    "challenger_position": {"family": "market", "inputs": {"hhi": DB, "sos": DB}},
    "fragmented_market_competition": {
        "family": "market",
        "inputs": {"hhi": DB, "competitor_count": KG_UNVERSIONED},
        "note": "competitor_count = KG get_competitors(brand) 개수. competesWith 616개가 "
        "kg_enricher 산출(날짜 버전 없음)이고 'LANEIGE'/'laneige' 주어가 따로 있어 "
        "조회 표기에 따라 개수가 달라진다",
    },
    "strong_avg_rank": {"family": "market", "inputs": {"avg_rank": DB, "is_target": TARGET}},
    "competitive_pressure": {
        "family": "market",
        "inputs": {"sos_change": UNDEFINED, "competitor_count": KG_UNVERSIONED},
        "note": "sos_change의 비교 기간을 정한 코드가 없다(_build_inference_context도 설정 안 함)",
    },
    # --- alert_rules.py ---
    "price_quality_mismatch": {
        "family": "alert",
        "inputs": {"cpi": DB, "rating_gap": DB},
        "note": "brand_metrics.cpi는 2026-08-31에 face_powder만 값이 있다(나머지 카테고리 NULL, "
        "lip_makeup은 raw_data price도 전부 NULL)",
    },
    "market_disruption": {
        "family": "alert",
        "inputs": {"has_rank_shock": HISTORY, "churn_rate": HISTORY},
        "note": "market_metrics.churn_rate 전부 NULL, product_metrics 0행. "
        "metric_calculator.calculate_churn_rate/calculate_rank_shock 공식은 있음",
    },
    "rank_decline_alert": {
        "family": "alert",
        "inputs": {"rank_change_7d": HISTORY, "rank_volatility": HISTORY},
    },
    # --- growth_rules.py ---
    "stable_growth": {
        "family": "growth",
        "inputs": {"streak_days": HISTORY, "rank_change_7d": HISTORY},
    },
    "trend_alignment_opportunity": {
        "family": "growth",
        "inputs": {"trend_keywords": ABSENT, "is_target": TARGET},
    },
    "top10_stability": {
        "family": "growth",
        "inputs": {"current_rank": DB, "streak_days": HISTORY, "rank_volatility": HISTORY},
    },
    "category_entry_opportunity": {
        "family": "growth",
        "inputs": {"hhi": DB, "sos": DB, "is_target": TARGET},
    },
    "rating_momentum_positive": {
        "family": "growth",
        "inputs": {"rating_trend": HISTORY, "review_count": DB},
    },
    "top3_achievement": {
        "family": "growth",
        "inputs": {"current_rank": DB, "is_target": TARGET},
    },
    "strong_rating_position": {
        "family": "growth",
        "inputs": {"rating_gap": DB, "is_target": TARGET},
    },
    # --- price_rules.py ---
    "value_position": {"family": "price", "inputs": {"cpi": DB, "rating_gap": DB}},
    "premium_price_position": {"family": "price", "inputs": {"cpi": DB, "rating_gap": DB}},
    "discount_dependent": {
        "family": "price",
        "inputs": {"discount_periods": UNDEFINED, "rank_improvements": UNDEFINED},
        "note": "raw_data.discount_percent도 2026-08-31 전부 NULL",
    },
    "viral_effect": {
        "family": "price",
        "inputs": {"price_stable": UNDEFINED, "rank_change_7d": HISTORY},
    },
    "bestseller_badge_effect": {
        "family": "price",
        "inputs": {"badge": EMPTY, "rank_change_7d": HISTORY},
    },
    "high_discount_dependency_score": {
        "family": "price",
        "inputs": {"product_history": UNDEFINED},
        "note": "점수 함수는 있으나 이력 구성(기간·정렬) 정의가 없고 discount_percent 전부 NULL",
    },
    "premium_defense_success": {
        "family": "price",
        "inputs": {"price": DB, "category_avg_price": DB, "rank": DB},
        "note": "market_metrics.category_avg_price는 2026-08-31 face_powder만 값이 있다",
    },
    # --- sentiment_rules.py ---
    **{
        name: {"family": "sentiment", "inputs": dict.fromkeys(keys, ABSENT)}
        for name, keys in {
            "sentiment_strength_hydration": ["sentiment_clusters"],
            "sentiment_value_advantage": ["sentiment_tags", "competitor_sentiment_tags"],
            "sentiment_weakness_packaging": [
                "sentiment_clusters",
                "competitor_sentiment_clusters",
            ],
            "sentiment_usability_strength": ["sentiment_clusters"],
            "sentiment_effectiveness_strong": ["sentiment_clusters"],
            "sentiment_gap_sensory": ["sentiment_clusters", "competitor_sentiment_clusters"],
            "customer_perception_positive": ["ai_summary"],
            "customer_perception_mixed": ["ai_summary"],
        }.items()
    },
    # --- ir_rules.py ---
    "ir_prime_day_impact": {
        "family": "ir",
        "inputs": {"ir_mentions_prime_day": ABSENT, "rank_change_during_event": UNDEFINED},
    },
    "ir_americas_revenue_correlation": {
        "family": "ir",
        "inputs": {"ir_americas_yoy": ABSENT, "sos_change": UNDEFINED, "is_target": TARGET},
    },
    "ir_growth_momentum": {
        "family": "ir",
        "inputs": {"ir_consecutive_growth_quarters": ABSENT},
    },
    "ir_growth_slowdown_warning": {
        "family": "ir",
        "inputs": {"ir_current_qtr_growth": ABSENT, "ir_prev_qtr_growth": ABSENT},
    },
    "ir_brand_campaign_effect": {
        "family": "ir",
        "inputs": {"ir_campaign_mentioned": ABSENT, "rank_change_7d": HISTORY},
    },
    "brand_ownership_verification": {
        "family": "ir",
        "inputs": {"parent_group": KG},
        "note": "KG ownedByGroup(config/brands.json)",
    },
}

SOS_SCALE_NOTE = (
    "DB sos는 0~100(%)이고 규칙 임계는 0~1이다(sos_above(0.15)의 설명 'SoS >= 15%'). "
    "정답은 규칙 의도대로 sos/100으로 판정했다. conclusion_if_sos_unconverted는 DB 값을 "
    "변환 없이 넣었을 때 규칙 엔진이 내는 결론이다."
)

# 지표 → metric_facts가 챗봇에 주는지 (src/rag/metric_facts.py 필드 목록)
METRIC_FACTS_SERVES = {
    "sos": "brand_share.sos",
    "hhi": "category_market.hhi",
    "category_avg_price": "category_market.category_avg_price",
    "current_rank": "brand_products.products[0].rank (브랜드 상위 3개 제품)",
    "price": "category_top_products/brand_products 의 price (상위 5·3개 제품만)",
    "avg_rank": None,
    "cpi": None,
    "rating_gap": None,
    "parent_group": None,
}


def rules_by_name() -> dict[str, Any]:
    from src.ontology.rules import ALL_BUSINESS_RULES

    return {r.name: r for r in ALL_BUSINESS_RULES}


# ---------------------------------------------------------------------------
# 후보 정의
# ---------------------------------------------------------------------------


@dataclass
class Inputs:
    """한 출처(raw 또는 지표 테이블)에서 읽은 규칙 입력."""

    source: str
    values: dict[str, Any] = field(default_factory=dict)


@dataclass
class Candidate:
    kind: str
    rule_ids: list[str]
    condition: str | None  # 조건 하나만 묻는 문항이면 RuleCondition.name
    category: str
    subject: str  # 브랜드명(raw_data 표기) 또는 카테고리 id 또는 "rank:<n>"
    question: str
    thresholds: dict[str, Any]
    expected: Callable[[dict[str, Any]], bool]  # 규칙 의도 단위 문맥 → 임계 산술 판정
    ev_keys: dict[str, str]  # expected_values 키 → 입력 이름
    domain: str
    difficulty: str
    concepts: list[str]
    requires_kg: bool = False
    gold_source: str = "snapshot"
    extra_inputs: tuple[str, ...] = ()  # 기대값에는 없지만 판정에 쓰는 입력


def raw_inputs(facts: gf.CategoryFacts, subject: str) -> dict[str, Any] | None:
    values: dict[str, Any] = {
        "hhi": facts.hhi,
        "category_avg_price": (round(facts.avg_price, 2) if facts.avg_price is not None else None),
    }
    if subject.startswith("rank:"):
        rank = int(subject.split(":")[1])
        product = next((p for p in facts.products if p["rank"] == rank), None)
        if product is None:
            return None
        values.update(
            {
                "rank": rank,
                "price": product["price"],
                "brand": product["brand"],
                "asin": product["asin"],
                "product_name": product["product_name"],
            }
        )
        return values
    if subject == facts.category:
        return values
    brand = facts.brands.get(subject)
    if brand is None:
        return None
    values.update(
        {
            "brand": brand.brand,
            "sos": brand.sos,
            "avg_rank": brand.avg_rank,
            "current_rank": brand.best_rank,
            "cpi": brand.cpi,
            "rating_gap": brand.rating_gap,
            "product_count": brand.count,
        }
    )
    return values


def table_inputs(table: dict, raw: dict[str, Any], subject: str) -> dict[str, Any]:
    """지표 테이블 값으로 같은 입력을 만든다. 테이블에 없는 입력(순위·제품 가격)은 raw와 같다."""
    market = table["market"] or {}
    values = dict(raw)
    values["hhi"] = market.get("hhi")
    values["category_avg_price"] = market.get("category_avg_price")
    if subject.startswith("rank:") or "brand" not in raw or "sos" not in raw:
        return values
    row = table["brands"].get(subject)
    values.update(
        {
            "sos": row["sos"] if row else None,
            "avg_rank": row["brand_avg_rank"] if row else None,
            "cpi": row["cpi"] if row else None,
            "rating_gap": row["avg_rating_gap"] if row else None,
            "product_count": row["product_count"] if row else None,
        }
    )
    return values


def rule_context(values: dict[str, Any], category: str, convert_sos: bool = True) -> dict:
    sos = values.get("sos")
    brand = values.get("brand", "")
    return {
        "brand": brand,
        "category": category,
        "is_target": str(brand).lower() == "laneige",
        "sos": (sos / 100 if convert_sos else sos) if sos is not None else None,
        "hhi": values.get("hhi"),
        "avg_rank": values.get("avg_rank"),
        "current_rank": values.get("current_rank"),
        "cpi": values.get("cpi"),
        "rating_gap": values.get("rating_gap"),
        "price": values.get("price"),
        "category_avg_price": values.get("category_avg_price"),
        "rank": values.get("rank"),
    }


def engine_verdict(rules: dict, cand: Candidate, ctx: dict) -> bool:
    if cand.condition:
        rule = rules[cand.rule_ids[0]]
        cond = next(c for c in rule.conditions if c.name == cand.condition)
        return bool(cond.evaluate(ctx))
    return all(rules[r].evaluate_conditions(ctx)[0] for r in cand.rule_ids)


# ---------------------------------------------------------------------------
# 후보 목록 (선택 순서 = 이 목록 순서, 종류별 상한 CAPS)
# ---------------------------------------------------------------------------

L = gf.CATEGORY_LABEL
D = gf.AS_OF


def build_candidates(facts: dict[str, gf.CategoryFacts], kg_brands_ap: set[str]) -> list[Candidate]:
    c: list[Candidate] = []

    # A. 분산 시장 조건 (HHI < 0.15) — 조건을 쓰는 규칙 3개
    for cat in ("lip_care", "skin_care", "lip_makeup", "face_powder", "beauty"):
        c.append(
            Candidate(
                kind="A_fragmented_market",
                rule_ids=[
                    "market_dominance_fragmented",
                    "fragmented_market_competition",
                    "category_entry_opportunity",
                ],
                condition="hhi_below_0.15",
                category=cat,
                subject=cat,
                question=f"{D} 기준 {L[cat]} 카테고리는 규칙 엔진 기준 분산 시장(HHI < 0.15)인가요?",
                thresholds={"hhi_below": 0.15},
                expected=lambda x: x["hhi"] < 0.15,
                ev_keys={"hhi": "hhi"},
                domain="market",
                difficulty="easy",
                concepts=["hhi", "market_concentration"],
            )
        )
    # B. 집중 시장 조건 (HHI >= 0.25)
    for cat in ("lip_makeup", "skin_care"):
        c.append(
            Candidate(
                kind="B_concentrated_market",
                rule_ids=["market_dominance_concentrated", "challenger_position"],
                condition="hhi_above_0.25",
                category=cat,
                subject=cat,
                question=f"{D} 기준 {L[cat]} 카테고리는 규칙 엔진 기준 집중 시장(HHI ≥ 0.25)인가요?",
                thresholds={"hhi_at_least": 0.25},
                expected=lambda x: x["hhi"] >= 0.25,
                ev_keys={"hhi": "hhi"},
                domain="market",
                difficulty="easy",
                concepts=["hhi", "market_concentration"],
            )
        )
    # C. 분산 시장 지배자 — 카테고리 SoS 1위 브랜드
    for cat in ("lip_care", "skin_care", "face_powder", "lip_makeup"):
        ranked = facts[cat].ranked_brands()
        if len(ranked) < 2 or ranked[0].sos == ranked[1].sos:
            continue  # 1위가 동률이면 "1위 브랜드"가 모호하다
        top = ranked[0].brand
        c.append(
            Candidate(
                kind="C_dominance_fragmented",
                rule_ids=["market_dominance_fragmented"],
                condition=None,
                category=cat,
                subject=top,
                question=(
                    f"{D} 기준 {top}는 {L[cat]}에서 규칙 market_dominance_fragmented"
                    "(SoS ≥ 15% 그리고 HHI < 0.15)의 '분산 시장 지배자' 포지션에 해당하나요?"
                ),
                thresholds={"sos_at_least_pct": 15, "hhi_below": 0.15},
                expected=lambda x: x["sos"] >= 0.15 and x["hhi"] < 0.15,
                ev_keys={"sos": "sos", "hhi": "hhi"},
                domain="brand",
                difficulty="medium",
                concepts=["sos", "hhi", "market_position"],
            )
        )
    # D. 도전자 포지션 — SoS 5~15% 브랜드
    for cat in ("skin_care", "lip_makeup"):
        mid = [b for b in facts[cat].ranked_brands() if b.sos is not None and 5 <= b.sos < 15]
        if not mid:
            continue
        b = mid[0].brand
        c.append(
            Candidate(
                kind="D_challenger",
                rule_ids=["challenger_position"],
                condition=None,
                category=cat,
                subject=b,
                question=(
                    f"{D} 기준 {b}는 {L[cat]}에서 규칙 challenger_position"
                    "(HHI ≥ 0.25 그리고 SoS 5% 이상 15% 미만)의 '도전자' 포지션에 해당하나요?"
                ),
                thresholds={"hhi_at_least": 0.25, "sos_pct_range": [5, 15]},
                expected=lambda x: x["hhi"] >= 0.25 and 0.05 <= x["sos"] < 0.15,
                ev_keys={"sos": "sos", "hhi": "hhi"},
                domain="brand",
                difficulty="medium",
                concepts=["sos", "hhi", "market_position"],
            )
        )
    laneige_cats = [
        cat
        for cat in ("lip_care", "lip_makeup", "face_powder", "skin_care", "beauty")
        if "LANEIGE" in facts[cat].brands
    ]
    # E. 평균 순위 우위
    for cat in laneige_cats:
        c.append(
            Candidate(
                kind="E_strong_avg_rank",
                rule_ids=["strong_avg_rank"],
                condition=None,
                category=cat,
                subject="LANEIGE",
                question=(
                    f"{D} 기준 LANEIGE의 {L[cat]} Top 100 제품 평균 순위는 규칙 strong_avg_rank"
                    "(타겟 브랜드의 평균 순위 < 20) 조건을 충족하나요?"
                ),
                thresholds={"avg_rank_below": 20},
                expected=lambda x: x["avg_rank"] < 20,
                ev_keys={"avg_rank": "avg_rank"},
                domain="brand",
                difficulty="medium",
                concepts=["product_ranking", "market_position"],
            )
        )
    # F. 카테고리 진입 기회
    for cat in laneige_cats:
        c.append(
            Candidate(
                kind="F_category_entry_opportunity",
                rule_ids=["category_entry_opportunity"],
                condition=None,
                category=cat,
                subject="LANEIGE",
                question=(
                    f"{D} 기준 {L[cat]}는 LANEIGE에게 규칙 category_entry_opportunity"
                    "(HHI < 0.15 그리고 LANEIGE SoS < 3%) 기준 '진입 확대 기회' 카테고리인가요?"
                ),
                thresholds={"hhi_below": 0.15, "sos_below_pct": 3},
                expected=lambda x: x["hhi"] < 0.15 and x["sos"] < 0.03,
                ev_keys={"sos": "sos", "hhi": "hhi"},
                domain="brand",
                difficulty="medium",
                concepts=["sos", "hhi", "market_opportunity"],
            )
        )
    # G. Top 3 달성
    for cat in laneige_cats:
        c.append(
            Candidate(
                kind="G_top3_achievement",
                rule_ids=["top3_achievement"],
                condition=None,
                category=cat,
                subject="LANEIGE",
                question=(
                    f"{D} 기준 LANEIGE는 {L[cat]}에서 규칙 top3_achievement"
                    "(최고 순위 제품이 Top 3 이내) 조건을 충족하나요?"
                ),
                thresholds={"current_rank_at_most": 3},
                expected=lambda x: x["current_rank"] <= 3,
                ev_keys={"best_rank": "current_rank"},
                domain="brand",
                difficulty="easy",
                concepts=["product_ranking", "top_3"],
            )
        )
    # H. 평점 경쟁 우위
    for cat in laneige_cats:
        c.append(
            Candidate(
                kind="H_strong_rating",
                rule_ids=["strong_rating_position"],
                condition=None,
                category=cat,
                subject="LANEIGE",
                question=(
                    f"{D} 기준 LANEIGE는 {L[cat]}에서 규칙 strong_rating_position"
                    "(브랜드 평균 평점 − 카테고리 평균 평점 > 0.05) 조건을 충족하나요?"
                ),
                thresholds={"rating_gap_above": 0.05},
                expected=lambda x: x["rating_gap"] > 0.05,
                ev_keys={"avg_rating_diff": "rating_gap"},
                domain="brand",
                difficulty="medium",
                concepts=["review_rating", "competitor_analysis"],
            )
        )
    # I~K. 가격 포지션 (CPI) — face_powder 브랜드, 이름순
    fp = facts["face_powder"]
    for b in sorted(fp.brands):
        c.append(
            Candidate(
                kind="I_price_quality_mismatch",
                rule_ids=["price_quality_mismatch"],
                condition=None,
                category="face_powder",
                subject=b,
                question=(
                    f"{D} 기준 {b}는 Face Powder에서 규칙 price_quality_mismatch"
                    "(CPI > 110 그리고 평점 갭 < 0)의 '가격-품질 불일치' 상태인가요?"
                ),
                thresholds={"cpi_above": 110, "rating_gap_below": 0},
                expected=lambda x: x["cpi"] > 110 and x["rating_gap"] < 0,
                ev_keys={"cpi": "cpi", "avg_rating_diff": "rating_gap"},
                domain="brand",
                difficulty="medium",
                concepts=["cpi", "price_positioning", "review_rating"],
            )
        )
        c.append(
            Candidate(
                kind="J_value_position",
                rule_ids=["value_position"],
                condition=None,
                category="face_powder",
                subject=b,
                question=(
                    f"{D} 기준 {b}는 Face Powder에서 규칙 value_position"
                    "(CPI < 90 그리고 평점 갭 > 0)의 '가성비 포지션'에 해당하나요?"
                ),
                thresholds={"cpi_below": 90, "rating_gap_above": 0},
                expected=lambda x: x["cpi"] < 90 and x["rating_gap"] > 0,
                ev_keys={"cpi": "cpi", "avg_rating_diff": "rating_gap"},
                domain="brand",
                difficulty="medium",
                concepts=["cpi", "price_positioning", "review_rating"],
            )
        )
        c.append(
            Candidate(
                kind="K_premium_position",
                rule_ids=["premium_price_position"],
                condition=None,
                category="face_powder",
                subject=b,
                question=(
                    f"{D} 기준 {b}는 Face Powder에서 규칙 premium_price_position"
                    "(CPI > 150 그리고 평점 갭 ≥ 0)의 '프리미엄 성공' 포지션에 해당하나요?"
                ),
                thresholds={"cpi_above": 150, "rating_gap_at_least": 0},
                expected=lambda x: x["cpi"] > 150 and x["rating_gap"] >= 0,
                ev_keys={"cpi": "cpi", "avg_rating_diff": "rating_gap"},
                domain="brand",
                difficulty="medium",
                concepts=["cpi", "price_positioning", "review_rating"],
            )
        )
    # L. 프리미엄 방어 — face_powder Top 10 제품
    for rank in range(1, 11):
        c.append(
            Candidate(
                kind="L_premium_defense",
                rule_ids=["premium_defense_success"],
                condition=None,
                category="face_powder",
                subject=f"rank:{rank}",
                question="",  # 제품명이 필요해 materialize에서 채운다
                thresholds={"price_premium_pct_above": 20, "rank_at_most": 10},
                expected=lambda x: (x["price"] - x["category_avg_price"])
                / x["category_avg_price"]
                * 100
                > 20
                and x["rank"] <= 10,
                ev_keys={"price": "price", "category_avg_price": "category_avg_price"},
                domain="product",
                difficulty="hard",
                concepts=["price_analysis", "product_ranking"],
                extra_inputs=("rank",),
            )
        )
    return c


CAPS = {
    "A_fragmented_market": 3,
    "B_concentrated_market": 1,
    "C_dominance_fragmented": 3,
    "D_challenger": 1,
    "E_strong_avg_rank": 3,
    "F_category_entry_opportunity": 3,
    "G_top3_achievement": 2,
    "H_strong_rating": 3,
    "I_price_quality_mismatch": 3,
    "J_value_position": 2,
    "K_premium_position": 2,
    "L_premium_defense": 3,
}


def _balanced_pick(accepted: list[dict], cap: int) -> list[dict]:
    """같은 종류 안에서 결론 참/거짓을 번갈아 고른다(입력 순서 유지, 결정적)."""
    yes = [a for a in accepted if a["fires"]]
    no = [a for a in accepted if not a["fires"]]
    picked: list[dict] = []
    while len(picked) < cap and (yes or no):
        for pool in (yes, no):
            if pool and len(picked) < cap:
                picked.append(pool.pop(0))
    return sorted(picked, key=lambda a: a["order"])


def fmt_value(key: str, value: Any) -> str:
    if value is None:
        return "없음"
    if key == "sos":
        return f"{value:g}%"
    if key in ("hhi",):
        return f"{value:.4f}"
    if key in ("avg_rank", "current_rank", "rank"):
        return f"{value:g}위"
    if key in ("price", "category_avg_price"):
        return f"${value:.2f}"
    if key == "rating_gap":
        return f"{value:+.3f}"
    return f"{value:g}" if isinstance(value, float) else str(value)


INPUT_LABEL = {
    "sos": "SoS",
    "hhi": "HHI",
    "avg_rank": "평균 순위",
    "current_rank": "최고 순위",
    "cpi": "CPI",
    "rating_gap": "평점 갭",
    "price": "가격",
    "category_avg_price": "카테고리 평균가",
    "rank": "순위",
}


def materialize(cand: Candidate, raw: dict[str, Any], fires: bool, order: int) -> dict[str, Any]:
    used = sorted(set(cand.ev_keys.values()) | set(cand.extra_inputs))
    facts_txt = ", ".join(f"{INPUT_LABEL[k]} {fmt_value(k, raw[k])}" for k in used)
    verdict = "네" if fires else "아니요"
    question = cand.question
    subject_txt = f"{L[cand.category]} {raw['brand']}" if raw.get("brand") else L[cand.category]
    if cand.kind == "L_premium_defense":
        name = (raw["product_name"] or "").split(",")[0].split(":")[0].strip()
        if name.lower().startswith(raw["brand"].lower()):
            name = name[len(raw["brand"]) :].strip()
        name = name[:50]
        question = (
            f"{D} 기준 Face Powder {raw['rank']}위 제품({raw['brand']} {name})은 규칙 "
            "premium_defense_success(가격이 카테고리 평균가보다 20% 초과 비싸고 Top 10 이내) "
            "조건에 해당하나요?"
        )
        premium = (raw["price"] - raw["category_avg_price"]) / raw["category_avg_price"] * 100
        facts_txt += f", 평균 대비 {premium:+.1f}%"
        subject_txt = f"Face Powder {raw['rank']}위 {raw['brand']} 제품"
    rule_txt = cand.condition or ", ".join(cand.rule_ids)
    answer = (
        f"{verdict}. {D} 스냅샷에서 {subject_txt} 기준 {facts_txt}입니다. 따라서 "
        f"{rule_txt} 조건을 {'충족합니다' if fires else '충족하지 않습니다'}."
    )
    return {"question": question, "answer": answer, "order": order}


def build(db_path: Path, kg_path: Path) -> tuple[list[dict], dict]:
    rules = rules_by_name()
    missing = set(rules) ^ set(RULE_INPUTS)
    if missing:
        raise ValueError(f"RULE_INPUTS와 규칙 목록 불일치: {sorted(missing)}")

    conn = gf.connect_ro(db_path)
    facts = {cat: gf.category_facts(conn, cat) for cat in gf.CATEGORIES}
    tables = {cat: gf.metric_table_facts(conn, cat) for cat in gf.CATEGORIES}
    triples = gf.load_curated_triples(kg_path)
    ap_brands = set(gf.subjects(triples, "ownedByGroup", "AMOREPACIFIC"))

    accepted: dict[str, list[dict]] = {}
    excluded: list[dict] = []
    for order, cand in enumerate(build_candidates(facts, ap_brands)):
        raw = raw_inputs(facts[cand.category], cand.subject)
        if raw is None:
            excluded.append(
                {
                    "kind": cand.kind,
                    "subject": cand.subject,
                    "category": cand.category,
                    "reason": "raw_data에 대상 없음",
                }
            )
            continue
        brand = raw.get("brand")
        if brand and brand in facts[cand.category].attribution_failures():
            excluded.append(
                {
                    "kind": cand.kind,
                    "subject": cand.subject,
                    "category": cand.category,
                    "reason": f"브랜드 귀속 검증 실패: '{brand}' 행의 제품명에 브랜드명 없음 "
                    f"{facts[cand.category].attribution_failures()[brand][:2]}",
                }
            )
            continue
        tab = table_inputs(tables[cand.category], raw, cand.subject)
        needed = sorted(set(cand.ev_keys.values()) | set(cand.extra_inputs))
        if any(raw.get(k) is None for k in needed):
            excluded.append(
                {
                    "kind": cand.kind,
                    "subject": cand.subject,
                    "category": cand.category,
                    "reason": f"raw 입력 결측 {[k for k in needed if raw.get(k) is None]}",
                }
            )
            continue
        ctx = rule_context(raw, cand.category)
        fires = engine_verdict(rules, cand, ctx)
        if fires != bool(cand.expected(ctx)):
            raise AssertionError(f"{cand.kind}/{cand.subject}: 규칙 엔진 판정과 임계 산술 불일치")

        reasons = []
        tab_missing = [k for k in needed if tab.get(k) is None]
        if tab_missing:
            reasons.append(f"지표 테이블 입력 결측 {tab_missing}")
        else:
            tab_fires = engine_verdict(rules, cand, rule_context(tab, cand.category))
            if tab_fires != fires:
                reasons.append(f"결론 불일치 raw={fires} 지표테이블={tab_fires}")
            for k in needed:
                if not gf.within_tolerance(raw[k], tab[k]):
                    reasons.append(f"{k} 값 불일치 raw={raw[k]} 지표테이블={tab[k]} (>10%)")
        if reasons:
            excluded.append(
                {
                    "kind": cand.kind,
                    "subject": cand.subject,
                    "category": cand.category,
                    "reason": "; ".join(reasons),
                    "raw": {k: raw[k] for k in needed},
                    "metric_tables": {k: tab.get(k) for k in needed},
                }
            )
            continue

        unconverted = None
        if "sos" in needed:
            unconverted = engine_verdict(rules, cand, rule_context(raw, cand.category, False))
        accepted.setdefault(cand.kind, []).append(
            {
                "cand": cand,
                "raw": raw,
                "tab": tab,
                "ctx": ctx,
                "fires": fires,
                "unconverted": unconverted,
                "order": order,
                "needed": needed,
                "attribution_failures": facts[cand.category].attribution_failures(),
            }
        )

    picked: list[dict] = []
    for kind, cap in CAPS.items():
        if kind in accepted:
            chosen = _balanced_pick(accepted[kind], cap)
            picked.extend(chosen)
            for extra in accepted[kind]:
                if extra not in chosen:
                    excluded.append(
                        {
                            "kind": kind,
                            "subject": extra["cand"].subject,
                            "category": extra["cand"].category,
                            "reason": f"종류별 상한 {cap} 초과(유효 후보)",
                        }
                    )

    records: list[dict] = []
    for a in picked:
        records.append(_record(len(records) + 1, a, db_path))
    records.extend(_ownership_records(len(records) + 1, triples, kg_path))

    log = {
        "as_of": gf.AS_OF,
        "generator": GENERATOR,
        "rule_input_availability": {
            name: {
                **spec,
                "generatable": all(s in GENERATABLE for s in spec["inputs"].values()),
            }
            for name, spec in sorted(RULE_INPUTS.items())
        },
        "metric_facts_serves": METRIC_FACTS_SERVES,
        "excluded": excluded,
        "counts": {
            "generated": len(records),
            "by_rule": _count_by_rule(records),
        },
    }
    return records, log


def _count_by_rule(records: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for r in records:
        for rid in r["metadata"]["rule_gold"]["rule_ids"]:
            counts[rid] = counts.get(rid, 0) + 1
    return dict(sorted(counts.items()))


def _record(n: int, a: dict, db_path: Path) -> dict:
    cand: Candidate = a["cand"]
    raw, tab = a["raw"], a["tab"]
    text = materialize(cand, raw, a["fires"], a["order"])
    brand = raw.get("brand")
    mf_brands = [brand] if brand and "sos" in raw else []
    mf = gf.metric_facts_provider_view(db_path, mf_brands, [cand.category])
    served = _metric_facts_values(mf, brand)
    params = {"d": gf.AS_OF, "c": cand.category}
    sql = [
        {"query": gf.SQL_TOTAL, "params": params},
        {"query": gf.SQL_BRAND_COUNTS, "params": params},
        {"query": gf.SQL_CATEGORY_PRICE_RATING, "params": params},
    ]
    if "sos" in raw:
        sql.append({"query": gf.SQL_BRAND_DETAIL, "params": {**params, "b": brand}})
    if cand.subject.startswith("rank:"):
        sql.append({"query": gf.SQL_PRODUCTS, "params": params})

    ev = {key: raw[inp] for key, inp in cand.ev_keys.items()}
    entities = [gf.norm_id(brand)] if brand else []
    entities.append(cand.category)
    rule_gold = {
        "rule_ids": cand.rule_ids,
        "condition": cand.condition,
        "expected_conclusion": {"fires": a["fires"]},
        "inputs": {k: raw[k] for k in a["needed"]},
        "rule_context": {k: v for k, v in a["ctx"].items() if v is not None},
        "thresholds": cand.thresholds,
        "formula": "scripts/golden_snapshot_facts.py 모듈 docstring의 정본 공식",
        "sql": sql,
        "as_of": gf.AS_OF,
        "cross_check": {
            "metric_tables": {k: tab.get(k) for k in a["needed"]},
            "metric_facts_provider": served,
            "metric_facts_not_served": [k for k in a["needed"] if not METRIC_FACTS_SERVES.get(k)],
        },
    }
    if a["unconverted"] is not None:
        rule_gold["conclusion_if_sos_unconverted"] = {"fires": a["unconverted"]}
        rule_gold["scale_note"] = SOS_SCALE_NOTE
    if raw.get("asin"):
        rule_gold["product"] = {"asin": raw["asin"], "brand": raw["brand"], "rank": raw["rank"]}
    failures = a["attribution_failures"]
    rule_gold["category_brand_attribution_failures"] = {
        "rows": sum(len(v) for v in failures.values()),
        "brands": sorted(failures),
        "note": "raw_data.brand 오귀속 행(제품명에 브랜드명 없음). SoS·HHI 정의는 이 필드를 "
        "그대로 쓰므로 정답도 정의를 따른다",
    }
    return gf.build_record(
        item_id=f"rg{n:03d}",
        question=text["question"],
        answer=text["answer"],
        question_type="rule",
        generator=GENERATOR,
        domain=cand.domain,
        difficulty=cand.difficulty,
        requires_kg=cand.requires_kg,
        gold_source=cand.gold_source,
        expected_values=ev,
        kg_entities=entities,
        concepts=cand.concepts,
        extra_metadata={"rule_gold": rule_gold},
    )


def _metric_facts_values(facts: list[dict], brand: str | None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for f in facts:
        if f["type"] == "category_market":
            out["hhi"] = f.get("hhi")
            out["category_avg_price"] = f.get("category_avg_price")
        elif f["type"] == "brand_share" and brand and f["brand"].lower() == brand.lower():
            out["sos"] = f.get("sos")
            out["present"] = f.get("present")
        elif f["type"] == "brand_products" and brand and f["brand"].lower() == brand.lower():
            out["current_rank"] = f["products"][0]["rank"]
    return out


def _ownership_records(start: int, triples: list[dict], kg_path: Path) -> list[dict]:
    """brand_ownership_verification — KG ownedByGroup(config/brands.json 출처)."""
    rules = rules_by_name()
    rule = rules["brand_ownership_verification"]
    brands_cfg = gf.REPO_ROOT / "config" / "brands.json"
    import json

    competitor_names = {
        b["name"] for b in json.loads(brands_cfg.read_text(encoding="utf-8"))["competitor_brands"]
    }
    out: list[dict] = []
    for brand in ("COSRX", "innisfree", "TIRTIR"):
        parents = gf.objects(triples, brand, "ownedByGroup")
        parent = parents[0] if len(parents) == 1 else None
        ctx = {"brand": brand, "parent_group": parent}
        fires = rule.evaluate_conditions(ctx)[0]
        if fires != (parent == "AMOREPACIFIC"):
            raise AssertionError("brand_ownership_verification 판정 불일치")
        if not fires and brand not in competitor_names:
            raise AssertionError(f"{brand}: 소속 부재를 뒷받침할 config 근거 없음")
        evidence = (
            f"KG '{brand} -ownedByGroup-> AMOREPACIFIC' (source=config/brands.json)"
            if fires
            else f"KG에 {brand}의 ownedByGroup 트리플이 없고 config/brands.json "
            "competitor_brands에 경쟁 브랜드로 등록"
        )
        answer = (
            f"{'네' if fires else '아니요'}. {evidence}이므로 brand_ownership_verification 조건"
            f"(parent_group == AMOREPACIFIC)을 {'충족합니다' if fires else '충족하지 않습니다'}."
        )
        rule_gold = {
            "rule_ids": ["brand_ownership_verification"],
            "condition": None,
            "expected_conclusion": {"fires": fires},
            "inputs": {"parent_group": parent},
            "rule_context": {k: v for k, v in ctx.items() if v is not None},
            "thresholds": {"parent_group_equals": "AMOREPACIFIC"},
            "kg_query": {
                "subject": brand,
                "predicate": "ownedByGroup",
                "sources": list(gf.CURATED_KG_SOURCES),
            },
            "as_of": None,
            "as_of_note": "config/brands.json 유래 정적 관계 — 스냅샷 날짜 없음",
            "cross_check": {
                "config_brands_competitor": brand in competitor_names,
                "metric_facts_not_served": ["parent_group"],
            },
        }
        out.append(
            gf.build_record(
                item_id=f"rg{start + len(out):03d}",
                question=(
                    f"{brand}는 아모레퍼시픽 그룹 소속 브랜드인가요? "
                    "규칙 brand_ownership_verification 기준으로 판정해 주세요."
                ),
                answer=answer,
                question_type="rule",
                generator=GENERATOR,
                domain="brand",
                difficulty="easy",
                requires_kg=True,
                gold_source="document",
                kg_entities=[gf.norm_id(brand), "amorepacific"],
                kg_edges=[gf.edge(brand, "ownedByGroup", "AMOREPACIFIC")] if fires else [],
                concepts=["brand_portfolio", "corporate_structure"],
                as_of=None,
                extra_metadata={"rule_gold": rule_gold},
            )
        )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--db", type=Path, default=gf.DEFAULT_DB)
    parser.add_argument("--kg", type=Path, default=gf.DEFAULT_KG)
    parser.add_argument("--out", type=Path, default=gf.TYPED_DIR)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    records, log = build(args.db, args.kg)
    print(f"생성 {len(records)}문항, 제외 후보 {len(log['excluded'])}")
    print(f"규칙별: {log['counts']['by_rule']}")
    return gf.write_or_check(
        {
            args.out / "generated_rule.jsonl": gf.dumps_jsonl(records),
            args.out / "generation_log_rule.json": gf.dumps_json(log),
        },
        args.check,
    )


if __name__ == "__main__":
    sys.exit(main())
