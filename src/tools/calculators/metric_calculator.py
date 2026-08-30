"""
Metric Calculator
=================
10개 전략 지표 계산기 (LANEIGE 경쟁력 분석용)

## 지표 계층 구조
```
┌─────────────────────────────────────────────────────────────────────────┐
│ Level 1: Market & Brand (시장/브랜드 수준)                              │
├─────────────────────────────────────────────────────────────────────────┤
│ SoS (Share of Shelf)    : Top 100 내 브랜드 제품 비중 (%)               │
│ HHI (Herfindahl Index)  : 시장 집중도 (0~1, 높을수록 과점)              │
│ Brand Avg Rank          : 브랜드 제품 평균 순위                          │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│ Level 2: Category & Price (카테고리/가격 수준)                          │
├─────────────────────────────────────────────────────────────────────────┤
│ CPI (Category Price Index) : 카테고리 평균가 대비 브랜드 가격 (100 기준)│
│ Churn Rate                 : 전일 대비 Top N 구성원 교체율               │
│ Avg Rating Gap             : 브랜드 평점 - 카테고리 평균 평점            │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│ Level 3: Product & Risk (제품/리스크 수준)                              │
├─────────────────────────────────────────────────────────────────────────┤
│ Rank Volatility : 7일간 순위 표준편차 (변동성)                          │
│ Rank Shock      : 전일 대비 순위 급변 여부 (±5 이상)                    │
│ Streak Days     : Top N 내 연속 체류 일수                               │
│ Rating Trend    : 7일간 평점 이동평균 기울기                            │
└─────────────────────────────────────────────────────────────────────────┘
```

## 해석 가이드
| 지표 | 좋음 | 주의 | 경고 |
|------|------|------|------|
| SoS | >10% | 5-10% | <5% |
| HHI | <0.15 (경쟁적) | 0.15-0.25 | >0.25 (집중) |
| CPI | 80-120 (적정) | 120-150 | >150 (과프리미엄) |
| Rating Gap | >0 (우위) | 0 | <0 (열위) |

## 사용 예
```python
calc = MetricCalculator()

# Level 1: 브랜드 지표
sos = calc.calculate_sos(records, "LANEIGE")  # 5.2%
hhi = calc.calculate_hhi(records)              # 0.08

# Level 2: 카테고리 지표
cpi = calc.calculate_cpi(records, "LANEIGE")   # 105

# Level 3: 제품 지표
volatility = calc.calculate_rank_volatility([1,2,3,2,1])  # 0.89

# 종합 지표
brand_metrics = calc.calculate_brand_metrics(records, "LANEIGE", "lip_care")
```

## 임계값 설정
config/thresholds.json 참조
"""

import json
from collections import defaultdict
from collections.abc import Iterable, Mapping
from datetime import date

import numpy as np

from src.domain.entities import BrandMetrics, MarketMetrics, ProductMetrics

# =============================================================================
# HHI 정본 구현 (D1: 정본 스케일 0-1)
# =============================================================================
# 이 모듈의 세 함수가 코드베이스 유일의 HHI 구현이다.
# 0-10000 포인트 스케일이 필요한 소비처(KG, 리포트)는 hhi_to_points()를 쓴다.
# 자체 계산을 다시 만들지 말 것 — tests/unit/tools/test_hhi_canonical.py가 막는다.

UNKNOWN_BRAND_LABELS = {"", "unknown", "n/a", "none"}

# SoS 계산에 필요한 최소 표본 수. 부분 수집일(카테고리당 60개 등)에는
# 실분모로 계산해도 값이 불안정하므로 아예 산출하지 않는다.
SOS_MIN_SAMPLE = 50


def calculate_sos_pct(count: int, total: int, min_sample: int = SOS_MIN_SAMPLE) -> float | None:
    """실분모 기준 SoS(%). 표본이 min_sample 미만이면 None.

    과거 API 구현은 `max(total, 100)`으로 분모에 바닥을 깔아, 부분 수집일에
    SoS를 조용히 과소 계산했다. 분모는 항상 실제 수집 건수를 쓴다.

    Args:
        count: 대상 브랜드 제품 수
        total: 전체 수집 제품 수 (실분모)
        min_sample: SoS를 산출할 최소 표본 수

    Returns:
        0-100 백분율, 표본 미달이면 None
    """
    if total <= 0 or total < min_sample:
        return None
    return round(count / total * 100, 2)


def count_brands(
    records: Iterable[Mapping], *, brand_key: str = "brand", exclude_unknown: bool = True
) -> dict[str, int]:
    """레코드 목록에서 브랜드별 제품 수를 센다.

    Args:
        records: 브랜드 키를 가진 dict 목록
        brand_key: 브랜드명이 담긴 키
        exclude_unknown: Unknown/빈 브랜드를 제외할지 여부

    Returns:
        {브랜드명: 제품 수}
    """
    counts: dict[str, int] = defaultdict(int)
    for record in records:
        brand = (record.get(brand_key) or "").strip()
        if exclude_unknown and brand.lower() in UNKNOWN_BRAND_LABELS:
            continue
        if not brand:
            continue
        counts[brand] += 1
    return dict(counts)


def calculate_hhi_from_counts(brand_counts: Mapping[str, int]) -> float:
    """브랜드별 제품 수로부터 HHI를 계산한다 (정본, 0-1 스케일).

    HHI = Σ(share_i)², share_i = count_i / Σcount
    분모는 반드시 집계된 카운트의 합이다. 제외한 브랜드를 분모에만 남기면
    점유율 합이 1 미만이 되어 HHI가 체계적으로 과소 계산된다.

    Returns:
        0.0 ≤ hhi ≤ 1.0
    """
    total = sum(brand_counts.values())
    if total <= 0:
        return 0.0
    return round(sum((count / total) ** 2 for count in brand_counts.values()), 4)


def hhi_to_points(hhi: float) -> int:
    """0-1 스케일 HHI를 0-10000 포인트 스케일(미국 DOJ 관례)로 변환."""
    return round(hhi * 10000)


class MetricCalculator:
    """전략 지표 계산기"""

    def __init__(self, config: str | dict = "./config/thresholds.json"):
        """
        Args:
            config: 임계값 설정 파일 경로(str) 또는 설정 dict
        """
        if isinstance(config, dict):
            self.config = config
        else:
            self.config = self._load_config(config)
        self.thresholds = self.config

    def _load_config(self, config_path: str) -> dict:
        """설정 파일 로드"""
        try:
            with open(config_path, encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            return self._default_thresholds()

    def _default_thresholds(self) -> dict:
        """기본 임계값"""
        return {
            "ranking": {
                "top_n_tiers": [3, 5, 10, 20, 50, 100],
                "significant_drop": 5,
                "significant_rise": 10,
            },
            "streak": {"weekly_highlight": 5, "monthly_highlight": 30},
            "monitoring": {"trend_analysis_window": 7},
        }

    # =========================================================================
    # Level 1: Market & Brand 지표
    # =========================================================================

    def calculate_sos(self, records: list[dict], brand: str, top_n: int = 100) -> float:
        """
        SoS (Share of Shelf) 계산

        정의: Top N 내에서 특정 브랜드가 차지하는 제품 비중 (%)

        Args:
            records: 순위 기록 리스트
            brand: 브랜드명
            top_n: Top N 기준 (기본 100)

        Returns:
            SoS 백분율 (0-100)
        """
        top_records = [r for r in records if int(r.get("rank", 999)) <= top_n]
        if not top_records:
            return 0.0

        brand_count = sum(1 for r in top_records if r.get("brand", "").lower() == brand.lower())

        return (brand_count / len(top_records)) * 100

    def calculate_hhi(self, records: list[dict], top_n: int = 100) -> float:
        """
        HHI (Herfindahl Index) 계산

        정의: 시장 집중도 지표 (각 브랜드 SoS의 제곱합)
        - 높을수록 집중 시장 (소수 브랜드 지배)
        - 낮을수록 분산 시장 (경쟁 치열)

        Args:
            records: 순위 기록 리스트
            top_n: Top N 기준

        Returns:
            HHI 값 (0-1, 1에 가까울수록 집중)
        """
        top_records = [r for r in records if int(r.get("rank", 999)) <= top_n]
        if not top_records:
            return 0.0

        # Unknown/빈 브랜드는 분자·분모 양쪽에서 제외한다.
        # 분모에만 남기면 점유율 합이 1 미만이 되어 HHI가 과소 계산된다.
        return calculate_hhi_from_counts(count_brands(top_records))

    def calculate_brand_avg_rank(self, records: list[dict], brand: str) -> float | None:
        """
        Brand Avg Rank 계산

        정의: 브랜드 제품들의 평균 순위

        Args:
            records: 순위 기록 리스트
            brand: 브랜드명

        Returns:
            평균 순위 (낮을수록 상위권)
        """
        brand_records = [r for r in records if r.get("brand", "").lower() == brand.lower()]

        if not brand_records:
            return None

        ranks = [int(r.get("rank", 0)) for r in brand_records if r.get("rank")]
        if not ranks:
            return None

        return round(sum(ranks) / len(ranks), 2)

    # =========================================================================
    # Level 2: Category & Price 지표
    # =========================================================================

    def calculate_cpi(self, records: list[dict], brand: str) -> float | None:
        """
        CPI (Category Price Index) 계산

        정의: 카테고리 평균 가격 대비 브랜드 평균 가격 (100 기준)
        - > 100: 프리미엄/고가 포지션
        - < 100: 가성비/저가 포지션

        Args:
            records: 순위 기록 리스트
            brand: 브랜드명

        Returns:
            CPI 값 (100 기준)
        """
        # 카테고리 평균 가격
        all_prices = [
            float(r.get("price", 0))
            for r in records
            if r.get("price") and float(r.get("price", 0)) > 0
        ]

        if not all_prices:
            return None

        category_avg = sum(all_prices) / len(all_prices)

        # 브랜드 평균 가격
        brand_prices = [
            float(r.get("price", 0))
            for r in records
            if r.get("brand", "").lower() == brand.lower()
            and r.get("price")
            and float(r.get("price", 0)) > 0
        ]

        if not brand_prices:
            return None

        brand_avg = sum(brand_prices) / len(brand_prices)

        return round((brand_avg / category_avg) * 100, 2)

    def calculate_churn_rate(
        self, today_records: list[dict], yesterday_records: list[dict], top_n: int = 100
    ) -> float:
        """
        Churn Rate (순위 교체율) 계산

        정의: 전일 대비 Top N 구성원 교체 비율

        Args:
            today_records: 오늘 순위 기록
            yesterday_records: 어제 순위 기록
            top_n: Top N 기준

        Returns:
            교체율 (0-1)
        """
        today_asins = {r.get("asin") for r in today_records if int(r.get("rank", 999)) <= top_n}

        yesterday_asins = {
            r.get("asin") for r in yesterday_records if int(r.get("rank", 999)) <= top_n
        }

        if not yesterday_asins:
            return 0.0

        # 신규 진입 + 이탈
        new_entries = today_asins - yesterday_asins
        exits = yesterday_asins - today_asins

        churn = (len(new_entries) + len(exits)) / (2 * top_n)

        return round(churn, 4)

    def calculate_avg_rating_gap(self, records: list[dict], brand: str) -> float | None:
        """
        Avg Rating Gap 계산

        정의: 브랜드 평균 평점 - 카테고리 평균 평점
        - 양수: 품질 인식 우위
        - 음수: 품질 인식 열위

        Args:
            records: 순위 기록 리스트
            brand: 브랜드명

        Returns:
            평점 격차
        """
        # 카테고리 평균 평점
        all_ratings = [
            float(r.get("rating", 0))
            for r in records
            if r.get("rating") and float(r.get("rating", 0)) > 0
        ]

        if not all_ratings:
            return None

        category_avg = sum(all_ratings) / len(all_ratings)

        # 브랜드 평균 평점
        brand_ratings = [
            float(r.get("rating", 0))
            for r in records
            if r.get("brand", "").lower() == brand.lower()
            and r.get("rating")
            and float(r.get("rating", 0)) > 0
        ]

        if not brand_ratings:
            return None

        brand_avg = sum(brand_ratings) / len(brand_ratings)

        return round(brand_avg - category_avg, 3)

    # =========================================================================
    # Level 3: Product & Risk 지표
    # =========================================================================

    def calculate_rank_volatility(self, rank_history: list[int], window: int = 7) -> float | None:
        """
        Rank Volatility 계산

        정의: 최근 N일간 순위의 표준편차

        Args:
            rank_history: 순위 히스토리 (최신순)
            window: 분석 기간 (기본 7일)

        Returns:
            순위 변동성 (표준편차)
        """
        if len(rank_history) < 2:
            return None

        recent_ranks = rank_history[:window]
        return round(float(np.std(recent_ranks)), 2)

    def calculate_rank_shock(
        self, today_rank: int, yesterday_rank: int, threshold: int | None = None
    ) -> bool:
        """
        Rank Shock 판단

        정의: 전일 대비 순위 급변 여부

        Args:
            today_rank: 오늘 순위
            yesterday_rank: 어제 순위
            threshold: 급변 기준 (기본: config에서 로드)

        Returns:
            급변 발생 여부
        """
        if threshold is None:
            threshold = self.thresholds.get("ranking", {}).get("significant_drop", 5)

        return abs(today_rank - yesterday_rank) >= threshold

    def calculate_rank_change(self, today_rank: int, yesterday_rank: int) -> int:
        """
        Rank Change 계산

        정의: 전일 대비 순위 변화
        - 양수: 하락 (순위 숫자 증가)
        - 음수: 상승 (순위 숫자 감소)

        Args:
            today_rank: 오늘 순위
            yesterday_rank: 어제 순위

        Returns:
            순위 변화량
        """
        return today_rank - yesterday_rank

    def calculate_streak_days(self, rank_history: list[dict], asin: str, top_n: int = 10) -> int:
        """
        Streak Days 계산

        정의: Top N 내 연속 체류 일수

        Args:
            rank_history: 날짜순 정렬된 순위 기록 (최신순)
            asin: 제품 ASIN
            top_n: Top N 기준

        Returns:
            연속 체류 일수
        """
        streak = 0

        for record in rank_history:
            if record.get("asin") != asin:
                continue

            rank = int(record.get("rank", 999))
            if rank <= top_n:
                streak += 1
            else:
                break

        return streak

    def calculate_rating_trend(self, rating_history: list[float], window: int = 7) -> float | None:
        """
        Rating Trend 계산

        정의: 최근 N일간 평점 이동평균의 기울기

        Args:
            rating_history: 평점 히스토리 (최신순)
            window: 분석 기간

        Returns:
            기울기 (양수: 상승, 음수: 하락)
        """
        if len(rating_history) < 2:
            return None

        recent_ratings = rating_history[:window]

        # 선형 회귀로 기울기 계산
        x = np.arange(len(recent_ratings))
        y = np.array(recent_ratings[::-1])  # 오래된 순으로 정렬

        if len(x) < 2:
            return None

        slope = np.polyfit(x, y, 1)[0]
        return round(float(slope), 4)

    def calculate_best_rank(self, rank_history: list[int]) -> int | None:
        """최고 순위 계산"""
        if not rank_history:
            return None
        return min(rank_history)

    def calculate_days_in_top_n(
        self, rank_history: list[int], top_n_tiers: list[int] | None = None
    ) -> dict[int, int]:
        """
        Top N별 체류일 계산

        Args:
            rank_history: 순위 히스토리
            top_n_tiers: Top N 기준 리스트

        Returns:
            {3: 5, 5: 10, 10: 20, ...}
        """
        if top_n_tiers is None:
            top_n_tiers = self.thresholds.get("ranking", {}).get(
                "top_n_tiers", [3, 5, 10, 20, 50, 100]
            )

        result = {}
        for n in top_n_tiers:
            days = sum(1 for rank in rank_history if rank <= n)
            result[n] = days

        return result

    # =========================================================================
    # 종합 계산 메서드
    # =========================================================================

    def calculate_brand_metrics(
        self, records: list[dict], brand: str, category_id: str
    ) -> BrandMetrics:
        """
        브랜드별 전체 지표 계산

        Args:
            records: 순위 기록 리스트
            brand: 브랜드명
            category_id: 카테고리 ID

        Returns:
            BrandMetrics 객체
        """
        brand_records = [r for r in records if r.get("brand", "").lower() == brand.lower()]

        return BrandMetrics(
            brand=brand,
            category_id=category_id,
            sos=self.calculate_sos(records, brand),
            brand_avg_rank=self.calculate_brand_avg_rank(records, brand),
            product_count=len(brand_records),
            cpi=self.calculate_cpi(records, brand),
            avg_rating_gap=self.calculate_avg_rating_gap(records, brand),
        )

    def calculate_product_metrics(
        self, rank_history: list[dict], asin: str, category_id: str
    ) -> ProductMetrics:
        """
        제품별 전체 지표 계산

        Args:
            rank_history: 날짜순 순위 기록 (최신순)
            asin: 제품 ASIN
            category_id: 카테고리 ID

        Returns:
            ProductMetrics 객체
        """
        # ASIN 기준 필터
        product_history = [r for r in rank_history if r.get("asin") == asin]

        if not product_history:
            return ProductMetrics(asin=asin, category_id=category_id)

        ranks = [int(r.get("rank", 0)) for r in product_history if r.get("rank")]
        ratings = [float(r.get("rating", 0)) for r in product_history if r.get("rating")]

        # 오늘/어제 순위
        today_rank = ranks[0] if ranks else None
        yesterday_rank = ranks[1] if len(ranks) > 1 else None

        return ProductMetrics(
            asin=asin,
            category_id=category_id,
            rank_volatility=self.calculate_rank_volatility(ranks),
            rank_shock=self.calculate_rank_shock(today_rank, yesterday_rank)
            if today_rank and yesterday_rank
            else False,
            rank_change=self.calculate_rank_change(today_rank, yesterday_rank)
            if today_rank and yesterday_rank
            else None,
            streak_days=self.calculate_streak_days(rank_history, asin),
            rating_trend=self.calculate_rating_trend(ratings),
            best_rank=self.calculate_best_rank(ranks),
            days_in_top_n=self.calculate_days_in_top_n(ranks),
        )

    def calculate_market_metrics(
        self,
        today_records: list[dict],
        yesterday_records: list[dict],
        category_id: str,
        snapshot_date: date,
    ) -> MarketMetrics:
        """
        시장(카테고리) 전체 지표 계산

        Args:
            today_records: 오늘 순위 기록
            yesterday_records: 어제 순위 기록
            category_id: 카테고리 ID
            snapshot_date: 스냅샷 날짜

        Returns:
            MarketMetrics 객체
        """
        # 카테고리 평균 가격
        prices = [
            float(r.get("price", 0))
            for r in today_records
            if r.get("price") and float(r.get("price", 0)) > 0
        ]
        category_avg_price = sum(prices) / len(prices) if prices else None

        # 카테고리 평균 평점
        ratings = [
            float(r.get("rating", 0))
            for r in today_records
            if r.get("rating") and float(r.get("rating", 0)) > 0
        ]
        category_avg_rating = sum(ratings) / len(ratings) if ratings else None

        return MarketMetrics(
            category_id=category_id,
            snapshot_date=snapshot_date,
            hhi=self.calculate_hhi(today_records),
            churn_rate=self.calculate_churn_rate(today_records, yesterday_records),
            category_avg_price=round(category_avg_price, 2) if category_avg_price else None,
            category_avg_rating=round(category_avg_rating, 2) if category_avg_rating else None,
        )
