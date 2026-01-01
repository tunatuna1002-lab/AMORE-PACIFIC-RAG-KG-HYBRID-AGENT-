"""
Metric Calculator
10개 전략 지표 계산기

Level 1: Market & Brand (SoS, HHI, Brand Avg Rank)
Level 2: Category & Price (CPI, Churn Rate, Avg Rating Gap)
Level 3: Product & Risk (Rank Volatility, Rank Shock, Streak Days, Rating Trend)
"""

import json
import numpy as np
from datetime import date, datetime, timedelta
from typing import List, Dict, Any, Optional
from collections import defaultdict

from ontology.schema import BrandMetrics, ProductMetrics, MarketMetrics


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
            with open(config_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            return self._default_thresholds()

    def _default_thresholds(self) -> dict:
        """기본 임계값"""
        return {
            "ranking": {
                "top_n_tiers": [3, 5, 10, 20, 50, 100],
                "significant_drop": 5,
                "significant_rise": 10
            },
            "streak": {
                "weekly_highlight": 5,
                "monthly_highlight": 30
            },
            "monitoring": {
                "trend_analysis_window": 7
            }
        }

    # =========================================================================
    # Level 1: Market & Brand 지표
    # =========================================================================

    def calculate_sos(self, records: List[Dict], brand: str, top_n: int = 100) -> float:
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

        brand_count = sum(
            1 for r in top_records
            if r.get("brand", "").lower() == brand.lower()
        )

        return (brand_count / len(top_records)) * 100

    def calculate_hhi(self, records: List[Dict], top_n: int = 100) -> float:
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

        # 브랜드별 카운트
        brand_counts = defaultdict(int)
        for r in top_records:
            brand = r.get("brand", "Unknown")
            brand_counts[brand] += 1

        total = len(top_records)

        # HHI = Σ(SoS_i)^2, SoS는 비율(0-1)로 계산
        hhi = sum((count / total) ** 2 for count in brand_counts.values())

        return round(hhi, 4)

    def calculate_brand_avg_rank(self, records: List[Dict], brand: str) -> Optional[float]:
        """
        Brand Avg Rank 계산

        정의: 브랜드 제품들의 평균 순위

        Args:
            records: 순위 기록 리스트
            brand: 브랜드명

        Returns:
            평균 순위 (낮을수록 상위권)
        """
        brand_records = [
            r for r in records
            if r.get("brand", "").lower() == brand.lower()
        ]

        if not brand_records:
            return None

        ranks = [int(r.get("rank", 0)) for r in brand_records if r.get("rank")]
        if not ranks:
            return None

        return round(sum(ranks) / len(ranks), 2)

    # =========================================================================
    # Level 2: Category & Price 지표
    # =========================================================================

    def calculate_cpi(self, records: List[Dict], brand: str) -> Optional[float]:
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
            and r.get("price") and float(r.get("price", 0)) > 0
        ]

        if not brand_prices:
            return None

        brand_avg = sum(brand_prices) / len(brand_prices)

        return round((brand_avg / category_avg) * 100, 2)

    def calculate_churn_rate(
        self,
        today_records: List[Dict],
        yesterday_records: List[Dict],
        top_n: int = 100
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
        today_asins = {
            r.get("asin")
            for r in today_records
            if int(r.get("rank", 999)) <= top_n
        }

        yesterday_asins = {
            r.get("asin")
            for r in yesterday_records
            if int(r.get("rank", 999)) <= top_n
        }

        if not yesterday_asins:
            return 0.0

        # 신규 진입 + 이탈
        new_entries = today_asins - yesterday_asins
        exits = yesterday_asins - today_asins

        churn = (len(new_entries) + len(exits)) / (2 * top_n)

        return round(churn, 4)

    def calculate_avg_rating_gap(self, records: List[Dict], brand: str) -> Optional[float]:
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
            and r.get("rating") and float(r.get("rating", 0)) > 0
        ]

        if not brand_ratings:
            return None

        brand_avg = sum(brand_ratings) / len(brand_ratings)

        return round(brand_avg - category_avg, 3)

    # =========================================================================
    # Level 3: Product & Risk 지표
    # =========================================================================

    def calculate_rank_volatility(self, rank_history: List[int], window: int = 7) -> Optional[float]:
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
        self,
        today_rank: int,
        yesterday_rank: int,
        threshold: Optional[int] = None
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

    def calculate_streak_days(
        self,
        rank_history: List[Dict],
        asin: str,
        top_n: int = 10
    ) -> int:
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

    def calculate_rating_trend(self, rating_history: List[float], window: int = 7) -> Optional[float]:
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

    def calculate_best_rank(self, rank_history: List[int]) -> Optional[int]:
        """최고 순위 계산"""
        if not rank_history:
            return None
        return min(rank_history)

    def calculate_days_in_top_n(
        self,
        rank_history: List[int],
        top_n_tiers: Optional[List[int]] = None
    ) -> Dict[int, int]:
        """
        Top N별 체류일 계산

        Args:
            rank_history: 순위 히스토리
            top_n_tiers: Top N 기준 리스트

        Returns:
            {3: 5, 5: 10, 10: 20, ...}
        """
        if top_n_tiers is None:
            top_n_tiers = self.thresholds.get("ranking", {}).get("top_n_tiers", [3, 5, 10, 20, 50, 100])

        result = {}
        for n in top_n_tiers:
            days = sum(1 for rank in rank_history if rank <= n)
            result[n] = days

        return result

    # =========================================================================
    # 종합 계산 메서드
    # =========================================================================

    def calculate_brand_metrics(
        self,
        records: List[Dict],
        brand: str,
        category_id: str
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
        brand_records = [
            r for r in records
            if r.get("brand", "").lower() == brand.lower()
        ]

        return BrandMetrics(
            brand=brand,
            category_id=category_id,
            sos=self.calculate_sos(records, brand),
            brand_avg_rank=self.calculate_brand_avg_rank(records, brand),
            product_count=len(brand_records),
            cpi=self.calculate_cpi(records, brand),
            avg_rating_gap=self.calculate_avg_rating_gap(records, brand)
        )

    def calculate_product_metrics(
        self,
        rank_history: List[Dict],
        asin: str,
        category_id: str
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
            rank_shock=self.calculate_rank_shock(today_rank, yesterday_rank) if today_rank and yesterday_rank else False,
            rank_change=self.calculate_rank_change(today_rank, yesterday_rank) if today_rank and yesterday_rank else None,
            streak_days=self.calculate_streak_days(rank_history, asin),
            rating_trend=self.calculate_rating_trend(ratings),
            best_rank=self.calculate_best_rank(ranks),
            days_in_top_n=self.calculate_days_in_top_n(ranks)
        )

    def calculate_market_metrics(
        self,
        today_records: List[Dict],
        yesterday_records: List[Dict],
        category_id: str,
        snapshot_date: date
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
            category_avg_rating=round(category_avg_rating, 2) if category_avg_rating else None
        )
