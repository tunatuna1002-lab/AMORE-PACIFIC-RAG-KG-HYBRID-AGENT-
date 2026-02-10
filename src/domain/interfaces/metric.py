"""
Metric Calculator Protocol
===========================
MetricCalculator에 대한 추상 인터페이스

구현체:
- MetricCalculator (src/tools/metric_calculator.py)
"""

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class MetricCalculatorProtocol(Protocol):
    """
    Metric Calculator Protocol

    10개 전략 지표 계산기 인터페이스 (LANEIGE 경쟁력 분석용).

    지표 계층 구조:
    - Level 1: Market & Brand (SoS, HHI, Brand Avg Rank)
    - Level 2: Category & Price (CPI, Churn Rate, Avg Rating Gap)
    - Level 3: Product & Risk (Rank Volatility, Rank Shock, Streak Days, Rating Trend)

    Methods:
        calculate_sos: Share of Shelf 계산
        calculate_hhi: Herfindahl-Hirschman Index 계산
        calculate_brand_avg_rank: 브랜드 평균 순위 계산
        calculate_cpi: Category Price Index 계산
        calculate_churn_rate: Churn Rate 계산
        calculate_avg_rating_gap: 평균 평점 격차 계산
        calculate_rank_volatility: 순위 변동성 계산
        calculate_rank_shock: 순위 급변 감지
        calculate_rank_change: 순위 변화 계산
        calculate_streak_days: 연속 체류 일수 계산
        calculate_rating_trend: 평점 추세 계산
        calculate_brand_metrics: 브랜드 종합 지표 계산
        calculate_product_metrics: 제품 종합 지표 계산
        calculate_market_metrics: 시장 종합 지표 계산
    """

    def calculate_sos(self, records: list[dict], brand: str, top_n: int = 100) -> float:
        """
        Share of Shelf (SoS)를 계산합니다.

        Top N 내 브랜드 제품 비중(%).

        Args:
            records: 제품 레코드 리스트
            brand: 브랜드명
            top_n: Top N 기준

        Returns:
            SoS 퍼센트 (0~100)
        """
        ...

    def calculate_hhi(self, records: list[dict], top_n: int = 100) -> float:
        """
        Herfindahl-Hirschman Index (HHI)를 계산합니다.

        시장 집중도 (0~1, 높을수록 과점).

        Args:
            records: 제품 레코드 리스트
            top_n: Top N 기준

        Returns:
            HHI 값 (0~1)
        """
        ...

    def calculate_brand_avg_rank(self, records: list[dict], brand: str) -> float | None:
        """
        브랜드 제품 평균 순위를 계산합니다.

        Args:
            records: 제품 레코드 리스트
            brand: 브랜드명

        Returns:
            평균 순위 (낮을수록 좋음)
        """
        ...

    def calculate_cpi(self, records: list[dict], brand: str) -> float | None:
        """
        Category Price Index (CPI)를 계산합니다.

        카테고리 평균가 대비 브랜드 가격 (100 기준).

        Args:
            records: 제품 레코드 리스트
            brand: 브랜드명

        Returns:
            CPI 값 (100 = 평균, >100 = 프리미엄, <100 = 저가)
        """
        ...

    def calculate_churn_rate(
        self,
        today_records: list[dict],
        yesterday_records: list[dict],
        top_n: int = 100,
    ) -> float:
        """
        Churn Rate를 계산합니다.

        전일 대비 Top N 구성원 교체율(%).

        Args:
            today_records: 오늘 레코드
            yesterday_records: 어제 레코드
            top_n: Top N 기준

        Returns:
            Churn Rate (0~100)
        """
        ...

    def calculate_avg_rating_gap(self, records: list[dict], brand: str) -> float | None:
        """
        평균 평점 격차를 계산합니다.

        브랜드 평점 - 카테고리 평균 평점.

        Args:
            records: 제품 레코드 리스트
            brand: 브랜드명

        Returns:
            평점 격차 (>0: 우위, <0: 열위)
        """
        ...

    def calculate_rank_volatility(self, rank_history: list[int], window: int = 7) -> float | None:
        """
        순위 변동성을 계산합니다.

        7일간 순위 표준편차.

        Args:
            rank_history: 순위 히스토리 (최신순)
            window: 윈도우 크기 (기본 7일)

        Returns:
            표준편차 (낮을수록 안정)
        """
        ...

    def calculate_rank_shock(
        self,
        today_rank: int,
        yesterday_rank: int,
        threshold: int = 5,
    ) -> bool:
        """
        순위 급변을 감지합니다.

        전일 대비 순위 급변 여부 (±threshold 이상).

        Args:
            today_rank: 오늘 순위
            yesterday_rank: 어제 순위
            threshold: 급변 기준

        Returns:
            True: 급변, False: 정상
        """
        ...

    def calculate_rank_change(self, today_rank: int, yesterday_rank: int) -> int:
        """
        순위 변화를 계산합니다.

        Args:
            today_rank: 오늘 순위
            yesterday_rank: 어제 순위

        Returns:
            순위 변화 (음수: 하락, 양수: 상승)
        """
        ...

    def calculate_streak_days(self, rank_history: list[dict], asin: str, top_n: int = 10) -> int:
        """
        연속 체류 일수를 계산합니다.

        Top N 내 연속 체류 일수.

        Args:
            rank_history: 순위 히스토리
            asin: 제품 ASIN
            top_n: Top N 기준

        Returns:
            연속 체류 일수
        """
        ...

    def calculate_rating_trend(self, rating_history: list[float], window: int = 7) -> float | None:
        """
        평점 추세를 계산합니다.

        7일간 평점 이동평균 기울기.

        Args:
            rating_history: 평점 히스토리
            window: 윈도우 크기 (기본 7일)

        Returns:
            기울기 (>0: 상승, <0: 하락)
        """
        ...

    def calculate_brand_metrics(
        self,
        records: list[dict],
        brand: str,
        category_id: str,
    ) -> Any:
        """
        브랜드 종합 지표를 계산합니다.

        Args:
            records: 제품 레코드 리스트
            brand: 브랜드명
            category_id: 카테고리 ID

        Returns:
            BrandMetrics 객체
        """
        ...

    def calculate_product_metrics(
        self,
        current_record: dict,
        history: list[dict],
        top_n: int = 100,
    ) -> Any:
        """
        제품 종합 지표를 계산합니다.

        Args:
            current_record: 현재 제품 레코드
            history: 제품 히스토리
            top_n: Top N 기준

        Returns:
            ProductMetrics 객체
        """
        ...

    def calculate_market_metrics(
        self,
        records: list[dict],
        category_id: str,
        snapshot_date: str,
    ) -> Any:
        """
        시장 종합 지표를 계산합니다.

        Args:
            records: 제품 레코드 리스트
            category_id: 카테고리 ID
            snapshot_date: 스냅샷 날짜

        Returns:
            MarketMetrics 객체
        """
        ...
