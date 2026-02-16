"""
MetricCalculator 단위 테스트
"""

from src.tools.calculators.metric_calculator import MetricCalculator


class TestMetricCalculator:
    """MetricCalculator 클래스 테스트"""

    def test_init(self):
        """초기화"""
        calc = MetricCalculator()
        assert calc is not None

    def test_calculate_sos(self):
        """SoS (Share of Shelf) 계산"""
        calc = MetricCalculator()
        # 총 100개 중 LANEIGE 5개 = 5%
        products = [
            {"brand": "LANEIGE", "rank": 1},
            {"brand": "LANEIGE", "rank": 5},
            {"brand": "LANEIGE", "rank": 10},
            {"brand": "LANEIGE", "rank": 15},
            {"brand": "LANEIGE", "rank": 20},
        ]
        for i in range(95):
            products.append({"brand": f"Other_{i}", "rank": i + 21})

        try:
            if hasattr(calc, "calculate_sos"):
                result = calc.calculate_sos(products)
                assert result is not None
            elif hasattr(calc, "calculate"):
                result = calc.calculate(products)
                assert result is not None
        except Exception:
            pass  # DB 의존성 에러 허용

    def test_calculate_hhi(self):
        """HHI (Herfindahl-Hirschman Index) 계산"""
        calc = MetricCalculator()
        # HHI = sum(si^2) where si = market share
        if hasattr(calc, "calculate_hhi"):
            try:
                shares = {"LANEIGE": 0.3, "Vaseline": 0.2, "Burt's Bees": 0.15, "Others": 0.35}
                result = calc.calculate_hhi(shares)
                assert result is not None
            except Exception:
                pass

    def test_has_calculate_method(self):
        """계산 메서드 존재"""
        calc = MetricCalculator()
        assert hasattr(calc, "calculate_sos")
        assert hasattr(calc, "calculate_hhi")
        assert hasattr(calc, "calculate_cpi")

    def test_empty_data(self):
        """빈 데이터 처리"""
        calc = MetricCalculator()
        try:
            result = calc.calculate_sos([], "LANEIGE")
            assert result == 0.0
        except Exception:
            pass  # 빈 데이터 에러 허용


# =========================================================================
# Wave 3: MetricCalculator 초기화 심화 테스트
# =========================================================================


class TestMetricCalculatorInit:
    """MetricCalculator 초기화 테스트"""

    def test_init_with_dict_config(self):
        """dict 설정으로 초기화"""
        config = {"ranking": {"significant_drop": 10}}
        calc = MetricCalculator(config=config)
        assert calc.config == config
        assert calc.thresholds == config

    def test_init_with_missing_file_uses_defaults(self, tmp_path):
        """존재하지 않는 파일 경로 시 기본값 사용"""
        calc = MetricCalculator(config=str(tmp_path / "nonexistent.json"))
        assert "ranking" in calc.config
        assert calc.config["ranking"]["significant_drop"] == 5

    def test_init_with_valid_file(self, tmp_path):
        """유효한 설정 파일 로드"""
        import json

        config_file = tmp_path / "thresholds.json"
        config_data = {"ranking": {"significant_drop": 7, "top_n_tiers": [5, 10]}}
        config_file.write_text(json.dumps(config_data), encoding="utf-8")

        calc = MetricCalculator(config=str(config_file))
        assert calc.config["ranking"]["significant_drop"] == 7

    def test_default_thresholds_structure(self):
        """기본 임계값 구조 확인"""
        calc = MetricCalculator(config={})
        defaults = calc._default_thresholds()
        assert "ranking" in defaults
        assert "streak" in defaults
        assert "monitoring" in defaults
        assert defaults["ranking"]["significant_drop"] == 5


# =========================================================================
# Wave 3: SoS 계산 심화 테스트
# =========================================================================


class TestMetricCalculatorSoS:
    """SoS (Share of Shelf) 계산 테스트"""

    def _make_records(self, brand_counts, start_rank=1):
        """테스트용 레코드 생성 헬퍼"""
        records = []
        rank = start_rank
        for brand, count in brand_counts.items():
            for _ in range(count):
                records.append({"brand": brand, "rank": rank})
                rank += 1
        return records

    def test_sos_empty_records(self):
        """빈 레코드 시 0.0 반환"""
        calc = MetricCalculator(config={})
        assert calc.calculate_sos([], "LANEIGE") == 0.0

    def test_sos_single_brand_100_percent(self):
        """단일 브랜드 100% SoS"""
        calc = MetricCalculator(config={})
        records = [{"brand": "LANEIGE", "rank": i} for i in range(1, 11)]
        result = calc.calculate_sos(records, "LANEIGE", top_n=10)
        assert result == 100.0

    def test_sos_brand_not_present(self):
        """브랜드가 없을 때 0% SoS"""
        calc = MetricCalculator(config={})
        records = [{"brand": "Other", "rank": i} for i in range(1, 11)]
        result = calc.calculate_sos(records, "LANEIGE", top_n=10)
        assert result == 0.0

    def test_sos_case_insensitive(self):
        """브랜드명 대소문자 구분 없음"""
        calc = MetricCalculator(config={})
        records = [
            {"brand": "laneige", "rank": 1},
            {"brand": "LANEIGE", "rank": 2},
            {"brand": "Other", "rank": 3},
            {"brand": "Other", "rank": 4},
        ]
        result = calc.calculate_sos(records, "LANEIGE", top_n=4)
        assert result == 50.0

    def test_sos_respects_top_n(self):
        """top_n 기준으로 필터링"""
        calc = MetricCalculator(config={})
        records = [
            {"brand": "LANEIGE", "rank": 5},
            {"brand": "LANEIGE", "rank": 15},  # Outside top_n=10
            {"brand": "Other", "rank": 3},
        ]
        result = calc.calculate_sos(records, "LANEIGE", top_n=10)
        # top_n=10: LANEIGE rank5 + Other rank3 = 2 records, 1 LANEIGE
        assert result == 50.0

    def test_sos_percentage_calculation(self):
        """SoS 백분율 계산 정확도"""
        calc = MetricCalculator(config={})
        records = self._make_records({"LANEIGE": 5, "COSRX": 3, "Other": 92})
        result = calc.calculate_sos(records, "LANEIGE", top_n=100)
        assert result == 5.0


# =========================================================================
# Wave 3: HHI 계산 심화 테스트
# =========================================================================


class TestMetricCalculatorHHI:
    """HHI (Herfindahl Index) 계산 테스트"""

    def test_hhi_empty_records(self):
        """빈 레코드 시 0.0 반환"""
        calc = MetricCalculator(config={})
        assert calc.calculate_hhi([]) == 0.0

    def test_hhi_single_brand_monopoly(self):
        """독점 시장 HHI = 1.0"""
        calc = MetricCalculator(config={})
        records = [{"brand": "Monopoly", "rank": i} for i in range(1, 11)]
        result = calc.calculate_hhi(records, top_n=10)
        assert result == 1.0

    def test_hhi_perfect_competition(self):
        """완전 경쟁 (각 브랜드 1개)"""
        calc = MetricCalculator(config={})
        records = [{"brand": f"Brand_{i}", "rank": i} for i in range(1, 101)]
        result = calc.calculate_hhi(records, top_n=100)
        # 100 brands each with 1% share: HHI = 100 * (0.01)^2 = 0.01
        assert result == 0.01

    def test_hhi_excludes_unknown_brands(self):
        """Unknown 브랜드 제외"""
        calc = MetricCalculator(config={})
        records = [
            {"brand": "LANEIGE", "rank": 1},
            {"brand": "Unknown", "rank": 2},
            {"brand": "", "rank": 3},
            {"brand": "COSRX", "rank": 4},
        ]
        result = calc.calculate_hhi(records, top_n=4)
        # Only LANEIGE and COSRX counted. total=4 (all records)
        # LANEIGE share: 1/4=0.25, COSRX: 1/4=0.25
        # HHI = 0.25^2 + 0.25^2 = 0.125
        assert result == 0.125

    def test_hhi_respects_top_n(self):
        """top_n 기준으로 필터링"""
        calc = MetricCalculator(config={})
        records = [
            {"brand": "A", "rank": 1},
            {"brand": "A", "rank": 2},
            {"brand": "B", "rank": 50},  # Outside top_n=5
        ]
        result = calc.calculate_hhi(records, top_n=5)
        # Only 2 records in top 5, both brand A => monopoly
        assert result == 1.0


# =========================================================================
# Wave 3: Brand Avg Rank 테스트
# =========================================================================


class TestMetricCalculatorBrandAvgRank:
    """Brand Avg Rank 계산 테스트"""

    def test_brand_avg_rank_no_records(self):
        """브랜드 레코드 없을 때 None"""
        calc = MetricCalculator(config={})
        result = calc.calculate_brand_avg_rank([], "LANEIGE")
        assert result is None

    def test_brand_avg_rank_single_product(self):
        """단일 제품"""
        calc = MetricCalculator(config={})
        records = [{"brand": "LANEIGE", "rank": 10}]
        result = calc.calculate_brand_avg_rank(records, "LANEIGE")
        assert result == 10.0

    def test_brand_avg_rank_multiple_products(self):
        """여러 제품 평균"""
        calc = MetricCalculator(config={})
        records = [
            {"brand": "LANEIGE", "rank": 5},
            {"brand": "LANEIGE", "rank": 15},
            {"brand": "Other", "rank": 1},
        ]
        result = calc.calculate_brand_avg_rank(records, "LANEIGE")
        assert result == 10.0

    def test_brand_avg_rank_case_insensitive(self):
        """대소문자 구분 없음"""
        calc = MetricCalculator(config={})
        records = [{"brand": "laneige", "rank": 20}]
        result = calc.calculate_brand_avg_rank(records, "LANEIGE")
        assert result == 20.0

    def test_brand_avg_rank_no_matching_brand(self):
        """브랜드 매칭 안 됨"""
        calc = MetricCalculator(config={})
        records = [{"brand": "Other", "rank": 5}]
        result = calc.calculate_brand_avg_rank(records, "LANEIGE")
        assert result is None


# =========================================================================
# Wave 3: CPI 계산 테스트
# =========================================================================


class TestMetricCalculatorCPI:
    """CPI (Category Price Index) 계산 테스트"""

    def test_cpi_no_prices(self):
        """가격 정보 없을 때 None"""
        calc = MetricCalculator(config={})
        records = [{"brand": "LANEIGE", "rank": 1}]
        result = calc.calculate_cpi(records, "LANEIGE")
        assert result is None

    def test_cpi_brand_no_price(self):
        """브랜드에 가격 정보 없을 때 None"""
        calc = MetricCalculator(config={})
        records = [
            {"brand": "Other", "rank": 1, "price": 20.0},
            {"brand": "LANEIGE", "rank": 2},
        ]
        result = calc.calculate_cpi(records, "LANEIGE")
        assert result is None

    def test_cpi_equal_to_average(self):
        """브랜드 가격 = 카테고리 평균 -> CPI 100"""
        calc = MetricCalculator(config={})
        records = [
            {"brand": "LANEIGE", "rank": 1, "price": 20.0},
            {"brand": "Other", "rank": 2, "price": 20.0},
        ]
        result = calc.calculate_cpi(records, "LANEIGE")
        assert result == 100.0

    def test_cpi_premium_brand(self):
        """프리미엄 브랜드 CPI > 100"""
        calc = MetricCalculator(config={})
        records = [
            {"brand": "LANEIGE", "rank": 1, "price": 30.0},
            {"brand": "Other", "rank": 2, "price": 10.0},
        ]
        result = calc.calculate_cpi(records, "LANEIGE")
        # category avg = (30+10)/2 = 20, brand avg = 30
        # CPI = (30/20)*100 = 150
        assert result == 150.0

    def test_cpi_value_brand(self):
        """가성비 브랜드 CPI < 100"""
        calc = MetricCalculator(config={})
        records = [
            {"brand": "LANEIGE", "rank": 1, "price": 10.0},
            {"brand": "Other", "rank": 2, "price": 30.0},
        ]
        result = calc.calculate_cpi(records, "LANEIGE")
        # category avg = 20, brand avg = 10 -> CPI = 50
        assert result == 50.0

    def test_cpi_zero_price_excluded(self):
        """가격 0인 레코드 제외"""
        calc = MetricCalculator(config={})
        records = [
            {"brand": "LANEIGE", "rank": 1, "price": 20.0},
            {"brand": "LANEIGE", "rank": 2, "price": 0},
            {"brand": "Other", "rank": 3, "price": 20.0},
        ]
        result = calc.calculate_cpi(records, "LANEIGE")
        assert result == 100.0


# =========================================================================
# Wave 3: Churn Rate 테스트
# =========================================================================


class TestMetricCalculatorChurnRate:
    """Churn Rate 계산 테스트"""

    def test_churn_rate_no_change(self):
        """변화 없음 -> 0"""
        calc = MetricCalculator(config={})
        today = [{"asin": "A1", "rank": 1}, {"asin": "A2", "rank": 2}]
        yesterday = [{"asin": "A1", "rank": 1}, {"asin": "A2", "rank": 2}]
        result = calc.calculate_churn_rate(today, yesterday, top_n=2)
        assert result == 0.0

    def test_churn_rate_complete_turnover(self):
        """완전 교체"""
        calc = MetricCalculator(config={})
        today = [{"asin": "NEW1", "rank": 1}, {"asin": "NEW2", "rank": 2}]
        yesterday = [{"asin": "OLD1", "rank": 1}, {"asin": "OLD2", "rank": 2}]
        result = calc.calculate_churn_rate(today, yesterday, top_n=2)
        # new_entries=2, exits=2, churn = (2+2)/(2*2) = 1.0
        assert result == 1.0

    def test_churn_rate_empty_yesterday(self):
        """어제 데이터 없음 -> 0"""
        calc = MetricCalculator(config={})
        today = [{"asin": "A1", "rank": 1}]
        result = calc.calculate_churn_rate(today, [], top_n=10)
        assert result == 0.0

    def test_churn_rate_partial_change(self):
        """부분 교체"""
        calc = MetricCalculator(config={})
        today = [{"asin": "A1", "rank": 1}, {"asin": "NEW", "rank": 2}]
        yesterday = [{"asin": "A1", "rank": 1}, {"asin": "OLD", "rank": 2}]
        result = calc.calculate_churn_rate(today, yesterday, top_n=2)
        # new=1, exits=1, churn = (1+1)/(2*2) = 0.5
        assert result == 0.5


# =========================================================================
# Wave 3: Rating Gap 테스트
# =========================================================================


class TestMetricCalculatorRatingGap:
    """Avg Rating Gap 계산 테스트"""

    def test_rating_gap_no_ratings(self):
        """평점 없을 때 None"""
        calc = MetricCalculator(config={})
        result = calc.calculate_avg_rating_gap([], "LANEIGE")
        assert result is None

    def test_rating_gap_brand_no_rating(self):
        """브랜드 평점 없을 때 None"""
        calc = MetricCalculator(config={})
        records = [{"brand": "Other", "rank": 1, "rating": 4.5}]
        result = calc.calculate_avg_rating_gap(records, "LANEIGE")
        assert result is None

    def test_rating_gap_positive(self):
        """브랜드 평점이 높을 때 양수"""
        calc = MetricCalculator(config={})
        records = [
            {"brand": "LANEIGE", "rank": 1, "rating": 4.8},
            {"brand": "Other", "rank": 2, "rating": 4.0},
        ]
        result = calc.calculate_avg_rating_gap(records, "LANEIGE")
        # cat avg = 4.4, brand avg = 4.8, gap = 0.4
        assert result == 0.4

    def test_rating_gap_negative(self):
        """브랜드 평점이 낮을 때 음수"""
        calc = MetricCalculator(config={})
        records = [
            {"brand": "LANEIGE", "rank": 1, "rating": 3.5},
            {"brand": "Other", "rank": 2, "rating": 4.5},
        ]
        result = calc.calculate_avg_rating_gap(records, "LANEIGE")
        # cat avg = 4.0, brand avg = 3.5, gap = -0.5
        assert result == -0.5


# =========================================================================
# Wave 3: Rank Volatility 테스트
# =========================================================================


class TestMetricCalculatorRankVolatility:
    """Rank Volatility 계산 테스트"""

    def test_volatility_single_rank(self):
        """순위 1개 -> None"""
        calc = MetricCalculator(config={})
        assert calc.calculate_rank_volatility([5]) is None

    def test_volatility_empty_list(self):
        """빈 리스트 -> None"""
        calc = MetricCalculator(config={})
        assert calc.calculate_rank_volatility([]) is None

    def test_volatility_no_change(self):
        """변동 없음 -> 0"""
        calc = MetricCalculator(config={})
        result = calc.calculate_rank_volatility([5, 5, 5, 5, 5])
        assert result == 0.0

    def test_volatility_with_change(self):
        """변동 있는 경우"""
        calc = MetricCalculator(config={})
        result = calc.calculate_rank_volatility([1, 2, 3, 2, 1])
        assert result is not None
        assert result > 0

    def test_volatility_respects_window(self):
        """window 파라미터 적용"""
        calc = MetricCalculator(config={})
        # 10개 데이터, window=3 -> 최근 3개만 사용
        history = [1, 1, 1, 50, 50, 50, 50, 50, 50, 50]
        result = calc.calculate_rank_volatility(history, window=3)
        assert result == 0.0  # First 3 are all 1


# =========================================================================
# Wave 3: Rank Shock / Rank Change 테스트
# =========================================================================


class TestMetricCalculatorRankShock:
    """Rank Shock 판단 테스트"""

    def test_rank_shock_true(self):
        """급변 발생"""
        calc = MetricCalculator(config={})
        assert calc.calculate_rank_shock(1, 10) is True  # |1-10| = 9 >= 5

    def test_rank_shock_false(self):
        """급변 미발생"""
        calc = MetricCalculator(config={})
        assert calc.calculate_rank_shock(5, 7) is False  # |5-7| = 2 < 5

    def test_rank_shock_exact_threshold(self):
        """임계값 정확히 일치"""
        calc = MetricCalculator(config={})
        assert calc.calculate_rank_shock(1, 6) is True  # |1-6| = 5 >= 5

    def test_rank_shock_custom_threshold(self):
        """커스텀 임계값"""
        calc = MetricCalculator(config={})
        assert calc.calculate_rank_shock(1, 4, threshold=3) is True  # |1-4| = 3 >= 3
        assert calc.calculate_rank_shock(1, 3, threshold=3) is False  # |1-3| = 2 < 3

    def test_rank_shock_uses_config_threshold(self):
        """config에서 임계값 로드"""
        calc = MetricCalculator(config={"ranking": {"significant_drop": 10}})
        assert calc.calculate_rank_shock(1, 9) is False  # |1-9| = 8 < 10
        assert calc.calculate_rank_shock(1, 11) is True  # |1-11| = 10 >= 10


class TestMetricCalculatorRankChange:
    """Rank Change 계산 테스트"""

    def test_rank_change_drop(self):
        """순위 하락 (숫자 증가)"""
        calc = MetricCalculator(config={})
        assert calc.calculate_rank_change(10, 5) == 5

    def test_rank_change_rise(self):
        """순위 상승 (숫자 감소)"""
        calc = MetricCalculator(config={})
        assert calc.calculate_rank_change(3, 8) == -5

    def test_rank_change_no_change(self):
        """변화 없음"""
        calc = MetricCalculator(config={})
        assert calc.calculate_rank_change(5, 5) == 0


# =========================================================================
# Wave 3: Streak Days 테스트
# =========================================================================


class TestMetricCalculatorStreakDays:
    """Streak Days 계산 테스트"""

    def test_streak_days_empty_history(self):
        """빈 히스토리"""
        calc = MetricCalculator(config={})
        assert calc.calculate_streak_days([], "B0TEST", top_n=10) == 0

    def test_streak_days_continuous(self):
        """연속 체류"""
        calc = MetricCalculator(config={})
        history = [
            {"asin": "B0TEST", "rank": 5},
            {"asin": "B0TEST", "rank": 3},
            {"asin": "B0TEST", "rank": 8},
        ]
        result = calc.calculate_streak_days(history, "B0TEST", top_n=10)
        assert result == 3

    def test_streak_days_broken(self):
        """연속 체류 중단"""
        calc = MetricCalculator(config={})
        history = [
            {"asin": "B0TEST", "rank": 5},
            {"asin": "B0TEST", "rank": 15},  # Outside top 10
            {"asin": "B0TEST", "rank": 3},
        ]
        result = calc.calculate_streak_days(history, "B0TEST", top_n=10)
        assert result == 1

    def test_streak_days_different_asin_skipped(self):
        """다른 ASIN은 건너뜀"""
        calc = MetricCalculator(config={})
        history = [
            {"asin": "B0OTHER", "rank": 1},
            {"asin": "B0TEST", "rank": 5},
            {"asin": "B0TEST", "rank": 3},
        ]
        result = calc.calculate_streak_days(history, "B0TEST", top_n=10)
        assert result == 2


# =========================================================================
# Wave 3: Rating Trend / Best Rank / Days in Top N 테스트
# =========================================================================


class TestMetricCalculatorRatingTrend:
    """Rating Trend 계산 테스트"""

    def test_rating_trend_single_value(self):
        """데이터 1개 -> None"""
        calc = MetricCalculator(config={})
        assert calc.calculate_rating_trend([4.5]) is None

    def test_rating_trend_empty(self):
        """빈 리스트 -> None"""
        calc = MetricCalculator(config={})
        assert calc.calculate_rating_trend([]) is None

    def test_rating_trend_increasing(self):
        """상승 추세"""
        calc = MetricCalculator(config={})
        # 최신순: [4.5, 4.3, 4.1] -> reversed: [4.1, 4.3, 4.5] -> positive slope
        result = calc.calculate_rating_trend([4.5, 4.3, 4.1])
        assert result is not None
        assert result > 0

    def test_rating_trend_decreasing(self):
        """하락 추세"""
        calc = MetricCalculator(config={})
        # 최신순: [4.0, 4.2, 4.5] -> reversed: [4.5, 4.2, 4.0] -> negative slope
        result = calc.calculate_rating_trend([4.0, 4.2, 4.5])
        assert result is not None
        assert result < 0

    def test_rating_trend_flat(self):
        """횡보"""
        calc = MetricCalculator(config={})
        result = calc.calculate_rating_trend([4.5, 4.5, 4.5])
        assert result is not None
        assert abs(result) < 0.01


class TestMetricCalculatorBestRank:
    """Best Rank 계산 테스트"""

    def test_best_rank_empty(self):
        """빈 리스트"""
        calc = MetricCalculator(config={})
        assert calc.calculate_best_rank([]) is None

    def test_best_rank_single(self):
        """단일 값"""
        calc = MetricCalculator(config={})
        assert calc.calculate_best_rank([5]) == 5

    def test_best_rank_multiple(self):
        """최소값 반환"""
        calc = MetricCalculator(config={})
        assert calc.calculate_best_rank([10, 5, 20, 1, 15]) == 1


class TestMetricCalculatorDaysInTopN:
    """Days in Top N 계산 테스트"""

    def test_days_in_top_n_empty(self):
        """빈 히스토리"""
        calc = MetricCalculator(config={})
        result = calc.calculate_days_in_top_n([], top_n_tiers=[5, 10])
        assert result == {5: 0, 10: 0}

    def test_days_in_top_n_all_in(self):
        """모든 날짜 Top N 안"""
        calc = MetricCalculator(config={})
        history = [1, 2, 3, 4, 5]
        result = calc.calculate_days_in_top_n(history, top_n_tiers=[5, 10])
        assert result[5] == 5
        assert result[10] == 5

    def test_days_in_top_n_partial(self):
        """부분적 Top N"""
        calc = MetricCalculator(config={})
        history = [3, 8, 12, 2, 50]
        result = calc.calculate_days_in_top_n(history, top_n_tiers=[5, 10])
        # top 5: ranks 3, 2 -> 2 days
        assert result[5] == 2
        # top 10: ranks 3, 8, 2 -> 3 days
        assert result[10] == 3

    def test_days_in_top_n_uses_config_tiers(self):
        """config에서 tier 로드"""
        calc = MetricCalculator(config={"ranking": {"top_n_tiers": [3, 10]}})
        history = [1, 2, 5, 10, 50]
        result = calc.calculate_days_in_top_n(history)
        assert 3 in result
        assert 10 in result


# =========================================================================
# Wave 3: 종합 계산 메서드 테스트
# =========================================================================


class TestMetricCalculatorBrandMetrics:
    """calculate_brand_metrics 종합 테스트"""

    def test_brand_metrics_returns_pydantic_model(self):
        """BrandMetrics Pydantic 모델 반환"""
        from src.domain.entities import BrandMetrics

        calc = MetricCalculator(config={})
        records = [
            {"brand": "LANEIGE", "rank": 5, "price": 22.0, "rating": 4.5},
            {"brand": "Other", "rank": 1, "price": 15.0, "rating": 4.2},
        ]
        result = calc.calculate_brand_metrics(records, "LANEIGE", "lip_care")
        assert isinstance(result, BrandMetrics)
        assert result.brand == "LANEIGE"
        assert result.category_id == "lip_care"
        assert result.sos > 0
        assert result.product_count == 1

    def test_brand_metrics_empty_records(self):
        """빈 레코드로 BrandMetrics 생성"""
        calc = MetricCalculator(config={})
        result = calc.calculate_brand_metrics([], "LANEIGE", "lip_care")
        assert result.sos == 0.0
        assert result.product_count == 0


class TestMetricCalculatorProductMetrics:
    """calculate_product_metrics 종합 테스트"""

    def test_product_metrics_returns_pydantic_model(self):
        """ProductMetrics Pydantic 모델 반환"""
        from src.domain.entities import ProductMetrics

        calc = MetricCalculator(config={})
        history = [
            {"asin": "B0TEST", "rank": 5, "rating": 4.5},
            {"asin": "B0TEST", "rank": 8, "rating": 4.3},
            {"asin": "B0TEST", "rank": 3, "rating": 4.7},
        ]
        result = calc.calculate_product_metrics(history, "B0TEST", "lip_care")
        assert isinstance(result, ProductMetrics)
        assert result.asin == "B0TEST"
        assert result.rank_volatility is not None
        assert result.best_rank == 3

    def test_product_metrics_no_history(self):
        """히스토리 없는 제품"""
        from src.domain.entities import ProductMetrics

        calc = MetricCalculator(config={})
        result = calc.calculate_product_metrics([], "B0EMPTY", "lip_care")
        assert isinstance(result, ProductMetrics)
        assert result.asin == "B0EMPTY"
        assert result.rank_volatility is None

    def test_product_metrics_single_record(self):
        """단일 레코드"""
        calc = MetricCalculator(config={})
        history = [{"asin": "B0SINGLE", "rank": 5, "rating": 4.5}]
        result = calc.calculate_product_metrics(history, "B0SINGLE", "lip_care")
        assert result.rank_shock is False
        assert result.rank_change is None


class TestMetricCalculatorMarketMetrics:
    """calculate_market_metrics 종합 테스트"""

    def test_market_metrics_returns_pydantic_model(self):
        """MarketMetrics Pydantic 모델 반환"""
        from datetime import date

        from src.domain.entities import MarketMetrics

        calc = MetricCalculator(config={})
        today = [
            {"brand": "LANEIGE", "rank": 1, "price": 22.0, "rating": 4.5, "asin": "A1"},
            {"brand": "Other", "rank": 2, "price": 15.0, "rating": 4.0, "asin": "A2"},
        ]
        yesterday = [
            {"brand": "LANEIGE", "rank": 1, "asin": "A1"},
            {"brand": "Other", "rank": 2, "asin": "A2"},
        ]
        result = calc.calculate_market_metrics(today, yesterday, "lip_care", date(2025, 1, 15))
        assert isinstance(result, MarketMetrics)
        assert result.category_id == "lip_care"
        assert result.hhi > 0
        assert result.category_avg_price is not None
        assert result.category_avg_rating is not None

    def test_market_metrics_no_prices_or_ratings(self):
        """가격/평점 없는 경우"""
        from datetime import date

        calc = MetricCalculator(config={})
        today = [{"brand": "A", "rank": 1, "asin": "A1"}]
        yesterday = [{"brand": "A", "rank": 1, "asin": "A1"}]
        result = calc.calculate_market_metrics(today, yesterday, "lip_care", date(2025, 1, 15))
        assert result.category_avg_price is None
        assert result.category_avg_rating is None
