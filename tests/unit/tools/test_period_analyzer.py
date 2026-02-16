"""
PeriodAnalyzer 단위 테스트
==========================
기간별 데이터 분석기 테스트
SQLite 의존성을 mock하여 순수 분석 로직을 검증합니다.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.tools.calculators.period_analyzer import PeriodAnalysis, PeriodAnalyzer

# =========================================================================
# Fixtures
# =========================================================================


@pytest.fixture
def analyzer():
    """mock storage를 가진 PeriodAnalyzer"""
    mock_storage = AsyncMock()
    return PeriodAnalyzer(sqlite_storage=mock_storage)


@pytest.fixture
def sample_daily_data():
    """테스트용 일별 데이터 (2일치)"""
    return {
        "2025-01-15": [
            {
                "asin": "B0A",
                "title": "LANEIGE Lip Sleeping Mask",
                "brand": "LANEIGE",
                "rank": 5,
                "category": "lip_care",
                "price": 22.0,
                "rating": 4.5,
                "reviews_count": 1000,
            },
            {
                "asin": "B0B",
                "title": "Vaseline Lip Therapy",
                "brand": "Vaseline",
                "rank": 1,
                "category": "lip_care",
                "price": 3.0,
                "rating": 4.3,
                "reviews_count": 5000,
            },
            {
                "asin": "B0C",
                "title": "COSRX Snail Mucin",
                "brand": "COSRX",
                "rank": 10,
                "category": "skin_care",
                "price": 15.0,
                "rating": 4.7,
                "reviews_count": 3000,
            },
        ],
        "2025-01-16": [
            {
                "asin": "B0A",
                "title": "LANEIGE Lip Sleeping Mask",
                "brand": "LANEIGE",
                "rank": 3,
                "category": "lip_care",
                "price": 22.0,
                "rating": 4.5,
                "reviews_count": 1010,
            },
            {
                "asin": "B0B",
                "title": "Vaseline Lip Therapy",
                "brand": "Vaseline",
                "rank": 2,
                "category": "lip_care",
                "price": 3.0,
                "rating": 4.3,
                "reviews_count": 5010,
            },
            {
                "asin": "B0D",
                "title": "e.l.f. Lip Gloss",
                "brand": "e.l.f.",
                "rank": 8,
                "category": "lip_makeup",
                "price": 6.0,
                "rating": 4.1,
                "reviews_count": 2000,
            },
        ],
    }


# =========================================================================
# PeriodAnalysis 데이터 클래스 테스트
# =========================================================================


class TestPeriodAnalysis:
    """PeriodAnalysis 데이터 클래스 테스트"""

    def test_period_analysis_default_fields(self):
        """기본 필드 초기값"""
        pa = PeriodAnalysis(start_date="2025-01-01", end_date="2025-01-07", total_days=7)
        assert pa.start_date == "2025-01-01"
        assert pa.end_date == "2025-01-07"
        assert pa.total_days == 7
        assert pa.laneige_metrics == {}
        assert pa.market_metrics == {}
        assert pa.brand_performance == []
        assert pa.category_analysis == {}
        assert pa.top_movers == {}
        assert pa.competitive_shifts == {}
        assert pa.daily_trends == []

    def test_period_analysis_custom_fields(self):
        """커스텀 필드 설정"""
        pa = PeriodAnalysis(
            start_date="2025-01-01",
            end_date="2025-01-07",
            total_days=7,
            laneige_metrics={"avg_sos": 5.0},
            market_metrics={"avg_hhi": 1000},
        )
        assert pa.laneige_metrics["avg_sos"] == 5.0
        assert pa.market_metrics["avg_hhi"] == 1000


# =========================================================================
# PeriodAnalyzer 초기화 테스트
# =========================================================================


class TestPeriodAnalyzerInit:
    """PeriodAnalyzer 초기화 테스트"""

    def test_init_with_storage(self):
        """storage 전달"""
        mock_storage = MagicMock()
        pa = PeriodAnalyzer(sqlite_storage=mock_storage)
        assert pa.storage is mock_storage

    def test_init_without_storage(self):
        """storage=None"""
        pa = PeriodAnalyzer()
        assert pa.storage is None

    def test_categories_list(self):
        """기본 카테고리 목록"""
        pa = PeriodAnalyzer()
        assert "lip_care" in pa._categories
        assert "skin_care" in pa._categories
        assert len(pa._categories) == 5


# =========================================================================
# analyze 메서드 테스트
# =========================================================================


class TestPeriodAnalyzerAnalyze:
    """PeriodAnalyzer.analyze 테스트"""

    @pytest.mark.asyncio
    async def test_analyze_no_data(self, analyzer):
        """데이터 없을 때 빈 결과"""
        analyzer.storage.get_raw_data = AsyncMock(return_value=[])
        result = await analyzer.analyze("2025-01-01", "2025-01-07")

        assert isinstance(result, PeriodAnalysis)
        assert result.total_days == 0

    @pytest.mark.asyncio
    async def test_analyze_with_data(self, analyzer):
        """데이터가 있을 때 전체 분석"""
        raw_data = [
            {
                "asin": "B0A",
                "product_name": "LANEIGE Mask",
                "brand": "LANEIGE",
                "rank": 5,
                "category_id": "lip_care",
                "snapshot_date": "2025-01-15",
                "price": 22.0,
                "rating": 4.5,
            },
            {
                "asin": "B0B",
                "product_name": "Vaseline",
                "brand": "Vaseline",
                "rank": 1,
                "category_id": "lip_care",
                "snapshot_date": "2025-01-15",
                "price": 3.0,
                "rating": 4.3,
            },
            {
                "asin": "B0A",
                "product_name": "LANEIGE Mask",
                "brand": "LANEIGE",
                "rank": 3,
                "category_id": "lip_care",
                "snapshot_date": "2025-01-16",
                "price": 22.0,
                "rating": 4.5,
            },
            {
                "asin": "B0B",
                "product_name": "Vaseline",
                "brand": "Vaseline",
                "rank": 2,
                "category_id": "lip_care",
                "snapshot_date": "2025-01-16",
                "price": 3.0,
                "rating": 4.3,
            },
        ]
        analyzer.storage.get_raw_data = AsyncMock(return_value=raw_data)
        result = await analyzer.analyze("2025-01-15", "2025-01-16")

        assert result.total_days == 2
        assert result.laneige_metrics != {}
        assert result.market_metrics != {}
        assert len(result.brand_performance) > 0
        assert len(result.daily_trends) == 2

    @pytest.mark.asyncio
    async def test_analyze_auto_creates_storage(self):
        """storage=None일 때 자동 생성"""
        pa = PeriodAnalyzer(sqlite_storage=None)
        with patch("src.tools.storage.sqlite_storage.SQLiteStorage") as MockStorage:
            mock_instance = AsyncMock()
            mock_instance.get_raw_data = AsyncMock(return_value=[])
            MockStorage.return_value = mock_instance

            result = await pa.analyze("2025-01-01", "2025-01-07")
            assert result.total_days == 0
            mock_instance.initialize.assert_called_once()


# =========================================================================
# _normalize_brand 테스트
# =========================================================================


class TestPeriodAnalyzerNormalizeBrand:
    """_normalize_brand 테스트"""

    def test_normalize_known_brand(self, analyzer):
        """정규화 매핑에 있는 브랜드"""
        assert analyzer._normalize_brand("burt's") == "Burt's Bees"
        assert analyzer._normalize_brand("wet") == "wet n wild"
        assert analyzer._normalize_brand("rare") == "Rare Beauty"

    def test_normalize_unknown_brand_passthrough(self, analyzer):
        """매핑에 없는 브랜드는 그대로 반환"""
        assert analyzer._normalize_brand("LANEIGE") == "LANEIGE"
        assert analyzer._normalize_brand("SomeNewBrand") == "SomeNewBrand"

    def test_normalize_empty_or_unknown(self, analyzer):
        """빈 문자열 또는 Unknown은 그대로 반환"""
        assert analyzer._normalize_brand("") == ""
        assert analyzer._normalize_brand("Unknown") == "Unknown"

    def test_normalize_case_insensitive_lookup(self, analyzer):
        """대소문자 구분 없이 매핑"""
        assert analyzer._normalize_brand("Burt's") == "Burt's Bees"
        assert analyzer._normalize_brand("RARE") == "Rare Beauty"

    def test_normalize_from_product_name(self, analyzer):
        """product_name에서 브랜드 추출"""
        result = analyzer._normalize_brand("some_short", "Charlotte Tilbury Matte Lipstick")
        assert result == "Charlotte Tilbury"

    def test_normalize_no_product_name_match(self, analyzer):
        """product_name에도 매칭 안 되면 원래 이름 반환"""
        result = analyzer._normalize_brand("xyz", "Totally Unknown Product")
        assert result == "xyz"


# =========================================================================
# _group_by_date 테스트
# =========================================================================


class TestPeriodAnalyzerGroupByDate:
    """_group_by_date 테스트"""

    def test_group_by_date_basic(self, analyzer):
        """기본 날짜 그룹핑"""
        data = [
            {
                "snapshot_date": "2025-01-15",
                "asin": "A1",
                "brand": "LANEIGE",
                "product_name": "",
                "rank": 5,
                "category_id": "lip_care",
            },
            {
                "snapshot_date": "2025-01-15",
                "asin": "A2",
                "brand": "COSRX",
                "product_name": "",
                "rank": 10,
                "category_id": "skin_care",
            },
            {
                "snapshot_date": "2025-01-16",
                "asin": "A1",
                "brand": "LANEIGE",
                "product_name": "",
                "rank": 3,
                "category_id": "lip_care",
            },
        ]
        result = analyzer._group_by_date(data)
        assert "2025-01-15" in result
        assert "2025-01-16" in result
        assert len(result["2025-01-15"]) == 2
        assert len(result["2025-01-16"]) == 1

    def test_group_by_date_sorts_keys(self, analyzer):
        """날짜 키가 정렬됨"""
        data = [
            {
                "snapshot_date": "2025-01-16",
                "asin": "A1",
                "brand": "B",
                "product_name": "",
                "rank": 1,
                "category_id": "c",
            },
            {
                "snapshot_date": "2025-01-15",
                "asin": "A2",
                "brand": "B",
                "product_name": "",
                "rank": 2,
                "category_id": "c",
            },
        ]
        result = analyzer._group_by_date(data)
        assert list(result.keys()) == ["2025-01-15", "2025-01-16"]

    def test_group_by_date_normalizes_datetime_format(self, analyzer):
        """datetime 형식에서 날짜만 추출"""
        data = [
            {
                "snapshot_date": "2025-01-15T10:30:00",
                "asin": "A1",
                "brand": "B",
                "product_name": "",
                "rank": 1,
                "category_id": "c",
            },
        ]
        result = analyzer._group_by_date(data)
        assert "2025-01-15" in result

    def test_group_by_date_uses_date_fallback(self, analyzer):
        """snapshot_date가 없으면 date 필드 사용"""
        data = [
            {
                "date": "2025-01-15",
                "asin": "A1",
                "brand": "B",
                "product_name": "",
                "rank": 1,
                "category_id": "c",
            },
        ]
        result = analyzer._group_by_date(data)
        assert "2025-01-15" in result

    def test_group_by_date_normalizes_brands(self, analyzer):
        """브랜드명 정규화 적용"""
        data = [
            {
                "snapshot_date": "2025-01-15",
                "asin": "A1",
                "brand": "burt's",
                "product_name": "",
                "rank": 1,
                "category_id": "c",
            },
        ]
        result = analyzer._group_by_date(data)
        assert result["2025-01-15"][0]["brand"] == "Burt's Bees"

    def test_group_by_date_empty_data(self, analyzer):
        """빈 데이터"""
        result = analyzer._group_by_date([])
        assert result == {}

    def test_group_by_date_missing_date_skipped(self, analyzer):
        """날짜 없는 레코드 건너뜀"""
        data = [{"asin": "A1", "brand": "B", "product_name": "", "rank": 1, "category_id": "c"}]
        result = analyzer._group_by_date(data)
        assert result == {}


# =========================================================================
# _analyze_laneige 테스트
# =========================================================================


class TestPeriodAnalyzerAnalyzeLaneige:
    """_analyze_laneige 테스트"""

    def test_analyze_laneige_basic(self, analyzer, sample_daily_data):
        """LANEIGE 기본 분석"""
        result = analyzer._analyze_laneige(sample_daily_data)

        assert "daily_sos" in result
        assert "avg_sos" in result
        assert "start_sos" in result
        assert "end_sos" in result
        assert "sos_change" in result
        assert "top_products" in result

    def test_analyze_laneige_sos_calculation(self, analyzer, sample_daily_data):
        """LANEIGE SoS 계산 (일별)"""
        result = analyzer._analyze_laneige(sample_daily_data)

        # Day 1: 1 LANEIGE out of 3 = 33.33%
        # Day 2: 1 LANEIGE out of 3 = 33.33%
        assert len(result["daily_sos"]) == 2
        assert result["daily_sos"][0]["sos"] == pytest.approx(33.33, abs=0.1)

    def test_analyze_laneige_no_laneige(self, analyzer):
        """LANEIGE 제품이 없는 경우"""
        daily = {
            "2025-01-15": [
                {
                    "asin": "B0X",
                    "title": "Other Product",
                    "brand": "Other",
                    "rank": 1,
                    "category": "lip_care",
                },
            ],
        }
        result = analyzer._analyze_laneige(daily)
        assert result["avg_sos"] == 0

    def test_analyze_laneige_rising_falling_products(self, analyzer):
        """상승/하락 제품 분류"""
        daily = {
            "2025-01-15": [
                {
                    "asin": "B0RISE",
                    "title": "LANEIGE Rising",
                    "brand": "LANEIGE",
                    "rank": 20,
                    "category": "lip_care",
                },
                {
                    "asin": "B0FALL",
                    "title": "LANEIGE Falling",
                    "brand": "LANEIGE",
                    "rank": 5,
                    "category": "lip_care",
                },
            ],
            "2025-01-16": [
                {
                    "asin": "B0RISE",
                    "title": "LANEIGE Rising",
                    "brand": "LANEIGE",
                    "rank": 5,
                    "category": "lip_care",
                },
                {
                    "asin": "B0FALL",
                    "title": "LANEIGE Falling",
                    "brand": "LANEIGE",
                    "rank": 20,
                    "category": "lip_care",
                },
            ],
        }
        result = analyzer._analyze_laneige(daily)
        assert len(result["top_products"]) == 2

    def test_analyze_laneige_sos_change_pct_zero_start(self, analyzer):
        """시작 SoS가 0일 때 변화율 0"""
        daily = {
            "2025-01-15": [
                {"asin": "B0X", "title": "Other", "brand": "Other", "rank": 1, "category": "c"},
            ],
            "2025-01-16": [
                {
                    "asin": "B0A",
                    "title": "LANEIGE X",
                    "brand": "LANEIGE",
                    "rank": 1,
                    "category": "c",
                },
            ],
        }
        result = analyzer._analyze_laneige(daily)
        assert result["sos_change_pct"] == 0


# =========================================================================
# _analyze_market 테스트
# =========================================================================


class TestPeriodAnalyzerAnalyzeMarket:
    """_analyze_market 테스트"""

    def test_analyze_market_hhi(self, analyzer, sample_daily_data):
        """시장 HHI 계산"""
        result = analyzer._analyze_market(sample_daily_data)
        assert "avg_hhi" in result
        assert "hhi_interpretation" in result
        assert "daily_hhi" in result
        assert len(result["daily_hhi"]) == 2

    def test_analyze_market_competitive(self, analyzer):
        """경쟁적 시장 (HHI < 1500)"""
        daily = {
            "2025-01-15": [
                {"brand": f"Brand_{i}", "rank": i, "asin": f"A{i}", "title": "", "category": ""}
                for i in range(1, 51)
            ],
        }
        result = analyzer._analyze_market(daily)
        assert result["avg_hhi"] < 1500
        assert "경쟁적" in result["hhi_interpretation"]

    def test_analyze_market_concentrated(self, analyzer):
        """고집중 시장"""
        daily = {
            "2025-01-15": [
                {"brand": "Monopoly", "rank": i, "asin": f"A{i}", "title": "", "category": ""}
                for i in range(1, 11)
            ],
        }
        result = analyzer._analyze_market(daily)
        # Single brand = HHI of 10000
        assert result["avg_hhi"] >= 2500
        assert "고집중" in result["hhi_interpretation"]

    def test_analyze_market_excludes_unknown(self, analyzer):
        """Unknown 브랜드 제외"""
        daily = {
            "2025-01-15": [
                {"brand": "LANEIGE", "rank": 1, "asin": "A1", "title": "", "category": ""},
                {"brand": "Unknown", "rank": 2, "asin": "A2", "title": "", "category": ""},
                {"brand": "", "rank": 3, "asin": "A3", "title": "", "category": ""},
            ],
        }
        result = analyzer._analyze_market(daily)
        # Only LANEIGE counted -> monopoly HHI=10000
        assert result["avg_hhi"] == 10000.0


# =========================================================================
# _analyze_brands 테스트
# =========================================================================


class TestPeriodAnalyzerAnalyzeBrands:
    """_analyze_brands 테스트"""

    def test_analyze_brands_basic(self, analyzer, sample_daily_data):
        """브랜드 분석 기본"""
        result = analyzer._analyze_brands(sample_daily_data)
        assert isinstance(result, list)
        assert len(result) > 0
        # Each brand has expected keys
        first = result[0]
        assert "brand" in first
        assert "avg_sos" in first
        assert "sos_change" in first
        assert "total_appearances" in first

    def test_analyze_brands_sorted_by_sos(self, analyzer, sample_daily_data):
        """SoS 내림차순 정렬"""
        result = analyzer._analyze_brands(sample_daily_data)
        if len(result) >= 2:
            assert result[0]["avg_sos"] >= result[1]["avg_sos"]

    def test_analyze_brands_excludes_unknown(self, analyzer):
        """Unknown 브랜드 제외"""
        daily = {
            "2025-01-15": [
                {"brand": "LANEIGE", "rank": 1, "asin": "A1", "title": "", "category": ""},
                {"brand": "Unknown", "rank": 2, "asin": "A2", "title": "", "category": ""},
            ],
        }
        result = analyzer._analyze_brands(daily)
        brand_names = [b["brand"] for b in result]
        assert "Unknown" not in brand_names

    def test_analyze_brands_max_20(self, analyzer):
        """최대 20개 브랜드 반환"""
        daily = {
            "2025-01-15": [
                {"brand": f"Brand_{i}", "rank": i, "asin": f"A{i}", "title": "", "category": ""}
                for i in range(1, 30)
            ],
        }
        result = analyzer._analyze_brands(daily)
        assert len(result) <= 20


# =========================================================================
# _analyze_categories 테스트
# =========================================================================


class TestPeriodAnalyzerAnalyzeCategories:
    """_analyze_categories 테스트"""

    def test_analyze_categories_basic(self, analyzer, sample_daily_data):
        """카테고리 분석"""
        result = analyzer._analyze_categories(sample_daily_data)
        assert isinstance(result, dict)
        assert "lip_care" in result
        assert "avg_sos" in result["lip_care"]

    def test_analyze_categories_multiple(self, analyzer, sample_daily_data):
        """여러 카테고리 분석"""
        result = analyzer._analyze_categories(sample_daily_data)
        # lip_care, skin_care, lip_makeup
        assert len(result) >= 2


# =========================================================================
# _identify_top_movers 테스트
# =========================================================================


class TestPeriodAnalyzerTopMovers:
    """_identify_top_movers 테스트"""

    def test_top_movers_basic(self, analyzer, sample_daily_data):
        """급등/급락 식별"""
        result = analyzer._identify_top_movers(sample_daily_data)
        assert "risers" in result
        assert "fallers" in result
        assert "laneige_risers" in result
        assert "laneige_fallers" in result

    def test_top_movers_calculates_change(self, analyzer):
        """순위 변화 계산"""
        daily = {
            "2025-01-15": [
                {
                    "asin": "B0RISE",
                    "title": "Rising Product",
                    "brand": "LANEIGE",
                    "rank": 50,
                    "category": "lip_care",
                },
            ],
            "2025-01-16": [
                {
                    "asin": "B0RISE",
                    "title": "Rising Product",
                    "brand": "LANEIGE",
                    "rank": 5,
                    "category": "lip_care",
                },
            ],
        }
        result = analyzer._identify_top_movers(daily)
        assert len(result["risers"]) == 1
        assert result["risers"][0]["change"] == 45  # 50 -> 5 = +45 상승

    def test_top_movers_single_day_no_movers(self, analyzer):
        """1일 데이터 -> 비교 불가"""
        daily = {
            "2025-01-15": [
                {"asin": "B0A", "title": "P", "brand": "B", "rank": 1, "category": "c"},
            ],
        }
        result = analyzer._identify_top_movers(daily)
        assert result["risers"] == []
        assert result["fallers"] == []

    def test_top_movers_laneige_filter(self, analyzer):
        """LANEIGE 제품만 필터"""
        daily = {
            "2025-01-15": [
                {
                    "asin": "B0L",
                    "title": "LANEIGE X",
                    "brand": "LANEIGE",
                    "rank": 20,
                    "category": "c",
                },
                {"asin": "B0O", "title": "Other X", "brand": "Other", "rank": 30, "category": "c"},
            ],
            "2025-01-16": [
                {
                    "asin": "B0L",
                    "title": "LANEIGE X",
                    "brand": "LANEIGE",
                    "rank": 5,
                    "category": "c",
                },
                {"asin": "B0O", "title": "Other X", "brand": "Other", "rank": 10, "category": "c"},
            ],
        }
        result = analyzer._identify_top_movers(daily)
        assert len(result["laneige_risers"]) == 1
        assert result["laneige_risers"][0]["asin"] == "B0L"


# =========================================================================
# _analyze_competitive_shifts 테스트
# =========================================================================


class TestPeriodAnalyzerCompetitiveShifts:
    """_analyze_competitive_shifts 테스트"""

    def test_competitive_shifts_basic(self, analyzer, sample_daily_data):
        """경쟁 구도 변화 분석"""
        result = analyzer._analyze_competitive_shifts(sample_daily_data)
        assert "new_entrants" in result
        assert "exits" in result
        assert "total_brands_start" in result
        assert "total_brands_end" in result

    def test_competitive_shifts_single_day(self, analyzer):
        """1일 데이터 -> 빈 결과"""
        daily = {
            "2025-01-15": [{"brand": "A", "asin": "A1", "rank": 1, "title": "", "category": ""}],
        }
        result = analyzer._analyze_competitive_shifts(daily)
        assert result["new_entrants"] == []
        assert result["exits"] == []

    def test_competitive_shifts_new_entrant(self, analyzer):
        """신규 진입 브랜드 감지"""
        daily = {
            "2025-01-15": [
                {"brand": "OldBrand", "asin": "A1", "rank": 1, "title": "", "category": ""},
            ],
            "2025-01-16": [
                {"brand": "OldBrand", "asin": "A1", "rank": 1, "title": "", "category": ""},
                {"brand": "NewBrand", "asin": "A2", "rank": 2, "title": "", "category": ""},
            ],
        }
        result = analyzer._analyze_competitive_shifts(daily)
        assert "NewBrand" in result["new_entrants"]

    def test_competitive_shifts_exit(self, analyzer):
        """이탈 브랜드 감지"""
        daily = {
            "2025-01-15": [
                {"brand": "StayBrand", "asin": "A1", "rank": 1, "title": "", "category": ""},
                {"brand": "LeaveBrand", "asin": "A2", "rank": 2, "title": "", "category": ""},
            ],
            "2025-01-16": [
                {"brand": "StayBrand", "asin": "A1", "rank": 1, "title": "", "category": ""},
            ],
        }
        result = analyzer._analyze_competitive_shifts(daily)
        assert "LeaveBrand" in result["exits"]

    def test_competitive_shifts_brand_counts(self, analyzer, sample_daily_data):
        """브랜드 수 카운트"""
        result = analyzer._analyze_competitive_shifts(sample_daily_data)
        assert result["total_brands_start"] == 3  # LANEIGE, Vaseline, COSRX
        assert result["total_brands_end"] == 3  # LANEIGE, Vaseline, e.l.f.


# =========================================================================
# _build_daily_trends 테스트
# =========================================================================


class TestPeriodAnalyzerDailyTrends:
    """_build_daily_trends 테스트"""

    def test_daily_trends_basic(self, analyzer, sample_daily_data):
        """일별 추이 데이터"""
        result = analyzer._build_daily_trends(sample_daily_data)
        assert len(result) == 2
        first = result[0]
        assert "date" in first
        assert "laneige_sos" in first
        assert "laneige_count" in first
        assert "total_products" in first
        assert "hhi" in first
        assert "brand_count" in first

    def test_daily_trends_sorted(self, analyzer, sample_daily_data):
        """날짜순 정렬"""
        result = analyzer._build_daily_trends(sample_daily_data)
        dates = [r["date"] for r in result]
        assert dates == sorted(dates)

    def test_daily_trends_sos_calculation(self, analyzer, sample_daily_data):
        """SoS 계산 정확도"""
        result = analyzer._build_daily_trends(sample_daily_data)
        # Day 1: 1 LANEIGE out of 3
        assert result[0]["laneige_count"] == 1
        assert result[0]["total_products"] == 3
        assert result[0]["laneige_sos"] == pytest.approx(33.33, abs=0.1)

    def test_daily_trends_empty(self, analyzer):
        """빈 데이터"""
        result = analyzer._build_daily_trends({})
        assert result == []


# =========================================================================
# _is_laneige 테스트
# =========================================================================


class TestPeriodAnalyzerIsLaneige:
    """_is_laneige 테스트"""

    def test_is_laneige_brand_match(self, analyzer):
        """브랜드명으로 매칭"""
        assert analyzer._is_laneige({"brand": "LANEIGE", "title": ""}) is True
        assert analyzer._is_laneige({"brand": "laneige", "title": ""}) is True

    def test_is_laneige_title_match(self, analyzer):
        """제품명으로 매칭"""
        assert analyzer._is_laneige({"brand": "Unknown", "title": "LANEIGE Lip Mask"}) is True

    def test_is_laneige_no_match(self, analyzer):
        """매칭 안 됨"""
        assert analyzer._is_laneige({"brand": "COSRX", "title": "Snail Mucin"}) is False

    def test_is_laneige_none_values(self, analyzer):
        """None 값 처리"""
        assert analyzer._is_laneige({"brand": None, "title": None}) is False
        assert analyzer._is_laneige({}) is False


# =========================================================================
# _get_category_name 테스트
# =========================================================================


class TestPeriodAnalyzerGetCategoryName:
    """_get_category_name 테스트"""

    def test_known_categories(self, analyzer):
        """알려진 카테고리 변환"""
        assert analyzer._get_category_name("lip_care") == "Lip Care"
        assert analyzer._get_category_name("skin_care") == "Skin Care"
        assert analyzer._get_category_name("lip_makeup") == "Lip Makeup"
        assert analyzer._get_category_name("face_powder") == "Face Powder"
        assert analyzer._get_category_name("beauty") == "Beauty & Personal Care"
        assert analyzer._get_category_name("beauty_personal_care") == "Beauty & Personal Care"

    def test_unknown_category_returns_id(self, analyzer):
        """알 수 없는 카테고리는 ID 그대로 반환"""
        assert analyzer._get_category_name("unknown_cat") == "unknown_cat"


# =========================================================================
# BRAND_NORMALIZATION 상수 테스트
# =========================================================================


class TestPeriodAnalyzerBrandNormalization:
    """BRAND_NORMALIZATION 매핑 테스트"""

    def test_normalization_map_not_empty(self):
        """매핑이 비어있지 않음"""
        assert len(PeriodAnalyzer.BRAND_NORMALIZATION) > 0

    def test_normalization_map_values_are_strings(self):
        """매핑 값은 모두 문자열"""
        for key, value in PeriodAnalyzer.BRAND_NORMALIZATION.items():
            assert isinstance(key, str)
            assert isinstance(value, str)

    def test_normalization_known_entries(self):
        """주요 매핑 항목 확인"""
        assert PeriodAnalyzer.BRAND_NORMALIZATION["rare"] == "Rare Beauty"
        assert PeriodAnalyzer.BRAND_NORMALIZATION["la"] == "La Roche-Posay"
        assert PeriodAnalyzer.BRAND_NORMALIZATION["fenty"] == "Fenty Beauty"
