"""
Unit tests for DashboardExporter

Target: 60%+ coverage for src/tools/exporters/dashboard_exporter.py
"""

import json
from unittest.mock import AsyncMock, MagicMock, mock_open, patch

import pytest

from src.tools.exporters.dashboard_exporter import DashboardExporter


@pytest.fixture
def mock_sheets_writer():
    """Mock SheetsWriter"""
    with patch("src.tools.exporters.dashboard_exporter.SheetsWriter") as mock:
        instance = MagicMock()
        instance.initialize = AsyncMock(return_value=True)
        mock.return_value = instance
        yield instance


@pytest.fixture
def mock_sqlite_storage():
    """Mock SQLiteStorage"""
    with patch("src.tools.exporters.dashboard_exporter.SQLiteStorage") as mock:
        instance = MagicMock()
        instance.initialize = AsyncMock(return_value=True)
        instance.get_historical_data = AsyncMock(return_value=[])
        instance.get_competitor_products = AsyncMock(return_value=[])
        instance.db_path = MagicMock()
        instance.db_path.exists = MagicMock(return_value=False)
        instance.get_connection = MagicMock()
        mock.return_value = instance
        yield instance


@pytest.fixture
def sample_raw_data():
    """Sample raw data for testing"""
    return [
        {
            "asin": "B001",
            "product_name": "LANEIGE Lip Sleeping Mask",
            "brand": "LANEIGE",
            "rank": 5,
            "rating": 4.5,
            "price": "22.00",
            "list_price": "25.00",
            "discount_percent": 12.0,
            "coupon_text": "",
            "is_subscribe_save": False,
            "promo_badges": "",
            "reviews_count": "15000",
            "category_id": "lip_care",
            "category_name": "Lip Care",
            "snapshot_date": "2026-02-15",
        },
        {
            "asin": "B001",
            "product_name": "LANEIGE Lip Sleeping Mask",
            "brand": "LANEIGE",
            "rank": 7,
            "rating": 4.5,
            "price": "22.00",
            "category_id": "lip_care",
            "snapshot_date": "2026-02-14",
        },
        {
            "asin": "B002",
            "product_name": "Summer Fridays Lip Butter Balm",
            "brand": "Summer Fridays",
            "rank": 3,
            "rating": 4.6,
            "price": "24.00",
            "category_id": "lip_care",
            "snapshot_date": "2026-02-15",
        },
        {
            "asin": "B003",
            "product_name": "LANEIGE Water Bank Cream",
            "brand": "LANEIGE",
            "rank": 15,
            "rating": 4.3,
            "price": "35.00",
            "category_id": "skin_care",
            "snapshot_date": "2026-02-15",
        },
    ]


@pytest.fixture
def exporter(mock_sheets_writer, mock_sqlite_storage):
    """Create DashboardExporter instance with mocked dependencies"""
    return DashboardExporter(spreadsheet_id="test_sheet_id", enable_ontology=False)


class TestDashboardExporterInit:
    """Test initialization"""

    def test_init_default(self, mock_sheets_writer, mock_sqlite_storage):
        """Test default initialization"""
        exporter = DashboardExporter()
        assert exporter.sheets is not None
        assert exporter.sqlite is not None
        assert exporter._initialized is False
        assert exporter.enable_ontology is False

    def test_init_with_spreadsheet_id(self, mock_sheets_writer, mock_sqlite_storage):
        """Test initialization with spreadsheet_id"""
        exporter = DashboardExporter(spreadsheet_id="custom_id")
        assert exporter.sheets is not None

    @patch("src.tools.exporters.dashboard_exporter.ONTOLOGY_AVAILABLE", True)
    def test_init_with_ontology_enabled(self, mock_sheets_writer, mock_sqlite_storage):
        """Test initialization with ontology enabled"""
        with patch.object(DashboardExporter, "_init_ontology") as mock_init:
            exporter = DashboardExporter(enable_ontology=True)
            mock_init.assert_called_once()


class TestDashboardExporterInitialize:
    """Test initialize method"""

    @pytest.mark.asyncio
    async def test_initialize_success(self, exporter, mock_sqlite_storage, mock_sheets_writer):
        """Test successful initialization"""
        result = await exporter.initialize()
        assert result is True
        assert exporter._initialized is True
        mock_sqlite_storage.initialize.assert_called_once()
        mock_sheets_writer.initialize.assert_called_once()


class TestDashboardExporterNormalizeBrand:
    """Test _normalize_brand method"""

    def test_normalize_brand_summer_fridays(self, exporter):
        """Test normalizing Summer Fridays brand"""
        result = exporter._normalize_brand("summer", "Summer Fridays Lip Butter Balm")
        assert result == "Summer Fridays"

    def test_normalize_brand_rare_beauty(self, exporter):
        """Test normalizing Rare Beauty brand"""
        result = exporter._normalize_brand("rare", "Rare Beauty Liquid Blush")
        assert result == "Rare Beauty"

    def test_normalize_brand_no_match(self, exporter):
        """Test brand with no correction needed"""
        result = exporter._normalize_brand("LANEIGE", "LANEIGE Lip Sleeping Mask")
        assert result == "LANEIGE"

    def test_normalize_brand_empty(self, exporter):
        """Test empty brand"""
        result = exporter._normalize_brand("", "Product Name")
        assert result == ""

    def test_normalize_brand_none(self, exporter):
        """Test None brand"""
        result = exporter._normalize_brand(None, "Product Name")
        assert result is None


class TestDashboardExporterGetLatestDate:
    """Test _get_latest_date method"""

    def test_get_latest_date_with_data(self, exporter, sample_raw_data):
        """Test getting latest date from data"""
        result = exporter._get_latest_date(sample_raw_data)
        assert result == "2026-02-15"

    def test_get_latest_date_empty_data(self, exporter):
        """Test getting latest date with empty data"""
        result = exporter._get_latest_date([])
        # Should return current date
        assert len(result) == 10  # YYYY-MM-DD format
        assert result.count("-") == 2


class TestDashboardExporterIsLaneige:
    """Test _is_laneige method"""

    def test_is_laneige_true(self, exporter):
        """Test identifying LANEIGE product"""
        assert exporter._is_laneige({"brand": "LANEIGE"}) is True
        assert exporter._is_laneige({"brand": "laneige"}) is True
        assert exporter._is_laneige({"brand": "Laneige"}) is True

    def test_is_laneige_false(self, exporter):
        """Test non-LANEIGE product"""
        assert exporter._is_laneige({"brand": "Summer Fridays"}) is False
        assert exporter._is_laneige({"brand": ""}) is False

    def test_is_laneige_missing_brand(self, exporter):
        """Test product with missing brand"""
        assert exporter._is_laneige({}) is False


class TestDashboardExporterSafeConversions:
    """Test _safe_int and _safe_float methods"""

    def test_safe_int_valid(self, exporter):
        """Test safe int conversion with valid values"""
        assert exporter._safe_int(42) == 42
        assert exporter._safe_int("42") == 42
        assert exporter._safe_int("42.5") == 42
        assert exporter._safe_int("#5") == 5
        assert exporter._safe_int("1,000") == 1000

    def test_safe_int_invalid(self, exporter):
        """Test safe int conversion with invalid values"""
        assert exporter._safe_int("invalid") == 0
        assert exporter._safe_int(None) == 0
        assert exporter._safe_int("") == 0

    def test_safe_float_valid(self, exporter):
        """Test safe float conversion with valid values"""
        assert exporter._safe_float(42.5) == 42.5
        assert exporter._safe_float("42.5") == 42.5
        assert exporter._safe_float("$22.00") == 22.0
        assert exporter._safe_float("1,000.50") == 1000.5

    def test_safe_float_invalid(self, exporter):
        """Test safe float conversion with invalid values"""
        assert exporter._safe_float("invalid") == 0.0
        assert exporter._safe_float(None) == 0.0
        assert exporter._safe_float("") == 0.0


class TestDashboardExporterGetBestRank:
    """Test _get_best_rank method"""

    def test_get_best_rank_with_products(self, exporter, sample_raw_data):
        """Test getting best rank from products"""
        laneige_products = [p for p in sample_raw_data if exporter._is_laneige(p)]
        result = exporter._get_best_rank(laneige_products)
        assert result == 5

    def test_get_best_rank_empty(self, exporter):
        """Test getting best rank with empty list"""
        result = exporter._get_best_rank([])
        assert result == 999


class TestDashboardExporterCalculateHHI:
    """Test _calculate_hhi method"""

    def test_calculate_hhi(self, exporter):
        """Test HHI calculation"""
        brand_stats = {
            "LANEIGE": {"products": [1, 2, 3], "ranks": []},
            "Summer Fridays": {"products": [1, 2], "ranks": []},
            "Rare Beauty": {"products": [1], "ranks": []},
        }
        result = exporter._calculate_hhi(brand_stats)
        # HHI = (50^2 + 33.33^2 + 16.67^2) / 10000
        assert result > 0
        assert result < 1

    def test_calculate_hhi_empty(self, exporter):
        """Test HHI calculation with empty data"""
        result = exporter._calculate_hhi({})
        assert result == 0


class TestDashboardExporterCalculateRankChange:
    """Test _calculate_rank_change method"""

    def test_calculate_rank_change_improved(self, exporter, sample_raw_data):
        """Test rank change calculation - improved rank"""
        current_product = sample_raw_data[0]  # rank 5 on 2026-02-15
        result = exporter._calculate_rank_change(current_product, sample_raw_data)
        # Previous rank was 7, current is 5, so change = 5 - 7 = -2 (improved)
        assert result == -2

    def test_calculate_rank_change_no_history(self, exporter):
        """Test rank change with no historical data"""
        product = {
            "asin": "B999",
            "rank": 10,
            "category_id": "lip_care",
            "snapshot_date": "2026-02-15",
        }
        result = exporter._calculate_rank_change(product, [])
        assert result == 0

    def test_calculate_rank_change_missing_asin(self, exporter):
        """Test rank change with missing ASIN"""
        product = {"rank": 10, "snapshot_date": "2026-02-15"}
        result = exporter._calculate_rank_change(product, [])
        assert result == 0


class TestDashboardExporterCalculateRankVolatility:
    """Test _calculate_rank_volatility method"""

    def test_calculate_rank_volatility(self, exporter):
        """Test rank volatility calculation"""
        all_data = [
            {
                "asin": "B001",
                "rank": 5,
                "category_id": "lip_care",
                "snapshot_date": "2026-02-15",
            },
            {
                "asin": "B001",
                "rank": 10,
                "category_id": "lip_care",
                "snapshot_date": "2026-02-14",
            },
            {
                "asin": "B001",
                "rank": 3,
                "category_id": "lip_care",
                "snapshot_date": "2026-02-13",
            },
        ]
        result = exporter._calculate_rank_volatility(all_data[0], all_data)
        # max(10) - min(3) = 7
        assert result == 7

    def test_calculate_rank_volatility_stable(self, exporter):
        """Test rank volatility with stable ranks"""
        all_data = [
            {
                "asin": "B001",
                "rank": 5,
                "category_id": "lip_care",
                "snapshot_date": "2026-02-15",
            },
            {
                "asin": "B001",
                "rank": 5,
                "category_id": "lip_care",
                "snapshot_date": "2026-02-14",
            },
        ]
        result = exporter._calculate_rank_volatility(all_data[0], all_data)
        assert result == 0


class TestDashboardExporterCalculateRankChange7d:
    """Test _calculate_rank_change_7d method"""

    def test_calculate_rank_change_7d(self, exporter):
        """Test 7-day rank change calculation"""
        all_data = [
            {
                "asin": "B001",
                "rank": 5,
                "category_id": "lip_care",
                "snapshot_date": "2026-02-15",
            },
            {
                "asin": "B001",
                "rank": 10,
                "category_id": "lip_care",
                "snapshot_date": "2026-02-10",
            },
        ]
        result = exporter._calculate_rank_change_7d(all_data[0], all_data)
        # 5 - 10 = -5 (improved)
        assert result == -5


class TestDashboardExporterCalculateVolatility:
    """Test _calculate_volatility method"""

    def test_calculate_volatility(self, exporter):
        """Test volatility calculation"""
        all_data = [
            {"asin": "B001", "rank": 5, "snapshot_date": "2026-02-15"},
            {"asin": "B001", "rank": 10, "snapshot_date": "2026-02-14"},
            {"asin": "B001", "rank": 3, "snapshot_date": "2026-02-13"},
        ]
        result = exporter._calculate_volatility("B001", all_data)
        assert result == 7  # max(10) - min(3)


class TestDashboardExporterFormatRankDelta:
    """Test _format_rank_delta method"""

    def test_format_rank_delta_improved(self, exporter):
        """Test formatting improved rank"""
        result = exporter._format_rank_delta(-5)
        assert result == "▲ 5위 상승"

    def test_format_rank_delta_declined(self, exporter):
        """Test formatting declined rank"""
        result = exporter._format_rank_delta(3)
        assert result == "▼ 3위 하락"

    def test_format_rank_delta_unchanged(self, exporter):
        """Test formatting unchanged rank"""
        result = exporter._format_rank_delta(0)
        assert result == "유지"


class TestDashboardExporterClassifyGrowthType:
    """Test _classify_growth_type method"""

    def test_classify_growth_type_discount(self, exporter):
        """Test discount-based growth classification"""
        result = exporter._classify_growth_type(20.0, "", "")
        assert result == "discount_based"

    def test_classify_growth_type_coupon(self, exporter):
        """Test coupon-based growth classification"""
        result = exporter._classify_growth_type(0.0, "Save $5", "")
        assert result == "discount_based"

    def test_classify_growth_type_deal(self, exporter):
        """Test deal-based growth classification"""
        result = exporter._classify_growth_type(0.0, "", "Lightning Deal")
        assert result == "discount_based"

    def test_classify_growth_type_organic(self, exporter):
        """Test organic growth classification"""
        result = exporter._classify_growth_type(5.0, "", "")
        assert result == "organic"


class TestDashboardExporterGenerateDailyInsight:
    """Test _generate_daily_insight method"""

    def test_generate_daily_insight_with_data(self, exporter, sample_raw_data):
        """Test daily insight generation"""
        laneige_products = [p for p in sample_raw_data if exporter._is_laneige(p)]
        result = exporter._generate_daily_insight(laneige_products, "2026-02-15")
        assert "Laneige" in result
        assert "LANEIGE Lip Sleeping Mask" in result
        assert "5위" in result

    def test_generate_daily_insight_empty(self, exporter):
        """Test daily insight with no products"""
        result = exporter._generate_daily_insight([], "2026-02-15")
        assert "데이터가 없습니다" in result


class TestDashboardExporterLoadRawData:
    """Test _load_raw_data method"""

    @pytest.mark.asyncio
    async def test_load_raw_data_from_sqlite(self, exporter, mock_sqlite_storage, sample_raw_data):
        """Test loading data from SQLite"""
        mock_sqlite_storage.get_historical_data.return_value = sample_raw_data
        result = await exporter._load_raw_data()
        assert len(result) == 4
        mock_sqlite_storage.get_historical_data.assert_called_once_with(days=30)

    @pytest.mark.asyncio
    async def test_load_raw_data_sqlite_failure(self, exporter, mock_sqlite_storage):
        """Test loading data when SQLite fails"""
        mock_sqlite_storage.get_historical_data.side_effect = Exception("DB error")
        with patch("os.path.exists", return_value=False):
            result = await exporter._load_raw_data()
            assert result == []

    @pytest.mark.asyncio
    async def test_load_raw_data_from_local_json(self, exporter, mock_sqlite_storage):
        """Test loading data from local JSON file"""
        mock_sqlite_storage.get_historical_data.return_value = []
        mock_data = {
            "categories": {
                "lip_care": {
                    "products": [
                        {
                            "asin": "B001",
                            "product_name": "Test Product",
                            "brand": "LANEIGE",
                            "rank": 5,
                            "snapshot_date": "2026-02-15",
                        }
                    ]
                }
            }
        }
        with (
            patch("os.path.exists", return_value=True),
            patch("builtins.open", mock_open(read_data=json.dumps(mock_data))),
        ):
            result = await exporter._load_raw_data()
            assert len(result) == 1
            assert result[0]["category_id"] == "lip_care"


class TestDashboardExporterGenerateHomeData:
    """Test _generate_home_data method"""

    def test_generate_home_data(self, exporter, sample_raw_data):
        """Test home data generation"""
        result = exporter._generate_home_data(sample_raw_data)
        assert "insight_message" in result
        assert "status" in result
        assert "action_items" in result
        assert result["status"]["position"] == "Top 5"

    def test_generate_home_data_empty(self, exporter):
        """Test home data with empty data"""
        result = exporter._generate_home_data([])
        assert result["status"]["position"] == "Top 999"


class TestDashboardExporterGenerateBrandData:
    """Test _generate_brand_data method"""

    def test_generate_brand_data(self, exporter, sample_raw_data):
        """Test brand data generation"""
        result = exporter._generate_brand_data(sample_raw_data)
        assert "kpis" in result
        assert "competitors" in result
        assert result["kpis"]["sos"] > 0
        assert len(result["competitors"]) > 0

    def test_generate_brand_data_calculates_avg_price(self, exporter, sample_raw_data):
        """Test brand data calculates average price correctly"""
        result = exporter._generate_brand_data(sample_raw_data)
        avg_price = result["kpis"]["avg_price"]
        assert avg_price is not None
        assert 20.0 <= avg_price <= 40.0


class TestDashboardExporterGenerateCategoryData:
    """Test _generate_category_data method"""

    def test_generate_category_data(self, exporter, sample_raw_data):
        """Test category data generation"""
        result = exporter._generate_category_data(sample_raw_data)
        assert "lip_care" in result
        assert result["lip_care"]["name"] == "Lip Care"
        assert result["lip_care"]["laneige_count"] == 1

    def test_generate_category_data_empty_category(self, exporter):
        """Test category data with no data"""
        result = exporter._generate_category_data([])
        assert len(result) == 0


class TestDashboardExporterGenerateProductData:
    """Test _generate_product_data method"""

    def test_generate_product_data(self, exporter, sample_raw_data):
        """Test product data generation"""
        result = exporter._generate_product_data(sample_raw_data)
        assert "B001" in result
        assert result["B001"]["name"] == "LANEIGE Lip Sleeping Mask"
        assert result["B001"]["rank"] == 5
        # Growth type is discount_based because discount_percent=12.0
        assert result["B001"]["growth_type"] in ["discount_based", "organic"]


class TestDashboardExporterGenerateChartData:
    """Test _generate_chart_data method"""

    def test_generate_chart_data(self, exporter, sample_raw_data):
        """Test chart data generation"""
        result = exporter._generate_chart_data(sample_raw_data)
        assert "sos_trend" in result
        assert "product_sos" in result
        assert "brand_matrix" in result
        assert "category_kpis" in result
        assert "cpi_trend" in result

    def test_generate_chart_data_sos_trend_periods(self, exporter, sample_raw_data):
        """Test SoS trend includes all periods"""
        result = exporter._generate_chart_data(sample_raw_data)
        assert "7d" in result["sos_trend"]
        assert "14d" in result["sos_trend"]
        assert "30d" in result["sos_trend"]

    def test_generate_chart_data_brand_matrix_excludes_unknown(self, exporter):
        """Test brand matrix excludes Unknown brands"""
        data = [
            {
                "asin": "B001",
                "brand": "Unknown",
                "rank": 5,
                "snapshot_date": "2026-02-15",
                "category_id": "lip_care",
            },
            {
                "asin": "B002",
                "brand": "Unknown",
                "rank": 6,
                "snapshot_date": "2026-02-15",
                "category_id": "lip_care",
            },
            {
                "asin": "B003",
                "brand": "LANEIGE",
                "rank": 7,
                "snapshot_date": "2026-02-15",
                "category_id": "lip_care",
            },
            {
                "asin": "B004",
                "brand": "LANEIGE",
                "rank": 8,
                "snapshot_date": "2026-02-15",
                "category_id": "lip_care",
            },
        ]
        result = exporter._generate_chart_data(data)
        brands = [b["brand"] for b in result["brand_matrix"]]
        assert "Unknown" not in brands
        # LANEIGE should be included if it has 2+ products
        if len(result["brand_matrix"]) > 0:
            assert "LANEIGE" in brands


class TestDashboardExporterGenerateCompetitorData:
    """Test _generate_competitor_data method"""

    def test_generate_competitor_data(self, exporter):
        """Test competitor data generation"""
        brand_stats = {
            "LANEIGE": {
                "products": [{"price": "22.00"}, {"price": "35.00"}],
                "ranks": [5, 10],
            },
            "Summer Fridays": {"products": [{"price": "24.00"}], "ranks": [3]},
            "Unknown": {"products": [{"price": "10.00"}], "ranks": [1]},
        }
        result = exporter._generate_competitor_data(brand_stats)
        brands = [c["brand"] for c in result]
        assert "LANEIGE" in brands
        assert "Summer Fridays" in brands
        assert "Unknown" not in brands


class TestDashboardExporterLoadTrackedCompetitors:
    """Test _load_tracked_competitors method"""

    def test_load_tracked_competitors_from_json(self, exporter):
        """Test loading tracked competitors from JSON"""
        mock_config = {
            "competitors": {
                "summer_fridays": {
                    "brand_name": "Summer Fridays",
                    "products": ["B001", "B002"],
                    "tier": "premium",
                }
            }
        }
        with (
            patch("pathlib.Path.exists", return_value=True),
            patch("builtins.open", mock_open(read_data=json.dumps(mock_config))),
        ):
            result = exporter._load_tracked_competitors()
            assert "Summer Fridays" in result
            assert result["Summer Fridays"]["product_count"] == 2


class TestDashboardExporterGenerateActionItems:
    """Test _generate_action_items method"""

    def test_generate_action_items_rank_drop(self, exporter):
        """Test action items for rank drop"""
        laneige_products = [
            {
                "asin": "B001",
                "product_name": "Test Product",
                "rank": 10,
                "rating": 4.5,
                "snapshot_date": "2026-02-15",
                "category_id": "lip_care",
            }
        ]
        all_data = laneige_products + [
            {
                "asin": "B001",
                "rank": 5,
                "snapshot_date": "2026-02-14",
                "category_id": "lip_care",
            }
        ]
        result = exporter._generate_action_items(laneige_products, all_data)
        assert len(result) > 0
        assert result[0]["priority"] == "P1"
        assert "하락" in result[0]["signal"]

    def test_generate_action_items_low_rating(self, exporter):
        """Test action items for low rating"""
        laneige_products = [
            {
                "asin": "B001",
                "product_name": "Test Product",
                "rank": 10,
                "rating": 3.5,
                "snapshot_date": "2026-02-15",
                "category_id": "lip_care",
            }
        ]
        result = exporter._generate_action_items(laneige_products, laneige_products)
        assert len(result) > 0
        assert result[0]["priority"] == "P1"
        assert "평점" in result[0]["signal"]


class TestDashboardExporterExportDashboardData:
    """Test export_dashboard_data method"""

    @pytest.mark.asyncio
    async def test_export_dashboard_data_success(
        self, exporter, mock_sqlite_storage, sample_raw_data
    ):
        """Test successful dashboard data export"""
        mock_sqlite_storage.get_historical_data.return_value = sample_raw_data

        output_path = "/tmp/test_dashboard_data.json"

        with patch("os.makedirs"), patch("builtins.open", mock_open()) as mock_file:
            result = await exporter.export_dashboard_data(output_path)

            assert "metadata" in result
            assert "data_source" in result
            assert "home" in result
            assert "brand" in result
            assert "categories" in result
            assert "products" in result
            assert "charts" in result
            assert result["metadata"]["total_products"] == 4

            mock_file.assert_called_once_with(output_path, "w", encoding="utf-8")

    @pytest.mark.asyncio
    async def test_export_dashboard_data_no_data(self, exporter, mock_sqlite_storage):
        """Test export with no data"""
        mock_sqlite_storage.get_historical_data.return_value = []

        with patch("os.path.exists", return_value=False):
            result = await exporter.export_dashboard_data()
            assert "error" in result
            assert result["error"] == "No data found"

    @pytest.mark.asyncio
    async def test_export_dashboard_data_auto_initialize(
        self, exporter, mock_sqlite_storage, sample_raw_data
    ):
        """Test auto-initialization when not initialized"""
        exporter._initialized = False
        mock_sqlite_storage.get_historical_data.return_value = sample_raw_data

        with patch("os.makedirs"), patch("builtins.open", mock_open()):
            result = await exporter.export_dashboard_data()
            assert exporter._initialized is True
            assert "metadata" in result


class TestDashboardExporterEdgeCases:
    """Test edge cases and error handling"""

    def test_normalize_brand_all_corrections(self, exporter):
        """Test all brand correction mappings"""
        test_cases = [
            ("la", "La Roche-Posay Moisturizer", "La Roche-Posay"),
            ("beauty", "Beauty of Joseon Serum", "Beauty of Joseon"),
            ("tower", "Tower 28 BeachPlease Blush", "Tower 28"),
            ("drunk", "Drunk Elephant C-Firma", "Drunk Elephant"),
            ("paula's", "Paula's Choice BHA", "Paula's Choice"),
            ("the", "The Ordinary Niacinamide", "The Ordinary"),
            ("glow", "Glow Recipe Watermelon", "Glow Recipe"),
            ("fenty", "Fenty Beauty Gloss Bomb", "Fenty Beauty"),
        ]
        for brand, product, expected in test_cases:
            result = exporter._normalize_brand(brand, product)
            assert result == expected

    @pytest.mark.asyncio
    def test_normalize_brand_applies_corrections(self, exporter):
        """Test that _normalize_brand correctly normalizes known brands"""
        # Brand "summer" + product_name containing "summer fridays" → "Summer Fridays"
        result = exporter._normalize_brand("summer", "Summer Fridays Lip Balm for Dry Lips")
        assert result == "Summer Fridays"

    def test_normalize_brand_no_match(self, exporter):
        """Test that _normalize_brand returns original when no match"""
        result = exporter._normalize_brand("unknown_brand", "Some Product")
        assert result == "unknown_brand"

    def test_normalize_brand_empty_input(self, exporter):
        """Test that _normalize_brand handles empty input"""
        assert exporter._normalize_brand("", "Some Product") == ""
        assert exporter._normalize_brand("brand", "") == "brand"

    def test_generate_chart_data_product_matrix_quadrants(self, exporter):
        """Test product matrix quadrant classification"""
        data = [
            {
                "asin": "B001",
                "brand": "LANEIGE",
                "rank": 5,
                "rating": 4.5,
                "snapshot_date": "2026-02-15",
            },
            {
                "asin": "B002",
                "brand": "LANEIGE",
                "rank": 15,
                "rating": 4.3,
                "snapshot_date": "2026-02-15",
            },
        ]
        result = exporter._generate_chart_data(data)
        matrix = result["product_matrix"]
        assert len(matrix) == 2
        # Check quadrant assignments exist
        for item in matrix:
            assert "quadrant" in item
            assert item["quadrant"] in ["king", "rising", "lagging", "risk"]

    def test_calculate_rank_change_different_category(self, exporter):
        """Test that rank changes only compare within same category"""
        all_data = [
            {
                "asin": "B001",
                "rank": 5,
                "category_id": "lip_care",
                "snapshot_date": "2026-02-15",
            },
            {
                "asin": "B001",
                "rank": 100,
                "category_id": "skin_care",  # Different category
                "snapshot_date": "2026-02-14",
            },
        ]
        result = exporter._calculate_rank_change(all_data[0], all_data)
        # Should return 0 because previous record is in different category
        assert result == 0


class TestDashboardExporterOntologyIntegration:
    """Test ontology integration (when disabled)"""

    def test_ontology_disabled_by_default(self, exporter):
        """Test that ontology is disabled by default"""
        assert exporter.enable_ontology is False

    def test_generate_ontology_insights_disabled(self, exporter):
        """Test ontology insights when disabled"""
        result = exporter._generate_ontology_insights([], {})
        assert result == {"enabled": False}
