"""
Unit tests for AmazonDealsScraper (non-browser tests)
"""

import pytest

from src.tools.scrapers.deals_scraper import AmazonDealsScraper, DealRecord, DealType


@pytest.fixture
def scraper():
    """Create scraper without browser initialization"""
    scraper = AmazonDealsScraper.__new__(AmazonDealsScraper)
    scraper.config = {}
    scraper.browser = None
    scraper.delay_seconds = 0
    return scraper


class TestDealRecord:
    """Test DealRecord dataclass"""

    def test_to_dict(self):
        record = DealRecord(
            snapshot_datetime="2026-02-16T10:00:00",
            asin="B084RGF8YJ",
            product_name="LANEIGE Lip Sleeping Mask",
            brand="LANEIGE",
            category="Beauty",
            deal_price=17.99,
            original_price=24.00,
            discount_percent=25.0,
            deal_type="lightning",
        )
        d = record.to_dict()

        assert d["asin"] == "B084RGF8YJ"
        assert d["deal_price"] == 17.99
        assert d["discount_percent"] == 25.0
        assert d["deal_type"] == "lightning"

    def test_default_values(self):
        record = DealRecord(
            snapshot_datetime="2026-02-16T10:00:00",
            asin="B084RGF8YJ",
            product_name="Test",
            brand="Test",
            category="Beauty",
            deal_price=10.0,
        )
        assert record.deal_type == "regular"
        assert record.original_price is None
        assert record.time_remaining is None
        assert record.claimed_percent is None


class TestDealType:
    """Test DealType enum"""

    def test_deal_types(self):
        assert DealType.LIGHTNING.value == "lightning"
        assert DealType.DEAL_OF_THE_DAY.value == "deal_of_day"
        assert DealType.BEST_DEAL.value == "best_deal"
        assert DealType.COUPON.value == "coupon"
        assert DealType.REGULAR.value == "regular"


class TestAmazonDealsScraper:
    """Test AmazonDealsScraper non-browser methods"""

    def test_extract_brand_known(self, scraper):
        """Test brand extraction for known brands"""
        assert scraper._extract_brand("LANEIGE Lip Sleeping Mask") == "LANEIGE"
        assert scraper._extract_brand("COSRX Snail Mucin Essence") == "COSRX"
        assert scraper._extract_brand("Maybelline Instant Age Rewind") == "Maybelline"

    def test_extract_brand_case_insensitive(self, scraper):
        """Test case-insensitive brand matching"""
        assert scraper._extract_brand("laneige lip mask") == "LANEIGE"
        assert scraper._extract_brand("cosrx snail cream") == "COSRX"

    def test_extract_brand_unknown(self, scraper):
        """Test brand extraction for unknown brands"""
        result = scraper._extract_brand("SomeNewBrand Face Cream")
        assert result == "SomeNewBrand"  # first word

    def test_extract_brand_empty(self, scraper):
        """Test brand extraction with empty string"""
        assert scraper._extract_brand("") == "Unknown"

    def test_parse_price_valid(self, scraper):
        """Test price parsing with valid inputs"""
        assert scraper._parse_price("$24.99") == 24.99
        assert scraper._parse_price("$1,299.99") is None  # > 1000
        assert scraper._parse_price("$0.99") == 0.99

    def test_parse_price_invalid(self, scraper):
        """Test price parsing with invalid inputs"""
        assert scraper._parse_price("") is None
        assert scraper._parse_price(None) is None
        assert scraper._parse_price("no price here") is None
        assert scraper._parse_price("€24.99") is None  # no dollar sign

    def test_parse_price_edge_cases(self, scraper):
        """Test price parsing edge cases"""
        assert scraper._parse_price("$0.49") is None  # below 0.50
        assert scraper._parse_price("$0.50") == 0.50

    def test_parse_time_remaining(self, scraper):
        """Test time remaining parsing"""
        assert scraper._parse_time_remaining("2h 30m") == 2 * 3600 + 30 * 60
        assert scraper._parse_time_remaining("1h") == 3600
        assert scraper._parse_time_remaining("45m") == 45 * 60
        assert scraper._parse_time_remaining("30s") == 30

    def test_parse_time_remaining_combined(self, scraper):
        """Test combined time parsing"""
        result = scraper._parse_time_remaining("1h 30m 15s")
        assert result == 3600 + 1800 + 15

    def test_parse_time_remaining_invalid(self, scraper):
        """Test time parsing with invalid input"""
        assert scraper._parse_time_remaining("no time") is None

    def test_is_beauty_product(self, scraper):
        """Test beauty product detection"""
        beauty_deal = DealRecord(
            snapshot_datetime="",
            asin="A1",
            product_name="LANEIGE Lip Sleeping Mask",
            brand="LANEIGE",
            category="Beauty",
            deal_price=20.0,
        )
        assert scraper._is_beauty_product(beauty_deal) is True

        non_beauty = DealRecord(
            snapshot_datetime="",
            asin="A2",
            product_name="Wireless Bluetooth Headphones",
            brand="Sony",
            category="Electronics",
            deal_price=50.0,
        )
        assert scraper._is_beauty_product(non_beauty) is False

    def test_is_beauty_product_various_keywords(self, scraper):
        """Test beauty detection with various keywords"""
        keywords = ["face cream", "serum bottle", "moisturizer set", "sunscreen SPF 50"]
        for name in keywords:
            deal = DealRecord(
                snapshot_datetime="",
                asin="A1",
                product_name=name,
                brand="Test",
                category="Beauty",
                deal_price=10.0,
            )
            assert scraper._is_beauty_product(deal) is True, f"Should detect '{name}' as beauty"

    def test_deals_urls_exist(self):
        """Test that DEALS_URLS are defined"""
        assert "all_deals" in AmazonDealsScraper.DEALS_URLS
        assert "beauty_deals" in AmazonDealsScraper.DEALS_URLS

    def test_watch_brands_include_laneige(self):
        """Test WATCH_BRANDS includes LANEIGE"""
        assert "LANEIGE" in AmazonDealsScraper.WATCH_BRANDS
        assert "COSRX" in AmazonDealsScraper.WATCH_BRANDS
