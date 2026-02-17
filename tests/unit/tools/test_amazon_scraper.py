"""
Unit tests for AmazonScraper

Target: 60%+ coverage for src/tools/scrapers/amazon_scraper.py
All external dependencies (Playwright, browser, network) are mocked.
"""

import asyncio
import json
from unittest.mock import AsyncMock, Mock, mock_open, patch

import pytest

from src.tools.scrapers.amazon_scraper import AmazonScraper, CircuitBreaker


@pytest.fixture
def mock_config():
    """Mock configuration data"""
    return {
        "categories": {
            "lip_care": {
                "name": "Lip Care",
                "url": "https://www.amazon.com/bestsellers/beauty/lip-care",
                "amazon_node_id": "3761351",
                "level": 2,
            },
            "skin_care": {
                "name": "Skin Care",
                "url": "https://www.amazon.com/bestsellers/beauty/skin-care",
                "amazon_node_id": "11060451",
                "level": 1,
            },
        },
        "system": {
            "crawler": {
                "delay_base_seconds": 8,
                "delay_random_max": 4,
                "category_delay_seconds": 45,
                "max_retries": 3,
            }
        },
    }


@pytest.fixture
def scraper(mock_config):
    """Create scraper with mocked config"""
    with patch("builtins.open", mock_open(read_data=json.dumps(mock_config))):
        scraper = AmazonScraper(config_path="./config/thresholds.json")
        scraper.brand_resolver = None  # Disable brand resolver
        return scraper


class TestCircuitBreaker:
    """Test CircuitBreaker functionality"""

    def test_initial_state(self):
        cb = CircuitBreaker(threshold=3, reset_minutes=30)
        assert cb.failures == 0
        assert cb.is_open is False
        assert cb.can_proceed() is True

    def test_record_failure(self):
        cb = CircuitBreaker(threshold=3, reset_minutes=30)
        cb.record_failure()
        assert cb.failures == 1
        assert cb.is_open is False

    def test_open_after_threshold(self):
        cb = CircuitBreaker(threshold=3, reset_minutes=30)
        cb.record_failure()
        cb.record_failure()
        cb.record_failure()
        assert cb.failures == 3
        assert cb.is_open is True
        assert cb.can_proceed() is False

    def test_record_success(self):
        cb = CircuitBreaker(threshold=3, reset_minutes=30)
        cb.record_failure()
        cb.record_failure()
        assert cb.failures == 2
        cb.record_success()
        assert cb.failures == 0
        assert cb.is_open is False

    def test_reset_after_timeout(self):
        cb = CircuitBreaker(threshold=3, reset_minutes=0)  # 0 minutes for instant reset
        cb.record_failure()
        cb.record_failure()
        cb.record_failure()
        assert cb.is_open is True

        # Wait slightly to trigger timeout
        import time

        time.sleep(0.1)

        assert cb.can_proceed() is True
        assert cb.is_open is False
        assert cb.failures == 0

    def test_get_backoff_seconds(self):
        cb = CircuitBreaker(threshold=5, reset_minutes=30)

        cb.record_failure()
        assert cb.get_backoff_seconds() == 60  # 60 * 2^0

        cb.record_failure()
        assert cb.get_backoff_seconds() == 120  # 60 * 2^1

        cb.record_failure()
        assert cb.get_backoff_seconds() == 240  # 60 * 2^2

        # Test max cap (600 seconds)
        for _ in range(10):
            cb.record_failure()
        assert cb.get_backoff_seconds() == 600


class TestAmazonScraper:
    """Test AmazonScraper methods"""

    def test_init_default_config(self):
        """Test initialization with default config"""
        with patch("builtins.open", mock_open(read_data="{}")):
            scraper = AmazonScraper(config_path="./config/thresholds.json")
            assert scraper.config == {}  # Empty JSON returns empty dict
            assert scraper.base_url == "https://www.amazon.com"
            assert scraper.browser is None

    def test_init_with_config(self, mock_config):
        """Test initialization with valid config"""
        with patch("builtins.open", mock_open(read_data=json.dumps(mock_config))):
            scraper = AmazonScraper(config_path="./config/thresholds.json")
            assert "lip_care" in scraper.config["categories"]
            assert scraper.delay_base_seconds == 8
            assert scraper.delay_random_max == 4
            assert scraper.category_delay_seconds == 45

    def test_load_config_file_not_found(self):
        """Test config loading when file doesn't exist"""
        with patch("builtins.open", side_effect=FileNotFoundError):
            scraper = AmazonScraper(config_path="./nonexistent.json")
            assert scraper.config == {"categories": {}}

    @pytest.mark.asyncio
    async def test_initialize(self, scraper):
        """Test browser initialization"""
        mock_playwright = AsyncMock()
        mock_browser = AsyncMock()
        mock_chromium = AsyncMock()
        mock_chromium.launch = AsyncMock(return_value=mock_browser)
        mock_playwright.chromium = mock_chromium

        with patch("src.tools.scrapers.amazon_scraper.async_playwright") as mock_ap:
            mock_ap.return_value.start = AsyncMock(return_value=mock_playwright)
            await scraper.initialize()

            assert scraper.browser == mock_browser
            mock_chromium.launch.assert_called_once()

    @pytest.mark.asyncio
    async def test_close(self, scraper):
        """Test browser close"""
        mock_browser = AsyncMock()
        scraper.browser = mock_browser

        await scraper.close()
        mock_browser.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_close_no_browser(self, scraper):
        """Test close when browser is None"""
        scraper.browser = None
        await scraper.close()  # Should not raise

    def test_normalize_brand(self, scraper):
        """Test brand normalization"""
        assert scraper._normalize_brand("laneige") == "LANEIGE"
        assert scraper._normalize_brand("cosrx") == "COSRX"
        assert scraper._normalize_brand("burt's") == "Burt's Bees"
        assert scraper._normalize_brand("wet") == "wet n wild"
        assert scraper._normalize_brand("covergirl") == "COVERGIRL"
        assert scraper._normalize_brand("eos") == "eos"
        assert scraper._normalize_brand("Unknown") == "Unknown"
        assert scraper._normalize_brand("") == ""

    def test_extract_brand_from_asin_cache(self, scraper):
        """Test brand extraction from ASIN cache"""
        mock_resolver = Mock()
        mock_resolver.get_brand = Mock(return_value="LANEIGE")
        scraper.brand_resolver = mock_resolver

        brand = scraper._extract_brand("Some Product Name", asin="B084RGF8YJ")
        assert brand == "LANEIGE"
        mock_resolver.get_brand.assert_called_once_with("B084RGF8YJ")

    def test_extract_brand_multi_word(self, scraper):
        """Test multi-word brand extraction"""
        assert scraper._extract_brand("Beauty of Joseon Relief Sun") == "Beauty of Joseon"
        assert scraper._extract_brand("La Roche-Posay Toleriane") == "La Roche-Posay"
        assert scraper._extract_brand("Sol de Janeiro Bum Bum Cream") == "Sol de Janeiro"
        assert scraper._extract_brand("drunk elephant protini") == "Drunk Elephant"

    def test_extract_brand_single_word(self, scraper):
        """Test single-word brand extraction"""
        assert scraper._extract_brand("LANEIGE Lip Sleeping Mask") == "LANEIGE"
        assert scraper._extract_brand("laneige lip mask") == "LANEIGE"
        assert scraper._extract_brand("COSRX Snail Mucin") == "COSRX"
        assert scraper._extract_brand("CeraVe Moisturizing Cream") == "CeraVe"
        assert scraper._extract_brand("Neutrogena Hydro Boost") == "Neutrogena"

    def test_extract_brand_unknown(self, scraper):
        """Test brand extraction with unknown brand"""
        result = scraper._extract_brand("UnknownBrand Face Cream")
        assert result == "Unknown"

    def test_extract_brand_empty(self, scraper):
        """Test brand extraction with empty string"""
        assert scraper._extract_brand("") == "Unknown"

    def test_parse_price_valid(self, scraper):
        """Test price parsing with valid inputs"""
        assert scraper._parse_price("$24.99") == 24.99
        assert scraper._parse_price("$1,234.56") is None  # > 500
        assert scraper._parse_price("$0.99") == 0.99
        assert scraper._parse_price("$17.50") == 17.50
        assert scraper._parse_price("$499.99") == 499.99

    def test_parse_price_invalid(self, scraper):
        """Test price parsing with invalid inputs"""
        assert scraper._parse_price("") is None
        assert scraper._parse_price(None) is None
        assert scraper._parse_price("no price") is None
        assert scraper._parse_price("€24.99") is None  # No dollar sign
        assert scraper._parse_price("24.99") is None  # No dollar sign
        assert scraper._parse_price("$0.49") is None  # Below minimum
        assert scraper._parse_price("$501.00") is None  # Above maximum

    def test_parse_price_edge_cases(self, scraper):
        """Test price parsing edge cases"""
        assert scraper._parse_price("$0.50") == 0.50  # Minimum
        assert scraper._parse_price("$500.00") == 500.00  # Maximum
        assert scraper._parse_price("Price: $24.99") == 24.99
        assert scraper._parse_price("Was $24.99 now") == 24.99

    def test_parse_rating_valid(self, scraper):
        """Test rating parsing with valid inputs"""
        assert scraper._parse_rating("4.7 out of 5 stars") == 4.7
        assert scraper._parse_rating("4.8 out of 5") == 4.8
        assert scraper._parse_rating("3.5/5") == 3.5
        assert scraper._parse_rating("4.9 out of 5 stars") == 4.9
        assert scraper._parse_rating("5.0 out of 5") == 5.0

    def test_parse_rating_invalid(self, scraper):
        """Test rating parsing with invalid inputs"""
        assert scraper._parse_rating("") is None
        assert scraper._parse_rating(None) is None
        assert scraper._parse_rating("no rating") is None
        assert scraper._parse_rating("5.5 out of 5") is None  # Above max
        # Note: "-1 out of 5" extracts "1" which is valid (0-5 range)

    def test_parse_rating_edge_cases(self, scraper):
        """Test rating parsing edge cases"""
        assert scraper._parse_rating("0.0 out of 5") == 0.0
        assert scraper._parse_rating("5.0 out of 5") == 5.0
        assert scraper._parse_rating("4.5") == 4.5  # Fallback pattern

    def test_parse_reviews_count_valid(self, scraper):
        """Test reviews count parsing with valid inputs"""
        assert scraper._parse_reviews_count("1,234") == 1234
        assert scraper._parse_reviews_count("89,234") == 89234
        assert scraper._parse_reviews_count("123") == 123
        assert scraper._parse_reviews_count("999,999") == 999999

    def test_parse_reviews_count_invalid(self, scraper):
        """Test reviews count parsing with invalid inputs"""
        assert scraper._parse_reviews_count("") is None
        assert scraper._parse_reviews_count(None) is None
        assert scraper._parse_reviews_count("no reviews") is None
        assert scraper._parse_reviews_count("1,000,001") is None  # Above max

    def test_parse_reviews_count_edge_cases(self, scraper):
        """Test reviews count parsing edge cases"""
        assert scraper._parse_reviews_count("0") == 0
        assert scraper._parse_reviews_count("1,000,000") == 1000000  # Maximum
        assert scraper._parse_reviews_count("123+") == 123  # Plus sign stripped

    def test_get_page2_url(self, scraper):
        """Test page 2 URL generation"""
        url1 = "https://www.amazon.com/bestsellers/beauty"
        assert scraper._get_page2_url(url1) == "https://www.amazon.com/bestsellers/beauty?pg=2"

        url2 = "https://www.amazon.com/bestsellers/beauty?ref=zg_bs"
        assert (
            scraper._get_page2_url(url2)
            == "https://www.amazon.com/bestsellers/beauty?ref=zg_bs&pg=2"
        )

        url3 = "https://www.amazon.com/bestsellers/beauty?pg=2"
        assert scraper._get_page2_url(url3) == url3  # Already has pg=2

    @pytest.mark.asyncio
    async def test_is_blocked(self, scraper):
        """Test Amazon block detection"""
        mock_page = AsyncMock()

        # Not blocked
        mock_page.content = AsyncMock(return_value="<html>Normal page content</html>")
        assert await scraper._is_blocked(mock_page) is False

        # Blocked - CAPTCHA
        mock_page.content = AsyncMock(return_value="Enter the characters you see below")
        assert await scraper._is_blocked(mock_page) is True

        # Blocked - Robot check
        mock_page.content = AsyncMock(
            return_value="Sorry, we just need to make sure you're not a robot"
        )
        assert await scraper._is_blocked(mock_page) is True

        # Blocked - Image CAPTCHA
        mock_page.content = AsyncMock(return_value="Type the characters you see in this image")
        assert await scraper._is_blocked(mock_page) is True

        # Blocked - Support email
        mock_page.content = AsyncMock(return_value="Contact api-services-support@amazon.com")
        assert await scraper._is_blocked(mock_page) is True

    @pytest.mark.asyncio
    async def test_random_delay(self, scraper):
        """Test random delay"""
        scraper.delay_seconds = 0.01  # Fast test

        start = asyncio.get_event_loop().time()
        await scraper._random_delay()
        elapsed = asyncio.get_event_loop().time() - start

        assert elapsed >= 0.01  # At least base delay

    def test_get_random_user_agent(self, scraper):
        """Test random user agent selection"""
        ua1 = scraper._get_random_user_agent()
        assert "Mozilla" in ua1
        assert "Chrome" in ua1 or "Safari" in ua1 or "Edge" in ua1

        # Test randomness
        agents = [scraper._get_random_user_agent() for _ in range(10)]
        assert len(set(agents)) > 1  # Should have variety

    @pytest.mark.asyncio
    async def test_random_delay_advanced(self, scraper):
        """Test advanced random delay with different types"""
        scraper.delay_base_seconds = 0.001
        scraper.delay_random_max = 0.001
        scraper.category_delay_seconds = 0.005

        # Test that each delay type executes without error
        await scraper._random_delay_advanced("base")
        await scraper._random_delay_advanced("detail")
        await scraper._random_delay_advanced("page")
        await scraper._random_delay_advanced("category")

        # Verify delay type is handled (no exceptions raised)

    @pytest.mark.asyncio
    async def test_simulate_human_behavior(self, scraper):
        """Test human behavior simulation"""
        mock_page = AsyncMock()
        mock_page.evaluate = AsyncMock()
        mock_page.mouse.move = AsyncMock()

        await scraper._simulate_human_behavior(mock_page)

        # Should have called evaluate for scrolling
        assert mock_page.evaluate.call_count >= 1
        mock_page.mouse.move.assert_called_once()

    @pytest.mark.asyncio
    async def test_simulate_human_behavior_error(self, scraper):
        """Test human behavior simulation handles errors"""
        mock_page = AsyncMock()
        mock_page.evaluate = AsyncMock(side_effect=Exception("Page error"))

        # Should not raise
        await scraper._simulate_human_behavior(mock_page)

    @pytest.mark.asyncio
    async def test_scroll_to_load_all(self, scraper):
        """Test scrolling to load lazy-loaded products"""
        mock_page = AsyncMock()
        mock_page.evaluate = AsyncMock(return_value=50)  # 50 products loaded

        await scraper._scroll_to_load_all(mock_page)

        # Should have called evaluate multiple times
        assert mock_page.evaluate.call_count >= 2

    @pytest.mark.asyncio
    async def test_scroll_to_load_all_error(self, scraper):
        """Test scroll handles errors gracefully"""
        mock_page = AsyncMock()
        mock_page.evaluate = AsyncMock(side_effect=Exception("Scroll error"))

        # Should not raise
        await scraper._scroll_to_load_all(mock_page)

    @pytest.mark.asyncio
    async def test_create_stealth_context(self, scraper):
        """Test stealth context creation"""
        mock_browser = AsyncMock()
        mock_context = AsyncMock()
        mock_browser.new_context = AsyncMock(return_value=mock_context)
        scraper.browser = mock_browser

        context = await scraper._create_stealth_context()

        assert context == mock_context
        mock_browser.new_context.assert_called_once()

        # Check that user_agent was set
        call_kwargs = mock_browser.new_context.call_args[1]
        assert "user_agent" in call_kwargs
        assert "viewport" in call_kwargs
        assert call_kwargs["locale"] == "en-US"

    @pytest.mark.asyncio
    async def test_create_stealth_page(self, scraper):
        """Test stealth page creation"""
        mock_context = AsyncMock()
        mock_page = AsyncMock()
        mock_context.new_page = AsyncMock(return_value=mock_page)

        page = await scraper._create_stealth_page(mock_context)

        assert page == mock_page
        mock_context.new_page.assert_called_once()

    @pytest.mark.asyncio
    async def test_extract_product_data(self, scraper):
        """Test product data extraction from card"""
        mock_card = AsyncMock()

        # Mock name
        mock_name = AsyncMock()
        mock_name.inner_text = AsyncMock(return_value="LANEIGE Lip Sleeping Mask")
        mock_card.query_selector = AsyncMock(
            side_effect=lambda sel: {
                ".p13n-sc-truncate, ._cDEzb_p13n-sc-css-line-clamp-3_g3dy1, .a-link-normal span": mock_name,
                ".p13n-sc-price": AsyncMock(inner_text=AsyncMock(return_value="$24.00")),
                "span.a-icon-alt": AsyncMock(
                    text_content=AsyncMock(return_value="4.7 out of 5 stars")
                ),
                ".a-badge-text, .p13n-best-seller-badge": AsyncMock(
                    inner_text=AsyncMock(return_value="Best Seller")
                ),
                "a.a-link-normal": AsyncMock(
                    get_attribute=AsyncMock(return_value="/dp/B084RGF8YJ")
                ),
            }.get(sel)
        )

        mock_card.query_selector_all = AsyncMock(
            return_value=[AsyncMock(text_content=AsyncMock(return_value="13,265"))]
        )
        mock_card.inner_html = AsyncMock(return_value="<div>product html</div>")

        result = await scraper._extract_product_data(
            mock_card, "B084RGF8YJ", 1, "lip_care", "2026-02-17"
        )

        assert result is not None
        assert result["asin"] == "B084RGF8YJ"
        assert result["rank"] == 1
        assert result["product_name"] == "LANEIGE Lip Sleeping Mask"
        assert result["brand"] == "LANEIGE"
        assert result["price"] == 24.00
        assert result["rating"] == 4.7
        assert result["reviews_count"] == 13265
        assert result["category_id"] == "lip_care"

    @pytest.mark.asyncio
    async def test_extract_product_data_error(self, scraper):
        """Test product data extraction handles errors"""
        mock_card = AsyncMock()
        mock_card.query_selector = AsyncMock(side_effect=Exception("Parse error"))

        result = await scraper._extract_product_data(
            mock_card, "B084RGF8YJ", 1, "lip_care", "2026-02-17"
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_parse_bestseller_page(self, scraper):
        """Test bestseller page parsing"""
        mock_page = AsyncMock()
        mock_container = AsyncMock()
        mock_page.query_selector = AsyncMock(return_value=mock_container)

        # Mock product cards
        mock_card1 = AsyncMock()
        mock_card1.get_attribute = AsyncMock(return_value="B084RGF8YJ")
        mock_badge1 = AsyncMock()
        mock_badge1.inner_text = AsyncMock(return_value="#1")
        mock_card1.query_selector = AsyncMock(
            side_effect=lambda sel: {
                "span.zg-bdg-text": mock_badge1,
                ".p13n-sc-truncate, ._cDEzb_p13n-sc-css-line-clamp-3_g3dy1, .a-link-normal span": AsyncMock(
                    inner_text=AsyncMock(return_value="LANEIGE Lip Mask")
                ),
                ".p13n-sc-price": AsyncMock(inner_text=AsyncMock(return_value="$24.00")),
                "span.a-icon-alt": AsyncMock(
                    text_content=AsyncMock(return_value="4.7 out of 5 stars")
                ),
                "a.a-link-normal": AsyncMock(
                    get_attribute=AsyncMock(return_value="/dp/B084RGF8YJ")
                ),
            }.get(sel)
        )
        mock_card1.query_selector_all = AsyncMock(
            return_value=[AsyncMock(text_content=AsyncMock(return_value="1,234"))]
        )
        mock_card1.inner_html = AsyncMock(return_value="<div>card1</div>")

        mock_container.query_selector_all = AsyncMock(return_value=[mock_card1])

        products = await scraper._parse_bestseller_page(mock_page, "lip_care", start_rank=1)

        assert len(products) == 1
        assert products[0]["asin"] == "B084RGF8YJ"
        assert products[0]["rank"] == 1

    @pytest.mark.asyncio
    async def test_scrape_category_success(self, scraper):
        """Test successful category scraping"""
        mock_browser = AsyncMock()
        mock_context = AsyncMock()
        mock_page = AsyncMock()

        scraper.browser = mock_browser
        mock_browser.new_context = AsyncMock(return_value=mock_context)
        mock_context.new_page = AsyncMock(return_value=mock_page)
        mock_context.close = AsyncMock()

        mock_page.goto = AsyncMock()
        mock_page.content = AsyncMock(return_value="<html>Normal page</html>")

        # Mock container and products
        mock_container = AsyncMock()
        mock_card = AsyncMock()
        mock_card.get_attribute = AsyncMock(return_value="B084RGF8YJ")
        mock_badge = AsyncMock()
        mock_badge.inner_text = AsyncMock(return_value="#1")
        mock_card.query_selector = AsyncMock(
            side_effect=lambda sel: {
                "span.zg-bdg-text": mock_badge,
                ".p13n-sc-truncate, ._cDEzb_p13n-sc-css-line-clamp-3_g3dy1, .a-link-normal span": AsyncMock(
                    inner_text=AsyncMock(return_value="LANEIGE Lip Mask")
                ),
                ".p13n-sc-price": AsyncMock(inner_text=AsyncMock(return_value="$24.00")),
                "span.a-icon-alt": AsyncMock(
                    text_content=AsyncMock(return_value="4.7 out of 5 stars")
                ),
                "a.a-link-normal": AsyncMock(
                    get_attribute=AsyncMock(return_value="/dp/B084RGF8YJ")
                ),
            }.get(sel)
        )
        mock_card.query_selector_all = AsyncMock(
            return_value=[AsyncMock(text_content=AsyncMock(return_value="1,234"))]
        )
        mock_card.inner_html = AsyncMock(return_value="<div>card</div>")

        mock_container.query_selector_all = AsyncMock(return_value=[mock_card])
        mock_page.query_selector = AsyncMock(return_value=mock_container)
        mock_page.evaluate = AsyncMock(return_value=1)

        result = await scraper.scrape_category(
            "lip_care", "https://www.amazon.com/bestsellers/beauty/lip-care"
        )

        assert result["success"] is True
        assert result["category_id"] == "lip_care"
        assert result["category"] == "Lip Care"
        assert len(result["products"]) >= 1

    @pytest.mark.asyncio
    async def test_scrape_category_blocked(self, scraper):
        """Test category scraping when blocked"""
        mock_browser = AsyncMock()
        mock_context = AsyncMock()
        mock_page = AsyncMock()

        scraper.browser = mock_browser
        mock_browser.new_context = AsyncMock(return_value=mock_context)
        mock_context.new_page = AsyncMock(return_value=mock_page)
        mock_context.close = AsyncMock()

        mock_page.goto = AsyncMock()
        mock_page.content = AsyncMock(return_value="Enter the characters you see below")

        with patch("asyncio.sleep", new_callable=AsyncMock):
            result = await scraper.scrape_category(
                "lip_care", "https://www.amazon.com/bestsellers/beauty/lip-care"
            )

        assert result["success"] is False
        assert result["error"] == "BLOCKED"

    @pytest.mark.asyncio
    async def test_scrape_all_categories(self, scraper):
        """Test scraping all categories"""
        mock_browser = AsyncMock()
        scraper.browser = mock_browser

        # Mock scrape_category to return success
        async def mock_scrape_category(cat_id, url):
            return {
                "success": True,
                "count": 10,
                "products": [{"asin": f"B{i}"} for i in range(10)],
                "category_id": cat_id,
            }

        scraper.scrape_category = mock_scrape_category

        result = await scraper.scrape_all_categories()

        assert result["success_count"] == 2  # lip_care and skin_care
        assert result["error_count"] == 0
        assert result["total_products"] == 20

    @pytest.mark.asyncio
    async def test_scrape_product_by_asin_success(self, scraper):
        """Test scraping individual product by ASIN"""
        mock_browser = AsyncMock()
        mock_context = AsyncMock()
        mock_page = AsyncMock()

        scraper.browser = mock_browser
        scraper.circuit_breaker.can_proceed = Mock(return_value=True)

        with patch.object(scraper, "_create_stealth_context", return_value=mock_context):
            with patch.object(scraper, "_create_stealth_page", return_value=mock_page):
                mock_page.goto = AsyncMock()
                mock_page.evaluate = AsyncMock(
                    return_value={
                        "product_name": "LANEIGE Lip Sleeping Mask",
                        "brand": "LANEIGE",
                        "price_text": "$24.00",
                        "list_price_text": "$32.00",
                        "savings_percent": "25%",
                        "rating_text": "4.7 out of 5 stars",
                        "reviews_text": "13,265 ratings",
                        "availability": "In Stock",
                        "image_url": "https://m.media-amazon.com/images/test.jpg",
                        "coupon_text": "Save 5% with coupon",
                        "has_subscribe_save": True,
                        "promo_badges": "Deal",
                    }
                )
                mock_context.close = AsyncMock()

                with patch.object(scraper, "_is_blocked", return_value=False):
                    result = await scraper.scrape_product_by_asin("B084RGF8YJ")

                assert result is not None
                assert result["asin"] == "B084RGF8YJ"
                assert result["product_name"] == "LANEIGE Lip Sleeping Mask"
                assert result["brand"] == "LANEIGE"
                assert result["price"] == 24.00
                assert result["list_price"] == 32.00
                assert result["discount_percent"] == 25.0
                assert result["rating"] == 4.7
                assert result["coupon_text"] == "Save 5% with coupon"
                assert result["is_subscribe_save"] is True

    @pytest.mark.asyncio
    async def test_scrape_product_by_asin_blocked(self, scraper):
        """Test ASIN scraping when blocked"""
        mock_browser = AsyncMock()
        mock_context = AsyncMock()
        mock_page = AsyncMock()

        scraper.browser = mock_browser
        scraper.circuit_breaker.can_proceed = Mock(return_value=True)

        with patch.object(scraper, "_create_stealth_context", return_value=mock_context):
            with patch.object(scraper, "_create_stealth_page", return_value=mock_page):
                mock_page.goto = AsyncMock()
                mock_context.close = AsyncMock()

                with patch.object(scraper, "_is_blocked", return_value=True):
                    result = await scraper.scrape_product_by_asin("B084RGF8YJ")

                assert result is None

    @pytest.mark.asyncio
    async def test_scrape_product_by_asin_circuit_breaker_open(self, scraper):
        """Test ASIN scraping when circuit breaker is open"""
        scraper.browser = AsyncMock()
        scraper.circuit_breaker.can_proceed = Mock(return_value=False)

        result = await scraper.scrape_product_by_asin("B084RGF8YJ")

        assert result is None

    @pytest.mark.asyncio
    async def test_scrape_category_with_details(self, scraper):
        """Test category scraping with details"""

        # Mock scrape_category
        async def mock_scrape_category(cat_id, url):
            return {
                "success": True,
                "products": [
                    {"asin": "B084RGF8YJ", "price": 24.00},  # pragma: allowlist secret
                    {"asin": "B123456789", "price": 15.00},  # pragma: allowlist secret
                ],
                "category_id": cat_id,
            }

        # Mock scrape_product_by_asin
        async def mock_scrape_asin(asin, metadata=None):
            return {
                "list_price": 32.00,
                "discount_percent": 25.0,
                "coupon_text": "Save 5%",
                "promo_badges": "Deal",
                "is_subscribe_save": True,
            }

        scraper.scrape_category = mock_scrape_category
        scraper.scrape_product_by_asin = mock_scrape_asin
        scraper.circuit_breaker.can_proceed = Mock(return_value=True)

        result = await scraper.scrape_category_with_details(
            "lip_care", "https://www.amazon.com/bestsellers/beauty/lip-care"
        )

        assert result["success"] is True
        assert len(result["products"]) == 2
        assert result["products"][0]["list_price"] == 32.00
        assert result["products"][0]["discount_percent"] == 25.0

    @pytest.mark.asyncio
    async def test_scrape_competitor_products(self, scraper):
        """Test competitor products scraping"""
        competitor_config = {
            "brand_name": "COSRX",
            "products": [
                {"asin": "B123", "category": "Skin Care", "product_type": "Serum"},
                {"asin": "B456", "category": "Skin Care", "product_type": "Cleanser"},
            ],
        }

        async def mock_scrape_asin(asin, metadata=None):
            return {
                "asin": asin,
                "product_name": f"Product {asin}",
                "brand": metadata.get("brand") if metadata else "Unknown",
                "price": 20.00,
            }

        scraper.scrape_product_by_asin = mock_scrape_asin
        scraper._random_delay = AsyncMock()

        results = await scraper.scrape_competitor_products(competitor_config)

        assert len(results) == 2
        assert results[0]["asin"] == "B123"
        assert results[0]["brand"] == "COSRX"
        assert results[1]["asin"] == "B456"


class TestScrapeBestsellersFunction:
    """Test standalone scrape_bestsellers function"""

    @pytest.mark.asyncio
    async def test_scrape_bestsellers(self, mock_config):
        """Test scrape_bestsellers convenience function"""
        from src.tools.scrapers.amazon_scraper import scrape_bestsellers

        with patch("builtins.open", mock_open(read_data=json.dumps(mock_config))):
            with patch("src.tools.scrapers.amazon_scraper.AmazonScraper") as MockScraper:
                mock_instance = AsyncMock()
                mock_instance.initialize = AsyncMock()
                mock_instance.scrape_category = AsyncMock(
                    return_value={
                        "success": True,
                        "products": [{"asin": "B084RGF8YJ"}],
                        "count": 1,
                    }
                )
                mock_instance.close = AsyncMock()
                MockScraper.return_value = mock_instance

                result = await scrape_bestsellers(
                    "https://www.amazon.com/bestsellers/beauty", "beauty"
                )

                assert result["success"] is True
                assert result["count"] == 1
                mock_instance.initialize.assert_called_once()
                mock_instance.close.assert_called_once()
