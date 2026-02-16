"""Unit tests for src/tools/utilities/brand_resolver.py"""

import json
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.tools.utilities.brand_resolver import BrandResolver, get_brand_resolver


@pytest.fixture
def mock_mapping_data():
    """Sample mapping data"""
    return {
        "version": "1.0",
        "updated_at": "2026-02-17T10:00:00",
        "description": "Test mappings",
        "mappings": {
            "B0C42HJRBF": {
                "brand": "LANEIGE",
                "verified": True,
                "verified_date": "2026-02-17",
                "product_name": "Lip Sleeping Mask",
            },
            "B0TESTID01": {
                "brand": "Test Brand",
                "verified": True,
                "verified_date": "2026-02-16",
                "product_name": "Test Product",
            },
        },
    }


@pytest.fixture
def temp_mapping_file(tmp_path, mock_mapping_data):
    """Create a temporary mapping file"""
    file_path = tmp_path / "config" / "asin_brand_mapping.json"
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(mock_mapping_data, f)
    return str(file_path)


class TestBrandResolverInit:
    """Test initialization"""

    def test_init_with_existing_file(self, temp_mapping_file, mock_mapping_data):
        """Test init loads existing mapping file"""
        resolver = BrandResolver(mapping_path=temp_mapping_file)
        assert len(resolver.mappings) == 2
        assert "B0C42HJRBF" in resolver.mappings
        assert resolver.mappings["B0C42HJRBF"]["brand"] == "LANEIGE"

    def test_init_with_missing_file(self, tmp_path):
        """Test init with non-existent file creates empty mappings"""
        file_path = tmp_path / "nonexistent.json"
        resolver = BrandResolver(mapping_path=str(file_path))
        assert resolver.mappings == {}

    def test_init_with_invalid_json(self, tmp_path):
        """Test init handles invalid JSON gracefully"""
        file_path = tmp_path / "invalid.json"
        file_path.write_text("not valid json {", encoding="utf-8")
        resolver = BrandResolver(mapping_path=str(file_path))
        assert resolver.mappings == {}


class TestLoadMappings:
    """Test _load_mappings method"""

    def test_load_mappings_success(self, temp_mapping_file):
        """Test successful loading"""
        resolver = BrandResolver(mapping_path=temp_mapping_file)
        assert len(resolver.mappings) == 2
        assert "B0TESTID01" in resolver.mappings

    def test_load_mappings_missing_mappings_key(self, tmp_path):
        """Test loading file without 'mappings' key"""
        file_path = tmp_path / "no_mappings.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump({"version": "1.0"}, f)
        resolver = BrandResolver(mapping_path=str(file_path))
        assert resolver.mappings == {}

    def test_load_mappings_io_error(self, tmp_path):
        """Test handling of I/O errors"""
        file_path = tmp_path / "test.json"
        resolver = BrandResolver.__new__(BrandResolver)
        resolver.mapping_path = Path(file_path)
        resolver.mappings = {}

        with patch("builtins.open", side_effect=OSError("Read error")):
            resolver._load_mappings()
            assert resolver.mappings == {}


class TestSaveMappings:
    """Test _save_mappings method"""

    def test_save_mappings_creates_directory(self, tmp_path):
        """Test save creates parent directory if missing"""
        file_path = tmp_path / "new_dir" / "mappings.json"
        resolver = BrandResolver(mapping_path=str(file_path))
        resolver.mappings = {"TEST": {"brand": "TestBrand", "verified": True}}
        resolver._save_mappings()

        assert file_path.exists()
        with open(file_path, encoding="utf-8") as f:
            data = json.load(f)
        assert "mappings" in data
        assert data["mappings"]["TEST"]["brand"] == "TestBrand"

    def test_save_mappings_includes_metadata(self, tmp_path):
        """Test save includes version and timestamp"""
        file_path = tmp_path / "mappings.json"
        resolver = BrandResolver(mapping_path=str(file_path))
        resolver.mappings = {"A": {"brand": "B"}}
        resolver._save_mappings()

        with open(file_path, encoding="utf-8") as f:
            data = json.load(f)
        assert data["version"] == "1.0"
        assert "updated_at" in data
        assert "description" in data

    def test_save_mappings_handles_error(self, tmp_path):
        """Test save handles write errors gracefully"""
        file_path = tmp_path / "test.json"
        resolver = BrandResolver(mapping_path=str(file_path))
        resolver.mappings = {"A": {"brand": "B"}}

        with patch("builtins.open", side_effect=OSError("Write error")):
            resolver._save_mappings()  # Should not raise


class TestGetBrand:
    """Test get_brand method"""

    def test_get_brand_found(self, temp_mapping_file):
        """Test retrieving existing brand"""
        resolver = BrandResolver(mapping_path=temp_mapping_file)
        brand = resolver.get_brand("B0C42HJRBF")
        assert brand == "LANEIGE"

    def test_get_brand_not_found(self, temp_mapping_file):
        """Test retrieving non-existent ASIN"""
        resolver = BrandResolver(mapping_path=temp_mapping_file)
        brand = resolver.get_brand("NONEXISTENT")
        assert brand is None

    def test_get_brand_empty_mappings(self, tmp_path):
        """Test get_brand with empty mappings"""
        file_path = tmp_path / "empty.json"
        resolver = BrandResolver(mapping_path=str(file_path))
        assert resolver.get_brand("ANY") is None


class TestAddMapping:
    """Test add_mapping method"""

    def test_add_mapping_with_auto_save(self, tmp_path):
        """Test adding mapping with auto-save enabled"""
        file_path = tmp_path / "mappings.json"
        resolver = BrandResolver(mapping_path=str(file_path))
        resolver.add_mapping("B0NEW", "NewBrand", "New Product", auto_save=True)

        assert "B0NEW" in resolver.mappings
        assert resolver.mappings["B0NEW"]["brand"] == "NewBrand"
        assert resolver.mappings["B0NEW"]["verified"] is True
        assert resolver.mappings["B0NEW"]["product_name"] == "New Product"
        assert file_path.exists()

    def test_add_mapping_without_auto_save(self, tmp_path):
        """Test adding mapping without auto-save"""
        file_path = tmp_path / "mappings.json"
        resolver = BrandResolver(mapping_path=str(file_path))
        resolver.add_mapping("B0NEW", "NewBrand", "New Product", auto_save=False)

        assert "B0NEW" in resolver.mappings
        assert not file_path.exists()

    def test_add_mapping_includes_date(self, tmp_path):
        """Test add_mapping includes verified_date"""
        file_path = tmp_path / "mappings.json"
        resolver = BrandResolver(mapping_path=str(file_path))
        today = datetime.now().strftime("%Y-%m-%d")
        resolver.add_mapping("B0NEW", "Brand", auto_save=False)

        assert resolver.mappings["B0NEW"]["verified_date"] == today

    def test_add_mapping_overwrites_existing(self, temp_mapping_file):
        """Test add_mapping overwrites existing entry"""
        resolver = BrandResolver(mapping_path=temp_mapping_file)
        resolver.add_mapping("B0C42HJRBF", "Updated Brand", auto_save=False)
        assert resolver.mappings["B0C42HJRBF"]["brand"] == "Updated Brand"


class TestAddMappingsBatch:
    """Test add_mappings_batch method"""

    def test_add_mappings_batch_success(self, tmp_path):
        """Test batch adding multiple mappings"""
        file_path = tmp_path / "mappings.json"
        resolver = BrandResolver(mapping_path=str(file_path))

        batch = [
            {"asin": "A1", "brand": "Brand1", "product_name": "Product 1"},
            {"asin": "A2", "brand": "Brand2", "product_name": "Product 2"},
            {"asin": "A3", "brand": "Brand3"},
        ]

        count = resolver.add_mappings_batch(batch)

        assert count == 3
        assert len(resolver.mappings) == 3
        assert resolver.mappings["A1"]["brand"] == "Brand1"
        assert resolver.mappings["A3"]["product_name"] == ""
        assert file_path.exists()

    def test_add_mappings_batch_skips_invalid(self, tmp_path):
        """Test batch skips entries without ASIN or brand"""
        file_path = tmp_path / "mappings.json"
        resolver = BrandResolver(mapping_path=str(file_path))

        batch = [
            {"asin": "A1", "brand": "Brand1"},
            {"asin": "A2"},  # Missing brand
            {"brand": "Brand3"},  # Missing ASIN
            {"asin": "", "brand": "Brand4"},  # Empty ASIN
        ]

        count = resolver.add_mappings_batch(batch)
        assert count == 1
        assert len(resolver.mappings) == 1

    def test_add_mappings_batch_empty_list(self, tmp_path):
        """Test batch with empty list"""
        file_path = tmp_path / "mappings.json"
        resolver = BrandResolver(mapping_path=str(file_path))
        count = resolver.add_mappings_batch([])
        assert count == 0
        assert not file_path.exists()


class TestExtractBrandFromText:
    """Test _extract_brand_from_text method"""

    def test_extract_brand_visit_store_pattern(self, tmp_path):
        """Test extracting brand from 'Visit the X Store' pattern"""
        resolver = BrandResolver(mapping_path=str(tmp_path / "test.json"))
        brand = resolver._extract_brand_from_text("Visit the LANEIGE Store")
        assert brand == "LANEIGE"

    def test_extract_brand_visit_store_case_insensitive(self, tmp_path):
        """Test case-insensitive matching"""
        resolver = BrandResolver(mapping_path=str(tmp_path / "test.json"))
        brand = resolver._extract_brand_from_text("visit the CeraVe store")
        assert brand == "CeraVe"

    def test_extract_brand_from_brand_prefix(self, tmp_path):
        """Test extracting brand from 'Brand: X' pattern"""
        resolver = BrandResolver(mapping_path=str(tmp_path / "test.json"))
        brand = resolver._extract_brand_from_text("Brand: Neutrogena")
        assert brand == "Neutrogena"

    def test_extract_brand_direct_text(self, tmp_path):
        """Test extracting brand from direct text"""
        resolver = BrandResolver(mapping_path=str(tmp_path / "test.json"))
        brand = resolver._extract_brand_from_text("CeraVe")
        assert brand == "CeraVe"

    def test_extract_brand_ignores_long_text(self, tmp_path):
        """Test ignoring long text that's not a brand"""
        resolver = BrandResolver(mapping_path=str(tmp_path / "test.json"))
        long_text = "This is a very long product description that exceeds fifty characters"
        brand = resolver._extract_brand_from_text(long_text)
        assert brand is None

    def test_extract_brand_empty_text(self, tmp_path):
        """Test handling empty text"""
        resolver = BrandResolver(mapping_path=str(tmp_path / "test.json"))
        assert resolver._extract_brand_from_text("") is None
        assert resolver._extract_brand_from_text(None) is None

    def test_extract_brand_strips_whitespace(self, tmp_path):
        """Test whitespace stripping"""
        resolver = BrandResolver(mapping_path=str(tmp_path / "test.json"))
        brand = resolver._extract_brand_from_text("  Visit the  LANEIGE  Store  ")
        assert brand == "LANEIGE"


class TestSearchBrandWeb:
    """Test _search_brand_web method"""

    @pytest.mark.asyncio
    async def test_search_brand_web_known_pattern(self, tmp_path):
        """Test web search with known pattern"""
        resolver = BrandResolver(mapping_path=str(tmp_path / "test.json"))
        brand = await resolver._search_brand_web("Summer Fridays Jet Lag Mask")
        assert brand == "Summer Fridays"

    @pytest.mark.asyncio
    async def test_search_brand_web_case_insensitive(self, tmp_path):
        """Test case-insensitive pattern matching"""
        resolver = BrandResolver(mapping_path=str(tmp_path / "test.json"))
        brand = await resolver._search_brand_web("DRUNK ELEPHANT Vitamin C Serum")
        assert brand == "Drunk Elephant"

    @pytest.mark.asyncio
    async def test_search_brand_web_unknown_pattern(self, tmp_path):
        """Test web search with unknown brand"""
        resolver = BrandResolver(mapping_path=str(tmp_path / "test.json"))
        brand = await resolver._search_brand_web("Unknown Brand Product Name")
        assert brand is None

    @pytest.mark.asyncio
    async def test_search_brand_web_multiple_patterns(self, tmp_path):
        """Test matching first found pattern"""
        resolver = BrandResolver(mapping_path=str(tmp_path / "test.json"))
        brand = await resolver._search_brand_web("The Ordinary Niacinamide")
        assert brand == "The Ordinary"

    @pytest.mark.asyncio
    async def test_search_brand_web_handles_error(self, tmp_path):
        """Test error handling in web search"""
        resolver = BrandResolver(mapping_path=str(tmp_path / "test.json"))
        # Pass invalid input to trigger error path
        brand = await resolver._search_brand_web("")
        assert brand is None


class TestFetchBrandFromAmazon:
    """Test _fetch_brand_from_amazon method"""

    @pytest.mark.asyncio
    async def test_fetch_brand_playwright_success(self, tmp_path):
        """Test successful brand fetch from Amazon"""
        resolver = BrandResolver(mapping_path=str(tmp_path / "test.json"))

        mock_element = AsyncMock()
        mock_element.inner_text = AsyncMock(return_value="Visit the LANEIGE Store")

        mock_page = AsyncMock()
        mock_page.goto = AsyncMock()
        mock_page.wait_for_load_state = AsyncMock()
        mock_page.query_selector = AsyncMock(return_value=mock_element)

        mock_context = AsyncMock()
        mock_context.new_page = AsyncMock(return_value=mock_page)

        mock_browser = AsyncMock()
        mock_browser.new_context = AsyncMock(return_value=mock_context)
        mock_browser.close = AsyncMock()

        mock_playwright = AsyncMock()
        mock_playwright.chromium.launch = AsyncMock(return_value=mock_browser)

        with patch("playwright.async_api.async_playwright") as mock_pw:
            mock_pw.return_value.__aenter__.return_value = mock_playwright
            brand = await resolver._fetch_brand_from_amazon("B0C42HJRBF")

        assert brand == "LANEIGE"

    @pytest.mark.asyncio
    async def test_fetch_brand_multiple_selectors(self, tmp_path):
        """Test trying multiple selectors"""
        resolver = BrandResolver(mapping_path=str(tmp_path / "test.json"))

        mock_element = AsyncMock()
        mock_element.inner_text = AsyncMock(return_value="CeraVe")

        mock_page = AsyncMock()
        mock_page.goto = AsyncMock()
        mock_page.wait_for_load_state = AsyncMock()
        # First selector returns None, second returns element
        mock_page.query_selector = AsyncMock(side_effect=[None, mock_element])

        mock_context = AsyncMock()
        mock_context.new_page = AsyncMock(return_value=mock_page)

        mock_browser = AsyncMock()
        mock_browser.new_context = AsyncMock(return_value=mock_context)
        mock_browser.close = AsyncMock()

        mock_playwright = AsyncMock()
        mock_playwright.chromium.launch = AsyncMock(return_value=mock_browser)

        with patch("playwright.async_api.async_playwright") as mock_pw:
            mock_pw.return_value.__aenter__.return_value = mock_playwright
            brand = await resolver._fetch_brand_from_amazon("B0TEST")

        assert brand == "CeraVe"

    @pytest.mark.asyncio
    async def test_fetch_brand_playwright_not_available(self, tmp_path):
        """Test handling missing Playwright"""
        resolver = BrandResolver(mapping_path=str(tmp_path / "test.json"))

        with patch("playwright.async_api.async_playwright", side_effect=ImportError):
            brand = await resolver._fetch_brand_from_amazon("B0TEST")

        assert brand is None

    @pytest.mark.asyncio
    async def test_fetch_brand_page_load_error(self, tmp_path):
        """Test handling page load errors"""
        resolver = BrandResolver(mapping_path=str(tmp_path / "test.json"))

        mock_page = AsyncMock()
        mock_page.goto = AsyncMock(side_effect=Exception("Timeout"))

        mock_context = AsyncMock()
        mock_context.new_page = AsyncMock(return_value=mock_page)

        mock_browser = AsyncMock()
        mock_browser.new_context = AsyncMock(return_value=mock_context)
        mock_browser.close = AsyncMock()

        mock_playwright = AsyncMock()
        mock_playwright.chromium.launch = AsyncMock(return_value=mock_browser)

        with patch("playwright.async_api.async_playwright") as mock_pw:
            mock_pw.return_value.__aenter__.return_value = mock_playwright
            brand = await resolver._fetch_brand_from_amazon("B0TEST")

        assert brand is None
        mock_browser.close.assert_called_once()


class TestVerifyUnknownBrands:
    """Test verify_unknown_brands method"""

    @pytest.mark.asyncio
    async def test_verify_unknown_brands_success(self, tmp_path):
        """Test successful verification of unknown brands"""
        resolver = BrandResolver(mapping_path=str(tmp_path / "test.json"))

        products = [
            {"asin": "A1", "brand": "Unknown", "product_name": "Product 1"},
            {"asin": "A2", "brand": "LANEIGE", "product_name": "Product 2"},
            {"asin": "A3", "brand": "Unknown", "product_name": "Product 3"},
        ]

        with patch.object(
            resolver, "_fetch_brand_from_amazon", AsyncMock(return_value="TestBrand")
        ):
            result = await resolver.verify_unknown_brands(
                products, use_websearch=False, delay_seconds=0
            )

        assert result["verified_count"] == 2
        assert result["failed_count"] == 0
        assert result["skipped_count"] == 1
        assert len(result["results"]) == 2

    @pytest.mark.asyncio
    async def test_verify_unknown_brands_with_websearch_fallback(self, tmp_path):
        """Test fallback to web search"""
        resolver = BrandResolver(mapping_path=str(tmp_path / "test.json"))

        products = [{"asin": "A1", "brand": "Unknown", "product_name": "Summer Fridays Product"}]

        with patch.object(resolver, "_fetch_brand_from_amazon", AsyncMock(return_value=None)):
            with patch.object(
                resolver, "_search_brand_web", AsyncMock(return_value="Summer Fridays")
            ):
                result = await resolver.verify_unknown_brands(products, delay_seconds=0)

        assert result["verified_count"] == 1
        assert result["results"][0]["source"] == "web_search"

    @pytest.mark.asyncio
    async def test_verify_unknown_brands_skips_mapped(self, tmp_path):
        """Test skipping already mapped ASINs"""
        resolver = BrandResolver(mapping_path=str(tmp_path / "test.json"))
        resolver.mappings["A1"] = {"brand": "Cached Brand"}

        products = [
            {"asin": "A1", "brand": "Unknown", "product_name": "Product 1"},
            {"asin": "A2", "brand": "Unknown", "product_name": "Product 2"},
        ]

        with patch.object(resolver, "_fetch_brand_from_amazon", AsyncMock(return_value="NewBrand")):
            result = await resolver.verify_unknown_brands(
                products, use_websearch=False, delay_seconds=0
            )

        assert result["verified_count"] == 1
        assert result["skipped_count"] == 1  # A1 is already mapped, so it's skipped

    @pytest.mark.asyncio
    async def test_verify_unknown_brands_handles_failures(self, tmp_path):
        """Test handling verification failures"""
        resolver = BrandResolver(mapping_path=str(tmp_path / "test.json"))

        products = [
            {"asin": "A1", "brand": "Unknown", "product_name": "Product 1"},
            {"asin": "A2", "brand": "Unknown", "product_name": "Product 2"},
        ]

        with patch.object(resolver, "_fetch_brand_from_amazon", AsyncMock(return_value=None)):
            with patch.object(resolver, "_search_brand_web", AsyncMock(return_value=None)):
                result = await resolver.verify_unknown_brands(products, delay_seconds=0)

        assert result["verified_count"] == 0
        assert result["failed_count"] == 2

    @pytest.mark.asyncio
    async def test_verify_unknown_brands_no_amazon(self, tmp_path):
        """Test skipping Amazon fetch"""
        resolver = BrandResolver(mapping_path=str(tmp_path / "test.json"))

        products = [{"asin": "A1", "brand": "Unknown", "product_name": "Rare Beauty Product"}]

        with patch.object(resolver, "_fetch_brand_from_amazon", AsyncMock()) as mock_amazon:
            with patch.object(resolver, "_search_brand_web", AsyncMock(return_value="Rare Beauty")):
                result = await resolver.verify_unknown_brands(
                    products, use_amazon=False, delay_seconds=0
                )

        mock_amazon.assert_not_called()
        assert result["verified_count"] == 1


class TestExtractBrandWithLLM:
    """Test extract_brand_with_llm method"""

    @pytest.mark.asyncio
    async def test_extract_brand_with_llm_success(self, tmp_path):
        """Test successful LLM brand extraction"""
        resolver = BrandResolver(mapping_path=str(tmp_path / "test.json"))

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"brand": "LANEIGE"}'

        with patch("litellm.acompletion", AsyncMock(return_value=mock_response)):
            brand = await resolver.extract_brand_with_llm(
                "LANEIGE Lip Sleeping Mask", asin="B0C42HJRBF"
            )

        assert brand == "LANEIGE"
        assert "B0C42HJRBF" in resolver.mappings

    @pytest.mark.asyncio
    async def test_extract_brand_with_llm_no_asin(self, tmp_path):
        """Test LLM extraction without ASIN caching"""
        resolver = BrandResolver(mapping_path=str(tmp_path / "test.json"))

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"brand": "CeraVe"}'

        with patch("litellm.acompletion", AsyncMock(return_value=mock_response)):
            brand = await resolver.extract_brand_with_llm("CeraVe Moisturizing Cream")

        assert brand == "CeraVe"
        assert len(resolver.mappings) == 0

    @pytest.mark.asyncio
    async def test_extract_brand_with_llm_unknown_result(self, tmp_path):
        """Test handling 'Unknown' from LLM"""
        resolver = BrandResolver(mapping_path=str(tmp_path / "test.json"))

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"brand": "Unknown"}'

        with patch("litellm.acompletion", AsyncMock(return_value=mock_response)):
            brand = await resolver.extract_brand_with_llm("Generic Product")

        assert brand is None

    @pytest.mark.asyncio
    async def test_extract_brand_with_llm_invalid_json(self, tmp_path):
        """Test handling invalid JSON from LLM"""
        resolver = BrandResolver(mapping_path=str(tmp_path / "test.json"))

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "not valid json"

        with patch("litellm.acompletion", AsyncMock(return_value=mock_response)):
            brand = await resolver.extract_brand_with_llm("Product Name")

        assert brand is None

    @pytest.mark.asyncio
    async def test_extract_brand_with_llm_litellm_not_available(self, tmp_path):
        """Test handling missing LiteLLM"""
        resolver = BrandResolver(mapping_path=str(tmp_path / "test.json"))

        with patch("litellm.acompletion", side_effect=ImportError):
            brand = await resolver.extract_brand_with_llm("Product Name")

        assert brand is None

    @pytest.mark.asyncio
    async def test_extract_brand_with_llm_filters_long_brand(self, tmp_path):
        """Test filtering out unreasonably long brand names"""
        resolver = BrandResolver(mapping_path=str(tmp_path / "test.json"))

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"brand": "' + "A" * 100 + '"}'

        with patch("litellm.acompletion", AsyncMock(return_value=mock_response)):
            brand = await resolver.extract_brand_with_llm("Product Name")

        assert brand is None

    @pytest.mark.asyncio
    async def test_extract_brand_with_llm_json_embedded_in_text(self, tmp_path):
        """Test extracting JSON from response with extra text"""
        resolver = BrandResolver(mapping_path=str(tmp_path / "test.json"))

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[
            0
        ].message.content = 'Sure! Here is the result: {"brand": "TestBrand"} - Hope this helps!'

        with patch("litellm.acompletion", AsyncMock(return_value=mock_response)):
            brand = await resolver.extract_brand_with_llm("Product Name")

        assert brand == "TestBrand"


class TestVerifyBrandsWithLLM:
    """Test verify_brands_with_llm method"""

    @pytest.mark.asyncio
    async def test_verify_brands_with_llm_success(self, tmp_path):
        """Test successful LLM batch verification"""
        resolver = BrandResolver(mapping_path=str(tmp_path / "test.json"))

        products = [
            {"asin": "A1", "brand": "Unknown", "product_name": "Product 1"},
            {"asin": "A2", "brand": "", "product_name": "Product 2"},
            {"asin": "A3", "brand": "Known Brand", "product_name": "Product 3"},
        ]

        with patch.object(
            resolver, "extract_brand_with_llm", AsyncMock(return_value="ExtractedBrand")
        ):
            result = await resolver.verify_brands_with_llm(products, delay_seconds=0)

        assert result["verified_count"] == 2
        assert result["failed_count"] == 0
        assert result["skipped_count"] == 1
        assert len(result["results"]) == 2

    @pytest.mark.asyncio
    async def test_verify_brands_with_llm_skips_mapped(self, tmp_path):
        """Test skipping already mapped products"""
        resolver = BrandResolver(mapping_path=str(tmp_path / "test.json"))
        resolver.mappings["A1"] = {"brand": "Cached"}

        products = [
            {"asin": "A1", "brand": "Unknown", "product_name": "Product 1"},
            {"asin": "A2", "brand": "Unknown", "product_name": "Product 2"},
        ]

        with patch.object(resolver, "extract_brand_with_llm", AsyncMock(return_value="NewBrand")):
            result = await resolver.verify_brands_with_llm(products, delay_seconds=0)

        assert result["verified_count"] == 1
        assert result["skipped_count"] == 1

    @pytest.mark.asyncio
    async def test_verify_brands_with_llm_updates_products(self, tmp_path):
        """Test updating products list with cached brands"""
        resolver = BrandResolver(mapping_path=str(tmp_path / "test.json"))

        products = [
            {"asin": "A1", "brand": "Unknown", "product_name": "Product 1"},
            {"asin": "A2", "brand": "Unknown", "product_name": "Product 2"},
        ]

        async def mock_extract(name, asin):
            resolver.mappings[asin] = {"brand": f"Brand{asin}", "verified": True}
            return f"Brand{asin}"

        with patch.object(resolver, "extract_brand_with_llm", side_effect=mock_extract):
            result = await resolver.verify_brands_with_llm(products, delay_seconds=0)

        updated = result["updated_products"]
        assert updated[0]["brand"] == "BrandA1"
        assert updated[1]["brand"] == "BrandA2"

    @pytest.mark.asyncio
    async def test_verify_brands_with_llm_handles_empty_product_name(self, tmp_path):
        """Test handling products without product_name"""
        resolver = BrandResolver(mapping_path=str(tmp_path / "test.json"))

        products = [
            {"asin": "A1", "brand": "Unknown", "product_name": ""},
            {"asin": "A2", "brand": "Unknown"},
        ]

        with patch.object(resolver, "extract_brand_with_llm", AsyncMock()) as mock_extract:
            result = await resolver.verify_brands_with_llm(products, delay_seconds=0)

        mock_extract.assert_not_called()
        assert result["failed_count"] == 2

    @pytest.mark.asyncio
    async def test_verify_brands_with_llm_handles_extraction_failure(self, tmp_path):
        """Test handling LLM extraction failures"""
        resolver = BrandResolver(mapping_path=str(tmp_path / "test.json"))

        products = [
            {"asin": "A1", "brand": "Unknown", "product_name": "Product 1"},
            {"asin": "A2", "brand": "Unknown", "product_name": "Product 2"},
        ]

        with patch.object(resolver, "extract_brand_with_llm", AsyncMock(return_value=None)):
            result = await resolver.verify_brands_with_llm(products, delay_seconds=0)

        assert result["verified_count"] == 0
        assert result["failed_count"] == 2


class TestGetStats:
    """Test get_stats method"""

    def test_get_stats_with_data(self, temp_mapping_file):
        """Test statistics with existing data"""
        resolver = BrandResolver(mapping_path=temp_mapping_file)
        # Add more mappings to test counting
        resolver.mappings["B0TEST02"] = {"brand": "LANEIGE", "verified": True}
        resolver.mappings["B0TEST03"] = {"brand": "CeraVe", "verified": True}

        stats = resolver.get_stats()

        assert stats["total_mappings"] == 4
        assert stats["unique_brands"] == 3
        assert len(stats["top_brands"]) <= 10
        # LANEIGE appears twice
        assert ("LANEIGE", 2) in stats["top_brands"]

    def test_get_stats_empty_mappings(self, tmp_path):
        """Test statistics with no mappings"""
        resolver = BrandResolver(mapping_path=str(tmp_path / "test.json"))
        stats = resolver.get_stats()

        assert stats["total_mappings"] == 0
        assert stats["unique_brands"] == 0
        assert stats["top_brands"] == []

    def test_get_stats_top_brands_sorted(self, tmp_path):
        """Test top brands are sorted by count descending"""
        resolver = BrandResolver(mapping_path=str(tmp_path / "test.json"))

        # Add brands with different counts
        for i in range(5):
            resolver.mappings[f"A{i}"] = {"brand": "BrandA"}
        for i in range(3):
            resolver.mappings[f"B{i}"] = {"brand": "BrandB"}
        resolver.mappings["C0"] = {"brand": "BrandC"}

        stats = resolver.get_stats()

        assert stats["top_brands"][0] == ("BrandA", 5)
        assert stats["top_brands"][1] == ("BrandB", 3)
        assert stats["top_brands"][2] == ("BrandC", 1)


class TestGetBrandResolverSingleton:
    """Test get_brand_resolver singleton function"""

    def test_get_brand_resolver_returns_instance(self):
        """Test singleton returns BrandResolver instance"""
        # Clear any existing singleton
        import src.tools.utilities.brand_resolver as resolver_module

        resolver_module._resolver_instance = None

        resolver = get_brand_resolver()
        assert isinstance(resolver, BrandResolver)

    def test_get_brand_resolver_returns_same_instance(self):
        """Test singleton returns same instance on repeated calls"""
        import src.tools.utilities.brand_resolver as resolver_module

        resolver_module._resolver_instance = None

        resolver1 = get_brand_resolver()
        resolver2 = get_brand_resolver()
        assert resolver1 is resolver2

    def test_get_brand_resolver_uses_default_path(self):
        """Test singleton uses default mapping path"""
        import src.tools.utilities.brand_resolver as resolver_module

        resolver_module._resolver_instance = None

        resolver = get_brand_resolver()
        assert str(resolver.mapping_path).endswith("asin_brand_mapping.json")
