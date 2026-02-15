"""
Unit tests for StorageAgent
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agents.storage_agent import StorageAgent


@pytest.fixture
def mock_sheets():
    """Mock SheetsWriter"""
    sheets = AsyncMock()
    sheets.append_rank_records = AsyncMock(return_value={"success": True})
    sheets.upsert_products_batch = AsyncMock(return_value={"created": 2, "updated": 1})
    sheets._append_row = MagicMock()
    sheets.get_rank_history = MagicMock(return_value=[])
    return sheets


@pytest.fixture
def mock_sqlite():
    """Mock SQLiteStorage"""
    sqlite = AsyncMock()
    sqlite.initialize = AsyncMock()
    sqlite.append_rank_records = AsyncMock(return_value={"success": True, "rows_added": 3})
    sqlite.save_competitor_products = AsyncMock(return_value={"success": True, "rows_added": 2})
    return sqlite


@pytest.fixture
def agent(mock_sheets, mock_sqlite):
    """Create StorageAgent with mocked dependencies"""
    with (
        patch("src.agents.storage_agent.SheetsWriter", return_value=mock_sheets),
        patch("src.agents.storage_agent.get_sqlite_storage", return_value=mock_sqlite),
    ):
        a = StorageAgent(spreadsheet_id="test-id", enable_sqlite=True)
        a.sheets = mock_sheets
        a.sqlite = mock_sqlite
        return a


@pytest.fixture
def sample_crawl_data():
    """Sample crawl data"""
    return {
        "categories": {
            "lip_care": {
                "rank_records": [
                    {"rank": 1, "brand": "LANEIGE", "asin": "B084RGF8YJ"},
                    {"rank": 3, "brand": "COSRX", "asin": "B0ABCDEF01"},
                ]
            }
        },
        "all_products": [
            {"asin": "B084RGF8YJ", "title": "LANEIGE Lip Mask", "brand": "LANEIGE"},
        ],
    }


class TestStorageAgent:
    """Test StorageAgent functionality"""

    @pytest.mark.asyncio
    async def test_execute_success(self, agent, sample_crawl_data):
        """Test successful execution"""
        result = await agent.execute(sample_crawl_data)

        assert result["status"] == "completed"
        assert result["raw_records"] == 2
        assert result["products_upserted"] == 3  # 2 created + 1 updated
        assert result["errors"] == []

    @pytest.mark.asyncio
    async def test_execute_with_sqlite(self, agent, sample_crawl_data):
        """Test that SQLite storage is called"""
        result = await agent.execute(sample_crawl_data)

        agent.sqlite.initialize.assert_called()
        agent.sqlite.append_rank_records.assert_called_once()
        assert result["sqlite_records"] == 3

    @pytest.mark.asyncio
    async def test_execute_sheets_failure(self, agent, sample_crawl_data):
        """Test handling of Sheets write failure"""
        agent.sheets.append_rank_records = AsyncMock(
            return_value={"success": False, "error": "API error"}
        )
        result = await agent.execute(sample_crawl_data)

        assert result["raw_records"] == 0
        assert len(result["errors"]) >= 1
        assert result["errors"][0]["step"] == "raw_data_sheets"

    @pytest.mark.asyncio
    async def test_execute_sqlite_failure(self, agent, sample_crawl_data):
        """Test handling of SQLite failure (still succeeds with Sheets)"""
        agent.sqlite.append_rank_records = AsyncMock(side_effect=Exception("DB error"))
        result = await agent.execute(sample_crawl_data)

        # Sheets should still succeed
        assert result["raw_records"] == 2
        # But SQLite error should be recorded
        sqlite_errors = [e for e in result["errors"] if "sqlite" in e["step"]]
        assert len(sqlite_errors) >= 1

    @pytest.mark.asyncio
    async def test_execute_empty_data(self, agent):
        """Test execution with empty crawl data"""
        result = await agent.execute({"categories": {}})

        assert result["status"] == "completed"
        assert result["raw_records"] == 0

    @pytest.mark.asyncio
    async def test_execute_with_competitor_products(self, agent):
        """Test saving competitor product data"""
        data = {
            "categories": {},
            "all_products": [],
            "competitor_products": [
                {"asin": "B0C42HJRBF", "brand": "Summer Fridays"},
            ],
        }
        result = await agent.execute(data)

        agent.sqlite.save_competitor_products.assert_called_once()

    @pytest.mark.asyncio
    async def test_execute_partial_failure(self, agent, sample_crawl_data):
        """Test status is 'partial' when some steps fail"""
        agent.sheets.upsert_products_batch = AsyncMock(side_effect=Exception("Batch error"))
        result = await agent.execute(sample_crawl_data)

        # raw_records saved but product upsert failed
        assert result["status"] == "partial"
        assert result["raw_records"] == 2

    def test_get_results_empty(self, agent):
        """Test get_results before execution"""
        assert agent.get_results() == {}

    def test_get_historical_data(self, agent):
        """Test historical data retrieval"""
        agent.sheets.get_rank_history = MagicMock(return_value=[{"rank": 5}])
        result = agent.get_historical_data("B084RGF8YJ", days=7)
        assert result == [{"rank": 5}]
        agent.sheets.get_rank_history.assert_called_once_with("B084RGF8YJ", 7)
