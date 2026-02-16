"""
Unit tests for src/tools/utilities/data_integrity_checker.py

Tests cover:
- DataIntegrityChecker initialization
- check_sync_status logic
- get_missing_dates logic
- get_date_record_counts logic
- run_full_check severity and recommendations
- _generate_recommendations logic
- Convenience function check_data_integrity
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.tools.utilities.data_integrity_checker import (
    DataIntegrityChecker,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def checker():
    """Uninitialized DataIntegrityChecker"""
    return DataIntegrityChecker(
        spreadsheet_id="test_sheet_id",
        sqlite_path="./data/test.db",
    )


@pytest.fixture
def mock_sheets():
    """Mock SheetsWriter"""
    m = MagicMock()
    m.initialize = AsyncMock(return_value=True)
    m.get_rank_history = AsyncMock(return_value=[])
    return m


@pytest.fixture
def mock_sqlite():
    """Mock SQLiteStorage"""
    m = MagicMock()
    m.initialize = AsyncMock(return_value=True)
    m.get_stats = MagicMock(return_value={"raw_data_count": 0, "date_range": {}})
    m.get_raw_data = AsyncMock(return_value=[])
    return m


@pytest.fixture
def initialized_checker(checker, mock_sheets, mock_sqlite):
    """DataIntegrityChecker with mocked dependencies"""
    checker._sheets = mock_sheets
    checker._sqlite = mock_sqlite
    checker._initialized = True
    return checker


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------
class TestDataIntegrityCheckerInit:
    """초기화 테스트"""

    def test_default_init(self):
        c = DataIntegrityChecker()
        assert c.sqlite_path == "./data/amore_data.db"
        assert c._initialized is False

    def test_custom_init(self, checker):
        assert checker.spreadsheet_id == "test_sheet_id"
        assert checker.sqlite_path == "./data/test.db"
        assert checker._initialized is False

    @pytest.mark.asyncio
    async def test_initialize_success(self, checker):
        with (
            patch("src.tools.storage.sheets_writer.SheetsWriter") as MockSheets,
            patch("src.tools.storage.sqlite_storage.SQLiteStorage") as MockSQLite,
        ):
            mock_sw = MagicMock()
            mock_sw.initialize = AsyncMock(return_value=True)
            MockSheets.return_value = mock_sw

            mock_sq = MagicMock()
            mock_sq.initialize = AsyncMock(return_value=True)
            MockSQLite.return_value = mock_sq

            result = await checker.initialize()
            assert result is True
            assert checker._initialized is True

    @pytest.mark.asyncio
    async def test_initialize_sheets_failure(self, checker):
        with (
            patch("src.tools.storage.sheets_writer.SheetsWriter") as MockSheets,
            patch("src.tools.storage.sqlite_storage.SQLiteStorage") as MockSQLite,
        ):
            mock_sw = MagicMock()
            mock_sw.initialize = AsyncMock(return_value=False)
            MockSheets.return_value = mock_sw

            mock_sq = MagicMock()
            mock_sq.initialize = AsyncMock(return_value=True)
            MockSQLite.return_value = mock_sq

            result = await checker.initialize()
            assert result is False
            assert checker._initialized is False

    @pytest.mark.asyncio
    async def test_initialize_exception(self, checker):
        with patch(
            "src.tools.storage.sheets_writer.SheetsWriter",
            side_effect=Exception("import error"),
        ):
            result = await checker.initialize()
            assert result is False


# ---------------------------------------------------------------------------
# check_sync_status
# ---------------------------------------------------------------------------
class TestCheckSyncStatus:
    """동기화 상태 확인 테스트"""

    @pytest.mark.asyncio
    async def test_synced_status(self, initialized_checker, mock_sheets, mock_sqlite):
        mock_sheets.get_rank_history.return_value = [
            {"snapshot_date": "2026-01-25"} for _ in range(100)
        ]
        mock_sqlite.get_stats.return_value = {
            "raw_data_count": 100,
            "date_range": {"min": "2026-01-01", "max": "2026-01-25"},
        }

        result = await initialized_checker.check_sync_status(days=30)
        assert result["sheets_count"] == 100
        assert result["sqlite_count"] == 100
        assert result["is_synced"] is True
        assert result["gap"] == 0

    @pytest.mark.asyncio
    async def test_unsynced_status(self, initialized_checker, mock_sheets, mock_sqlite):
        mock_sheets.get_rank_history.return_value = [
            {"snapshot_date": "2026-01-25"} for _ in range(500)
        ]
        mock_sqlite.get_stats.return_value = {
            "raw_data_count": 200,
            "date_range": {},
        }

        result = await initialized_checker.check_sync_status(days=30)
        assert result["is_synced"] is False
        assert result["gap"] == 300

    @pytest.mark.asyncio
    async def test_sync_status_empty_data(self, initialized_checker, mock_sheets, mock_sqlite):
        mock_sheets.get_rank_history.return_value = []
        mock_sqlite.get_stats.return_value = {"raw_data_count": 0, "date_range": {}}

        result = await initialized_checker.check_sync_status()
        assert result["sheets_count"] == 0
        assert result["sqlite_count"] == 0
        assert result["is_synced"] is True

    @pytest.mark.asyncio
    async def test_sync_status_none_records(self, initialized_checker, mock_sheets, mock_sqlite):
        mock_sheets.get_rank_history.return_value = None
        mock_sqlite.get_stats.return_value = {"raw_data_count": 0, "date_range": {}}

        result = await initialized_checker.check_sync_status()
        assert result["sheets_count"] == 0

    @pytest.mark.asyncio
    async def test_sync_status_date_ranges(self, initialized_checker, mock_sheets, mock_sqlite):
        mock_sheets.get_rank_history.return_value = [
            {"snapshot_date": "2026-01-20"},
            {"snapshot_date": "2026-01-25"},
        ]
        mock_sqlite.get_stats.return_value = {
            "raw_data_count": 2,
            "date_range": {"min": "2026-01-20", "max": "2026-01-25"},
        }

        result = await initialized_checker.check_sync_status()
        assert result["sheets_date_range"]["min"] == "2026-01-20"
        assert result["sheets_date_range"]["max"] == "2026-01-25"

    @pytest.mark.asyncio
    async def test_sync_status_exception(self, initialized_checker, mock_sheets):
        mock_sheets.get_rank_history.side_effect = Exception("API error")

        result = await initialized_checker.check_sync_status()
        assert "error" in result
        assert result["is_synced"] is False

    @pytest.mark.asyncio
    async def test_auto_initialize_if_needed(self, checker):
        """초기화 안 됐으면 자동으로 초기화"""
        mock_init = AsyncMock(return_value=False)
        with patch.object(checker, "initialize", mock_init):
            result = await checker.check_sync_status()
            # Should have tried to initialize
            mock_init.assert_called_once()


# ---------------------------------------------------------------------------
# get_missing_dates
# ---------------------------------------------------------------------------
class TestGetMissingDates:
    """누락 날짜 조회 테스트"""

    @pytest.mark.asyncio
    async def test_no_missing_dates(self, initialized_checker, mock_sheets, mock_sqlite):
        mock_sheets.get_rank_history.return_value = [
            {"snapshot_date": "2026-01-24"},
            {"snapshot_date": "2026-01-25"},
        ]
        mock_sqlite.get_raw_data.return_value = [
            {"snapshot_date": "2026-01-24"},
            {"snapshot_date": "2026-01-25"},
        ]

        result = await initialized_checker.get_missing_dates(days=7)
        assert result == []

    @pytest.mark.asyncio
    async def test_missing_dates_found(self, initialized_checker, mock_sheets, mock_sqlite):
        mock_sheets.get_rank_history.return_value = [
            {"snapshot_date": "2026-01-23"},
            {"snapshot_date": "2026-01-24"},
            {"snapshot_date": "2026-01-25"},
        ]
        mock_sqlite.get_raw_data.return_value = [
            {"snapshot_date": "2026-01-25"},
        ]

        result = await initialized_checker.get_missing_dates(days=7)
        assert "2026-01-23" in result
        assert "2026-01-24" in result
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_missing_dates_sorted(self, initialized_checker, mock_sheets, mock_sqlite):
        mock_sheets.get_rank_history.return_value = [
            {"snapshot_date": "2026-01-25"},
            {"snapshot_date": "2026-01-20"},
            {"snapshot_date": "2026-01-22"},
        ]
        mock_sqlite.get_raw_data.return_value = []

        result = await initialized_checker.get_missing_dates(days=30)
        assert result == sorted(result)

    @pytest.mark.asyncio
    async def test_missing_dates_exception(self, initialized_checker, mock_sheets):
        mock_sheets.get_rank_history.side_effect = Exception("error")
        result = await initialized_checker.get_missing_dates()
        assert result == []

    @pytest.mark.asyncio
    async def test_missing_dates_none_records(self, initialized_checker, mock_sheets, mock_sqlite):
        mock_sheets.get_rank_history.return_value = None
        mock_sqlite.get_raw_data.return_value = None

        result = await initialized_checker.get_missing_dates()
        assert result == []


# ---------------------------------------------------------------------------
# get_date_record_counts
# ---------------------------------------------------------------------------
class TestGetDateRecordCounts:
    """날짜별 레코드 수 비교 테스트"""

    @pytest.mark.asyncio
    async def test_matching_counts(self, initialized_checker, mock_sheets, mock_sqlite):
        mock_sheets.get_rank_history.return_value = [
            {"snapshot_date": "2026-01-25"},
            {"snapshot_date": "2026-01-25"},
        ]
        mock_sqlite.get_raw_data.return_value = [
            {"snapshot_date": "2026-01-25"},
            {"snapshot_date": "2026-01-25"},
        ]

        result = await initialized_checker.get_date_record_counts(days=7)
        assert "2026-01-25" in result
        assert result["2026-01-25"]["sheets"] == 2
        assert result["2026-01-25"]["sqlite"] == 2
        assert result["2026-01-25"]["diff"] == 0

    @pytest.mark.asyncio
    async def test_mismatched_counts(self, initialized_checker, mock_sheets, mock_sqlite):
        mock_sheets.get_rank_history.return_value = [
            {"snapshot_date": "2026-01-25"},
            {"snapshot_date": "2026-01-25"},
            {"snapshot_date": "2026-01-25"},
        ]
        mock_sqlite.get_raw_data.return_value = [
            {"snapshot_date": "2026-01-25"},
        ]

        result = await initialized_checker.get_date_record_counts(days=7)
        assert result["2026-01-25"]["diff"] == 2

    @pytest.mark.asyncio
    async def test_empty_counts(self, initialized_checker, mock_sheets, mock_sqlite):
        mock_sheets.get_rank_history.return_value = []
        mock_sqlite.get_raw_data.return_value = []

        result = await initialized_checker.get_date_record_counts()
        assert result == {}

    @pytest.mark.asyncio
    async def test_counts_exception(self, initialized_checker, mock_sheets):
        mock_sheets.get_rank_history.side_effect = Exception("error")
        result = await initialized_checker.get_date_record_counts()
        assert result == {}


# ---------------------------------------------------------------------------
# run_full_check
# ---------------------------------------------------------------------------
class TestRunFullCheck:
    """전체 정합성 검사 테스트"""

    @pytest.mark.asyncio
    async def test_severity_ok(self, initialized_checker, mock_sheets, mock_sqlite):
        mock_sheets.get_rank_history.return_value = [
            {"snapshot_date": "2026-01-25"} for _ in range(100)
        ]
        mock_sqlite.get_stats.return_value = {"raw_data_count": 100, "date_range": {}}
        mock_sqlite.get_raw_data.return_value = [
            {"snapshot_date": "2026-01-25"} for _ in range(100)
        ]

        result = await initialized_checker.run_full_check()
        assert result["severity"] == "OK"
        assert "checked_at" in result

    @pytest.mark.asyncio
    async def test_severity_warning(self, initialized_checker, mock_sheets, mock_sqlite):
        mock_sheets.get_rank_history.return_value = [
            {"snapshot_date": "2026-01-25"},
            {"snapshot_date": "2026-01-24"},
        ]
        mock_sqlite.get_stats.return_value = {"raw_data_count": 2, "date_range": {}}
        mock_sqlite.get_raw_data.return_value = [
            {"snapshot_date": "2026-01-25"},
        ]

        result = await initialized_checker.run_full_check()
        assert result["severity"] == "WARNING"

    @pytest.mark.asyncio
    async def test_severity_critical_missing_dates(
        self, initialized_checker, mock_sheets, mock_sqlite
    ):
        mock_sheets.get_rank_history.return_value = [
            {"snapshot_date": f"2026-01-{20+i}"} for i in range(5)
        ]
        mock_sqlite.get_stats.return_value = {"raw_data_count": 5, "date_range": {}}
        mock_sqlite.get_raw_data.return_value = [
            {"snapshot_date": "2026-01-20"},
        ]

        result = await initialized_checker.run_full_check()
        assert result["severity"] == "CRITICAL"

    @pytest.mark.asyncio
    async def test_severity_critical_large_gap(self, initialized_checker, mock_sheets, mock_sqlite):
        mock_sheets.get_rank_history.return_value = [
            {"snapshot_date": "2026-01-25"} for _ in range(600)
        ]
        mock_sqlite.get_stats.return_value = {"raw_data_count": 50, "date_range": {}}
        mock_sqlite.get_raw_data.return_value = [
            {"snapshot_date": "2026-01-25"} for _ in range(600)
        ]

        result = await initialized_checker.run_full_check()
        assert result["severity"] == "CRITICAL"


# ---------------------------------------------------------------------------
# _generate_recommendations
# ---------------------------------------------------------------------------
class TestGenerateRecommendations:
    """권고사항 생성 테스트"""

    def test_no_issues(self, checker):
        recs = checker._generate_recommendations({"gap": 0, "is_synced": True}, [])
        assert len(recs) == 1
        assert "정상" in recs[0]

    def test_gap_recommendation(self, checker):
        recs = checker._generate_recommendations({"gap": 200, "is_synced": False}, [])
        assert any("200" in r for r in recs)
        assert any("sync" in r.lower() or "동기화" in r for r in recs)

    def test_missing_dates_recommendation(self, checker):
        recs = checker._generate_recommendations(
            {"gap": 0, "is_synced": True},
            ["2026-01-20", "2026-01-21"],
        )
        assert any("2" in r and "누락" in r for r in recs)

    def test_missing_dates_truncation(self, checker):
        dates = [f"2026-01-{10+i}" for i in range(10)]
        recs = checker._generate_recommendations({"gap": 0, "is_synced": True}, dates)
        assert any("..." in r for r in recs)

    def test_not_synced_recommendation(self, checker):
        recs = checker._generate_recommendations({"gap": 50, "is_synced": False}, [])
        assert any("동기화" in r or "sync" in r.lower() for r in recs)
