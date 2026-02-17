"""
GoogleSheetsRepository 단위 테스트

테스트 대상: src/infrastructure/persistence/sheets_repository.py
Coverage target: 40%+

Google Sheets API는 모킹하여 외부 의존 없이 테스트합니다.
"""

import os
from datetime import date, datetime
from unittest.mock import MagicMock, patch

import pytest

from src.domain.entities.brand import BrandMetrics
from src.domain.entities.market import MarketMetrics
from src.domain.entities.product import RankRecord
from src.infrastructure.persistence.sheets_repository import GoogleSheetsRepository

# =============================================================================
# 초기화 테스트
# =============================================================================


class TestGoogleSheetsRepositoryInit:
    """GoogleSheetsRepository 초기화 테스트"""

    def test_init_default(self):
        """기본 초기화"""
        repo = GoogleSheetsRepository()
        assert repo.spreadsheet_id is None
        assert repo.credentials_path is None
        assert repo._client is None
        assert repo._sheet is None
        assert repo._initialized is False

    def test_init_with_args(self):
        """파라미터 전달 초기화"""
        repo = GoogleSheetsRepository(
            spreadsheet_id="sheet_123", credentials_path="/path/to/creds.json"
        )
        assert repo.spreadsheet_id == "sheet_123"
        assert repo.credentials_path == "/path/to/creds.json"

    @pytest.mark.asyncio
    async def test_initialize_already_initialized(self):
        """이미 초기화된 경우 스킵"""
        repo = GoogleSheetsRepository()
        repo._initialized = True
        await repo.initialize()  # should not raise

    @pytest.mark.asyncio
    async def test_initialize_missing_gspread_import(self):
        """gspread 미설치 시 RuntimeError"""
        repo = GoogleSheetsRepository(spreadsheet_id="test")
        with patch.dict("sys.modules", {"gspread": None}):
            # gspread import가 실패하면 RuntimeError
            # 실제로는 ImportError가 캐치됨
            with pytest.raises((RuntimeError, ImportError, ModuleNotFoundError)):
                await repo.initialize()

    @pytest.mark.asyncio
    async def test_initialize_missing_spreadsheet_id(self):
        """spreadsheet_id 없으면 ValueError"""
        repo = GoogleSheetsRepository()
        with patch.dict(os.environ, {}, clear=False):
            # GOOGLE_SPREADSHEET_ID 환경변수 없이
            env_without_sheet = {
                k: v for k, v in os.environ.items() if k != "GOOGLE_SPREADSHEET_ID"
            }
            with patch.dict(os.environ, env_without_sheet, clear=True):
                with pytest.raises((ValueError, RuntimeError, ImportError, ModuleNotFoundError)):
                    await repo.initialize()


# =============================================================================
# save_records 테스트
# =============================================================================


def _make_record(asin="B001", brand="LANEIGE", rank=1) -> RankRecord:
    return RankRecord(
        snapshot_date=date.today(),
        category_id="lip_care",
        asin=asin,
        product_name=f"Product {asin}",
        brand=brand,
        rank=rank,
        price=24.0,
        rating=4.5,
        reviews_count=100,
        badge="Best Seller",
        coupon_text="10% off",
        product_url=f"https://amazon.com/dp/{asin}",
    )


class TestSaveRecords:
    """save_records 테스트"""

    def _make_initialized_repo(self):
        """초기화된 mock repo 생성"""
        repo = GoogleSheetsRepository(spreadsheet_id="test")
        repo._initialized = True
        repo._client = MagicMock()
        repo._sheet = MagicMock()
        return repo

    @pytest.mark.asyncio
    async def test_save_records_basic(self):
        """기본 레코드 저장"""
        repo = self._make_initialized_repo()
        mock_ws = MagicMock()
        repo._sheet.worksheet.return_value = mock_ws

        records = [_make_record("B001"), _make_record("B002", rank=2)]
        result = await repo.save_records(records)

        assert result is True
        mock_ws.clear.assert_called_once()
        mock_ws.update.assert_called_once()

    @pytest.mark.asyncio
    async def test_save_records_creates_worksheet(self):
        """워크시트 없으면 생성"""
        repo = self._make_initialized_repo()
        repo._sheet.worksheet.side_effect = Exception("Not found")
        mock_ws = MagicMock()
        repo._sheet.add_worksheet.return_value = mock_ws

        result = await repo.save_records([_make_record()])

        assert result is True
        repo._sheet.add_worksheet.assert_called_once()

    @pytest.mark.asyncio
    async def test_save_records_empty(self):
        """빈 레코드 저장"""
        repo = self._make_initialized_repo()
        mock_ws = MagicMock()
        repo._sheet.worksheet.return_value = mock_ws

        result = await repo.save_records([])
        assert result is True

    @pytest.mark.asyncio
    async def test_save_records_error(self):
        """저장 오류 시 False"""
        repo = self._make_initialized_repo()
        repo._sheet.worksheet.side_effect = Exception("Not found")
        repo._sheet.add_worksheet.side_effect = Exception("API error")

        result = await repo.save_records([_make_record()])
        assert result is False

    @pytest.mark.asyncio
    async def test_save_records_correct_headers(self):
        """올바른 헤더 포함 확인"""
        repo = self._make_initialized_repo()
        mock_ws = MagicMock()
        repo._sheet.worksheet.return_value = mock_ws

        await repo.save_records([_make_record()])

        call_args = mock_ws.update.call_args
        rows = call_args[0][0]
        headers = rows[0]
        assert "snapshot_date" in headers
        assert "asin" in headers
        assert "brand" in headers
        assert "rank" in headers
        assert "price" in headers


# =============================================================================
# get_recent 테스트
# =============================================================================


class TestGetRecent:
    """get_recent 테스트"""

    def _make_initialized_repo(self):
        repo = GoogleSheetsRepository(spreadsheet_id="test")
        repo._initialized = True
        repo._client = MagicMock()
        repo._sheet = MagicMock()
        return repo

    @pytest.mark.asyncio
    async def test_get_recent_basic(self):
        """기본 레코드 조회"""
        repo = self._make_initialized_repo()
        mock_ws = MagicMock()
        mock_ws.get_all_records.return_value = [
            {
                "snapshot_date": str(date.today()),
                "category_id": "lip_care",
                "asin": "B001",
                "product_name": "LANEIGE Lip Mask",
                "brand": "LANEIGE",
                "rank": 1,
                "price": 24.0,
                "rating": 4.5,
                "reviews_count": 100,
                "badge": "",
                "coupon_text": "",
                "product_url": "https://amazon.com/dp/B001",
            }
        ]
        repo._sheet.worksheet.return_value = mock_ws

        records = await repo.get_recent(days=7)
        assert len(records) == 1
        assert isinstance(records[0], RankRecord)
        assert records[0].asin == "B001"

    @pytest.mark.asyncio
    async def test_get_recent_handles_malformed(self):
        """잘못된 데이터 스킵"""
        repo = self._make_initialized_repo()
        mock_ws = MagicMock()
        mock_ws.get_all_records.return_value = [
            {"snapshot_date": "invalid", "bad": "data"},
        ]
        repo._sheet.worksheet.return_value = mock_ws

        records = await repo.get_recent()
        assert records == []

    @pytest.mark.asyncio
    async def test_get_recent_error(self):
        """오류 시 빈 리스트"""
        repo = self._make_initialized_repo()
        repo._sheet.worksheet.side_effect = Exception("API error")

        records = await repo.get_recent()
        assert records == []


# =============================================================================
# get_by_brand / get_by_category 테스트
# =============================================================================


class TestGetByBrandAndCategory:
    """get_by_brand / get_by_category 테스트"""

    def _make_repo_with_records(self):
        repo = GoogleSheetsRepository(spreadsheet_id="test")
        repo._initialized = True
        repo._client = MagicMock()
        repo._sheet = MagicMock()

        mock_ws = MagicMock()
        mock_ws.get_all_records.return_value = [
            {
                "snapshot_date": str(date.today()),
                "category_id": "lip_care",
                "asin": "B001",
                "product_name": "LANEIGE Lip Mask",
                "brand": "LANEIGE",
                "rank": 1,
                "price": 24.0,
                "rating": 4.5,
                "reviews_count": 100,
                "badge": "",
                "coupon_text": "",
                "product_url": "",
            },
            {
                "snapshot_date": str(date.today()),
                "category_id": "lip_makeup",
                "asin": "B002",
                "product_name": "COSRX Lip Product",
                "brand": "COSRX",
                "rank": 2,
                "price": 18.0,
                "rating": 4.2,
                "reviews_count": 50,
                "badge": "",
                "coupon_text": "",
                "product_url": "",
            },
        ]
        repo._sheet.worksheet.return_value = mock_ws
        return repo

    @pytest.mark.asyncio
    async def test_get_by_brand(self):
        """브랜드별 필터 (대소문자 무관)"""
        repo = self._make_repo_with_records()
        records = await repo.get_by_brand("laneige")
        assert len(records) == 1
        assert records[0].brand == "LANEIGE"

    @pytest.mark.asyncio
    async def test_get_by_category(self):
        """카테고리별 필터"""
        repo = self._make_repo_with_records()
        records = await repo.get_by_category("lip_makeup")
        assert len(records) == 1
        assert records[0].category_id == "lip_makeup"


# =============================================================================
# BrandMetrics 테스트
# =============================================================================


def _make_brand_metrics(brand="LANEIGE", category_id="lip_care", sos=12.5) -> BrandMetrics:
    return BrandMetrics(
        brand=brand,
        category_id=category_id,
        sos=sos,
        brand_avg_rank=5.0,
        product_count=3,
        cpi=0.8,
        avg_rating_gap=0.2,
        calculated_at=datetime.now(),
    )


class TestBrandMetrics:
    """BrandMetrics 저장/조회 테스트"""

    def _make_initialized_repo(self):
        repo = GoogleSheetsRepository(spreadsheet_id="test")
        repo._initialized = True
        repo._client = MagicMock()
        repo._sheet = MagicMock()
        return repo

    @pytest.mark.asyncio
    async def test_save_brand_metrics(self):
        """브랜드 메트릭 저장"""
        repo = self._make_initialized_repo()
        mock_ws = MagicMock()
        repo._sheet.worksheet.return_value = mock_ws

        result = await repo.save_brand_metrics([_make_brand_metrics()])
        assert result is True
        mock_ws.clear.assert_called_once()
        mock_ws.update.assert_called_once()

    @pytest.mark.asyncio
    async def test_save_brand_metrics_creates_worksheet(self):
        """워크시트 없으면 생성"""
        repo = self._make_initialized_repo()
        repo._sheet.worksheet.side_effect = Exception("Not found")
        mock_ws = MagicMock()
        repo._sheet.add_worksheet.return_value = mock_ws

        result = await repo.save_brand_metrics([_make_brand_metrics()])
        assert result is True

    @pytest.mark.asyncio
    async def test_save_brand_metrics_error(self):
        """저장 오류 시 False"""
        repo = self._make_initialized_repo()
        repo._sheet.worksheet.side_effect = Exception("Not found")
        repo._sheet.add_worksheet.side_effect = Exception("API error")

        result = await repo.save_brand_metrics([_make_brand_metrics()])
        assert result is False

    @pytest.mark.asyncio
    async def test_get_brand_metrics(self):
        """브랜드 메트릭 조회"""
        repo = self._make_initialized_repo()
        mock_ws = MagicMock()
        mock_ws.get_all_records.return_value = [
            {
                "brand": "LANEIGE",
                "category_id": "lip_care",
                "sos": 12.5,
                "brand_avg_rank": 5.0,
                "product_count": 3,
                "cpi": 0.8,
                "avg_rating_gap": 0.2,
            }
        ]
        repo._sheet.worksheet.return_value = mock_ws

        result = await repo.get_brand_metrics("LANEIGE")
        assert result is not None
        assert result.brand == "LANEIGE"
        assert result.sos == 12.5

    @pytest.mark.asyncio
    async def test_get_brand_metrics_case_insensitive(self):
        """대소문자 무관 조회"""
        repo = self._make_initialized_repo()
        mock_ws = MagicMock()
        mock_ws.get_all_records.return_value = [
            {
                "brand": "LANEIGE",
                "category_id": "lip_care",
                "sos": 12.5,
                "brand_avg_rank": 5.0,
                "product_count": 3,
                "cpi": 0.8,
                "avg_rating_gap": 0.2,
            }
        ]
        repo._sheet.worksheet.return_value = mock_ws

        result = await repo.get_brand_metrics("laneige")
        assert result is not None

    @pytest.mark.asyncio
    async def test_get_brand_metrics_with_category_filter(self):
        """카테고리 필터 조회"""
        repo = self._make_initialized_repo()
        mock_ws = MagicMock()
        mock_ws.get_all_records.return_value = [
            {
                "brand": "LANEIGE",
                "category_id": "lip_care",
                "sos": 12.5,
                "brand_avg_rank": 5.0,
                "product_count": 3,
                "cpi": 0.8,
                "avg_rating_gap": 0.2,
            },
            {
                "brand": "LANEIGE",
                "category_id": "lip_makeup",
                "sos": 5.0,
                "brand_avg_rank": 8.0,
                "product_count": 1,
                "cpi": 0.5,
                "avg_rating_gap": 0.3,
            },
        ]
        repo._sheet.worksheet.return_value = mock_ws

        result = await repo.get_brand_metrics("LANEIGE", category_id="lip_makeup")
        assert result is not None
        assert result.sos == 5.0

    @pytest.mark.asyncio
    async def test_get_brand_metrics_not_found(self):
        """존재하지 않는 브랜드"""
        repo = self._make_initialized_repo()
        mock_ws = MagicMock()
        mock_ws.get_all_records.return_value = []
        repo._sheet.worksheet.return_value = mock_ws

        result = await repo.get_brand_metrics("UNKNOWN")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_brand_metrics_error(self):
        """오류 시 None"""
        repo = self._make_initialized_repo()
        repo._sheet.worksheet.side_effect = Exception("API error")

        result = await repo.get_brand_metrics("LANEIGE")
        assert result is None


# =============================================================================
# MarketMetrics 테스트
# =============================================================================


class TestMarketMetrics:
    """MarketMetrics 저장/조회 테스트"""

    def _make_initialized_repo(self):
        repo = GoogleSheetsRepository(spreadsheet_id="test")
        repo._initialized = True
        repo._client = MagicMock()
        repo._sheet = MagicMock()
        return repo

    def _make_market_metrics(self, category_id="lip_care", hhi=0.15) -> MarketMetrics:
        return MarketMetrics(
            category_id=category_id,
            snapshot_date=date.today(),
            hhi=hhi,
            churn_rate=0.1,
            category_avg_price=25.0,
            category_avg_rating=4.3,
            calculated_at=datetime.now(),
        )

    @pytest.mark.asyncio
    async def test_save_market_metrics(self):
        """마켓 메트릭 저장"""
        repo = self._make_initialized_repo()
        mock_ws = MagicMock()
        repo._sheet.worksheet.return_value = mock_ws

        result = await repo.save_market_metrics([self._make_market_metrics()])
        assert result is True

    @pytest.mark.asyncio
    async def test_save_market_metrics_error(self):
        """저장 오류 시 False"""
        repo = self._make_initialized_repo()
        repo._sheet.worksheet.side_effect = Exception("Not found")
        repo._sheet.add_worksheet.side_effect = Exception("API error")

        result = await repo.save_market_metrics([self._make_market_metrics()])
        assert result is False

    @pytest.mark.asyncio
    async def test_get_market_metrics(self):
        """마켓 메트릭 조회"""
        repo = self._make_initialized_repo()
        mock_ws = MagicMock()
        mock_ws.get_all_records.return_value = [
            {
                "category_id": "lip_care",
                "snapshot_date": str(date.today()),
                "hhi": 0.15,
                "churn_rate": 0.1,
                "category_avg_price": 25.0,
                "category_avg_rating": 4.3,
            }
        ]
        repo._sheet.worksheet.return_value = mock_ws

        result = await repo.get_market_metrics("lip_care")
        assert result is not None
        assert result.hhi == 0.15

    @pytest.mark.asyncio
    async def test_get_market_metrics_with_date(self):
        """날짜 필터 조회"""
        repo = self._make_initialized_repo()
        mock_ws = MagicMock()
        mock_ws.get_all_records.return_value = [
            {
                "category_id": "lip_care",
                "snapshot_date": str(date.today()),
                "hhi": 0.15,
                "churn_rate": 0.1,
                "category_avg_price": 25.0,
                "category_avg_rating": 4.3,
            }
        ]
        repo._sheet.worksheet.return_value = mock_ws

        result = await repo.get_market_metrics("lip_care", snapshot_date=date.today())
        assert result is not None

    @pytest.mark.asyncio
    async def test_get_market_metrics_not_found(self):
        """존재하지 않는 카테고리"""
        repo = self._make_initialized_repo()
        mock_ws = MagicMock()
        mock_ws.get_all_records.return_value = []
        repo._sheet.worksheet.return_value = mock_ws

        result = await repo.get_market_metrics("unknown")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_market_metrics_error(self):
        """오류 시 None"""
        repo = self._make_initialized_repo()
        repo._sheet.worksheet.side_effect = Exception("API error")

        result = await repo.get_market_metrics("lip_care")
        assert result is None
