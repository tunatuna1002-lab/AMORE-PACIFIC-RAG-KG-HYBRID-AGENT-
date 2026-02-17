"""
JsonFileRepository 단위 테스트

테스트 대상: src/infrastructure/persistence/json_repository.py
Coverage target: 40%+
"""

from datetime import date, datetime, timedelta
from pathlib import Path

import pytest

from src.domain.entities.brand import BrandMetrics
from src.domain.entities.market import MarketMetrics
from src.domain.entities.product import RankRecord
from src.infrastructure.persistence.json_repository import (
    JsonFileRepository,
    _read_json,
    _write_json,
)

# =============================================================================
# Helper JSON functions
# =============================================================================


class TestReadWriteJson:
    """_read_json / _write_json 헬퍼 테스트"""

    def test_write_and_read_roundtrip(self, tmp_path):
        """JSON 쓰기/읽기 왕복"""
        path = tmp_path / "test.json"
        data = {"key": "value", "number": 42, "list": [1, 2, 3]}
        _write_json(path, data)
        result = _read_json(path)
        assert result == data

    def test_write_json_ensure_ascii_false(self, tmp_path):
        """한글이 유니코드 이스케이프 없이 저장"""
        path = tmp_path / "test.json"
        data = {"name": "라네즈"}
        _write_json(path, data)
        content = path.read_text(encoding="utf-8")
        assert "라네즈" in content
        assert "\\u" not in content

    def test_read_json_utf8(self, tmp_path):
        """UTF-8 인코딩 파일 읽기"""
        path = tmp_path / "test.json"
        path.write_text('{"brand": "라네즈"}', encoding="utf-8")
        result = _read_json(path)
        assert result["brand"] == "라네즈"


# =============================================================================
# 초기화 테스트
# =============================================================================


class TestJsonFileRepositoryInit:
    """JsonFileRepository 초기화 테스트"""

    def test_init_default_path(self):
        """기본 데이터 디렉토리"""
        repo = JsonFileRepository()
        assert repo.data_dir == Path("./data")

    def test_init_custom_path(self, tmp_path):
        """커스텀 데이터 디렉토리"""
        repo = JsonFileRepository(data_dir=tmp_path)
        assert repo.data_dir == tmp_path

    def test_init_creates_data_dir(self, tmp_path):
        """데이터 디렉토리 자동 생성"""
        data_dir = tmp_path / "new_dir"
        assert not data_dir.exists()
        JsonFileRepository(data_dir=data_dir)
        assert data_dir.exists()

    def test_file_paths_set(self, tmp_path):
        """내부 파일 경로 설정"""
        repo = JsonFileRepository(data_dir=tmp_path)
        assert repo._records_file == tmp_path / "rank_records.json"
        assert repo._brand_metrics_file == tmp_path / "brand_metrics.json"
        assert repo._market_metrics_file == tmp_path / "market_metrics.json"

    @pytest.mark.asyncio
    async def test_initialize_noop(self, tmp_path):
        """initialize는 noop (에러 없이 완료)"""
        repo = JsonFileRepository(data_dir=tmp_path)
        await repo.initialize()  # should not raise


# =============================================================================
# date serializer 테스트
# =============================================================================


class TestDateSerializer:
    """_date_serializer 테스트"""

    def test_serialize_date(self, tmp_path):
        """date 객체 직렬화"""
        repo = JsonFileRepository(data_dir=tmp_path)
        result = repo._date_serializer(date(2026, 1, 15))
        assert result == "2026-01-15"

    def test_serialize_datetime(self, tmp_path):
        """datetime 객체 직렬화"""
        repo = JsonFileRepository(data_dir=tmp_path)
        dt = datetime(2026, 1, 15, 10, 30, 0)
        result = repo._date_serializer(dt)
        assert "2026-01-15" in result

    def test_serialize_unsupported_type(self, tmp_path):
        """지원되지 않는 타입은 TypeError"""
        repo = JsonFileRepository(data_dir=tmp_path)
        with pytest.raises(TypeError, match="not JSON serializable"):
            repo._date_serializer("string")


# =============================================================================
# save_records / get_recent 테스트
# =============================================================================


def _make_record(
    asin="B001",
    brand="LANEIGE",
    rank=1,
    category_id="lip_care",
    snapshot_date=None,
    price=24.0,
) -> RankRecord:
    """테스트용 RankRecord 생성"""
    return RankRecord(
        snapshot_date=snapshot_date or date.today(),
        category_id=category_id,
        asin=asin,
        product_name=f"Product {asin}",
        brand=brand,
        rank=rank,
        price=price,
        rating=4.5,
        reviews_count=100,
        badge="Best Seller",
        coupon_text="",
        product_url=f"https://amazon.com/dp/{asin}",
    )


class TestSaveRecords:
    """save_records 테스트"""

    @pytest.mark.asyncio
    async def test_save_records_basic(self, tmp_path):
        """기본 레코드 저장"""
        repo = JsonFileRepository(data_dir=tmp_path)
        records = [_make_record("B001"), _make_record("B002", rank=2)]
        result = await repo.save_records(records)
        assert result is True
        assert repo._records_file.exists()

    @pytest.mark.asyncio
    async def test_save_records_appends(self, tmp_path):
        """기존 데이터에 추가 (같은 날짜는 교체)"""
        repo = JsonFileRepository(data_dir=tmp_path)
        yesterday = date.today() - timedelta(days=1)

        await repo.save_records([_make_record("B001", snapshot_date=yesterday)])
        await repo.save_records([_make_record("B002")])

        data = _read_json(repo._records_file)
        # yesterday's record + today's record
        dates = {r["snapshot_date"] for r in data}
        assert str(yesterday) in dates
        assert str(date.today()) in dates

    @pytest.mark.asyncio
    async def test_save_records_replaces_same_date(self, tmp_path):
        """같은 날짜 레코드는 교체"""
        repo = JsonFileRepository(data_dir=tmp_path)
        await repo.save_records([_make_record("B001")])
        await repo.save_records([_make_record("B002")])

        data = _read_json(repo._records_file)
        today_records = [r for r in data if r["snapshot_date"] == str(date.today())]
        # 두 번째 저장이 오늘 날짜를 교체
        assert len(today_records) == 1
        assert today_records[0]["asin"] == "B002"

    @pytest.mark.asyncio
    async def test_save_records_30day_retention(self, tmp_path):
        """30일 이전 데이터 자동 삭제"""
        repo = JsonFileRepository(data_dir=tmp_path)
        old_date = date.today() - timedelta(days=31)

        # 먼저 오래된 데이터 직접 저장
        old_data = [
            {
                "snapshot_date": str(old_date),
                "category_id": "lip_care",
                "asin": "OLD",
                "product_name": "Old",
                "brand": "OLD",
                "rank": 1,
                "price": 10.0,
                "list_price": None,
                "discount_percent": None,
                "rating": 4.0,
                "reviews_count": 50,
                "badge": "",
                "coupon_text": "",
                "is_subscribe_save": False,
                "promo_badges": "",
                "product_url": "",
            }
        ]
        _write_json(repo._records_file, old_data)

        # 새 레코드 저장 시 오래된 데이터 삭제
        await repo.save_records([_make_record("B001")])

        data = _read_json(repo._records_file)
        assert all(r["asin"] != "OLD" for r in data)

    @pytest.mark.asyncio
    async def test_save_records_error_returns_false(self, tmp_path):
        """저장 오류 시 False 반환"""
        repo = JsonFileRepository(data_dir=tmp_path)
        # records_file을 디렉토리로 만들어서 쓰기 실패 유도
        repo._records_file.mkdir(parents=True)
        result = await repo.save_records([_make_record()])
        assert result is False


class TestGetRecent:
    """get_recent 테스트"""

    @pytest.mark.asyncio
    async def test_get_recent_no_file(self, tmp_path):
        """파일 없으면 빈 리스트"""
        repo = JsonFileRepository(data_dir=tmp_path)
        records = await repo.get_recent()
        assert records == []

    @pytest.mark.asyncio
    async def test_get_recent_basic(self, tmp_path):
        """기본 최근 레코드 조회"""
        repo = JsonFileRepository(data_dir=tmp_path)
        await repo.save_records([_make_record("B001"), _make_record("B002", rank=2)])

        records = await repo.get_recent(days=7)
        assert len(records) == 2
        assert all(isinstance(r, RankRecord) for r in records)

    @pytest.mark.asyncio
    async def test_get_recent_filters_by_days(self, tmp_path):
        """일수 필터링"""
        repo = JsonFileRepository(data_dir=tmp_path)
        old = date.today() - timedelta(days=10)
        recent = date.today() - timedelta(days=2)

        await repo.save_records([_make_record("B001", snapshot_date=old)])
        # 직접 추가 (save_records는 같은 날짜 교체하므로)
        data = _read_json(repo._records_file)
        data.append(
            {
                "snapshot_date": str(recent),
                "category_id": "lip_care",
                "asin": "B002",
                "product_name": "Product B002",
                "brand": "LANEIGE",
                "rank": 2,
                "price": 24.0,
                "list_price": None,
                "discount_percent": None,
                "rating": 4.5,
                "reviews_count": 100,
                "badge": "",
                "coupon_text": "",
                "is_subscribe_save": False,
                "promo_badges": "",
                "product_url": "",
            }
        )
        _write_json(repo._records_file, data)

        records = await repo.get_recent(days=5)
        assert len(records) == 1
        assert records[0].asin == "B002"

    @pytest.mark.asyncio
    async def test_get_recent_handles_malformed_row(self, tmp_path):
        """잘못된 행은 스킵"""
        repo = JsonFileRepository(data_dir=tmp_path)
        data = [
            {"snapshot_date": str(date.today()), "category_id": "x"},  # missing fields
            {
                "snapshot_date": str(date.today()),
                "category_id": "lip_care",
                "asin": "B001",
                "product_name": "Product",
                "brand": "LANEIGE",
                "rank": 1,
                "price": 24.0,
                "rating": 4.5,
                "reviews_count": 100,
                "badge": "",
                "coupon_text": "",
                "product_url": "",
            },
        ]
        _write_json(repo._records_file, data)

        records = await repo.get_recent(days=7)
        assert len(records) == 1

    @pytest.mark.asyncio
    async def test_get_recent_error_returns_empty(self, tmp_path):
        """읽기 오류 시 빈 리스트"""
        repo = JsonFileRepository(data_dir=tmp_path)
        repo._records_file.write_text("invalid json{{{", encoding="utf-8")
        records = await repo.get_recent()
        assert records == []


# =============================================================================
# get_by_brand / get_by_category 테스트
# =============================================================================


class TestGetByBrandAndCategory:
    """get_by_brand / get_by_category 테스트"""

    @pytest.mark.asyncio
    async def test_get_by_brand(self, tmp_path):
        """브랜드별 필터"""
        repo = JsonFileRepository(data_dir=tmp_path)
        await repo.save_records(
            [_make_record("B001", brand="LANEIGE"), _make_record("B002", brand="COSRX", rank=2)]
        )

        # get_by_brand 호출 전에 수동으로 두 번째 레코드 추가
        # (save_records는 같은 날짜를 교체하므로)
        data = _read_json(repo._records_file)
        data.append(
            {
                "snapshot_date": str(date.today()),
                "category_id": "lip_care",
                "asin": "B002",
                "product_name": "Product B002",
                "brand": "COSRX",
                "rank": 2,
                "price": 20.0,
                "list_price": None,
                "discount_percent": None,
                "rating": 4.0,
                "reviews_count": 50,
                "badge": "",
                "coupon_text": "",
                "is_subscribe_save": False,
                "promo_badges": "",
                "product_url": "",
            }
        )
        _write_json(repo._records_file, data)

        records = await repo.get_by_brand("laneige")  # case insensitive
        assert all(r.brand.upper() == "LANEIGE" for r in records)

    @pytest.mark.asyncio
    async def test_get_by_category(self, tmp_path):
        """카테고리별 필터"""
        repo = JsonFileRepository(data_dir=tmp_path)
        data = []
        for cat in ["lip_care", "lip_makeup"]:
            data.append(
                {
                    "snapshot_date": str(date.today()),
                    "category_id": cat,
                    "asin": f"B-{cat}",
                    "product_name": f"Product {cat}",
                    "brand": "LANEIGE",
                    "rank": 1,
                    "price": 24.0,
                    "list_price": None,
                    "discount_percent": None,
                    "rating": 4.5,
                    "reviews_count": 100,
                    "badge": "",
                    "coupon_text": "",
                    "is_subscribe_save": False,
                    "promo_badges": "",
                    "product_url": "",
                }
            )
        _write_json(repo._records_file, data)

        records = await repo.get_by_category("lip_care")
        assert len(records) == 1
        assert records[0].category_id == "lip_care"


# =============================================================================
# BrandMetrics 저장/조회 테스트
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

    @pytest.mark.asyncio
    async def test_save_brand_metrics(self, tmp_path):
        """브랜드 메트릭 저장"""
        repo = JsonFileRepository(data_dir=tmp_path)
        metrics = [_make_brand_metrics(), _make_brand_metrics("COSRX", sos=8.0)]
        result = await repo.save_brand_metrics(metrics)
        assert result is True
        assert repo._brand_metrics_file.exists()

    @pytest.mark.asyncio
    async def test_save_brand_metrics_error(self, tmp_path):
        """저장 오류 시 False"""
        repo = JsonFileRepository(data_dir=tmp_path)
        repo._brand_metrics_file.mkdir(parents=True)
        result = await repo.save_brand_metrics([_make_brand_metrics()])
        assert result is False

    @pytest.mark.asyncio
    async def test_get_brand_metrics(self, tmp_path):
        """브랜드 메트릭 조회"""
        repo = JsonFileRepository(data_dir=tmp_path)
        await repo.save_brand_metrics([_make_brand_metrics()])

        result = await repo.get_brand_metrics("LANEIGE")
        assert result is not None
        assert result.brand == "LANEIGE"
        assert result.sos == 12.5

    @pytest.mark.asyncio
    async def test_get_brand_metrics_case_insensitive(self, tmp_path):
        """대소문자 무관 조회"""
        repo = JsonFileRepository(data_dir=tmp_path)
        await repo.save_brand_metrics([_make_brand_metrics()])

        result = await repo.get_brand_metrics("laneige")
        assert result is not None
        assert result.brand == "LANEIGE"

    @pytest.mark.asyncio
    async def test_get_brand_metrics_with_category(self, tmp_path):
        """카테고리 필터 조회"""
        repo = JsonFileRepository(data_dir=tmp_path)
        await repo.save_brand_metrics(
            [
                _make_brand_metrics(category_id="lip_care"),
                _make_brand_metrics(category_id="lip_makeup", sos=5.0),
            ]
        )

        result = await repo.get_brand_metrics("LANEIGE", category_id="lip_makeup")
        assert result is not None
        assert result.sos == 5.0

    @pytest.mark.asyncio
    async def test_get_brand_metrics_not_found(self, tmp_path):
        """존재하지 않는 브랜드"""
        repo = JsonFileRepository(data_dir=tmp_path)
        await repo.save_brand_metrics([_make_brand_metrics()])

        result = await repo.get_brand_metrics("UNKNOWN")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_brand_metrics_no_file(self, tmp_path):
        """파일 없으면 None"""
        repo = JsonFileRepository(data_dir=tmp_path)
        result = await repo.get_brand_metrics("LANEIGE")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_brand_metrics_error(self, tmp_path):
        """읽기 오류 시 None"""
        repo = JsonFileRepository(data_dir=tmp_path)
        repo._brand_metrics_file.write_text("invalid", encoding="utf-8")
        result = await repo.get_brand_metrics("LANEIGE")
        assert result is None


# =============================================================================
# MarketMetrics 저장/조회 테스트
# =============================================================================


def _make_market_metrics(category_id="lip_care", hhi=0.15) -> MarketMetrics:
    return MarketMetrics(
        category_id=category_id,
        snapshot_date=date.today(),
        hhi=hhi,
        churn_rate=0.1,
        category_avg_price=25.0,
        category_avg_rating=4.3,
        calculated_at=datetime.now(),
    )


class TestMarketMetrics:
    """MarketMetrics 저장/조회 테스트"""

    @pytest.mark.asyncio
    async def test_save_market_metrics(self, tmp_path):
        """마켓 메트릭 저장"""
        repo = JsonFileRepository(data_dir=tmp_path)
        metrics = [_make_market_metrics(), _make_market_metrics("lip_makeup", hhi=0.2)]
        result = await repo.save_market_metrics(metrics)
        assert result is True
        assert repo._market_metrics_file.exists()

    @pytest.mark.asyncio
    async def test_save_market_metrics_error(self, tmp_path):
        """저장 오류 시 False"""
        repo = JsonFileRepository(data_dir=tmp_path)
        repo._market_metrics_file.mkdir(parents=True)
        result = await repo.save_market_metrics([_make_market_metrics()])
        assert result is False

    @pytest.mark.asyncio
    async def test_get_market_metrics(self, tmp_path):
        """마켓 메트릭 조회"""
        repo = JsonFileRepository(data_dir=tmp_path)
        await repo.save_market_metrics([_make_market_metrics()])

        result = await repo.get_market_metrics("lip_care")
        assert result is not None
        assert result.category_id == "lip_care"
        assert result.hhi == 0.15

    @pytest.mark.asyncio
    async def test_get_market_metrics_with_date(self, tmp_path):
        """날짜 필터 조회"""
        repo = JsonFileRepository(data_dir=tmp_path)
        await repo.save_market_metrics([_make_market_metrics()])

        result = await repo.get_market_metrics("lip_care", snapshot_date=date.today())
        assert result is not None

    @pytest.mark.asyncio
    async def test_get_market_metrics_wrong_date(self, tmp_path):
        """날짜 불일치 시 None"""
        repo = JsonFileRepository(data_dir=tmp_path)
        await repo.save_market_metrics([_make_market_metrics()])

        yesterday = date.today() - timedelta(days=1)
        result = await repo.get_market_metrics("lip_care", snapshot_date=yesterday)
        assert result is None

    @pytest.mark.asyncio
    async def test_get_market_metrics_not_found(self, tmp_path):
        """존재하지 않는 카테고리"""
        repo = JsonFileRepository(data_dir=tmp_path)
        await repo.save_market_metrics([_make_market_metrics()])

        result = await repo.get_market_metrics("unknown_category")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_market_metrics_no_file(self, tmp_path):
        """파일 없으면 None"""
        repo = JsonFileRepository(data_dir=tmp_path)
        result = await repo.get_market_metrics("lip_care")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_market_metrics_error(self, tmp_path):
        """읽기 오류 시 None"""
        repo = JsonFileRepository(data_dir=tmp_path)
        repo._market_metrics_file.write_text("invalid", encoding="utf-8")
        result = await repo.get_market_metrics("lip_care")
        assert result is None
