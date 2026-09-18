"""
일일 크롤 지표 영속화 회귀 테스트

회귀 배경 (2026-09):
- launchd 일일 크롤(scripts/daily_crawl.py)에 지표 저장 스텝이 없어
  2026-09-01 이후 brand_metrics / market_metrics가 한 행도 쓰이지 않았다
  (지표는 BatchWorkflow STORE_METRICS와 수동 백필에서만 쓰였다).
- 2026-08-31 지표는 01시 수동 크롤 raw_data로 계산됐고, 22시 정기 크롤이
  raw_data를 INSERT OR REPLACE로 덮은 뒤에도 재계산되지 않아 두 테이블이 어긋났다.
"""

import importlib.util
import sqlite3
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.tools.calculators.metric_calculator import calculate_hhi_from_counts, count_brands
from src.tools.calculators.metric_snapshot import build_metric_rows, persist_metrics_for_dates
from src.tools.storage.sqlite_storage import SQLiteStorage

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _load_daily_crawl():
    spec = importlib.util.spec_from_file_location(
        "daily_crawl_under_test", PROJECT_ROOT / "scripts" / "daily_crawl.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _records(date: str, category: str, brands: list[str]) -> list[dict]:
    return [
        {
            "snapshot_date": date,
            "category_id": category,
            "rank": i + 1,
            "asin": f"{category}-{date}-{i}",
            "product_name": f"{brand} product {i}",
            "brand": brand,
            "price": 10.0 + i,
            "rating": 4.5,
            "reviews_count": 100,
            "product_url": "",
        }
        for i, brand in enumerate(brands)
    ]


def _db_rows(db_path: Path, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        return conn.execute(sql, params).fetchall()
    finally:
        conn.close()


@pytest.fixture
async def storage(tmp_path: Path) -> SQLiteStorage:
    s = SQLiteStorage(db_path=str(tmp_path / "metrics.db"))
    await s.initialize()
    return s


# 카테고리당 50행 이상이어야 SoS 최소 표본(SOS_MIN_SAMPLE)을 넘는다
DAY1_LIP = ["LANEIGE"] * 20 + ["Aquaphor"] * 20 + ["Burt's Bees"] * 20
DAY1_POWDER = ["Laura Mercier"] * 15 + ["Unknown"] * 15


class TestPersistMetricsForDates:
    async def test_writes_rows_for_each_crawled_date(self, storage: SQLiteStorage):
        await storage.append_rank_records(
            _records("2026-09-01", "lip_care", DAY1_LIP)
            + _records("2026-09-01", "face_powder", DAY1_POWDER)
            + _records("2026-09-02", "lip_care", DAY1_LIP)
        )

        summary = await persist_metrics_for_dates(storage, ["2026-09-01", "2026-09-02"])

        assert summary["2026-09-01"]["market_rows"] == 2
        assert summary["2026-09-02"]["market_rows"] == 1
        market = await storage.get_market_metrics("2026-09-01", "2026-09-02")
        assert {(m["snapshot_date"], m["category_id"]) for m in market} == {
            ("2026-09-01", "lip_care"),
            ("2026-09-01", "face_powder"),
            ("2026-09-02", "lip_care"),
        }
        brands = await storage.get_brand_metrics("2026-09-01", "2026-09-01", "lip_care")
        assert {b["brand"] for b in brands} == {"LANEIGE", "Aquaphor", "Burt's Bees"}

    async def test_hhi_matches_canonical_calculator(self, storage: SQLiteStorage):
        records = _records("2026-09-01", "face_powder", DAY1_POWDER)
        await storage.append_rank_records(records)

        await persist_metrics_for_dates(storage, ["2026-09-01"])

        (row,) = await storage.get_market_metrics("2026-09-01", "2026-09-01")
        assert row["hhi"] == calculate_hhi_from_counts(count_brands(records))
        # Unknown 제외 → 단일 브랜드 독점
        assert row["hhi"] == 1.0

    async def test_recrawl_same_date_replaces_stale_metrics(self, storage: SQLiteStorage):
        """2026-08-31 회귀: 같은 날짜 raw_data가 덮이면 지표도 그 스냅샷으로 바뀌어야 한다."""
        await storage.append_rank_records(_records("2026-08-31", "lip_care", DAY1_LIP))
        await persist_metrics_for_dates(storage, ["2026-08-31"])

        recrawl = ["LANEIGE"] * 40 + ["Vaseline"] * 20  # Aquaphor·Burt's Bees 이탈
        await storage.append_rank_records(_records("2026-08-31", "lip_care", recrawl))
        await persist_metrics_for_dates(storage, ["2026-08-31"])

        (market,) = await storage.get_market_metrics("2026-08-31", "2026-08-31")
        assert market["hhi"] == calculate_hhi_from_counts({"LANEIGE": 40, "Vaseline": 20})
        brands = await storage.get_brand_metrics("2026-08-31", "2026-08-31")
        assert {b["brand"] for b in brands} == {"LANEIGE", "Vaseline"}

    async def test_date_without_raw_data_keeps_existing_metrics(self, storage: SQLiteStorage):
        await storage.save_market_metrics(
            [{"snapshot_date": "2026-08-31", "category_id": "lip_care", "hhi": 0.0681}]
        )

        summary = await persist_metrics_for_dates(storage, ["2026-08-31"])

        assert summary["2026-08-31"] == {"raw_rows": 0, "brand_rows": 0, "market_rows": 0}
        (row,) = await storage.get_market_metrics("2026-08-31", "2026-08-31")
        assert row["hhi"] == 0.0681

    def test_build_metric_rows_groups_by_date_and_category(self):
        brand_rows, market_rows = build_metric_rows(
            _records("2026-09-01", "lip_care", DAY1_LIP)
            + _records("2026-09-02", "lip_care", DAY1_LIP)
        )
        assert [(m["snapshot_date"], m["category_id"]) for m in market_rows] == [
            ("2026-09-01", "lip_care"),
            ("2026-09-02", "lip_care"),
        ]
        assert len(brand_rows) == 6
        assert all(b["sos"] == pytest.approx(33.33) for b in brand_rows)


class TestDailyCrawlStoresMetrics:
    """launchd가 실제로 실행하는 scripts/daily_crawl.py 파이프라인 회귀 테스트"""

    def test_snapshot_dates_come_from_rank_records(self):
        daily_crawl = _load_daily_crawl()
        crawl_result = {
            "categories": {
                "lip_care": {"rank_records": [{"snapshot_date": "2026-09-17"}]},
                "face_powder": {"rank_records": [{"snapshot_date": "2026-09-16"}]},
                "beauty": {"rank_records": []},
            }
        }
        assert daily_crawl._crawl_snapshot_dates(crawl_result) == {"2026-09-16", "2026-09-17"}
        assert daily_crawl._crawl_snapshot_dates({"categories": {}}) == set()

    async def test_run_pipeline_persists_metrics_before_dashboard_export(
        self, storage: SQLiteStorage
    ):
        daily_crawl = _load_daily_crawl()
        date = "2026-09-02"
        records = _records(date, "lip_care", DAY1_LIP)
        crawl_result = {
            "status": "completed",
            "total_products": len(records),
            "categories": {"lip_care": {"rank_records": records}},
        }

        crawler = MagicMock()
        crawler.scraper.initialize = AsyncMock()
        crawler.scraper.close = AsyncMock()
        crawler.execute = AsyncMock(return_value=crawl_result)

        async def storage_agent_writes_raw_data(result: dict) -> int:
            # 실제 파이프라인에서 raw_data는 STEP 3(StorageAgent)이 쓴다
            await storage.append_rank_records(records)
            return len(records)

        call_order: list[str] = []

        async def export_dashboard() -> None:
            market = await storage.get_market_metrics(date, date)
            call_order.append(f"export(market_rows={len(market)})")

        with (
            patch("src.infrastructure.container.Container.get_crawler_agent", return_value=crawler),
            patch("src.tools.storage.sqlite_storage.get_sqlite_storage", return_value=storage),
            patch.object(daily_crawl, "_save_crawl_json"),
            patch.object(daily_crawl, "_save_to_sqlite", AsyncMock(return_value=0)),
            patch.object(daily_crawl, "_save_to_sheets", side_effect=storage_agent_writes_raw_data),
            patch.object(daily_crawl, "_export_dashboard", side_effect=export_dashboard),
        ):
            result = await daily_crawl.run_pipeline()

        assert result["status"] == "completed", result["errors"]
        assert result["metrics"][date]["market_rows"] == 1
        assert result["metrics"][date]["brand_rows"] == 3
        assert call_order == ["export(market_rows=1)"]
        assert (
            _db_rows(
                storage.db_path,
                "SELECT COUNT(*) AS n FROM brand_metrics WHERE snapshot_date = ?",
                (date,),
            )[0]["n"]
            == 3
        )

    async def test_run_pipeline_flags_date_with_no_raw_data(self, storage: SQLiteStorage):
        daily_crawl = _load_daily_crawl()
        records = _records("2026-09-03", "lip_care", DAY1_LIP)
        crawl_result = {
            "status": "completed",
            "total_products": len(records),
            "categories": {"lip_care": {"rank_records": records}},
        }
        crawler = MagicMock()
        crawler.scraper.initialize = AsyncMock()
        crawler.scraper.close = AsyncMock()
        crawler.execute = AsyncMock(return_value=crawl_result)

        with (
            patch("src.infrastructure.container.Container.get_crawler_agent", return_value=crawler),
            patch("src.tools.storage.sqlite_storage.get_sqlite_storage", return_value=storage),
            patch.object(daily_crawl, "_save_crawl_json"),
            patch.object(daily_crawl, "_save_to_sqlite", AsyncMock(return_value=0)),
            patch.object(daily_crawl, "_save_to_sheets", AsyncMock(return_value=0)),
            patch.object(daily_crawl, "_export_dashboard", AsyncMock()),
        ):
            result = await daily_crawl.run_pipeline()

        assert result["status"] == "completed_with_warnings"
        assert any(e.startswith("metrics: 2026-09-03") for e in result["errors"])
