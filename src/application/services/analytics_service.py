"""
Analytics Service (KPI / SoS)
=============================
The aggregation behind ``/api/category/kpi`` and the ``/api/sos/*`` endpoints (F6).
The routes only validate the query string and hand the call over; everything that
turns rows into a response payload lives here and is testable without FastAPI.

Data sources are injected, never imported:

- ``sqlite_factory`` -> the storage object (``initialize()`` + ``get_connection()``)
- ``crawl_loader``   -> ``latest_crawl_result.json`` as a dict, or None

so a caller can point the service at any storage and the route keeps its patch
points. Every synchronous sqlite3 block runs through ``asyncio.to_thread`` (D20).

Date defaults come from ``date_range.resolve_date_range`` (KST) and category
metadata from ``category_names.monitored_category_meta`` — both single sources.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from typing import Any

from src.application.services.category_names import monitored_category_meta
from src.application.services.date_range import resolve_date_range
from src.application.services.sql_rows import fetch_all
from src.domain.brand import is_target_brand

logger = logging.getLogger(__name__)

UNKNOWN_CATEGORY_META: dict[str, Any] = {
    "name": "",
    "level": 0,
    "parent_id": None,
    "indent": 0,
    "order": 99,
}


class AnalyticsService:
    """KPI / SoS aggregation over the raw_data snapshots."""

    def __init__(
        self,
        sqlite_factory: Callable[[], Any],
        crawl_loader: Callable[[], dict | None],
    ):
        self._sqlite_factory = sqlite_factory
        self._crawl_loader = crawl_loader

    async def _storage(self) -> Any:
        sqlite = self._sqlite_factory()
        await sqlite.initialize()
        return sqlite

    def _crawl_data(self) -> dict | None:
        return self._crawl_loader()

    # ------------------------------------------------------------ category KPI
    async def category_kpi(
        self,
        category_id: str,
        start_date: str | None = None,
        end_date: str | None = None,
        brand: str = "LANEIGE",
    ) -> dict[str, Any]:
        """SoS / best rank / CPI / new competitors for one category over a window."""
        try:
            # 날짜 범위 설정 (기본: 최근 7일, KST 기준)
            start_date, end_date = resolve_date_range(start_date, end_date, default_days=7)

            rows: list = []

            # SQLite에서 데이터 조회
            try:
                sqlite = await self._storage()
                query = """
                    SELECT snapshot_date, rank, brand, price
                    FROM raw_data
                    WHERE snapshot_date BETWEEN ? AND ?
                    AND category_id = ?
                    ORDER BY snapshot_date DESC, rank ASC
                """
                rows = await asyncio.to_thread(
                    fetch_all, sqlite, query, (start_date, end_date, category_id)
                )
            except Exception as db_err:
                logger.warning(f"SQLite query failed for category KPI: {db_err}")

            # JSON fallback
            if not rows:
                crawl_data = self._crawl_data()
                if crawl_data and crawl_data.get("categories", {}).get(category_id):
                    cat_data = crawl_data["categories"][category_id]
                    snapshot_date = crawl_data.get("snapshot_date", end_date)
                    for product in cat_data.get("products", []):
                        rows.append(
                            (
                                snapshot_date,
                                product.get("rank", 100),
                                product.get("brand", "Unknown"),
                                product.get("price"),
                            )
                        )

            if not rows:
                return {
                    "success": True,
                    "message": f"해당 기간({start_date} ~ {end_date})에 데이터가 없습니다.",
                    "data": None,
                    "period": {"start": start_date, "end": end_date},
                }

            # KPI 계산
            total_products = len(rows)
            brand_products = [r for r in rows if r[2] and brand.lower() in r[2].lower()]
            brand_count = len(brand_products)

            # SoS (Share of Shelf)
            sos = (brand_count / total_products * 100) if total_products > 0 else 0

            # Best Rank
            brand_ranks = [r[1] for r in brand_products if r[1]]
            best_rank = min(brand_ranks) if brand_ranks else None

            # CPI (Competitive Price Index) - 브랜드 평균가 / 전체 평균가 * 100
            brand_prices = [r[3] for r in brand_products if r[3] and r[3] > 0]
            all_prices = [r[3] for r in rows if r[3] and r[3] > 0]

            if brand_prices and all_prices:
                brand_avg_price = sum(brand_prices) / len(brand_prices)
                all_avg_price = sum(all_prices) / len(all_prices)
                cpi = (brand_avg_price / all_avg_price * 100) if all_avg_price > 0 else 100
            else:
                cpi = 100

            # New Competitors (최근 7일 내 신규 진입 - 간소화된 계산)
            new_competitors = max(0, total_products - brand_count - 50)

            return {
                "success": True,
                "data": {
                    "category_id": category_id,
                    "sos": round(sos, 1),
                    "best_rank": best_rank,
                    "cpi": round(cpi, 0),
                    "new_competitors": new_competitors,
                    "brand": brand,
                    "product_count": brand_count,
                    "total_products": total_products,
                },
                "period": {"start": start_date, "end": end_date},
            }

        except Exception as e:
            logger.error(f"Category KPI API error: {e}")
            return {"success": False, "error": str(e), "data": None}

    # ------------------------------------------------------------- SoS by category
    async def sos_by_category(
        self,
        start_date: str | None = None,
        end_date: str | None = None,
        compare_brands: str | None = None,
    ) -> dict[str, Any]:
        """Per-category Share of Shelf (target brand + optional comparison brands)."""
        try:
            # 비교 브랜드 파싱
            compare_brand_list = []
            if compare_brands:
                compare_brand_list = [b.strip() for b in compare_brands.split(",") if b.strip()]

            # 날짜 범위 설정 (기본: 당일, KST 기준)
            start_date, end_date = resolve_date_range(start_date, end_date, default_days=0)

            # SQLite 먼저 시도
            rows: list = []
            try:
                sqlite = await self._storage()
                query = """
                    SELECT snapshot_date, category_id, brand, COUNT(*) as product_count
                    FROM raw_data
                    WHERE snapshot_date BETWEEN ? AND ?
                    GROUP BY snapshot_date, category_id, brand
                    ORDER BY snapshot_date DESC, category_id, product_count DESC
                """
                rows = await asyncio.to_thread(fetch_all, sqlite, query, (start_date, end_date))
            except Exception as db_err:
                logger.warning(f"SQLite query failed, using JSON fallback: {db_err}")

            # SQLite 데이터 없으면 JSON fallback
            if not rows:
                crawl_data = self._crawl_data()
                if crawl_data and crawl_data.get("categories"):
                    snapshot_date = crawl_data.get("snapshot_date", end_date)
                    for cat_id, cat_data in crawl_data.get("categories", {}).items():
                        for product in cat_data.get("products", []):
                            brand = product.get("brand", "Unknown")
                            rows.append((snapshot_date, cat_id, brand, 1))

            if not rows:
                return {
                    "success": True,
                    "message": f"해당 기간({start_date} ~ {end_date})에 데이터가 없습니다.",
                    "data": [],
                    "period": {"start": start_date, "end": end_date},
                }

            # 데이터 집계
            category_data: dict[str, dict[str, Any]] = {}
            dates_set = set()

            for row in rows:
                snapshot_date, category_id, brand, count = row[0], row[1], row[2], row[3]
                dates_set.add(snapshot_date)

                if category_id not in category_data:
                    category_data[category_id] = {}
                if brand not in category_data[category_id]:
                    category_data[category_id][brand] = {"dates": {}, "total_count": 0}

                category_data[category_id][brand]["dates"][snapshot_date] = count
                category_data[category_id][brand]["total_count"] += count

            # SoS 계산 (기간 평균)
            num_dates = len(dates_set)
            result_data = []

            # 카테고리 메타 정보 (계층 구조 포함)
            # config/category_hierarchy.json 단일 출처 (F6: 라우트에 박혀 있던
            # 5개 카테고리 리터럴 맵 제거 — 같은 name/level/parent/indent/order)
            category_meta = monitored_category_meta()

            for category_id, brands in category_data.items():
                # 해당 카테고리의 총 제품 수 (기간 합계)
                total_products_in_category = sum(b["total_count"] for b in brands.values())

                # 타겟 브랜드 SoS (F6: 3개 철자 변형 목록 -> is_target_brand)
                target_count = 0
                target_dates: set[str] = set()
                for brand_name, brand_data in brands.items():
                    if is_target_brand(brand_name):
                        target_count += brand_data["total_count"]
                        target_dates.update(brand_data.get("dates", {}))
                target_appearance_days = len(target_dates)

                target_sos = (
                    (target_count / total_products_in_category * 100)
                    if total_products_in_category > 0
                    else 0
                )

                # 평균 SoS (전체 브랜드 수 기준)
                num_brands = len(brands)
                avg_sos = (100 / num_brands) if num_brands > 0 else 0

                # 비교 브랜드 SoS
                compare_sos = {}
                for compare_brand in compare_brand_list:
                    brand_count = 0
                    for brand_name, brand_data in brands.items():
                        if compare_brand.lower() in brand_name.lower():
                            brand_count += brand_data["total_count"]
                    compare_sos[compare_brand] = (
                        (brand_count / total_products_in_category * 100)
                        if total_products_in_category > 0
                        else 0
                    )

                # 카테고리 메타 정보 가져오기
                meta = category_meta.get(
                    category_id, {**UNKNOWN_CATEGORY_META, "name": category_id}
                )

                result_data.append(
                    {
                        "category_id": category_id,
                        "category_name": meta["name"],
                        "level": meta["level"],
                        "parent_id": meta["parent_id"],
                        "indent": meta["indent"],
                        "order": meta["order"],
                        "total_products": total_products_in_category // num_dates
                        if num_dates > 0
                        else 0,
                        "laneige_sos": round(target_sos, 2),
                        "laneige_count": round(target_count / num_dates, 1) if num_dates > 0 else 0,
                        "laneige_appearance_days": target_appearance_days,
                        "laneige_appearance_rate": round(
                            target_appearance_days / num_dates * 100, 1
                        )
                        if num_dates > 0
                        else 0,
                        "avg_sos": round(avg_sos, 2),
                        "compare_brands": compare_sos,
                        "num_dates": num_dates,
                    }
                )

            # 계층 구조 순서대로 정렬
            result_data.sort(key=lambda x: x.get("order", 99))

            return {
                "success": True,
                "period": {"start": start_date, "end": end_date, "days": num_dates},
                "data": result_data,
                "compare_brands": compare_brand_list,
                "hierarchy_info": {
                    "description": "각 카테고리는 자체 Top 100 기준으로 독립 계산됩니다.",
                    "note": "상위 카테고리와 하위 카테고리의 SoS는 서로 다른 랭킹에서 계산됩니다.",
                },
            }

        except Exception as e:
            logger.error(f"SoS category API error: {e}")
            return {"success": False, "error": str(e)}

    # ------------------------------------------------------------------- brands
    async def available_brands(
        self, category_id: str | None = None, min_count: int = 1
    ) -> dict[str, Any]:
        """Brands present in the last week's Top 100 (comparison picker)."""
        try:
            start_date, end_date = resolve_date_range(None, None, default_days=7)

            rows: list = []
            # SQLite 먼저 시도
            try:
                sqlite = await self._storage()
                if category_id:
                    query = """
                        SELECT brand, COUNT(DISTINCT asin) as product_count,
                               COUNT(DISTINCT snapshot_date) as days_present
                        FROM raw_data
                        WHERE snapshot_date BETWEEN ? AND ?
                        AND category_id = ?
                        AND LOWER(brand) != 'unknown'
                        GROUP BY brand
                        HAVING product_count >= ?
                        ORDER BY product_count DESC
                    """
                    params: tuple = (start_date, end_date, category_id, min_count)
                else:
                    query = """
                        SELECT brand, COUNT(DISTINCT asin) as product_count,
                               COUNT(DISTINCT snapshot_date) as days_present
                        FROM raw_data
                        WHERE snapshot_date BETWEEN ? AND ?
                        AND LOWER(brand) != 'unknown'
                        GROUP BY brand
                        HAVING product_count >= ?
                        ORDER BY product_count DESC
                    """
                    params = (start_date, end_date, min_count)

                rows = await asyncio.to_thread(fetch_all, sqlite, query, params)
            except Exception as db_err:
                logger.warning(f"SQLite query failed for brands: {db_err}")

            # SQLite 데이터 없으면 JSON fallback
            brands: list[dict[str, Any]] = []
            if not rows:
                crawl_data = self._crawl_data()
                if crawl_data and crawl_data.get("categories"):
                    brand_counts: dict[str, int] = {}
                    for cat_id, cat_data in crawl_data.get("categories", {}).items():
                        if category_id and cat_id != category_id:
                            continue
                        for product in cat_data.get("products", []):
                            brand = product.get("brand", "Unknown")
                            if brand:
                                brand_counts[brand] = brand_counts.get(brand, 0) + 1

                    for brand_name, count in sorted(brand_counts.items(), key=lambda x: -x[1]):
                        if (
                            count >= min_count
                            and brand_name.strip()
                            and brand_name.lower() != "unknown"
                        ):
                            brands.append(
                                {
                                    "name": brand_name,
                                    "product_count": count,
                                    "days_present": 1,
                                    "is_laneige": is_target_brand(brand_name),
                                }
                            )
            else:
                for row in rows:
                    brand_name, product_count, days_present = row
                    if brand_name and brand_name.strip() and brand_name.lower() != "unknown":
                        brands.append(
                            {
                                "name": brand_name,
                                "product_count": product_count,
                                "days_present": days_present,
                                "is_laneige": is_target_brand(brand_name),
                            }
                        )

            return {
                "success": True,
                "period": {"start": start_date, "end": end_date},
                "category_id": category_id,
                "brands": brands,
                "total_brands": len(brands),
            }

        except Exception as e:
            logger.error(f"SoS brands API error: {e}")
            return {"success": False, "error": str(e)}
