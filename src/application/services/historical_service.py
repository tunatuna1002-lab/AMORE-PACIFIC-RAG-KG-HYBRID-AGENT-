"""
Historical Data Service
=======================
The aggregation behind ``GET /api/historical`` (F6): daily SoS history, rank
history and the brand matrix, from SQLite first, Google Sheets second and the
local JSON caches last.

The route only validates the query string; everything below runs without FastAPI.
Collaborators are injected so a caller can point the service at any storage:

- ``sqlite_factory``    -> storage with ``initialize()`` / ``get_raw_data()`` / ``get_stats()``
- ``sheets_factory``    -> Google Sheets writer (``initialize()`` / ``get_raw_data()``)
- ``dashboard_loader``  -> ``dashboard_data.json`` as a dict (``{}`` when absent)
- ``data_service``      -> :class:`DashboardDataService` for data-directory paths
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from datetime import datetime
from typing import Any

from src.application.services.brand_matrix import (
    brand_metrics_for_period,
    brand_metrics_from_dashboard,
)
from src.application.services.date_range import DATE_FMT, today_kst

logger = logging.getLogger(__name__)


class HistoricalService:
    """Historical metrics for the dashboard's period view."""

    def __init__(
        self,
        sqlite_factory: Callable[[], Any],
        sheets_factory: Callable[[], Any],
        dashboard_loader: Callable[[], dict],
        data_service: Any,
    ):
        self._sqlite_factory = sqlite_factory
        self._sheets_factory = sheets_factory
        self._dashboard_loader = dashboard_loader
        self._data_service = data_service

    # --------------------------------------------------------------- composite
    async def historical(
        self,
        start_date: str,
        end_date: str,
        category_id: str | None = None,
        brand: str | None = "LANEIGE",
    ) -> dict[str, Any]:
        """
        히스토리컬 데이터 조회 (SQLite 우선, Google Sheets fallback)

        Returns:
            - data: 날짜별 지표 데이터
            - sos_history: SoS 추이 데이터
            - raw_data: 순위 추이 데이터
        """
        try:
            records: list[dict] = []
            data_source = None

            # 1차: SQLite에서 조회 (빠름)
            try:
                sqlite = self._sqlite_factory()
                await sqlite.initialize()
                records = await sqlite.get_raw_data(
                    start_date=start_date,
                    end_date=end_date,
                    category_id=category_id,
                    limit=50000,
                )
                if records:
                    data_source = "sqlite"
                    logger.info(
                        f"Historical: loaded {len(records)} records from SQLite "
                        f"({start_date} ~ {end_date})"
                    )
            except Exception as sqlite_err:
                logger.warning(f"Historical: SQLite 조회 실패: {sqlite_err}")

            # 2차: SQLite 실패/빈 결과 시 Google Sheets fallback
            if not records:
                try:
                    sheets_writer = self._sheets_factory()
                    if not sheets_writer._initialized:
                        await sheets_writer.initialize()
                    records = await sheets_writer.get_raw_data(
                        start_date=start_date, end_date=end_date, category_id=category_id
                    )
                    if records:
                        data_source = "sheets"
                        logger.info(
                            f"Historical: loaded {len(records)} records from Sheets "
                            f"({start_date} ~ {end_date})"
                        )
                except Exception as sheets_err:
                    logger.warning(f"Historical: Google Sheets 조회 실패: {sheets_err}")

            if not records:
                return await self.from_local(start_date, end_date, brand)

            # 날짜 범위 계산
            start_dt = datetime.strptime(start_date, DATE_FMT)
            end_dt = datetime.strptime(end_date, DATE_FMT)
            days = (end_dt - start_dt).days + 1

            # 날짜별 데이터 집계 (특정 브랜드 필터링)
            daily_data: dict[str, dict[str, Any]] = {}
            brand_lower = brand.lower() if brand else ""
            for record in records:
                snapshot_date = record.get("snapshot_date", "")
                if not snapshot_date or snapshot_date < start_date or snapshot_date > end_date:
                    continue

                record_brand = record.get("brand", "")
                if brand_lower and record_brand.lower() != brand_lower:
                    continue

                if snapshot_date not in daily_data:
                    daily_data[snapshot_date] = {
                        "date": snapshot_date,
                        "products": [],
                        "total_count": 0,
                        "top10_count": 0,
                    }

                rank = int(record.get("rank", 0)) if record.get("rank") else 0
                daily_data[snapshot_date]["products"].append(
                    {
                        "asin": record.get("asin", ""),
                        "product_name": record.get("product_name", ""),
                        "brand": record_brand,
                        "rank": rank,
                        "price": record.get("price", ""),
                        "rating": record.get("rating", ""),
                    }
                )
                daily_data[snapshot_date]["total_count"] += 1
                if rank <= 10:
                    daily_data[snapshot_date]["top10_count"] += 1

            # SoS 추이 계산
            sos_history = []
            raw_data = []
            for date_str in sorted(daily_data.keys()):
                day_data = daily_data[date_str]
                products = day_data["products"]

                sos = round(len(products) / 100 * 100, 1) if products else 0
                sos_history.append(
                    {
                        "date": date_str,
                        "sos": sos,
                        "product_count": len(products),
                        "top10_count": day_data["top10_count"],
                    }
                )

                if products:
                    avg_rank = round(sum(p["rank"] for p in products) / len(products), 1)
                    raw_data.append(
                        {
                            "date": date_str,
                            "rank": avg_rank,
                            "best_rank": min(p["rank"] for p in products),
                            "worst_rank": max(p["rank"] for p in products),
                        }
                    )

            available_dates = sorted(daily_data.keys())

            # brand_metrics 계산 (전체 기간 통합 - 모든 브랜드 포함)
            brand_metrics = brand_metrics_for_period(records, brand)

            # rank_history 생성 (Product View 차트용)
            rank_history: dict[str, dict[str, Any]] = {}
            for record in records:
                snapshot_date = record.get("snapshot_date", "")
                if not snapshot_date or snapshot_date < start_date or snapshot_date > end_date:
                    continue

                if snapshot_date not in rank_history:
                    rank_history[snapshot_date] = {"products": []}

                rank = int(record.get("rank", 0)) if record.get("rank") else 0
                price = _parse_price(record.get("price", 0))

                rank_history[snapshot_date]["products"].append(
                    {
                        "name": record.get("product_name", ""),
                        "product_name": record.get("product_name", ""),
                        "brand": record.get("brand", ""),
                        "asin": record.get("asin", ""),
                        "rank": rank,
                        "price": price,
                        "rating": record.get("rating", ""),
                        "discount_percent": record.get("discount_percent", 0),
                    }
                )

            # 전체 데이터의 사용 가능한 날짜 범위 조회
            available_date_range: dict[str, Any] = {"min": None, "max": None}
            try:
                sqlite = self._sqlite_factory()
                stats = sqlite.get_stats()
                if "date_range" in stats:
                    available_date_range = stats["date_range"]
            except Exception:
                pass

            return {
                "success": True,
                "available_dates": available_dates,
                "available_date_range": available_date_range,
                "data_source": data_source,
                "brand_metrics": brand_metrics,
                "rank_history": rank_history,
                "data": {
                    "sos_history": sos_history,
                    "raw_data": raw_data,
                    "daily_data": list(daily_data.values()),
                    "period": {"start": start_date, "end": end_date, "days": days},
                    "brand": brand,
                },
            }

        except Exception as e:
            logger.error(f"Historical data error: {e}")
            return await self.from_local(start_date, end_date, brand)

    # ------------------------------------------------------------ local fallback
    async def from_local(
        self, start_date: str, end_date: str, brand: str = "LANEIGE"
    ) -> dict[str, Any]:
        """로컬 JSON 파일에서 히스토리컬 데이터 조회 (폴백)"""
        try:
            data = self._dashboard_loader()
            sos_history: list[dict[str, Any]] = []
            raw_data: list[dict[str, Any]] = []

            # 1. 대시보드 데이터에서 현재 SoS/순위 정보 추출
            if data:
                brand_kpis = data.get("brand", {}).get("kpis", {})
                current_sos = brand_kpis.get("sos", 0)
                data_date = data.get("metadata", {}).get(
                    "data_date", today_kst().strftime(DATE_FMT)
                )

                if start_date <= data_date <= end_date:
                    sos_history.append(
                        {
                            "date": data_date,
                            "sos": current_sos,
                            "product_count": brand_kpis.get("product_count", 0),
                            "top10_count": brand_kpis.get("top10_count", 0),
                        }
                    )

                    avg_rank = brand_kpis.get("avg_rank", 0)
                    if avg_rank:
                        raw_data.append(
                            {
                                "date": data_date,
                                "rank": avg_rank,
                                "best_rank": brand_kpis.get("best_rank", avg_rank),
                                "worst_rank": brand_kpis.get("worst_rank", avg_rank),
                            }
                        )

            # 2. latest_crawl_result.json에서 데이터 추출
            latest_crawl_path = self._data_service.latest_crawl_json_path
            if latest_crawl_path.exists():
                try:
                    with open(latest_crawl_path, encoding="utf-8") as f:
                        crawl_data = json.load(f)

                    brand_products = []
                    crawl_date = None

                    for _cat_id, cat_data in crawl_data.get("categories", {}).items():
                        for product in cat_data.get("products", []):
                            product_brand = product.get("brand", "")
                            product_name = product.get("product_name", "")

                            if (
                                brand.upper() in product_brand.upper()
                                or brand.upper() in product_name.upper()
                            ):
                                brand_products.append(product)
                                if not crawl_date:
                                    crawl_date = product.get("snapshot_date")

                    if brand_products and crawl_date and start_date <= crawl_date <= end_date:
                        if not any(h["date"] == crawl_date for h in sos_history):
                            total_products = sum(
                                len(cat.get("products", []))
                                for cat in crawl_data.get("categories", {}).values()
                            )

                            sos = round(len(brand_products) / max(total_products, 100) * 100, 2)
                            avg_rank = round(
                                sum(p.get("rank", 0) for p in brand_products) / len(brand_products),
                                1,
                            )

                            sos_history.append(
                                {
                                    "date": crawl_date,
                                    "sos": sos,
                                    "product_count": len(brand_products),
                                    "top10_count": sum(
                                        1 for p in brand_products if p.get("rank", 100) <= 10
                                    ),
                                }
                            )
                            raw_data.append(
                                {
                                    "date": crawl_date,
                                    "rank": avg_rank,
                                    "best_rank": min(p.get("rank", 100) for p in brand_products),
                                    "worst_rank": max(p.get("rank", 100) for p in brand_products),
                                }
                            )

                except (json.JSONDecodeError, ValueError) as e:
                    logger.warning(f"Failed to parse latest_crawl_result.json: {e}")

            # 3. raw_products 폴더에서 날짜별 데이터 검색
            raw_data_dir = self._data_service.path_for("raw_products")
            if raw_data_dir.exists():
                for json_file in raw_data_dir.glob("*.json"):
                    try:
                        file_date = json_file.stem
                        if start_date <= file_date <= end_date:
                            with open(json_file, encoding="utf-8") as f:
                                daily_raw = json.load(f)

                            brand_products = [
                                p
                                for p in daily_raw
                                if brand.upper() in p.get("brand", "").upper()
                                or brand.upper() in p.get("product_name", "").upper()
                            ]

                            if brand_products:
                                sos = round(len(brand_products) / 100 * 100, 1)
                                avg_rank = round(
                                    sum(p.get("rank", 0) for p in brand_products)
                                    / len(brand_products),
                                    1,
                                )

                                if not any(h["date"] == file_date for h in sos_history):
                                    sos_history.append(
                                        {
                                            "date": file_date,
                                            "sos": sos,
                                            "product_count": len(brand_products),
                                            "top10_count": sum(
                                                1
                                                for p in brand_products
                                                if p.get("rank", 100) <= 10
                                            ),
                                        }
                                    )
                                    raw_data.append(
                                        {
                                            "date": file_date,
                                            "rank": avg_rank,
                                            "best_rank": min(
                                                p.get("rank", 100) for p in brand_products
                                            ),
                                            "worst_rank": max(
                                                p.get("rank", 100) for p in brand_products
                                            ),
                                        }
                                    )
                    except (json.JSONDecodeError, ValueError):
                        continue

            sos_history.sort(key=lambda x: x["date"])
            raw_data.sort(key=lambda x: x["date"])

            available_dates = [h["date"] for h in sos_history]
            brand_metrics = brand_metrics_from_dashboard(data, brand)

            # rank_history 생성 (CPI 차트용)
            rank_history: dict[str, dict[str, Any]] = {}
            latest_crawl_path = self._data_service.latest_crawl_json_path
            if latest_crawl_path.exists():
                try:
                    with open(latest_crawl_path, encoding="utf-8") as f:
                        crawl_data = json.load(f)
                    for _cat_id, cat_data in crawl_data.get("categories", {}).items():
                        for product in cat_data.get("products", []):
                            snap_date = product.get("snapshot_date", "")
                            if not snap_date or snap_date < start_date or snap_date > end_date:
                                continue
                            if snap_date not in rank_history:
                                rank_history[snap_date] = {"products": []}
                            rank_history[snap_date]["products"].append(
                                {
                                    "name": product.get("product_name", ""),
                                    "brand": product.get("brand", ""),
                                    "rank": product.get("rank", 0),
                                    "price": _parse_price(product.get("price", 0)),
                                }
                            )
                except (json.JSONDecodeError, ValueError) as e:
                    logger.warning(f"Failed to build rank_history from local: {e}")

            if not sos_history:
                return {
                    "success": False,
                    "error": "No historical data found for the specified period",
                    "available_dates": [],
                    "brand_metrics": [],
                    "rank_history": rank_history,
                    "data": None,
                }

            return {
                "success": True,
                "available_dates": available_dates,
                "brand_metrics": brand_metrics,
                "rank_history": rank_history,
                "data": {
                    "sos_history": sos_history,
                    "raw_data": raw_data,
                    "period": {"start": start_date, "end": end_date},
                    "brand": brand,
                    "source": "local",
                },
            }

        except Exception as e:
            logger.error(f"Local historical data error: {e}")
            return {
                "success": False,
                "error": str(e),
                "available_dates": [],
                "brand_metrics": [],
                "data": None,
            }


def _parse_price(value: Any) -> float:
    """``"$24.00"`` / ``24.0`` / ``""`` -> float (0 when unparsable)."""
    try:
        return float(str(value).replace("$", "").replace(",", "")) if value else 0
    except (ValueError, TypeError):
        return 0
