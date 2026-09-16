"""
SoS Trend Service
=================
Daily Share-of-Shelf series behind ``/api/sos/trend`` and
``/api/sos/trend/competitors-avg`` (F6). Split out of ``analytics_service`` so
each module stays one screenful of concern: point-in-time KPI/SoS there, time
series here.

The storage factory is injected; the synchronous sqlite3 work runs through
``asyncio.to_thread`` (D20) and date defaults come from ``date_range`` (KST).
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from collections.abc import Callable
from typing import Any

from src.application.services.date_range import resolve_date_range
from src.application.services.sql_rows import daily_totals_query, fetch_pair

logger = logging.getLogger(__name__)


class SosTrendService:
    """Daily SoS series for the target brand and for its competitors."""

    def __init__(self, sqlite_factory: Callable[[], Any]):
        self._sqlite_factory = sqlite_factory

    async def _storage(self) -> Any:
        sqlite = self._sqlite_factory()
        await sqlite.initialize()
        return sqlite

    # -------------------------------------------------------------------- trend
    async def sos_trend(
        self,
        brand: str = "LANEIGE",
        category_id: str | None = None,
        days: int = 7,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> dict[str, Any]:
        """Daily SoS series for one brand."""
        try:
            sqlite = await self._storage()

            # start_date/end_date가 제공되면 사용, 아니면 days 기반으로 계산
            if not (start_date and end_date):
                start_date, end_date = resolve_date_range(None, None, default_days=days)

            # 일별 전체 제품 수
            total_query, total_params = daily_totals_query(category_id, start_date, end_date)

            if category_id:
                brand_query = """
                    SELECT snapshot_date, COUNT(*) as brand_count
                    FROM raw_data
                    WHERE snapshot_date BETWEEN ? AND ?
                    AND category_id = ?
                    AND LOWER(brand) LIKE ?
                    GROUP BY snapshot_date
                    ORDER BY snapshot_date
                """
                brand_params: tuple = (start_date, end_date, category_id, f"%{brand.lower()}%")
            else:
                brand_query = """
                    SELECT snapshot_date, COUNT(*) as brand_count
                    FROM raw_data
                    WHERE snapshot_date BETWEEN ? AND ?
                    AND LOWER(brand) LIKE ?
                    GROUP BY snapshot_date
                    ORDER BY snapshot_date
                """
                brand_params = (start_date, end_date, f"%{brand.lower()}%")

            total_rows, brand_rows = await asyncio.to_thread(
                fetch_pair, sqlite, (total_query, total_params), (brand_query, brand_params)
            )
            total_by_date = {row[0]: row[1] for row in total_rows}
            brand_by_date = {row[0]: row[1] for row in brand_rows}

            # SoS 계산
            trend_data = []
            for date, total in sorted(total_by_date.items()):
                brand_count = brand_by_date.get(date, 0)
                sos = (brand_count / total * 100) if total > 0 else 0
                trend_data.append(
                    {
                        "date": date,
                        "total_products": total,
                        "brand_count": brand_count,
                        "sos": round(sos, 2),
                    }
                )

            return {
                "success": True,
                "brand": brand,
                "category_id": category_id,
                "period": {"start": start_date, "end": end_date, "days": days},
                "trend": trend_data,
            }

        except Exception as e:
            logger.error(f"SoS trend API error: {e}")
            return {"success": False, "error": str(e)}

    # ------------------------------------------------------- competitors average
    async def competitors_avg_sos_trend(
        self,
        category_id: str | None = None,
        days: int = 7,
        start_date: str | None = None,
        end_date: str | None = None,
        top_n: int = 10,
        exclude_brand: str = "LANEIGE",
    ) -> dict[str, Any]:
        """Daily average SoS of the Top N competitor brands (target brand excluded)."""
        try:
            sqlite = await self._storage()

            # 날짜 범위 결정
            if not (start_date and end_date):
                start_date, end_date = resolve_date_range(None, None, default_days=days)

            # 일별 전체 제품 수 쿼리
            total_query, total_params = daily_totals_query(category_id, start_date, end_date)

            if category_id:
                brand_daily_query = """
                    SELECT snapshot_date, brand, COUNT(*) as brand_count
                    FROM raw_data
                    WHERE snapshot_date BETWEEN ? AND ?
                    AND category_id = ?
                    AND LOWER(brand) NOT LIKE ?
                    AND brand IS NOT NULL
                    AND brand != ''
                    GROUP BY snapshot_date, brand
                    ORDER BY snapshot_date, brand_count DESC
                """
                brand_daily_params: tuple = (
                    start_date,
                    end_date,
                    category_id,
                    f"%{exclude_brand.lower()}%",
                )
            else:
                brand_daily_query = """
                    SELECT snapshot_date, brand, COUNT(*) as brand_count
                    FROM raw_data
                    WHERE snapshot_date BETWEEN ? AND ?
                    AND LOWER(brand) NOT LIKE ?
                    AND brand IS NOT NULL
                    AND brand != ''
                    GROUP BY snapshot_date, brand
                    ORDER BY snapshot_date, brand_count DESC
                """
                brand_daily_params = (start_date, end_date, f"%{exclude_brand.lower()}%")

            total_rows, brand_rows = await asyncio.to_thread(
                fetch_pair,
                sqlite,
                (total_query, total_params),
                (brand_daily_query, brand_daily_params),
            )
            total_by_date = {row[0]: row[1] for row in total_rows}

            # 일별로 Top N 브랜드의 평균 SoS 계산
            daily_brands: dict[str, list[tuple[str, int]]] = defaultdict(list)
            for date, brand, count in brand_rows:
                daily_brands[date].append((brand, count))

            trend_data = []
            for date, total in sorted(total_by_date.items()):
                brands_for_date = daily_brands.get(date, [])
                top_brands = brands_for_date[:top_n]

                if top_brands and total > 0:
                    sos_values = [(count / total * 100) for _, count in top_brands]
                    avg_sos = sum(sos_values) / len(sos_values)
                else:
                    avg_sos = 0

                trend_data.append(
                    {
                        "date": date,
                        "total_products": total,
                        "top_brands_count": len(top_brands),
                        "avg_sos": round(avg_sos, 2),
                    }
                )

            return {
                "success": True,
                "category_id": category_id,
                "excluded_brand": exclude_brand,
                "top_n": top_n,
                "period": {"start": start_date, "end": end_date, "days": days},
                "trend": trend_data,
            }

        except Exception as e:
            logger.error(f"Competitors avg SoS trend API error: {e}")
            return {"success": False, "error": str(e)}
