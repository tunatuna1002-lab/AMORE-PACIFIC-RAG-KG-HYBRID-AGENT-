"""
스냅샷 지표 영속화 (raw_data → brand_metrics / market_metrics)

raw_data에 저장된 크롤 레코드를 정본 계산기(count_brands / calculate_hhi_from_counts /
calculate_sos_pct)로 집계해 지표 테이블을 채운다.

사용처:
- scripts/daily_crawl.py (launchd 일일 크롤): 크롤 저장 직후 해당 날짜 지표 저장
- scripts/backfill_metrics.py: 과거 날짜 일괄 재계산

지표를 크롤 결과(메모리)가 아니라 raw_data에서 다시 읽어 계산하는 이유:
raw_data는 UNIQUE(snapshot_date, category_id, rank)에 INSERT OR REPLACE로 쓰인다.
같은 날짜에 크롤이 두 번 돌면 raw_data는 나중 크롤로 덮이는데, 지표가 먼저 계산돼
있으면 테이블 사이 수치가 어긋난다 (2026-08-31: 01시 수동 크롤로 계산된 지표가
22시 정기 크롤 이후에도 남아 있었다). 항상 저장된 raw_data 기준으로 날짜 단위
교체하면 두 테이블이 같은 스냅샷을 가리킨다.
"""

import logging
from collections import defaultdict
from collections.abc import Iterable
from typing import Any, Protocol

from src.tools.calculators.metric_calculator import (
    calculate_hhi_from_counts,
    calculate_sos_pct,
    count_brands,
)

logger = logging.getLogger(__name__)

# 한 날짜의 raw_data 행 수 상한 (5 카테고리 × Top 100 = 500행, 여유분 포함)
_MAX_ROWS_PER_DATE = 100_000


class MetricSnapshotStorage(Protocol):
    """persist_metrics_for_dates가 요구하는 저장소 인터페이스 (SQLiteStorage 호환)"""

    async def get_raw_data(
        self,
        start_date: str | None = None,
        end_date: str | None = None,
        category_id: str | None = None,
        brand: str | None = None,
        limit: int = 1000,
    ) -> list[dict[str, Any]]: ...

    async def replace_metrics_for_date(
        self,
        snapshot_date: str,
        brand_rows: list[dict[str, Any]],
        market_rows: list[dict[str, Any]],
    ) -> tuple[int, int]: ...


def _safe_float(value: Any) -> float | None:
    try:
        cleaned = str(value).replace("$", "").replace(",", "").strip()
        return float(cleaned) if cleaned else None
    except (ValueError, TypeError):
        return None


def build_metric_rows(records: list[dict]) -> tuple[list[dict], list[dict]]:
    """raw_data 레코드에서 (brand_metrics 행, market_metrics 행)을 만든다."""
    by_day_cat: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in records:
        date = r.get("snapshot_date")
        category = r.get("category_id")
        if date and category:
            by_day_cat[(date, category)].append(r)

    brand_rows: list[dict] = []
    market_rows: list[dict] = []

    for (date, category), rows in sorted(by_day_cat.items()):
        total = len(rows)
        brand_counts = count_brands(rows)

        # --- 시장 지표 ---
        prices = [p for p in (_safe_float(r.get("price")) for r in rows) if p and 0.5 <= p <= 500]
        ratings = [p for p in (_safe_float(r.get("rating")) for r in rows) if p is not None]
        market_rows.append(
            {
                "snapshot_date": date,
                "category_id": category,
                "hhi": calculate_hhi_from_counts(brand_counts),
                "churn_rate": None,  # 전일 비교 필요 — 스냅샷 단위 집계 범위 밖
                "category_avg_price": round(sum(prices) / len(prices), 2) if prices else None,
                "category_avg_rating": round(sum(ratings) / len(ratings), 2) if ratings else None,
            }
        )

        # --- 브랜드 지표 ---
        cat_avg_price = sum(prices) / len(prices) if prices else None
        cat_avg_rating = sum(ratings) / len(ratings) if ratings else None

        by_brand: dict[str, list[dict]] = defaultdict(list)
        for r in rows:
            name = (r.get("brand") or "").strip()
            if name in brand_counts:
                by_brand[name].append(r)

        for name, brand_rows_for_name in by_brand.items():
            ranks = [int(r["rank"]) for r in brand_rows_for_name if r.get("rank")]
            b_prices = [
                p
                for p in (_safe_float(r.get("price")) for r in brand_rows_for_name)
                if p and 0.5 <= p <= 500
            ]
            b_ratings = [
                p
                for p in (_safe_float(r.get("rating")) for r in brand_rows_for_name)
                if p is not None
            ]

            cpi = None
            if b_prices and cat_avg_price:
                cpi = round(sum(b_prices) / len(b_prices) / cat_avg_price * 100, 1)

            rating_gap = None
            if b_ratings and cat_avg_rating is not None:
                rating_gap = round(sum(b_ratings) / len(b_ratings) - cat_avg_rating, 3)

            brand_rows.append(
                {
                    "snapshot_date": date,
                    "category_id": category,
                    "brand": name,
                    "sos": calculate_sos_pct(len(brand_rows_for_name), total),
                    "brand_avg_rank": round(sum(ranks) / len(ranks), 2) if ranks else None,
                    "product_count": len(brand_rows_for_name),
                    "cpi": cpi,
                    "avg_rating_gap": rating_gap,
                }
            )

    return brand_rows, market_rows


async def persist_metrics_for_dates(
    storage: MetricSnapshotStorage, dates: Iterable[str]
) -> dict[str, dict[str, int]]:
    """날짜별로 raw_data를 읽어 지표를 재계산하고 해당 날짜 지표를 통째로 교체한다.

    raw_data가 0행인 날짜는 건너뛴다 (기존 지표를 지우지 않는다).

    Returns:
        {snapshot_date: {"raw_rows": n, "brand_rows": n, "market_rows": n}}
        — 건너뛴 날짜는 모두 0
    """
    summary: dict[str, dict[str, int]] = {}
    for date in sorted(set(dates)):
        records = await storage.get_raw_data(
            start_date=date, end_date=date, limit=_MAX_ROWS_PER_DATE
        )
        if not records:
            logger.warning(f"raw_data에 {date} 레코드가 없어 지표 저장을 건너뜁니다")
            summary[date] = {"raw_rows": 0, "brand_rows": 0, "market_rows": 0}
            continue

        brand_rows, market_rows = build_metric_rows(records)
        saved_brand, saved_market = await storage.replace_metrics_for_date(
            date, brand_rows, market_rows
        )
        logger.info(
            f"Metrics persisted from raw_data: {date} "
            f"(raw={len(records)}, brand={saved_brand}, market={saved_market})"
        )
        summary[date] = {
            "raw_rows": len(records),
            "brand_rows": saved_brand,
            "market_rows": saved_market,
        }
    return summary
