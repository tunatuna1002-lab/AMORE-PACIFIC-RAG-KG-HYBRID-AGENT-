"""
지표 백필 스크립트 (Phase 5 / §5.2)

brand_metrics / market_metrics 테이블이 0행인 과거 날짜를 raw_data에서 재계산해 채운다.
STORE_METRICS 스텝(BatchWorkflow)이 도입되기 전 데이터가 대상이며, 1회만 실행하면 된다.

사용법:
    python3 scripts/backfill_metrics.py                    # 전체 기간
    python3 scripts/backfill_metrics.py --days 30          # 최근 30일
    python3 scripts/backfill_metrics.py --dry-run          # 저장 없이 집계만 출력
    python3 scripts/backfill_metrics.py --overwrite        # 이미 있는 날짜도 다시 계산
"""

import argparse
import asyncio
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tools.calculators.metric_calculator import (  # noqa: E402
    calculate_hhi_from_counts,
    calculate_sos_pct,
    count_brands,
)
from src.tools.storage.sqlite_storage import get_sqlite_storage  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _safe_float(value: Any) -> float | None:
    try:
        cleaned = str(value).replace("$", "").replace(",", "").strip()
        return float(cleaned) if cleaned else None
    except (ValueError, TypeError):
        return None


def build_rows(records: list[dict]) -> tuple[list[dict], list[dict]]:
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
                "churn_rate": None,  # 전일 비교 필요 — 백필 범위 밖
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


async def main() -> None:
    parser = argparse.ArgumentParser(description="brand_metrics / market_metrics 백필")
    parser.add_argument("--days", type=int, default=None, help="최근 N일만 처리 (기본: 전체)")
    parser.add_argument("--dry-run", action="store_true", help="저장하지 않고 집계만 출력")
    parser.add_argument(
        "--overwrite", action="store_true", help="이미 지표가 있는 날짜도 다시 계산"
    )
    args = parser.parse_args()

    storage = get_sqlite_storage()
    await storage.initialize()

    where = "WHERE snapshot_date >= date('now', ?)" if args.days else ""
    params = (f"-{args.days} day",) if args.days else ()

    async with storage.get_async_connection() as conn:
        cursor = await conn.execute(f"SELECT * FROM raw_data {where}", params)
        records = [dict(row) for row in await cursor.fetchall()]

    if not records:
        logger.warning("raw_data에 대상 레코드가 없습니다")
        return

    dates = sorted({r["snapshot_date"] for r in records if r.get("snapshot_date")})
    logger.info(f"대상: {len(records)}행, {len(dates)}일 ({dates[0]} ~ {dates[-1]})")

    if not args.overwrite:
        existing = {
            row["snapshot_date"] for row in await storage.get_brand_metrics(dates[0], dates[-1])
        }
        if existing:
            logger.info(
                f"이미 지표가 있는 날짜 {len(existing)}일은 건너뜁니다 (--overwrite로 무시)"
            )
            records = [r for r in records if r.get("snapshot_date") not in existing]
            if not records:
                logger.info("모든 날짜가 이미 처리돼 있습니다")
                return

    brand_rows, market_rows = build_rows(records)
    logger.info(f"생성: brand_metrics {len(brand_rows)}행, market_metrics {len(market_rows)}행")

    if args.dry_run:
        for row in market_rows[:5]:
            logger.info(f"  [dry-run] {row}")
        logger.info("--dry-run 이므로 저장하지 않았습니다")
        return

    saved_brand = await storage.save_brand_metrics(brand_rows) if brand_rows else 0
    saved_market = await storage.save_market_metrics(market_rows) if market_rows else 0
    logger.info(f"저장 완료: brand_metrics {saved_brand}행, market_metrics {saved_market}행")


if __name__ == "__main__":
    asyncio.run(main())
