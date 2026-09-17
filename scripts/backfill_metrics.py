"""
지표 백필 스크립트 (Phase 5 / §5.2)

brand_metrics / market_metrics 테이블이 0행인 과거 날짜를 raw_data에서 재계산해 채운다.
대상: STORE_METRICS 스텝 도입 전 데이터, 그리고 launchd 일일 크롤(scripts/daily_crawl.py)에
지표 스텝이 없던 기간(2026-09-01~)의 누락 날짜. 행 생성 로직은
src/tools/calculators/metric_snapshot.py(일일 크롤과 공용)에 있다.

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

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tools.calculators.metric_snapshot import build_metric_rows  # noqa: E402
from src.tools.storage.sqlite_storage import get_sqlite_storage  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


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

    brand_rows, market_rows = build_metric_rows(records)
    logger.info(f"생성: brand_metrics {len(brand_rows)}행, market_metrics {len(market_rows)}행")

    if args.dry_run:
        for row in market_rows[:5]:
            logger.info(f"  [dry-run] {row}")
        logger.info("--dry-run 이므로 저장하지 않았습니다")
        return

    # 날짜 단위 교체: --overwrite 시 재크롤로 사라진 브랜드 행이 남지 않게 한다
    brand_by_date: dict[str, list[dict]] = defaultdict(list)
    market_by_date: dict[str, list[dict]] = defaultdict(list)
    for row in brand_rows:
        brand_by_date[row["snapshot_date"]].append(row)
    for row in market_rows:
        market_by_date[row["snapshot_date"]].append(row)

    saved_brand = saved_market = 0
    for date in sorted(set(brand_by_date) | set(market_by_date)):
        b, m = await storage.replace_metrics_for_date(
            date, brand_by_date[date], market_by_date[date]
        )
        saved_brand += b
        saved_market += m
    logger.info(f"저장 완료: brand_metrics {saved_brand}행, market_metrics {saved_market}행")


if __name__ == "__main__":
    asyncio.run(main())
