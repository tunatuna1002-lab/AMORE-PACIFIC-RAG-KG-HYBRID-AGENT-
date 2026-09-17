"""
Metric Facts Provider
=====================
크롤 DB(SQLite, 지표의 정본)에서 질의 엔티티에 해당하는 수치를 조회해 검색 컨텍스트에 싣는다.

배경 (사이클 10, docs/experiments/eval_cycle10_2026-09-12.md §2)
-------------------------------------------------------------
챗봇은 DB에 있는 SoS·HHI·순위·리뷰 수를 "데이터에 없습니다"라고 답했다. 수치가
프롬프트에 닿는 경로가 네 군데서 끊겨 있었다: current_metrics가 읽는 키가 운영
데이터(dashboard_data.json)에 없고, KG 메트릭 엣지는 렌더링되지 않고, 카테고리 주어
엣지는 조회되지 않고, KG의 메트릭 트리플은 버전 없이 낡은 값이 섞여 있다.
그래서 수치는 KG가 아니라 SQLite에서 직접, 스냅샷 날짜를 붙여 가져온다.

데이터 시점
-----------
기본은 테이블별 최신 스냅샷이다. `AMORE_DATA_AS_OF`(YYYY-MM-DD)가 설정되면 그 날짜
이하의 최신 스냅샷을 쓴다 — 골든셋 snapshot 문항의 골드가 특정 날짜 DB에서 생성됐기
때문에 평가는 같은 날짜를 읽어야 비교가 성립한다(eval/cli.py가 설정한다).

원칙
----
- 값이 없으면(NULL) 사실에 넣지 않는다. 0이나 기본값으로 채우지 않는다.
- 브랜드 부재("Top 100 내 없음")는 해당 카테고리·날짜에 데이터가 실제로 있을 때만
  단정한다. 표본 부족으로 산출하지 않은 날을 0%로 오독하지 않기 위해서다.
- DB는 읽기 전용으로 연다. 파일이 없으면 아무것도 조회하지 않는다(빈 DB를 만들지 않는다).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

AS_OF_ENV = "AMORE_DATA_AS_OF"

MAX_CATEGORIES = 3
MAX_BRANDS = 3
TOP_BRANDS = 5
TOP_PRODUCTS = 5
BRAND_PRODUCTS = 3

_TABLES = ("brand_metrics", "market_metrics", "raw_data")
_MARKET_FIELDS = ("hhi", "churn_rate", "category_avg_price", "category_avg_rating")
# 질의 브랜드의 brand_share에만 싣는 지표 (규칙 입력: strong_avg_rank·price_quality_mismatch·
# value_position·premium_price_position·strong_rating_position). 상위 브랜드 목록에는 싣지 않는다.
_BRAND_EXTRA_FIELDS = ("brand_avg_rank", "cpi", "avg_rating_gap")


def _product(row: Any) -> dict[str, Any]:
    name = (row["product_name"] or "").split(":")[0].strip()[:60]
    product: dict[str, Any] = {"rank": row["rank"], "brand": row["brand"], "name": name}
    for key in ("price", "rating", "reviews_count"):
        if row[key] is not None:
            product[key] = row[key]
    return product


class MetricFactsProvider:
    """질의 엔티티(브랜드·카테고리)에 대한 크롤 DB 수치 사실."""

    def __init__(self, db_path: str | Path | None = None, as_of: str | None = None):
        """
        Args:
            db_path: SQLite 경로. None이면 SQLiteStorage의 경로 해석(Railway/로컬)을 따른다.
            as_of: 데이터 시점 상한. None이면 AMORE_DATA_AS_OF, 그것도 없으면 최신.
        """
        self._db_path = Path(db_path) if db_path else None
        self._as_of = as_of

    @property
    def as_of(self) -> str | None:
        return self._as_of or os.environ.get(AS_OF_ENV) or None

    def _resolve_db_path(self) -> Path:
        if self._db_path is not None:
            return self._db_path
        from src.tools.storage.sqlite_storage import get_sqlite_storage

        return Path(get_sqlite_storage().db_path)

    async def collect(self, entities: dict[str, list[str]]) -> list[dict[str, Any]]:
        """엔티티에 해당하는 수치 사실 목록. 엔티티가 없거나 DB가 없으면 빈 리스트."""
        brands = [b for b in (entities.get("brands") or []) if b][:MAX_BRANDS]
        categories = [c for c in (entities.get("categories") or []) if c]
        if not brands and not categories:
            return []

        db_path = self._resolve_db_path()
        if not db_path.exists():
            logger.debug(f"Metric facts skipped: DB not found at {db_path}")
            return []

        import aiosqlite

        async with aiosqlite.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
            conn.row_factory = aiosqlite.Row
            return await self._collect(conn, brands, categories)

    async def _latest(self, conn: Any, table: str) -> str | None:
        assert table in _TABLES  # 테이블명은 고정 목록에서만 온다
        if self.as_of:
            cursor = await conn.execute(
                f"SELECT MAX(snapshot_date) FROM {table} WHERE snapshot_date <= ?", (self.as_of,)
            )
        else:
            cursor = await conn.execute(f"SELECT MAX(snapshot_date) FROM {table}")
        row = await cursor.fetchone()
        return row[0] if row and row[0] else None

    async def _collect(
        self, conn: Any, brands: list[str], categories: list[str]
    ) -> list[dict[str, Any]]:
        bm_date = await self._latest(conn, "brand_metrics")
        mm_date = await self._latest(conn, "market_metrics")
        raw_date = await self._latest(conn, "raw_data")
        lowered = [b.lower() for b in brands]

        if not categories and brands and bm_date:
            # 브랜드만 링크됐으면 그 브랜드가 진입한 카테고리를 점유율 순으로
            placeholders = ",".join("?" * len(lowered))
            cursor = await conn.execute(
                f"""
                SELECT category_id, MAX(sos) AS top_sos FROM brand_metrics
                 WHERE snapshot_date = ? AND LOWER(brand) IN ({placeholders})
                 GROUP BY category_id ORDER BY top_sos DESC
                """,
                (bm_date, *lowered),
            )
            categories = [row["category_id"] for row in await cursor.fetchall()]

        facts: list[dict[str, Any]] = []
        for category in categories[:MAX_CATEGORIES]:
            if mm_date:
                facts.extend(await self._market_fact(conn, category, mm_date))
            if bm_date:
                facts.extend(await self._share_facts(conn, category, bm_date, brands))
            if raw_date:
                facts.extend(await self._product_facts(conn, category, raw_date, brands))
        return facts

    async def _market_fact(self, conn: Any, category: str, date: str) -> list[dict[str, Any]]:
        cursor = await conn.execute(
            """
            SELECT hhi, churn_rate, category_avg_price, category_avg_rating
              FROM market_metrics WHERE snapshot_date = ? AND category_id = ?
            """,
            (date, category),
        )
        row = await cursor.fetchone()
        if not row:
            return []
        fact: dict[str, Any] = {
            "type": "category_market",
            "category": category,
            "snapshot_date": date,
        }
        for key in _MARKET_FIELDS:
            if row[key] is not None:
                fact[key] = row[key]
        return [fact] if len(fact) > 3 else []

    async def _share_facts(
        self, conn: Any, category: str, date: str, brands: list[str]
    ) -> list[dict[str, Any]]:
        cursor = await conn.execute(
            """
            SELECT brand, sos, product_count FROM brand_metrics
             WHERE snapshot_date = ? AND category_id = ? AND sos IS NOT NULL
             ORDER BY sos DESC, brand LIMIT ?
            """,
            (date, category, TOP_BRANDS),
        )
        top = await cursor.fetchall()
        if not top:
            return []  # 이 날짜·카테고리에 점유율이 없다 — 부재도 단정하지 않는다

        facts: list[dict[str, Any]] = [
            {
                "type": "category_top_brands",
                "category": category,
                "snapshot_date": date,
                "brands": [
                    {"brand": r["brand"], "sos": r["sos"], "product_count": r["product_count"]}
                    for r in top
                ],
            }
        ]
        for brand in brands:
            cursor = await conn.execute(
                """
                SELECT b1.brand, b1.sos, b1.product_count,
                       b1.brand_avg_rank, b1.cpi, b1.avg_rating_gap,
                       (SELECT COUNT(*) + 1 FROM brand_metrics b2
                         WHERE b2.snapshot_date = b1.snapshot_date
                           AND b2.category_id = b1.category_id AND b2.sos > b1.sos) AS brand_rank
                  FROM brand_metrics b1
                 WHERE b1.snapshot_date = ? AND b1.category_id = ? AND LOWER(b1.brand) = LOWER(?)
                """,
                (date, category, brand),
            )
            row = await cursor.fetchone()
            if row and row["sos"] is not None:
                share: dict[str, Any] = {
                    "type": "brand_share",
                    "brand": row["brand"],
                    "category": category,
                    "snapshot_date": date,
                    "present": True,
                    "sos": row["sos"],
                    "product_count": row["product_count"],
                    "brand_rank": row["brand_rank"],
                }
                for key in _BRAND_EXTRA_FIELDS:
                    if row[key] is not None:  # NULL은 넣지 않는다 (0으로 채우지 않음)
                        share[key] = row[key]
                facts.append(share)
            else:
                facts.append(
                    {
                        "type": "brand_share",
                        "brand": brand,
                        "category": category,
                        "snapshot_date": date,
                        "present": False,
                    }
                )
        return facts

    async def _product_facts(
        self, conn: Any, category: str, date: str, brands: list[str]
    ) -> list[dict[str, Any]]:
        columns = "rank, brand, product_name, price, rating, reviews_count"
        cursor = await conn.execute(
            f"SELECT {columns} FROM raw_data WHERE snapshot_date = ? AND category_id = ?"
            " ORDER BY rank LIMIT ?",
            (date, category, TOP_PRODUCTS),
        )
        top = await cursor.fetchall()
        if not top:
            return []

        facts: list[dict[str, Any]] = [
            {
                "type": "category_top_products",
                "category": category,
                "snapshot_date": date,
                "products": [_product(r) for r in top],
            }
        ]
        for brand in brands:
            cursor = await conn.execute(
                f"SELECT {columns} FROM raw_data WHERE snapshot_date = ? AND category_id = ?"
                " AND LOWER(brand) = LOWER(?) ORDER BY rank LIMIT ?",
                (date, category, brand, BRAND_PRODUCTS),
            )
            rows = await cursor.fetchall()
            if rows:
                facts.append(
                    {
                        "type": "brand_products",
                        "brand": rows[0]["brand"],
                        "category": category,
                        "snapshot_date": date,
                        "products": [_product(r) for r in rows],
                    }
                )
        return facts
