"""
JSON File Repository Implementation
===================================
ProductRepository Protocol 구현 - 로컬 JSON 파일 백엔드

로컬 개발 및 테스트용으로 사용됩니다.
"""

import asyncio
import json
import logging
from datetime import date, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

from src.domain.entities.brand import BrandMetrics
from src.domain.entities.market import MarketMetrics
from src.domain.entities.product import RankRecord
from src.domain.interfaces.repository import MetricsRepository, ProductRepository


def _read_json(path):
    """동기 JSON 읽기 (to_thread에서 사용)"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _write_json(path, data, **kwargs):
    """동기 JSON 쓰기 (to_thread에서 사용)"""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, **kwargs)


class JsonFileRepository(ProductRepository, MetricsRepository):
    """
    JSON 파일을 이용한 Repository 구현

    로컬 개발 및 테스트용입니다.
    """

    def __init__(self, data_dir: Path = None):
        self.data_dir = data_dir or Path("./data")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self._records_file = self.data_dir / "rank_records.json"
        self._brand_metrics_file = self.data_dir / "brand_metrics.json"
        self._market_metrics_file = self.data_dir / "market_metrics.json"

    async def initialize(self) -> None:
        """초기화 (파일 기반이므로 별도 초기화 불필요)"""
        pass

    def _date_serializer(self, obj: Any) -> str:
        """JSON 직렬화 헬퍼"""
        if isinstance(obj, (date, datetime)):
            return obj.isoformat()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    async def save_records(self, records: list[RankRecord]) -> bool:
        """레코드 저장"""
        try:
            # Load existing records
            existing = []
            if self._records_file.exists():
                existing = await asyncio.to_thread(_read_json, self._records_file)

            # Convert new records to dicts
            new_records = [
                {
                    "snapshot_date": str(r.snapshot_date),
                    "category_id": r.category_id,
                    "asin": r.asin,
                    "product_name": r.product_name,
                    "brand": r.brand,
                    "rank": r.rank,
                    "price": r.price,
                    "list_price": r.list_price,
                    "discount_percent": r.discount_percent,
                    "rating": r.rating,
                    "reviews_count": r.reviews_count,
                    "badge": r.badge,
                    "coupon_text": r.coupon_text,
                    "is_subscribe_save": r.is_subscribe_save,
                    "promo_badges": r.promo_badges,
                    "product_url": r.product_url,
                }
                for r in records
            ]

            # Append (or replace by date)
            today = str(date.today())
            existing = [r for r in existing if r.get("snapshot_date") != today]
            existing.extend(new_records)

            # Keep only recent 30 days
            from datetime import timedelta

            cutoff = str(date.today() - timedelta(days=30))
            existing = [r for r in existing if r.get("snapshot_date", "") >= cutoff]

            # Save
            await asyncio.to_thread(_write_json, self._records_file, existing)

            return True

        except Exception as e:
            logger.error(f"Error saving records: {e}")
            return False

    async def get_recent(self, days: int = 7) -> list[RankRecord]:
        """최근 N일 레코드 조회"""
        try:
            if not self._records_file.exists():
                return []

            data = await asyncio.to_thread(_read_json, self._records_file)

            from datetime import timedelta

            cutoff = str(date.today() - timedelta(days=days))

            result = []
            for row in data:
                if row.get("snapshot_date", "") >= cutoff:
                    try:
                        record = RankRecord(
                            snapshot_date=date.fromisoformat(row["snapshot_date"]),
                            category_id=row["category_id"],
                            asin=row["asin"],
                            product_name=row["product_name"],
                            brand=row["brand"],
                            rank=int(row["rank"]),
                            price=float(row["price"]) if row.get("price") else None,
                            list_price=float(row["list_price"]) if row.get("list_price") else None,
                            discount_percent=float(row["discount_percent"])
                            if row.get("discount_percent")
                            else None,
                            rating=float(row["rating"]) if row.get("rating") else None,
                            reviews_count=int(row["reviews_count"])
                            if row.get("reviews_count")
                            else None,
                            badge=row.get("badge", ""),
                            coupon_text=row.get("coupon_text", ""),
                            is_subscribe_save=row.get("is_subscribe_save", False),
                            promo_badges=row.get("promo_badges", ""),
                            product_url=row.get("product_url", ""),
                        )
                        result.append(record)
                    except Exception:
                        continue

            return result

        except Exception as e:
            logger.error(f"Error getting records: {e}")
            return []

    async def get_by_brand(self, brand: str, days: int = 7) -> list[RankRecord]:
        """브랜드별 레코드 조회"""
        records = await self.get_recent(days)
        return [r for r in records if r.brand.upper() == brand.upper()]

    async def get_by_category(self, category_id: str, days: int = 7) -> list[RankRecord]:
        """카테고리별 레코드 조회"""
        records = await self.get_recent(days)
        return [r for r in records if r.category_id == category_id]

    # MetricsRepository Implementation

    async def save_brand_metrics(self, metrics: list[BrandMetrics]) -> bool:
        """브랜드 메트릭 저장"""
        try:
            data = [
                {
                    "brand": m.brand,
                    "category_id": m.category_id,
                    "sos": m.sos,
                    "brand_avg_rank": m.brand_avg_rank,
                    "product_count": m.product_count,
                    "cpi": m.cpi,
                    "avg_rating_gap": m.avg_rating_gap,
                    "calculated_at": m.calculated_at.isoformat(),
                }
                for m in metrics
            ]

            await asyncio.to_thread(_write_json, self._brand_metrics_file, data)

            return True

        except Exception as e:
            logger.error(f"Error saving brand metrics: {e}")
            return False

    async def save_market_metrics(self, metrics: list[MarketMetrics]) -> bool:
        """마켓 메트릭 저장"""
        try:
            data = [
                {
                    "category_id": m.category_id,
                    "snapshot_date": str(m.snapshot_date),
                    "hhi": m.hhi,
                    "churn_rate": m.churn_rate,
                    "category_avg_price": m.category_avg_price,
                    "category_avg_rating": m.category_avg_rating,
                    "calculated_at": m.calculated_at.isoformat(),
                }
                for m in metrics
            ]

            await asyncio.to_thread(_write_json, self._market_metrics_file, data)

            return True

        except Exception as e:
            logger.error(f"Error saving market metrics: {e}")
            return False

    async def get_brand_metrics(
        self, brand: str, category_id: str | None = None
    ) -> BrandMetrics | None:
        """브랜드 메트릭 조회"""
        try:
            if not self._brand_metrics_file.exists():
                return None

            data = await asyncio.to_thread(_read_json, self._brand_metrics_file)

            for row in data:
                if row["brand"].upper() == brand.upper():
                    if category_id and row["category_id"] != category_id:
                        continue

                    return BrandMetrics(
                        brand=row["brand"],
                        category_id=row["category_id"],
                        sos=float(row["sos"]),
                        brand_avg_rank=float(row["brand_avg_rank"])
                        if row.get("brand_avg_rank")
                        else None,
                        product_count=int(row["product_count"]),
                        cpi=float(row["cpi"]) if row.get("cpi") else None,
                        avg_rating_gap=float(row["avg_rating_gap"])
                        if row.get("avg_rating_gap")
                        else None,
                    )

            return None

        except Exception as e:
            logger.error(f"Error getting brand metrics: {e}")
            return None

    async def get_market_metrics(
        self, category_id: str, snapshot_date: date | None = None
    ) -> MarketMetrics | None:
        """마켓 메트릭 조회"""
        try:
            if not self._market_metrics_file.exists():
                return None

            data = await asyncio.to_thread(_read_json, self._market_metrics_file)

            for row in data:
                if row["category_id"] == category_id:
                    if snapshot_date and row["snapshot_date"] != str(snapshot_date):
                        continue

                    return MarketMetrics(
                        category_id=row["category_id"],
                        snapshot_date=date.fromisoformat(row["snapshot_date"]),
                        hhi=float(row["hhi"]),
                        churn_rate=float(row["churn_rate"]) if row.get("churn_rate") else None,
                        category_avg_price=float(row["category_avg_price"])
                        if row.get("category_avg_price")
                        else None,
                        category_avg_rating=float(row["category_avg_rating"])
                        if row.get("category_avg_rating")
                        else None,
                    )

            return None

        except Exception as e:
            logger.error(f"Error getting market metrics: {e}")
            return None
