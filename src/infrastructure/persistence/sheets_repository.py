"""
Google Sheets Repository Implementation
=======================================
ProductRepository Protocol 구현 - Google Sheets 백엔드
"""

import logging
from datetime import date
from pathlib import Path

logger = logging.getLogger(__name__)

from src.domain.entities.brand import BrandMetrics
from src.domain.entities.market import MarketMetrics
from src.domain.entities.product import RankRecord
from src.domain.interfaces.repository import MetricsRepository, ProductRepository


class GoogleSheetsRepository(ProductRepository, MetricsRepository):
    """
    Google Sheets를 이용한 Repository 구현

    ProductRepository와 MetricsRepository Protocol을 구현합니다.
    """

    def __init__(self, spreadsheet_id: str | None = None, credentials_path: str | None = None):
        self.spreadsheet_id = spreadsheet_id
        self.credentials_path = credentials_path
        self._client = None
        self._sheet = None
        self._initialized = False

    async def initialize(self) -> None:
        """Google Sheets 연결 초기화"""
        if self._initialized:
            return

        # Lazy import to avoid dependency issues
        try:
            import gspread
            from google.oauth2.service_account import Credentials
        except ImportError:
            raise RuntimeError(
                "Google Sheets dependencies not installed. Run: pip install gspread google-auth"
            )

        if not self.spreadsheet_id:
            import os

            self.spreadsheet_id = os.environ.get("GOOGLE_SPREADSHEET_ID")

        if not self.credentials_path:
            import os

            self.credentials_path = os.environ.get(
                "GOOGLE_APPLICATION_CREDENTIALS", "./config/google_credentials.json"
            )

        if not self.spreadsheet_id:
            raise ValueError("spreadsheet_id is required")

        if not Path(self.credentials_path).exists():
            raise FileNotFoundError(f"Credentials file not found: {self.credentials_path}")

        scopes = [
            "https://www.googleapis.com/auth/spreadsheets",
            "https://www.googleapis.com/auth/drive",
        ]

        creds = Credentials.from_service_account_file(self.credentials_path, scopes=scopes)
        self._client = gspread.authorize(creds)
        self._sheet = self._client.open_by_key(self.spreadsheet_id)
        self._initialized = True

    async def save_records(self, records: list[RankRecord]) -> bool:
        """레코드 저장"""
        await self.initialize()

        try:
            # Get or create worksheet
            worksheet_name = "RankRecords"
            try:
                worksheet = self._sheet.worksheet(worksheet_name)
            except Exception:
                worksheet = self._sheet.add_worksheet(title=worksheet_name, rows=1000, cols=20)

            # Convert to rows
            if not records:
                return True

            # Headers
            headers = [
                "snapshot_date",
                "category_id",
                "asin",
                "product_name",
                "brand",
                "rank",
                "price",
                "rating",
                "reviews_count",
                "badge",
                "coupon_text",
                "product_url",
            ]

            # Data rows
            rows = [headers]
            for record in records:
                row = [
                    str(record.snapshot_date),
                    record.category_id,
                    record.asin,
                    record.product_name,
                    record.brand,
                    record.rank,
                    record.price or "",
                    record.rating or "",
                    record.reviews_count or "",
                    record.badge,
                    record.coupon_text,
                    record.product_url,
                ]
                rows.append(row)

            # Update worksheet
            worksheet.clear()
            worksheet.update(rows, "A1")

            return True

        except Exception as e:
            logger.error(f"Error saving records to Sheets: {e}")
            return False

    async def get_recent(self, days: int = 7) -> list[RankRecord]:
        """최근 N일 레코드 조회"""
        await self.initialize()

        try:
            worksheet = self._sheet.worksheet("RankRecords")
            records = worksheet.get_all_records()

            # Filter by date
            date.today().isoformat()
            result = []

            for row in records:
                try:
                    record = RankRecord(
                        snapshot_date=date.fromisoformat(row["snapshot_date"]),
                        category_id=row["category_id"],
                        asin=row["asin"],
                        product_name=row["product_name"],
                        brand=row["brand"],
                        rank=int(row["rank"]),
                        price=float(row["price"]) if row.get("price") else None,
                        rating=float(row["rating"]) if row.get("rating") else None,
                        reviews_count=int(row["reviews_count"])
                        if row.get("reviews_count")
                        else None,
                        badge=row.get("badge", ""),
                        coupon_text=row.get("coupon_text", ""),
                        product_url=row.get("product_url", ""),
                    )
                    result.append(record)
                except Exception:
                    continue

            return result

        except Exception as e:
            logger.error(f"Error getting records from Sheets: {e}")
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
        await self.initialize()

        try:
            worksheet_name = "BrandMetrics"
            try:
                worksheet = self._sheet.worksheet(worksheet_name)
            except Exception:
                worksheet = self._sheet.add_worksheet(title=worksheet_name, rows=500, cols=15)

            headers = [
                "brand",
                "category_id",
                "sos",
                "brand_avg_rank",
                "product_count",
                "cpi",
                "avg_rating_gap",
                "calculated_at",
            ]

            rows = [headers]
            for m in metrics:
                row = [
                    m.brand,
                    m.category_id,
                    m.sos,
                    m.brand_avg_rank or "",
                    m.product_count,
                    m.cpi or "",
                    m.avg_rating_gap or "",
                    m.calculated_at.isoformat(),
                ]
                rows.append(row)

            worksheet.clear()
            worksheet.update(rows, "A1")

            return True

        except Exception as e:
            logger.error(f"Error saving brand metrics: {e}")
            return False

    async def save_market_metrics(self, metrics: list[MarketMetrics]) -> bool:
        """마켓 메트릭 저장"""
        await self.initialize()

        try:
            worksheet_name = "MarketMetrics"
            try:
                worksheet = self._sheet.worksheet(worksheet_name)
            except Exception:
                worksheet = self._sheet.add_worksheet(title=worksheet_name, rows=500, cols=15)

            headers = [
                "category_id",
                "snapshot_date",
                "hhi",
                "churn_rate",
                "category_avg_price",
                "category_avg_rating",
                "calculated_at",
            ]

            rows = [headers]
            for m in metrics:
                row = [
                    m.category_id,
                    str(m.snapshot_date),
                    m.hhi,
                    m.churn_rate or "",
                    m.category_avg_price or "",
                    m.category_avg_rating or "",
                    m.calculated_at.isoformat(),
                ]
                rows.append(row)

            worksheet.clear()
            worksheet.update(rows, "A1")

            return True

        except Exception as e:
            logger.error(f"Error saving market metrics: {e}")
            return False

    async def get_brand_metrics(
        self, brand: str, category_id: str | None = None
    ) -> BrandMetrics | None:
        """브랜드 메트릭 조회"""
        await self.initialize()

        try:
            worksheet = self._sheet.worksheet("BrandMetrics")
            records = worksheet.get_all_records()

            for row in records:
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
        await self.initialize()

        try:
            worksheet = self._sheet.worksheet("MarketMetrics")
            records = worksheet.get_all_records()

            for row in records:
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
