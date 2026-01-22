"""
SQLite Storage
로컬 SQLite 데이터베이스를 통한 데이터 저장/조회

Google Sheets 대체용:
- 빠른 읽기/쓰기 (API 호출 없음)
- SQL 쿼리 지원 (복잡한 분석 가능)
- 오프라인 동작
- 파일 기반 (백업 용이)

Usage:
    storage = SQLiteStorage()
    await storage.initialize()
    await storage.append_rank_records(records)

    # 엑셀 내보내기
    storage.export_to_excel("./exports/report.xlsx")
"""

import os
import sqlite3
import json
import logging
from datetime import date, datetime, timedelta, timezone
from typing import List, Dict, Any, Optional
from pathlib import Path
from contextlib import contextmanager

logger = logging.getLogger(__name__)

# 한국 시간대 (UTC+9)
KST = timezone(timedelta(hours=9))


class SQLiteStorage:
    """SQLite 기반 데이터 저장소"""

    # 데이터베이스 스키마
    SCHEMA = """
    -- 원본 크롤링 데이터 (Google Sheets RawData와 동일)
    CREATE TABLE IF NOT EXISTS raw_data (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        snapshot_date TEXT NOT NULL,
        category_id TEXT NOT NULL,
        rank INTEGER NOT NULL,
        asin TEXT NOT NULL,
        product_name TEXT,
        brand TEXT,
        price REAL,
        list_price REAL,
        discount_percent REAL,
        rating REAL,
        reviews_count INTEGER,
        badge TEXT,
        coupon_text TEXT,
        is_subscribe_save INTEGER DEFAULT 0,
        promo_badges TEXT,
        product_url TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(snapshot_date, category_id, rank)
    );

    -- 제품 마스터 데이터
    CREATE TABLE IF NOT EXISTS products (
        asin TEXT PRIMARY KEY,
        product_name TEXT,
        brand TEXT,
        first_seen_date TEXT,
        launch_date TEXT,
        product_url TEXT,
        updated_at TEXT DEFAULT CURRENT_TIMESTAMP
    );

    -- 브랜드 메트릭 (일별)
    CREATE TABLE IF NOT EXISTS brand_metrics (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        snapshot_date TEXT NOT NULL,
        category_id TEXT NOT NULL,
        brand TEXT NOT NULL,
        sos REAL,
        brand_avg_rank REAL,
        product_count INTEGER,
        cpi REAL,
        avg_rating_gap REAL,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(snapshot_date, category_id, brand)
    );

    -- 제품 메트릭 (일별)
    CREATE TABLE IF NOT EXISTS product_metrics (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        snapshot_date TEXT NOT NULL,
        category_id TEXT NOT NULL,
        asin TEXT NOT NULL,
        rank_volatility REAL,
        rank_shock INTEGER DEFAULT 0,
        rank_change INTEGER,
        streak_days INTEGER DEFAULT 0,
        rating_trend TEXT,
        best_rank INTEGER,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(snapshot_date, category_id, asin)
    );

    -- 시장 메트릭 (일별)
    CREATE TABLE IF NOT EXISTS market_metrics (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        snapshot_date TEXT NOT NULL,
        category_id TEXT NOT NULL,
        hhi REAL,
        churn_rate REAL,
        category_avg_price REAL,
        category_avg_rating REAL,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(snapshot_date, category_id)
    );

    -- Amazon Deals 데이터 (경쟁사 할인 모니터링)
    CREATE TABLE IF NOT EXISTS deals (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        snapshot_datetime TEXT NOT NULL,
        asin TEXT NOT NULL,
        product_name TEXT,
        brand TEXT,
        category TEXT,
        deal_price REAL,
        original_price REAL,
        discount_percent REAL,
        deal_type TEXT,
        deal_badge TEXT,
        time_remaining TEXT,
        time_remaining_seconds INTEGER,
        claimed_percent INTEGER,
        deal_end_time TEXT,
        product_url TEXT,
        rating REAL,
        reviews_count INTEGER,
        is_competitor INTEGER DEFAULT 0,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(snapshot_datetime, asin)
    );

    -- 할인 히스토리 (일별 집계)
    CREATE TABLE IF NOT EXISTS deals_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        snapshot_date TEXT NOT NULL,
        brand TEXT NOT NULL,
        total_deals INTEGER DEFAULT 0,
        lightning_deals INTEGER DEFAULT 0,
        avg_discount_percent REAL,
        max_discount_percent REAL,
        products_on_deal TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(snapshot_date, brand)
    );

    -- 할인 알림 로그
    CREATE TABLE IF NOT EXISTS deals_alerts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        alert_datetime TEXT NOT NULL,
        brand TEXT NOT NULL,
        asin TEXT,
        product_name TEXT,
        deal_type TEXT,
        discount_percent REAL,
        alert_type TEXT,
        alert_message TEXT,
        is_sent INTEGER DEFAULT 0,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP
    );

    -- 경쟁사 추적 제품 (Tracked Competitors)
    CREATE TABLE IF NOT EXISTS competitor_products (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        snapshot_date TEXT NOT NULL,
        asin TEXT NOT NULL,
        product_name TEXT,
        brand TEXT NOT NULL,
        price REAL,
        rating REAL,
        reviews_count INTEGER,
        availability TEXT,
        image_url TEXT,
        product_url TEXT,
        category_id TEXT,
        product_type TEXT,
        laneige_competitor TEXT,
        created_at TEXT DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(snapshot_date, asin)
    );

    -- 인덱스 생성 (쿼리 성능 최적화)
    CREATE INDEX IF NOT EXISTS idx_raw_data_date ON raw_data(snapshot_date);
    CREATE INDEX IF NOT EXISTS idx_raw_data_category ON raw_data(category_id);
    CREATE INDEX IF NOT EXISTS idx_raw_data_brand ON raw_data(brand);
    CREATE INDEX IF NOT EXISTS idx_raw_data_asin ON raw_data(asin);
    CREATE INDEX IF NOT EXISTS idx_brand_metrics_date ON brand_metrics(snapshot_date);
    CREATE INDEX IF NOT EXISTS idx_product_metrics_date ON product_metrics(snapshot_date);
    CREATE INDEX IF NOT EXISTS idx_market_metrics_date ON market_metrics(snapshot_date);
    CREATE INDEX IF NOT EXISTS idx_deals_datetime ON deals(snapshot_datetime);
    CREATE INDEX IF NOT EXISTS idx_deals_brand ON deals(brand);
    CREATE INDEX IF NOT EXISTS idx_deals_type ON deals(deal_type);
    CREATE INDEX IF NOT EXISTS idx_deals_history_date ON deals_history(snapshot_date);
    CREATE INDEX IF NOT EXISTS idx_deals_alerts_datetime ON deals_alerts(alert_datetime);
    CREATE INDEX IF NOT EXISTS idx_competitor_date ON competitor_products(snapshot_date);
    CREATE INDEX IF NOT EXISTS idx_competitor_brand ON competitor_products(brand);
    CREATE INDEX IF NOT EXISTS idx_competitor_asin ON competitor_products(asin);
    """

    def __init__(self, db_path: Optional[str] = None):
        """
        Args:
            db_path: SQLite 데이터베이스 파일 경로
                     None이면 환경에 따라 자동 선택:
                     - Railway: /data/amore_data.db (Volume)
                     - Local: ./data/amore_data.db
        """
        if db_path is None:
            # Railway 환경 감지
            if os.environ.get("RAILWAY_ENVIRONMENT"):
                db_path = "/data/amore_data.db"
                logger.info("Railway environment detected, using Volume path: /data/amore_data.db")
            else:
                db_path = "./data/amore_data.db"

        self.db_path = Path(db_path)
        self._initialized = False

    @contextmanager
    def get_connection(self):
        """컨텍스트 매니저로 DB 연결 관리"""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row  # 딕셔너리 형태로 결과 반환
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()

    async def initialize(self) -> bool:
        """
        데이터베이스 초기화

        Returns:
            초기화 성공 여부
        """
        try:
            # 디렉토리 생성
            self.db_path.parent.mkdir(parents=True, exist_ok=True)

            # 스키마 생성
            with self.get_connection() as conn:
                conn.executescript(self.SCHEMA)

            self._initialized = True
            logger.info(f"SQLite database initialized: {self.db_path}")
            return True

        except Exception as e:
            logger.error(f"SQLite 초기화 실패: {e}")
            return False

    async def append_rank_records(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        순위 기록 추가 (SheetsWriter와 동일한 인터페이스)

        Args:
            records: RankRecord 딕셔너리 리스트

        Returns:
            {
                "success": True,
                "rows_added": 100,
                "table": "raw_data"
            }
        """
        if not self._initialized:
            await self.initialize()

        sql = """
        INSERT OR REPLACE INTO raw_data (
            snapshot_date, category_id, rank, asin, product_name,
            brand, price, list_price, discount_percent, rating,
            reviews_count, badge, coupon_text, is_subscribe_save,
            promo_badges, product_url
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """

        rows_added = 0
        try:
            with self.get_connection() as conn:
                for record in records:
                    conn.execute(sql, (
                        record.get("snapshot_date"),
                        record.get("category_id"),
                        record.get("rank"),
                        record.get("asin"),
                        record.get("product_name"),
                        record.get("brand"),
                        self._to_float(record.get("price")),
                        self._to_float(record.get("list_price")),
                        self._to_float(record.get("discount_percent")),
                        self._to_float(record.get("rating")),
                        record.get("reviews_count"),
                        record.get("badge"),
                        record.get("coupon_text"),
                        1 if record.get("is_subscribe_save") else 0,
                        json.dumps(record.get("promo_badges", [])) if record.get("promo_badges") else None,
                        record.get("product_url")
                    ))
                    rows_added += 1

                # 제품 마스터 테이블도 업데이트
                await self._update_products(conn, records)

            logger.info(f"SQLite: {rows_added} rows added to raw_data")
            return {
                "success": True,
                "rows_added": rows_added,
                "table": "raw_data"
            }

        except Exception as e:
            logger.error(f"SQLite append failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "rows_added": rows_added
            }

    def _to_float(self, value) -> Optional[float]:
        """값을 float로 변환 (None, 빈 문자열, 잘못된 형식 처리)"""
        if value is None or value == "" or value == "N/A":
            return None
        try:
            # 문자열에서 $ 기호 제거
            if isinstance(value, str):
                value = value.replace("$", "").replace(",", "").strip()
            return float(value)
        except (ValueError, TypeError):
            return None

    async def _update_products(self, conn: sqlite3.Connection, records: List[Dict[str, Any]]) -> int:
        """제품 마스터 테이블 업데이트"""
        sql = """
        INSERT OR IGNORE INTO products (asin, product_name, brand, first_seen_date, product_url)
        VALUES (?, ?, ?, ?, ?)
        """

        today = datetime.now(KST).strftime("%Y-%m-%d")
        updated = 0

        for record in records:
            asin = record.get("asin")
            if asin:
                conn.execute(sql, (
                    asin,
                    record.get("product_name"),
                    record.get("brand"),
                    today,
                    record.get("product_url")
                ))
                updated += 1

        return updated

    async def get_raw_data(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        category_id: Optional[str] = None,
        brand: Optional[str] = None,
        limit: int = 1000
    ) -> List[Dict[str, Any]]:
        """
        원본 데이터 조회

        Args:
            start_date: 시작 날짜 (YYYY-MM-DD)
            end_date: 종료 날짜 (YYYY-MM-DD)
            category_id: 카테고리 필터
            brand: 브랜드 필터
            limit: 최대 결과 수

        Returns:
            데이터 딕셔너리 리스트
        """
        if not self._initialized:
            await self.initialize()

        conditions = []
        params = []

        if start_date:
            conditions.append("snapshot_date >= ?")
            params.append(start_date)
        if end_date:
            conditions.append("snapshot_date <= ?")
            params.append(end_date)
        if category_id:
            conditions.append("category_id = ?")
            params.append(category_id)
        if brand:
            conditions.append("LOWER(brand) = LOWER(?)")
            params.append(brand)

        where_clause = " AND ".join(conditions) if conditions else "1=1"
        params.append(limit)

        sql = f"""
        SELECT * FROM raw_data
        WHERE {where_clause}
        ORDER BY snapshot_date DESC, category_id, rank
        LIMIT ?
        """

        with self.get_connection() as conn:
            cursor = conn.execute(sql, params)
            return [dict(row) for row in cursor.fetchall()]

    async def get_latest_data(self, category_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """가장 최근 날짜의 데이터 조회"""
        if not self._initialized:
            await self.initialize()

        with self.get_connection() as conn:
            # 최근 날짜 조회
            cursor = conn.execute("SELECT MAX(snapshot_date) as latest FROM raw_data")
            row = cursor.fetchone()
            if not row or not row["latest"]:
                return []

            latest_date = row["latest"]

            # 해당 날짜 데이터 조회
            if category_id:
                cursor = conn.execute(
                    "SELECT * FROM raw_data WHERE snapshot_date = ? AND category_id = ? ORDER BY rank",
                    (latest_date, category_id)
                )
            else:
                cursor = conn.execute(
                    "SELECT * FROM raw_data WHERE snapshot_date = ? ORDER BY category_id, rank",
                    (latest_date,)
                )

            return [dict(row) for row in cursor.fetchall()]

    async def get_historical_data(
        self,
        days: int = 30,
        category_id: Optional[str] = None,
        brand: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """히스토리컬 데이터 조회 (최근 N일)"""
        end_date = datetime.now(KST).strftime("%Y-%m-%d")
        start_date = (datetime.now(KST) - timedelta(days=days)).strftime("%Y-%m-%d")

        return await self.get_raw_data(
            start_date=start_date,
            end_date=end_date,
            category_id=category_id,
            brand=brand,
            limit=10000
        )

    async def save_brand_metrics(self, metrics: List[Dict[str, Any]]) -> int:
        """브랜드 메트릭 저장"""
        if not self._initialized:
            await self.initialize()

        sql = """
        INSERT OR REPLACE INTO brand_metrics (
            snapshot_date, category_id, brand, sos, brand_avg_rank,
            product_count, cpi, avg_rating_gap
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """

        with self.get_connection() as conn:
            for m in metrics:
                conn.execute(sql, (
                    m.get("snapshot_date"),
                    m.get("category_id"),
                    m.get("brand"),
                    m.get("sos"),
                    m.get("brand_avg_rank"),
                    m.get("product_count"),
                    m.get("cpi"),
                    m.get("avg_rating_gap")
                ))

        return len(metrics)

    async def save_market_metrics(self, metrics: List[Dict[str, Any]]) -> int:
        """시장 메트릭 저장"""
        if not self._initialized:
            await self.initialize()

        sql = """
        INSERT OR REPLACE INTO market_metrics (
            snapshot_date, category_id, hhi, churn_rate,
            category_avg_price, category_avg_rating
        ) VALUES (?, ?, ?, ?, ?, ?)
        """

        with self.get_connection() as conn:
            for m in metrics:
                conn.execute(sql, (
                    m.get("snapshot_date"),
                    m.get("category_id"),
                    m.get("hhi"),
                    m.get("churn_rate"),
                    m.get("category_avg_price"),
                    m.get("category_avg_rating")
                ))

        return len(metrics)

    async def save_competitor_products(self, products: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        경쟁사 추적 제품 저장

        Args:
            products: 경쟁사 제품 리스트 (scrape_competitor_products 결과)

        Returns:
            {"success": True, "rows_added": N}
        """
        if not self._initialized:
            await self.initialize()

        sql = """
        INSERT OR REPLACE INTO competitor_products (
            snapshot_date, asin, product_name, brand, price, rating,
            reviews_count, availability, image_url, product_url,
            category_id, product_type, laneige_competitor
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """

        rows_added = 0
        try:
            with self.get_connection() as conn:
                for p in products:
                    conn.execute(sql, (
                        p.get("snapshot_date"),
                        p.get("asin"),
                        p.get("product_name"),
                        p.get("brand"),
                        self._to_float(p.get("price")),
                        self._to_float(p.get("rating")),
                        p.get("reviews_count"),
                        p.get("availability"),
                        p.get("image_url"),
                        p.get("product_url"),
                        p.get("category_id"),
                        p.get("product_type"),
                        p.get("laneige_competitor")
                    ))
                    rows_added += 1

            logger.info(f"SQLite: saved {rows_added} competitor products")
            return {"success": True, "rows_added": rows_added}

        except Exception as e:
            logger.error(f"Failed to save competitor products: {e}")
            return {"success": False, "error": str(e), "rows_added": rows_added}

    async def get_competitor_products(
        self,
        brand: Optional[str] = None,
        snapshot_date: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        경쟁사 제품 조회

        Args:
            brand: 브랜드 필터 (예: "Summer Fridays")
            snapshot_date: 특정 날짜 (없으면 최신)

        Returns:
            경쟁사 제품 리스트
        """
        if not self._initialized:
            await self.initialize()

        conditions = []
        params = []

        if brand:
            conditions.append("brand = ?")
            params.append(brand)

        if snapshot_date:
            conditions.append("snapshot_date = ?")
            params.append(snapshot_date)
        else:
            # 최신 날짜
            conditions.append("snapshot_date = (SELECT MAX(snapshot_date) FROM competitor_products)")

        where_clause = " AND ".join(conditions) if conditions else "1=1"

        sql = f"""
        SELECT * FROM competitor_products
        WHERE {where_clause}
        ORDER BY brand, product_type
        """

        try:
            with self.get_connection() as conn:
                cursor = conn.execute(sql, params)
                return [dict(row) for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Failed to get competitor products: {e}")
            return []

    def get_data_date(self) -> Optional[str]:
        """가장 최근 데이터 날짜 반환"""
        try:
            with self.get_connection() as conn:
                cursor = conn.execute("SELECT MAX(snapshot_date) as latest FROM raw_data")
                row = cursor.fetchone()
                return row["latest"] if row else None
        except Exception:
            return None

    def get_stats(self) -> Dict[str, Any]:
        """데이터베이스 통계"""
        try:
            with self.get_connection() as conn:
                stats = {}

                # 테이블별 행 수
                for table in ["raw_data", "products", "brand_metrics", "market_metrics"]:
                    cursor = conn.execute(f"SELECT COUNT(*) as cnt FROM {table}")
                    row = cursor.fetchone()
                    stats[f"{table}_count"] = row["cnt"] if row else 0

                # 날짜 범위
                cursor = conn.execute(
                    "SELECT MIN(snapshot_date) as min_date, MAX(snapshot_date) as max_date FROM raw_data"
                )
                row = cursor.fetchone()
                stats["date_range"] = {
                    "min": row["min_date"] if row else None,
                    "max": row["max_date"] if row else None
                }

                # 파일 크기
                if self.db_path.exists():
                    stats["file_size_mb"] = round(self.db_path.stat().st_size / (1024 * 1024), 2)

                return stats

        except Exception as e:
            return {"error": str(e)}

    # =========================================================================
    # 엑셀 내보내기
    # =========================================================================

    def export_to_excel(
        self,
        output_path: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        include_metrics: bool = True
    ) -> Dict[str, Any]:
        """
        데이터를 엑셀 파일로 내보내기

        Args:
            output_path: 출력 파일 경로 (.xlsx)
            start_date: 시작 날짜 (없으면 최근 7일)
            end_date: 종료 날짜 (없으면 오늘)
            include_metrics: 메트릭 시트 포함 여부

        Returns:
            {
                "success": True,
                "file_path": "...",
                "sheets": ["Summary", "Beauty", ...],
                "total_rows": 500
            }
        """
        try:
            import pandas as pd
        except ImportError:
            return {"success": False, "error": "pandas not installed"}

        try:
            # 날짜 기본값
            if not end_date:
                end_date = datetime.now(KST).strftime("%Y-%m-%d")
            if not start_date:
                start_date = (datetime.now(KST) - timedelta(days=7)).strftime("%Y-%m-%d")

            # 출력 디렉토리 생성
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            sheets_created = []
            total_rows = 0

            with pd.ExcelWriter(str(output_path), engine="openpyxl") as writer:
                with self.get_connection() as conn:
                    # 1. Summary 시트
                    summary_data = self._create_summary_data(conn, start_date, end_date)
                    if summary_data:
                        df_summary = pd.DataFrame(summary_data)
                        df_summary.to_excel(writer, sheet_name="Summary", index=False)
                        sheets_created.append("Summary")
                        total_rows += len(df_summary)

                    # 2. 카테고리별 시트
                    categories = {
                        "beauty": "Beauty & Personal Care",
                        "skin_care": "Skin Care",
                        "lip_care": "Lip Care",
                        "lip_makeup": "Lip Makeup",
                        "face_powder": "Face Powder"
                    }

                    for cat_id, cat_name in categories.items():
                        df = pd.read_sql_query(
                            """
                            SELECT snapshot_date, rank, asin, product_name, brand,
                                   price, rating, reviews_count, badge
                            FROM raw_data
                            WHERE category_id = ? AND snapshot_date BETWEEN ? AND ?
                            ORDER BY snapshot_date DESC, rank
                            """,
                            conn,
                            params=(cat_id, start_date, end_date)
                        )

                        if not df.empty:
                            # 시트 이름 제한 (31자)
                            sheet_name = cat_name[:31]
                            df.to_excel(writer, sheet_name=sheet_name, index=False)
                            sheets_created.append(sheet_name)
                            total_rows += len(df)

                    # 3. 브랜드 메트릭 시트 (옵션)
                    if include_metrics:
                        df_brand = pd.read_sql_query(
                            """
                            SELECT * FROM brand_metrics
                            WHERE snapshot_date BETWEEN ? AND ?
                            ORDER BY snapshot_date DESC, sos DESC
                            """,
                            conn,
                            params=(start_date, end_date)
                        )

                        if not df_brand.empty:
                            df_brand.to_excel(writer, sheet_name="Brand Metrics", index=False)
                            sheets_created.append("Brand Metrics")
                            total_rows += len(df_brand)

                    # 4. 데이터가 없는 경우 안내 시트 생성 (openpyxl 요구사항)
                    if not sheets_created:
                        # 사용 가능한 날짜 범위 조회
                        cursor = conn.execute(
                            "SELECT MIN(snapshot_date) as min_date, MAX(snapshot_date) as max_date FROM raw_data"
                        )
                        row = cursor.fetchone()
                        available_min = row["min_date"] if row and row["min_date"] else "N/A"
                        available_max = row["max_date"] if row and row["max_date"] else "N/A"

                        no_data_info = [
                            {"항목": "요청 기간", "값": f"{start_date} ~ {end_date}"},
                            {"항목": "결과", "값": "해당 기간에 데이터가 없습니다"},
                            {"항목": "사용 가능한 데이터 기간", "값": f"{available_min} ~ {available_max}"},
                            {"항목": "안내", "값": "날짜 범위를 조정하여 다시 시도해주세요"}
                        ]
                        df_no_data = pd.DataFrame(no_data_info)
                        df_no_data.to_excel(writer, sheet_name="No Data", index=False)
                        sheets_created.append("No Data")

            logger.info(f"Excel exported: {output_path} ({total_rows} rows)")
            return {
                "success": True,
                "file_path": str(output_path),
                "sheets": sheets_created,
                "total_rows": total_rows,
                "date_range": {"start": start_date, "end": end_date}
            }

        except Exception as e:
            logger.error(f"Excel export failed: {e}")
            return {"success": False, "error": str(e)}

    def _create_summary_data(
        self,
        conn: sqlite3.Connection,
        start_date: str,
        end_date: str
    ) -> List[Dict[str, Any]]:
        """Summary 시트용 데이터 생성 (기간 내 브랜드별 집계)"""
        # 브랜드별 SoS 집계 - 지정된 기간의 데이터 사용
        cursor = conn.execute(
            """
            SELECT
                brand,
                COUNT(*) as product_count,
                ROUND(COUNT(*) * 100.0 / (
                    SELECT COUNT(*) FROM raw_data
                    WHERE snapshot_date BETWEEN ? AND ?
                ), 2) as sos,
                ROUND(AVG(rank), 1) as avg_rank,
                ROUND(AVG(rating), 2) as avg_rating,
                MIN(snapshot_date) as period_start,
                MAX(snapshot_date) as period_end
            FROM raw_data
            WHERE snapshot_date BETWEEN ? AND ?
            GROUP BY brand
            ORDER BY product_count DESC
            LIMIT 20
            """,
            (start_date, end_date, start_date, end_date)
        )

        return [dict(row) for row in cursor.fetchall()]

    # =========================================================================
    # Deals 관련 메서드
    # =========================================================================

    async def save_deals(self, deals: List[Dict[str, Any]], is_competitor: bool = False) -> Dict[str, Any]:
        """
        Deals 데이터 저장

        Args:
            deals: DealRecord 딕셔너리 리스트
            is_competitor: 경쟁사 딜 여부

        Returns:
            {"success": True, "rows_added": 50}
        """
        if not self._initialized:
            await self.initialize()

        sql = """
        INSERT OR REPLACE INTO deals (
            snapshot_datetime, asin, product_name, brand, category,
            deal_price, original_price, discount_percent, deal_type, deal_badge,
            time_remaining, time_remaining_seconds, claimed_percent, deal_end_time,
            product_url, rating, reviews_count, is_competitor
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """

        rows_added = 0
        try:
            with self.get_connection() as conn:
                for deal in deals:
                    conn.execute(sql, (
                        deal.get("snapshot_datetime"),
                        deal.get("asin"),
                        deal.get("product_name"),
                        deal.get("brand"),
                        deal.get("category"),
                        deal.get("deal_price"),
                        deal.get("original_price"),
                        deal.get("discount_percent"),
                        deal.get("deal_type"),
                        deal.get("deal_badge"),
                        deal.get("time_remaining"),
                        deal.get("time_remaining_seconds"),
                        deal.get("claimed_percent"),
                        deal.get("deal_end_time"),
                        deal.get("product_url"),
                        deal.get("rating"),
                        deal.get("reviews_count"),
                        1 if is_competitor else 0
                    ))
                    rows_added += 1

            logger.info(f"SQLite: {rows_added} deals saved")
            return {"success": True, "rows_added": rows_added}

        except Exception as e:
            logger.error(f"Deals save failed: {e}")
            return {"success": False, "error": str(e), "rows_added": rows_added}

    async def save_deals_history(self, history: Dict[str, Any]) -> bool:
        """일별 Deals 히스토리 저장"""
        if not self._initialized:
            await self.initialize()

        sql = """
        INSERT OR REPLACE INTO deals_history (
            snapshot_date, brand, total_deals, lightning_deals,
            avg_discount_percent, max_discount_percent, products_on_deal
        ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """

        try:
            with self.get_connection() as conn:
                conn.execute(sql, (
                    history.get("snapshot_date"),
                    history.get("brand"),
                    history.get("total_deals", 0),
                    history.get("lightning_deals", 0),
                    history.get("avg_discount_percent"),
                    history.get("max_discount_percent"),
                    json.dumps(history.get("products_on_deal", []))
                ))
            return True
        except Exception as e:
            logger.error(f"Deals history save failed: {e}")
            return False

    async def save_deal_alert(self, alert: Dict[str, Any]) -> int:
        """할인 알림 저장 (ID 반환)"""
        if not self._initialized:
            await self.initialize()

        sql = """
        INSERT INTO deals_alerts (
            alert_datetime, brand, asin, product_name,
            deal_type, discount_percent, alert_type, alert_message
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """

        try:
            with self.get_connection() as conn:
                cursor = conn.execute(sql, (
                    alert.get("alert_datetime", datetime.now(KST).isoformat()),
                    alert.get("brand"),
                    alert.get("asin"),
                    alert.get("product_name"),
                    alert.get("deal_type"),
                    alert.get("discount_percent"),
                    alert.get("alert_type"),
                    alert.get("alert_message")
                ))
                return cursor.lastrowid
        except Exception as e:
            logger.error(f"Deal alert save failed: {e}")
            return -1

    async def get_competitor_deals(
        self,
        brand: Optional[str] = None,
        hours: int = 24
    ) -> List[Dict[str, Any]]:
        """경쟁사 딜 조회"""
        if not self._initialized:
            await self.initialize()

        cutoff_time = (datetime.now(KST) - timedelta(hours=hours)).isoformat()

        if brand:
            sql = """
            SELECT * FROM deals
            WHERE is_competitor = 1 AND LOWER(brand) LIKE LOWER(?) AND snapshot_datetime >= ?
            ORDER BY discount_percent DESC
            """
            params = (f"%{brand}%", cutoff_time)
        else:
            sql = """
            SELECT * FROM deals
            WHERE is_competitor = 1 AND snapshot_datetime >= ?
            ORDER BY discount_percent DESC
            """
            params = (cutoff_time,)

        with self.get_connection() as conn:
            cursor = conn.execute(sql, params)
            return [dict(row) for row in cursor.fetchall()]

    async def get_deals_summary(self, days: int = 7) -> Dict[str, Any]:
        """Deals 요약 통계"""
        if not self._initialized:
            await self.initialize()

        cutoff_date = (datetime.now(KST) - timedelta(days=days)).strftime("%Y-%m-%d")

        with self.get_connection() as conn:
            # 브랜드별 딜 현황
            cursor = conn.execute("""
                SELECT
                    brand,
                    COUNT(*) as total_deals,
                    SUM(CASE WHEN deal_type = 'lightning' THEN 1 ELSE 0 END) as lightning_deals,
                    ROUND(AVG(discount_percent), 1) as avg_discount,
                    MAX(discount_percent) as max_discount
                FROM deals
                WHERE DATE(snapshot_datetime) >= ?
                GROUP BY brand
                ORDER BY total_deals DESC
                LIMIT 20
            """, (cutoff_date,))

            by_brand = [dict(row) for row in cursor.fetchall()]

            # 일별 추이
            cursor = conn.execute("""
                SELECT
                    DATE(snapshot_datetime) as date,
                    COUNT(*) as total_deals,
                    SUM(CASE WHEN deal_type = 'lightning' THEN 1 ELSE 0 END) as lightning_deals,
                    ROUND(AVG(discount_percent), 1) as avg_discount
                FROM deals
                WHERE DATE(snapshot_datetime) >= ?
                GROUP BY DATE(snapshot_datetime)
                ORDER BY date DESC
            """, (cutoff_date,))

            by_date = [dict(row) for row in cursor.fetchall()]

            return {
                "by_brand": by_brand,
                "by_date": by_date,
                "period_days": days
            }

    async def get_unsent_alerts(self, limit: int = 50) -> List[Dict[str, Any]]:
        """미발송 알림 조회"""
        if not self._initialized:
            await self.initialize()

        with self.get_connection() as conn:
            cursor = conn.execute("""
                SELECT * FROM deals_alerts
                WHERE is_sent = 0
                ORDER BY alert_datetime DESC
                LIMIT ?
            """, (limit,))
            return [dict(row) for row in cursor.fetchall()]

    async def mark_alert_sent(self, alert_id: int) -> bool:
        """알림 발송 완료 표시"""
        try:
            with self.get_connection() as conn:
                conn.execute("UPDATE deals_alerts SET is_sent = 1 WHERE id = ?", (alert_id,))
            return True
        except Exception:
            return False

    def export_deals_report(
        self,
        output_path: str,
        days: int = 7
    ) -> Dict[str, Any]:
        """
        Deals 분석 리포트 Excel 내보내기

        Args:
            output_path: 출력 파일 경로
            days: 분석 기간 (일)

        Returns:
            {"success": True, "file_path": "...", "sheets": [...]}
        """
        try:
            import pandas as pd
        except ImportError:
            return {"success": False, "error": "pandas not installed"}

        try:
            cutoff_date = (datetime.now(KST) - timedelta(days=days)).strftime("%Y-%m-%d")
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            sheets_created = []

            with pd.ExcelWriter(str(output_path), engine="openpyxl") as writer:
                with self.get_connection() as conn:
                    # 1. Summary 시트
                    df_summary = pd.read_sql_query("""
                        SELECT
                            brand as "브랜드",
                            COUNT(*) as "총 딜 수",
                            SUM(CASE WHEN deal_type = 'lightning' THEN 1 ELSE 0 END) as "Lightning Deals",
                            ROUND(AVG(discount_percent), 1) as "평균 할인율(%)",
                            MAX(discount_percent) as "최대 할인율(%)",
                            COUNT(DISTINCT asin) as "제품 수"
                        FROM deals
                        WHERE DATE(snapshot_datetime) >= ?
                        GROUP BY brand
                        ORDER BY COUNT(*) DESC
                    """, conn, params=(cutoff_date,))

                    if not df_summary.empty:
                        df_summary.to_excel(writer, sheet_name="Summary", index=False)
                        sheets_created.append("Summary")

                    # 2. 일별 추이
                    df_daily = pd.read_sql_query("""
                        SELECT
                            DATE(snapshot_datetime) as "날짜",
                            COUNT(*) as "총 딜",
                            SUM(CASE WHEN deal_type = 'lightning' THEN 1 ELSE 0 END) as "Lightning",
                            ROUND(AVG(discount_percent), 1) as "평균 할인율"
                        FROM deals
                        WHERE DATE(snapshot_datetime) >= ?
                        GROUP BY DATE(snapshot_datetime)
                        ORDER BY DATE(snapshot_datetime) DESC
                    """, conn, params=(cutoff_date,))

                    if not df_daily.empty:
                        df_daily.to_excel(writer, sheet_name="Daily Trend", index=False)
                        sheets_created.append("Daily Trend")

                    # 3. 경쟁사 딜 상세
                    df_competitor = pd.read_sql_query("""
                        SELECT
                            snapshot_datetime as "수집시각",
                            brand as "브랜드",
                            product_name as "제품명",
                            deal_price as "할인가",
                            original_price as "원가",
                            discount_percent as "할인율(%)",
                            deal_type as "딜 타입",
                            time_remaining as "남은시간",
                            claimed_percent as "판매율(%)"
                        FROM deals
                        WHERE is_competitor = 1 AND DATE(snapshot_datetime) >= ?
                        ORDER BY snapshot_datetime DESC
                        LIMIT 500
                    """, conn, params=(cutoff_date,))

                    if not df_competitor.empty:
                        df_competitor.to_excel(writer, sheet_name="Competitor Deals", index=False)
                        sheets_created.append("Competitor Deals")

                    # 4. Lightning Deals
                    df_lightning = pd.read_sql_query("""
                        SELECT
                            snapshot_datetime as "수집시각",
                            brand as "브랜드",
                            product_name as "제품명",
                            deal_price as "할인가",
                            discount_percent as "할인율(%)",
                            time_remaining as "남은시간",
                            claimed_percent as "판매율(%)",
                            deal_end_time as "종료예정"
                        FROM deals
                        WHERE deal_type = 'lightning' AND DATE(snapshot_datetime) >= ?
                        ORDER BY snapshot_datetime DESC
                        LIMIT 200
                    """, conn, params=(cutoff_date,))

                    if not df_lightning.empty:
                        df_lightning.to_excel(writer, sheet_name="Lightning Deals", index=False)
                        sheets_created.append("Lightning Deals")

                    # 5. 데이터가 없는 경우 안내 시트 생성 (openpyxl 요구사항)
                    if not sheets_created:
                        no_data_info = [
                            {"항목": "분석 기간", "값": f"최근 {days}일"},
                            {"항목": "결과", "값": "해당 기간에 딜 데이터가 없습니다"},
                            {"항목": "안내", "값": "딜 모니터링이 시작된 후 데이터가 수집됩니다"}
                        ]
                        df_no_data = pd.DataFrame(no_data_info)
                        df_no_data.to_excel(writer, sheet_name="No Data", index=False)
                        sheets_created.append("No Data")

            logger.info(f"Deals report exported: {output_path}")
            return {
                "success": True,
                "file_path": str(output_path),
                "sheets": sheets_created,
                "period_days": days
            }

        except Exception as e:
            logger.error(f"Deals report export failed: {e}")
            return {"success": False, "error": str(e)}


# =============================================================================
# 싱글톤 인스턴스
# =============================================================================

_storage_instance: Optional[SQLiteStorage] = None


def get_sqlite_storage() -> SQLiteStorage:
    """SQLiteStorage 싱글톤 반환"""
    global _storage_instance
    if _storage_instance is None:
        _storage_instance = SQLiteStorage()
    return _storage_instance
