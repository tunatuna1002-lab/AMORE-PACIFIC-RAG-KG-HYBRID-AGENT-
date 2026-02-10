"""
Data Sync Routes
================
Railway Volume 데이터 동기화 엔드포인트
"""

import logging
import os
import re

from fastapi import APIRouter, HTTPException, Request

from src.tools.storage.sqlite_storage import get_sqlite_storage

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Sync"])


@router.get("/api/sync/status")
async def sync_status():
    """
    Railway Volume의 데이터 현황 반환

    Returns:
        - latest: 최신 데이터 날짜
        - oldest: 가장 오래된 데이터 날짜
        - total_days: 총 일수
        - total_records: SQLite raw_data 총 레코드 수
    """
    try:
        sqlite = get_sqlite_storage()
        if not sqlite:
            raise HTTPException(status_code=500, detail="SQLite not available")

        await sqlite.initialize()

        # raw_data 테이블에서 날짜 범위 조회
        with sqlite.get_connection() as conn:
            cursor = conn.execute("""
                SELECT
                    MIN(snapshot_date) as oldest,
                    MAX(snapshot_date) as latest,
                    COUNT(DISTINCT snapshot_date) as total_days,
                    COUNT(*) as total_records
                FROM raw_data
            """)
            row = cursor.fetchone()

            if not row or not row[0]:
                return {
                    "success": True,
                    "latest": None,
                    "oldest": None,
                    "total_days": 0,
                    "total_records": 0,
                    "message": "No data available",
                }

            return {
                "success": True,
                "latest": row[1],
                "oldest": row[0],
                "total_days": row[2],
                "total_records": row[3],
            }
    except Exception as e:
        logging.error(f"Sync status error: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/api/sync/dates")
async def sync_dates():
    """
    사용 가능한 모든 날짜 목록 반환 (정렬됨)

    Returns:
        - dates: ["2026-01-17", "2026-01-18", ..., "2026-01-25"]
    """
    try:
        sqlite = get_sqlite_storage()
        if not sqlite:
            raise HTTPException(status_code=500, detail="SQLite not available")

        await sqlite.initialize()

        with sqlite.get_connection() as conn:
            cursor = conn.execute("""
                SELECT DISTINCT snapshot_date
                FROM raw_data
                ORDER BY snapshot_date
            """)
            dates = [row[0] for row in cursor.fetchall()]

        return {"success": True, "dates": dates, "count": len(dates)}
    except Exception as e:
        logging.error(f"Sync dates error: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.get("/api/sync/download/{date}")
async def sync_download(date: str):
    """
    특정 날짜의 raw_data를 JSON으로 다운로드

    Args:
        date: 날짜 (YYYY-MM-DD 형식)

    Returns:
        JSON array of raw_data records for the specified date
    """
    # 날짜 형식 검증
    if not re.match(r"^\d{4}-\d{2}-\d{2}$", date):
        raise HTTPException(status_code=400, detail="Invalid date format. Use YYYY-MM-DD")

    try:
        sqlite = get_sqlite_storage()
        if not sqlite:
            raise HTTPException(status_code=500, detail="SQLite not available")

        await sqlite.initialize()

        with sqlite.get_connection() as conn:
            # 컬럼명 가져오기
            cursor = conn.execute("PRAGMA table_info(raw_data)")
            columns = [row[1] for row in cursor.fetchall()]

            # 해당 날짜 데이터 조회
            cursor = conn.execute(
                """
                SELECT * FROM raw_data
                WHERE snapshot_date = ?
                ORDER BY category_id, rank
            """,
                (date,),
            )
            rows = cursor.fetchall()

        if not rows:
            raise HTTPException(status_code=404, detail=f"No data found for date: {date}")

        # 딕셔너리 변환
        records = []
        for row in rows:
            record = dict(zip(columns, row, strict=False))
            records.append(record)

        return {"success": True, "date": date, "count": len(records), "records": records}
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Sync download error for {date}: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/api/sync/upload")
async def sync_upload(request: Request):
    """
    로컬에서 Railway로 raw_data 업로드

    Request Body:
        {
            "records": [...],  # raw_data 레코드 배열
            "api_key": "..."   # 인증 키 (선택)
        }

    Returns:
        {"success": True, "inserted": N, "updated": M}
    """
    try:
        body = await request.json()
        records = body.get("records", [])
        api_key = body.get("api_key", "")

        # API 키 검증 (설정된 경우)
        expected_key = os.getenv("API_KEY", "")
        if expected_key and api_key != expected_key:
            raise HTTPException(status_code=401, detail="Invalid API key")

        if not records:
            raise HTTPException(status_code=400, detail="No records provided")

        sqlite = get_sqlite_storage()
        if not sqlite:
            raise HTTPException(status_code=500, detail="SQLite not available")

        await sqlite.initialize()

        inserted = 0
        updated = 0

        with sqlite.get_connection() as conn:
            # 스키마 마이그레이션: 누락된 컬럼 추가
            migration_columns = [
                ("image_url", "TEXT"),
                ("is_best_seller", "INTEGER DEFAULT 0"),
                ("is_amazon_choice", "INTEGER DEFAULT 0"),
            ]
            for col_name, col_type in migration_columns:
                try:
                    conn.execute(f"ALTER TABLE raw_data ADD COLUMN {col_name} {col_type}")
                    conn.commit()
                    logging.info(f"Added {col_name} column to raw_data table")
                except Exception:
                    pass  # 이미 존재하면 무시

            for record in records:
                # UPSERT: snapshot_date + category_id + asin 조합이 unique key
                cursor = conn.execute(
                    """
                    SELECT id FROM raw_data
                    WHERE snapshot_date = ? AND category_id = ? AND asin = ?
                """,
                    (
                        record.get("snapshot_date"),
                        record.get("category_id"),
                        record.get("asin"),
                    ),
                )
                existing = cursor.fetchone()

                if existing:
                    # UPDATE
                    conn.execute(
                        """
                        UPDATE raw_data SET
                            rank = ?, product_name = ?, brand = ?, price = ?,
                            rating = ?, reviews_count = ?, product_url = ?,
                            image_url = ?, is_best_seller = ?, is_amazon_choice = ?
                        WHERE id = ?
                    """,
                        (
                            record.get("rank"),
                            record.get("product_name"),
                            record.get("brand"),
                            record.get("price"),
                            record.get("rating"),
                            record.get("reviews_count"),
                            record.get("product_url"),
                            record.get("image_url"),
                            record.get("is_best_seller", 0),
                            record.get("is_amazon_choice", 0),
                            existing[0],
                        ),
                    )
                    updated += 1
                else:
                    # INSERT
                    conn.execute(
                        """
                        INSERT INTO raw_data (
                            snapshot_date, category_id, asin, rank, product_name,
                            brand, price, rating, reviews_count, product_url,
                            image_url, is_best_seller, is_amazon_choice
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                        (
                            record.get("snapshot_date"),
                            record.get("category_id"),
                            record.get("asin"),
                            record.get("rank"),
                            record.get("product_name"),
                            record.get("brand"),
                            record.get("price"),
                            record.get("rating"),
                            record.get("reviews_count"),
                            record.get("product_url"),
                            record.get("image_url"),
                            record.get("is_best_seller", 0),
                            record.get("is_amazon_choice", 0),
                        ),
                    )
                    inserted += 1

            conn.commit()

        logging.info(f"Sync upload: inserted={inserted}, updated={updated}")
        return {"success": True, "inserted": inserted, "updated": updated}

    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Sync upload error: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e
