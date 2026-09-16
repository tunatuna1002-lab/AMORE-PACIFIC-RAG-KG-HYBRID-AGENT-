"""
Shared SQLite row helpers (analytics)
=====================================
The synchronous sqlite3 calls the analytics services hand to ``asyncio.to_thread``
(D20: nothing blocks the event loop), plus the one daily-totals query both the
KPI and the trend services need.
"""

from __future__ import annotations

from typing import Any


def fetch_all(sqlite: Any, query: str, params: tuple) -> list:
    """(sync) 단일 SELECT - asyncio.to_thread로 호출해 이벤트 루프를 막지 않는다"""
    with sqlite.get_connection() as conn:
        return conn.execute(query, params).fetchall()


def fetch_pair(
    sqlite: Any, first: tuple[str, tuple], second: tuple[str, tuple]
) -> tuple[list, list]:
    """(sync) 같은 연결로 두 SELECT 실행 - asyncio.to_thread로 호출"""
    with sqlite.get_connection() as conn:
        first_rows = conn.execute(*first).fetchall()
        second_rows = conn.execute(*second).fetchall()
    return first_rows, second_rows


def daily_totals_query(
    category_id: str | None, start_date: str, end_date: str
) -> tuple[str, tuple]:
    """일별 전체 제품 수 (카테고리 필터는 선택)."""
    if category_id:
        return (
            """
                SELECT snapshot_date, COUNT(*) as total_count
                FROM raw_data
                WHERE snapshot_date BETWEEN ? AND ?
                AND category_id = ?
                GROUP BY snapshot_date
                ORDER BY snapshot_date
            """,
            (start_date, end_date, category_id),
        )
    return (
        """
                SELECT snapshot_date, COUNT(*) as total_count
                FROM raw_data
                WHERE snapshot_date BETWEEN ? AND ?
                GROUP BY snapshot_date
                ORDER BY snapshot_date
        """,
        (start_date, end_date),
    )
