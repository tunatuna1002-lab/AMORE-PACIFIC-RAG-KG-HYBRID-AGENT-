#!/usr/bin/env python3
"""
로컬 → Railway SQLite 동기화 스크립트

로컬 SQLite 데이터를 Railway 서버로 업로드합니다.
Railway에 없는 날짜의 데이터만 업로드합니다.

Usage:
    python scripts/sync_to_railway.py
    python scripts/sync_to_railway.py --force      # 전체 재동기화
    python scripts/sync_to_railway.py --dry-run    # 실제 동기화 없이 확인만
    python scripts/sync_to_railway.py --date 2026-01-15  # 특정 날짜만

환경변수:
    RAILWAY_API_URL: Railway API URL (기본값: production URL)
    API_KEY: Railway API 인증 키
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path

# 프로젝트 루트를 path에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

import httpx

from src.tools.sqlite_storage import SQLiteStorage

# Railway Production URL
DEFAULT_RAILWAY_URL = "https://amore-pacific-rag-kg-hybrid-agent-production.up.railway.app"


async def get_remote_dates(base_url: str) -> list[str]:
    """Railway 서버에서 사용 가능한 날짜 목록 조회"""
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(f"{base_url}/api/sync/dates")
        response.raise_for_status()
        data = response.json()
        return data.get("dates", [])


async def get_local_dates(sqlite: SQLiteStorage) -> list[str]:
    """로컬 SQLite에서 사용 가능한 날짜 목록 조회"""
    with sqlite.get_connection() as conn:
        cursor = conn.execute("""
            SELECT DISTINCT snapshot_date
            FROM raw_data
            ORDER BY snapshot_date
        """)
        return [row[0] for row in cursor.fetchall()]


async def get_local_data_for_date(sqlite: SQLiteStorage, date: str) -> list[dict]:
    """로컬 SQLite에서 특정 날짜 데이터 조회"""
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

    return [dict(zip(columns, row, strict=False)) for row in rows]


async def upload_to_railway(base_url: str, records: list[dict], api_key: str = "") -> dict:
    """Railway 서버로 데이터 업로드"""
    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(
            f"{base_url}/api/sync/upload",
            json={"records": records, "api_key": api_key},
        )
        response.raise_for_status()
        return response.json()


async def sync_to_railway(
    base_url: str = DEFAULT_RAILWAY_URL,
    force: bool = False,
    dry_run: bool = False,
    specific_date: str | None = None,
) -> bool:
    """로컬 → Railway SQLite 동기화 실행"""

    print("=" * 60)
    print("Local → Railway SQLite 동기화")
    print("=" * 60)
    print(f"Railway URL: {base_url}")
    print(f"Force mode: {force}")
    print(f"Dry run: {dry_run}")
    if specific_date:
        print(f"Specific date: {specific_date}")
    print()

    # API 키
    api_key = os.getenv("API_KEY", "")

    # 1. 로컬 SQLite 초기화
    print("1. 로컬 SQLite 연결 중...")
    sqlite = SQLiteStorage()
    await sqlite.initialize()
    local_dates = await get_local_dates(sqlite)
    print(f"   로컬 데이터: {len(local_dates)}일")
    if local_dates:
        print(f"   기간: {local_dates[0]} ~ {local_dates[-1]}")

    # 2. Railway 날짜 목록 조회
    print("\n2. Railway 서버 상태 확인 중...")
    try:
        remote_dates = await get_remote_dates(base_url)
        print(f"   Railway 데이터: {len(remote_dates)}일")
        if remote_dates:
            print(f"   기간: {remote_dates[0]} ~ {remote_dates[-1]}")
    except Exception as e:
        print(f"   ⚠️  Railway 연결 실패: {e}")
        remote_dates = []

    # 3. 업로드 대상 날짜 결정
    print("\n3. 업로드 대상 분석 중...")
    if specific_date:
        # 특정 날짜만
        if specific_date in local_dates:
            dates_to_upload = [specific_date]
        else:
            print(f"   ❌ 날짜 {specific_date}가 로컬에 없습니다.")
            return False
    elif force:
        # 전체 로컬 날짜
        dates_to_upload = local_dates
    else:
        # Railway에 없는 날짜만
        remote_set = set(remote_dates)
        dates_to_upload = [d for d in local_dates if d not in remote_set]

    print(f"   업로드 대상: {len(dates_to_upload)}일")
    if dates_to_upload:
        print(f"   날짜: {dates_to_upload[0]} ~ {dates_to_upload[-1]}")

    if not dates_to_upload:
        print("\n✅ 동기화할 데이터가 없습니다. Railway가 최신 상태입니다.")
        return True

    if dry_run:
        print("\n[Dry Run] 실제 업로드하지 않습니다.")
        for date in dates_to_upload:
            records = await get_local_data_for_date(sqlite, date)
            print(f"   {date}: {len(records)} 레코드")
        return True

    # 4. 날짜별 업로드
    print("\n4. 데이터 업로드 중...")
    total_inserted = 0
    total_updated = 0
    failed_dates = []

    for i, date in enumerate(dates_to_upload, 1):
        try:
            records = await get_local_data_for_date(sqlite, date)
            print(
                f"   [{i}/{len(dates_to_upload)}] {date}: {len(records)} 레코드 업로드 중...",
                end=" ",
            )

            result = await upload_to_railway(base_url, records, api_key)

            inserted = result.get("inserted", 0)
            updated = result.get("updated", 0)
            total_inserted += inserted
            total_updated += updated
            print(f"✓ (inserted={inserted}, updated={updated})")

        except Exception as e:
            print(f"✗ 실패: {e}")
            failed_dates.append(date)

    # 5. 결과 요약
    print("\n" + "=" * 60)
    print("동기화 완료")
    print("=" * 60)
    print(f"총 업로드 대상: {len(dates_to_upload)}일")
    print(f"성공: {len(dates_to_upload) - len(failed_dates)}일")
    print(f"실패: {len(failed_dates)}일")
    print(f"신규 삽입: {total_inserted} 레코드")
    print(f"업데이트: {total_updated} 레코드")

    if failed_dates:
        print(f"\n실패한 날짜: {failed_dates}")
        return False

    return True


def main():
    parser = argparse.ArgumentParser(description="로컬 → Railway SQLite 동기화")
    parser.add_argument(
        "--url",
        default=os.getenv("RAILWAY_API_URL", DEFAULT_RAILWAY_URL),
        help="Railway API URL",
    )
    parser.add_argument("--force", action="store_true", help="전체 재동기화")
    parser.add_argument("--dry-run", action="store_true", help="실제 동기화 없이 확인만")
    parser.add_argument("--date", help="특정 날짜만 동기화 (YYYY-MM-DD)")
    args = parser.parse_args()

    success = asyncio.run(
        sync_to_railway(
            base_url=args.url,
            force=args.force,
            dry_run=args.dry_run,
            specific_date=args.date,
        )
    )
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
