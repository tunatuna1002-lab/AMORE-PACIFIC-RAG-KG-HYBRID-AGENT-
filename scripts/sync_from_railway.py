#!/usr/bin/env python3
"""
Railway → 로컬 SQLite 동기화 스크립트

Railway 서버의 SQLite 데이터를 로컬로 동기화합니다.
누락된 날짜만 자동으로 감지하여 다운로드합니다.

Usage:
    python scripts/sync_from_railway.py
    python scripts/sync_from_railway.py --force      # 전체 재동기화
    python scripts/sync_from_railway.py --dry-run    # 실제 동기화 없이 확인만
    python scripts/sync_from_railway.py --url URL    # 커스텀 Railway URL

환경변수:
    RAILWAY_API_URL: Railway API URL (기본값: production URL)
"""

import argparse
import asyncio
import sys
from pathlib import Path

# 프로젝트 루트를 path에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

import httpx

from src.tools.storage.sqlite_storage import SQLiteStorage

# Railway Production URL
DEFAULT_RAILWAY_URL = "https://amore-pacific-rag-kg-hybrid-agent-production.up.railway.app"


async def get_remote_dates(base_url: str) -> list[str]:
    """Railway 서버에서 사용 가능한 날짜 목록 조회"""
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(f"{base_url}/api/sync/dates")
        response.raise_for_status()
        data = response.json()
        return data.get("dates", [])


async def get_remote_status(base_url: str) -> dict:
    """Railway 서버의 데이터 현황 조회"""
    async with httpx.AsyncClient(timeout=30.0) as client:
        response = await client.get(f"{base_url}/api/sync/status")
        response.raise_for_status()
        return response.json()


async def download_date_data(base_url: str, date: str) -> list[dict]:
    """특정 날짜의 데이터 다운로드"""
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.get(f"{base_url}/api/sync/download/{date}")
        response.raise_for_status()
        data = response.json()
        return data.get("records", [])


async def get_local_dates(sqlite: SQLiteStorage) -> list[str]:
    """로컬 SQLite에서 사용 가능한 날짜 목록 조회"""
    with sqlite.get_connection() as conn:
        cursor = conn.execute("""
            SELECT DISTINCT snapshot_date
            FROM raw_data
            ORDER BY snapshot_date
        """)
        return [row[0] for row in cursor.fetchall()]


async def sync_from_railway(
    base_url: str = DEFAULT_RAILWAY_URL, force: bool = False, dry_run: bool = False
) -> bool:
    """Railway → 로컬 SQLite 동기화 실행"""

    print("=" * 60)
    print("Railway → Local SQLite 동기화")
    print("=" * 60)
    print(f"\nRailway URL: {base_url}")

    # 1. Railway 상태 확인
    print("\n[1/4] Railway 서버 상태 확인 중...")
    try:
        remote_status = await get_remote_status(base_url)
        if not remote_status.get("success"):
            print("❌ Railway 서버에 데이터가 없습니다")
            return False

        print(f"✅ Railway: {remote_status.get('total_days', 0)}일치 데이터")
        print(f"   날짜 범위: {remote_status.get('oldest')} ~ {remote_status.get('latest')}")
        print(f"   총 레코드: {remote_status.get('total_records', 0):,}")
    except httpx.HTTPError as e:
        print(f"❌ Railway 연결 실패: {e}")
        return False

    # 2. 로컬 SQLite 상태 확인
    print("\n[2/4] 로컬 SQLite 상태 확인 중...")
    sqlite = SQLiteStorage()
    if not await sqlite.initialize():
        print("❌ SQLite 초기화 실패")
        return False

    local_dates = await get_local_dates(sqlite)
    if local_dates:
        print(f"✅ 로컬: {len(local_dates)}일치 데이터")
        print(f"   날짜 범위: {local_dates[0]} ~ {local_dates[-1]}")
    else:
        print("⚠️ 로컬: 데이터 없음")

    # 3. 누락된 날짜 확인
    print("\n[3/4] 누락된 날짜 확인 중...")
    remote_dates = await get_remote_dates(base_url)
    local_dates_set = set(local_dates)
    remote_dates_set = set(remote_dates)

    if force:
        # 강제 모드: 모든 날짜 재동기화
        missing_dates = sorted(remote_dates)
        print(f"⚠️ 강제 모드: 모든 {len(missing_dates)}일치 데이터 재동기화")
    else:
        # 일반 모드: 누락된 날짜만
        missing_dates = sorted(remote_dates_set - local_dates_set)

    if not missing_dates:
        print("✅ 동기화 필요 없음 - 로컬 데이터가 최신입니다")
        return True

    print(f"📥 누락된 날짜: {len(missing_dates)}일")
    for date in missing_dates:
        print(f"   - {date}")

    if dry_run:
        print("\n[DRY RUN] 실제 동기화는 실행하지 않습니다")
        return True

    # 4. 데이터 다운로드 및 삽입
    print("\n[4/4] 데이터 동기화 중...")
    total_synced = 0
    total_records = 0

    for i, date in enumerate(missing_dates, 1):
        try:
            print(f"   [{i}/{len(missing_dates)}] {date} 다운로드 중...", end=" ", flush=True)

            # 다운로드
            records = await download_date_data(base_url, date)
            if not records:
                print("⚠️ 데이터 없음")
                continue

            # SQLite 삽입
            result = await sqlite.append_rank_records(records)
            if result.get("success"):
                rows_added = result.get("rows_added", 0)
                total_records += rows_added
                total_synced += 1
                print(f"✅ {rows_added} records")
            else:
                print(f"⚠️ 삽입 실패: {result.get('error')}")

        except httpx.HTTPError as e:
            print(f"❌ 다운로드 실패: {e}")
        except Exception as e:
            print(f"❌ 에러: {e}")

    # 5. 결과 출력
    print("\n" + "=" * 60)
    print("동기화 완료!")
    print("=" * 60)
    print("\n결과:")
    print(f"  - 동기화된 날짜: {total_synced}/{len(missing_dates)}")
    print(f"  - 추가된 레코드: {total_records:,}")

    # 최종 상태 확인
    final_dates = await get_local_dates(sqlite)
    print("\n로컬 SQLite 최종 상태:")
    print(f"  - 총 일수: {len(final_dates)}")
    if final_dates:
        print(f"  - 날짜 범위: {final_dates[0]} ~ {final_dates[-1]}")

    return total_synced == len(missing_dates)


def main():
    parser = argparse.ArgumentParser(description="Railway → 로컬 SQLite 동기화 스크립트")
    parser.add_argument("--url", default=DEFAULT_RAILWAY_URL, help="Railway API URL")
    parser.add_argument(
        "--force", action="store_true", help="모든 날짜 재동기화 (기존 데이터 덮어쓰기)"
    )
    parser.add_argument("--dry-run", action="store_true", help="실제 동기화 없이 확인만")

    args = parser.parse_args()

    success = asyncio.run(
        sync_from_railway(base_url=args.url, force=args.force, dry_run=args.dry_run)
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
