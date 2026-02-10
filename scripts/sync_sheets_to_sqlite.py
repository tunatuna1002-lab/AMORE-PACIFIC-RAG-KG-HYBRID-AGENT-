#!/usr/bin/env python3
"""
Google Sheets → SQLite 동기화 스크립트

Google Sheets의 RawData 시트 전체 데이터를 SQLite로 복사합니다.

Usage:
    python scripts/sync_sheets_to_sqlite.py

환경변수:
    GOOGLE_SHEETS_SPREADSHEET_ID: Google Sheets ID (또는 스크립트 내 하드코딩)
"""

import asyncio
import sys
from pathlib import Path

# 프로젝트 루트를 path에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tools.storage.sheets_writer import SheetsWriter
from src.tools.storage.sqlite_storage import SQLiteStorage

# Google Sheets ID (URL에서 추출)
# https://docs.google.com/spreadsheets/d/1cNr3E2WSSbO83XXh_9V92jwc6nfsxjAogcswlHcjV9w/edit
SPREADSHEET_ID = "1cNr3E2WSSbO83XXh_9V92jwc6nfsxjAogcswlHcjV9w"


async def sync_sheets_to_sqlite():
    """Google Sheets 데이터를 SQLite로 동기화"""

    print("=" * 60)
    print("Google Sheets → SQLite 동기화")
    print("=" * 60)

    # 1. Google Sheets 연결
    print("\n[1/4] Google Sheets 연결 중...")
    sheets = SheetsWriter(spreadsheet_id=SPREADSHEET_ID)
    if not await sheets.initialize():
        print("❌ Google Sheets 연결 실패")
        return False
    print("✅ Google Sheets 연결 성공")

    # 2. 전체 데이터 조회 (365일 = 약 1년치)
    print("\n[2/4] Google Sheets에서 데이터 조회 중...")
    records = await sheets.get_rank_history(days=365)

    if not records:
        print("❌ Google Sheets에 데이터가 없습니다")
        return False

    # 날짜 범위 확인
    dates = sorted(set(r.get("snapshot_date", "") for r in records if r.get("snapshot_date")))
    print(f"✅ 조회 완료: {len(records)} records")
    print(f"   날짜 범위: {dates[0]} ~ {dates[-1]}")
    print(f"   총 {len(dates)}일치 데이터")

    # 3. SQLite 연결
    print("\n[3/4] SQLite 초기화 중...")
    sqlite = SQLiteStorage()
    if not await sqlite.initialize():
        print("❌ SQLite 초기화 실패")
        return False

    # 기존 데이터 확인
    existing_stats = sqlite.get_stats()
    print("✅ SQLite 연결 성공")
    print(f"   기존 데이터: {existing_stats.get('raw_data_count', 0)} records")
    if existing_stats.get("date_range", {}).get("min"):
        print(
            f"   기존 날짜 범위: {existing_stats['date_range']['min']} ~ {existing_stats['date_range']['max']}"
        )

    # 4. 데이터 삽입
    print("\n[4/4] SQLite에 데이터 삽입 중...")

    # 배치 크기 설정 (메모리 효율성)
    batch_size = 500
    total_inserted = 0

    for i in range(0, len(records), batch_size):
        batch = records[i : i + batch_size]
        result = await sqlite.append_rank_records(batch)

        if result.get("success"):
            total_inserted += result.get("rows_added", 0)
            print(
                f"   진행: {min(i + batch_size, len(records))}/{len(records)} ({total_inserted} inserted)"
            )
        else:
            print(f"   ⚠️ 배치 삽입 실패: {result.get('error')}")

    # 5. 결과 확인
    print("\n" + "=" * 60)
    print("동기화 완료!")
    print("=" * 60)

    final_stats = sqlite.get_stats()
    print("\nSQLite 최종 상태:")
    print(f"  - 총 레코드: {final_stats.get('raw_data_count', 0)}")
    print(
        f"  - 날짜 범위: {final_stats.get('date_range', {}).get('min')} ~ {final_stats.get('date_range', {}).get('max')}"
    )
    print(f"  - DB 파일 크기: {final_stats.get('file_size_mb', 0)} MB")

    return True


if __name__ == "__main__":
    success = asyncio.run(sync_sheets_to_sqlite())
    sys.exit(0 if success else 1)
