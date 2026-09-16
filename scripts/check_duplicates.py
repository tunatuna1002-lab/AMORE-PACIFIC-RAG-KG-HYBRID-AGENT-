"""
Google Sheets 중복 데이터 확인 스크립트

사용법:
    python scripts/check_duplicates.py

환경 변수:
    GOOGLE_SHEETS_CREDENTIALS_JSON 또는 ./config/google_credentials.json
    GOOGLE_SHEETS_SPREADSHEET_ID
"""

import asyncio
import os
import sys
from collections import Counter, defaultdict

# 프로젝트 루트를 path에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.domain.brand import is_target_brand
from src.tools.storage.sheets_writer import SheetsWriter


async def check_duplicates():
    print("=" * 60)
    print("Google Sheets 중복 데이터 확인")
    print("=" * 60)

    # Sheets 연결
    sheets = SheetsWriter()
    await sheets.initialize()

    # 최근 30일 데이터 로드
    print("\n📊 데이터 로드 중...")
    records = await sheets.get_rank_history(days=30)
    print(f"   총 레코드 수: {len(records)}")

    # 1. 날짜별 레코드 수
    print("\n📅 날짜별 레코드 수:")
    date_counts = Counter(r.get("snapshot_date", "unknown") for r in records)
    for date_str, count in sorted(date_counts.items(), reverse=True)[:10]:
        expected = 60  # 5 카테고리 × 약 12개
        status = "✅" if count <= expected else "⚠️ 중복 의심"
        print(f"   {date_str}: {count}개 {status}")

    # 2. 중복 레코드 찾기 (같은 날짜 + 같은 ASIN + 같은 카테고리)
    print("\n🔍 중복 레코드 확인:")
    seen = defaultdict(list)
    for r in records:
        key = (r.get("snapshot_date"), r.get("asin"), r.get("category_id"))
        seen[key].append(r)

    duplicates = {k: v for k, v in seen.items() if len(v) > 1}

    if duplicates:
        print(f"   ⚠️ 중복 발견: {len(duplicates)}개 그룹")
        print("\n   중복 상세 (최대 10개):")
        for i, (key, items) in enumerate(list(duplicates.items())[:10]):
            date_str, asin, category = key
            print(f"\n   [{i + 1}] {date_str} / {category} / {asin}")
            for item in items:
                rank = item.get("rank", "?")
                name = item.get("product_name", "Unknown")[:30]
                print(f"       - 순위 {rank}: {name}")
    else:
        print("   ✅ 중복 없음")

    # 3. 2026-01-02 상세 분석 (문제가 된 날짜)
    print("\n📋 2026-01-02 상세 분석:")
    jan02_records = [r for r in records if r.get("snapshot_date") == "2026-01-02"]

    if jan02_records:
        print(f"   총 레코드: {len(jan02_records)}개")

        # 카테고리별
        cat_counts = Counter(r.get("category_id", "unknown") for r in jan02_records)
        print("\n   카테고리별:")
        for cat, count in sorted(cat_counts.items()):
            expected_per_cat = 12  # 대략
            status = "✅" if count <= 20 else "⚠️"
            print(f"      {cat}: {count}개 {status}")

        # LANEIGE 제품
        laneige = [r for r in jan02_records if is_target_brand(r.get("brand"))]
        print(f"\n   LANEIGE 제품: {len(laneige)}개")
    else:
        print("   데이터 없음")

    # 4. 2026-01-03 상세 분석 (수정 후 날짜)
    print("\n📋 2026-01-03 상세 분석:")
    jan03_records = [r for r in records if r.get("snapshot_date") == "2026-01-03"]

    if jan03_records:
        print(f"   총 레코드: {len(jan03_records)}개")

        cat_counts = Counter(r.get("category_id", "unknown") for r in jan03_records)
        print("\n   카테고리별:")
        for cat, count in sorted(cat_counts.items()):
            print(f"      {cat}: {count}개")
    else:
        print("   데이터 없음 (아직 새 크롤링 전)")

    print("\n" + "=" * 60)
    print("확인 완료")
    print("=" * 60)

    return duplicates


if __name__ == "__main__":
    duplicates = asyncio.run(check_duplicates())

    if duplicates:
        print("\n💡 중복 데이터 정리 방법:")
        print("   1. Google Sheets에서 수동 삭제")
        print("   2. 또는 scripts/remove_duplicates.py 실행 (별도 작성 필요)")
