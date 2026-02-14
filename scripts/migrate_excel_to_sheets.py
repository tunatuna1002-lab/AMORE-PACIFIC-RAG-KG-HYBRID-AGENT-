"""
엑셀 데이터를 Google Sheets로 마이그레이션하는 스크립트
"""

import os
import re
from collections import Counter

import pandas as pd

# 환경변수 로드
from dotenv import load_dotenv

load_dotenv(".env")

from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build

# ===== 1. 엑셀 데이터 로드 =====
print("1. 엑셀 데이터 로드 중...")
df = pd.read_excel("ranking_data.xlsx", sheet_name=0)
print(f"   총 {len(df)}개 레코드")

# ===== 2. 카테고리 매핑 =====
CATEGORY_MAP = {
    "lip_care_products": "lip_care",
    "beauty_overall": "beauty",
    "skin_care_products": "skin_care",
    "lip_makeup": "lip_makeup",
    "face_powder": "face_powder",
    "skincare_cream": "skin_care",
    "anti_aging": "skin_care",
    "face_mask": "skin_care",
    "foundation": "face_powder",
    "lip_gloss": "lip_care",
    "overall_ranking": "beauty",
    "item_800": "beauty",
    "item_1005": "beauty",
    "item_904": "beauty",
    "item_803": "beauty",
}


# ===== 3. ASIN 추출 함수 =====
def extract_asin(url):
    if pd.isna(url):
        return ""
    match = re.search(r"/dp/([A-Z0-9]{10})", str(url))
    return match.group(1) if match else ""


# ===== 4. 데이터 변환 =====
print("2. 데이터 변환 중...")
converted_rows = []

for _, row in df.iterrows():
    # 카테고리 변환
    category = CATEGORY_MAP.get(row["category_key"], "beauty")

    # ASIN 추출
    asin = extract_asin(row["product_url"])

    # 누락된 필드는 빈값 또는 기본값
    converted_row = [
        str(row["date_kst"])[:10],  # snapshot_date (YYYY-MM-DD)
        category,  # category_id
        int(row["rank"]) if pd.notna(row["rank"]) else 999,  # rank
        asin,  # asin
        str(row["product_name_raw"])[:200]
        if pd.notna(row["product_name_raw"])
        else "",  # product_name
        str(row["brand_raw"]) if pd.notna(row["brand_raw"]) else "",  # brand
        "",  # price (없음)
        "",  # rating (없음)
        "",  # reviews_count (없음)
        "",  # badge (없음)
        str(row["product_url"]) if pd.notna(row["product_url"]) else "",  # product_url
    ]
    converted_rows.append(converted_row)

print(f"   변환 완료: {len(converted_rows)}개")

# ===== 5. 기존 데이터와 중복 제거 =====
print("3. 기존 Google Sheets 데이터 확인 중...")

SCOPES = ["https://www.googleapis.com/auth/spreadsheets"]
creds = Credentials.from_service_account_file(
    os.getenv("GOOGLE_SHEETS_CREDENTIALS_PATH", "./config/google_credentials.json"), scopes=SCOPES
)
service = build("sheets", "v4", credentials=creds)
spreadsheet_id = os.getenv("GOOGLE_SHEETS_SPREADSHEET_ID")

# 기존 날짜 확인
result = (
    service.spreadsheets().values().get(spreadsheetId=spreadsheet_id, range="RawData!A:A").execute()
)
existing_dates = {row[0] for row in result.get("values", [])[1:] if row}
print(f"   기존 날짜: {sorted(existing_dates)}")

# 새로운 데이터만 필터링 (기존 날짜 제외)
new_rows = [row for row in converted_rows if row[0] not in existing_dates]
print(f"   새로 추가할 데이터: {len(new_rows)}개")

# 날짜별 분포
new_dates = Counter(row[0] for row in new_rows)
print(f"   날짜별 분포: {dict(sorted(new_dates.items()))}")

# ===== 6. Google Sheets에 추가 =====
if new_rows:
    print("4. Google Sheets에 데이터 추가 중...")

    # 배치로 나누어 추가 (한 번에 1000개씩)
    batch_size = 1000
    total_added = 0

    for i in range(0, len(new_rows), batch_size):
        batch = new_rows[i : i + batch_size]

        body = {"values": batch}
        result = (
            service.spreadsheets()
            .values()
            .append(
                spreadsheetId=spreadsheet_id,
                range="RawData!A:K",
                valueInputOption="RAW",
                insertDataOption="INSERT_ROWS",
                body=body,
            )
            .execute()
        )

        total_added += len(batch)
        print(f"   {total_added}/{len(new_rows)} 추가 완료")

    print(f"\n✅ 완료! 총 {total_added}개 레코드가 추가되었습니다.")
else:
    print("\n⚠️  추가할 새 데이터가 없습니다 (모두 중복)")

# ===== 7. 최종 확인 =====
print("\n5. 최종 데이터 확인...")
result = (
    service.spreadsheets().values().get(spreadsheetId=spreadsheet_id, range="RawData!A:A").execute()
)
all_dates = [row[0] for row in result.get("values", [])[1:] if row]
date_counts = Counter(all_dates)
print("=== Google Sheets 최종 날짜별 현황 ===")
for date in sorted(date_counts.keys()):
    print(f"   {date}: {date_counts[date]}개")
print(f"\n총 레코드 수: {len(all_dates)}")
