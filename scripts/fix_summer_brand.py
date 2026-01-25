#!/usr/bin/env python3
"""
Summer Fridays 브랜드명 데이터 정정 스크립트

문제: 크롤러의 fallback 로직으로 인해 "Summer Fridays"가 "Summer"로 잘못 저장됨
해결: product_name에 "Summer Fridays"가 포함된 경우 brand를 "Summer Fridays"로 수정

사용법:
    # Google Sheets 정정 (dry-run)
    python scripts/fix_summer_brand.py --dry-run

    # Google Sheets 정정 (실제 적용)
    python scripts/fix_summer_brand.py

    # 로컬 JSON 파일 정정
    python scripts/fix_summer_brand.py --local data/raw_products/
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# 프로젝트 루트 추가
sys.path.insert(0, str(Path(__file__).parent.parent))


def fix_local_json_files(data_dir: str, dry_run: bool = True) -> dict:
    """로컬 JSON 파일에서 Summer 브랜드 정정

    Args:
        data_dir: JSON 파일이 있는 디렉토리
        dry_run: True면 변경 없이 미리보기만

    Returns:
        정정 결과 통계
    """
    stats = {"files_checked": 0, "records_fixed": 0, "products_found": []}

    data_path = Path(data_dir)
    if not data_path.exists():
        print(f"[ERROR] 디렉토리가 존재하지 않습니다: {data_dir}")
        return stats

    json_files = list(data_path.glob("**/*.json"))
    print(f"[INFO] {len(json_files)}개의 JSON 파일 검사 중...")

    for json_file in json_files:
        stats["files_checked"] += 1
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            modified = False

            # 리스트 형태의 데이터
            if isinstance(data, list):
                for item in data:
                    if _should_fix_brand(item):
                        stats["records_fixed"] += 1
                        stats["products_found"].append({
                            "file": str(json_file),
                            "asin": item.get("asin", "N/A"),
                            "product_name": item.get("product_name", item.get("title", "N/A"))[:50],
                            "old_brand": item.get("brand"),
                        })
                        if not dry_run:
                            item["brand"] = "Summer Fridays"
                            modified = True

            # 딕셔너리 형태의 데이터 (products 키 포함)
            elif isinstance(data, dict):
                products = data.get("products", [])
                for item in products:
                    if _should_fix_brand(item):
                        stats["records_fixed"] += 1
                        stats["products_found"].append({
                            "file": str(json_file),
                            "asin": item.get("asin", "N/A"),
                            "product_name": item.get("product_name", item.get("title", "N/A"))[:50],
                            "old_brand": item.get("brand"),
                        })
                        if not dry_run:
                            item["brand"] = "Summer Fridays"
                            modified = True

            # 변경사항 저장
            if modified and not dry_run:
                with open(json_file, "w", encoding="utf-8") as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                print(f"  [UPDATED] {json_file}")

        except (json.JSONDecodeError, IOError) as e:
            print(f"  [WARN] 파일 읽기 실패: {json_file} - {e}")

    return stats


def _should_fix_brand(item: dict) -> bool:
    """브랜드 정정이 필요한지 확인

    조건:
    1. brand가 "Summer"인 경우
    2. product_name(또는 title)에 "Summer Fridays"가 포함된 경우
    """
    brand = item.get("brand", "")
    product_name = item.get("product_name", item.get("title", ""))

    if brand and brand.lower() == "summer":
        if "summer fridays" in product_name.lower():
            return True

    return False


def fix_google_sheets(dry_run: bool = True) -> dict:
    """Google Sheets에서 Summer 브랜드 정정

    Args:
        dry_run: True면 변경 없이 미리보기만

    Returns:
        정정 결과 통계
    """
    stats = {"sheets_checked": 0, "records_fixed": 0, "products_found": []}

    try:
        from src.tools.sheets_writer import SheetsWriter
    except ImportError:
        print("[ERROR] SheetsWriter를 import할 수 없습니다.")
        print("        Google Sheets 연동을 위해서는 credentials가 필요합니다.")
        return stats

    try:
        writer = SheetsWriter()
    except Exception as e:
        print(f"[ERROR] SheetsWriter 초기화 실패: {e}")
        return stats

    # 카테고리별 시트 검사
    categories = ["lip_care", "skin_care", "beauty", "lip_makeup", "face_powder"]

    for category in categories:
        stats["sheets_checked"] += 1
        sheet_name = category

        try:
            # 데이터 읽기
            data = writer.read_all_data(sheet_name)
            if not data:
                print(f"  [INFO] {sheet_name}: 데이터 없음")
                continue

            print(f"  [INFO] {sheet_name}: {len(data)}개 레코드 검사 중...")

            # 수정이 필요한 행 찾기
            rows_to_fix = []
            for idx, row in enumerate(data):
                if _should_fix_brand(row):
                    stats["records_fixed"] += 1
                    stats["products_found"].append({
                        "sheet": sheet_name,
                        "row": idx + 2,  # 헤더 + 1-based index
                        "asin": row.get("asin", "N/A"),
                        "product_name": row.get("product_name", row.get("title", "N/A"))[:50],
                        "old_brand": row.get("brand"),
                    })
                    rows_to_fix.append(idx + 2)  # 1-based row number with header

            if rows_to_fix and not dry_run:
                # 브랜드 열 인덱스 찾기
                headers = writer.get_headers(sheet_name)
                brand_col_idx = headers.index("brand") if "brand" in headers else -1

                if brand_col_idx >= 0:
                    for row_num in rows_to_fix:
                        col_letter = chr(ord('A') + brand_col_idx)
                        cell = f"{col_letter}{row_num}"
                        writer.update_cell(sheet_name, cell, "Summer Fridays")
                        print(f"    [UPDATED] {sheet_name}!{cell} = 'Summer Fridays'")
                else:
                    print(f"  [WARN] {sheet_name}: 'brand' 열을 찾을 수 없음")

        except Exception as e:
            print(f"  [ERROR] {sheet_name} 처리 실패: {e}")

    return stats


def print_report(stats: dict, dry_run: bool):
    """정정 결과 리포트 출력"""
    print("\n" + "=" * 60)
    print("Summer Fridays 브랜드 정정 결과")
    print("=" * 60)

    if dry_run:
        print("[MODE] DRY-RUN (변경 없이 미리보기만)")
    else:
        print("[MODE] LIVE (실제 데이터 변경 적용)")

    print(f"\n검사한 파일/시트 수: {stats.get('files_checked', 0) + stats.get('sheets_checked', 0)}")
    print(f"정정 대상 레코드 수: {stats['records_fixed']}")

    if stats["products_found"]:
        print("\n정정 대상 제품 목록:")
        print("-" * 60)
        for i, product in enumerate(stats["products_found"][:20], 1):
            location = product.get("file") or f"{product.get('sheet')}!Row{product.get('row')}"
            print(f"  {i}. [{product['old_brand']}] → [Summer Fridays]")
            print(f"     ASIN: {product['asin']}")
            print(f"     제품명: {product['product_name']}...")
            print(f"     위치: {location}")
            print()

        if len(stats["products_found"]) > 20:
            print(f"  ... 외 {len(stats['products_found']) - 20}개 더")

    if dry_run and stats["records_fixed"] > 0:
        print("\n[TIP] 실제 적용하려면 --dry-run 옵션을 제거하세요:")
        print("      python scripts/fix_summer_brand.py")


def main():
    parser = argparse.ArgumentParser(
        description="Summer Fridays 브랜드명 데이터 정정 스크립트"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="변경 없이 미리보기만 (기본값: False)"
    )
    parser.add_argument(
        "--local",
        type=str,
        default=None,
        help="로컬 JSON 파일 경로 (예: data/raw_products/)"
    )
    parser.add_argument(
        "--sheets",
        action="store_true",
        help="Google Sheets 정정 (기본값)"
    )

    args = parser.parse_args()

    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Summer Fridays 브랜드 정정 시작")

    total_stats = {
        "files_checked": 0,
        "sheets_checked": 0,
        "records_fixed": 0,
        "products_found": []
    }

    # 로컬 JSON 파일 정정
    if args.local:
        print(f"\n[STEP 1] 로컬 JSON 파일 정정: {args.local}")
        local_stats = fix_local_json_files(args.local, args.dry_run)
        total_stats["files_checked"] = local_stats["files_checked"]
        total_stats["records_fixed"] += local_stats["records_fixed"]
        total_stats["products_found"].extend(local_stats["products_found"])

    # Google Sheets 정정 (기본)
    if args.sheets or not args.local:
        print("\n[STEP 2] Google Sheets 정정")
        sheets_stats = fix_google_sheets(args.dry_run)
        total_stats["sheets_checked"] = sheets_stats["sheets_checked"]
        total_stats["records_fixed"] += sheets_stats["records_fixed"]
        total_stats["products_found"].extend(sheets_stats["products_found"])

    # 결과 리포트
    print_report(total_stats, args.dry_run)


if __name__ == "__main__":
    main()
