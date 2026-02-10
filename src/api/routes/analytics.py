"""
Analytics Routes (KPI / SoS)
=============================
카테고리별 KPI, SoS(Share of Shelf), 브랜드 비교 엔드포인트
"""

import json
import logging
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

from fastapi import APIRouter

from src.tools.storage.sqlite_storage import get_sqlite_storage

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Analytics"])


# ============= Helper =============


def _load_crawl_data_for_sos():
    """JSON 파일에서 크롤링 데이터 로드 (SQLite fallback)"""
    crawl_path = Path("./data/latest_crawl_result.json")
    if crawl_path.exists():
        with open(crawl_path, encoding="utf-8") as f:
            return json.load(f)
    return None


# ============= Category KPI =============


@router.get("/api/category/kpi")
async def get_category_kpi(
    category_id: str,
    start_date: str | None = None,
    end_date: str | None = None,
    brand: str = "LANEIGE",
):
    """
    카테고리별 KPI 데이터 조회 (기간 필터링 지원)

    Args:
        category_id: 카테고리 ID (beauty_personal_care, skin_care, lip_care, lip_makeup, face_powder)
        start_date: 시작일 (YYYY-MM-DD)
        end_date: 종료일 (YYYY-MM-DD)
        brand: 타겟 브랜드 (기본값: LANEIGE)

    Returns:
        KPI 데이터: sos, best_rank, cpi, new_competitors
    """
    try:
        # 날짜 범위 설정
        if not end_date:
            end_date = datetime.now().strftime("%Y-%m-%d")
        if not start_date:
            start_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")

        rows = []

        # SQLite에서 데이터 조회
        try:
            sqlite = get_sqlite_storage()
            await sqlite.initialize()

            query = """
                SELECT snapshot_date, rank, brand, price
                FROM raw_data
                WHERE snapshot_date BETWEEN ? AND ?
                AND category_id = ?
                ORDER BY snapshot_date DESC, rank ASC
            """
            with sqlite.get_connection() as conn:
                cursor = conn.execute(query, (start_date, end_date, category_id))
                rows = cursor.fetchall()
        except Exception as db_err:
            logging.warning(f"SQLite query failed for category KPI: {db_err}")

        # JSON fallback
        if not rows:
            crawl_data = _load_crawl_data_for_sos()
            if crawl_data and crawl_data.get("categories", {}).get(category_id):
                cat_data = crawl_data["categories"][category_id]
                snapshot_date = crawl_data.get("snapshot_date", end_date)
                for product in cat_data.get("products", []):
                    rows.append(
                        (
                            snapshot_date,
                            product.get("rank", 100),
                            product.get("brand", "Unknown"),
                            product.get("price"),
                        )
                    )

        if not rows:
            return {
                "success": True,
                "message": f"해당 기간({start_date} ~ {end_date})에 데이터가 없습니다.",
                "data": None,
                "period": {"start": start_date, "end": end_date},
            }

        # KPI 계산
        total_products = len(rows)
        brand_products = [r for r in rows if r[2] and brand.lower() in r[2].lower()]
        brand_count = len(brand_products)

        # SoS (Share of Shelf)
        sos = (brand_count / total_products * 100) if total_products > 0 else 0

        # Best Rank
        brand_ranks = [r[1] for r in brand_products if r[1]]
        best_rank = min(brand_ranks) if brand_ranks else None

        # CPI (Competitive Price Index) - 브랜드 평균가 / 전체 평균가 * 100
        brand_prices = [r[3] for r in brand_products if r[3] and r[3] > 0]
        all_prices = [r[3] for r in rows if r[3] and r[3] > 0]

        if brand_prices and all_prices:
            brand_avg_price = sum(brand_prices) / len(brand_prices)
            all_avg_price = sum(all_prices) / len(all_prices)
            cpi = (brand_avg_price / all_avg_price * 100) if all_avg_price > 0 else 100
        else:
            cpi = 100

        # New Competitors (최근 7일 내 신규 진입 - 간소화된 계산)
        new_competitors = max(0, total_products - brand_count - 50)

        return {
            "success": True,
            "data": {
                "category_id": category_id,
                "sos": round(sos, 1),
                "best_rank": best_rank,
                "cpi": round(cpi, 0),
                "new_competitors": new_competitors,
                "brand": brand,
                "product_count": brand_count,
                "total_products": total_products,
            },
            "period": {"start": start_date, "end": end_date},
        }

    except Exception as e:
        logging.error(f"Category KPI API error: {e}")
        return {"success": False, "error": str(e), "data": None}


# ============= SoS (Share of Shelf) =============


@router.get("/api/sos/category")
async def get_sos_by_category(
    start_date: str | None = None,
    end_date: str | None = None,
    compare_brands: str | None = None,
):
    """
    카테고리별 SoS (Share of Shelf) 데이터 조회

    SoS = (해당 브랜드 제품 수 / Top 100) * 100

    Args:
        start_date: 시작일 (YYYY-MM-DD)
        end_date: 종료일 (YYYY-MM-DD)
        compare_brands: 비교할 브랜드 (콤마로 구분)

    Returns:
        카테고리별 SoS 데이터
    """
    try:
        # 비교 브랜드 파싱
        compare_brand_list = []
        if compare_brands:
            compare_brand_list = [b.strip() for b in compare_brands.split(",") if b.strip()]

        # 날짜 범위 설정
        if not end_date:
            end_date = datetime.now().strftime("%Y-%m-%d")
        if not start_date:
            start_date = end_date

        # SQLite 먼저 시도
        rows = []
        try:
            sqlite = get_sqlite_storage()
            await sqlite.initialize()

            query = """
                SELECT snapshot_date, category_id, brand, COUNT(*) as product_count
                FROM raw_data
                WHERE snapshot_date BETWEEN ? AND ?
                GROUP BY snapshot_date, category_id, brand
                ORDER BY snapshot_date DESC, category_id, product_count DESC
            """
            with sqlite.get_connection() as conn:
                cursor = conn.execute(query, (start_date, end_date))
                rows = cursor.fetchall()
        except Exception as db_err:
            logging.warning(f"SQLite query failed, using JSON fallback: {db_err}")

        # SQLite 데이터 없으면 JSON fallback
        if not rows:
            crawl_data = _load_crawl_data_for_sos()
            if crawl_data and crawl_data.get("categories"):
                snapshot_date = crawl_data.get("snapshot_date", end_date)
                for cat_id, cat_data in crawl_data.get("categories", {}).items():
                    for product in cat_data.get("products", []):
                        brand = product.get("brand", "Unknown")
                        rows.append((snapshot_date, cat_id, brand, 1))

        if not rows:
            return {
                "success": True,
                "message": f"해당 기간({start_date} ~ {end_date})에 데이터가 없습니다.",
                "data": [],
                "period": {"start": start_date, "end": end_date},
            }

        # 데이터 집계
        category_data = {}
        dates_set = set()

        for row in rows:
            if len(row) == 4:
                snapshot_date, category_id, brand, count = row
            else:
                snapshot_date, category_id, brand, count = row[0], row[1], row[2], row[3]
            dates_set.add(snapshot_date)

            if category_id not in category_data:
                category_data[category_id] = {}
            if brand not in category_data[category_id]:
                category_data[category_id][brand] = {"dates": {}, "total_count": 0}

            category_data[category_id][brand]["dates"][snapshot_date] = count
            category_data[category_id][brand]["total_count"] += count

        # SoS 계산 (기간 평균)
        num_dates = len(dates_set)
        result_data = []

        # 카테고리 계층 구조 로드
        hierarchy_path = Path("./config/category_hierarchy.json")
        hierarchy_data = {}
        if hierarchy_path.exists():
            with open(hierarchy_path, encoding="utf-8") as f:
                hierarchy_data = json.load(f).get("categories", {})

        # 카테고리 메타 정보 (계층 구조 포함)
        category_meta = {
            "beauty": {
                "name": "Beauty & Personal Care",
                "level": 0,
                "parent_id": None,
                "indent": 0,
                "order": 0,
            },
            "skin_care": {
                "name": "Skin Care",
                "level": 1,
                "parent_id": "beauty",
                "indent": 1,
                "order": 1,
            },
            "lip_care": {
                "name": "Lip Care",
                "level": 2,
                "parent_id": "skin_care",
                "indent": 2,
                "order": 2,
            },
            "lip_makeup": {
                "name": "Lip Makeup",
                "level": 2,
                "parent_id": "makeup",
                "indent": 1,
                "order": 3,
            },
            "face_powder": {
                "name": "Face Powder",
                "level": 3,
                "parent_id": "face_makeup",
                "indent": 2,
                "order": 4,
            },
        }

        # hierarchy_data에서 정보 업데이트
        for cat_id, meta in category_meta.items():
            if cat_id in hierarchy_data:
                meta["name"] = hierarchy_data[cat_id].get("name", meta["name"])
                meta["level"] = hierarchy_data[cat_id].get("level", meta["level"])
                meta["parent_id"] = hierarchy_data[cat_id].get("parent_id", meta["parent_id"])

        for category_id, brands in category_data.items():
            # 해당 카테고리의 총 제품 수 (기간 합계)
            total_products_in_category = sum(b["total_count"] for b in brands.values())

            # LANEIGE SoS
            laneige_count = 0
            laneige_dates = set()
            laneige_variants = ["LANEIGE", "Laneige", "laneige"]
            for variant in laneige_variants:
                if variant in brands:
                    laneige_count += brands[variant]["total_count"]
                    if "dates" in brands[variant]:
                        laneige_dates.update(brands[variant]["dates"])
            laneige_appearance_days = len(laneige_dates)

            laneige_sos = (
                (laneige_count / total_products_in_category * 100)
                if total_products_in_category > 0
                else 0
            )

            # 평균 SoS (전체 브랜드 수 기준)
            num_brands = len(brands)
            avg_sos = (100 / num_brands) if num_brands > 0 else 0

            # 비교 브랜드 SoS
            compare_sos = {}
            for compare_brand in compare_brand_list:
                brand_count = 0
                for brand_name, brand_data in brands.items():
                    if compare_brand.lower() in brand_name.lower():
                        brand_count += brand_data["total_count"]
                compare_sos[compare_brand] = (
                    (brand_count / total_products_in_category * 100)
                    if total_products_in_category > 0
                    else 0
                )

            # 카테고리 메타 정보 가져오기
            meta = category_meta.get(
                category_id,
                {"name": category_id, "level": 0, "parent_id": None, "indent": 0, "order": 99},
            )

            result_data.append(
                {
                    "category_id": category_id,
                    "category_name": meta["name"],
                    "level": meta["level"],
                    "parent_id": meta["parent_id"],
                    "indent": meta["indent"],
                    "order": meta["order"],
                    "total_products": total_products_in_category // num_dates
                    if num_dates > 0
                    else 0,
                    "laneige_sos": round(laneige_sos, 2),
                    "laneige_count": round(laneige_count / num_dates, 1) if num_dates > 0 else 0,
                    "laneige_appearance_days": laneige_appearance_days,
                    "laneige_appearance_rate": round(laneige_appearance_days / num_dates * 100, 1)
                    if num_dates > 0
                    else 0,
                    "avg_sos": round(avg_sos, 2),
                    "compare_brands": compare_sos,
                    "num_dates": num_dates,
                }
            )

        # 계층 구조 순서대로 정렬
        result_data.sort(key=lambda x: x.get("order", 99))

        return {
            "success": True,
            "period": {"start": start_date, "end": end_date, "days": num_dates},
            "data": result_data,
            "compare_brands": compare_brand_list,
            "hierarchy_info": {
                "description": "각 카테고리는 자체 Top 100 기준으로 독립 계산됩니다.",
                "note": "상위 카테고리와 하위 카테고리의 SoS는 서로 다른 랭킹에서 계산됩니다.",
            },
        }

    except Exception as e:
        logging.error(f"SoS category API error: {e}")
        return {"success": False, "error": str(e)}


@router.get("/api/sos/brands")
async def get_available_brands(category_id: str | None = None, min_count: int = 1):
    """
    비교 가능한 브랜드 목록 조회 (Top 100에 포함된 브랜드들)

    Args:
        category_id: 특정 카테고리만 조회 (선택)
        min_count: 최소 제품 수 (기본: 1)

    Returns:
        브랜드 목록 (제품 수 기준 정렬)
    """
    try:
        end_date = datetime.now().strftime("%Y-%m-%d")
        start_date = (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")

        rows = []
        # SQLite 먼저 시도
        try:
            sqlite = get_sqlite_storage()
            await sqlite.initialize()

            if category_id:
                query = """
                    SELECT brand, COUNT(DISTINCT asin) as product_count,
                           COUNT(DISTINCT snapshot_date) as days_present
                    FROM raw_data
                    WHERE snapshot_date BETWEEN ? AND ?
                    AND category_id = ?
                    AND LOWER(brand) != 'unknown'
                    GROUP BY brand
                    HAVING product_count >= ?
                    ORDER BY product_count DESC
                """
                params = (start_date, end_date, category_id, min_count)
            else:
                query = """
                    SELECT brand, COUNT(DISTINCT asin) as product_count,
                           COUNT(DISTINCT snapshot_date) as days_present
                    FROM raw_data
                    WHERE snapshot_date BETWEEN ? AND ?
                    AND LOWER(brand) != 'unknown'
                    GROUP BY brand
                    HAVING product_count >= ?
                    ORDER BY product_count DESC
                """
                params = (start_date, end_date, min_count)

            with sqlite.get_connection() as conn:
                cursor = conn.execute(query, params)
                rows = cursor.fetchall()
        except Exception as db_err:
            logging.warning(f"SQLite query failed for brands: {db_err}")

        # SQLite 데이터 없으면 JSON fallback
        brands = []
        if not rows:
            crawl_data = _load_crawl_data_for_sos()
            if crawl_data and crawl_data.get("categories"):
                brand_counts = {}
                for cat_id, cat_data in crawl_data.get("categories", {}).items():
                    if category_id and cat_id != category_id:
                        continue
                    for product in cat_data.get("products", []):
                        brand = product.get("brand", "Unknown")
                        if brand:
                            brand_counts[brand] = brand_counts.get(brand, 0) + 1

                for brand_name, count in sorted(brand_counts.items(), key=lambda x: -x[1]):
                    if (
                        count >= min_count
                        and brand_name.strip()
                        and brand_name.lower() != "unknown"
                    ):
                        brands.append(
                            {
                                "name": brand_name,
                                "product_count": count,
                                "days_present": 1,
                                "is_laneige": "laneige" in brand_name.lower(),
                            }
                        )
        else:
            for row in rows:
                brand_name, product_count, days_present = row
                if brand_name and brand_name.strip() and brand_name.lower() != "unknown":
                    brands.append(
                        {
                            "name": brand_name,
                            "product_count": product_count,
                            "days_present": days_present,
                            "is_laneige": "laneige" in brand_name.lower(),
                        }
                    )

        return {
            "success": True,
            "period": {"start": start_date, "end": end_date},
            "category_id": category_id,
            "brands": brands,
            "total_brands": len(brands),
        }

    except Exception as e:
        logging.error(f"SoS brands API error: {e}")
        return {"success": False, "error": str(e)}


@router.get("/api/sos/trend")
async def get_sos_trend(
    brand: str = "LANEIGE",
    category_id: str | None = None,
    days: int = 7,
    start_date: str | None = None,
    end_date: str | None = None,
):
    """
    브랜드의 SoS 추세 데이터 (일별)

    Args:
        brand: 브랜드명 (기본: LANEIGE)
        category_id: 카테고리 (선택, 없으면 전체)
        days: 조회 기간 (기본: 7일, start_date/end_date가 없을 때만 사용)
        start_date: 시작 날짜 (YYYY-MM-DD)
        end_date: 종료 날짜 (YYYY-MM-DD)

    Returns:
        일별 SoS 추세 데이터
    """
    try:
        sqlite = get_sqlite_storage()
        await sqlite.initialize()

        # start_date/end_date가 제공되면 사용, 아니면 days 기반으로 계산
        if start_date and end_date:
            pass  # 그대로 사용
        else:
            end_date = datetime.now().strftime("%Y-%m-%d")
            start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

        # 일별 전체 제품 수
        if category_id:
            total_query = """
                SELECT snapshot_date, COUNT(*) as total_count
                FROM raw_data
                WHERE snapshot_date BETWEEN ? AND ?
                AND category_id = ?
                GROUP BY snapshot_date
                ORDER BY snapshot_date
            """
            total_params = (start_date, end_date, category_id)

            brand_query = """
                SELECT snapshot_date, COUNT(*) as brand_count
                FROM raw_data
                WHERE snapshot_date BETWEEN ? AND ?
                AND category_id = ?
                AND LOWER(brand) LIKE ?
                GROUP BY snapshot_date
                ORDER BY snapshot_date
            """
            brand_params = (start_date, end_date, category_id, f"%{brand.lower()}%")
        else:
            total_query = """
                SELECT snapshot_date, COUNT(*) as total_count
                FROM raw_data
                WHERE snapshot_date BETWEEN ? AND ?
                GROUP BY snapshot_date
                ORDER BY snapshot_date
            """
            total_params = (start_date, end_date)

            brand_query = """
                SELECT snapshot_date, COUNT(*) as brand_count
                FROM raw_data
                WHERE snapshot_date BETWEEN ? AND ?
                AND LOWER(brand) LIKE ?
                GROUP BY snapshot_date
                ORDER BY snapshot_date
            """
            brand_params = (start_date, end_date, f"%{brand.lower()}%")

        with sqlite.get_connection() as conn:
            # 전체 카운트
            cursor = conn.execute(total_query, total_params)
            total_rows = cursor.fetchall()
            total_by_date = {row[0]: row[1] for row in total_rows}

            # 브랜드 카운트
            cursor = conn.execute(brand_query, brand_params)
            brand_rows = cursor.fetchall()
            brand_by_date = {row[0]: row[1] for row in brand_rows}

        # SoS 계산
        trend_data = []
        for date, total in sorted(total_by_date.items()):
            brand_count = brand_by_date.get(date, 0)
            sos = (brand_count / total * 100) if total > 0 else 0
            trend_data.append(
                {
                    "date": date,
                    "total_products": total,
                    "brand_count": brand_count,
                    "sos": round(sos, 2),
                }
            )

        return {
            "success": True,
            "brand": brand,
            "category_id": category_id,
            "period": {"start": start_date, "end": end_date, "days": days},
            "trend": trend_data,
        }

    except Exception as e:
        logging.error(f"SoS trend API error: {e}")
        return {"success": False, "error": str(e)}


@router.get("/api/sos/trend/competitors-avg")
async def get_competitors_avg_sos_trend(
    category_id: str | None = None,
    days: int = 7,
    start_date: str | None = None,
    end_date: str | None = None,
    top_n: int = 10,
    exclude_brand: str = "LANEIGE",
):
    """
    경쟁 브랜드 평균 SoS 추세 데이터 (일별)
    Top N 브랜드(LANEIGE 제외)의 평균 시장점유율 추이

    Args:
        category_id: 카테고리 (선택, 없으면 전체)
        days: 조회 기간 (기본: 7일)
        start_date: 시작 날짜 (YYYY-MM-DD)
        end_date: 종료 날짜 (YYYY-MM-DD)
        top_n: 상위 몇 개 브랜드 (기본: 10)
        exclude_brand: 제외할 브랜드 (기본: LANEIGE)

    Returns:
        경쟁 브랜드 평균 SoS 추세 데이터
    """
    try:
        sqlite = get_sqlite_storage()
        await sqlite.initialize()

        # 날짜 범위 결정
        if start_date and end_date:
            pass
        else:
            end_date = datetime.now().strftime("%Y-%m-%d")
            start_date = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")

        # 일별 전체 제품 수 쿼리
        if category_id:
            total_query = """
                SELECT snapshot_date, COUNT(*) as total_count
                FROM raw_data
                WHERE snapshot_date BETWEEN ? AND ?
                AND category_id = ?
                GROUP BY snapshot_date
                ORDER BY snapshot_date
            """
            total_params = (start_date, end_date, category_id)

            brand_daily_query = """
                SELECT snapshot_date, brand, COUNT(*) as brand_count
                FROM raw_data
                WHERE snapshot_date BETWEEN ? AND ?
                AND category_id = ?
                AND LOWER(brand) NOT LIKE ?
                AND brand IS NOT NULL
                AND brand != ''
                GROUP BY snapshot_date, brand
                ORDER BY snapshot_date, brand_count DESC
            """
            brand_daily_params = (start_date, end_date, category_id, f"%{exclude_brand.lower()}%")
        else:
            total_query = """
                SELECT snapshot_date, COUNT(*) as total_count
                FROM raw_data
                WHERE snapshot_date BETWEEN ? AND ?
                GROUP BY snapshot_date
                ORDER BY snapshot_date
            """
            total_params = (start_date, end_date)

            brand_daily_query = """
                SELECT snapshot_date, brand, COUNT(*) as brand_count
                FROM raw_data
                WHERE snapshot_date BETWEEN ? AND ?
                AND LOWER(brand) NOT LIKE ?
                AND brand IS NOT NULL
                AND brand != ''
                GROUP BY snapshot_date, brand
                ORDER BY snapshot_date, brand_count DESC
            """
            brand_daily_params = (start_date, end_date, f"%{exclude_brand.lower()}%")

        with sqlite.get_connection() as conn:
            # 전체 카운트
            cursor = conn.execute(total_query, total_params)
            total_rows = cursor.fetchall()
            total_by_date = {row[0]: row[1] for row in total_rows}

            # 일별/브랜드별 카운트
            cursor = conn.execute(brand_daily_query, brand_daily_params)
            brand_rows = cursor.fetchall()

        # 일별로 Top N 브랜드의 평균 SoS 계산
        daily_brands: dict[str, list[tuple[str, int]]] = defaultdict(list)
        for date, brand, count in brand_rows:
            daily_brands[date].append((brand, count))

        # 일별 경쟁 브랜드 평균 SoS 계산
        trend_data = []
        for date, total in sorted(total_by_date.items()):
            brands_for_date = daily_brands.get(date, [])
            top_brands = brands_for_date[:top_n]

            if top_brands and total > 0:
                sos_values = [(count / total * 100) for _, count in top_brands]
                avg_sos = sum(sos_values) / len(sos_values)
            else:
                avg_sos = 0

            trend_data.append(
                {
                    "date": date,
                    "total_products": total,
                    "top_brands_count": len(top_brands),
                    "avg_sos": round(avg_sos, 2),
                }
            )

        return {
            "success": True,
            "category_id": category_id,
            "excluded_brand": exclude_brand,
            "top_n": top_n,
            "period": {"start": start_date, "end": end_date, "days": days},
            "trend": trend_data,
        }

    except Exception as e:
        logging.error(f"Competitors avg SoS trend API error: {e}")
        return {"success": False, "error": str(e)}
