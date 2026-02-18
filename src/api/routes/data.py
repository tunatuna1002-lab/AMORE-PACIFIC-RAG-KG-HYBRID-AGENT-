"""
Data Routes
===========
Dashboard data and historical data endpoints (SQLite-first, Sheets/local fallback)
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request

from src.api.dependencies import get_sheets_writer, limiter, load_dashboard_data
from src.tools.storage.sqlite_storage import get_sqlite_storage

logger = logging.getLogger(__name__)

router = APIRouter(tags=["data"])


@router.get("/api/data")
@limiter.limit("30/minute")
async def get_data(request: Request):
    """대시보드 데이터 조회"""
    data = load_dashboard_data()
    if not data:
        raise HTTPException(status_code=404, detail="Dashboard data not found")
    return data


@router.get("/api/historical")
@limiter.limit("30/minute")
async def get_historical_data(
    request: Request,
    start_date: str,
    end_date: str,
    category_id: str | None = None,
    brand: str | None = "LANEIGE",
):
    """
    히스토리컬 데이터 조회 (SQLite 우선, Google Sheets fallback)

    Args:
        start_date: 시작 날짜 (YYYY-MM-DD)
        end_date: 종료 날짜 (YYYY-MM-DD)
        category_id: 카테고리 필터 (선택)
        brand: 브랜드 필터 (기본값: LANEIGE)

    Returns:
        - data: 날짜별 지표 데이터
        - sos_history: SoS 추이 데이터
        - raw_data: 순위 추이 데이터
    """
    try:
        records = []
        data_source = None

        # 1차: SQLite에서 조회 (빠름)
        try:
            sqlite = get_sqlite_storage()
            await sqlite.initialize()
            records = await sqlite.get_raw_data(
                start_date=start_date,
                end_date=end_date,
                category_id=category_id,
                limit=50000,
            )
            if records:
                data_source = "sqlite"
                logging.info(
                    f"Historical: loaded {len(records)} records from SQLite ({start_date} ~ {end_date})"
                )
        except Exception as sqlite_err:
            logging.warning(f"Historical: SQLite 조회 실패: {sqlite_err}")

        # 2차: SQLite 실패/빈 결과 시 Google Sheets fallback
        if not records:
            try:
                sheets_writer = get_sheets_writer()
                if not sheets_writer._initialized:
                    await sheets_writer.initialize()
                records = await sheets_writer.get_raw_data(
                    start_date=start_date, end_date=end_date, category_id=category_id
                )
                if records:
                    data_source = "sheets"
                    logging.info(
                        f"Historical: loaded {len(records)} records from Sheets ({start_date} ~ {end_date})"
                    )
            except Exception as sheets_err:
                logging.warning(f"Historical: Google Sheets 조회 실패: {sheets_err}")

        if not records:
            return await _get_historical_from_local(start_date, end_date, brand)

        # 날짜 범위 계산
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        days = (end_dt - start_dt).days + 1

        # 날짜별 데이터 집계 (특정 브랜드 필터링)
        daily_data = {}
        brand_lower = brand.lower() if brand else ""
        for record in records:
            snapshot_date = record.get("snapshot_date", "")
            if not snapshot_date or snapshot_date < start_date or snapshot_date > end_date:
                continue

            record_brand = record.get("brand", "")
            if brand_lower and record_brand.lower() != brand_lower:
                continue

            if snapshot_date not in daily_data:
                daily_data[snapshot_date] = {
                    "date": snapshot_date,
                    "products": [],
                    "total_count": 0,
                    "top10_count": 0,
                }

            rank = int(record.get("rank", 0)) if record.get("rank") else 0
            daily_data[snapshot_date]["products"].append(
                {
                    "asin": record.get("asin", ""),
                    "product_name": record.get("product_name", ""),
                    "brand": record_brand,
                    "rank": rank,
                    "price": record.get("price", ""),
                    "rating": record.get("rating", ""),
                }
            )
            daily_data[snapshot_date]["total_count"] += 1
            if rank <= 10:
                daily_data[snapshot_date]["top10_count"] += 1

        # SoS 추이 계산
        sos_history = []
        raw_data = []
        for date_str in sorted(daily_data.keys()):
            day_data = daily_data[date_str]
            products = day_data["products"]

            sos = round(len(products) / 100 * 100, 1) if products else 0
            sos_history.append(
                {
                    "date": date_str,
                    "sos": sos,
                    "product_count": len(products),
                    "top10_count": day_data["top10_count"],
                }
            )

            if products:
                avg_rank = round(sum(p["rank"] for p in products) / len(products), 1)
                raw_data.append(
                    {
                        "date": date_str,
                        "rank": avg_rank,
                        "best_rank": min(p["rank"] for p in products),
                        "worst_rank": max(p["rank"] for p in products),
                    }
                )

        available_dates = sorted(daily_data.keys())

        # brand_metrics 계산 (전체 기간 통합 - 모든 브랜드 포함)
        brand_metrics = await _calculate_brand_metrics_for_period(records, daily_data, brand)

        # rank_history 생성 (Product View 차트용)
        rank_history = {}
        for record in records:
            snapshot_date = record.get("snapshot_date", "")
            if not snapshot_date or snapshot_date < start_date or snapshot_date > end_date:
                continue

            if snapshot_date not in rank_history:
                rank_history[snapshot_date] = {"products": []}

            rank = int(record.get("rank", 0)) if record.get("rank") else 0
            price_val = record.get("price", 0)
            try:
                price = float(str(price_val).replace("$", "").replace(",", "")) if price_val else 0
            except (ValueError, TypeError):
                price = 0

            rank_history[snapshot_date]["products"].append(
                {
                    "name": record.get("product_name", ""),
                    "product_name": record.get("product_name", ""),
                    "brand": record.get("brand", ""),
                    "asin": record.get("asin", ""),
                    "rank": rank,
                    "price": price,
                    "rating": record.get("rating", ""),
                    "discount_percent": record.get("discount_percent", 0),
                }
            )

        # 전체 데이터의 사용 가능한 날짜 범위 조회
        available_date_range = {"min": None, "max": None}
        try:
            sqlite = get_sqlite_storage()
            stats = sqlite.get_stats()
            if "date_range" in stats:
                available_date_range = stats["date_range"]
        except Exception:
            pass

        return {
            "success": True,
            "available_dates": available_dates,
            "available_date_range": available_date_range,
            "data_source": data_source,
            "brand_metrics": brand_metrics,
            "rank_history": rank_history,
            "data": {
                "sos_history": sos_history,
                "raw_data": raw_data,
                "daily_data": list(daily_data.values()),
                "period": {"start": start_date, "end": end_date, "days": days},
                "brand": brand,
            },
        }

    except Exception as e:
        logging.error(f"Historical data error: {e}")
        return await _get_historical_from_local(start_date, end_date, brand)


# ============= Helper Functions =============


async def _calculate_brand_metrics_for_period(
    records: list[dict], daily_data: dict, target_brand: str
) -> list[dict]:
    """
    기간 내 모든 브랜드의 메트릭 계산 (SoS x Avg Rank 차트용)

    Note:
        기간 조회 시 동일 ASIN이 여러 날짜에 중복 등장하므로,
        ASIN 기준 유니크 카운트를 적용하여 정확한 제품 수 계산
    """
    brand_data = {}
    brand_unique_asins: dict[str, set] = {}

    for record in records:
        brand_name = record.get("brand", "Unknown")
        asin = record.get("asin", "")
        rank = int(record.get("rank", 0)) if record.get("rank") else 0

        if not brand_name or brand_name.lower() == "unknown" or rank == 0:
            continue

        if brand_name not in brand_data:
            brand_data[brand_name] = {
                "brand": brand_name,
                "ranks": [],
                "prices": [],
                "product_count": 0,
            }
            brand_unique_asins[brand_name] = set()

        brand_data[brand_name]["ranks"].append(rank)

        price = record.get("price")
        if price is not None:
            try:
                price_val = float(price)
                if 0.5 <= price_val <= 500:
                    brand_data[brand_name]["prices"].append(price_val)
            except (ValueError, TypeError):
                pass

        if asin and asin not in brand_unique_asins[brand_name]:
            brand_unique_asins[brand_name].add(asin)
            brand_data[brand_name]["product_count"] += 1
        elif not asin:
            brand_data[brand_name]["product_count"] += 1

    total_products = sum(b["product_count"] for b in brand_data.values())

    brand_metrics = []
    for brand_name, data in brand_data.items():
        if not data["ranks"]:
            continue

        sos = round(data["product_count"] / max(total_products, 100) * 100, 2)
        avg_rank = round(sum(data["ranks"]) / len(data["ranks"]), 1)

        prices = data.get("prices", [])
        avg_price = round(sum(prices) / len(prices), 2) if prices else None

        bubble_size = max(5, min(25, data["product_count"] * 2))
        is_laneige = target_brand.upper() in brand_name.upper()

        brand_metrics.append(
            {
                "brand": brand_name,
                "sos": sos,
                "avg_rank": avg_rank,
                "product_count": data["product_count"],
                "avg_price": avg_price,
                "bubble_size": bubble_size,
                "is_laneige": is_laneige,
            }
        )

    brand_metrics.sort(key=lambda x: x["sos"], reverse=True)
    top_10 = brand_metrics[:10]

    # LANEIGE가 top_10에 없으면 추가
    laneige_in_top10 = any(b.get("is_laneige") for b in top_10)
    if not laneige_in_top10 and target_brand:
        laneige_data = None
        for key in [
            target_brand,
            target_brand.upper(),
            target_brand.lower(),
            target_brand.capitalize(),
        ]:
            if key in brand_data:
                laneige_data = brand_data[key]
                break

        if laneige_data and laneige_data["ranks"]:
            sos = round(laneige_data["product_count"] / max(total_products, 100) * 100, 2)
            avg_rank = round(sum(laneige_data["ranks"]) / len(laneige_data["ranks"]), 1)
            l_prices = laneige_data.get("prices", [])
            l_avg_price = round(sum(l_prices) / len(l_prices), 2) if l_prices else None
            bubble_size = max(5, min(25, laneige_data["product_count"] * 2))
            top_10.append(
                {
                    "brand": target_brand,
                    "sos": sos,
                    "avg_rank": avg_rank,
                    "product_count": laneige_data["product_count"],
                    "avg_price": l_avg_price,
                    "bubble_size": bubble_size,
                    "is_laneige": True,
                }
            )
            top_10.sort(key=lambda x: x["sos"], reverse=True)

    # Summer Fridays 특별 처리 (tracked competitor)
    TRACKED_COMPETITORS = ["Summer Fridays"]
    for tracked_brand in TRACKED_COMPETITORS:
        tracked_in_top = any(b.get("brand") == tracked_brand for b in top_10)
        if not tracked_in_top and tracked_brand in brand_data:
            tracked_data = brand_data[tracked_brand]
            if tracked_data["ranks"]:
                sos = round(tracked_data["product_count"] / max(total_products, 100) * 100, 2)
                avg_rank = round(sum(tracked_data["ranks"]) / len(tracked_data["ranks"]), 1)
                t_prices = tracked_data.get("prices", [])
                t_avg_price = round(sum(t_prices) / len(t_prices), 2) if t_prices else None
                bubble_size = max(5, min(25, tracked_data["product_count"] * 2))
                top_10.append(
                    {
                        "brand": tracked_brand,
                        "sos": sos,
                        "avg_rank": avg_rank,
                        "product_count": tracked_data["product_count"],
                        "avg_price": t_avg_price,
                        "bubble_size": bubble_size,
                        "is_laneige": False,
                        "is_tracked": True,
                    }
                )
        elif not tracked_in_top:
            top_10.append(
                {
                    "brand": tracked_brand,
                    "sos": 0,
                    "avg_rank": None,
                    "product_count": 0,
                    "bubble_size": 5,
                    "is_laneige": False,
                    "is_tracked": True,
                    "no_data": True,
                }
            )

    top_10.sort(key=lambda x: (not x.get("is_tracked", False), x["sos"]), reverse=True)
    return top_10


def _get_brand_metrics_from_dashboard(dashboard_data: dict | None, target_brand: str) -> list[dict]:
    """대시보드 데이터에서 브랜드 메트릭 추출 (로컬 폴백용)"""
    if not dashboard_data:
        return []

    brand_matrix = dashboard_data.get("charts", {}).get("brand_matrix", [])
    if brand_matrix:
        return brand_matrix

    competitors = dashboard_data.get("brand", {}).get("competitors", [])
    if not competitors:
        return []

    brand_metrics = []
    for comp in competitors:
        brand_metrics.append(
            {
                "brand": comp.get("brand", "Unknown"),
                "sos": comp.get("sos", 0),
                "avg_rank": comp.get("avg_rank", 50),
                "product_count": comp.get("product_count", 0),
                "bubble_size": max(5, min(25, comp.get("product_count", 0) * 2)),
                "is_laneige": target_brand.upper() in comp.get("brand", "").upper(),
            }
        )

    return brand_metrics


async def _get_historical_from_local(
    start_date: str, end_date: str, brand: str = "LANEIGE"
) -> dict[str, Any]:
    """
    로컬 JSON 파일에서 히스토리컬 데이터 조회 (폴백)
    """
    try:
        data = load_dashboard_data()
        sos_history = []
        raw_data = []

        # 1. 대시보드 데이터에서 현재 SoS/순위 정보 추출
        if data:
            brand_kpis = data.get("brand", {}).get("kpis", {})
            current_sos = brand_kpis.get("sos", 0)
            data_date = data.get("metadata", {}).get(
                "data_date", datetime.now().strftime("%Y-%m-%d")
            )

            if start_date <= data_date <= end_date:
                sos_history.append(
                    {
                        "date": data_date,
                        "sos": current_sos,
                        "product_count": brand_kpis.get("product_count", 0),
                        "top10_count": brand_kpis.get("top10_count", 0),
                    }
                )

                avg_rank = brand_kpis.get("avg_rank", 0)
                if avg_rank:
                    raw_data.append(
                        {
                            "date": data_date,
                            "rank": avg_rank,
                            "best_rank": brand_kpis.get("best_rank", avg_rank),
                            "worst_rank": brand_kpis.get("worst_rank", avg_rank),
                        }
                    )

        # 2. latest_crawl_result.json에서 데이터 추출
        latest_crawl_path = Path("./data/latest_crawl_result.json")
        if latest_crawl_path.exists():
            try:
                with open(latest_crawl_path, encoding="utf-8") as f:
                    crawl_data = json.load(f)

                brand_products = []
                crawl_date = None

                for _cat_id, cat_data in crawl_data.get("categories", {}).items():
                    for product in cat_data.get("products", []):
                        product_brand = product.get("brand", "")
                        product_name = product.get("product_name", "")

                        if (
                            brand.upper() in product_brand.upper()
                            or brand.upper() in product_name.upper()
                        ):
                            brand_products.append(product)
                            if not crawl_date:
                                crawl_date = product.get("snapshot_date")

                if brand_products and crawl_date and start_date <= crawl_date <= end_date:
                    if not any(h["date"] == crawl_date for h in sos_history):
                        total_products = sum(
                            len(cat.get("products", []))
                            for cat in crawl_data.get("categories", {}).values()
                        )

                        sos = round(len(brand_products) / max(total_products, 100) * 100, 2)
                        avg_rank = round(
                            sum(p.get("rank", 0) for p in brand_products) / len(brand_products), 1
                        )

                        sos_history.append(
                            {
                                "date": crawl_date,
                                "sos": sos,
                                "product_count": len(brand_products),
                                "top10_count": sum(
                                    1 for p in brand_products if p.get("rank", 100) <= 10
                                ),
                            }
                        )
                        raw_data.append(
                            {
                                "date": crawl_date,
                                "rank": avg_rank,
                                "best_rank": min(p.get("rank", 100) for p in brand_products),
                                "worst_rank": max(p.get("rank", 100) for p in brand_products),
                            }
                        )

            except (json.JSONDecodeError, ValueError) as e:
                logging.warning(f"Failed to parse latest_crawl_result.json: {e}")

        # 3. raw_products 폴더에서 날짜별 데이터 검색
        raw_data_dir = Path("./data/raw_products")
        if raw_data_dir.exists():
            for json_file in raw_data_dir.glob("*.json"):
                try:
                    file_date = json_file.stem
                    if start_date <= file_date <= end_date:
                        with open(json_file, encoding="utf-8") as f:
                            daily_raw = json.load(f)

                        brand_products = [
                            p
                            for p in daily_raw
                            if brand.upper() in p.get("brand", "").upper()
                            or brand.upper() in p.get("product_name", "").upper()
                        ]

                        if brand_products:
                            sos = round(len(brand_products) / 100 * 100, 1)
                            avg_rank = round(
                                sum(p.get("rank", 0) for p in brand_products) / len(brand_products),
                                1,
                            )

                            if not any(h["date"] == file_date for h in sos_history):
                                sos_history.append(
                                    {
                                        "date": file_date,
                                        "sos": sos,
                                        "product_count": len(brand_products),
                                        "top10_count": sum(
                                            1 for p in brand_products if p.get("rank", 100) <= 10
                                        ),
                                    }
                                )
                                raw_data.append(
                                    {
                                        "date": file_date,
                                        "rank": avg_rank,
                                        "best_rank": min(
                                            p.get("rank", 100) for p in brand_products
                                        ),
                                        "worst_rank": max(
                                            p.get("rank", 100) for p in brand_products
                                        ),
                                    }
                                )
                except (json.JSONDecodeError, ValueError):
                    continue

        sos_history.sort(key=lambda x: x["date"])
        raw_data.sort(key=lambda x: x["date"])

        available_dates = [h["date"] for h in sos_history]
        brand_metrics = _get_brand_metrics_from_dashboard(data, brand)

        # rank_history 생성 (CPI 차트용)
        rank_history = {}
        latest_crawl_path = Path("./data/latest_crawl_result.json")
        if latest_crawl_path.exists():
            try:
                with open(latest_crawl_path, encoding="utf-8") as f:
                    crawl_data = json.load(f)
                for _cat_id, cat_data in crawl_data.get("categories", {}).items():
                    for product in cat_data.get("products", []):
                        snap_date = product.get("snapshot_date", "")
                        if not snap_date or snap_date < start_date or snap_date > end_date:
                            continue
                        if snap_date not in rank_history:
                            rank_history[snap_date] = {"products": []}
                        price_val = product.get("price", 0)
                        try:
                            price = (
                                float(str(price_val).replace("$", "").replace(",", ""))
                                if price_val
                                else 0
                            )
                        except (ValueError, TypeError):
                            price = 0
                        rank_history[snap_date]["products"].append(
                            {
                                "name": product.get("product_name", ""),
                                "brand": product.get("brand", ""),
                                "rank": product.get("rank", 0),
                                "price": price,
                            }
                        )
            except (json.JSONDecodeError, ValueError) as e:
                logging.warning(f"Failed to build rank_history from local: {e}")

        if not sos_history:
            return {
                "success": False,
                "error": "No historical data found for the specified period",
                "available_dates": [],
                "brand_metrics": [],
                "rank_history": rank_history,
                "data": None,
            }

        return {
            "success": True,
            "available_dates": available_dates,
            "brand_metrics": brand_metrics,
            "rank_history": rank_history,
            "data": {
                "sos_history": sos_history,
                "raw_data": raw_data,
                "period": {"start": start_date, "end": end_date},
                "brand": brand,
                "source": "local",
            },
        }

    except Exception as e:
        logging.error(f"Local historical data error: {e}")
        return {
            "success": False,
            "error": str(e),
            "available_dates": [],
            "brand_metrics": [],
            "data": None,
        }
