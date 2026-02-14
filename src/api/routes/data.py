"""
Data Routes
===========
Dashboard data endpoints
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException

from src.tools.storage.sheets_writer import SheetsWriter

logger = logging.getLogger(__name__)
# Data path
DATA_PATH = "./data/dashboard_data.json"

# Router
router = APIRouter()

# SheetsWriter singleton instance
_sheets_writer: SheetsWriter | None = None


def get_sheets_writer() -> SheetsWriter:
    """SheetsWriter singleton instance"""
    global _sheets_writer
    if _sheets_writer is None:
        _sheets_writer = SheetsWriter()
    return _sheets_writer


def load_dashboard_data() -> dict[str, Any]:
    """Load dashboard data"""
    try:
        with open(DATA_PATH, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


@router.get("/api/data")
async def get_data():
    """Dashboard data endpoint"""
    data = load_dashboard_data()
    if not data:
        raise HTTPException(status_code=404, detail="Dashboard data not found")
    return data


@router.get("/api/historical")
async def get_historical_data(
    start_date: str, end_date: str, category_id: str | None = None, brand: str | None = "LANEIGE"
):
    """
    Historical data endpoint (from Google Sheets)

    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        category_id: Category filter (optional)
        brand: Brand filter (default: LANEIGE)

    Returns:
        - data: Date-wise metric data
        - sos_history: SoS trend data
        - rank_history: Rank trend data
    """
    try:
        sheets_writer = get_sheets_writer()
        if not sheets_writer._initialized:
            await sheets_writer.initialize()

        # Calculate date range
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")
        days = (end_dt - start_dt).days + 1

        # Get historical data from Google Sheets
        records = await sheets_writer.get_rank_history(
            category_id=category_id, brand=brand, days=days
        )

        if not records:
            # If no data in Google Sheets, try local JSON files
            return await _get_historical_from_local(start_date, end_date, brand)

        # Aggregate data by date
        daily_data = {}
        for record in records:
            snapshot_date = record.get("snapshot_date", "")
            if not snapshot_date or snapshot_date < start_date or snapshot_date > end_date:
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
                    "rank": rank,
                    "price": record.get("price", ""),
                    "rating": record.get("rating", ""),
                }
            )
            daily_data[snapshot_date]["total_count"] += 1
            if rank <= 10:
                daily_data[snapshot_date]["top10_count"] += 1

        # Calculate SoS trend (based on Top 100)
        sos_history = []
        rank_history = []
        for date_str in sorted(daily_data.keys()):
            day_data = daily_data[date_str]
            products = day_data["products"]

            # SoS = (LANEIGE product count / 100) * 100
            sos = round(len(products) / 100 * 100, 1) if products else 0
            sos_history.append(
                {
                    "date": date_str,
                    "sos": sos,
                    "product_count": len(products),
                    "top10_count": day_data["top10_count"],
                }
            )

            # Average rank (if available)
            if products:
                avg_rank = round(sum(p["rank"] for p in products) / len(products), 1)
                rank_history.append(
                    {
                        "date": date_str,
                        "rank": avg_rank,
                        "best_rank": min(p["rank"] for p in products),
                        "worst_rank": max(p["rank"] for p in products),
                    }
                )

        # Calculate available_dates
        available_dates = sorted(daily_data.keys())

        # Calculate brand_metrics (aggregate all brands for entire period)
        brand_metrics = await _calculate_brand_metrics_for_period(records, daily_data, brand)

        return {
            "success": True,
            "available_dates": available_dates,
            "brand_metrics": brand_metrics,
            "data": {
                "sos_history": sos_history,
                "rank_history": rank_history,
                "daily_data": list(daily_data.values()),
                "period": {"start": start_date, "end": end_date, "days": days},
                "brand": brand,
            },
        }

    except Exception as e:
        logging.error(f"Historical data error: {e}")
        # Fallback: try local data
        return await _get_historical_from_local(start_date, end_date, brand)


async def _calculate_brand_metrics_for_period(
    records: list[dict], daily_data: dict, target_brand: str
) -> list[dict]:
    """
    Calculate metrics for all brands within period (for SoS × Avg Rank chart)

    Returns:
        Brand-wise SoS, average rank, product count, etc.
    """
    # Aggregate all product data (all brands)
    brand_data = {}

    for record in records:
        brand_name = record.get("brand", "Unknown")
        rank = int(record.get("rank", 0)) if record.get("rank") else 0

        if not brand_name or rank == 0:
            continue

        if brand_name not in brand_data:
            brand_data[brand_name] = {
                "brand": brand_name,
                "ranks": [],
                "prices": [],
                "product_count": 0,
            }

        brand_data[brand_name]["ranks"].append(rank)
        brand_data[brand_name]["product_count"] += 1

        # Collect prices (valid USD range only)
        price = record.get("price")
        if price is not None:
            try:
                price_val = float(price)
                if 0.5 <= price_val <= 500:
                    brand_data[brand_name]["prices"].append(price_val)
            except (ValueError, TypeError):
                logger.warning("Suppressed Exception", exc_info=True)

    # Total product count (all brands)
    total_products = sum(b["product_count"] for b in brand_data.values())

    # Calculate metrics
    brand_metrics = []
    for brand_name, data in brand_data.items():
        if not data["ranks"]:
            continue

        sos = round(data["product_count"] / max(total_products, 100) * 100, 2)
        avg_rank = round(sum(data["ranks"]) / len(data["ranks"]), 1)

        # Average price
        prices = data.get("prices", [])
        avg_price = round(sum(prices) / len(prices), 2) if prices else None

        # Bubble size: based on product count (min 5, max 25)
        bubble_size = max(5, min(25, data["product_count"] * 2))

        brand_metrics.append(
            {
                "brand": brand_name,
                "sos": sos,
                "avg_rank": avg_rank,
                "product_count": data["product_count"],
                "avg_price": avg_price,
                "bubble_size": bubble_size,
                "is_laneige": target_brand.upper() in brand_name.upper(),
            }
        )

    # Sort by SoS descending, return top 10 only
    brand_metrics.sort(key=lambda x: x["sos"], reverse=True)
    return brand_metrics[:10]


def _get_brand_metrics_from_dashboard(dashboard_data: dict | None, target_brand: str) -> list[dict]:
    """
    Extract brand metrics from dashboard data (for local fallback)
    """
    if not dashboard_data:
        return []

    # Use brand_matrix data from dashboard
    brand_matrix = dashboard_data.get("charts", {}).get("brand_matrix", [])
    if brand_matrix:
        return brand_matrix

    # Generate from competitor data
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
    Get historical data from local JSON files (fallback)

    Uses date-wise JSON files in data/ folder or historical data from dashboard_data.json
    """
    try:
        # Load main dashboard data
        data = load_dashboard_data()
        sos_history = []
        rank_history = []

        # 1. Extract current SoS/rank info from dashboard data
        if data:
            brand_kpis = data.get("brand", {}).get("kpis", {})
            current_sos = brand_kpis.get("sos", 0)
            data_date = data.get("metadata", {}).get(
                "data_date", datetime.now().strftime("%Y-%m-%d")
            )

            # Add if current date is within requested range
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
                    rank_history.append(
                        {
                            "date": data_date,
                            "rank": avg_rank,
                            "best_rank": brand_kpis.get("best_rank", avg_rank),
                            "worst_rank": brand_kpis.get("worst_rank", avg_rank),
                        }
                    )

        # 2. Extract data from latest_crawl_result.json
        latest_crawl_path = Path("./data/latest_crawl_result.json")
        if latest_crawl_path.exists():
            try:
                with open(latest_crawl_path, encoding="utf-8") as f:
                    crawl_data = json.load(f)

                # Find brand products across all categories
                brand_products = []
                crawl_date = None

                for _cat_id, cat_data in crawl_data.get("categories", {}).items():
                    for product in cat_data.get("products", []):
                        product_brand = product.get("brand", "")
                        product_name = product.get("product_name", "")

                        # Brand matching (case insensitive, partial match)
                        if (
                            brand.upper() in product_brand.upper()
                            or brand.upper() in product_name.upper()
                        ):
                            brand_products.append(product)
                            if not crawl_date:
                                crawl_date = product.get("snapshot_date")

                if brand_products and crawl_date and start_date <= crawl_date <= end_date:
                    # Check for duplicates
                    if not any(h["date"] == crawl_date for h in sos_history):
                        # Total products per category (based on Top 100)
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
                        rank_history.append(
                            {
                                "date": crawl_date,
                                "rank": avg_rank,
                                "best_rank": min(p.get("rank", 100) for p in brand_products),
                                "worst_rank": max(p.get("rank", 100) for p in brand_products),
                            }
                        )

            except (json.JSONDecodeError, ValueError) as e:
                logging.warning(f"Failed to parse latest_crawl_result.json: {e}")

        # 3. Search date-wise data in raw_products folder (existing logic)
        raw_data_dir = Path("./data/raw_products")
        if raw_data_dir.exists():
            for json_file in raw_data_dir.glob("*.json"):
                try:
                    file_date = json_file.stem  # Assume filename is YYYY-MM-DD format
                    if start_date <= file_date <= end_date:
                        with open(json_file, encoding="utf-8") as f:
                            daily_raw = json.load(f)

                        # Filter brand products only
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

                            # Remove duplicates
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
                                rank_history.append(
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

        # Sort by date
        sos_history.sort(key=lambda x: x["date"])
        rank_history.sort(key=lambda x: x["date"])

        # Calculate available_dates
        available_dates = [h["date"] for h in sos_history]

        # Calculate brand_metrics (from current dashboard data)
        brand_metrics = _get_brand_metrics_from_dashboard(data, brand)

        if not sos_history:
            return {
                "success": False,
                "error": "No historical data found for the specified period",
                "available_dates": [],
                "brand_metrics": [],
                "data": None,
            }

        return {
            "success": True,
            "available_dates": available_dates,
            "brand_metrics": brand_metrics,
            "data": {
                "sos_history": sos_history,
                "rank_history": rank_history,
                "period": {"start": start_date, "end": end_date},
                "brand": brand,
                "source": "local",
            },
        }

    except Exception as e:
        logging.error(f"Error loading historical data from local: {e}")
        return {
            "success": False,
            "error": str(e),
            "available_dates": [],
            "brand_metrics": [],
            "data": None,
        }
