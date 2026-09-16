"""
Brand Matrix
============
The "SoS x average rank" bubble-chart rows the dashboard's period view draws
(``GET /api/historical`` -> ``brand_metrics``), from raw snapshot rows or, when
there are none, from the cached dashboard JSON.

Pure functions: no I/O, no framework.
"""

from __future__ import annotations

from typing import Any

from src.domain.brand import is_target_brand

# 항상 브랜드 매트릭스에 표시하는 경쟁사 (데이터가 없어도 자리를 남긴다)
TRACKED_COMPETITORS = ["Summer Fridays"]


def brand_metrics_for_period(records: list[dict], target_brand: str) -> list[dict]:
    """
    기간 내 모든 브랜드의 메트릭 계산 (SoS x Avg Rank 차트용)

    Note:
        기간 조회 시 동일 ASIN이 여러 날짜에 중복 등장하므로,
        ASIN 기준 유니크 카운트를 적용하여 정확한 제품 수 계산
    """
    brand_data: dict[str, dict[str, Any]] = {}
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
        is_laneige = is_target_brand(brand_name, target_brand)

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

    # 타겟 브랜드가 top_10에 없으면 추가
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

    # 추적 경쟁사 특별 처리 (tracked competitor)
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


def brand_metrics_from_dashboard(dashboard_data: dict | None, target_brand: str) -> list[dict]:
    """대시보드 데이터에서 브랜드 메트릭 추출 (로컬 폴백용)"""
    if not dashboard_data:
        return []

    brand_matrix = dashboard_data.get("charts", {}).get("brand_matrix", [])
    if brand_matrix:
        return brand_matrix

    competitors = dashboard_data.get("brand", {}).get("competitors", [])
    if not competitors:
        return []

    return [
        {
            "brand": comp.get("brand", "Unknown"),
            "sos": comp.get("sos", 0),
            "avg_rank": comp.get("avg_rank", 50),
            "product_count": comp.get("product_count", 0),
            "bubble_size": max(5, min(25, comp.get("product_count", 0) * 2)),
            "is_laneige": is_target_brand(comp.get("brand"), target_brand),
        }
        for comp in competitors
    ]
