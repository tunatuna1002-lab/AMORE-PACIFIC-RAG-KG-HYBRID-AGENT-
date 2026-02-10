"""
Competitor Comparison Routes
============================
경쟁사 추적 및 비교 엔드포인트
"""

import json
import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException

from src.api.dependencies import load_dashboard_data
from src.tools.storage.sqlite_storage import get_sqlite_storage

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Competitors"])


def _detect_product_type(product_name: str) -> str:
    """제품명에서 제품 타입 추출"""
    name_lower = product_name.lower()

    if "lip sleeping" in name_lower or "lip mask" in name_lower:
        return "lip_balm"
    elif "lip glowy" in name_lower or "lip butter" in name_lower or "lip balm" in name_lower:
        return "lip_balm"
    elif "water sleeping" in name_lower or "sleeping mask" in name_lower:
        return "sleeping_mask"
    elif "water bank" in name_lower or "cream" in name_lower or "moisturizer" in name_lower:
        return "moisturizer"
    elif "toner" in name_lower or "cream skin" in name_lower:
        return "toner"
    elif "serum" in name_lower:
        return "serum"
    else:
        return "other"


@router.get("/api/competitors")
async def get_competitor_data(brand: str | None = None):
    """
    경쟁사 추적 데이터 조회

    Args:
        brand: 브랜드 필터 (예: "Summer Fridays")

    Returns:
        경쟁사 제품 목록 및 LANEIGE 비교 데이터
    """
    try:
        result = {"competitors": {}, "laneige_products": {}, "comparison": []}

        # 1. SQLite에서 경쟁사 데이터 조회 시도
        try:
            sqlite = get_sqlite_storage()
            await sqlite.initialize()
            comp_products = await sqlite.get_competitor_products(brand=brand)

            if comp_products:
                # 브랜드별로 그룹화
                for p in comp_products:
                    brand_name = p.get("brand", "Unknown")
                    if brand_name not in result["competitors"]:
                        result["competitors"][brand_name] = {
                            "brand": brand_name,
                            "products": [],
                            "product_count": 0,
                            "avg_price": 0,
                            "avg_rating": 0,
                        }
                    result["competitors"][brand_name]["products"].append(p)

                # 브랜드별 집계
                for _brand_name, brand_data in result["competitors"].items():
                    products = brand_data["products"]
                    brand_data["product_count"] = len(products)
                    prices = [p["price"] for p in products if p.get("price")]
                    ratings = [p["rating"] for p in products if p.get("rating")]
                    brand_data["avg_price"] = round(sum(prices) / len(prices), 2) if prices else 0
                    brand_data["avg_rating"] = (
                        round(sum(ratings) / len(ratings), 1) if ratings else 0
                    )

        except Exception as sqlite_err:
            logging.warning(f"SQLite competitor query failed: {sqlite_err}")

        # 2. JSON 파일에서 폴백
        if not result["competitors"]:
            json_path = Path("./data/competitor_products.json")
            if json_path.exists():
                with open(json_path, encoding="utf-8") as f:
                    json_data = json.load(f)
                    for p in json_data.get("products", []):
                        brand_name = p.get("brand", "Unknown")
                        if brand and brand_name != brand:
                            continue
                        if brand_name not in result["competitors"]:
                            result["competitors"][brand_name] = {
                                "brand": brand_name,
                                "products": [],
                                "product_count": 0,
                            }
                        result["competitors"][brand_name]["products"].append(p)

        # 3. LANEIGE 제품 데이터 로드 (최신 크롤링 데이터에서)
        data = load_dashboard_data()
        if data:
            # 카테고리별 LANEIGE 제품 추출
            for cat_id, cat_data in data.get("category", {}).items():
                for product in cat_data.get("top_products", []):
                    if "laneige" in product.get("brand", "").lower():
                        product_type = _detect_product_type(product.get("product_name", ""))
                        if product_type not in result["laneige_products"]:
                            result["laneige_products"][product_type] = []
                        result["laneige_products"][product_type].append(
                            {**product, "category_id": cat_id, "product_type": product_type}
                        )

        # 4. 제품 타입별 비교 데이터 생성
        for brand_name, brand_data in result["competitors"].items():
            for comp_product in brand_data["products"]:
                laneige_match = comp_product.get("laneige_competitor")
                product_type = comp_product.get("product_type", "")

                comparison_item = {
                    "competitor_brand": brand_name,
                    "competitor_product": comp_product.get("product_name", ""),
                    "competitor_price": comp_product.get("price"),
                    "competitor_rating": comp_product.get("rating"),
                    "competitor_reviews": comp_product.get("reviews_count"),
                    "product_type": product_type,
                    "laneige_product": laneige_match,
                    "laneige_price": None,
                    "laneige_rating": None,
                    "laneige_reviews": None,
                    "price_diff": None,
                    "rating_diff": None,
                }

                # LANEIGE 매칭 제품 찾기
                if product_type in result["laneige_products"]:
                    for lp in result["laneige_products"][product_type]:
                        comparison_item["laneige_price"] = lp.get("price")
                        comparison_item["laneige_rating"] = lp.get("rating")
                        comparison_item["laneige_reviews"] = lp.get("reviews_count")

                        # 차이 계산
                        if comparison_item["competitor_price"] and comparison_item["laneige_price"]:
                            comparison_item["price_diff"] = round(
                                comparison_item["laneige_price"]
                                - comparison_item["competitor_price"],
                                2,
                            )
                        if (
                            comparison_item["competitor_rating"]
                            and comparison_item["laneige_rating"]
                        ):
                            comparison_item["rating_diff"] = round(
                                comparison_item["laneige_rating"]
                                - comparison_item["competitor_rating"],
                                1,
                            )
                        break

                result["comparison"].append(comparison_item)

        return result

    except Exception as e:
        logger.error(f"Competitor data error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail="경쟁사 데이터 조회 중 오류가 발생했습니다"
        ) from e


@router.get("/api/competitors/brands")
async def get_tracked_brands():
    """추적 중인 경쟁사 브랜드 목록"""
    try:
        config_path = Path("./config/tracked_competitors.json")
        if not config_path.exists():
            return {"brands": []}

        with open(config_path, encoding="utf-8") as f:
            config = json.load(f)

        brands = []
        for brand_name, brand_config in config.get("competitors", {}).items():
            brands.append(
                {
                    "name": brand_name,
                    "tier": brand_config.get("tier", ""),
                    "product_count": len(brand_config.get("products", [])),
                }
            )

        return {"brands": brands}

    except Exception as e:
        logging.error(f"Get tracked brands error: {e}")
        return {"brands": []}
