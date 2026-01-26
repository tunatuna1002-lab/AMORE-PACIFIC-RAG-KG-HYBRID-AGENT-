"""
Apify Amazon Scraper (Hybrid)
==============================
Apify Primary + Playwright Fallback 하이브리드 스크래퍼

## 아키텍처
```
┌───────────────────────────────────────────────────────┐
│                 Apify Amazon Scraper                  │
├───────────────────────────────────────────────────────┤
│  1. Apify API 시도 (junglee/amazon-bestsellers)       │
│     ↓ 실패 시                                         │
│  2. 브랜드 인식률 검증 (>=90%)                        │
│     ↓ 미달 시                                         │
│  3. Playwright Fallback (기존 AmazonScraper)          │
└───────────────────────────────────────────────────────┘
```

## 장점
- 8배 빠름 (80분 → 10분)
- 차단 리스크 최소화
- 무료 크레딧으로 운영 ($5/월)
- 기존 브랜드 인식 로직 재사용

## 사용 예
```python
scraper = ApifyAmazonScraper()
result = await scraper.scrape_category("lip_care", url)
```
"""

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

# Apify 클라이언트
try:
    from apify_client import ApifyClient
    APIFY_AVAILABLE = True
except ImportError:
    APIFY_AVAILABLE = False
    logger.warning("apify-client not installed")

# 한국 시간대 (UTC+9)
KST = timezone(timedelta(hours=9))


class ApifyAmazonScraper:
    """
    Apify 기반 Amazon 베스트셀러 스크래퍼 (하이브리드)

    Apify를 주로 사용하고, 실패 시 기존 Playwright 스크래퍼로 폴백합니다.
    브랜드 인식률이 90% 미만이면 폴백합니다.
    """

    # Apify Actor ID
    ACTOR_ID = "junglee/amazon-bestsellers"

    # 브랜드 인식률 임계값
    BRAND_RECOGNITION_THRESHOLD = 0.90  # 90%

    # 기존 AmazonScraper의 브랜드 목록 재사용
    MULTI_WORD_BRANDS = [
        # K-Beauty
        "Beauty of Joseon", "SKIN1004", "Round Lab", "Some By Mi",
        "Dear Klairs", "I'm From", "By Wishtrend", "Pyunkang Yul",
        # Premium Skincare
        "Sunday Riley", "Drunk Elephant", "Tatcha", "Fresh",
        "Paula's Choice", "The Ordinary", "La Roche-Posay", "Youth To The People",
        "First Aid Beauty", "Summer Fridays", "Rare Beauty", "Tower 28",
        "Glow Recipe", "Supergoop",
        # Makeup
        "Charlotte Tilbury", "Too Faced", "Urban Decay",
        "Fenty Beauty", "Huda Beauty", "Anastasia Beverly Hills",
        "Benefit Cosmetics", "MAC Cosmetics", "IT Cosmetics",
        "Kylie Cosmetics",
        # Drugstore
        "CeraVe", "La Roche Posay", "Neutrogena", "Cetaphil",
        # Hair
        "Mielle Organics", "Olaplex", "Moroccanoil",
    ]

    SINGLE_WORD_BRANDS = [
        # K-Beauty
        "LANEIGE", "Laneige", "COSRX", "TIRTIR", "Anua", "ANUA",
        "BIODANCE", "Innisfree", "MISSHA", "ETUDE", "SKINFOOD",
        "Benton", "Purito", "Klairs", "Heimish", "Isntree",
        "Rovectin", "Torriden", "mixsoon", "Numbuzin", "Haruharu",
        "Neogen", "Mediheal", "Banila", "Holika", "Peripera",
        "Romand", "Espoir", "Clio", "Moonshot", "Hera", "Sulwhasoo",
        "MEDICUBE", "medicube", "Abib",
        # US Brands
        "CeraVe", "Neutrogena", "Cetaphil", "Aveeno", "Olay",
        "Garnier", "Nivea", "Vaseline", "Dove",
        "NYX", "Maybelline", "Revlon", "Covergirl", "Milani",
        "ColourPop", "Morphe", "Tarte", "Smashbox", "Hourglass",
        "Glossier", "Nars", "Stila",
        "elf", "e.l.f.",
    ]

    def __init__(self, config_path: str = "./config/thresholds.json"):
        """
        Args:
            config_path: 설정 파일 경로
        """
        self._client: Optional[ApifyClient] = None
        self._playwright_scraper = None  # Lazy load
        self._enabled = os.getenv("ENABLE_APIFY_SCRAPER", "false").lower() == "true"
        self._api_token = os.getenv("APIFY_API_TOKEN")
        self.config_path = config_path

        # 데이터 저장 경로
        self.data_dir = Path("data/market_intelligence/amazon_apify")
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # 통계
        self.stats = {
            "apify_success": 0,
            "apify_fallback": 0,
            "playwright_fallback": 0,
            "brand_recognition_rate": 0.0
        }

    def _get_client(self) -> Optional[ApifyClient]:
        """Apify 클라이언트 반환 (lazy initialization)"""
        if not APIFY_AVAILABLE:
            logger.error("apify-client not available")
            return None

        if not self._api_token:
            logger.warning("APIFY_API_TOKEN not set, will use Playwright fallback")
            return None

        if self._client is None:
            self._client = ApifyClient(self._api_token)

        return self._client

    async def _get_playwright_scraper(self):
        """Playwright 스크래퍼 반환 (lazy initialization)"""
        if self._playwright_scraper is None:
            from src.tools.amazon_scraper import AmazonScraper
            self._playwright_scraper = AmazonScraper(self.config_path)
            await self._playwright_scraper.initialize()
        return self._playwright_scraper

    def _extract_brand(self, product_name: str) -> str:
        """
        제품명에서 브랜드 추출 (기존 AmazonScraper 로직 재사용)
        """
        if not product_name:
            return "Unknown"

        # Multi-word 브랜드 먼저 체크
        for brand in self.MULTI_WORD_BRANDS:
            if brand.lower() in product_name.lower():
                return brand

        # Single-word 브랜드
        for brand in self.SINGLE_WORD_BRANDS:
            if brand.lower() in product_name.lower():
                return brand

        return "Unknown"

    def _enrich_brand_info(self, products: List[Dict]) -> List[Dict]:
        """
        Apify 결과에 브랜드 정보 추가

        Apify는 brand 필드를 반환하지 않을 수 있으므로,
        기존 브랜드 인식 로직으로 보강합니다.
        """
        enriched = []
        recognized_count = 0

        for product in products:
            title = product.get("title", "") or product.get("name", "")

            # 기존 brand가 있으면 사용, 없으면 추출
            brand = product.get("brand")
            if not brand or brand == "Unknown":
                brand = self._extract_brand(title)

            if brand != "Unknown":
                recognized_count += 1

            enriched_product = {
                **product,
                "brand": brand,
                "brand_recognized": brand != "Unknown"
            }
            enriched.append(enriched_product)

        # 브랜드 인식률 계산
        recognition_rate = recognized_count / len(products) if products else 0
        self.stats["brand_recognition_rate"] = recognition_rate

        return enriched

    async def _scrape_via_apify(self, category_url: str, max_results: int = 100) -> Optional[List[Dict]]:
        """
        Apify를 통한 스크래핑

        Args:
            category_url: Amazon 베스트셀러 URL
            max_results: 최대 결과 수

        Returns:
            제품 리스트 또는 None (실패 시)
        """
        client = self._get_client()
        if client is None:
            return None

        try:
            logger.info(f"Starting Apify scrape for: {category_url}")

            run_input = {
                "categoryUrls": [{"url": category_url}],
                "maxItemsPerCategory": max_results,
                "proxy": {
                    "useApifyProxy": True,
                    "apifyProxyGroups": ["RESIDENTIAL"]
                }
            }

            # 동기 API를 비동기로 래핑
            def _run_actor():
                run = client.actor(self.ACTOR_ID).call(run_input=run_input)
                return list(client.dataset(run["defaultDatasetId"]).iterate_items())

            items = await asyncio.get_event_loop().run_in_executor(None, _run_actor)

            if items:
                logger.info(f"Apify returned {len(items)} products")
                self.stats["apify_success"] += 1
                return items
            else:
                logger.warning("Apify returned no results")
                return None

        except Exception as e:
            logger.error(f"Apify scraping failed: {e}")
            return None

    async def _scrape_via_playwright(self, category_id: str, category_url: str) -> Dict[str, Any]:
        """
        Playwright를 통한 폴백 스크래핑
        """
        logger.info(f"Falling back to Playwright for: {category_id}")
        self.stats["playwright_fallback"] += 1

        scraper = await self._get_playwright_scraper()
        return await scraper.scrape_category(category_id, category_url)

    async def scrape_category(
        self,
        category_id: str,
        category_url: str,
        force_apify: bool = False,
        force_playwright: bool = False
    ) -> Dict[str, Any]:
        """
        카테고리 베스트셀러 스크래핑 (하이브리드)

        Args:
            category_id: 카테고리 ID (예: "lip_care")
            category_url: Amazon 베스트셀러 URL
            force_apify: Apify만 사용 (테스트용)
            force_playwright: Playwright만 사용 (폴백 강제)

        Returns:
            {
                "products": [...],
                "count": int,
                "source": "apify" | "playwright",
                "brand_recognition_rate": float,
                "scraped_at": str
            }
        """
        now = datetime.now(KST).isoformat()

        # Feature flag 확인
        if not self._enabled and not force_apify:
            logger.info("Apify scraper disabled, using Playwright")
            result = await self._scrape_via_playwright(category_id, category_url)
            result["source"] = "playwright"
            return result

        # Playwright 강제 사용
        if force_playwright:
            result = await self._scrape_via_playwright(category_id, category_url)
            result["source"] = "playwright"
            return result

        # 1. Apify 시도
        apify_products = await self._scrape_via_apify(category_url)

        if apify_products is None:
            # Apify 실패 → Playwright 폴백
            logger.warning("Apify failed, falling back to Playwright")
            self.stats["apify_fallback"] += 1
            result = await self._scrape_via_playwright(category_id, category_url)
            result["source"] = "playwright"
            return result

        # 2. 브랜드 정보 보강
        enriched_products = self._enrich_brand_info(apify_products)

        # 3. 브랜드 인식률 검증
        recognition_rate = self.stats["brand_recognition_rate"]

        if recognition_rate < self.BRAND_RECOGNITION_THRESHOLD:
            logger.warning(
                f"Brand recognition rate ({recognition_rate:.1%}) below threshold "
                f"({self.BRAND_RECOGNITION_THRESHOLD:.0%}), falling back to Playwright"
            )
            self.stats["apify_fallback"] += 1
            result = await self._scrape_via_playwright(category_id, category_url)
            result["source"] = "playwright"
            result["apify_recognition_rate"] = recognition_rate
            return result

        # 4. 성공 - Apify 결과 반환
        logger.info(f"Apify scrape successful with {recognition_rate:.1%} brand recognition")

        # RankRecord 형식으로 변환
        products = []
        for i, item in enumerate(enriched_products, 1):
            product = {
                "asin": item.get("asin", ""),
                "product_name": item.get("title", "") or item.get("name", ""),
                "brand": item.get("brand", "Unknown"),
                "rank": i,
                "price": self._parse_price(item.get("price")),
                "rating": item.get("rating"),
                "reviews_count": item.get("reviewsCount") or item.get("numberOfReviews"),
                "url": item.get("url", ""),
                "category_id": category_id,
                "scraped_at": now,
                "source": "apify"
            }
            products.append(product)

        return {
            "products": products,
            "count": len(products),
            "source": "apify",
            "brand_recognition_rate": recognition_rate,
            "scraped_at": now
        }

    def _parse_price(self, price_value: Any) -> Optional[float]:
        """가격 파싱"""
        if price_value is None:
            return None

        if isinstance(price_value, (int, float)):
            return float(price_value)

        if isinstance(price_value, str):
            import re
            match = re.search(r'[\d,]+\.?\d*', price_value.replace(',', ''))
            if match:
                try:
                    return float(match.group())
                except ValueError:
                    pass

        return None

    async def scrape_all_categories(self) -> Dict[str, Any]:
        """
        모든 카테고리 스크래핑

        Returns:
            {
                "categories": {
                    "lip_care": {...},
                    "skin_care": {...},
                    ...
                },
                "stats": {...},
                "scraped_at": str
            }
        """
        # config에서 카테고리 URL 로드
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
        except FileNotFoundError:
            logger.error(f"Config file not found: {self.config_path}")
            return {"error": "Config not found"}

        categories = config.get("categories", {})
        results = {}

        for category_id, category_info in categories.items():
            url = category_info.get("url")
            if url:
                logger.info(f"Scraping category: {category_id}")
                result = await self.scrape_category(category_id, url)
                results[category_id] = result

                # 카테고리 간 딜레이
                await asyncio.sleep(5)

        return {
            "categories": results,
            "stats": self.stats,
            "scraped_at": datetime.now(KST).isoformat()
        }

    async def save_results(self, results: Dict[str, Any], filename: Optional[str] = None) -> Path:
        """결과 저장"""
        if filename is None:
            date_str = datetime.now(KST).strftime("%Y-%m-%d")
            filename = f"amazon_apify_{date_str}.json"

        filepath = self.data_dir / filename

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)

        logger.info(f"Saved results to {filepath}")
        return filepath

    async def close(self):
        """리소스 정리"""
        if self._playwright_scraper:
            await self._playwright_scraper.close()


# 테스트용 메인
if __name__ == "__main__":
    async def main():
        scraper = ApifyAmazonScraper()

        # 테스트 URL (Lip Care)
        test_url = "https://www.amazon.com/Best-Sellers-Lip-Care/zgbs/beauty/11060711"

        print("Testing Apify Amazon Scraper...")
        result = await scraper.scrape_category("lip_care", test_url)

        print(f"\nSource: {result.get('source')}")
        print(f"Products: {result.get('count')}")
        print(f"Brand Recognition: {result.get('brand_recognition_rate', 0):.1%}")

        # 상위 5개 제품
        for product in result.get("products", [])[:5]:
            print(f"\n  #{product['rank']} {product['brand']}: {product['product_name'][:50]}...")

        print(f"\nStats: {scraper.stats}")

        await scraper.close()

    asyncio.run(main())
