"""
Amazon Bestseller Scraper
Playwright 기반 Amazon Top 100 크롤러

Errors:
    - "BLOCKED": Amazon 차단됨 - IP 변경 또는 대기 필요
    - "TIMEOUT": 응답 없음 - 재시도 권장
    - "PARSE_ERROR": HTML 구조 변경 - 파서 업데이트 필요
"""

import asyncio
import re
from datetime import date, datetime, timezone, timedelta
from typing import List, Dict, Optional, Any
from playwright.async_api import async_playwright, Page, Browser, TimeoutError as PlaywrightTimeout
import json
import os
import random

# 한국 시간대 (UTC+9)
KST = timezone(timedelta(hours=9))

from src.ontology.schema import RankRecord


class AmazonScraper:
    """Amazon 베스트셀러 Top 100 크롤러"""

    def __init__(self, config_path: str = "./config/thresholds.json"):
        """
        Args:
            config_path: 설정 파일 경로
        """
        self.config = self._load_config(config_path)
        self.base_url = "https://www.amazon.com"
        self.delay_seconds = float(os.getenv("SCRAPE_DELAY_SECONDS", "2"))
        self.browser: Optional[Browser] = None

    def _load_config(self, config_path: str) -> dict:
        """설정 파일 로드"""
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            return {"categories": {}}

    async def initialize(self) -> None:
        """브라우저 초기화"""
        playwright = await async_playwright().start()
        self.browser = await playwright.chromium.launch(
            headless=True,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--no-sandbox",
                "--disable-dev-shm-usage"
            ]
        )

    async def close(self) -> None:
        """브라우저 종료"""
        if self.browser:
            await self.browser.close()

    async def scrape_category(self, category_id: str, category_url: str) -> Dict[str, Any]:
        """
        단일 카테고리 베스트셀러 Top 100 크롤링

        Args:
            category_id: 카테고리 ID (예: "lip_care")
            category_url: Amazon 베스트셀러 URL

        Returns:
            {
                "products": [...],  # 최대 100개 RankRecord
                "count": 100,
                "category": "Lip Care",
                "category_id": "lip_care",
                "snapshot_date": "2025-01-15",
                "success": True,
                "error": None
            }

        Errors:
            - "BLOCKED": Amazon 차단됨 - IP 변경 또는 대기 필요
            - "TIMEOUT": 응답 없음 - 재시도 권장
            - "PARSE_ERROR": HTML 구조 변경 - 파서 업데이트 필요
        """
        result = {
            "products": [],
            "count": 0,
            "category_id": category_id,
            "category": category_id,
            "snapshot_date": datetime.now(KST).date().isoformat(),
            "success": False,
            "error": None
        }

        if not self.browser:
            await self.initialize()

        context = await self.browser.new_context(
            user_agent=self._get_random_user_agent(),
            viewport={"width": 1920, "height": 1080},
            locale="en-US"
        )

        page = await context.new_page()

        try:
            # 페이지 1 로드 (1-50위)
            await page.goto(category_url, wait_until="domcontentloaded", timeout=30000)
            await self._random_delay()

            # 차단 감지
            if await self._is_blocked(page):
                result["error"] = "BLOCKED"
                return result

            # 첫 페이지 파싱 (1-50위)
            products_page1 = await self._parse_bestseller_page(page, category_id, start_rank=1)
            result["products"].extend(products_page1)

            # 페이지 2 로드 (51-100위)
            page2_url = self._get_page2_url(category_url)
            if page2_url:
                await page.goto(page2_url, wait_until="domcontentloaded", timeout=30000)
                await self._random_delay()

                if not await self._is_blocked(page):
                    products_page2 = await self._parse_bestseller_page(page, category_id, start_rank=51)
                    result["products"].extend(products_page2)

            result["count"] = len(result["products"])
            result["success"] = True

        except PlaywrightTimeout:
            result["error"] = "TIMEOUT"
        except Exception as e:
            result["error"] = f"PARSE_ERROR: {str(e)}"
        finally:
            await context.close()

        return result

    async def scrape_all_categories(self) -> Dict[str, Any]:
        """
        모든 카테고리 크롤링

        Returns:
            {
                "categories": {
                    "lip_care": {...},
                    "skin_care": {...},
                    ...
                },
                "total_products": 500,
                "snapshot_date": "2025-01-15",
                "success_count": 5,
                "error_count": 0
            }
        """
        results = {
            "categories": {},
            "total_products": 0,
            "snapshot_date": datetime.now(KST).date().isoformat(),
            "success_count": 0,
            "error_count": 0
        }

        categories = self.config.get("categories", {})

        for cat_id, cat_info in categories.items():
            result = await self.scrape_category(cat_id, cat_info["url"])
            results["categories"][cat_id] = result

            if result["success"]:
                results["success_count"] += 1
                results["total_products"] += result["count"]
            else:
                results["error_count"] += 1

            # 카테고리 간 딜레이
            await asyncio.sleep(self.delay_seconds * 2)

        return results

    async def _parse_bestseller_page(self, page: Page, category_id: str, start_rank: int = 1) -> List[Dict]:
        """베스트셀러 페이지 파싱"""
        products = []
        snapshot_date = datetime.now(KST).date().isoformat()

        # 제품 카드 선택자들 (Amazon 구조에 따라 조정 필요)
        product_cards = await page.query_selector_all('[data-asin]:not([data-asin=""])')

        rank = start_rank
        for card in product_cards:
            if rank > start_rank + 49:  # 페이지당 최대 50개
                break

            try:
                asin = await card.get_attribute("data-asin")
                if not asin:
                    continue

                product_data = await self._extract_product_data(card, asin, rank, category_id, snapshot_date)
                if product_data:
                    products.append(product_data)
                    rank += 1

            except Exception:
                continue

        return products

    async def _extract_product_data(self, card, asin: str, rank: int, category_id: str, snapshot_date: str) -> Optional[Dict]:
        """개별 제품 데이터 추출"""
        try:
            # 제품명
            name_elem = await card.query_selector(".p13n-sc-truncate, ._cDEzb_p13n-sc-css-line-clamp-3_g3dy1, .a-link-normal span")
            product_name = await name_elem.inner_text() if name_elem else "Unknown"
            product_name = product_name.strip()

            # 브랜드 추출 (제품명에서 추출 시도)
            brand = self._extract_brand(product_name)

            # 가격
            price = None
            price_elem = await card.query_selector(".p13n-sc-price, ._cDEzb_p13n-sc-price_3mJ9Z, .a-price .a-offscreen")
            if price_elem:
                price_text = await price_elem.inner_text()
                price = self._parse_price(price_text)

            # 평점
            rating = None
            rating_elem = await card.query_selector(".a-icon-star-small, .a-icon-alt")
            if rating_elem:
                rating_text = await rating_elem.get_attribute("aria-label") or await rating_elem.inner_text()
                rating = self._parse_rating(rating_text)

            # 리뷰 수
            reviews_count = None
            reviews_elem = await card.query_selector(".a-size-small a span")
            if reviews_elem:
                reviews_text = await reviews_elem.inner_text()
                reviews_count = self._parse_reviews_count(reviews_text)

            # 뱃지
            badge = ""
            badge_elem = await card.query_selector(".a-badge-text, .p13n-best-seller-badge")
            if badge_elem:
                badge = await badge_elem.inner_text()

            # URL
            link_elem = await card.query_selector("a.a-link-normal")
            href = await link_elem.get_attribute("href") if link_elem else ""
            product_url = f"{self.base_url}{href}" if href and not href.startswith("http") else href

            return {
                "snapshot_date": snapshot_date,
                "category_id": category_id,
                "asin": asin,
                "product_name": product_name,
                "brand": brand,
                "rank": rank,
                "price": price,
                "rating": rating,
                "reviews_count": reviews_count,
                "badge": badge,
                "product_url": product_url or f"{self.base_url}/dp/{asin}"
            }

        except Exception:
            return None

    def _extract_brand(self, product_name: str) -> str:
        """제품명에서 브랜드 추출"""
        # 일반적으로 첫 단어가 브랜드인 경우가 많음
        known_brands = ["LANEIGE", "Laneige", "COSRX", "TIRTIR", "Rare Beauty",
                       "e.l.f.", "NYX", "Maybelline", "L'Oreal", "Neutrogena",
                       "CeraVe", "La Roche-Posay", "SKIN1004", "Beauty of Joseon"]

        for brand in known_brands:
            if brand.lower() in product_name.lower():
                return brand

        # 첫 단어를 브랜드로 추정
        words = product_name.split()
        return words[0] if words else "Unknown"

    def _parse_price(self, price_text: str) -> Optional[float]:
        """가격 문자열 파싱"""
        try:
            # "$24.00" -> 24.00
            match = re.search(r'\$?([\d,]+\.?\d*)', price_text.replace(",", ""))
            if match:
                return float(match.group(1))
        except:
            pass
        return None

    def _parse_rating(self, rating_text: str) -> Optional[float]:
        """평점 문자열 파싱"""
        try:
            # "4.7 out of 5 stars" -> 4.7
            match = re.search(r'([\d.]+)\s*(?:out of|\/)', rating_text)
            if match:
                return float(match.group(1))
            # "4.7" -> 4.7
            match = re.search(r'^([\d.]+)$', rating_text.strip())
            if match:
                return float(match.group(1))
        except:
            pass
        return None

    def _parse_reviews_count(self, reviews_text: str) -> Optional[int]:
        """리뷰 수 문자열 파싱"""
        try:
            # "89,234" -> 89234
            clean_text = reviews_text.replace(",", "").replace("+", "")
            match = re.search(r'(\d+)', clean_text)
            if match:
                return int(match.group(1))
        except:
            pass
        return None

    def _get_page2_url(self, url: str) -> Optional[str]:
        """페이지 2 URL 생성"""
        if "pg=2" in url:
            return url
        separator = "&" if "?" in url else "?"
        return f"{url}{separator}pg=2"

    async def _is_blocked(self, page: Page) -> bool:
        """차단 여부 확인"""
        content = await page.content()
        block_indicators = [
            "Enter the characters you see below",
            "Sorry, we just need to make sure you're not a robot",
            "Type the characters you see in this image",
            "api-services-support@amazon.com"
        ]
        return any(indicator in content for indicator in block_indicators)

    async def _random_delay(self) -> None:
        """랜덤 딜레이"""
        delay = self.delay_seconds + random.uniform(0.5, 1.5)
        await asyncio.sleep(delay)

    def _get_random_user_agent(self) -> str:
        """랜덤 User-Agent 반환"""
        user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Edge/120.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15"
        ]
        return random.choice(user_agents)


# 편의 함수
async def scrape_bestsellers(category_url: str, category_id: str = "unknown") -> Dict[str, Any]:
    """
    Amazon 베스트셀러 Top 100 크롤링 (단일 함수 인터페이스)

    Args:
        category_url: Amazon 카테고리 URL
        category_id: 카테고리 식별자

    Returns:
        {
            "products": [...],  # 최대 100개
            "count": 100,
            "category": "Lip Care",
            "snapshot_date": "2025-01-15"
        }

    Errors:
        - "BLOCKED": Amazon 차단됨 - IP 변경 또는 대기 필요
        - "TIMEOUT": 응답 없음 - 재시도 권장
        - "PARSE_ERROR": HTML 구조 변경 - 파서 업데이트 필요
    """
    scraper = AmazonScraper()
    try:
        await scraper.initialize()
        result = await scraper.scrape_category(category_id, category_url)
        return result
    finally:
        await scraper.close()
