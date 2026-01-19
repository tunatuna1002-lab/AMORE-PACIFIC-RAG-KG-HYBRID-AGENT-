"""
Amazon Product Detail Scraper
상품 상세 페이지에서 AI Customers Say 및 추가 정보 수집

이 모듈은 amazon_scraper.py(베스트셀러 목록 크롤러)와 분리되어 있으며,
상품 상세 페이지(/dp/ASIN)에서만 접근 가능한 정보를 수집합니다.

수집 정보:
    - AI Customers Say: Amazon의 AI 리뷰 요약
    - 리뷰 감성 태그: 고객들이 언급한 키워드 (예: "Value for money", "Easy to use")
    - 상세 제품 정보: 용량, 성분, 제조사 등

Errors:
    - "BLOCKED": Amazon 차단됨 - IP 변경 또는 대기 필요
    - "TIMEOUT": 응답 없음 - 재시도 권장
    - "NOT_FOUND": 상품 페이지 없음
    - "PARSE_ERROR": HTML 구조 변경 - 파서 업데이트 필요
"""

import asyncio
import re
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Optional, Any
from playwright.async_api import async_playwright, Page, Browser, TimeoutError as PlaywrightTimeout
import json
import os
import random

# 한국 시간대 (UTC+9)
KST = timezone(timedelta(hours=9))


class AmazonProductScraper:
    """Amazon 상품 상세 페이지 크롤러 (AI Customers Say 수집 전용)"""

    def __init__(self, config_path: str = "./config/thresholds.json"):
        """
        Args:
            config_path: 설정 파일 경로
        """
        self.config = self._load_config(config_path)
        self.base_url = "https://www.amazon.com"
        self.delay_seconds = float(os.getenv("SCRAPE_DELAY_SECONDS", "3"))  # 상세 페이지는 더 긴 딜레이
        self.browser: Optional[Browser] = None

    def _load_config(self, config_path: str) -> dict:
        """설정 파일 로드"""
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            return {"target_brands": ["LANEIGE"]}

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

    async def scrape_product(self, asin: str) -> Dict[str, Any]:
        """
        단일 상품 상세 페이지 크롤링

        Args:
            asin: Amazon 상품 ID

        Returns:
            {
                "asin": "B0BSHRYY1S",
                "ai_customers_say": "Customers like the moisturizing...",
                "sentiment_tags": ["Moisturizing", "Value for money", "Easy to use"],
                "product_details": {...},
                "collected_at": "2025-01-15T10:30:00+09:00",
                "success": True,
                "error": None
            }
        """
        result = {
            "asin": asin,
            "ai_customers_say": None,
            "sentiment_tags": [],
            "product_details": {},
            "collected_at": datetime.now(KST).isoformat(),
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
            product_url = f"{self.base_url}/dp/{asin}"
            await page.goto(product_url, wait_until="domcontentloaded", timeout=30000)
            await self._random_delay()

            # 차단 감지
            if await self._is_blocked(page):
                result["error"] = "BLOCKED"
                return result

            # 404 감지
            if await self._is_not_found(page):
                result["error"] = "NOT_FOUND"
                return result

            # AI Customers Say 추출
            ai_summary = await self._extract_ai_customers_say(page)
            result["ai_customers_say"] = ai_summary

            # 감성 태그 추출
            sentiment_tags = await self._extract_sentiment_tags(page)
            result["sentiment_tags"] = sentiment_tags

            # 상세 정보 추출
            product_details = await self._extract_product_details(page)
            result["product_details"] = product_details

            result["success"] = True

        except PlaywrightTimeout:
            result["error"] = "TIMEOUT"
        except Exception as e:
            result["error"] = f"PARSE_ERROR: {str(e)}"
        finally:
            await context.close()

        return result

    async def scrape_products(self, asins: List[str], max_concurrent: int = 1) -> Dict[str, Any]:
        """
        여러 상품 크롤링

        Args:
            asins: ASIN 목록
            max_concurrent: 동시 크롤링 수 (차단 방지를 위해 1 권장)

        Returns:
            {
                "products": {...},  # ASIN별 결과
                "total": 20,
                "success_count": 18,
                "error_count": 2,
                "collected_at": "2025-01-15"
            }
        """
        results = {
            "products": {},
            "total": len(asins),
            "success_count": 0,
            "error_count": 0,
            "collected_at": datetime.now(KST).date().isoformat()
        }

        for asin in asins:
            product_result = await self.scrape_product(asin)
            results["products"][asin] = product_result

            if product_result["success"]:
                results["success_count"] += 1
            else:
                results["error_count"] += 1

            # 상품 간 딜레이 (차단 방지)
            await asyncio.sleep(self.delay_seconds * 2)

            # 차단 감지시 중단
            if product_result.get("error") == "BLOCKED":
                break

        return results

    async def scrape_top_products_by_category(
        self,
        category_products: Dict[str, List[Dict]],
        top_n: int = 20,
        brands_filter: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        카테고리별 Top N 상품의 상세 정보 수집

        Args:
            category_products: 카테고리별 상품 목록 (베스트셀러 크롤링 결과)
                {
                    "lip_care": [{"asin": "...", "rank": 1, "brand": "..."}, ...],
                    "skin_care": [...]
                }
            top_n: 각 카테고리에서 수집할 상위 N개 (기본 20)
            brands_filter: 특정 브랜드만 수집 (None이면 전체)

        Returns:
            {
                "categories": {
                    "lip_care": {...},
                    "skin_care": {...}
                },
                "total_collected": 100,
                "collected_at": "2025-01-15"
            }
        """
        results = {
            "categories": {},
            "total_collected": 0,
            "collected_at": datetime.now(KST).date().isoformat()
        }

        for category_id, products in category_products.items():
            # Top N 필터링
            filtered_products = products[:top_n]

            # 브랜드 필터링 (선택사항)
            if brands_filter:
                filtered_products = [
                    p for p in filtered_products
                    if any(b.lower() in p.get("brand", "").lower() for b in brands_filter)
                ]

            asins = [p["asin"] for p in filtered_products if p.get("asin")]

            category_result = await self.scrape_products(asins)
            results["categories"][category_id] = category_result
            results["total_collected"] += category_result["success_count"]

            # 카테고리 간 딜레이
            await asyncio.sleep(self.delay_seconds * 3)

        return results

    async def _extract_ai_customers_say(self, page: Page) -> Optional[str]:
        """AI Customers Say 텍스트 추출"""
        # Amazon의 AI 리뷰 요약 섹션 선택자들
        selectors = [
            # AI Customers Say 섹션
            "#cm-cr-summarization-attributes-slot p",
            "[data-hook='cr-ai-summary-text']",
            "#cr-ai-summary-text",
            ".cr-ai-summary-text",
            # "What customers say" 섹션
            "[data-hook='summarization-text']",
            ".cr-summarization-text",
            # 대체 선택자
            "#cr-customer-insights-content"
        ]

        for selector in selectors:
            try:
                elem = await page.query_selector(selector)
                if elem:
                    text = await elem.text_content()
                    if text and len(text.strip()) > 20:  # 최소 길이 검증
                        return text.strip()
            except Exception:
                continue

        return None

    async def _extract_sentiment_tags(self, page: Page) -> List[str]:
        """
        고객 감성 태그 추출
        (예: "Moisturizing", "Value for money", "Easy to use")
        """
        tags = []

        # 감성 태그 컨테이너 선택자들
        tag_selectors = [
            # CR (Customer Review) 인사이트 태그
            "#cr-customer-insights-slot .a-declarative span",
            "[data-hook='cr-insights-widget-tag-text']",
            ".cr-insights-widget-tag-text",
            # 리뷰 키워드 태그
            "[data-hook='review-highlight']",
            ".cr-lighthouse-term"
        ]

        for selector in tag_selectors:
            try:
                elements = await page.query_selector_all(selector)
                for elem in elements:
                    text = await elem.text_content()
                    if text:
                        tag = text.strip()
                        # 유효한 태그인지 검증 (너무 길거나 짧은 것 제외)
                        if 2 <= len(tag) <= 50 and tag not in tags:
                            tags.append(tag)
            except Exception:
                continue

        return tags[:10]  # 최대 10개

    async def _extract_product_details(self, page: Page) -> Dict[str, str]:
        """상세 제품 정보 추출"""
        details = {}

        # Product Information 테이블
        try:
            rows = await page.query_selector_all("#productDetails_techSpec_section_1 tr, #productDetails_detailBullets_sections1 tr")
            for row in rows:
                th = await row.query_selector("th")
                td = await row.query_selector("td")
                if th and td:
                    key = (await th.text_content()).strip()
                    value = (await td.text_content()).strip()
                    if key and value:
                        details[key] = value
        except Exception:
            pass

        # Feature Bullets (About this item)
        try:
            bullets = await page.query_selector_all("#feature-bullets ul li span")
            features = []
            for bullet in bullets:
                text = await bullet.text_content()
                if text and text.strip():
                    features.append(text.strip())
            if features:
                details["features"] = features[:5]  # 상위 5개
        except Exception:
            pass

        # A+ Content에서 주요 정보 추출
        try:
            aplus_headings = await page.query_selector_all("#aplus h3, #aplus h4")
            aplus_titles = []
            for heading in aplus_headings[:3]:
                text = await heading.text_content()
                if text and text.strip():
                    aplus_titles.append(text.strip())
            if aplus_titles:
                details["aplus_highlights"] = aplus_titles
        except Exception:
            pass

        return details

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

    async def _is_not_found(self, page: Page) -> bool:
        """404 페이지 확인"""
        content = await page.content()
        not_found_indicators = [
            "Page Not Found",
            "Looking for something?",
            "We're sorry. The Web address you entered is not a functioning page"
        ]
        return any(indicator in content for indicator in not_found_indicators)

    async def _random_delay(self) -> None:
        """랜덤 딜레이"""
        delay = self.delay_seconds + random.uniform(1.0, 2.0)
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
async def scrape_ai_customers_say(asin: str) -> Dict[str, Any]:
    """
    단일 상품의 AI Customers Say 수집 (단일 함수 인터페이스)

    Args:
        asin: Amazon 상품 ID

    Returns:
        {
            "asin": "B0BSHRYY1S",
            "ai_customers_say": "Customers like the moisturizing...",
            "sentiment_tags": ["Moisturizing", "Value for money"],
            "success": True
        }

    Errors:
        - "BLOCKED": Amazon 차단됨
        - "TIMEOUT": 응답 없음
        - "NOT_FOUND": 상품 없음
    """
    scraper = AmazonProductScraper()
    try:
        await scraper.initialize()
        result = await scraper.scrape_product(asin)
        return result
    finally:
        await scraper.close()


async def scrape_laneige_ai_summaries(
    category_products: Dict[str, List[Dict]],
    top_n: int = 20
) -> Dict[str, Any]:
    """
    LANEIGE 제품들의 AI Customers Say 수집

    Args:
        category_products: 카테고리별 상품 목록
        top_n: 각 카테고리에서 수집할 상위 N개

    Returns:
        카테고리별 LANEIGE 제품의 AI 리뷰 요약
    """
    scraper = AmazonProductScraper()
    try:
        await scraper.initialize()
        # LANEIGE 브랜드만 필터링
        result = await scraper.scrape_top_products_by_category(
            category_products,
            top_n=top_n,
            brands_filter=scraper.config.get("target_brands", ["LANEIGE"])
        )
        return result
    finally:
        await scraper.close()
