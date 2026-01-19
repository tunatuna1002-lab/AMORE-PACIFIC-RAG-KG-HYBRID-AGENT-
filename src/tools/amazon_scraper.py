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
        # 카테고리 설정에서 정보 가져오기
        cat_config = self.config.get("categories", {}).get(category_id, {})
        category_name = cat_config.get("name", category_id)
        amazon_node_id = cat_config.get("amazon_node_id", category_id)
        category_level = cat_config.get("level", 0)

        result = {
            "products": [],
            "count": 0,
            "category_id": category_id,
            "category": category_name,
            "amazon_node_id": amazon_node_id,
            "category_level": category_level,
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
        """개별 제품 데이터 추출 (프로모션 정보 포함)"""
        try:
            # 제품명
            name_elem = await card.query_selector(".p13n-sc-truncate, ._cDEzb_p13n-sc-css-line-clamp-3_g3dy1, .a-link-normal span")
            product_name = await name_elem.inner_text() if name_elem else "Unknown"
            product_name = product_name.strip()

            # 브랜드 추출 (제품명에서 추출 시도)
            brand = self._extract_brand(product_name)

            # 가격 (현재 판매가) - 더 구체적인 선택자 사용
            price = None
            # Amazon 베스트셀러 페이지의 가격 선택자들 (우선순위 순)
            price_selectors = [
                ".p13n-sc-price",  # 베스트셀러 페이지 기본
                "._cDEzb_p13n-sc-price_3mJ9Z",  # 대체 클래스
                ".a-price:not([data-a-strike]) .a-offscreen",  # 할인가 (취소선 없는 가격)
                ".a-color-price"  # 가격 색상 클래스
            ]
            for selector in price_selectors:
                price_elem = await card.query_selector(selector)
                if price_elem:
                    price_text = await price_elem.inner_text()
                    price = self._parse_price(price_text)
                    if price:  # 유효한 가격을 찾으면 중단
                        break

            # 원가 (정가, 할인 전 가격)
            list_price = None
            list_price_selectors = [
                ".a-text-price .a-offscreen",  # 취소선 가격
                ".a-price[data-a-strike='true'] .a-offscreen",
                "span.a-text-price span.a-offscreen"
            ]
            for selector in list_price_selectors:
                list_price_elem = await card.query_selector(selector)
                if list_price_elem:
                    list_price_text = await list_price_elem.inner_text()
                    list_price = self._parse_price(list_price_text)
                    if list_price:
                        break

            # 할인율 계산
            discount_percent = None
            if price and list_price and list_price > price:
                discount_percent = round((1 - price / list_price) * 100, 1)

            # 평점 - span.a-icon-alt에서 "X out of 5 stars" 텍스트 찾기
            rating = None
            # 방법 1: span.a-icon-alt (베스트셀러 페이지 주요 패턴)
            rating_alt = await card.query_selector("span.a-icon-alt")
            if rating_alt:
                rating_text = await rating_alt.text_content()  # aria-hidden 대응
                rating = self._parse_rating(rating_text)

            # 방법 2: i 태그의 aria-label (대체 패턴)
            if not rating:
                rating_selectors = [
                    ".a-icon-star-small",
                    ".a-icon-star-mini",
                    "i.a-icon-star"
                ]
                for selector in rating_selectors:
                    rating_elem = await card.query_selector(selector)
                    if rating_elem:
                        rating_text = await rating_elem.get_attribute("aria-label")
                        if rating_text:
                            rating = self._parse_rating(rating_text)
                            if rating:
                                break

            # 리뷰 수 - span.a-size-small에서 숫자만 있는 텍스트 찾기
            reviews_count = None
            # 방법 1: 직접 span.a-size-small 선택 (베스트셀러 페이지 주요 패턴)
            # aria-hidden="true" 속성이 있으므로 text_content() 사용
            review_spans = await card.query_selector_all("span.a-size-small")
            for review_span in review_spans:
                reviews_text = await review_span.text_content()  # aria-hidden 대응
                reviews_text = reviews_text.strip() if reviews_text else ""
                # 숫자와 콤마만 있는지 확인 (리뷰 수 패턴: "13,265")
                # [0-9,]+ 패턴 사용 (일부 환경에서 \d가 다르게 동작할 수 있음)
                if reviews_text and re.match(r'^[0-9,]+$', reviews_text):
                    reviews_count = self._parse_reviews_count(reviews_text)
                    if reviews_count:
                        break

            # 방법 2: 링크 안에 있는 경우 (대체 패턴)
            if not reviews_count:
                reviews_selectors = [
                    ".a-size-small .a-link-normal",
                    "a.a-size-small"
                ]
                for selector in reviews_selectors:
                    reviews_elem = await card.query_selector(selector)
                    if reviews_elem:
                        reviews_text = await reviews_elem.text_content()
                        reviews_count = self._parse_reviews_count(reviews_text)
                        if reviews_count:
                            break

            # 뱃지 (Best Seller, Amazon's Choice 등)
            badge = ""
            badge_elem = await card.query_selector(".a-badge-text, .p13n-best-seller-badge")
            if badge_elem:
                badge = await badge_elem.inner_text()

            # === 프로모션 정보 추출 ===

            # 쿠폰 정보 (예: "Save 5% with coupon", "Save $2.00 with coupon")
            coupon_text = ""
            coupon_elem = await card.query_selector(".s-coupon-highlight-color, .couponBadge, [data-component-type='s-coupon-component']")
            if coupon_elem:
                coupon_text = (await coupon_elem.inner_text()).strip()

            # Subscribe & Save 여부
            is_subscribe_save = False
            sns_elem = await card.query_selector(".s-subscriptions-terms, [data-component-type='s-subscribe-and-save']")
            if sns_elem:
                is_subscribe_save = True
            # 텍스트로도 확인
            card_html = await card.inner_html()
            if "Subscribe & Save" in card_html or "Subscribe &amp; Save" in card_html:
                is_subscribe_save = True

            # 프로모션 배지들 (Limited Time Deal, Lightning Deal 등)
            promo_badges = []

            # Limited Time Deal
            ltd_elem = await card.query_selector(".a-badge-limited-time-deal, [data-badge='limited-time-deal']")
            if ltd_elem or "Limited time deal" in card_html:
                promo_badges.append("Limited Time Deal")

            # Lightning Deal
            if "Lightning Deal" in card_html:
                promo_badges.append("Lightning Deal")

            # Prime Day Deal / Holiday Deal 등
            deal_badge = await card.query_selector(".dealBadge, .a-badge-deal")
            if deal_badge:
                deal_text = await deal_badge.inner_text()
                if deal_text and deal_text.strip():
                    promo_badges.append(deal_text.strip())

            # Climate Pledge Friendly
            if "Climate Pledge" in card_html:
                promo_badges.append("Climate Pledge Friendly")

            # Amazon's Choice (badge와 별도로 저장)
            if "Amazon's Choice" in card_html or "Amazons Choice" in card_html:
                if "Amazon's Choice" not in promo_badges:
                    promo_badges.append("Amazon's Choice")

            # 프로모션 배지를 콤마로 연결
            promo_badges_str = ", ".join(promo_badges) if promo_badges else ""

            # URL
            link_elem = await card.query_selector("a.a-link-normal")
            href = await link_elem.get_attribute("href") if link_elem else ""
            product_url = f"{self.base_url}{href}" if href and not href.startswith("http") else href

            # 카테고리 설정에서 amazon_node_id 가져오기
            cat_config = self.config.get("categories", {}).get(category_id, {})
            amazon_node_id = cat_config.get("amazon_node_id", category_id)
            category_level = cat_config.get("level", 0)
            category_name = cat_config.get("name", category_id)

            return {
                "snapshot_date": snapshot_date,
                "collected_at": datetime.now(KST).isoformat(),  # 정확한 수집 시간 (ISO 8601 형식)
                "category_id": category_id,
                "amazon_node_id": amazon_node_id,
                "category_name": category_name,
                "category_level": category_level,
                "asin": asin,
                "product_name": product_name,
                "brand": brand,
                "rank": rank,
                "price": price,
                "list_price": list_price,
                "discount_percent": discount_percent,
                "rating": rating,
                "reviews_count": reviews_count,
                "badge": badge,
                "coupon_text": coupon_text,
                "is_subscribe_save": is_subscribe_save,
                "promo_badges": promo_badges_str,
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
        """가격 문자열 파싱 - $ 기호가 있는 가격만 인식"""
        try:
            if not price_text:
                return None

            # $ 기호가 반드시 있어야 함 (리뷰 수 등 다른 숫자와 구분)
            if '$' not in price_text:
                return None

            # "$24.00" -> 24.00, "$1,234.56" -> 1234.56
            match = re.search(r'\$([\d,]+\.?\d*)', price_text)
            if match:
                price = float(match.group(1).replace(",", ""))
                # 합리적인 가격 범위 검증 (뷰티 제품: $0.50 ~ $500)
                if 0.50 <= price <= 500:
                    return price
        except (ValueError, TypeError, AttributeError):
            pass
        return None

    def _parse_rating(self, rating_text: str) -> Optional[float]:
        """평점 문자열 파싱 - 5점 만점 평점만 인식"""
        try:
            if not rating_text:
                return None

            # "4.7 out of 5 stars" 또는 "4.7 out of 5" 패턴
            match = re.search(r'([0-9.]+)\s*out of\s*5', rating_text, re.IGNORECASE)
            if match:
                rating = float(match.group(1))
                # 평점 범위 검증 (0.0 ~ 5.0)
                if 0.0 <= rating <= 5.0:
                    return round(rating, 1)

            # "4.7/5" 패턴
            match = re.search(r'([0-9.]+)\s*/\s*5', rating_text)
            if match:
                rating = float(match.group(1))
                if 0.0 <= rating <= 5.0:
                    return round(rating, 1)

            # Fallback: 문자열 시작 부분에서 숫자 추출 (예: "4.8 out of 5 stars")
            # parseFloat 스타일 파싱
            first_word = rating_text.strip().split()[0] if rating_text.strip() else ""
            if first_word and re.match(r'^[0-9.]+$', first_word):
                rating = float(first_word)
                if 0.0 <= rating <= 5.0:
                    return round(rating, 1)

        except (ValueError, TypeError, AttributeError):
            pass
        return None

    def _parse_reviews_count(self, reviews_text: str) -> Optional[int]:
        """리뷰 수 문자열 파싱 - 합리적인 범위 검증"""
        try:
            if not reviews_text:
                return None

            # 콤마 제거하고 숫자 추출: "89,234" -> 89234
            clean_text = reviews_text.replace(",", "").replace("+", "").strip()
            match = re.search(r'^(\d+)$', clean_text)
            if match:
                count = int(match.group(1))
                # 합리적인 리뷰 수 범위 (0 ~ 1,000,000)
                if 0 <= count <= 1000000:
                    return count
        except (ValueError, TypeError, AttributeError):
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
