"""
Amazon Deals Scraper
Playwright 기반 Amazon Deals 페이지 크롤러

목적:
- 경쟁사 할인 모니터링
- Lightning Deals 시간 한정 할인 추적
- 뷰티 카테고리 할인 상품 수집

수집 데이터:
- 상품 정보 (ASIN, 제품명, 브랜드)
- 할인 정보 (원가, 할인가, 할인율)
- Lightning Deal 정보 (남은 시간, 판매율)
- Deal 타입 (Lightning, Deal of the Day, Best Deal)
"""

import asyncio
import json
import logging
import os
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any

from playwright.async_api import Browser, Page, async_playwright
from playwright.async_api import TimeoutError as PlaywrightTimeout

logger = logging.getLogger(__name__)

# 한국 시간대 (UTC+9)
KST = timezone(timedelta(hours=9))


class DealType(Enum):
    """딜 타입"""

    LIGHTNING = "lightning"  # 시간 한정 Lightning Deal
    DEAL_OF_THE_DAY = "deal_of_day"  # 오늘의 딜
    BEST_DEAL = "best_deal"  # Best Deal
    COUPON = "coupon"  # 쿠폰 할인
    REGULAR = "regular"  # 일반 할인


@dataclass
class DealRecord:
    """할인 정보 레코드"""

    snapshot_datetime: str  # 수집 시각 (ISO format)
    asin: str  # Amazon 상품 ID
    product_name: str  # 상품명
    brand: str  # 브랜드
    category: str  # 카테고리 (Beauty 등)

    # 가격 정보
    deal_price: float  # 할인가
    original_price: float | None = None  # 원가
    discount_percent: float | None = None  # 할인율

    # 딜 정보
    deal_type: str = "regular"  # DealType
    deal_badge: str | None = None  # "Limited time deal" 등

    # Lightning Deal 전용
    time_remaining: str | None = None  # "2h 30m" 남은 시간
    time_remaining_seconds: int | None = None  # 초 단위
    claimed_percent: int | None = None  # 판매된 비율 (%)
    deal_end_time: str | None = None  # 종료 예정 시각

    # 메타데이터
    product_url: str | None = None
    image_url: str | None = None
    rating: float | None = None
    reviews_count: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class AmazonDealsScraper:
    """Amazon Deals 페이지 크롤러"""

    # 뷰티 관련 Deals URL
    DEALS_URLS = {
        "all_deals": "https://www.amazon.com/gp/goldbox",
        "beauty_deals": "https://www.amazon.com/gp/goldbox?ref=nav_cs_gb&deals-widget=%257B%2522version%2522%253A1%252C%2522viewIndex%2522%253A0%252C%2522presetId%2522%253A%2522deals-collection-all-702324011%2522%252C%2522sorting%2522%253A%2522BY_SCORE%2522%257D",
        # 뷰티 카테고리 node ID: 3760911
        "beauty_lightning": "https://www.amazon.com/gp/goldbox?ref_=nav_cs_gb&gb_f_deal_c=1&gb_f_c=1&gb_f_h=1&gb_s=RELEVANCE&gb_f_nc=3760911",
    }

    # 관심 브랜드 (경쟁사)
    WATCH_BRANDS = [
        "e.l.f.",
        "elf",
        "E.L.F.",
        "Maybelline",
        "MAYBELLINE",
        "L'Oreal",
        "L'OREAL",
        "Loreal",
        "NYX",
        "nyx",
        "Neutrogena",
        "NEUTROGENA",
        "CeraVe",
        "CERAVE",
        "COSRX",
        "cosrx",  # 대소문자 변형 매칭
        "La Roche-Posay",
        "Cetaphil",
        "CETAPHIL",
        "The Ordinary",
        "LANEIGE",
        "Laneige",  # 자사 브랜드도 모니터링
    ]

    def __init__(self, config_path: str = "./config/thresholds.json"):
        self.config = self._load_config(config_path)
        self.browser: Browser | None = None
        self.delay_seconds = float(os.getenv("DEALS_SCRAPE_DELAY", "3"))

    def _load_config(self, config_path: str) -> dict:
        try:
            with open(config_path, encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            return {}

    async def initialize(self) -> None:
        """브라우저 초기화"""
        playwright = await async_playwright().start()
        self.browser = await playwright.chromium.launch(
            headless=True,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--no-sandbox",
                "--disable-dev-shm-usage",
            ],
        )
        logger.info("DealsScraper browser initialized")

    async def close(self) -> None:
        """브라우저 종료"""
        if self.browser:
            await self.browser.close()
            logger.info("DealsScraper browser closed")

    async def scrape_deals(
        self, url: str | None = None, max_items: int = 50, beauty_only: bool = True
    ) -> dict[str, Any]:
        """
        Amazon Deals 페이지 크롤링

        Args:
            url: Deals 페이지 URL (없으면 기본 뷰티 딜)
            max_items: 최대 수집 개수
            beauty_only: 뷰티 카테고리만 필터링

        Returns:
            {
                "deals": [...],  # DealRecord 리스트
                "count": 50,
                "lightning_count": 10,
                "competitor_deals": [...],  # 경쟁사 딜만
                "snapshot_datetime": "2025-01-20T10:30:00+09:00",
                "success": True,
                "error": None
            }
        """
        if not self.browser:
            await self.initialize()

        target_url = url or self.DEALS_URLS["beauty_deals"]
        snapshot_time = datetime.now(KST).isoformat()

        result = {
            "deals": [],
            "count": 0,
            "lightning_count": 0,
            "competitor_deals": [],
            "snapshot_datetime": snapshot_time,
            "success": False,
            "error": None,
        }

        context = None
        page = None

        try:
            # 브라우저 컨텍스트 생성
            context = await self.browser.new_context(
                viewport={"width": 1920, "height": 1080},
                user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            )
            page = await context.new_page()

            # 페이지 로드
            logger.info(f"Loading deals page: {target_url}")
            await page.goto(target_url, wait_until="domcontentloaded", timeout=30000)
            await asyncio.sleep(2)  # JavaScript 렌더링 대기

            # 차단 확인
            if await self._is_blocked(page):
                result["error"] = "BLOCKED"
                logger.warning("Amazon blocked the request")
                return result

            # 스크롤하여 더 많은 딜 로드
            await self._scroll_page(page, scroll_count=3)

            # 딜 카드 파싱
            deals = await self._parse_deal_cards(page, max_items, snapshot_time)

            # 뷰티 필터링
            if beauty_only:
                deals = [d for d in deals if self._is_beauty_product(d)]

            # 경쟁사 딜 필터링
            competitor_deals = [
                d
                for d in deals
                if any(brand.lower() in d.brand.lower() for brand in self.WATCH_BRANDS)
            ]

            # Lightning Deal 카운트
            lightning_count = sum(1 for d in deals if d.deal_type == DealType.LIGHTNING.value)

            result["deals"] = [d.to_dict() for d in deals]
            result["count"] = len(deals)
            result["lightning_count"] = lightning_count
            result["competitor_deals"] = [d.to_dict() for d in competitor_deals]
            result["success"] = True

            logger.info(
                f"Scraped {len(deals)} deals, {lightning_count} lightning, {len(competitor_deals)} competitors"
            )

        except PlaywrightTimeout:
            result["error"] = "TIMEOUT"
            logger.error("Deals page timeout")
        except Exception as e:
            result["error"] = f"PARSE_ERROR: {str(e)}"
            logger.error(f"Deals scraping error: {e}")
        finally:
            if page:
                await page.close()
            if context:
                await context.close()

        return result

    async def _is_blocked(self, page: Page) -> bool:
        """차단 여부 확인"""
        content = await page.content()
        blocked_indicators = [
            "Enter the characters you see below",
            "Sorry, we just need to make sure",
            "Type the characters you see in this image",
            "Robot Check",
        ]
        return any(indicator in content for indicator in blocked_indicators)

    async def _scroll_page(self, page: Page, scroll_count: int = 3) -> None:
        """페이지 스크롤하여 더 많은 콘텐츠 로드"""
        for _i in range(scroll_count):
            await page.evaluate("window.scrollBy(0, window.innerHeight)")
            await asyncio.sleep(1)

    async def _parse_deal_cards(
        self, page: Page, max_items: int, snapshot_time: str
    ) -> list[DealRecord]:
        """딜 카드 파싱"""
        deals = []

        # 딜 카드 선택자들 (Amazon Deals 페이지 구조)
        card_selectors = [
            "[data-testid='deal-card']",
            ".DealCard-module__dealCard",
            ".a-section.octopus-dlp-asin-section",
            "[data-component-type='s-search-result']",
            ".DealGridItem-module__dealItem",
        ]

        for selector in card_selectors:
            cards = await page.query_selector_all(selector)
            if cards:
                logger.info(f"Found {len(cards)} deal cards with selector: {selector}")
                break
        else:
            # 폴백: 일반적인 상품 카드 구조
            cards = await page.query_selector_all("[data-asin]")
            logger.info(f"Fallback: Found {len(cards)} cards with data-asin")

        for i, card in enumerate(cards[:max_items]):
            try:
                deal = await self._parse_single_deal(card, snapshot_time)
                if deal and deal.asin:
                    deals.append(deal)
            except Exception as e:
                logger.debug(f"Failed to parse deal card {i}: {e}")
                continue

        return deals

    async def _parse_single_deal(self, card, snapshot_time: str) -> DealRecord | None:
        """단일 딜 카드 파싱"""
        # ASIN 추출
        asin = await card.get_attribute("data-asin")
        if not asin:
            asin_link = await card.query_selector("a[href*='/dp/']")
            if asin_link:
                href = await asin_link.get_attribute("href")
                asin_match = re.search(r"/dp/([A-Z0-9]{10})", href or "")
                asin = asin_match.group(1) if asin_match else None

        if not asin:
            return None

        # 제품명
        product_name = ""
        name_selectors = [
            "[data-component-type='s-product-title'] span",
            ".a-link-normal span.a-text-normal",
            ".DealContent-module__title",
            "h2 a span",
            ".a-size-base-plus",
        ]
        for selector in name_selectors:
            elem = await card.query_selector(selector)
            if elem:
                product_name = (await elem.inner_text()).strip()
                if product_name:
                    break

        # 브랜드 추출
        brand = self._extract_brand(product_name)

        # 가격 정보
        deal_price = None
        original_price = None

        # 할인가
        price_selectors = [
            ".a-price:not([data-a-strike]) .a-offscreen",
            ".DealPrice-module__dealPrice",
            ".a-color-price .a-offscreen",
        ]
        for selector in price_selectors:
            elem = await card.query_selector(selector)
            if elem:
                price_text = await elem.inner_text()
                deal_price = self._parse_price(price_text)
                if deal_price:
                    break

        # 원가
        orig_selectors = [
            ".a-price[data-a-strike] .a-offscreen",
            ".a-text-price .a-offscreen",
            ".DealPrice-module__originalPrice",
        ]
        for selector in orig_selectors:
            elem = await card.query_selector(selector)
            if elem:
                price_text = await elem.inner_text()
                original_price = self._parse_price(price_text)
                if original_price:
                    break

        # 할인율 계산
        discount_percent = None
        if deal_price and original_price and original_price > deal_price:
            discount_percent = round((1 - deal_price / original_price) * 100, 1)

        # 딜 타입 및 배지
        deal_type = DealType.REGULAR.value
        deal_badge = None
        time_remaining = None
        time_remaining_seconds = None
        claimed_percent = None

        # Lightning Deal 확인
        lightning_elem = await card.query_selector(
            ".Lightning, [data-component-type='lightning-deal']"
        )
        if lightning_elem:
            deal_type = DealType.LIGHTNING.value

        # 딜 배지 확인
        badge_selectors = [
            ".DealBadge-module__badge",
            ".a-badge-text",
            "[data-component-type='s-deal-badge']",
        ]
        for selector in badge_selectors:
            elem = await card.query_selector(selector)
            if elem:
                deal_badge = (await elem.inner_text()).strip()
                if "Lightning" in deal_badge:
                    deal_type = DealType.LIGHTNING.value
                elif "Deal of the Day" in deal_badge:
                    deal_type = DealType.DEAL_OF_THE_DAY.value
                break

        # 남은 시간 (Lightning Deal)
        time_selectors = [".DealTimer-module__timer", ".a-size-mini.a-color-secondary"]
        for selector in time_selectors:
            elem = await card.query_selector(selector)
            if elem:
                time_text = (await elem.inner_text()).strip()
                if re.search(r"\d+[hm]", time_text):
                    time_remaining = time_text
                    time_remaining_seconds = self._parse_time_remaining(time_text)
                    deal_type = DealType.LIGHTNING.value
                    break

        # 판매율 (Claimed %)
        claimed_elem = await card.query_selector(".DealProgressBar-module__text, .a-meter-bar")
        if claimed_elem:
            claimed_text = await claimed_elem.inner_text()
            claimed_match = re.search(r"(\d+)%", claimed_text)
            if claimed_match:
                claimed_percent = int(claimed_match.group(1))

        # 평점
        rating = None
        rating_elem = await card.query_selector(".a-icon-alt, [data-a-popover*='rating']")
        if rating_elem:
            rating_text = await rating_elem.inner_text()
            rating_match = re.search(r"([\d.]+)\s*out of", rating_text)
            if rating_match:
                rating = float(rating_match.group(1))

        # 리뷰 수
        reviews_count = None
        reviews_elem = await card.query_selector(".a-size-small .a-link-normal")
        if reviews_elem:
            reviews_text = await reviews_elem.inner_text()
            reviews_text = reviews_text.replace(",", "")
            reviews_match = re.search(r"([\d,]+)", reviews_text)
            if reviews_match:
                reviews_count = int(reviews_match.group(1).replace(",", ""))

        # 상품 URL
        product_url = None
        link_elem = await card.query_selector("a[href*='/dp/']")
        if link_elem:
            href = await link_elem.get_attribute("href")
            if href:
                product_url = f"https://www.amazon.com{href}" if href.startswith("/") else href

        # 종료 예정 시각 계산
        deal_end_time = None
        if time_remaining_seconds:
            end_dt = datetime.now(KST) + timedelta(seconds=time_remaining_seconds)
            deal_end_time = end_dt.isoformat()

        return DealRecord(
            snapshot_datetime=snapshot_time,
            asin=asin,
            product_name=product_name[:500] if product_name else "",
            brand=brand,
            category="Beauty",  # 뷰티 카테고리 고정 (필터링 후)
            deal_price=deal_price or 0,
            original_price=original_price,
            discount_percent=discount_percent,
            deal_type=deal_type,
            deal_badge=deal_badge,
            time_remaining=time_remaining,
            time_remaining_seconds=time_remaining_seconds,
            claimed_percent=claimed_percent,
            deal_end_time=deal_end_time,
            product_url=product_url,
            rating=rating,
            reviews_count=reviews_count,
        )

    def _extract_brand(self, product_name: str) -> str:
        """제품명에서 브랜드 추출"""
        if not product_name:
            return "Unknown"

        # 알려진 브랜드 먼저 확인
        for brand in self.WATCH_BRANDS:
            if brand.lower() in product_name.lower():
                return brand

        # 첫 단어를 브랜드로 추정
        words = product_name.split()
        return words[0] if words else "Unknown"

    def _parse_price(self, price_text: str) -> float | None:
        """가격 문자열 파싱"""
        if not price_text or "$" not in price_text:
            return None
        try:
            match = re.search(r"\$([\d,]+\.?\d*)", price_text)
            if match:
                price = float(match.group(1).replace(",", ""))
                if 0.50 <= price <= 1000:
                    return price
        except (ValueError, TypeError):
            pass
        return None

    def _parse_time_remaining(self, time_text: str) -> int | None:
        """남은 시간을 초 단위로 변환"""
        try:
            total_seconds = 0

            # 시간 파싱
            hours_match = re.search(r"(\d+)\s*h", time_text)
            if hours_match:
                total_seconds += int(hours_match.group(1)) * 3600

            # 분 파싱
            mins_match = re.search(r"(\d+)\s*m", time_text)
            if mins_match:
                total_seconds += int(mins_match.group(1)) * 60

            # 초 파싱
            secs_match = re.search(r"(\d+)\s*s", time_text)
            if secs_match:
                total_seconds += int(secs_match.group(1))

            return total_seconds if total_seconds > 0 else None
        except Exception:
            # 시간 파싱 실패 시 None 반환
            return None

    def _is_beauty_product(self, deal: DealRecord) -> bool:
        """뷰티 카테고리 제품인지 확인"""
        beauty_keywords = [
            "lip",
            "skin",
            "face",
            "beauty",
            "makeup",
            "cream",
            "serum",
            "moisturizer",
            "cleanser",
            "toner",
            "mask",
            "lotion",
            "powder",
            "foundation",
            "concealer",
            "mascara",
            "eyeliner",
            "lipstick",
            "blush",
            "bronzer",
            "primer",
            "sunscreen",
            "spf",
        ]

        name_lower = deal.product_name.lower()
        return any(kw in name_lower for kw in beauty_keywords)

    async def scrape_competitor_deals(self) -> dict[str, Any]:
        """경쟁사 딜만 수집"""
        result = await self.scrape_deals(beauty_only=True, max_items=100)

        if result["success"]:
            # 경쟁사 브랜드만 필터링
            competitor_deals = result["competitor_deals"]

            # 브랜드별 그룹화
            by_brand = {}
            for deal in competitor_deals:
                brand = deal.get("brand", "Unknown")
                if brand not in by_brand:
                    by_brand[brand] = []
                by_brand[brand].append(deal)

            result["deals_by_brand"] = by_brand

        return result


# =============================================================================
# 싱글톤 인스턴스
# =============================================================================

_deals_scraper: AmazonDealsScraper | None = None


async def get_deals_scraper() -> AmazonDealsScraper:
    """DealsScraper 싱글톤 반환"""
    global _deals_scraper
    if _deals_scraper is None:
        _deals_scraper = AmazonDealsScraper()
        await _deals_scraper.initialize()
    return _deals_scraper


# =============================================================================
# CLI 테스트
# =============================================================================

if __name__ == "__main__":

    async def test():
        scraper = AmazonDealsScraper()
        await scraper.initialize()

        try:
            result = await scraper.scrape_deals(max_items=20, beauty_only=True)
            print("\n=== Deals Scraping Result ===")
            print(f"Success: {result['success']}")
            print(f"Total deals: {result['count']}")
            print(f"Lightning deals: {result['lightning_count']}")
            print(f"Competitor deals: {len(result['competitor_deals'])}")

            if result["deals"]:
                print("\nSample deal:")
                print(json.dumps(result["deals"][0], indent=2, ensure_ascii=False))
        finally:
            await scraper.close()

    asyncio.run(test())
