"""
Amazon Bestseller Scraper
=========================
Playwright 기반 Amazon Top 100 크롤러 (Headless Chromium)

## 크롤링 대상 카테고리 (config/thresholds.json 참조)
```
Beauty & Personal Care (L0)         ← 전체 뷰티 랭킹
├── Skin Care (L1)                  ← 스킨케어
│   └── Lip Care (L2)               ← 립케어 (LANEIGE 핵심)
└── Make Up (L1)
    ├── Lip Makeup (L2)             ← 립메이크업/색조
    └── Face (L2)
        └── Face Powder (L3)        ← 파우더
```

## 데이터 수집 항목
- 제품 정보: ASIN, 제품명, 브랜드, URL
- 순위 정보: rank (1-100), 카테고리별
- 가격 정보: price (현재가), list_price (정가), discount_percent
- 평가 정보: rating (5점 만점), reviews_count
- 프로모션: coupon_text, is_subscribe_save, promo_badges
- 뱃지: Best Seller, Amazon's Choice 등

## 사용 예
```python
scraper = AmazonScraper()
await scraper.initialize()

# 단일 카테고리
result = await scraper.scrape_category("lip_care", url)

# 모든 카테고리 (config 기반)
results = await scraper.scrape_all_categories()

await scraper.close()
```

## 에러 코드
- "BLOCKED": Amazon 차단됨 → IP 변경 또는 대기 필요
- "TIMEOUT": 응답 없음 → 재시도 권장
- "PARSE_ERROR": HTML 구조 변경 → 파서 업데이트 필요

## 주의사항
- KST (UTC+9) 기준 날짜 사용
- 페이지당 50개, 총 2페이지 크롤링 (Top 100)
- 랜덤 User-Agent 및 딜레이로 차단 회피
"""

import asyncio
import json
import logging
import os
import random
import re
from datetime import datetime, timedelta, timezone
from typing import Any

from playwright.async_api import Browser, BrowserContext, Page, async_playwright
from playwright.async_api import TimeoutError as PlaywrightTimeout

# Anti-bot / Stealth mode packages
try:
    from playwright_stealth import stealth_async

    STEALTH_AVAILABLE = True
except ImportError:
    STEALTH_AVAILABLE = False

try:
    from browserforge.fingerprints import FingerprintGenerator
    from browserforge.headers import HeaderGenerator

    BROWSERFORGE_AVAILABLE = True
except ImportError:
    BROWSERFORGE_AVAILABLE = False

try:
    from fake_useragent import UserAgent

    FAKE_UA_AVAILABLE = True
except ImportError:
    FAKE_UA_AVAILABLE = False

logger = logging.getLogger(__name__)

# 한국 시간대 (UTC+9)
KST = timezone(timedelta(hours=9))


class CircuitBreaker:
    """연속 실패 시 자동 중단하는 회로 차단기"""

    def __init__(self, threshold: int = 3, reset_minutes: int = 30):
        self.threshold = threshold
        self.reset_minutes = reset_minutes
        self.failures = 0
        self.last_failure: datetime | None = None
        self.is_open = False

    def record_failure(self) -> None:
        """실패 기록"""
        self.failures += 1
        self.last_failure = datetime.now()

        if self.failures >= self.threshold:
            self.is_open = True
            logger.error(f"Circuit breaker OPEN after {self.failures} consecutive failures")

    def record_success(self) -> None:
        """성공 시 리셋"""
        self.failures = 0
        self.is_open = False

    def can_proceed(self) -> bool:
        """진행 가능 여부 확인"""
        if not self.is_open:
            return True

        # 리셋 시간 확인
        if self.last_failure:
            elapsed = (datetime.now() - self.last_failure).total_seconds() / 60
            if elapsed >= self.reset_minutes:
                self.is_open = False
                self.failures = 0
                logger.info("Circuit breaker reset after timeout")
                return True

        return False

    def get_backoff_seconds(self) -> int:
        """지수 백오프 시간 계산"""
        return min(60 * (2 ** (self.failures - 1)), 600)  # 최대 10분


class AmazonScraper:
    """Amazon 베스트셀러 Top 100 크롤러"""

    def __init__(self, config_path: str = "./config/thresholds.json"):
        """
        Args:
            config_path: 설정 파일 경로
        """
        self.config = self._load_config(config_path)
        self.base_url = "https://www.amazon.com"

        # crawler 설정 로드 (system.crawler 섹션)
        crawler_config = self.config.get("system", {}).get("crawler", {})
        self.delay_base_seconds = crawler_config.get("delay_base_seconds", 8)
        self.delay_random_max = crawler_config.get("delay_random_max", 4)
        self.category_delay_seconds = crawler_config.get("category_delay_seconds", 45)
        self.max_retries = crawler_config.get("max_retries", 3)

        # 환경변수 오버라이드 (호환성 유지)
        self.delay_seconds = float(os.getenv("SCRAPE_DELAY_SECONDS", str(self.delay_base_seconds)))
        self.browser: Browser | None = None

        # Anti-bot 도구 초기화
        self.ua = UserAgent(browsers=["chrome", "firefox", "edge"]) if FAKE_UA_AVAILABLE else None
        self.header_generator = HeaderGenerator() if BROWSERFORGE_AVAILABLE else None
        self.fingerprint_generator = FingerprintGenerator() if BROWSERFORGE_AVAILABLE else None

        # 회로 차단기
        self.circuit_breaker = CircuitBreaker(threshold=3, reset_minutes=30)

        # 브랜드 매핑 resolver (캐시된 ASIN→Brand 조회)
        try:
            from src.tools.utilities.brand_resolver import get_brand_resolver

            self.brand_resolver = get_brand_resolver()
        except ImportError:
            self.brand_resolver = None

    def _load_config(self, config_path: str) -> dict:
        """설정 파일 로드"""
        try:
            with open(config_path, encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            return {"categories": {}}

    async def initialize(self) -> None:
        """브라우저 초기화 (Stealth 모드)"""
        playwright = await async_playwright().start()
        self.browser = await playwright.chromium.launch(
            headless=True,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--no-sandbox",
                "--disable-dev-shm-usage",
                "--disable-infobars",
                "--disable-extensions",
                "--disable-features=IsolateOrigins,site-per-process",
            ],
        )
        logger.info(
            f"Browser initialized (Stealth: {STEALTH_AVAILABLE}, BrowserForge: {BROWSERFORGE_AVAILABLE})"
        )

    async def close(self) -> None:
        """브라우저 종료"""
        if self.browser:
            await self.browser.close()

    async def _create_stealth_context(self) -> BrowserContext:
        """Stealth 모드가 적용된 브라우저 컨텍스트 생성"""
        # 실제 브라우저 핑거프린트 생성
        viewport_width = random.randint(1200, 1920)
        viewport_height = random.randint(800, 1080)

        # User-Agent 선택
        if self.ua:
            user_agent = self.ua.random
        else:
            user_agent = self._get_random_user_agent()

        # 헤더 생성
        extra_headers = {
            "Accept-Language": "en-US,en;q=0.9",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "Cache-Control": "max-age=0",
            "sec-ch-ua": '"Chromium";v="120", "Not A(Brand";v="99"',
            "sec-ch-ua-mobile": "?0",
            "sec-ch-ua-platform": '"Windows"',
        }

        if self.header_generator:
            try:
                generated = self.header_generator.generate(browser="chrome", os="windows")
                if generated:
                    extra_headers.update(
                        {
                            k: v
                            for k, v in generated.items()
                            if k.lower() not in ["host", "content-length", "content-type"]
                        }
                    )
            except Exception as e:
                logger.debug(f"Header generation failed: {e}")

        context = await self.browser.new_context(
            user_agent=user_agent,
            viewport={"width": viewport_width, "height": viewport_height},
            locale="en-US",
            timezone_id="America/New_York",  # 미국 동부 시간대
            geolocation={"longitude": -73.935242, "latitude": 40.730610},  # NYC
            permissions=["geolocation"],
            extra_http_headers=extra_headers,
        )

        return context

    async def _create_stealth_page(self, context: BrowserContext) -> Page:
        """Stealth 모드가 적용된 페이지 생성"""
        page = await context.new_page()

        # Stealth 적용 (navigator.webdriver 제거, HeadlessChrome 숨김 등)
        if STEALTH_AVAILABLE:
            try:
                await stealth_async(page)
            except Exception as e:
                logger.debug(f"Stealth application failed: {e}")

        return page

    async def _simulate_human_behavior(self, page: Page) -> None:
        """인간처럼 행동하는 시뮬레이션"""
        try:
            # 1. 랜덤 스크롤
            scroll_amount = random.randint(100, 500)
            await page.evaluate(f"window.scrollBy(0, {scroll_amount})")
            await asyncio.sleep(random.uniform(0.5, 1.5))

            # 2. 마우스 움직임 (랜덤 좌표)
            x = random.randint(100, 800)
            y = random.randint(100, 600)
            await page.mouse.move(x, y)
            await asyncio.sleep(random.uniform(0.3, 0.8))

            # 3. 랜덤 대기 (읽는 척)
            await asyncio.sleep(random.uniform(1, 3))

        except Exception as e:
            logger.debug(f"Human behavior simulation failed: {e}")

    async def _scroll_to_load_all(self, page: Page) -> None:
        """페이지를 끝까지 스크롤하여 lazy-loaded 제품 카드를 모두 로드"""
        try:
            prev_count = 0
            for i in range(10):
                await page.evaluate(f"window.scrollBy(0, {800 + i * 200})")
                await asyncio.sleep(random.uniform(0.4, 0.8))

                # 현재 로드된 카드 수 확인
                count = await page.evaluate(
                    "document.querySelectorAll('[data-asin]:not([data-asin=\"\"])').length"
                )
                if count >= 50 or (count == prev_count and i >= 3):
                    break
                prev_count = count

            # 마지막에 약간 위로 스크롤 (자연스러운 행동)
            await page.evaluate("window.scrollBy(0, -200)")
            await asyncio.sleep(random.uniform(0.3, 0.6))
        except Exception as e:
            logger.debug(f"Scroll to load all failed: {e}")

    async def _random_delay_advanced(self, delay_type: str = "base") -> None:
        """더 자연스러운 랜덤 딜레이 (안티봇용) - 설정 파일에서 로드"""
        # 설정 파일 값 기반 딜레이 (config/thresholds.json → system.crawler)
        base_delay = self.delay_base_seconds  # 기본 8초
        random_max = self.delay_random_max  # 랜덤 최대 4초
        category_delay = self.category_delay_seconds  # 카테고리 전환 45초

        delays = {
            "base": (base_delay - 3, random_max - 1),  # ~5-8초
            "detail": (base_delay, random_max),  # ~8-12초
            "page": (base_delay + 4, random_max - 1),  # ~12-15초
            "category": (category_delay, 15),  # ~45-60초
        }
        base, jitter = delays.get(delay_type, (base_delay - 3, random_max - 1))

        # 가끔 더 긴 딜레이 (인간처럼) - 10% 확률
        if random.random() < 0.1:
            base *= 2

        delay = base + random.uniform(0, jitter)
        logger.debug(f"Waiting {delay:.1f}s ({delay_type})")
        await asyncio.sleep(delay)

    async def scrape_category(self, category_id: str, category_url: str) -> dict[str, Any]:
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
            "error": None,
        }

        if not self.browser:
            await self.initialize()

        context = await self.browser.new_context(
            user_agent=self._get_random_user_agent(),
            viewport={"width": 1920, "height": 1080},
            locale="en-US",
            extra_http_headers={
                "Accept-Language": "en-US,en;q=0.9",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                "Accept-Encoding": "gzip, deflate, br",
                "Connection": "keep-alive",
                "Upgrade-Insecure-Requests": "1",
                "Cache-Control": "max-age=0",
            },
        )

        page = await context.new_page()

        # 재시도 로직 (최대 3회)
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # 페이지 1 로드 (1-50위)
                await page.goto(category_url, wait_until="domcontentloaded", timeout=30000)
                await self._random_delay()

                # 차단 감지
                if await self._is_blocked(page):
                    if attempt < max_retries - 1:
                        # 재시도 전 긴 대기 (10-20초)
                        wait_time = 10 + random.uniform(5, 10)
                        logger.warning(
                            f"Blocked on {category_id}, retrying in {wait_time:.1f}s (attempt {attempt + 1}/{max_retries})"
                        )
                        await asyncio.sleep(wait_time)
                        continue
                    result["error"] = "BLOCKED"
                    return result
                break  # 성공시 루프 탈출
            except Exception:
                if attempt < max_retries - 1:
                    await asyncio.sleep(5)
                    continue
                raise

        try:
            # 스크롤하여 lazy-loaded 제품 모두 로드 (Amazon은 초기 30개만 표시)
            await self._scroll_to_load_all(page)

            # 첫 페이지 파싱 (1-50위)
            products_page1 = await self._parse_bestseller_page(page, category_id, start_rank=1)
            result["products"].extend(products_page1)
            logger.info(f"Page 1 loaded: {len(products_page1)} products for {category_id}")

            # 페이지 2 로드 (51-100위)
            page2_url = self._get_page2_url(category_url)
            if page2_url:
                await page.goto(page2_url, wait_until="domcontentloaded", timeout=30000)
                await self._random_delay()

                if not await self._is_blocked(page):
                    # 페이지 2도 스크롤하여 lazy-loaded 제품 로드
                    await self._scroll_to_load_all(page)

                    products_page2 = await self._parse_bestseller_page(
                        page, category_id, start_rank=51
                    )
                    result["products"].extend(products_page2)
                    logger.info(f"Page 2 loaded: {len(products_page2)} products for {category_id}")
                else:
                    logger.warning(f"Page 2 blocked for {category_id}")

            result["count"] = len(result["products"])
            result["success"] = True

        except PlaywrightTimeout:
            result["error"] = "TIMEOUT"
        except Exception as e:
            result["error"] = f"PARSE_ERROR: {str(e)}"
        finally:
            await context.close()

        return result

    async def scrape_all_categories(self) -> dict[str, Any]:
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
            "error_count": 0,
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

            # 카테고리 간 딜레이 (5-10초)
            delay = 5 + random.uniform(2, 5)
            await asyncio.sleep(delay)

        return results

    async def _parse_bestseller_page(
        self, page: Page, category_id: str, start_rank: int = 1
    ) -> list[dict]:
        """베스트셀러 페이지 파싱 (순위 배지 기반)"""
        products = []
        snapshot_date = datetime.now(KST).date().isoformat()

        # #zg-right-col 내의 실제 제품 카드만 선택 (광고 제외)
        container = await page.query_selector("#zg-right-col")
        if not container:
            container = page

        product_cards = await container.query_selector_all('[data-asin]:not([data-asin=""])')

        for card in product_cards:
            try:
                asin = await card.get_attribute("data-asin")
                if not asin:
                    continue

                # 순위 배지에서 실제 rank 읽기 (Amazon이 부여한 순위)
                badge = await card.query_selector("span.zg-bdg-text")
                if badge:
                    badge_text = (await badge.inner_text()).strip().replace("#", "")
                    try:
                        rank = int(badge_text)
                    except ValueError:
                        continue
                else:
                    # 배지 없는 카드는 광고/스폰서 → 스킵
                    continue

                product_data = await self._extract_product_data(
                    card, asin, rank, category_id, snapshot_date
                )
                if product_data:
                    products.append(product_data)

            except Exception as e:
                logger.warning(f"Product parsing failed: {str(e)[:100]}")
                continue

        return products

    async def _extract_product_data(
        self, card, asin: str, rank: int, category_id: str, snapshot_date: str
    ) -> dict | None:
        """개별 제품 데이터 추출 (프로모션 정보 포함)"""
        try:
            # 제품명
            name_elem = await card.query_selector(
                ".p13n-sc-truncate, ._cDEzb_p13n-sc-css-line-clamp-3_g3dy1, .a-link-normal span"
            )
            product_name = await name_elem.inner_text() if name_elem else "Unknown"
            product_name = product_name.strip()

            # 브랜드 추출 (ASIN 매핑 우선, 제품명에서 추출 fallback)
            brand = self._extract_brand(product_name, asin=asin)

            # 가격 (현재 판매가) - 더 구체적인 선택자 사용
            price = None
            # Amazon 베스트셀러 페이지의 가격 선택자들 (우선순위 순)
            price_selectors = [
                ".p13n-sc-price",  # 베스트셀러 페이지 기본
                "._cDEzb_p13n-sc-price_3mJ9Z",  # 대체 클래스
                ".a-price:not([data-a-strike]) .a-offscreen",  # 할인가 (취소선 없는 가격)
                ".a-color-price",  # 가격 색상 클래스
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
                # 2026년 Amazon 신규 구조 (사용자 스크린샷 기반)
                "span.a-price.a-text-price span.a-offscreen",  # 취소선 가격
                ".a-text-price .a-offscreen",  # 기존 취소선 가격
                ".a-price[data-a-strike='true'] .a-offscreen",
                "span.a-text-price span.a-offscreen",
                # 상세 페이지 스타일 (베스트셀러에서도 간혹 사용)
                ".basisPrice .a-offscreen",
            ]
            for selector in list_price_selectors:
                list_price_elem = await card.query_selector(selector)
                if list_price_elem:
                    list_price_text = await list_price_elem.inner_text()
                    list_price = self._parse_price(list_price_text)
                    if list_price:
                        break

            # 할인율 직접 추출 (2026년 Amazon 신규 구조)
            discount_percent = None
            discount_selectors = [
                # 2026년 Amazon 신규 구조 (사용자 스크린샷 기반)
                "span.savingsPriceOverride.savingsPercentage",  # "-20%" 형태
                "span.savingsPercentage",  # 대체 패턴
                ".a-color-price .a-text-bold",  # 할인율 텍스트
            ]
            for selector in discount_selectors:
                discount_elem = await card.query_selector(selector)
                if discount_elem:
                    discount_text = await discount_elem.inner_text()
                    # "-20%" 또는 "20%" 형태에서 숫자 추출
                    match = re.search(r"[-−]?(\d+)%", discount_text)
                    if match:
                        discount_percent = float(match.group(1))
                        break

            # 직접 추출 실패 시 가격으로 계산 (fallback)
            if discount_percent is None and price and list_price and list_price > price:
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
                rating_selectors = [".a-icon-star-small", ".a-icon-star-mini", "i.a-icon-star"]
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
                if reviews_text and re.match(r"^[0-9,]+$", reviews_text):
                    reviews_count = self._parse_reviews_count(reviews_text)
                    if reviews_count:
                        break

            # 방법 2: 링크 안에 있는 경우 (대체 패턴)
            if not reviews_count:
                reviews_selectors = [".a-size-small .a-link-normal", "a.a-size-small"]
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
            coupon_elem = await card.query_selector(
                ".s-coupon-highlight-color, .couponBadge, [data-component-type='s-coupon-component']"
            )
            if coupon_elem:
                coupon_text = (await coupon_elem.inner_text()).strip()

            # Subscribe & Save 여부
            is_subscribe_save = False
            sns_elem = await card.query_selector(
                ".s-subscriptions-terms, [data-component-type='s-subscribe-and-save']"
            )
            if sns_elem:
                is_subscribe_save = True
            # 텍스트로도 확인
            card_html = await card.inner_html()
            if "Subscribe & Save" in card_html or "Subscribe &amp; Save" in card_html:
                is_subscribe_save = True

            # 프로모션 배지들 (Limited Time Deal, Lightning Deal 등)
            promo_badges = []

            # Limited Time Deal
            ltd_elem = await card.query_selector(
                ".a-badge-limited-time-deal, [data-badge='limited-time-deal']"
            )
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
                "product_url": product_url or f"{self.base_url}/dp/{asin}",
            }

        except Exception:
            return None

    # 브랜드명 정규화 매핑 (부분 매칭/대소문자 불일치 교정)
    BRAND_NORMALIZATION = {
        # 잘린 브랜드명 → 전체 브랜드명
        "burt's": "Burt's Bees",
        "wet": "wet n wild",
        "tree": "Tree Hut",
        "clean": "Clean Skin Club",
        "physicians": "Physicians Formula",
        "mighty": "Mighty Patch",
        "hero": "Hero Cosmetics",
        "sol": "Sol de Janeiro",
        "grace": "grace & stella",
        "mrs.": "Mrs. Meyer's",
        "dr.": "Dr.Melaxin",
        "amazon": "Amazon Basics",
        # 대소문자 통일
        "covergirl": "COVERGIRL",
        "medicube": "MEDICUBE",
        "biodance": "BIODANCE",
        "laneige": "LANEIGE",
        "cosrx": "COSRX",
        "tirtir": "TIRTIR",
        "anua": "ANUA",
        "carslan": "CARSLAN",
        "eos": "eos",  # 소문자가 공식
    }

    def _normalize_brand(self, brand: str) -> str:
        """브랜드명 정규화 - 대소문자 통일 및 잘린 이름 교정"""
        if not brand or brand == "Unknown":
            return brand

        brand_lower = brand.lower().strip()

        # 정규화 매핑에서 찾기
        if brand_lower in self.BRAND_NORMALIZATION:
            return self.BRAND_NORMALIZATION[brand_lower]

        return brand

    def _extract_brand(self, product_name: str, asin: str = None) -> str:
        """제품명에서 브랜드 추출

        우선순위:
        1. ASIN→Brand 매핑 테이블 조회 (캐시)
        2. 하드코딩된 브랜드 리스트 매칭
        3. Unknown 반환 (나중에 배치 검증)
        """
        # 1. ASIN 매핑 테이블에서 캐시된 브랜드 조회
        if asin and self.brand_resolver:
            cached_brand = self.brand_resolver.get_brand(asin)
            if cached_brand:
                return self._normalize_brand(cached_brand)

        # 2. 두 단어 이상 브랜드 먼저 체크 (순서 중요!)
        # Amazon Top 100 Beauty 기준 브랜드 목록 (2024-2025 기준)
        multi_word_brands = [
            # === K-Beauty ===
            "Beauty of Joseon",
            "SKIN1004",
            "Thank You Farmer",
            "I'm From",
            "I DEW CARE",
            "Some By Mi",
            "By Wishtrend",
            "Dear Klairs",
            "Sulwhasoo",
            "Amore Pacific",
            # === Premium Skincare ===
            "La Roche-Posay",
            "Drunk Elephant",
            "Paula's Choice",
            "The Ordinary",
            "Glow Recipe",
            "Youth To The People",
            "Peter Thomas Roth",
            "Sunday Riley",
            "First Aid Beauty",
            "Sol de Janeiro",
            "Clean Skin Club",
            "Hero Cosmetics",
            "Summer Fridays",
            "Rare Beauty",
            "Tower 28",
            # === Classic Luxury ===
            "Kiehl's",
            "Tatcha",
            "Fresh",
            "Olehenriksen",
            "Origins",
            "Clinique",
            "Estee Lauder",
            "Lancome",
            "Shiseido",
            "SK-II",
            "Elizabeth Arden",
            "Clarins",
            "Dermalogica",
            # === Makeup Premium ===
            "Charlotte Tilbury",
            "Too Faced",
            "Urban Decay",
            "Fenty Beauty",
            "Huda Beauty",
            "Anastasia Beverly Hills",
            "Benefit Cosmetics",
            "MAC Cosmetics",
            "IT Cosmetics",
            "Bobbi Brown",
            "Laura Mercier",
            "Bare Minerals",
            "Kylie Cosmetics",
            "KVD Vegan Beauty",
            # === Sunscreen/Derm ===
            "EltaMD",
            "Supergoop",
            "La Roche Posay",
            "Blue Lizard",
            "Banana Boat",
            "Sun Bum",
            "Australian Gold",
            # === Hair Care ===
            "Mielle Organics",
            "Olaplex",
            "Moroccanoil",
            "Aussie",
            "OGX",
            "Not Your Mother's",
            "Bed Head",
            "Living Proof",
            "Briogeo",
            "Verb",
            "Color Wow",
            "Pureology",
            # === Body Care ===
            "Tree Hut",
            "Necessaire",
            "Soap & Glory",
            "Jergens",
            "Gold Bond",
            "Eucerin",
            "Aquaphor",
            "Lubriderm",
            # === Fragrance ===
            "Bath & Body Works",
            "Victoria's Secret",
            # === Misc ===
            "NARS",
            "Burt's Bees",
            "e.l.f. Cosmetics",
        ]

        for brand in multi_word_brands:
            if brand.lower() in product_name.lower():
                return self._normalize_brand(brand)

        # 단일 단어 브랜드 (대소문자 구분 없이 매칭)
        single_word_brands = [
            # === K-Beauty ===
            "LANEIGE",
            "Laneige",
            "COSRX",
            "TIRTIR",
            "Anua",
            "ANUA",
            "BIODANCE",
            "Innisfree",
            "MISSHA",
            "ETUDE",
            "SKINFOOD",
            "Benton",
            "Purito",
            "Klairs",
            "Heimish",
            "Isntree",
            "Rovectin",
            "Torriden",
            "mixsoon",
            "Numbuzin",
            "Haruharu",
            "Neogen",
            "Mediheal",
            "Banila",
            "Holika",
            "Peripera",
            "Romand",
            "Espoir",
            "Clio",
            "Moonshot",
            "Hera",
            "Sulwhasoo",
            "MEDICUBE",
            "medicube",
            "SKIN1004",
            "Abib",
            "Round Lab",
            # === Drugstore US ===
            "CeraVe",
            "Neutrogena",
            "Cetaphil",
            "Aveeno",
            "Olay",
            "Garnier",
            "Nivea",
            "Vaseline",
            "Dove",
            "Pond's",
            "Differin",
            "Bioderma",
            "Vichy",
            "Vanicream",
            # === Makeup Drugstore ===
            "e.l.f.",
            "elf",
            "NYX",
            "Maybelline",
            "L'Oreal",
            "Loreal",
            "Revlon",
            "Covergirl",
            "Milani",
            "ColourPop",
            "Morphe",
            "Wet n Wild",
            "Almay",
            "Rimmel",
            "Physicians Formula",
            "Essence",
            "Catrice",
            "Makeup Revolution",
            # === Makeup Premium ===
            "Tarte",
            "Smashbox",
            "Hourglass",
            "Glossier",
            "Nars",
            "Stila",
            "Becca",
            "Jouer",
            "Natasha Denona",
            # === Skincare Prestige ===
            "TruSkin",
            "InstaNatural",
            "Murad",
            "Obagi",
            "SkinMedica",
            "iS Clinical",
            "SkinCeuticals",
            "Alastin",
            "ZO Skin",
            # === Hair ===
            "Olaplex",
            "Nizoral",
            "Mielle",
            "OGX",
            "TRESemme",
            "Pantene",
            "Herbal Essences",
            "Aussie",
            "Suave",
            "Head & Shoulders",
            "Redken",
            "Kerastase",
            "Aveda",
            "CHI",
            "Kenra",
            "Joico",
            "Sebastian",
            "Matrix",
            "Nioxin",
            "Nutrafol",
            "Vegamour",
            "Rogaine",
            # === Body/Bath ===
            "Jergens",
            "Eucerin",
            "Aquaphor",
            "Lubriderm",
            "Palmer's",
            "Aveeno",
            "Curel",
            "Vanicream",
            "Lottabody",
            "Cantu",
            "SheaMoisture",
            # === Lip Care ===
            "Aquaphor",
            "Blistex",
            "Chapstick",
            "Carmex",
            "Burt's Bees",
            "eos",
            "Laneige",
            "Tatcha",
            # === Tools/Accessories ===
            "Revlon",
            "Conair",
            "Remington",
            "Hot Tools",
            "BaByliss",
            "T3",
            "Dyson",
            "GHD",
            # === Nails ===
            "OPI",
            "Essie",
            "Sally Hansen",
            "ORLY",
            "Zoya",
            # === Men's ===
            "Gillette",
            "Nivea",
            "Jack Black",
            "Bulldog",
            "Duke Cannon",
            "Every Man Jack",
            "Cremo",
            # === Specialty ===
            "Sacheu",
            "Patchology",
            "Starface",
            "ZitSticka",
            "Peace Out",
            "Bliss",
            "Mario Badescu",
            "Origins",
        ]

        for brand in single_word_brands:
            if brand.lower() in product_name.lower():
                return self._normalize_brand(brand)

        # Fallback: 브랜드를 특정할 수 없으면 "Unknown" 반환
        # 기존에 words[0]을 반환했으나, "Summer Fridays" -> "Summer" 버그 발생
        return "Unknown"

    def _parse_price(self, price_text: str) -> float | None:
        """가격 문자열 파싱 - $ 기호가 있는 가격만 인식"""
        try:
            if not price_text:
                return None

            # $ 기호가 반드시 있어야 함 (리뷰 수 등 다른 숫자와 구분)
            if "$" not in price_text:
                return None

            # "$24.00" -> 24.00, "$1,234.56" -> 1234.56
            match = re.search(r"\$([\d,]+\.?\d*)", price_text)
            if match:
                price = float(match.group(1).replace(",", ""))
                # 합리적인 가격 범위 검증 (뷰티 제품: $0.50 ~ $500)
                if 0.50 <= price <= 500:
                    return price
        except (ValueError, TypeError, AttributeError):
            pass
        return None

    def _parse_rating(self, rating_text: str) -> float | None:
        """평점 문자열 파싱 - 5점 만점 평점만 인식"""
        try:
            if not rating_text:
                return None

            # "4.7 out of 5 stars" 또는 "4.7 out of 5" 패턴
            match = re.search(r"([0-9.]+)\s*out of\s*5", rating_text, re.IGNORECASE)
            if match:
                rating = float(match.group(1))
                # 평점 범위 검증 (0.0 ~ 5.0)
                if 0.0 <= rating <= 5.0:
                    return round(rating, 1)

            # "4.7/5" 패턴
            match = re.search(r"([0-9.]+)\s*/\s*5", rating_text)
            if match:
                rating = float(match.group(1))
                if 0.0 <= rating <= 5.0:
                    return round(rating, 1)

            # Fallback: 문자열 시작 부분에서 숫자 추출 (예: "4.8 out of 5 stars")
            # parseFloat 스타일 파싱
            first_word = rating_text.strip().split()[0] if rating_text.strip() else ""
            if first_word and re.match(r"^[0-9.]+$", first_word):
                rating = float(first_word)
                if 0.0 <= rating <= 5.0:
                    return round(rating, 1)

        except (ValueError, TypeError, AttributeError):
            pass
        return None

    def _parse_reviews_count(self, reviews_text: str) -> int | None:
        """리뷰 수 문자열 파싱 - 합리적인 범위 검증"""
        try:
            if not reviews_text:
                return None

            # 콤마 제거하고 숫자 추출: "89,234" -> 89234
            clean_text = reviews_text.replace(",", "").replace("+", "").strip()
            match = re.search(r"^(\d+)$", clean_text)
            if match:
                count = int(match.group(1))
                # 합리적인 리뷰 수 범위 (0 ~ 1,000,000)
                if 0 <= count <= 1000000:
                    return count
        except (ValueError, TypeError, AttributeError):
            pass
        return None

    def _get_page2_url(self, url: str) -> str | None:
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
            "api-services-support@amazon.com",
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
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Safari/605.1.15",
        ]
        return random.choice(user_agents)

    async def scrape_product_by_asin(
        self, asin: str, metadata: dict | None = None
    ) -> dict[str, Any] | None:
        """
        개별 ASIN으로 상품 상세 페이지 크롤링 (할인 정보 포함)

        Args:
            asin: Amazon ASIN
            metadata: 추가 메타데이터 (brand, category, product_type 등)

        Returns:
            상품 정보 딕셔너리 (할인 정보 포함) 또는 None
        """
        # 버그 수정: _initialized → self.browser
        if not self.browser:
            await self.initialize()

        # 회로 차단기 확인
        if not self.circuit_breaker.can_proceed():
            logger.warning(f"Circuit breaker is OPEN - skipping ASIN {asin}")
            return None

        url = f"https://www.amazon.com/dp/{asin}"
        today = datetime.now(timezone(timedelta(hours=9))).strftime("%Y-%m-%d")

        context = None
        try:
            # Stealth 컨텍스트 생성
            context = await self._create_stealth_context()
            page = await self._create_stealth_page(context)

            # 인간 행동 시뮬레이션 + 딜레이
            await self._random_delay_advanced("detail")
            await page.goto(url, wait_until="domcontentloaded", timeout=30000)
            await self._simulate_human_behavior(page)

            # 차단 확인
            if await self._is_blocked(page):
                self.circuit_breaker.record_failure()
                logger.warning(f"Blocked while scraping ASIN: {asin}")
                await context.close()
                return None

            self.circuit_breaker.record_success()

            # 상품 정보 추출 (할인 정보 포함)
            product_data = await page.evaluate("""() => {
                const data = {};

                // 제품명
                const titleEl = document.querySelector('#productTitle');
                data.product_name = titleEl ? titleEl.textContent.trim() : '';

                // 브랜드 (bylineInfo에서 추출)
                const brandEl = document.querySelector('#bylineInfo');
                if (brandEl) {
                    const brandText = brandEl.textContent.trim();
                    const visitMatch = brandText.match(/Visit the (.+?) Store/);
                    const brandMatch = brandText.match(/Brand:\\s*(.+)/);
                    data.brand = visitMatch ? visitMatch[1] : (brandMatch ? brandMatch[1].trim() : brandText);
                } else {
                    data.brand = '';
                }

                // 현재가
                const priceSelectors = [
                    '#corePrice_feature_div .a-price .a-offscreen',
                    '.a-price .a-offscreen',
                    '#priceblock_ourprice',
                    '#priceblock_dealprice'
                ];
                for (const sel of priceSelectors) {
                    const el = document.querySelector(sel);
                    if (el && el.textContent.includes('$')) {
                        data.price_text = el.textContent.trim();
                        break;
                    }
                }

                // === 할인 정보 (핵심 추가) ===

                // 정가 (취소선 가격) - 2026년 Amazon 신규 구조 반영
                const listPriceSelectors = [
                    // 2026년 Amazon 신규 구조
                    'span.a-price.a-text-price span.a-offscreen',
                    '.a-text-price .a-offscreen',
                    // 기존 상세 페이지 선택자
                    '.basisPrice .a-offscreen',
                    '#listPrice',
                    '.a-text-price[data-a-strike] .a-offscreen',
                    'span[data-a-strike="true"] .a-offscreen',
                    '#priceblock_listprice'
                ];
                for (const sel of listPriceSelectors) {
                    const el = document.querySelector(sel);
                    if (el && el.textContent.includes('$')) {
                        data.list_price_text = el.textContent.trim();
                        break;
                    }
                }

                // 할인율 (직접 표시) - 2026년 Amazon 신규 구조 반영
                const savingsSelectors = [
                    '#savingsPercentage',  // 기존 상세 페이지
                    'span.savingsPriceOverride.savingsPercentage',  // 2026년 신규 구조
                    'span.savingsPercentage',  // 대체 패턴
                    '.a-color-price .a-text-bold'  // 할인율 텍스트
                ];
                data.savings_percent = '';
                for (const sel of savingsSelectors) {
                    const el = document.querySelector(sel);
                    if (el && el.textContent.includes('%')) {
                        data.savings_percent = el.textContent.trim();
                        break;
                    }
                }

                // 쿠폰
                const couponSelectors = [
                    '#promoPriceBlockMessage_feature_div .a-color-success',
                    '#vpcButton',
                    '.couponBadge',
                    '[data-csa-c-content-id="coupon"]'
                ];
                for (const sel of couponSelectors) {
                    const el = document.querySelector(sel);
                    if (el && el.textContent.toLowerCase().includes('coupon')) {
                        data.coupon_text = el.textContent.trim();
                        break;
                    }
                }

                // Subscribe & Save
                const snsEl = document.querySelector('#sns-base-price, #snsPrice, .sns-price');
                data.has_subscribe_save = !!snsEl;

                // 프로모션 배지
                const badges = [];
                if (document.querySelector('.dealBadge')) badges.push('Deal');
                if (document.querySelector('.a-badge-limited-time-deal')) badges.push('Limited Time Deal');
                if (document.body.textContent.includes('Lightning Deal')) badges.push('Lightning Deal');
                if (document.body.textContent.includes('Prime Day')) badges.push('Prime Day Deal');
                data.promo_badges = badges.join(', ');

                // 평점
                const ratingEl = document.querySelector('#acrPopover');
                data.rating_text = ratingEl ? (ratingEl.getAttribute('title') || ratingEl.textContent.trim()) : '';

                // 리뷰 수
                const reviewsEl = document.querySelector('#acrCustomerReviewText');
                data.reviews_text = reviewsEl ? reviewsEl.textContent.trim() : '';

                // 재고 상태
                const availEl = document.querySelector('#availability span');
                data.availability = availEl ? availEl.textContent.trim() : '';

                // 이미지 URL
                const imgEl = document.querySelector('#landingImage');
                data.image_url = imgEl ? imgEl.getAttribute('src') : '';

                return data;
            }""")

            await context.close()

            if not product_data.get("product_name"):
                logger.warning(f"Failed to parse product page for ASIN: {asin}")
                return None

            # 메타데이터 병합
            meta = metadata or {}

            # 할인율 파싱
            discount_percent = None
            if product_data.get("savings_percent"):
                match = re.search(r"(\d+)%", product_data["savings_percent"])
                if match:
                    discount_percent = float(match.group(1))

            result = {
                "snapshot_date": today,
                "asin": asin,
                "product_name": product_data.get("product_name", ""),
                "brand": self._normalize_brand(meta.get("brand") or "")
                or self._normalize_brand(product_data.get("brand") or "")
                or self._extract_brand(product_data.get("product_name", ""), asin=asin),
                "price": self._parse_price(product_data.get("price_text", "")),
                "list_price": self._parse_price(product_data.get("list_price_text", "")),
                "discount_percent": discount_percent,
                "coupon_text": product_data.get("coupon_text", ""),
                "is_subscribe_save": product_data.get("has_subscribe_save", False),
                "promo_badges": product_data.get("promo_badges", ""),
                "rating": self._parse_rating(product_data.get("rating_text", "")),
                "reviews_count": self._parse_reviews_count(
                    product_data.get("reviews_text", "")
                    .replace(" ratings", "")
                    .replace(" rating", "")
                ),
                "availability": product_data.get("availability", ""),
                "image_url": product_data.get("image_url", ""),
                "product_url": url,
                "category_id": meta.get("category", ""),
                "product_type": meta.get("product_type", ""),
                "laneige_competitor": meta.get("laneige_competitor", ""),
                "source": "detail_page",
            }

            logger.debug(
                f"Scraped ASIN {asin}: list_price={result.get('list_price')}, coupon={result.get('coupon_text')}"
            )
            return result

        except Exception as e:
            self.circuit_breaker.record_failure()
            logger.error(f"Error scraping ASIN {asin}: {e}")
            if context:
                await context.close()
            return None

    async def scrape_category_with_details(
        self, category_id: str, category_url: str
    ) -> dict[str, Any]:
        """
        베스트셀러 + 상세 페이지 통합 크롤링 (할인 정보 포함)

        1. 베스트셀러 목록에서 Top 100 수집
        2. 각 제품의 상세 페이지에서 할인 정보 수집

        Args:
            category_id: 카테고리 ID
            category_url: Amazon 베스트셀러 URL

        Returns:
            할인 정보가 포함된 제품 목록
        """
        # 회로 차단기 확인
        if not self.circuit_breaker.can_proceed():
            logger.warning("Circuit breaker is OPEN - skipping category crawl")
            return {"products": [], "success": False, "error": "CIRCUIT_BREAKER_OPEN"}

        # 1. 베스트셀러 목록 크롤링
        result = await self.scrape_category(category_id, category_url)

        if not result["success"]:
            return result

        # 2. 모든 제품의 상세 페이지 크롤링 (천천히, 안전하게)
        product_count = len(result["products"])
        estimated_minutes = (product_count * 10) // 60
        logger.info(
            f"Fetching details for {product_count} products (estimated: ~{estimated_minutes} minutes)..."
        )

        for i, product in enumerate(result["products"]):
            # 회로 차단기 확인
            if not self.circuit_breaker.can_proceed():
                logger.warning(f"Circuit breaker opened at product {i + 1} - stopping")
                break

            asin = product.get("asin")
            if not asin:
                continue

            # 상세 페이지 크롤링
            detail = await self.scrape_product_by_asin(asin)

            if detail:
                # 할인 정보 병합 (상세 페이지에서만 얻을 수 있는 정보)
                product["list_price"] = detail.get("list_price") or product.get("list_price")
                product["discount_percent"] = detail.get("discount_percent") or product.get(
                    "discount_percent"
                )
                product["coupon_text"] = detail.get("coupon_text") or product.get("coupon_text")
                product["promo_badges"] = detail.get("promo_badges") or product.get("promo_badges")
                product["is_subscribe_save"] = detail.get("is_subscribe_save") or product.get(
                    "is_subscribe_save"
                )

            # 진행률 로깅 (10개마다)
            if (i + 1) % 10 == 0:
                logger.info(
                    f"Progress: {i + 1}/{product_count} ({(i + 1) / product_count * 100:.1f}%)"
                )

        return result

    async def scrape_competitor_products(
        self, competitor_config: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """
        경쟁사 제품 목록 크롤링

        Args:
            competitor_config: tracked_competitors.json의 경쟁사 설정

        Returns:
            경쟁사 제품 정보 리스트
        """
        results = []
        brand_name = competitor_config.get("brand_name", "Unknown")
        products = competitor_config.get("products", [])

        logger.info(f"Scraping {len(products)} products for competitor: {brand_name}")

        for product in products:
            asin = product.get("asin")
            if not asin:
                continue

            metadata = {
                "brand": brand_name,
                "category": product.get("category", ""),
                "product_type": product.get("product_type", ""),
                "laneige_competitor": product.get("laneige_competitor", ""),
            }

            product_data = await self.scrape_product_by_asin(asin, metadata)
            if product_data:
                results.append(product_data)

            # 요청 간 딜레이
            await self._random_delay()

        logger.info(f"Completed scraping {len(results)}/{len(products)} products for {brand_name}")
        return results


# 편의 함수
async def scrape_bestsellers(category_url: str, category_id: str = "unknown") -> dict[str, Any]:
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
