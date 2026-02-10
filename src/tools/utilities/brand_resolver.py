"""
Brand Resolver
==============
ASIN → Brand 매핑 관리 및 자동 검증

## 사용법
```python
resolver = BrandResolver()

# 1. 캐시된 브랜드 조회 (크롤링 시)
brand = resolver.get_brand("B0C42HJRBF")

# 2. Unknown 브랜드 일괄 검증 (크롤링 후)
verified_count = await resolver.verify_unknown_brands(products)
```

## 검증 방식
1. Amazon 상세 페이지에서 bylineInfo 추출 (우선)
2. 실패 시 웹검색 fallback
"""

import asyncio
import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class BrandResolver:
    """ASIN → Brand 매핑 관리 및 자동 검증"""

    def __init__(self, mapping_path: str = "./config/asin_brand_mapping.json"):
        """
        Args:
            mapping_path: 브랜드 매핑 JSON 파일 경로
        """
        self.mapping_path = Path(mapping_path)
        self.mappings: dict[str, dict[str, Any]] = {}
        self._load_mappings()

    def _load_mappings(self) -> None:
        """매핑 파일 로드"""
        if self.mapping_path.exists():
            try:
                with open(self.mapping_path, encoding="utf-8") as f:
                    data = json.load(f)
                    self.mappings = data.get("mappings", {})
                    logger.info(f"Loaded {len(self.mappings)} brand mappings")
            except Exception as e:
                logger.error(f"Failed to load brand mappings: {e}")
                self.mappings = {}
        else:
            logger.warning(f"Brand mapping file not found: {self.mapping_path}")
            self.mappings = {}

    def _save_mappings(self) -> None:
        """매핑 파일 저장"""
        try:
            # 디렉토리 생성
            self.mapping_path.parent.mkdir(parents=True, exist_ok=True)

            data = {
                "version": "1.0",
                "updated_at": datetime.now().isoformat(),
                "description": "ASIN → Brand 검증된 매핑",
                "mappings": self.mappings,
            }

            with open(self.mapping_path, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

            logger.info(f"Saved {len(self.mappings)} brand mappings")
        except Exception as e:
            logger.error(f"Failed to save brand mappings: {e}")

    def get_brand(self, asin: str) -> str | None:
        """
        매핑 테이블에서 브랜드 조회 (O(1))

        Args:
            asin: Amazon 제품 ID

        Returns:
            브랜드명 또는 None (매핑 없음)
        """
        if asin in self.mappings:
            return self.mappings[asin].get("brand")
        return None

    def add_mapping(
        self, asin: str, brand: str, product_name: str = "", auto_save: bool = True
    ) -> None:
        """
        브랜드 매핑 추가

        Args:
            asin: Amazon 제품 ID
            brand: 확정된 브랜드명
            product_name: 제품명 (참고용)
            auto_save: 자동 저장 여부
        """
        self.mappings[asin] = {
            "brand": brand,
            "verified": True,
            "verified_date": datetime.now().strftime("%Y-%m-%d"),
            "product_name": product_name,
        }

        if auto_save:
            self._save_mappings()

    def add_mappings_batch(self, mappings: list[dict[str, str]]) -> int:
        """
        브랜드 매핑 일괄 추가

        Args:
            mappings: [{"asin": "...", "brand": "...", "product_name": "..."}] 리스트

        Returns:
            추가된 매핑 개수
        """
        count = 0
        for item in mappings:
            asin = item.get("asin")
            brand = item.get("brand")
            if asin and brand:
                self.add_mapping(
                    asin=asin,
                    brand=brand,
                    product_name=item.get("product_name", ""),
                    auto_save=False,
                )
                count += 1

        if count > 0:
            self._save_mappings()

        return count

    async def verify_unknown_brands(
        self,
        products: list[dict[str, Any]],
        use_amazon: bool = True,
        use_websearch: bool = True,
        delay_seconds: float = 2.0,
    ) -> dict[str, Any]:
        """
        Unknown 브랜드 일괄 검증 (크롤링 후 호출)

        Args:
            products: 제품 리스트 [{"asin": "...", "brand": "...", "product_name": "..."}]
            use_amazon: Amazon 상세 페이지 검색 사용
            use_websearch: 웹검색 fallback 사용
            delay_seconds: 요청 간 딜레이

        Returns:
            {
                "verified_count": 10,
                "failed_count": 2,
                "skipped_count": 5,
                "results": [{"asin": "...", "brand": "...", "source": "..."}]
            }
        """
        # Unknown 브랜드 중 매핑에 없는 것만 필터링
        unknown_products = [
            p
            for p in products
            if p.get("brand") == "Unknown" and p.get("asin") not in self.mappings
        ]

        logger.info(f"Found {len(unknown_products)} unknown brands to verify")

        results = []
        verified_count = 0
        failed_count = 0

        for product in unknown_products:
            asin = product.get("asin")
            product_name = product.get("product_name", "")

            brand = None
            source = None

            # 1. Amazon 상세 페이지에서 브랜드 추출 시도
            if use_amazon and not brand:
                brand = await self._fetch_brand_from_amazon(asin)
                if brand:
                    source = "amazon_detail_page"

            # 2. 웹검색 fallback
            if use_websearch and not brand:
                brand = await self._search_brand_web(product_name)
                if brand:
                    source = "web_search"

            # 결과 처리
            if brand and brand != "Unknown":
                self.add_mapping(asin, brand, product_name, auto_save=False)
                verified_count += 1
                results.append(
                    {"asin": asin, "brand": brand, "source": source, "product_name": product_name}
                )
                logger.info(f"Verified brand: {asin} → {brand} (via {source})")
            else:
                failed_count += 1
                logger.warning(f"Failed to verify brand: {asin} - {product_name[:50]}")

            # 딜레이
            if delay_seconds > 0:
                await asyncio.sleep(delay_seconds)

        # 일괄 저장
        if verified_count > 0:
            self._save_mappings()

        return {
            "verified_count": verified_count,
            "failed_count": failed_count,
            "skipped_count": len(products) - len(unknown_products),
            "results": results,
        }

    async def _fetch_brand_from_amazon(self, asin: str) -> str | None:
        """
        Amazon 상세 페이지에서 브랜드 추출

        Args:
            asin: Amazon 제품 ID

        Returns:
            브랜드명 또는 None
        """
        try:
            from playwright.async_api import async_playwright

            url = f"https://www.amazon.com/dp/{asin}"

            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                context = await browser.new_context(
                    user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
                )
                page = await context.new_page()

                try:
                    await page.goto(url, timeout=30000)
                    await page.wait_for_load_state("domcontentloaded")

                    # bylineInfo에서 브랜드 추출
                    brand_selectors = [
                        "#bylineInfo",
                        "a#bylineInfo",
                        "#bylineInfo_feature_div a",
                        ".po-brand .a-span9 span",
                    ]

                    for selector in brand_selectors:
                        try:
                            element = await page.query_selector(selector)
                            if element:
                                text = await element.inner_text()
                                # "Brand: XXX" 또는 "Visit the XXX Store" 패턴 처리
                                brand = self._extract_brand_from_text(text)
                                if brand:
                                    return brand
                        except Exception:
                            continue

                except Exception as e:
                    logger.debug(f"Failed to fetch Amazon page for {asin}: {e}")

                finally:
                    await browser.close()

        except ImportError:
            logger.warning("Playwright not available for Amazon brand fetch")
        except Exception as e:
            logger.error(f"Error fetching brand from Amazon: {e}")

        return None

    def _extract_brand_from_text(self, text: str) -> str | None:
        """Amazon 페이지 텍스트에서 브랜드 추출"""
        if not text:
            return None

        text = text.strip()

        # "Visit the XXX Store" 패턴
        match = re.search(r"Visit the (.+?) Store", text, re.IGNORECASE)
        if match:
            return match.group(1).strip()

        # "Brand: XXX" 패턴
        match = re.search(r"Brand:\s*(.+)", text, re.IGNORECASE)
        if match:
            return match.group(1).strip()

        # 링크 텍스트 자체가 브랜드인 경우
        if len(text) < 50 and not text.startswith("Visit"):
            return text

        return None

    async def _search_brand_web(self, product_name: str) -> str | None:
        """
        웹검색으로 브랜드 확인

        Args:
            product_name: 제품명

        Returns:
            브랜드명 또는 None
        """
        try:
            # 제품명에서 첫 2-3 단어로 검색 쿼리 생성
            words = product_name.split()[:3]
            query = " ".join(words) + " brand"

            # TODO: 실제 웹검색 API 연동
            # 현재는 placeholder - 향후 WebSearch 도구 연동
            logger.debug(f"Web search query: {query}")

            # 간단한 패턴 매칭으로 브랜드 추정
            known_patterns = {
                "summer fridays": "Summer Fridays",
                "rare beauty": "Rare Beauty",
                "beauty of joseon": "Beauty of Joseon",
                "la roche": "La Roche-Posay",
                "drunk elephant": "Drunk Elephant",
                "paula's choice": "Paula's Choice",
                "the ordinary": "The Ordinary",
                "glow recipe": "Glow Recipe",
                "tower 28": "Tower 28",
                "first aid beauty": "First Aid Beauty",
                "it cosmetics": "IT Cosmetics",
                "too faced": "Too Faced",
                "fenty beauty": "Fenty Beauty",
                "huda beauty": "Huda Beauty",
            }

            product_lower = product_name.lower()
            for pattern, brand in known_patterns.items():
                if pattern in product_lower:
                    return brand

        except Exception as e:
            logger.error(f"Error in web search: {e}")

        return None

    async def extract_brand_with_llm(self, product_name: str, asin: str = None) -> str | None:
        """
        LLM을 사용하여 제품명에서 브랜드 추출 (Unknown 브랜드 검증용)

        Args:
            product_name: Amazon 제품명
            asin: 제품 ASIN (캐싱용)

        Returns:
            추출된 브랜드명 또는 None
        """
        try:
            from litellm import acompletion

            prompt = f"""Amazon 제품명에서 브랜드를 추출하세요.

제품명: {product_name}

규칙:
1. 제품명 앞부분에 브랜드가 있는 경우가 대부분입니다
2. 브랜드가 명확하지 않으면 "Unknown" 반환
3. 일반 단어(예: Hydrating, Professional)는 브랜드가 아닙니다
4. Amazon Basics, Amazon 등은 브랜드입니다

JSON 형식으로만 응답: {{"brand": "브랜드명"}}"""

            response = await acompletion(
                model="gpt-4.1-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=50,
                temperature=0,
            )

            # 응답 파싱
            content = response.choices[0].message.content.strip()

            # JSON 추출
            import json

            # JSON 블록 찾기
            if "{" in content and "}" in content:
                json_str = content[content.find("{") : content.rfind("}") + 1]
                result = json.loads(json_str)
                brand = result.get("brand", "").strip()

                if brand and brand != "Unknown" and len(brand) < 50:
                    logger.info(f"LLM extracted brand: '{brand}' from '{product_name[:50]}...'")

                    # 캐싱
                    if asin:
                        self.add_mapping(asin, brand, product_name, auto_save=True)

                    return brand

        except ImportError:
            logger.warning("LiteLLM not available for brand extraction")
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse LLM response: {e}")
        except Exception as e:
            logger.error(f"Error in LLM brand extraction: {e}")

        return None

    async def verify_brands_with_llm(
        self, products: list[dict[str, Any]], max_concurrent: int = 5, delay_seconds: float = 0.5
    ) -> dict[str, Any]:
        """
        LLM을 사용하여 Unknown/빈 브랜드 일괄 검증 (크롤링 직후 호출)

        Args:
            products: 제품 리스트 [{"asin": "...", "brand": "...", "product_name": "..."}]
            max_concurrent: 최대 동시 요청 수
            delay_seconds: 요청 간 딜레이

        Returns:
            {
                "verified_count": int,
                "failed_count": int,
                "skipped_count": int,
                "updated_products": List[Dict]  # 업데이트된 제품 리스트
            }
        """
        # Unknown 또는 빈 브랜드만 필터링
        unknown_products = [
            p
            for p in products
            if not p.get("brand") or p.get("brand") == "Unknown" or p.get("brand") == ""
        ]

        # 이미 매핑에 있는 것은 제외
        to_verify = [p for p in unknown_products if p.get("asin") not in self.mappings]

        logger.info(
            f"LLM brand verification: {len(to_verify)} products to verify (out of {len(unknown_products)} unknown)"
        )

        verified_count = 0
        failed_count = 0
        results = []

        for product in to_verify:
            asin = product.get("asin")
            product_name = product.get("product_name", "")

            if not product_name:
                failed_count += 1
                continue

            brand = await self.extract_brand_with_llm(product_name, asin)

            if brand and brand != "Unknown":
                verified_count += 1
                results.append(
                    {"asin": asin, "brand": brand, "product_name": product_name, "source": "llm"}
                )
            else:
                failed_count += 1

            # 딜레이 (API 제한 방지)
            if delay_seconds > 0:
                await asyncio.sleep(delay_seconds)

        # 원본 products 리스트 업데이트
        updated_products = []
        for p in products:
            asin = p.get("asin")
            # 캐시에서 브랜드 조회
            cached_brand = self.get_brand(asin)
            if cached_brand:
                p["brand"] = cached_brand
            updated_products.append(p)

        return {
            "verified_count": verified_count,
            "failed_count": failed_count,
            "skipped_count": len(products) - len(to_verify),
            "results": results,
            "updated_products": updated_products,
        }

    def get_stats(self) -> dict[str, Any]:
        """매핑 통계 반환"""
        brands = [m.get("brand") for m in self.mappings.values()]
        unique_brands = set(brands)

        return {
            "total_mappings": len(self.mappings),
            "unique_brands": len(unique_brands),
            "top_brands": sorted(
                [(b, brands.count(b)) for b in unique_brands], key=lambda x: x[1], reverse=True
            )[:10],
        }


# 싱글톤 인스턴스
_resolver_instance: BrandResolver | None = None


def get_brand_resolver() -> BrandResolver:
    """BrandResolver 싱글톤 인스턴스 반환"""
    global _resolver_instance
    if _resolver_instance is None:
        _resolver_instance = BrandResolver()
    return _resolver_instance
