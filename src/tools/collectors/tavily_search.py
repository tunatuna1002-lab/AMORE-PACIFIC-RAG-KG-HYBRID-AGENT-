"""
Tavily Search API Client
========================
실시간 뉴스 및 웹 검색 기능을 제공하는 Tavily API 클라이언트

## 기능
- 뷰티/화장품 관련 뉴스 검색
- 브랜드별 검색
- 검색 결과 신뢰도 평가
- ExternalSignal 변환 지원

## 사용 예
```python
client = TavilySearchClient()
results = await client.search_beauty_news(
    brands=["LANEIGE", "COSRX"],
    topics=["K-Beauty trends"],
    days=7
)

for result in results:
    print(f"{result.title} - {result.source} (신뢰도: {result.reliability_score})")
```

## 환경변수
- TAVILY_API_KEY: Tavily API 키 (필수)
- ENABLE_TAVILY_SEARCH: 활성화 여부 (기본값: true)
"""

import asyncio
import logging
import os
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

try:
    import httpx

    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False

logger = logging.getLogger(__name__)


# 한국 시간대 (UTC+9)
@dataclass
class TavilySearchResult:
    """
    Tavily 검색 결과

    Attributes:
        title: 기사/페이지 제목
        url: 원본 URL
        content: 본문 요약 (최대 500자)
        score: Tavily 관련성 점수 (0-1)
        published_date: 발행일 (YYYY-MM-DD 또는 None)
        source: 출처 도메인 (예: allure.com)
        reliability_score: 신뢰도 점수 (0-1, 소스 기반)
    """

    title: str
    url: str
    content: str
    score: float
    published_date: str | None
    source: str
    reliability_score: float = 0.7

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "url": self.url,
            "content": self.content,
            "score": self.score,
            "published_date": self.published_date,
            "source": self.source,
            "reliability_score": self.reliability_score,
        }


class TavilySearchClient:
    """
    Tavily Search API 클라이언트

    뷰티 산업 특화 뉴스 검색 기능을 제공합니다.
    신뢰도 기반 소스 가중치를 적용하여 검색 결과를 정렬합니다.
    """

    BASE_URL = "https://api.tavily.com"

    # 뷰티 산업 신뢰 소스 (신뢰도 가중치) - 확장 버전
    TRUSTED_SOURCES = {
        # Tier 1: 최고 신뢰도 (전문 뷰티 매체)
        "allure.com": 0.95,
        "wwd.com": 0.95,
        "beautyindependent.com": 0.92,
        "cosmeticsdesign.com": 0.90,
        "cosmeticsdesign-asia.com": 0.90,
        "cosmeticsdesign-europe.com": 0.90,
        "premiumbeautynews.com": 0.88,
        "cosmeticsbusiness.com": 0.88,
        "globalcosmeticsnews.com": 0.86,
        "beautypackaging.com": 0.85,
        "happi.com": 0.85,
        # Tier 2: 높은 신뢰도 (주요 언론)
        "reuters.com": 0.95,
        "bloomberg.com": 0.95,
        "forbes.com": 0.85,
        "businessinsider.com": 0.82,
        "cnbc.com": 0.85,
        "yahoo.com": 0.80,
        # Tier 3: 중간 신뢰도 (생활/패션 매체)
        "vogue.com": 0.80,
        "elle.com": 0.78,
        "harpersbazaar.com": 0.78,
        "byrdie.com": 0.75,
        "refinery29.com": 0.72,
        "glamour.com": 0.72,
        "people.com": 0.70,
        "instyle.com": 0.72,
        "cosmopolitan.com": 0.70,
        "teenvogue.com": 0.68,
        "self.com": 0.70,
        "womenshealthmag.com": 0.70,
        # Tier 4: 기본 신뢰도 (일반 뉴스)
        "cnn.com": 0.75,
        "nytimes.com": 0.85,
        "washingtonpost.com": 0.85,
        "usatoday.com": 0.75,
        # 한국/아시아 매체
        "koreaherald.com": 0.82,
        "kedglobal.com": 0.80,
        "koreatimes.co.kr": 0.78,
        "koreajoongangdaily.joins.com": 0.78,
        "en.yna.co.kr": 0.80,  # 연합뉴스 영문
        "scmp.com": 0.78,  # South China Morning Post
    }

    # 뷰티 산업 기본 키워드
    BEAUTY_KEYWORDS = [
        "K-Beauty",
        "Korean skincare",
        "LANEIGE",
        "COSRX",
        "lip sleeping mask",
        "skincare routine",
        "beauty trends",
        "Amazon bestseller beauty",
        "skincare ingredient",
    ]

    def __init__(self, api_key: str | None = None):
        """
        Args:
            api_key: Tavily API 키 (환경변수 TAVILY_API_KEY로도 설정 가능)
        """
        self.api_key = api_key or os.getenv("TAVILY_API_KEY")
        self._enabled = os.getenv("ENABLE_TAVILY_SEARCH", "true").lower() == "true"
        self._client: httpx.AsyncClient | None = None

        if not self.api_key:
            logger.warning("TAVILY_API_KEY not configured. Tavily search will be disabled.")
            self._enabled = False

    async def _get_client(self) -> httpx.AsyncClient | None:
        """HTTP 클라이언트 반환 (lazy initialization)"""
        if not HTTPX_AVAILABLE:
            logger.error("httpx not installed. Install with: pip install httpx")
            return None

        if not self._enabled:
            return None

        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(30.0),
                headers={
                    "Content-Type": "application/json",
                    "User-Agent": "AmorePacificMarketIntelligence/1.0",
                },
            )

        return self._client

    async def search(
        self,
        query: str,
        search_depth: str = "advanced",
        max_results: int = 10,
        include_domains: list[str] | None = None,
        exclude_domains: list[str] | None = None,
        days: int = 7,
    ) -> list[TavilySearchResult]:
        """
        일반 검색 실행

        Args:
            query: 검색 쿼리 (예: "LANEIGE lip sleeping mask review")
            search_depth: 검색 깊이 (basic: 빠름, advanced: 정확)
            max_results: 최대 결과 수 (기본: 10)
            include_domains: 포함할 도메인 리스트
            exclude_domains: 제외할 도메인 리스트
            days: 검색 기간 (최근 N일)

        Returns:
            TavilySearchResult 리스트
        """
        client = await self._get_client()
        if not client:
            logger.warning("Tavily client not available")
            return []

        payload = {
            "api_key": self.api_key,
            "query": query,
            "search_depth": search_depth,
            "max_results": max_results,
            "include_answer": False,
            "include_raw_content": False,
            "days": days,
        }

        if include_domains:
            payload["include_domains"] = include_domains
        if exclude_domains:
            payload["exclude_domains"] = exclude_domains

        try:
            logger.info(f"Tavily search: '{query}' (days={days}, max={max_results})")

            response = await client.post(f"{self.BASE_URL}/search", json=payload)
            response.raise_for_status()
            data = response.json()

            results = []
            for item in data.get("results", []):
                domain = self._extract_domain(item.get("url", ""))
                reliability = self.TRUSTED_SOURCES.get(domain, 0.7)

                result = TavilySearchResult(
                    title=item.get("title", ""),
                    url=item.get("url", ""),
                    content=item.get("content", "")[:500],  # 500자 제한
                    score=item.get("score", 0.5),
                    published_date=item.get("published_date"),
                    source=domain,
                    reliability_score=reliability,
                )
                results.append(result)

            logger.info(f"Tavily returned {len(results)} results")
            return results

        except httpx.HTTPStatusError as e:
            logger.error(f"Tavily API HTTP error: {e.response.status_code}")
            return []
        except Exception as e:
            logger.error(f"Tavily search failed: {e}")
            return []

    async def search_beauty_news(
        self,
        brands: list[str] | None = None,
        topics: list[str] | None = None,
        days: int = 7,
        max_results_per_query: int = 5,
    ) -> list[TavilySearchResult]:
        """
        뷰티 산업 특화 뉴스 검색

        Args:
            brands: 브랜드명 리스트 (예: ["LANEIGE", "COSRX"])
            topics: 토픽 리스트 (예: ["K-Beauty", "skincare trends"])
            days: 검색 기간 (최근 N일)
            max_results_per_query: 쿼리당 최대 결과 수

        Returns:
            중복 제거 및 신뢰도 정렬된 TavilySearchResult 리스트
        """
        if not self._enabled:
            logger.info("Tavily search is disabled")
            return []

        queries = []

        # 브랜드별 쿼리 생성 (최적화된 검색어)
        if brands:
            for brand in brands:
                # 더 구체적이고 뉴스성 있는 쿼리
                queries.append(f'"{brand}" beauty news')
                queries.append(f'"{brand}" Amazon skincare bestseller')

        # 토픽별 쿼리 생성
        if topics:
            for topic in topics:
                queries.append(f"{topic} news")

        # 기본 쿼리 (K-Beauty 특화)
        if not queries:
            queries = [
                "K-Beauty skincare trends",
                "LANEIGE lip sleeping mask",
                "Korean beauty Amazon bestseller",
                "Amorepacific beauty news",
            ]

        # API 비용 고려: 최대 6개 쿼리 (더 다양한 결과)
        queries = queries[:6]

        all_results = []
        for query in queries:
            results = await self.search(
                query=query,
                days=days,
                max_results=max_results_per_query,
                # 신뢰도 높은 뷰티/뉴스 소스 확장 (30개)
                include_domains=list(self.TRUSTED_SOURCES.keys())[:30],
            )
            all_results.extend(results)

            # API 호출 간 딜레이
            await asyncio.sleep(0.3)

        # 중복 제거 및 신뢰도 기반 정렬
        unique_results = self._deduplicate_and_rank(all_results)

        logger.info(
            f"Beauty news search: {len(unique_results)} unique results from {len(queries)} queries"
        )
        return unique_results

    async def search_brand_mentions(
        self, brand: str, days: int = 30, max_results: int = 20
    ) -> list[TavilySearchResult]:
        """
        특정 브랜드 언급 검색

        Args:
            brand: 브랜드명 (예: "LANEIGE")
            days: 검색 기간
            max_results: 최대 결과 수

        Returns:
            TavilySearchResult 리스트
        """
        queries = [
            f'"{brand}" beauty news',
            f'"{brand}" skincare review',
            f'"{brand}" Amazon bestseller',
        ]

        all_results = []
        for query in queries:
            results = await self.search(query=query, days=days, max_results=max_results // 3)
            all_results.extend(results)
            await asyncio.sleep(0.3)

        return self._deduplicate_and_rank(all_results)

    def _extract_domain(self, url: str) -> str:
        """URL에서 도메인 추출"""
        try:
            parsed = urlparse(url)
            domain = parsed.netloc.replace("www.", "")
            return domain
        except Exception:
            return "unknown"

    def _deduplicate_and_rank(self, results: list[TavilySearchResult]) -> list[TavilySearchResult]:
        """중복 제거 및 신뢰도 기반 랭킹"""
        seen_urls = set()
        unique = []

        for result in results:
            if result.url not in seen_urls:
                seen_urls.add(result.url)
                # 최종 점수 = Tavily 관련성 * 신뢰도
                result.score = result.score * result.reliability_score
                unique.append(result)

        # 점수 기준 정렬
        unique.sort(key=lambda x: x.score, reverse=True)
        return unique

    def get_source_reliability(self, domain: str) -> float:
        """소스의 신뢰도 점수 반환"""
        return self.TRUSTED_SOURCES.get(domain, 0.7)

    async def close(self):
        """클라이언트 종료"""
        if self._client:
            await self._client.aclose()
            self._client = None

    def is_enabled(self) -> bool:
        """Tavily 검색 활성화 여부"""
        return self._enabled and bool(self.api_key)


# 편의 함수
async def search_beauty_news(
    brands: list[str] | None = None, topics: list[str] | None = None, days: int = 7
) -> list[TavilySearchResult]:
    """
    뷰티 뉴스 검색 편의 함수

    Args:
        brands: 브랜드명 리스트
        topics: 토픽 리스트
        days: 검색 기간

    Returns:
        TavilySearchResult 리스트
    """
    client = TavilySearchClient()
    try:
        return await client.search_beauty_news(brands, topics, days)
    finally:
        await client.close()
