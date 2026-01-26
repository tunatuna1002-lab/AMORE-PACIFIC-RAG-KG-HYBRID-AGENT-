# 뉴스 및 보고서 검색 기능 구현 계획

> **작성일**: 2026-01-26
> **목표**: Tavily API, Apify API, 공공데이터 API를 활용한 통합 뉴스/보고서 검색 시스템 구축

---

## 1. 현재 상태 분석

### 1.1 API 연동 현황

| API | 구현 상태 | 코드 위치 | 비고 |
|-----|----------|-----------|------|
| **Tavily API** | ❌ 미구현 | 설정만 존재 (`config/public_apis.json`) | 뉴스 검색용, ~$10/월 |
| **Apify API** | ✅ 구현됨 | `src/tools/apify_amazon_scraper.py` | 기본 비활성화 |
| **YouTube Collector** | ✅ 구현됨 | `src/tools/youtube_collector.py` | 미통합 |
| **공공데이터** | ⚠️ 프레임워크만 | `src/tools/public_data_collector.py` | API 호출 미구현 |
| **RSS 피드** | ✅ 작동 중 | `src/tools/external_signal_collector.py` | 23개 소스 |
| **Reddit API** | ✅ 작동 중 | `src/tools/external_signal_collector.py` | JSON API (무료) |

### 1.2 핵심 문제점

1. **Tavily API 미구현**: 설정 파일에만 존재, 실제 검색 로직 없음
2. **데이터 파이프라인 손실**: 외부 신호의 메타데이터(신뢰도, 관련성)가 LLM 프롬프트에 전달 안됨
3. **ReferenceTracker 미통합**: 외부 신호가 참고자료 섹션에 자동 추가 안됨
4. **신호-지표 상관분석 부재**: 외부 신호와 SoS/HHI 변화 연결 분석 없음

---

## 2. 구현 목표

### 2.1 Phase 1: Tavily API 통합 (우선순위: 높음)

**목표**: 실시간 뉴스 및 웹 검색 기능 추가

```
사용자 쿼리 → Tavily 검색 → 결과 정제 → 인사이트 생성 → 보고서 반영
```

### 2.2 Phase 2: 외부 신호 파이프라인 강화 (우선순위: 높음)

**목표**: 수집된 신호의 메타데이터를 보고서에 완전히 반영

```
외부 신호 수집 → 메타데이터 보존 → LLM 프롬프트 전달 → ReferenceTracker 자동 등록
```

### 2.3 Phase 3: 공공데이터 API 연동 (우선순위: 중간)

**목표**: 한국 공공데이터 (관세청, 식약처, KOSIS) 실제 연동

```
공공 API 호출 → 데이터 파싱 → 시장 분석 통합 → 보고서 반영
```

### 2.4 Phase 4: YouTube Collector 통합 (우선순위: 중간)

**목표**: YouTube 뷰티 콘텐츠 트렌드 분석 연동

```
YouTube 검색 → 영상 메타데이터 수집 → 트렌드 분석 → 보고서 반영
```

---

## 3. 상세 구현 계획

### 3.1 Phase 1: Tavily API 통합

#### 3.1.1 새 파일 생성: `src/tools/tavily_search.py`

```python
"""
Tavily Search API 통합
=====================
실시간 뉴스 및 웹 검색 기능 제공

Features:
- 뷰티/화장품 관련 뉴스 검색
- 브랜드별 검색
- 검색 결과 신뢰도 평가
- ReferenceTracker 자동 연동
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import httpx
import os

@dataclass
class TavilySearchResult:
    """Tavily 검색 결과"""
    title: str
    url: str
    content: str
    score: float  # 관련성 점수 (0-1)
    published_date: Optional[str]
    source: str  # 출처 도메인

class TavilySearchClient:
    """Tavily Search API 클라이언트"""

    BASE_URL = "https://api.tavily.com"

    # 뷰티 산업 신뢰 소스 (신뢰도 가중치 적용)
    TRUSTED_SOURCES = {
        "allure.com": 0.95,
        "wwd.com": 0.95,
        "beautyindependent.com": 0.90,
        "cosmeticsdesign.com": 0.90,
        "reuters.com": 0.95,
        "bloomberg.com": 0.95,
        "forbes.com": 0.85,
    }

    def __init__(self):
        self.api_key = os.getenv("TAVILY_API_KEY")
        if not self.api_key:
            raise ValueError("TAVILY_API_KEY 환경변수가 설정되지 않았습니다")
        self.client = httpx.AsyncClient(timeout=30.0)

    async def search_news(
        self,
        query: str,
        search_depth: str = "advanced",  # basic or advanced
        max_results: int = 10,
        include_domains: List[str] = None,
        exclude_domains: List[str] = None,
        days: int = 7  # 최근 N일
    ) -> List[TavilySearchResult]:
        """
        뉴스 검색 실행

        Args:
            query: 검색 쿼리 (예: "LANEIGE lip sleeping mask review")
            search_depth: 검색 깊이 (basic: 빠름, advanced: 정확)
            max_results: 최대 결과 수
            include_domains: 포함할 도메인 리스트
            exclude_domains: 제외할 도메인 리스트
            days: 검색 기간 (최근 N일)

        Returns:
            검색 결과 리스트
        """
        payload = {
            "api_key": self.api_key,
            "query": query,
            "search_depth": search_depth,
            "max_results": max_results,
            "include_answer": True,
            "include_raw_content": False,
            "days": days
        }

        if include_domains:
            payload["include_domains"] = include_domains
        if exclude_domains:
            payload["exclude_domains"] = exclude_domains

        response = await self.client.post(
            f"{self.BASE_URL}/search",
            json=payload
        )
        response.raise_for_status()
        data = response.json()

        results = []
        for item in data.get("results", []):
            result = TavilySearchResult(
                title=item.get("title", ""),
                url=item.get("url", ""),
                content=item.get("content", ""),
                score=item.get("score", 0.5),
                published_date=item.get("published_date"),
                source=self._extract_domain(item.get("url", ""))
            )
            results.append(result)

        return results

    async def search_beauty_news(
        self,
        brands: List[str] = None,
        topics: List[str] = None,
        days: int = 7
    ) -> List[TavilySearchResult]:
        """
        뷰티 산업 특화 뉴스 검색

        Args:
            brands: 브랜드명 리스트 (예: ["LANEIGE", "COSRX"])
            topics: 토픽 리스트 (예: ["K-Beauty", "skincare trends"])
            days: 검색 기간
        """
        queries = []

        # 브랜드별 쿼리 생성
        if brands:
            for brand in brands:
                queries.append(f"{brand} beauty news")
                queries.append(f"{brand} skincare review")

        # 토픽별 쿼리 생성
        if topics:
            for topic in topics:
                queries.append(f"{topic} 2026")

        # 기본 쿼리
        if not queries:
            queries = [
                "K-Beauty trends 2026",
                "LANEIGE Amazon bestseller",
                "Korean skincare news"
            ]

        all_results = []
        for query in queries[:5]:  # API 비용 고려 최대 5개 쿼리
            results = await self.search_news(
                query=query,
                days=days,
                include_domains=list(self.TRUSTED_SOURCES.keys())
            )
            all_results.extend(results)

        # 중복 제거 및 신뢰도 기반 정렬
        unique_results = self._deduplicate_and_rank(all_results)
        return unique_results

    def _extract_domain(self, url: str) -> str:
        """URL에서 도메인 추출"""
        from urllib.parse import urlparse
        parsed = urlparse(url)
        return parsed.netloc.replace("www.", "")

    def _deduplicate_and_rank(
        self,
        results: List[TavilySearchResult]
    ) -> List[TavilySearchResult]:
        """중복 제거 및 신뢰도 기반 랭킹"""
        seen_urls = set()
        unique = []

        for result in results:
            if result.url not in seen_urls:
                seen_urls.add(result.url)
                # 신뢰도 가중치 적용
                trust_weight = self.TRUSTED_SOURCES.get(result.source, 0.7)
                result.score = result.score * trust_weight
                unique.append(result)

        # 점수 기준 정렬
        unique.sort(key=lambda x: x.score, reverse=True)
        return unique

    async def close(self):
        await self.client.aclose()
```

#### 3.1.2 ExternalSignalCollector 통합

**파일**: `src/tools/external_signal_collector.py`

```python
# 기존 클래스에 추가

class ExternalSignalCollector:
    def __init__(self):
        # ... 기존 코드 ...
        self.tavily_client = None

    async def initialize(self):
        # ... 기존 코드 ...
        # Tavily 클라이언트 초기화
        try:
            from src.tools.tavily_search import TavilySearchClient
            self.tavily_client = TavilySearchClient()
            logger.info("Tavily Search API 초기화 완료")
        except ValueError as e:
            logger.warning(f"Tavily API 미설정: {e}")

    async def fetch_tavily_news(
        self,
        brands: List[str] = None,
        topics: List[str] = None,
        days: int = 7
    ) -> List[ExternalSignal]:
        """Tavily API로 뉴스 검색"""
        if not self.tavily_client:
            logger.warning("Tavily 클라이언트 미초기화")
            return []

        results = await self.tavily_client.search_beauty_news(
            brands=brands,
            topics=topics,
            days=days
        )

        signals = []
        for result in results:
            signal = ExternalSignal(
                source="tavily",
                tier=3,  # Tier 3: Authority
                title=result.title,
                url=result.url,
                content=result.content,
                date=result.published_date or datetime.now().isoformat(),
                reliability_score=result.score,
                relevance_score=result.score,
                metadata={
                    "domain": result.source,
                    "search_type": "news"
                }
            )
            signals.append(signal)

        return signals
```

---

### 3.2 Phase 2: 외부 신호 파이프라인 강화

#### 3.2.1 PeriodInsightAgent 수정

**파일**: `src/agents/period_insight_agent.py`

**현재 문제** (라인 428-469):
```python
# 현재 코드 - 메타데이터 손실
async def _generate_external_signals(self, signals):
    signal_count = len(signals)
    sources = [s.source for s in signals]
    # 신뢰도, 관련성 점수가 전달 안됨!
```

**수정 계획**:
```python
async def _generate_external_signals(
    self,
    signals: List[ExternalSignal]
) -> Dict[str, Any]:
    """외부 신호 분석 (메타데이터 완전 보존)"""

    # 신호를 Tier별로 그룹화
    tier_groups = {1: [], 2: [], 3: [], 4: []}
    for signal in signals:
        tier_groups[signal.tier].append(signal)

    # LLM 프롬프트용 상세 컨텍스트 생성
    signal_context = []
    for tier, tier_signals in tier_groups.items():
        tier_name = {
            1: "바이럴 신호 (TikTok/Instagram)",
            2: "검증/리뷰 (YouTube/Reddit)",
            3: "권위 있는 출처 (뉴스/전문지)",
            4: "PR/실시간 (Twitter/보도자료)"
        }.get(tier, "기타")

        for s in tier_signals:
            signal_context.append({
                "tier": tier,
                "tier_name": tier_name,
                "source": s.source,
                "title": s.title,
                "date": s.date,
                "reliability": s.reliability_score,  # 신뢰도 보존
                "relevance": s.relevance_score,      # 관련성 보존
                "content_preview": s.content[:200] if s.content else "",
                "url": s.url
            })

    # LLM 프롬프트에 전체 메타데이터 전달
    prompt = f"""
## 외부 신호 분석

총 {len(signals)}개의 외부 신호를 분석합니다.

### 신호 상세 (신뢰도/관련성 점수 포함)

{json.dumps(signal_context, ensure_ascii=False, indent=2)}

### 분석 요청
1. Tier별 핵심 트렌드 요약
2. 신뢰도 높은 출처(0.8 이상)의 핵심 메시지
3. LANEIGE와 직접 관련된 신호 강조
4. 시장 동향과 연결되는 인사이트
"""

    return {
        "signal_count": len(signals),
        "tier_breakdown": {k: len(v) for k, v in tier_groups.items()},
        "prompt_context": prompt,
        "raw_signals": signal_context  # 원본 보존
    }
```

#### 3.2.2 ReferenceTracker 자동 연동

**파일**: `src/tools/reference_tracker.py`

**추가할 메서드**:
```python
def add_external_signals(
    self,
    signals: List[ExternalSignal],
    auto_categorize: bool = True
) -> int:
    """
    외부 신호를 참고자료에 자동 추가

    Args:
        signals: ExternalSignal 객체 리스트
        auto_categorize: 자동 카테고리 분류 여부

    Returns:
        추가된 참고자료 수
    """
    added_count = 0

    # Tier → ReferenceType 매핑
    tier_to_type = {
        1: ReferenceType.SOCIAL,   # TikTok/Instagram
        2: ReferenceType.SOCIAL,   # YouTube/Reddit
        3: ReferenceType.ARTICLE,  # News/전문지
        4: ReferenceType.ARTICLE   # PR/Twitter
    }

    for signal in signals:
        ref_type = tier_to_type.get(signal.tier, ReferenceType.ARTICLE)

        # 중복 체크
        if self._is_duplicate(signal.url):
            continue

        reference = Reference(
            type=ref_type,
            title=signal.title,
            source=signal.source,
            url=signal.url,
            date=signal.date,
            metadata={
                "tier": signal.tier,
                "reliability_score": signal.reliability_score,
                "relevance_score": signal.relevance_score
            }
        )

        self.add_reference(reference)
        added_count += 1

    return added_count

def _is_duplicate(self, url: str) -> bool:
    """URL 기반 중복 체크"""
    for ref in self.references:
        if ref.url == url:
            return True
    return False
```

---

### 3.3 Phase 3: 공공데이터 API 연동

#### 3.3.1 PublicDataCollector 완성

**파일**: `src/tools/public_data_collector.py`

**현재 상태**: 프레임워크만 존재, 실제 API 호출 없음

**구현 계획**:

```python
class PublicDataCollector:
    """
    한국 공공데이터 API 연동

    지원 API:
    1. 관세청 수출입 통계 (화장품 HS코드: 3304)
    2. 식약처 화장품 원료/제품 데이터
    3. KOSIS 소비자물가지수
    """

    async def fetch_customs_export_data(
        self,
        hs_code: str = "3304",  # 화장품
        country: str = "US",
        start_date: str = None,
        end_date: str = None
    ) -> Dict[str, Any]:
        """
        관세청 화장품 수출 데이터 조회

        Returns:
            {
                "period": "2025-12",
                "export_amount_usd": 1234567890,
                "yoy_change": 12.5,
                "top_items": [
                    {"name": "립스틱", "amount": ...},
                    ...
                ]
            }
        """
        api_key = os.getenv("DATA_GO_KR_API_KEY")
        if not api_key:
            raise ValueError("DATA_GO_KR_API_KEY 환경변수 필요")

        # 실제 API 호출 구현
        url = "http://apis.data.go.kr/1220000/tradestatistics"
        params = {
            "serviceKey": api_key,
            "searchStDt": start_date,
            "searchEdDt": end_date,
            "hsCode": hs_code,
            "cntyCd": country
        }

        async with httpx.AsyncClient() as client:
            response = await client.get(url, params=params)
            data = response.json()

        return self._parse_customs_data(data)

    async def fetch_mfds_cosmetics_data(
        self,
        category: str = "기능성화장품"
    ) -> Dict[str, Any]:
        """
        식약처 화장품 허가/등록 데이터

        Returns:
            최근 허가된 화장품 리스트, 성분 트렌드 등
        """
        api_key = os.getenv("MFDS_API_KEY")
        # ... 구현

    async def fetch_consumer_price_index(
        self,
        item_code: str = "화장품"
    ) -> Dict[str, Any]:
        """
        KOSIS 소비자물가지수 (화장품)

        Returns:
            물가지수 추이, 전년 대비 변화율
        """
        api_key = os.getenv("KOSIS_API_KEY")
        # ... 구현
```

---

### 3.4 Phase 4: YouTube Collector 통합

#### 3.4.1 Market Intelligence Engine 연동

**파일**: `src/tools/market_intelligence.py`

```python
class MarketIntelligenceEngine:
    """4계층 시장 인텔리전스 엔진"""

    async def collect_all_signals(
        self,
        brands: List[str],
        period_days: int = 7
    ) -> Dict[str, Any]:
        """모든 소스에서 신호 수집"""

        results = {
            "tavily_news": [],
            "youtube_trends": [],
            "reddit_discussions": [],
            "rss_articles": [],
            "public_data": {}
        }

        # 1. Tavily 뉴스 검색
        if self.external_collector.tavily_client:
            results["tavily_news"] = await self.external_collector.fetch_tavily_news(
                brands=brands,
                days=period_days
            )

        # 2. YouTube 트렌드 (기존 youtube_collector.py 활용)
        if self.youtube_collector:
            results["youtube_trends"] = await self.youtube_collector.search_beauty_videos(
                queries=[f"{brand} review" for brand in brands],
                max_results=20
            )

        # 3. Reddit 토론
        results["reddit_discussions"] = await self.external_collector.fetch_reddit_trends(
            subreddits=["SkincareAddiction", "AsianBeauty", "MakeupAddiction"]
        )

        # 4. RSS 기사
        results["rss_articles"] = await self.external_collector.fetch_rss_articles(
            keywords=brands
        )

        # 5. 공공데이터 (한국 수출 통계)
        if self.public_data_collector:
            results["public_data"] = await self.public_data_collector.fetch_customs_export_data()

        return results
```

---

## 4. 파일 변경 요약

| 파일 | 작업 | 설명 |
|------|------|------|
| `src/tools/tavily_search.py` | **CREATE** | Tavily API 클라이언트 |
| `src/tools/external_signal_collector.py` | **MODIFY** | Tavily 통합, 메타데이터 보존 |
| `src/agents/period_insight_agent.py` | **MODIFY** | LLM 프롬프트에 전체 메타데이터 전달 |
| `src/tools/reference_tracker.py` | **MODIFY** | `add_external_signals()` 메서드 추가 |
| `src/tools/public_data_collector.py` | **MODIFY** | 실제 API 호출 구현 |
| `src/tools/market_intelligence.py` | **MODIFY** | YouTube Collector 통합 |
| `config/public_apis.json` | **MODIFY** | Tavily enabled: true 변경 |

---

## 5. 환경변수 요구사항

```bash
# .env 파일 필수 설정

# Tavily API (뉴스 검색)
TAVILY_API_KEY=tvly-...

# Apify API (Amazon/YouTube 스크래핑)
APIFY_API_TOKEN=apify_api_...

# 한국 공공데이터 포털
DATA_GO_KR_API_KEY=...

# 식약처 API (선택)
MFDS_API_KEY=...

# KOSIS API (선택)
KOSIS_API_KEY=...
```

---

## 6. 테스트 계획

### 6.1 단위 테스트

```bash
# Tavily API 테스트
python -m pytest tests/test_tavily_search.py -v

# 외부 신호 파이프라인 테스트
python -m pytest tests/test_external_signal_pipeline.py -v

# 참고자료 자동 등록 테스트
python -m pytest tests/test_reference_tracker_integration.py -v
```

### 6.2 통합 테스트

```bash
# 전체 보고서 생성 테스트
curl -X POST http://localhost:8001/api/export/analyst-report \
  -H "Content-Type: application/json" \
  -d '{
    "start_date": "2026-01-14",
    "end_date": "2026-01-25",
    "include_external_signals": true
  }'
```

### 6.3 검증 항목

- [ ] Tavily 검색 결과가 보고서 Section 5에 반영되는지
- [ ] 외부 신호 신뢰도/관련성 점수가 보고서에 표시되는지
- [ ] ReferenceTracker가 외부 신호를 Section 8에 자동 추가하는지
- [ ] YouTube 트렌드가 분석에 포함되는지
- [ ] 공공데이터 (수출 통계)가 시장 분석에 반영되는지

---

## 7. 예상 일정

| Phase | 예상 소요 | 우선순위 |
|-------|----------|----------|
| Phase 1: Tavily API 통합 | 2-3시간 | 🔴 높음 |
| Phase 2: 파이프라인 강화 | 2-3시간 | 🔴 높음 |
| Phase 3: 공공데이터 연동 | 3-4시간 | 🟡 중간 |
| Phase 4: YouTube 통합 | 1-2시간 | 🟡 중간 |
| 테스트 및 검증 | 2시간 | 🔴 높음 |

**총 예상 소요**: 10-14시간

---

## 8. 아키텍처 다이어그램

```
┌─────────────────────────────────────────────────────────────────┐
│                    Market Intelligence Engine                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │ Tavily API  │  │  Apify API  │  │ 공공데이터   │             │
│  │ (뉴스 검색)  │  │ (Amazon/YT) │  │ (관세청 등)  │             │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘             │
│         │                │                │                     │
│         └────────────────┼────────────────┘                     │
│                          ▼                                      │
│              ┌───────────────────────┐                         │
│              │ ExternalSignalCollector│                         │
│              │ (신호 수집 + 정규화)    │                         │
│              └───────────┬───────────┘                         │
│                          │                                      │
│         ┌────────────────┼────────────────┐                     │
│         ▼                ▼                ▼                     │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │ RSS Feeds   │  │ Reddit API  │  │ Manual Input│             │
│  │ (23개 소스)  │  │ (JSON API)  │  │ (TikTok 등) │             │
│  └─────────────┘  └─────────────┘  └─────────────┘             │
│                                                                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│              ┌───────────────────────┐                         │
│              │ PeriodInsightAgent    │                         │
│              │ (LLM 인사이트 생성)    │◄──── 메타데이터 완전 보존  │
│              └───────────┬───────────┘                         │
│                          │                                      │
│              ┌───────────▼───────────┐                         │
│              │ ReferenceTracker      │◄──── 자동 참고자료 등록   │
│              └───────────┬───────────┘                         │
│                          │                                      │
│              ┌───────────▼───────────┐                         │
│              │ DOCX Report Generator │                         │
│              │ (애널리스트 보고서)    │                         │
│              └───────────────────────┘                         │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 9. 리스크 및 대응

| 리스크 | 영향 | 대응 방안 |
|--------|------|----------|
| Tavily API 비용 초과 | 중간 | 일일 쿼리 제한 (50회), 캐싱 적용 |
| 공공데이터 API 불안정 | 낮음 | Fallback 데이터 준비, 재시도 로직 |
| API 응답 지연 | 중간 | 타임아웃 설정, 비동기 병렬 처리 |
| 데이터 품질 이슈 | 중간 | 신뢰도 기반 필터링, 검증 로직 |

---

## 10. 성공 기준

1. **Tavily 검색 동작**: 브랜드명으로 검색 시 관련 뉴스 5개 이상 반환
2. **메타데이터 보존**: 보고서에 신뢰도/관련성 점수 표시
3. **참고자료 자동화**: 외부 신호 → Section 8 자동 추가
4. **보고서 품질**: 외부 신호 섹션이 Tier별로 구조화되어 표시
5. **응답 시간**: 전체 보고서 생성 3분 이내

---

*이 계획은 사용자의 요청에 따라 작성되었습니다. 구현 시작 전 추가 요구사항이 있으면 말씀해주세요.*
