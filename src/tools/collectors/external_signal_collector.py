"""
External Signal Collector
==========================
뷰티 전문 매체 및 SNS에서 트렌드 신호를 수집하는 모듈

## 아키텍처
```
┌─────────────────────────────────────────────────────────┐
│           External Signal Sources (뷰티 특화)            │
├─────────────────────────────────────────────────────────┤
│  [Tier 1] 발견/바이럴 감지 - TikTok, Instagram          │
│  [Tier 2] 검증/심층 리뷰 - YouTube, Reddit              │
│  [Tier 3] 전문 매체 (보고서 근거) - Allure, WWD, People │
│  [Tier 4] 보조/PR - X (Twitter)                         │
└─────────────────────────────────────────────────────────┘
```

## 수집 방법
- Tier 3 (전문 매체): RSS 피드 + 키워드 필터링 (무료)
- Tier 1/2 (SNS): TikTok Creative Center, Reddit API (무료)
- 유료 API: NewsAPI, Bing News (주석 처리)

## 사용 예
```python
collector = ExternalSignalCollector()
await collector.initialize()

# RSS에서 뷰티 기사 수집
articles = await collector.fetch_beauty_articles(["LANEIGE", "K-Beauty"])

# Reddit에서 트렌드 수집
reddit_posts = await collector.fetch_reddit_trends(["SkincareAddiction"])

# 수동 입력 (주간 트렌드 레이더 작성 시)
collector.add_manual_media_input(media_data)
```

## 데이터 출력 (보고서용)
```
■ 전문 매체 근거:
• Allure (1월 10일): "Lipification of Beauty 현상 가속화"
• People (1월 12일): "LANEIGE가 글래스 스킨 트렌드 선도"

■ 소비자 트렌드:
• TikTok #LipBasting: 520만 조회 (1월 14일 기준)
• Reddit r/SkincareAddiction: 립마스크 추천글 2,400 업보트
```
"""

import asyncio
import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any

from src.shared.constants import KST

# HTTP 클라이언트
try:
    import aiohttp

    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False

try:
    import feedparser

    FEEDPARSER_AVAILABLE = True
except ImportError:
    FEEDPARSER_AVAILABLE = False


logger = logging.getLogger(__name__)


# 한국 시간대 (UTC+9)
class SignalTier(Enum):
    """신호 출처 등급"""

    TIER1_VIRAL = "tier1_viral"  # TikTok, Instagram - 바이럴 감지
    TIER2_VALIDATION = "tier2_validation"  # YouTube, Reddit - 검증/리뷰
    TIER3_AUTHORITY = "tier3_authority"  # Allure, WWD, People - 권위 있는 근거
    TIER4_PR = "tier4_pr"  # X (Twitter) - PR/실시간 이슈


class SignalSource(Enum):
    """신호 출처"""

    # Tier 1: 바이럴
    TIKTOK = "tiktok"
    INSTAGRAM = "instagram"

    # Tier 2: 검증
    YOUTUBE = "youtube"
    REDDIT = "reddit"

    # Tier 3: 전문 매체 (글로벌)
    ALLURE = "allure"
    VOGUE_BEAUTY = "vogue_beauty"
    WWD_BEAUTY = "wwd_beauty"
    PEOPLE = "people"
    BYRDIE = "byrdie"
    REFINERY29 = "refinery29"
    COSMETICS_BUSINESS = "cosmetics_business"
    BEAUTY_INDEPENDENT = "beauty_independent"
    GLOSSY = "glossy"

    # Tier 3: 뷰티 산업 전문 매체 (추가)
    COSMETICS_DESIGN_ASIA = "cosmetics_design_asia"
    COSMETICS_DESIGN_EUROPE = "cosmetics_design_europe"
    KEDGLOBAL = "kedglobal"
    KOREA_HERALD = "korea_herald"
    COSINKOREA = "cosinkorea"
    COSMORNING = "cosmorning"

    # Tier 3: 뷰티 산업 전문 매체 (Phase 1.2 추가)
    PREMIUM_BEAUTY_NEWS = "premium_beauty_news"
    HAPPI = "happi"
    BEAUTY_PACKAGING = "beauty_packaging"
    GLOBAL_COSMETICS_NEWS = "global_cosmetics_news"

    # Tier 4: PR
    TWITTER = "twitter"

    # 수동 입력
    MANUAL = "manual"


@dataclass
class ExternalSignal:
    """
    외부 신호 데이터 클래스

    Attributes:
        signal_id: 고유 ID (SID-{source}-{date}-{seq})
        source: 출처 (allure, tiktok, reddit 등)
        tier: 신호 등급 (tier1~tier4)
        title: 제목/헤드라인
        content: 본문/요약
        url: 원본 URL
        published_at: 발행일
        collected_at: 수집일
        keywords: 매칭된 키워드
        relevance_score: 관련성 점수 (0.0~1.0)
        metadata: 추가 메타데이터 (조회수, 업보트 등)
    """

    signal_id: str
    source: str
    tier: str
    title: str
    content: str
    url: str
    published_at: str
    collected_at: str
    keywords: list[str] = field(default_factory=list)
    relevance_score: float = 0.5
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_report_format(self) -> str:
        """보고서용 자연어 포맷"""
        date_str = self.published_at[:10] if self.published_at else "날짜 미상"
        source_name = self.source.replace("_", " ").title()

        # 메타데이터에서 추가 정보 추출
        views = self.metadata.get("views", "")
        upvotes = self.metadata.get("upvotes", "")

        extra = ""
        if views:
            extra = f" (조회수 {views:,})"
        elif upvotes:
            extra = f" ({upvotes:,} 업보트)"

        return f'• {source_name} ({date_str}): "{self.title}"{extra}'


# =============================================================================
# RSS 피드 설정 (무료)
# =============================================================================

RSS_FEEDS = {
    # Tier 3: 전문 매체 RSS (무료로 사용 가능)
    SignalSource.ALLURE: "https://www.allure.com/feed/rss",
    SignalSource.BYRDIE: "https://www.byrdie.com/feed",
    SignalSource.REFINERY29: "https://www.refinery29.com/en-us/beauty/rss.xml",
    # 뷰티 산업 전문 매체 (추가) - 인사이트 시스템 고도화
    SignalSource.COSMETICS_DESIGN_ASIA: "https://www.cosmeticsdesign-asia.com/Info/RSS/",
    SignalSource.COSMETICS_DESIGN_EUROPE: "https://www.cosmeticsdesign-europe.com/Info/RSS/",
    SignalSource.KEDGLOBAL: "https://www.kedglobal.com/rss/all.xml",
    SignalSource.KOREA_HERALD: "https://www.koreaherald.com/rss/028040600.xml",  # Business & Economy
    # Phase 1.2: 추가 뷰티 산업 매체 (2026-01)
    SignalSource.COSMETICS_BUSINESS: "https://www.cosmeticsbusiness.com/rss/news.xml",
    SignalSource.PREMIUM_BEAUTY_NEWS: "https://www.premiumbeautynews.com/en/?format=feed&type=rss",
    SignalSource.HAPPI: "https://www.happi.com/rss/",
    SignalSource.BEAUTY_PACKAGING: "https://www.beautypackaging.com/rss/news.xml",
    SignalSource.GLOBAL_COSMETICS_NEWS: "https://www.globalcosmeticsnews.com/feed/",
    # 일부 매체는 RSS를 제공하지 않거나 제한적
    # WWD, People, Vogue는 RSS가 없거나 유료 구독 필요
    # COSINKOREA, COSMORNING - 한국 매체는 RSS 없음, 스크래핑 필요
}

# 뷰티 관련 키워드 (필터링용)
BEAUTY_KEYWORDS = [
    # 아모레퍼시픽 브랜드 (우선순위)
    "laneige",
    "라네즈",
    "amorepacific",
    "아모레퍼시픽",
    "sulwhasoo",
    "설화수",
    "hera",
    "헤라",
    "iope",
    "아이오페",
    "illiyoon",
    "일리윤",
    "cosrx",
    "코스알엑스",
    "aestura",
    "에스트라",
    "mise en scene",
    "미장센",
    "hanyul",
    "한율",
    # K-Beauty 키워드
    "k-beauty",
    "korean beauty",
    "k beauty",
    "korean cosmetics",
    "korean skincare",
    "korean makeup",
    # 경쟁 브랜드
    "tirtir",
    "medicube",
    "biodance",
    "skin1004",
    "anua",
    "summer fridays",
    "rare beauty",
    "glow recipe",
    "drunk elephant",
    "tatcha",
    "the ordinary",
    "cerave",
    "la roche posay",
    # 제품 카테고리
    "lip care",
    "lip mask",
    "lip balm",
    "lip sleeping",
    "skin care",
    "skincare",
    "moisturizer",
    "serum",
    "sunscreen",
    "spf",
    "toner",
    "essence",
    "sleeping mask",
    "sheet mask",
    "face mask",
    # 트렌드/성분
    "glass skin",
    "peptide",
    "niacinamide",
    "retinol",
    "vitamin c",
    "hyaluronic",
    "ceramide",
    "snail mucin",
    "pdrn",
    "centella",
    "cica",
    "collagen",
    "probiotics",
    "slow aging",
    "pre-aging",
    "skin barrier",
    # 트렌드 키워드
    "beauty trend",
    "skincare trend",
    "tiktok beauty",
    "viral beauty",
    "best seller",
    "bestseller",
    "amazon beauty",
    "prime day",
    "sephora",
    # 시장/산업 키워드
    "cosmetics export",
    "beauty market",
    "cosmetics industry",
    "beauty brand",
    "beauty growth",
    "beauty sales",
]

# K-Beauty 전용 키워드 (글로벌 매체에서 한국 관련 기사 필터링)
KBEAUTY_KEYWORDS = [
    "k-beauty",
    "korean beauty",
    "korean cosmetics",
    "korean skincare",
    "laneige",
    "amorepacific",
    "cosrx",
    "sulwhasoo",
    "innisfree",
    "tirtir",
    "medicube",
    "biodance",
    "skin1004",
    "anua",
    "korea",
    "korean",
    "seoul",
    "gangnam",
    "k beauty",
    "k-skincare",
]


class ExternalSignalCollector:
    """
    외부 신호 수집기

    기능:
    1. RSS 피드에서 뷰티 기사 수집 (무료)
    2. Reddit API로 트렌드 수집 (무료)
    3. 수동 입력 지원 (주간 트렌드 레이더)
    4. 유료 API 지원 (NewsAPI, Bing News - 주석 처리)
    """

    def __init__(self, data_dir: str = "./data/external_signals"):
        """
        Args:
            data_dir: 신호 저장 디렉토리
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # 수집된 신호 저장
        self.signals: list[ExternalSignal] = []

        # 신호 ID 시퀀스
        self._signal_seq = 0

        # HTTP 세션
        self._session: aiohttp.ClientSession | None = None

        # Tavily 클라이언트 (lazy initialization)
        self._tavily_client = None

    async def initialize(self) -> None:
        """비동기 초기화"""
        if AIOHTTP_AVAILABLE and not self._session:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=30),
                headers={"User-Agent": "Mozilla/5.0 (compatible; BeautyTrendBot/1.0)"},
            )

        # Tavily 클라이언트 초기화
        try:
            from src.tools.collectors.tavily_search import TavilySearchClient

            self._tavily_client = TavilySearchClient()
            if self._tavily_client.is_enabled():
                logger.info("Tavily Search API initialized successfully")
            else:
                logger.warning("Tavily Search API not configured (TAVILY_API_KEY not set)")
                self._tavily_client = None
        except ImportError as e:
            logger.warning(f"Tavily module not available: {e}")
            self._tavily_client = None
        except Exception as e:
            logger.warning(f"Tavily initialization failed: {e}")
            self._tavily_client = None

        # 기존 신호 로드
        self._load_signals()

    async def close(self) -> None:
        """세션 종료"""
        if self._session:
            await self._session.close()
            self._session = None

    # =========================================================================
    # Tier 3: 전문 매체 (RSS - 무료)
    # =========================================================================

    async def fetch_rss_articles(
        self, source: SignalSource, keywords: list[str] | None = None, max_articles: int = 10
    ) -> list[ExternalSignal]:
        """
        RSS 피드에서 기사 수집

        Args:
            source: 매체 소스
            keywords: 필터링 키워드 (없으면 기본 뷰티 키워드 사용)
            max_articles: 최대 수집 기사 수

        Returns:
            ExternalSignal 리스트
        """
        if not FEEDPARSER_AVAILABLE:
            logger.warning("feedparser not installed. Run: pip install feedparser")
            return []

        feed_url = RSS_FEEDS.get(source)
        if not feed_url:
            logger.warning(f"No RSS feed configured for {source}")
            return []

        keywords = keywords or BEAUTY_KEYWORDS
        keywords_lower = [k.lower() for k in keywords]

        try:
            # RSS 파싱
            feed = feedparser.parse(feed_url)

            if feed.bozo:
                logger.warning(f"RSS parse error for {source}: {feed.bozo_exception}")

            signals = []
            for entry in feed.entries[: max_articles * 2]:  # 필터링 고려해서 2배 수집
                # 제목과 요약에서 키워드 매칭
                title = entry.get("title", "")
                summary = entry.get("summary", entry.get("description", ""))
                content = f"{title} {summary}".lower()

                matched_keywords = [k for k in keywords_lower if k in content]

                if not matched_keywords:
                    continue

                # 발행일 파싱
                published = entry.get("published_parsed") or entry.get("updated_parsed")
                if published:
                    published_dt = datetime(*published[:6])
                    published_str = published_dt.strftime("%Y-%m-%d")
                else:
                    published_str = datetime.now(KST).strftime("%Y-%m-%d")

                signal = ExternalSignal(
                    signal_id=self._generate_signal_id(source.value),
                    source=source.value,
                    tier=SignalTier.TIER3_AUTHORITY.value,
                    title=title,
                    content=summary[:500] if summary else "",
                    url=entry.get("link", ""),
                    published_at=published_str,
                    collected_at=datetime.now(KST).isoformat(),
                    keywords=matched_keywords,
                    relevance_score=min(len(matched_keywords) * 0.2, 1.0),
                    metadata={
                        "author": entry.get("author", ""),
                        "tags": [t.get("term", "") for t in entry.get("tags", [])],
                    },
                )
                signals.append(signal)

                if len(signals) >= max_articles:
                    break

            logger.info(f"Fetched {len(signals)} articles from {source.value}")
            return signals

        except Exception as e:
            logger.error(f"Failed to fetch RSS from {source}: {e}")
            return []

    async def fetch_all_rss_feeds(self, keywords: list[str] | None = None) -> list[ExternalSignal]:
        """모든 RSS 피드에서 기사 수집"""
        all_signals = []

        for source in RSS_FEEDS.keys():
            signals = await self.fetch_rss_articles(source, keywords)
            all_signals.extend(signals)
            await asyncio.sleep(1)  # Rate limiting

        self.signals.extend(all_signals)
        self._save_signals()

        return all_signals

    # =========================================================================
    # Tier 2: Reddit (무료 API)
    # =========================================================================

    async def fetch_reddit_trends(
        self, subreddits: list[str] = None, keywords: list[str] | None = None, max_posts: int = 10
    ) -> list[ExternalSignal]:
        """
        Reddit에서 트렌드 수집 (JSON API - 무료)

        Args:
            subreddits: 서브레딧 목록
            keywords: 필터링 키워드
            max_posts: 최대 수집 게시물 수

        Returns:
            ExternalSignal 리스트
        """
        if not self._session:
            await self.initialize()

        subreddits = subreddits or ["SkincareAddiction", "AsianBeauty", "MakeupAddiction"]
        keywords = keywords or BEAUTY_KEYWORDS
        keywords_lower = [k.lower() for k in keywords]

        signals = []

        for subreddit in subreddits:
            try:
                # Reddit JSON API (인증 불필요)
                url = f"https://www.reddit.com/r/{subreddit}/hot.json?limit=25"

                async with self._session.get(url) as response:
                    if response.status != 200:
                        logger.warning(f"Reddit API error for r/{subreddit}: {response.status}")
                        continue

                    data = await response.json()

                posts = data.get("data", {}).get("children", [])

                for post in posts:
                    post_data = post.get("data", {})
                    title = post_data.get("title", "")
                    selftext = post_data.get("selftext", "")
                    content = f"{title} {selftext}".lower()

                    matched_keywords = [k for k in keywords_lower if k in content]

                    if not matched_keywords:
                        continue

                    # 타임스탬프 변환
                    created_utc = post_data.get("created_utc", 0)
                    published_str = datetime.fromtimestamp(created_utc).strftime("%Y-%m-%d")

                    signal = ExternalSignal(
                        signal_id=self._generate_signal_id("reddit"),
                        source=SignalSource.REDDIT.value,
                        tier=SignalTier.TIER2_VALIDATION.value,
                        title=title,
                        content=selftext[:500] if selftext else "",
                        url=f"https://reddit.com{post_data.get('permalink', '')}",
                        published_at=published_str,
                        collected_at=datetime.now(KST).isoformat(),
                        keywords=matched_keywords,
                        relevance_score=min(len(matched_keywords) * 0.2, 1.0),
                        metadata={
                            "subreddit": subreddit,
                            "upvotes": post_data.get("ups", 0),
                            "comments": post_data.get("num_comments", 0),
                            "author": post_data.get("author", ""),
                        },
                    )
                    signals.append(signal)

                    if len(signals) >= max_posts:
                        break

                await asyncio.sleep(2)  # Reddit rate limiting

            except Exception as e:
                logger.error(f"Failed to fetch Reddit r/{subreddit}: {e}")
                continue

        self.signals.extend(signals)
        self._save_signals()

        logger.info(f"Fetched {len(signals)} posts from Reddit")
        return signals

    # =========================================================================
    # Tier 1: TikTok (제한적 - 공식 API 없음)
    # =========================================================================

    async def fetch_tiktok_trends(self, hashtags: list[str] = None) -> list[ExternalSignal]:
        """
        TikTok 트렌드 수집

        NOTE: TikTok은 공식 API가 없어 수동 입력 또는
        TikTok Creative Center (https://ads.tiktok.com/business/creativecenter)
        에서 데이터를 복사해서 입력해야 합니다.

        Returns:
            빈 리스트 (수동 입력 권장)
        """
        logger.info(
            "TikTok API is not available. "
            "Please use manual input or TikTok Creative Center for trend data."
        )
        # TikTok Creative Center에서 수동으로 데이터 확인 후
        # add_manual_media_input()으로 입력하는 것을 권장
        return []

    # =========================================================================
    # 수동 입력 (주간 트렌드 레이더 작성 시)
    # =========================================================================

    def add_manual_media_input(self, media_data: dict[str, Any]) -> ExternalSignal:
        """
        수동 매체 입력

        주간 트렌드 레이더 작성 시 Allure, People, Dash Social 등에서
        수동으로 수집한 데이터를 입력합니다.

        Args:
            media_data: {
                "source": "allure",
                "date": "2026-01-10",
                "title": "2026 Skincare Trends",
                "url": "https://allure.com/...",
                "quotes": ["펩타이드가 2026년 트렌드", ...],
                "keywords": ["peptide", "lip care"],
                "views": 50000  # optional
            }

        Returns:
            생성된 ExternalSignal
        """
        source_str = media_data.get("source", "manual")

        # 소스에 따른 Tier 결정
        tier = SignalTier.TIER3_AUTHORITY.value
        if source_str in ["tiktok", "instagram"]:
            tier = SignalTier.TIER1_VIRAL.value
        elif source_str in ["youtube", "reddit"]:
            tier = SignalTier.TIER2_VALIDATION.value
        elif source_str in ["twitter", "x"]:
            tier = SignalTier.TIER4_PR.value

        signal = ExternalSignal(
            signal_id=self._generate_signal_id(source_str),
            source=source_str,
            tier=tier,
            title=media_data.get("title", ""),
            content="\n".join(media_data.get("quotes", [])),
            url=media_data.get("url", ""),
            published_at=media_data.get("date", datetime.now(KST).strftime("%Y-%m-%d")),
            collected_at=datetime.now(KST).isoformat(),
            keywords=media_data.get("keywords", []),
            relevance_score=0.9,  # 수동 입력은 관련성 높음
            metadata={
                "views": media_data.get("views"),
                "upvotes": media_data.get("upvotes"),
                "manual_input": True,
            },
        )

        self.signals.append(signal)
        self._save_signals()

        logger.info(f"Added manual signal: {signal.signal_id}")
        return signal

    def add_weekly_trend_radar(self, radar_data: list[dict[str, Any]]) -> list[ExternalSignal]:
        """
        주간 트렌드 레이더 데이터 일괄 입력

        미국 뷰티 트렌드 레이더.md 형식의 데이터를 입력합니다.

        Args:
            radar_data: [
                {
                    "rank": 1,
                    "keyword": "립케어·#lipcare",
                    "source": "Dash Social / TikTok",
                    "date": "2025-12-21",
                    "consumer_need": "겨울 건조로 인한 입술 보습 관심 급증",
                    "laneige_connection": "Lip Sleeping Mask 강점 강조",
                    "evidence": "TikTok #lipcare(2K), #lipoil(2K) 상위 태그"
                },
                ...
            ]

        Returns:
            생성된 ExternalSignal 리스트
        """
        signals = []

        for item in radar_data:
            signal = ExternalSignal(
                signal_id=self._generate_signal_id("trend_radar"),
                source="trend_radar",
                tier=SignalTier.TIER3_AUTHORITY.value,
                title=item.get("keyword", ""),
                content=f"소비자 니즈: {item.get('consumer_need', '')}\n"
                f"LANEIGE 연결: {item.get('laneige_connection', '')}\n"
                f"근거: {item.get('evidence', '')}",
                url="",
                published_at=item.get("date", datetime.now(KST).strftime("%Y-%m-%d")),
                collected_at=datetime.now(KST).isoformat(),
                keywords=[item.get("keyword", "")],
                relevance_score=1.0,
                metadata={
                    "rank": item.get("rank"),
                    "source_detail": item.get("source"),
                    "manual_input": True,
                },
            )
            signals.append(signal)

        self.signals.extend(signals)
        self._save_signals()

        logger.info(f"Added {len(signals)} trend radar entries")
        return signals

    # =========================================================================
    # 유료 API (주석 처리 - 필요 시 활성화)
    # =========================================================================

    # ---------------------------------------------------------------------
    # NewsAPI ($449/월 비즈니스 플랜 또는 $0 개발자 플랜 - 제한적)
    # 웹사이트: https://newsapi.org/
    # ---------------------------------------------------------------------
    #
    # async def fetch_newsapi_articles(
    #     self,
    #     query: str = "LANEIGE OR K-Beauty OR lip care",
    #     domains: str = "allure.com,vogue.com,wwd.com,people.com",
    #     api_key: Optional[str] = None
    # ) -> List[ExternalSignal]:
    #     """
    #     NewsAPI에서 기사 수집 (유료)
    #
    #     비용: 개발자 플랜 무료 (100 req/day, 1개월 지연)
    #           비즈니스 플랜 $449/월 (실시간)
    #
    #     환경변수: NEWSAPI_KEY
    #     """
    #     api_key = api_key or os.getenv("NEWSAPI_KEY")
    #     if not api_key:
    #         logger.warning("NewsAPI key not configured. Set NEWSAPI_KEY env var.")
    #         return []
    #
    #     url = "https://newsapi.org/v2/everything"
    #     params = {
    #         "q": query,
    #         "domains": domains,
    #         "language": "en",
    #         "sortBy": "publishedAt",
    #         "apiKey": api_key
    #     }
    #
    #     async with self._session.get(url, params=params) as response:
    #         if response.status != 200:
    #             logger.error(f"NewsAPI error: {response.status}")
    #             return []
    #
    #         data = await response.json()
    #
    #     signals = []
    #     for article in data.get("articles", [])[:10]:
    #         signal = ExternalSignal(
    #             signal_id=self._generate_signal_id("newsapi"),
    #             source=article.get("source", {}).get("name", "unknown"),
    #             tier=SignalTier.TIER3_AUTHORITY.value,
    #             title=article.get("title", ""),
    #             content=article.get("description", ""),
    #             url=article.get("url", ""),
    #             published_at=article.get("publishedAt", "")[:10],
    #             collected_at=datetime.now(KST).isoformat(),
    #             keywords=[],
    #             relevance_score=0.8
    #         )
    #         signals.append(signal)
    #
    #     return signals

    # ---------------------------------------------------------------------
    # Bing News Search API ($3/1000 transactions)
    # 웹사이트: https://www.microsoft.com/en-us/bing/apis/bing-news-search-api
    # ---------------------------------------------------------------------
    #
    # async def fetch_bing_news(
    #     self,
    #     query: str = "LANEIGE lip care",
    #     api_key: Optional[str] = None
    # ) -> List[ExternalSignal]:
    #     """
    #     Bing News Search API에서 기사 수집 (유료)
    #
    #     비용: $3 / 1,000 transactions
    #     무료 티어: 1,000 transactions/월
    #
    #     환경변수: BING_NEWS_API_KEY
    #     """
    #     api_key = api_key or os.getenv("BING_NEWS_API_KEY")
    #     if not api_key:
    #         logger.warning("Bing News API key not configured.")
    #         return []
    #
    #     url = "https://api.bing.microsoft.com/v7.0/news/search"
    #     headers = {"Ocp-Apim-Subscription-Key": api_key}
    #     params = {
    #         "q": query,
    #         "mkt": "en-US",
    #         "count": 10
    #     }
    #
    #     async with self._session.get(url, headers=headers, params=params) as response:
    #         if response.status != 200:
    #             logger.error(f"Bing News API error: {response.status}")
    #             return []
    #
    #         data = await response.json()
    #
    #     signals = []
    #     for article in data.get("value", []):
    #         signal = ExternalSignal(
    #             signal_id=self._generate_signal_id("bing_news"),
    #             source=article.get("provider", [{}])[0].get("name", "unknown"),
    #             tier=SignalTier.TIER3_AUTHORITY.value,
    #             title=article.get("name", ""),
    #             content=article.get("description", ""),
    #             url=article.get("url", ""),
    #             published_at=article.get("datePublished", "")[:10],
    #             collected_at=datetime.now(KST).isoformat(),
    #             keywords=[],
    #             relevance_score=0.8
    #         )
    #         signals.append(signal)
    #
    #     return signals

    # ---------------------------------------------------------------------
    # YouTube Data API (무료 10,000 quota/day)
    # 웹사이트: https://developers.google.com/youtube/v3
    # ---------------------------------------------------------------------
    #
    # async def fetch_youtube_trends(
    #     self,
    #     query: str = "LANEIGE lip sleeping mask review",
    #     api_key: Optional[str] = None,
    #     max_results: int = 10
    # ) -> List[ExternalSignal]:
    #     """
    #     YouTube Data API에서 영상 검색 (무료 할당량)
    #
    #     비용: 무료 (10,000 quota/day, 검색 1회 = 100 quota)
    #
    #     환경변수: YOUTUBE_API_KEY
    #     """
    #     api_key = api_key or os.getenv("YOUTUBE_API_KEY")
    #     if not api_key:
    #         logger.warning("YouTube API key not configured.")
    #         return []
    #
    #     url = "https://www.googleapis.com/youtube/v3/search"
    #     params = {
    #         "part": "snippet",
    #         "q": query,
    #         "type": "video",
    #         "order": "date",
    #         "maxResults": max_results,
    #         "key": api_key
    #     }
    #
    #     async with self._session.get(url, params=params) as response:
    #         if response.status != 200:
    #             logger.error(f"YouTube API error: {response.status}")
    #             return []
    #
    #         data = await response.json()
    #
    #     signals = []
    #     for item in data.get("items", []):
    #         snippet = item.get("snippet", {})
    #         video_id = item.get("id", {}).get("videoId", "")
    #
    #         signal = ExternalSignal(
    #             signal_id=self._generate_signal_id("youtube"),
    #             source=SignalSource.YOUTUBE.value,
    #             tier=SignalTier.TIER2_VALIDATION.value,
    #             title=snippet.get("title", ""),
    #             content=snippet.get("description", "")[:500],
    #             url=f"https://www.youtube.com/watch?v={video_id}",
    #             published_at=snippet.get("publishedAt", "")[:10],
    #             collected_at=datetime.now(KST).isoformat(),
    #             keywords=[],
    #             relevance_score=0.7,
    #             metadata={
    #                 "channel": snippet.get("channelTitle", ""),
    #                 "video_id": video_id
    #             }
    #         )
    #         signals.append(signal)
    #
    #     return signals

    # =========================================================================
    # 보고서 생성
    # =========================================================================

    def generate_report_section(
        self, tier_filter: list[SignalTier] | None = None, days: int = 7
    ) -> str:
        """
        보고서용 섹션 생성

        Args:
            tier_filter: 포함할 Tier (없으면 전체)
            days: 최근 N일 내 신호만

        Returns:
            보고서 섹션 문자열
        """
        cutoff_date = (datetime.now(KST) - timedelta(days=days)).strftime("%Y-%m-%d")

        filtered_signals = [s for s in self.signals if s.published_at >= cutoff_date]

        if tier_filter:
            tier_values = [t.value for t in tier_filter]
            filtered_signals = [s for s in filtered_signals if s.tier in tier_values]

        # Tier별 그룹핑
        by_tier = {
            SignalTier.TIER3_AUTHORITY.value: [],
            SignalTier.TIER2_VALIDATION.value: [],
            SignalTier.TIER1_VIRAL.value: [],
            SignalTier.TIER4_PR.value: [],
        }

        for signal in filtered_signals:
            if signal.tier in by_tier:
                by_tier[signal.tier].append(signal)

        # 보고서 생성
        sections = []

        # Tier 3: 전문 매체
        if by_tier[SignalTier.TIER3_AUTHORITY.value]:
            sections.append("■ 전문 매체 근거:")
            for signal in by_tier[SignalTier.TIER3_AUTHORITY.value][:5]:
                sections.append(signal.to_report_format())
            sections.append("")

        # Tier 2: 검증/리뷰
        if by_tier[SignalTier.TIER2_VALIDATION.value]:
            sections.append("■ 소비자 검증 (YouTube/Reddit):")
            for signal in by_tier[SignalTier.TIER2_VALIDATION.value][:5]:
                sections.append(signal.to_report_format())
            sections.append("")

        # Tier 1: 바이럴
        if by_tier[SignalTier.TIER1_VIRAL.value]:
            sections.append("■ 소비자 트렌드 (TikTok/Instagram):")
            for signal in by_tier[SignalTier.TIER1_VIRAL.value][:5]:
                sections.append(signal.to_report_format())
            sections.append("")

        return "\n".join(sections) if sections else "수집된 외부 신호가 없습니다."

    # =========================================================================
    # K-Beauty 전용 메서드 (인사이트 시스템 고도화)
    # =========================================================================

    async def fetch_kbeauty_news(self, max_articles: int = 20) -> list[ExternalSignal]:
        """
        K-Beauty 관련 뉴스만 필터링하여 수집

        글로벌 뷰티 매체에서 K-Beauty 관련 기사만 추출합니다.

        Args:
            max_articles: 최대 수집 기사 수

        Returns:
            ExternalSignal 리스트
        """
        all_signals = []

        # K-Beauty 전용 매체 우선
        kbeauty_sources = [
            SignalSource.COSMETICS_DESIGN_ASIA,
            SignalSource.KEDGLOBAL,
            SignalSource.KOREA_HERALD,
        ]

        for source in kbeauty_sources:
            if source in RSS_FEEDS:
                signals = await self.fetch_rss_articles(
                    source,
                    keywords=KBEAUTY_KEYWORDS,
                    max_articles=max_articles // len(kbeauty_sources),
                )
                all_signals.extend(signals)
                await asyncio.sleep(1)

        # 글로벌 매체에서도 K-Beauty 키워드로 필터링
        global_sources = [
            SignalSource.ALLURE,
            SignalSource.BYRDIE,
            SignalSource.COSMETICS_DESIGN_EUROPE,
        ]

        for source in global_sources:
            if source in RSS_FEEDS:
                signals = await self.fetch_rss_articles(
                    source, keywords=KBEAUTY_KEYWORDS, max_articles=5
                )
                all_signals.extend(signals)
                await asyncio.sleep(1)

        logger.info(f"Fetched {len(all_signals)} K-Beauty news articles")
        return all_signals

    async def fetch_industry_signals(
        self, keywords: list[str] | None = None
    ) -> list[ExternalSignal]:
        """
        뷰티 산업 전반의 신호 수집

        Args:
            keywords: 필터링 키워드 (없으면 기본 뷰티 키워드)

        Returns:
            ExternalSignal 리스트
        """
        all_signals = []

        # 모든 산업 매체에서 수집
        industry_sources = [
            SignalSource.COSMETICS_DESIGN_ASIA,
            SignalSource.COSMETICS_DESIGN_EUROPE,
            SignalSource.KEDGLOBAL,
        ]

        for source in industry_sources:
            if source in RSS_FEEDS:
                signals = await self.fetch_rss_articles(source, keywords=keywords, max_articles=10)
                all_signals.extend(signals)
                await asyncio.sleep(1)

        # Reddit에서도 수집
        reddit_signals = await self.fetch_reddit_trends(
            subreddits=["SkincareAddiction", "AsianBeauty"], keywords=keywords, max_posts=10
        )
        all_signals.extend(reddit_signals)

        logger.info(f"Fetched {len(all_signals)} industry signals")
        return all_signals

    def get_source_reliability(self, source: str) -> float:
        """
        매체별 신뢰도 점수 반환

        출처 관리 시스템 연동용

        Args:
            source: 매체 이름

        Returns:
            신뢰도 점수 (0.0 ~ 1.0)
        """
        reliability_map = {
            # 전문 매체 (Tier 3)
            "allure": 0.8,
            "byrdie": 0.8,
            "refinery29": 0.75,
            "cosmetics_design_asia": 0.85,
            "cosmetics_design_europe": 0.85,
            "kedglobal": 0.8,
            "korea_herald": 0.8,
            "cosinkorea": 0.8,
            "wwd_beauty": 0.85,
            "vogue_beauty": 0.8,
            # Tavily 뉴스 (신뢰도는 도메인별로 다름, 기본값)
            "tavily_news": 0.85,
            # 검증/리뷰 (Tier 2)
            "youtube": 0.6,
            "reddit": 0.5,
            # 바이럴 (Tier 1)
            "tiktok": 0.5,
            "instagram": 0.5,
            # PR (Tier 4)
            "twitter": 0.4,
            # 수동 입력
            "manual": 0.9,
            "trend_radar": 0.9,
        }

        return reliability_map.get(source.lower(), 0.5)

    def create_source_reference(self, signal: ExternalSignal) -> dict[str, Any]:
        """
        출처 참조 객체 생성 (SourceManager 연동용)

        Args:
            signal: ExternalSignal 객체

        Returns:
            Source 형식 딕셔너리
        """
        return {
            "id": signal.signal_id,
            "title": signal.title,
            "publisher": signal.source.replace("_", " ").title(),
            "date": signal.published_at,
            "url": signal.url,
            "source_type": "news" if signal.tier == SignalTier.TIER3_AUTHORITY.value else "sns",
            "reliability_score": self.get_source_reliability(signal.source),
        }

    # =========================================================================
    # Tavily API 검색 (실시간 뉴스)
    # =========================================================================

    async def fetch_tavily_news(
        self,
        brands: list[str] | None = None,
        topics: list[str] | None = None,
        days: int = 7,
        max_results: int = 10,
    ) -> list[ExternalSignal]:
        """
        Tavily API를 통한 실시간 뉴스 검색

        Args:
            brands: 브랜드명 리스트 (예: ["LANEIGE", "COSRX"])
            topics: 토픽 리스트 (예: ["K-Beauty", "skincare trends"])
            days: 검색 기간 (최근 N일)
            max_results: 최대 결과 수

        Returns:
            ExternalSignal 리스트

        Example:
            >>> signals = await collector.fetch_tavily_news(
            ...     brands=["LANEIGE", "COSRX"],
            ...     topics=["K-Beauty trends"],
            ...     days=7
            ... )
        """
        if not self._tavily_client:
            logger.warning("Tavily client not initialized. Skipping Tavily news search.")
            return []

        try:
            # Tavily 검색 실행
            results = await self._tavily_client.search_beauty_news(
                brands=brands,
                topics=topics,
                days=days,
                max_results_per_query=max_results // 2 if max_results > 2 else max_results,
            )

            signals = []
            for result in results[:max_results]:
                # Tavily 결과 → ExternalSignal 변환
                signal = ExternalSignal(
                    signal_id=self._generate_signal_id("tavily"),
                    source="tavily_news",
                    tier=SignalTier.TIER3_AUTHORITY.value,  # 뉴스 = Tier 3
                    title=result.title,
                    content=result.content[:500] if result.content else "",
                    url=result.url,
                    published_at=result.published_date or datetime.now(KST).strftime("%Y-%m-%d"),
                    collected_at=datetime.now(KST).isoformat(),
                    keywords=brands or [],
                    relevance_score=result.score,
                    metadata={
                        "domain": result.source,
                        "reliability_score": result.reliability_score,
                        "tavily_score": result.score,
                        "search_type": "tavily_api",
                    },
                )
                signals.append(signal)

            # 수집된 신호 저장
            self.signals.extend(signals)
            self._save_signals()

            logger.info(f"Fetched {len(signals)} news articles via Tavily API")
            return signals

        except Exception as e:
            logger.error(f"Tavily news fetch failed: {e}")
            return []

    async def fetch_all_news(
        self,
        brands: list[str] | None = None,
        keywords: list[str] | None = None,
        days: int = 7,
        include_tavily: bool = True,
        include_rss: bool = True,
        include_reddit: bool = True,
    ) -> list[ExternalSignal]:
        """
        모든 소스에서 뉴스/신호 수집 (통합 메서드)

        Args:
            brands: 브랜드명 리스트
            keywords: 키워드 리스트
            days: 검색 기간
            include_tavily: Tavily 뉴스 포함 여부
            include_rss: RSS 피드 포함 여부
            include_reddit: Reddit 포함 여부

        Returns:
            ExternalSignal 리스트 (중복 제거, 신뢰도 정렬)
        """
        all_signals = []

        # 1. Tavily API (실시간 뉴스)
        if include_tavily and self._tavily_client:
            tavily_signals = await self.fetch_tavily_news(brands=brands, topics=keywords, days=days)
            all_signals.extend(tavily_signals)

        # 2. RSS 피드 (전문 매체)
        if include_rss:
            rss_signals = await self.fetch_all_rss_feeds(keywords=keywords or BEAUTY_KEYWORDS)
            all_signals.extend(rss_signals)

        # 3. Reddit (소비자 검증)
        if include_reddit:
            reddit_signals = await self.fetch_reddit_trends(
                subreddits=["SkincareAddiction", "AsianBeauty", "MakeupAddiction"],
                keywords=keywords,
                max_posts=10,
            )
            all_signals.extend(reddit_signals)

        # 중복 제거 (URL 기준)
        seen_urls = set()
        unique_signals = []
        for signal in all_signals:
            if signal.url not in seen_urls:
                seen_urls.add(signal.url)
                unique_signals.append(signal)

        # 신뢰도/관련성 기반 정렬
        unique_signals.sort(
            key=lambda s: (
                s.metadata.get("reliability_score", 0.5) * 0.6 + s.relevance_score * 0.4
            ),
            reverse=True,
        )

        logger.info(f"Fetched {len(unique_signals)} unique signals from all sources")
        return unique_signals

    # =========================================================================
    # 유틸리티
    # =========================================================================

    def _generate_signal_id(self, source: str) -> str:
        """신호 ID 생성"""
        self._signal_seq += 1
        date_str = datetime.now(KST).strftime("%Y%m%d")
        return f"SID-{source.upper()[:3]}-{date_str}-{self._signal_seq:04d}"

    def _save_signals(self) -> None:
        """신호 저장"""
        filepath = self.data_dir / "signals.json"
        data = {
            "signals": [s.to_dict() for s in self.signals],
            "updated_at": datetime.now(KST).isoformat(),
            "count": len(self.signals),
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _load_signals(self) -> None:
        """신호 로드"""
        filepath = self.data_dir / "signals.json"
        if not filepath.exists():
            return

        try:
            with open(filepath, encoding="utf-8") as f:
                data = json.load(f)

            self.signals = [ExternalSignal(**s) for s in data.get("signals", [])]
            self._signal_seq = len(self.signals)
            logger.info(f"Loaded {len(self.signals)} signals")
        except Exception as e:
            logger.warning(f"Failed to load signals: {e}")

    def get_signals_for_kg(self) -> list[dict[str, Any]]:
        """지식그래프에 추가할 형식으로 반환"""
        return [s.to_dict() for s in self.signals]

    def get_stats(self) -> dict[str, Any]:
        """통계 반환"""
        by_tier = {}
        by_source = {}

        for signal in self.signals:
            by_tier[signal.tier] = by_tier.get(signal.tier, 0) + 1
            by_source[signal.source] = by_source.get(signal.source, 0) + 1

        return {
            "total_signals": len(self.signals),
            "by_tier": by_tier,
            "by_source": by_source,
            "last_updated": self.signals[-1].collected_at if self.signals else None,
        }
