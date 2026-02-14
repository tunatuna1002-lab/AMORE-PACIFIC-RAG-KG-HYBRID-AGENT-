"""
Reddit Collector
================
Reddit에서 K-Beauty/LANEIGE 관련 포스트와 댓글을 수집하는 모듈

## 사용 기술
- Reddit JSON API (API 키 불필요)
- .json 엔드포인트 활용

## 수집 대상
- 서브레딧: r/AsianBeauty, r/SkincareAddiction, r/MakeupAddiction
- 키워드: LANEIGE, K-Beauty, Korean skincare

## 사용 예시
```python
collector = RedditCollector()
await collector.initialize()

# 서브레딧 포스트
posts = await collector.get_subreddit_posts("AsianBeauty", limit=50)

# 키워드 검색
results = await collector.search("LANEIGE review", limit=30)
```

## 주의사항
- Rate limit: 분당 30 요청 권장
- 공개 데이터만 수집
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from urllib.parse import quote

from src.shared.constants import KST

try:
    import aiohttp

    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False

logger = logging.getLogger(__name__)


# 한국 시간대
@dataclass
class RedditPost:
    """Reddit 포스트 데이터"""

    post_id: str
    title: str
    author: str
    subreddit: str
    selftext: str
    score: int
    upvote_ratio: float
    num_comments: int
    created_utc: datetime
    url: str
    permalink: str
    is_self: bool
    flair: str | None = None
    collected_at: datetime = field(default_factory=lambda: datetime.now(KST))

    def to_dict(self) -> dict[str, Any]:
        return {
            "post_id": self.post_id,
            "title": self.title,
            "author": self.author,
            "subreddit": self.subreddit,
            "selftext": self.selftext[:500] if self.selftext else "",
            "score": self.score,
            "upvote_ratio": self.upvote_ratio,
            "num_comments": self.num_comments,
            "created_utc": self.created_utc.isoformat(),
            "url": self.url,
            "permalink": self.permalink,
            "is_self": self.is_self,
            "flair": self.flair,
            "collected_at": self.collected_at.isoformat(),
        }


class RedditCollector:
    """
    Reddit 데이터 수집기

    Reddit JSON API 기반 (API 키 불필요)
    """

    # K-Beauty 관련 서브레딧
    KBEAUTY_SUBREDDITS = [
        "AsianBeauty",
        "SkincareAddiction",
        "MakeupAddiction",
        "KoreanBeauty",
        "30PlusSkinCare",
    ]

    # 검색 키워드
    SEARCH_KEYWORDS = [
        "LANEIGE",
        "lip sleeping mask",
        "water bank",
        "Korean skincare",
        "K-Beauty",
        "COSRX",
        "TIRTIR",
    ]

    BASE_URL = "https://www.reddit.com"

    def __init__(self, request_delay: float = 2.0, user_agent: str = "KBeautyAnalyzer/1.0"):
        """
        Args:
            request_delay: 요청 간 딜레이 (초)
            user_agent: User-Agent 헤더
        """
        self.request_delay = request_delay
        self.user_agent = user_agent
        self._session: aiohttp.ClientSession | None = None
        self._posts: list[RedditPost] = []
        self._initialized = False

    async def initialize(self) -> bool:
        """세션 초기화"""
        if not AIOHTTP_AVAILABLE:
            logger.warning("aiohttp not available. Reddit collection disabled.")
            return False

        try:
            self._session = aiohttp.ClientSession(headers={"User-Agent": self.user_agent})
            self._initialized = True
            logger.info("Reddit collector initialized")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize Reddit collector: {e}")
            return False

    async def close(self):
        """세션 종료"""
        if self._session:
            await self._session.close()
            self._session = None
            self._initialized = False

    async def _fetch_json(self, url: str) -> dict[str, Any] | None:
        """JSON 데이터 가져오기"""
        if not self._session:
            return None

        try:
            async with self._session.get(url, timeout=aiohttp.ClientTimeout(total=30)) as response:
                if response.status == 200:
                    return await response.json()
                elif response.status == 429:
                    logger.warning("Reddit rate limit hit, waiting...")
                    await asyncio.sleep(60)
                    return None
                else:
                    logger.warning(f"Reddit API returned {response.status}")
                    return None

        except TimeoutError:
            logger.warning(f"Timeout fetching {url}")
            return None
        except Exception as e:
            logger.error(f"Error fetching {url}: {e}")
            return None

    def _parse_post(self, data: dict[str, Any]) -> RedditPost | None:
        """JSON 데이터를 RedditPost로 변환"""
        try:
            post_data = data.get("data", data)

            created_utc = datetime.fromtimestamp(post_data.get("created_utc", 0), tz=UTC)

            return RedditPost(
                post_id=post_data.get("id", ""),
                title=post_data.get("title", ""),
                author=post_data.get("author", "[deleted]"),
                subreddit=post_data.get("subreddit", ""),
                selftext=post_data.get("selftext", "")[:1000],
                score=post_data.get("score", 0),
                upvote_ratio=post_data.get("upvote_ratio", 0.0),
                num_comments=post_data.get("num_comments", 0),
                created_utc=created_utc,
                url=post_data.get("url", ""),
                permalink=f"https://www.reddit.com{post_data.get('permalink', '')}",
                is_self=post_data.get("is_self", False),
                flair=post_data.get("link_flair_text"),
            )

        except Exception as e:
            logger.debug(f"Error parsing post: {e}")
            return None

    async def get_subreddit_posts(
        self, subreddit: str, sort: str = "hot", limit: int = 50, time_filter: str = "week"
    ) -> list[RedditPost]:
        """
        서브레딧 포스트 가져오기

        Args:
            subreddit: 서브레딧 이름 (r/ 제외)
            sort: hot, new, top, rising
            limit: 최대 개수 (max 100)
            time_filter: hour, day, week, month, year, all (sort=top일 때만)

        Returns:
            RedditPost 리스트
        """
        if not self._initialized:
            await self.initialize()

        posts = []
        limit = min(limit, 100)

        url = f"{self.BASE_URL}/r/{subreddit}/{sort}.json?limit={limit}"
        if sort == "top":
            url += f"&t={time_filter}"

        data = await self._fetch_json(url)

        if data and "data" in data and "children" in data["data"]:
            for child in data["data"]["children"]:
                post = self._parse_post(child)
                if post:
                    posts.append(post)

        logger.info(f"Fetched {len(posts)} posts from r/{subreddit}")
        return posts

    async def search(
        self,
        query: str,
        subreddit: str | None = None,
        sort: str = "relevance",
        time_filter: str = "month",
        limit: int = 50,
    ) -> list[RedditPost]:
        """
        Reddit 검색

        Args:
            query: 검색 쿼리
            subreddit: 특정 서브레딧 (선택)
            sort: relevance, hot, top, new, comments
            time_filter: hour, day, week, month, year, all
            limit: 최대 개수

        Returns:
            RedditPost 리스트
        """
        if not self._initialized:
            await self.initialize()

        posts = []
        limit = min(limit, 100)
        encoded_query = quote(query)

        if subreddit:
            url = f"{self.BASE_URL}/r/{subreddit}/search.json?q={encoded_query}&restrict_sr=1&sort={sort}&t={time_filter}&limit={limit}"
        else:
            url = f"{self.BASE_URL}/search.json?q={encoded_query}&sort={sort}&t={time_filter}&limit={limit}"

        data = await self._fetch_json(url)

        if data and "data" in data and "children" in data["data"]:
            for child in data["data"]["children"]:
                post = self._parse_post(child)
                if post:
                    posts.append(post)

        logger.info(f"Found {len(posts)} posts for '{query}'")
        return posts

    async def search_kbeauty(self, limit: int = 100) -> list[RedditPost]:
        """
        K-Beauty 관련 포스트 종합 수집

        Args:
            limit: 총 수집 개수

        Returns:
            RedditPost 리스트
        """
        all_posts = []

        # 1. 주요 서브레딧에서 최신 포스트
        for subreddit in self.KBEAUTY_SUBREDDITS[:3]:
            posts = await self.get_subreddit_posts(subreddit, sort="hot", limit=20)
            all_posts.extend(posts)
            await asyncio.sleep(self.request_delay)

        # 2. LANEIGE 관련 검색
        for keyword in ["LANEIGE", "lip sleeping mask"]:
            posts = await self.search(
                keyword, subreddit="AsianBeauty", sort="relevance", time_filter="month", limit=20
            )
            all_posts.extend(posts)
            await asyncio.sleep(self.request_delay)

        # 중복 제거
        seen_ids = set()
        unique_posts = []
        for post in all_posts:
            if post.post_id not in seen_ids:
                seen_ids.add(post.post_id)
                unique_posts.append(post)

        self._posts = unique_posts[:limit]
        return self._posts

    def analyze_sentiment_keywords(self) -> dict[str, int]:
        """
        포스트에서 감성 키워드 분석

        Returns:
            키워드 빈도 딕셔너리
        """
        positive_keywords = [
            "love",
            "amazing",
            "best",
            "holy grail",
            "recommend",
            "favorite",
            "obsessed",
            "repurchase",
            "works",
            "effective",
            "hydrating",
        ]
        negative_keywords = [
            "hate",
            "worst",
            "disappointed",
            "broke out",
            "irritation",
            "waste",
            "overrated",
            "not worth",
            "returned",
            "allergic",
        ]

        keyword_counts = {"positive": {}, "negative": {}}

        for post in self._posts:
            text = (post.title + " " + post.selftext).lower()

            for kw in positive_keywords:
                if kw in text:
                    keyword_counts["positive"][kw] = keyword_counts["positive"].get(kw, 0) + 1

            for kw in negative_keywords:
                if kw in text:
                    keyword_counts["negative"][kw] = keyword_counts["negative"].get(kw, 0) + 1

        return keyword_counts

    def generate_report_section(self) -> str:
        """리포트용 섹션 생성"""
        if not self._posts:
            return "Reddit 데이터가 수집되지 않았습니다."

        total_score = sum(p.score for p in self._posts)
        total_comments = sum(p.num_comments for p in self._posts)
        avg_upvote_ratio = (
            sum(p.upvote_ratio for p in self._posts) / len(self._posts) if self._posts else 0
        )

        # 서브레딧별 분포
        subreddit_counts: dict[str, int] = {}
        for post in self._posts:
            subreddit_counts[post.subreddit] = subreddit_counts.get(post.subreddit, 0) + 1

        # 인기 포스트
        top_posts = sorted(self._posts, key=lambda x: x.score, reverse=True)[:5]

        # 감성 분석
        sentiment = self.analyze_sentiment_keywords()
        positive_total = sum(sentiment["positive"].values())
        negative_total = sum(sentiment["negative"].values())

        lines = [
            "### Reddit 커뮤니티 분석",
            f"- 분석 포스트: {len(self._posts)}개",
            f"- 총 점수: {total_score:,}",
            f"- 총 댓글: {total_comments:,}",
            f"- 평균 지지율: {avg_upvote_ratio:.1%}",
            "",
            "**서브레딧 분포:**",
        ]

        for sub, count in sorted(subreddit_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
            lines.append(f"- r/{sub}: {count}개")

        lines.append("")
        lines.append("**감성 지표:**")
        lines.append(f"- 긍정 키워드: {positive_total}회")
        lines.append(f"- 부정 키워드: {negative_total}회")

        if positive_total + negative_total > 0:
            sentiment_ratio = positive_total / (positive_total + negative_total)
            lines.append(f"- 긍정 비율: {sentiment_ratio:.1%}")

        if top_posts:
            lines.append("")
            lines.append("**인기 포스트:**")
            for post in top_posts[:3]:
                lines.append(f"- [{post.title[:40]}...] {post.score:,} points")

        return "\n".join(lines)

    @property
    def posts(self) -> list[RedditPost]:
        return self._posts


# 테스트용
async def main():
    collector = RedditCollector()

    if await collector.initialize():
        # 서브레딧 포스트
        posts = await collector.get_subreddit_posts("AsianBeauty", limit=10)

        for post in posts:
            print(f"[{post.score:>5}] r/{post.subreddit} | {post.title[:50]}...")

        # 검색
        print("\n--- LANEIGE 검색 ---")
        search_results = await collector.search("LANEIGE", subreddit="AsianBeauty", limit=5)

        for post in search_results:
            print(f"[{post.score:>5}] {post.title[:50]}...")

        print("\n" + collector.generate_report_section())

        await collector.close()


if __name__ == "__main__":
    asyncio.run(main())
