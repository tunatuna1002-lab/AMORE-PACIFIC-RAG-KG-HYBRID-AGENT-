"""
TikTok Collector
================
TikTok에서 K-Beauty/LANEIGE 관련 콘텐츠를 수집하는 모듈

## 사용 기술
- Playwright 기반 스크래핑 (TikTok-Api 불안정으로 직접 구현)
- JSON API 엔드포인트 활용

## 수집 대상
- 해시태그: #laneige, #kbeauty, #skincare, #lipsleepmask
- 프로필: 뷰티 인플루언서

## 사용 예시
```python
collector = TikTokCollector()
await collector.initialize()

# 해시태그 검색
posts = await collector.search_hashtag("laneige", limit=100)

# 트렌딩 검색
trending = await collector.get_trending_beauty(limit=50)
```
"""

import asyncio
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.parse import quote

try:
    from playwright.async_api import Browser, Page, async_playwright

    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False

try:
    import importlib.util

    AIOHTTP_AVAILABLE = importlib.util.find_spec("aiohttp") is not None
except Exception:
    AIOHTTP_AVAILABLE = False

logger = logging.getLogger(__name__)

# 한국 시간대
KST = timezone(timedelta(hours=9))


@dataclass
class TikTokPost:
    """TikTok 포스트 데이터"""

    post_id: str
    author: str
    author_id: str
    description: str
    hashtags: list[str]
    likes: int
    comments: int
    shares: int
    views: int
    music_title: str | None = None
    created_at: datetime | None = None
    video_url: str | None = None
    thumbnail_url: str | None = None
    collected_at: datetime = field(default_factory=lambda: datetime.now(KST))

    def to_dict(self) -> dict[str, Any]:
        return {
            "post_id": self.post_id,
            "author": self.author,
            "author_id": self.author_id,
            "description": self.description,
            "hashtags": self.hashtags,
            "likes": self.likes,
            "comments": self.comments,
            "shares": self.shares,
            "views": self.views,
            "music_title": self.music_title,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "video_url": self.video_url,
            "thumbnail_url": self.thumbnail_url,
            "collected_at": self.collected_at.isoformat(),
        }

    def get_engagement_rate(self) -> float:
        """인게이지먼트율 계산"""
        if self.views == 0:
            return 0.0
        return (self.likes + self.comments + self.shares) / self.views * 100


class TikTokCollector:
    """
    TikTok 데이터 수집기

    Playwright 기반으로 TikTok 웹에서 데이터 수집
    """

    # K-Beauty 관련 해시태그
    KBEAUTY_HASHTAGS = [
        "laneige",
        "kbeauty",
        "koreanbeauty",
        "koreanskincare",
        "lipsleepmask",
        "waterbank",
        "라네즈",
        "skincare",
        "skincareroutine",
        "glowyskin",
        "cosrx",
        "tirtir",
        "biodance",
        "anua",
    ]

    # 뷰티 인플루언서 탐지 키워드
    BEAUTY_KEYWORDS = [
        "skincare",
        "beauty",
        "makeup",
        "routine",
        "review",
        "glow",
        "hydration",
        "moisturizer",
        "serum",
        "cream",
    ]

    def __init__(self, headless: bool = True, proxy: str | None = None, request_delay: float = 2.0):
        """
        Args:
            headless: 브라우저 헤드리스 모드
            proxy: 프록시 서버 URL (선택)
            request_delay: 요청 간 딜레이 (초)
        """
        self.headless = headless
        self.proxy = proxy
        self.request_delay = request_delay
        self._browser: Browser | None = None
        self._page: Page | None = None
        self._initialized = False
        self._posts: list[TikTokPost] = []

    async def initialize(self) -> bool:
        """브라우저 초기화"""
        if not PLAYWRIGHT_AVAILABLE:
            logger.warning("Playwright not available. TikTok collection disabled.")
            return False

        try:
            playwright = await async_playwright().start()

            launch_options = {
                "headless": self.headless,
            }

            if self.proxy:
                launch_options["proxy"] = {"server": self.proxy}

            self._browser = await playwright.chromium.launch(**launch_options)
            self._page = await self._browser.new_page()

            # User Agent 설정
            await self._page.set_extra_http_headers(
                {
                    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                }
            )

            self._initialized = True
            logger.info("TikTok collector initialized")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize TikTok collector: {e}")
            return False

    async def close(self):
        """브라우저 종료"""
        if self._browser:
            await self._browser.close()
            self._browser = None
            self._page = None
            self._initialized = False

    async def search_hashtag(self, hashtag: str, limit: int = 50) -> list[TikTokPost]:
        """
        해시태그로 포스트 검색

        Args:
            hashtag: 검색할 해시태그 (# 제외)
            limit: 최대 수집 개수

        Returns:
            TikTokPost 리스트
        """
        if not self._initialized:
            logger.warning("Collector not initialized")
            return []

        posts = []
        hashtag_clean = hashtag.replace("#", "").lower()

        try:
            url = f"https://www.tiktok.com/tag/{quote(hashtag_clean)}"
            await self._page.goto(url, wait_until="networkidle", timeout=30000)
            await asyncio.sleep(self.request_delay)

            # 스크롤하며 포스트 로드
            scroll_count = min(limit // 10 + 1, 5)  # 최대 5번 스크롤

            for _ in range(scroll_count):
                await self._page.evaluate("window.scrollBy(0, 1000)")
                await asyncio.sleep(1)

            # 포스트 데이터 추출
            posts_data = await self._page.evaluate("""
                () => {
                    const posts = [];
                    const items = document.querySelectorAll('[data-e2e="challenge-item"]');

                    items.forEach((item, index) => {
                        try {
                            const link = item.querySelector('a');
                            const desc = item.querySelector('[data-e2e="challenge-item-desc"]');
                            const stats = item.querySelectorAll('[data-e2e="video-views"]');

                            posts.push({
                                post_id: link?.href?.split('/video/')[1]?.split('?')[0] || `post_${index}`,
                                description: desc?.textContent || '',
                                views_text: stats[0]?.textContent || '0',
                                url: link?.href || ''
                            });
                        } catch (e) {
                            console.error(e);
                        }
                    });

                    return posts;
                }
            """)

            # TikTokPost 객체로 변환
            for data in posts_data[:limit]:
                post = TikTokPost(
                    post_id=data.get("post_id", ""),
                    author="",  # 상세 페이지에서 추출 필요
                    author_id="",
                    description=data.get("description", ""),
                    hashtags=self._extract_hashtags(data.get("description", "")),
                    likes=0,
                    comments=0,
                    shares=0,
                    views=self._parse_views(data.get("views_text", "0")),
                    video_url=data.get("url", ""),
                )
                posts.append(post)

            logger.info(f"Collected {len(posts)} posts for #{hashtag_clean}")

        except Exception as e:
            logger.error(f"Error searching hashtag {hashtag}: {e}")

        return posts

    async def search_kbeauty(self, limit: int = 100) -> list[TikTokPost]:
        """
        K-Beauty 관련 포스트 종합 수집

        Args:
            limit: 총 수집 개수

        Returns:
            TikTokPost 리스트
        """
        all_posts = []
        per_hashtag = limit // len(self.KBEAUTY_HASHTAGS[:5])  # 상위 5개 해시태그

        for hashtag in self.KBEAUTY_HASHTAGS[:5]:
            posts = await self.search_hashtag(hashtag, limit=per_hashtag)
            all_posts.extend(posts)
            await asyncio.sleep(self.request_delay)

        # 중복 제거
        seen_ids = set()
        unique_posts = []
        for post in all_posts:
            if post.post_id not in seen_ids:
                seen_ids.add(post.post_id)
                unique_posts.append(post)

        self._posts = unique_posts
        return unique_posts

    def _extract_hashtags(self, text: str) -> list[str]:
        """텍스트에서 해시태그 추출"""
        return re.findall(r"#(\w+)", text.lower())

    def _parse_views(self, views_text: str) -> int:
        """조회수 텍스트를 숫자로 변환"""
        views_text = views_text.lower().strip()

        try:
            if "k" in views_text:
                return int(float(views_text.replace("k", "")) * 1000)
            elif "m" in views_text:
                return int(float(views_text.replace("m", "")) * 1000000)
            elif "b" in views_text:
                return int(float(views_text.replace("b", "")) * 1000000000)
            else:
                return int(views_text.replace(",", ""))
        except (ValueError, AttributeError):
            return 0

    def generate_report_section(self) -> str:
        """리포트용 섹션 생성"""
        if not self._posts:
            return "TikTok 데이터가 수집되지 않았습니다."

        total_views = sum(p.views for p in self._posts)
        sum(p.likes for p in self._posts)
        avg_engagement = (
            sum(p.get_engagement_rate() for p in self._posts) / len(self._posts)
            if self._posts
            else 0
        )

        # 해시태그 빈도
        hashtag_freq: dict[str, int] = {}
        for post in self._posts:
            for tag in post.hashtags:
                hashtag_freq[tag] = hashtag_freq.get(tag, 0) + 1

        top_hashtags = sorted(hashtag_freq.items(), key=lambda x: x[1], reverse=True)[:10]

        lines = [
            "### TikTok 트렌드 분석",
            f"- 분석 포스트: {len(self._posts)}개",
            f"- 총 조회수: {total_views:,}",
            f"- 평균 인게이지먼트율: {avg_engagement:.2f}%",
            "",
            "**인기 해시태그:**",
        ]

        for tag, count in top_hashtags[:5]:
            lines.append(f"- #{tag}: {count}회")

        return "\n".join(lines)

    @property
    def posts(self) -> list[TikTokPost]:
        return self._posts


# 테스트용
async def main():
    collector = TikTokCollector(headless=True)

    if await collector.initialize():
        posts = await collector.search_hashtag("laneige", limit=10)

        for post in posts:
            print(f"[{post.post_id}] Views: {post.views:,} | {post.description[:50]}...")

        print("\n" + collector.generate_report_section())

        await collector.close()


if __name__ == "__main__":
    asyncio.run(main())
