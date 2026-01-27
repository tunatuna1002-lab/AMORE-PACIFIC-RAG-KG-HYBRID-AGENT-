"""
Instagram Collector
===================
Instagram에서 K-Beauty/LANEIGE 관련 콘텐츠를 수집하는 모듈

## 사용 기술
- Instaloader 라이브러리 (무료, 오픈소스)
- 로그인 없이 공개 데이터 수집

## 수집 대상
- 해시태그: #laneige, #kbeauty, #라네즈
- 공개 프로필 포스트

## 사용 예시
```python
collector = InstagramCollector()

# 해시태그 검색
posts = await collector.search_hashtag("laneige", limit=50)

# K-Beauty 종합 검색
all_posts = await collector.search_kbeauty(limit=100)
```

## 주의사항
- Instagram은 rate limiting이 엄격함
- 프록시 사용 권장
- 개인 계정 사용 금지 (차단 위험)
"""

import asyncio
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor

try:
    import instaloader
    INSTALOADER_AVAILABLE = True
except ImportError:
    INSTALOADER_AVAILABLE = False

logger = logging.getLogger(__name__)

# 한국 시간대
KST = timezone(timedelta(hours=9))


@dataclass
class InstagramPost:
    """Instagram 포스트 데이터"""
    post_id: str
    shortcode: str
    author: str
    caption: str
    hashtags: List[str]
    likes: int
    comments: int
    is_video: bool
    video_view_count: Optional[int] = None
    created_at: Optional[datetime] = None
    url: Optional[str] = None
    thumbnail_url: Optional[str] = None
    collected_at: datetime = field(default_factory=lambda: datetime.now(KST))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "post_id": self.post_id,
            "shortcode": self.shortcode,
            "author": self.author,
            "caption": self.caption,
            "hashtags": self.hashtags,
            "likes": self.likes,
            "comments": self.comments,
            "is_video": self.is_video,
            "video_view_count": self.video_view_count,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "url": self.url,
            "thumbnail_url": self.thumbnail_url,
            "collected_at": self.collected_at.isoformat(),
        }

    def get_engagement_rate(self, follower_count: int = 0) -> float:
        """인게이지먼트율 계산"""
        if follower_count == 0:
            return 0.0
        return (self.likes + self.comments) / follower_count * 100


class InstagramCollector:
    """
    Instagram 데이터 수집기

    Instaloader 기반으로 공개 데이터 수집
    """

    # K-Beauty 관련 해시태그
    KBEAUTY_HASHTAGS = [
        "laneige", "kbeauty", "koreanbeauty", "koreanskincare",
        "lipsleepmask", "waterbank", "라네즈",
        "skincare", "glowyskin", "skincarekorea",
        "cosrx", "tirtir", "biodance", "anua"
    ]

    def __init__(
        self,
        request_delay: float = 3.0,
        max_retries: int = 3
    ):
        """
        Args:
            request_delay: 요청 간 딜레이 (초)
            max_retries: 재시도 횟수
        """
        self.request_delay = request_delay
        self.max_retries = max_retries
        self._loader: Optional[instaloader.Instaloader] = None
        self._posts: List[InstagramPost] = []
        self._initialized = False
        self._executor = ThreadPoolExecutor(max_workers=1)

    def initialize(self) -> bool:
        """Instaloader 초기화"""
        if not INSTALOADER_AVAILABLE:
            logger.warning("Instaloader not available. Instagram collection disabled.")
            return False

        try:
            self._loader = instaloader.Instaloader(
                download_pictures=False,
                download_videos=False,
                download_video_thumbnails=False,
                download_geotags=False,
                download_comments=False,
                save_metadata=False,
                compress_json=False,
                quiet=True,
                request_timeout=30
            )

            # Rate limit 설정
            self._loader.context._session.headers.update({
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
            })

            self._initialized = True
            logger.info("Instagram collector initialized")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize Instagram collector: {e}")
            return False

    def _search_hashtag_sync(self, hashtag: str, limit: int) -> List[InstagramPost]:
        """동기 방식 해시태그 검색 (스레드에서 실행)"""
        posts = []
        hashtag_clean = hashtag.replace("#", "").lower()

        try:
            hashtag_obj = instaloader.Hashtag.from_name(self._loader.context, hashtag_clean)
            post_iterator = hashtag_obj.get_posts()

            count = 0
            for post in post_iterator:
                if count >= limit:
                    break

                try:
                    ig_post = InstagramPost(
                        post_id=str(post.mediaid),
                        shortcode=post.shortcode,
                        author=post.owner_username,
                        caption=post.caption or "",
                        hashtags=list(post.caption_hashtags) if post.caption_hashtags else [],
                        likes=post.likes,
                        comments=post.comments,
                        is_video=post.is_video,
                        video_view_count=post.video_view_count if post.is_video else None,
                        created_at=post.date_utc.replace(tzinfo=timezone.utc) if post.date_utc else None,
                        url=f"https://www.instagram.com/p/{post.shortcode}/",
                        thumbnail_url=post.url if not post.is_video else None
                    )
                    posts.append(ig_post)
                    count += 1

                    # Rate limiting
                    if count % 10 == 0:
                        import time
                        time.sleep(self.request_delay)

                except Exception as e:
                    logger.debug(f"Error processing post: {e}")
                    continue

        except instaloader.exceptions.QueryReturnedNotFoundException:
            logger.warning(f"Hashtag not found: {hashtag_clean}")
        except instaloader.exceptions.ConnectionException as e:
            logger.error(f"Connection error for #{hashtag_clean}: {e}")
        except Exception as e:
            logger.error(f"Error searching hashtag {hashtag_clean}: {e}")

        return posts

    async def search_hashtag(self, hashtag: str, limit: int = 50) -> List[InstagramPost]:
        """
        해시태그로 포스트 검색

        Args:
            hashtag: 검색할 해시태그 (# 제외)
            limit: 최대 수집 개수

        Returns:
            InstagramPost 리스트
        """
        if not self._initialized:
            if not self.initialize():
                return []

        loop = asyncio.get_event_loop()
        posts = await loop.run_in_executor(
            self._executor,
            self._search_hashtag_sync,
            hashtag,
            limit
        )

        logger.info(f"Collected {len(posts)} posts for #{hashtag}")
        return posts

    async def search_kbeauty(self, limit: int = 100) -> List[InstagramPost]:
        """
        K-Beauty 관련 포스트 종합 수집

        Args:
            limit: 총 수집 개수

        Returns:
            InstagramPost 리스트
        """
        all_posts = []
        per_hashtag = max(limit // len(self.KBEAUTY_HASHTAGS[:5]), 10)

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

        self._posts = unique_posts[:limit]
        return self._posts

    def generate_report_section(self) -> str:
        """리포트용 섹션 생성"""
        if not self._posts:
            return "Instagram 데이터가 수집되지 않았습니다."

        total_likes = sum(p.likes for p in self._posts)
        total_comments = sum(p.comments for p in self._posts)
        video_count = sum(1 for p in self._posts if p.is_video)

        # 해시태그 빈도
        hashtag_freq: Dict[str, int] = {}
        for post in self._posts:
            for tag in post.hashtags:
                tag_lower = tag.lower()
                hashtag_freq[tag_lower] = hashtag_freq.get(tag_lower, 0) + 1

        top_hashtags = sorted(hashtag_freq.items(), key=lambda x: x[1], reverse=True)[:10]

        # 인기 포스트
        top_posts = sorted(self._posts, key=lambda x: x.likes, reverse=True)[:3]

        lines = [
            "### Instagram 트렌드 분석",
            f"- 분석 포스트: {len(self._posts)}개 (동영상: {video_count}개)",
            f"- 총 좋아요: {total_likes:,}",
            f"- 총 댓글: {total_comments:,}",
            "",
            "**인기 해시태그:**"
        ]

        for tag, count in top_hashtags[:5]:
            lines.append(f"- #{tag}: {count}회")

        if top_posts:
            lines.append("")
            lines.append("**인기 포스트:**")
            for post in top_posts:
                lines.append(f"- @{post.author}: {post.likes:,} likes")

        return "\n".join(lines)

    @property
    def posts(self) -> List[InstagramPost]:
        return self._posts

    def __del__(self):
        """리소스 정리"""
        if self._executor:
            self._executor.shutdown(wait=False)


# 테스트용
async def main():
    collector = InstagramCollector()

    if collector.initialize():
        posts = await collector.search_hashtag("laneige", limit=10)

        for post in posts:
            print(f"[@{post.author}] {post.likes:,} likes | {post.caption[:50]}...")

        print("\n" + collector.generate_report_section())


if __name__ == "__main__":
    asyncio.run(main())
