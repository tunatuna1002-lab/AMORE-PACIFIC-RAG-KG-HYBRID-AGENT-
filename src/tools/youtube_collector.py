"""
YouTube Collector
==================
Apify 기반 YouTube 리뷰/튜토리얼 수집기

## 기능
- LANEIGE 및 K-Beauty 관련 YouTube 영상 수집
- 리뷰, 튜토리얼, 비교 영상 분석
- 조회수, 좋아요, 댓글수 추적

## 사용 예
```python
collector = YouTubeCollector()
videos = await collector.fetch_laneige_reviews()
signals = collector.to_external_signals(videos)
```

## 비용
- Apify 무료 크레딧으로 10,000 결과 가능
- 월 200-500개 영상 수집 시 무료
"""

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

# Apify 클라이언트
try:
    from apify_client import ApifyClient
    APIFY_AVAILABLE = True
except ImportError:
    APIFY_AVAILABLE = False
    logger.warning("apify-client not installed. Install with: pip install apify-client")

# 한국 시간대 (UTC+9)
KST = timezone(timedelta(hours=9))


@dataclass
class YouTubeVideo:
    """YouTube 영상 데이터"""
    video_id: str
    title: str
    channel_name: str
    channel_id: str
    view_count: int
    like_count: int
    comment_count: int
    published_at: str
    description: str = ""
    thumbnail_url: str = ""
    duration: str = ""
    url: str = ""
    collected_at: str = ""
    keywords_matched: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class YouTubeCollector:
    """
    YouTube 영상 수집기 (Apify 기반)

    Apify의 YouTube Scraper Actor를 사용하여 영상 데이터를 수집합니다.
    무료 크레딧 ($5/월)으로 약 10,000개 결과 수집 가능합니다.
    """

    # Apify Actor ID
    ACTOR_ID = "streamers/youtube-scraper"

    # 기본 검색 키워드
    DEFAULT_SEARCH_KEYWORDS = [
        "LANEIGE review",
        "LANEIGE lip sleeping mask",
        "Korean skincare routine",
        "K-beauty haul",
        "COSRX review"
    ]

    # LANEIGE 특화 키워드
    LANEIGE_KEYWORDS = [
        "LANEIGE review",
        "LANEIGE lip sleeping mask review",
        "LANEIGE water bank",
        "LANEIGE cream skin",
        "라네즈 리뷰"
    ]

    def __init__(self):
        """초기화"""
        self._client: Optional[ApifyClient] = None
        self._enabled = os.getenv("ENABLE_YOUTUBE_COLLECTOR", "true").lower() == "true"
        self._api_token = os.getenv("APIFY_API_TOKEN")

        # 데이터 저장 경로
        self.data_dir = Path("data/market_intelligence/youtube")
        self.data_dir.mkdir(parents=True, exist_ok=True)

    def _get_client(self) -> Optional[ApifyClient]:
        """Apify 클라이언트 반환 (lazy initialization)"""
        if not APIFY_AVAILABLE:
            logger.error("apify-client not available")
            return None

        if not self._api_token:
            logger.error("APIFY_API_TOKEN not set")
            return None

        if self._client is None:
            self._client = ApifyClient(self._api_token)

        return self._client

    async def fetch_reviews(
        self,
        keywords: List[str],
        max_results: int = 50,
        sort_by: str = "date"
    ) -> List[YouTubeVideo]:
        """
        YouTube 리뷰 영상 수집

        Args:
            keywords: 검색 키워드 리스트
            max_results: 키워드당 최대 결과 수
            sort_by: 정렬 방식 (date, relevance, viewCount)

        Returns:
            YouTubeVideo 리스트
        """
        if not self._enabled:
            logger.info("YouTube collector disabled")
            return []

        client = self._get_client()
        if client is None:
            return []

        videos = []
        now = datetime.now(KST).isoformat()

        try:
            # Apify Actor 실행
            run_input = {
                "searchKeywords": keywords,
                "maxResults": max_results,
                "sortBy": sort_by,
                "maxResultsShorts": 0,  # Shorts 제외
                "channelType": "any",
                "resultsType": "videos"
            }

            logger.info(f"Starting YouTube scrape for keywords: {keywords}")

            # 동기 API를 비동기로 래핑
            def _run_actor():
                run = client.actor(self.ACTOR_ID).call(run_input=run_input)
                return list(client.dataset(run["defaultDatasetId"]).iterate_items())

            items = await asyncio.get_event_loop().run_in_executor(None, _run_actor)

            for item in items:
                video = YouTubeVideo(
                    video_id=item.get("id", ""),
                    title=item.get("title", ""),
                    channel_name=item.get("channelName", ""),
                    channel_id=item.get("channelId", ""),
                    view_count=item.get("viewCount", 0) or 0,
                    like_count=item.get("likes", 0) or 0,
                    comment_count=item.get("commentsCount", 0) or 0,
                    published_at=item.get("date", ""),
                    description=item.get("text", "")[:500],  # 설명 500자 제한
                    thumbnail_url=item.get("thumbnailUrl", ""),
                    duration=item.get("duration", ""),
                    url=item.get("url", f"https://www.youtube.com/watch?v={item.get('id', '')}"),
                    collected_at=now,
                    keywords_matched=self._match_keywords(item.get("title", ""), keywords)
                )
                videos.append(video)

            logger.info(f"Fetched {len(videos)} YouTube videos")

        except Exception as e:
            logger.error(f"Error fetching YouTube videos: {e}")

        return videos

    def _match_keywords(self, title: str, keywords: List[str]) -> List[str]:
        """제목에서 매칭된 키워드 추출"""
        title_lower = title.lower()
        matched = []
        for kw in keywords:
            if kw.lower() in title_lower:
                matched.append(kw)
        return matched

    async def fetch_laneige_reviews(self, max_results: int = 50) -> List[YouTubeVideo]:
        """LANEIGE 관련 리뷰 수집"""
        return await self.fetch_reviews(self.LANEIGE_KEYWORDS, max_results)

    async def fetch_beauty_trends(self, max_results: int = 30) -> List[YouTubeVideo]:
        """일반 뷰티 트렌드 영상 수집"""
        return await self.fetch_reviews(self.DEFAULT_SEARCH_KEYWORDS, max_results)

    def to_external_signals(self, videos: List[YouTubeVideo]) -> List[Dict[str, Any]]:
        """
        ExternalSignal 형식으로 변환 (Tier 2)

        Args:
            videos: YouTubeVideo 리스트

        Returns:
            ExternalSignal 호환 딕셔너리 리스트
        """
        signals = []

        for i, video in enumerate(videos):
            signal = {
                "signal_id": f"SID-youtube-{video.published_at[:10] if video.published_at else 'unknown'}-{i:03d}",
                "source": "youtube",
                "tier": "tier2_validation",
                "title": video.title,
                "content": video.description[:200] if video.description else "",
                "url": video.url,
                "published_at": video.published_at,
                "collected_at": video.collected_at,
                "keywords": video.keywords_matched,
                "relevance_score": self._calculate_relevance(video),
                "metadata": {
                    "channel_name": video.channel_name,
                    "view_count": video.view_count,
                    "like_count": video.like_count,
                    "comment_count": video.comment_count,
                    "duration": video.duration
                }
            }
            signals.append(signal)

        return signals

    def _calculate_relevance(self, video: YouTubeVideo) -> float:
        """관련성 점수 계산 (0.0~1.0)"""
        score = 0.5  # 기본 점수

        # 키워드 매칭 보너스
        score += len(video.keywords_matched) * 0.1

        # 조회수 보너스 (10만 이상)
        if video.view_count >= 100000:
            score += 0.1
        elif video.view_count >= 10000:
            score += 0.05

        # 참여율 보너스 (좋아요/조회수)
        if video.view_count > 0:
            engagement = (video.like_count + video.comment_count) / video.view_count
            if engagement > 0.05:  # 5% 이상
                score += 0.1

        return min(score, 1.0)

    def generate_insight_section(self, videos: List[YouTubeVideo]) -> str:
        """
        인사이트 보고서용 섹션 생성

        Returns:
            마크다운 형식의 YouTube 인사이트
        """
        if not videos:
            return ""

        lines = ["### YouTube 리뷰 트렌드\n"]

        # 상위 5개 영상
        top_videos = sorted(videos, key=lambda v: v.view_count, reverse=True)[:5]

        for video in top_videos:
            views_str = f"{video.view_count:,}" if video.view_count else "N/A"
            lines.append(f"- **{video.channel_name}**: \"{video.title[:50]}...\" ({views_str} views)")

        # 통계
        total_views = sum(v.view_count for v in videos)
        avg_engagement = sum(v.like_count + v.comment_count for v in videos) / len(videos) if videos else 0

        lines.append(f"\n**총 조회수**: {total_views:,} | **평균 참여**: {avg_engagement:.0f}")

        return "\n".join(lines)

    async def save_videos(self, videos: List[YouTubeVideo], filename: Optional[str] = None) -> Path:
        """영상 데이터 저장"""
        if filename is None:
            date_str = datetime.now(KST).strftime("%Y-%m-%d")
            filename = f"youtube_{date_str}.json"

        filepath = self.data_dir / filename

        data = {
            "collected_at": datetime.now(KST).isoformat(),
            "count": len(videos),
            "videos": [v.to_dict() for v in videos]
        }

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info(f"Saved {len(videos)} videos to {filepath}")
        return filepath


# 테스트용 메인
if __name__ == "__main__":
    async def main():
        collector = YouTubeCollector()

        print("Fetching LANEIGE reviews...")
        videos = await collector.fetch_laneige_reviews(max_results=10)

        for video in videos[:5]:
            print(f"\n{video.title}")
            print(f"  Channel: {video.channel_name}")
            print(f"  Views: {video.view_count:,}")
            print(f"  URL: {video.url}")

        # 인사이트 생성
        print("\n" + collector.generate_insight_section(videos))

        # ExternalSignal 변환
        signals = collector.to_external_signals(videos)
        print(f"\nConverted to {len(signals)} external signals")

        # 저장
        await collector.save_videos(videos)

    asyncio.run(main())
