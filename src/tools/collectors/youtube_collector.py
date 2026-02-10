"""
YouTube Collector
=================
YouTube에서 K-Beauty/LANEIGE 관련 비디오 메타데이터를 수집하는 모듈

## 사용 기술
- yt-dlp 라이브러리 (무료, 오픈소스)
- API 키 불필요

## 수집 대상
- 검색 쿼리: "LANEIGE review", "K-Beauty routine"
- 채널 비디오 메타데이터

## 사용 예시
```python
collector = YouTubeCollector()

# 검색
videos = await collector.search("LANEIGE lip sleeping mask review", limit=20)

# K-Beauty 종합 검색
all_videos = await collector.search_kbeauty(limit=50)
```

## 주의사항
- Rate limiting 준수
- 메타데이터만 추출 (비디오 다운로드 X)
"""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta, timezone
from typing import Any

try:
    import yt_dlp

    YTDLP_AVAILABLE = True
except ImportError:
    YTDLP_AVAILABLE = False

logger = logging.getLogger(__name__)

# 한국 시간대
KST = timezone(timedelta(hours=9))


@dataclass
class YouTubeVideo:
    """YouTube 비디오 데이터"""

    video_id: str
    title: str
    channel: str
    channel_id: str
    description: str
    views: int
    likes: int
    comments: int
    duration: int  # seconds
    upload_date: datetime | None = None
    thumbnail_url: str | None = None
    tags: list[str] = field(default_factory=list)
    url: str = ""
    collected_at: datetime = field(default_factory=lambda: datetime.now(KST))

    def to_dict(self) -> dict[str, Any]:
        return {
            "video_id": self.video_id,
            "title": self.title,
            "channel": self.channel,
            "channel_id": self.channel_id,
            "description": self.description,
            "views": self.views,
            "likes": self.likes,
            "comments": self.comments,
            "duration": self.duration,
            "upload_date": self.upload_date.isoformat() if self.upload_date else None,
            "thumbnail_url": self.thumbnail_url,
            "tags": self.tags,
            "url": self.url,
            "collected_at": self.collected_at.isoformat(),
        }

    def get_engagement_rate(self) -> float:
        """인게이지먼트율 계산"""
        if self.views == 0:
            return 0.0
        return (self.likes + self.comments) / self.views * 100


class YouTubeCollector:
    """
    YouTube 데이터 수집기

    yt-dlp 기반으로 비디오 메타데이터 수집
    """

    # K-Beauty 관련 검색 쿼리
    KBEAUTY_QUERIES = [
        "LANEIGE review",
        "LANEIGE lip sleeping mask",
        "K-Beauty routine 2026",
        "Korean skincare routine",
        "best Korean skincare",
        "COSRX review",
        "TIRTIR review",
    ]

    def __init__(self, request_delay: float = 2.0, max_results_per_query: int = 20):
        """
        Args:
            request_delay: 요청 간 딜레이 (초)
            max_results_per_query: 쿼리당 최대 결과 수
        """
        self.request_delay = request_delay
        self.max_results_per_query = max_results_per_query
        self._videos: list[YouTubeVideo] = []
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._initialized = False

    def initialize(self) -> bool:
        """초기화"""
        if not YTDLP_AVAILABLE:
            logger.warning("yt-dlp not available. YouTube collection disabled.")
            return False

        self._initialized = True
        logger.info("YouTube collector initialized")
        return True

    def _get_ydl_opts(self) -> dict[str, Any]:
        """yt-dlp 옵션"""
        return {
            "quiet": True,
            "no_warnings": True,
            "extract_flat": True,
            "skip_download": True,
            "ignoreerrors": True,
            "no_color": True,
        }

    def _search_sync(self, query: str, limit: int) -> list[YouTubeVideo]:
        """동기 방식 검색 (스레드에서 실행)"""
        videos = []

        try:
            search_url = f"ytsearch{limit}:{query}"

            with yt_dlp.YoutubeDL(self._get_ydl_opts()) as ydl:
                result = ydl.extract_info(search_url, download=False)

                if not result or "entries" not in result:
                    return videos

                for entry in result["entries"]:
                    if not entry:
                        continue

                    try:
                        video = self._parse_entry(entry)
                        if video:
                            videos.append(video)
                    except Exception as e:
                        logger.debug(f"Error parsing entry: {e}")
                        continue

        except Exception as e:
            logger.error(f"Error searching YouTube for '{query}': {e}")

        return videos

    def _parse_entry(self, entry: dict[str, Any]) -> YouTubeVideo | None:
        """yt-dlp 결과를 YouTubeVideo로 변환"""
        try:
            video_id = entry.get("id", "")
            if not video_id:
                return None

            # 업로드 날짜 파싱
            upload_date = None
            date_str = entry.get("upload_date")
            if date_str and len(date_str) == 8:
                try:
                    upload_date = datetime.strptime(date_str, "%Y%m%d").replace(tzinfo=UTC)
                except ValueError:
                    pass

            return YouTubeVideo(
                video_id=video_id,
                title=entry.get("title", ""),
                channel=entry.get("uploader", entry.get("channel", "")),
                channel_id=entry.get("channel_id", entry.get("uploader_id", "")),
                description=entry.get("description", "")[:500] if entry.get("description") else "",
                views=entry.get("view_count", 0) or 0,
                likes=entry.get("like_count", 0) or 0,
                comments=entry.get("comment_count", 0) or 0,
                duration=entry.get("duration", 0) or 0,
                upload_date=upload_date,
                thumbnail_url=entry.get("thumbnail"),
                tags=entry.get("tags", []) or [],
                url=f"https://www.youtube.com/watch?v={video_id}",
            )

        except Exception as e:
            logger.debug(f"Error parsing video entry: {e}")
            return None

    async def search(self, query: str, limit: int = 20) -> list[YouTubeVideo]:
        """
        YouTube 검색

        Args:
            query: 검색 쿼리
            limit: 최대 결과 수

        Returns:
            YouTubeVideo 리스트
        """
        if not self._initialized:
            if not self.initialize():
                return []

        loop = asyncio.get_event_loop()
        videos = await loop.run_in_executor(
            self._executor, self._search_sync, query, min(limit, self.max_results_per_query)
        )

        logger.info(f"Found {len(videos)} videos for '{query}'")
        return videos

    async def search_kbeauty(self, limit: int = 50) -> list[YouTubeVideo]:
        """
        K-Beauty 관련 비디오 종합 검색

        Args:
            limit: 총 수집 개수

        Returns:
            YouTubeVideo 리스트
        """
        all_videos = []
        per_query = max(limit // len(self.KBEAUTY_QUERIES[:4]), 5)

        for query in self.KBEAUTY_QUERIES[:4]:
            videos = await self.search(query, limit=per_query)
            all_videos.extend(videos)
            await asyncio.sleep(self.request_delay)

        # 중복 제거
        seen_ids = set()
        unique_videos = []
        for video in all_videos:
            if video.video_id not in seen_ids:
                seen_ids.add(video.video_id)
                unique_videos.append(video)

        self._videos = unique_videos[:limit]
        return self._videos

    async def get_video_info(self, video_id: str) -> YouTubeVideo | None:
        """
        특정 비디오 상세 정보

        Args:
            video_id: YouTube 비디오 ID

        Returns:
            YouTubeVideo 또는 None
        """
        if not self._initialized:
            if not self.initialize():
                return None

        def _get_info():
            try:
                url = f"https://www.youtube.com/watch?v={video_id}"
                with yt_dlp.YoutubeDL(self._get_ydl_opts()) as ydl:
                    info = ydl.extract_info(url, download=False)
                    return self._parse_entry(info) if info else None
            except Exception as e:
                logger.error(f"Error getting video info: {e}")
                return None

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, _get_info)

    def generate_report_section(self) -> str:
        """리포트용 섹션 생성"""
        if not self._videos:
            return "YouTube 데이터가 수집되지 않았습니다."

        total_views = sum(v.views for v in self._videos)
        total_likes = sum(v.likes for v in self._videos)
        avg_duration = (
            sum(v.duration for v in self._videos) / len(self._videos) if self._videos else 0
        )

        # 인기 비디오
        top_videos = sorted(self._videos, key=lambda x: x.views, reverse=True)[:5]

        # 채널 빈도
        channel_freq: dict[str, int] = {}
        for video in self._videos:
            channel_freq[video.channel] = channel_freq.get(video.channel, 0) + 1

        top_channels = sorted(channel_freq.items(), key=lambda x: x[1], reverse=True)[:5]

        lines = [
            "### YouTube 트렌드 분석",
            f"- 분석 비디오: {len(self._videos)}개",
            f"- 총 조회수: {total_views:,}",
            f"- 총 좋아요: {total_likes:,}",
            f"- 평균 길이: {int(avg_duration // 60)}분 {int(avg_duration % 60)}초",
            "",
            "**인기 비디오:**",
        ]

        for video in top_videos[:3]:
            lines.append(f"- [{video.title[:40]}...] {video.views:,} views")

        if top_channels:
            lines.append("")
            lines.append("**활발한 채널:**")
            for channel, count in top_channels[:3]:
                lines.append(f"- {channel}: {count}개 비디오")

        return "\n".join(lines)

    @property
    def videos(self) -> list[YouTubeVideo]:
        return self._videos

    def __del__(self):
        """리소스 정리"""
        if self._executor:
            self._executor.shutdown(wait=False)


# 테스트용
async def main():
    collector = YouTubeCollector()

    if collector.initialize():
        videos = await collector.search("LANEIGE review", limit=5)

        for video in videos:
            print(f"[{video.video_id}] {video.views:,} views | {video.title[:50]}...")

        print("\n" + collector.generate_report_section())


if __name__ == "__main__":
    asyncio.run(main())
