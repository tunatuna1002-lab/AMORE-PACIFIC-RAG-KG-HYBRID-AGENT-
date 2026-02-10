"""
Signal Collector Protocol
==========================
ExternalSignalCollector에 대한 추상 인터페이스

구현체:
- ExternalSignalCollector (src/tools/external_signal_collector.py)
"""

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class SignalCollectorProtocol(Protocol):
    """
    External Signal Collector Protocol

    외부 신호 수집 인터페이스.
    뉴스, 소셜 미디어, 트렌드 등 외부 데이터를 수집하고 통합합니다.

    Methods:
        initialize: 수집기 초기화
        close: 수집기 종료
        fetch_rss_articles: RSS 피드 기사 수집
        fetch_all_rss_feeds: 모든 RSS 피드 수집
        fetch_reddit_trends: Reddit 트렌드 수집
        fetch_tiktok_trends: TikTok 트렌드 수집
        fetch_kbeauty_news: K-Beauty 뉴스 수집
        fetch_industry_signals: 산업 신호 수집
        fetch_tavily_news: Tavily API 뉴스 수집
        fetch_all_news: 모든 뉴스 수집
        add_manual_media_input: 수동 미디어 입력 추가
        add_weekly_trend_radar: 주간 트렌드 레이더 추가
        generate_report_section: 리포트 섹션 생성
        get_source_reliability: 소스 신뢰도 조회
        create_source_reference: 소스 참조 생성
        get_signals_for_kg: Knowledge Graph용 신호 조회
        get_stats: 통계 조회
    """

    async def initialize(self) -> None:
        """
        수집기를 초기화합니다.

        HTTP 클라이언트, 캐시 등을 설정합니다.
        """
        ...

    async def close(self) -> None:
        """
        수집기를 종료합니다.

        리소스를 정리하고 연결을 닫습니다.
        """
        ...

    async def fetch_rss_articles(
        self,
        feed_url: str,
        keywords: list[str],
        limit: int = 10,
    ) -> list[Any]:
        """
        RSS 피드에서 기사를 수집합니다.

        Args:
            feed_url: RSS 피드 URL
            keywords: 필터링 키워드
            limit: 최대 수집 수

        Returns:
            ExternalSignal 객체 리스트
        """
        ...

    async def fetch_all_rss_feeds(
        self,
        keywords: list[str] | None = None,
    ) -> list[Any]:
        """
        모든 RSS 피드에서 기사를 수집합니다.

        Args:
            keywords: 필터링 키워드 (선택)

        Returns:
            ExternalSignal 객체 리스트
        """
        ...

    async def fetch_reddit_trends(
        self,
        subreddit: str = "AsianBeauty",
        keywords: list[str] | None = None,
        limit: int = 50,
    ) -> list[Any]:
        """
        Reddit에서 트렌드를 수집합니다.

        Args:
            subreddit: 서브레딧 이름
            keywords: 필터링 키워드 (선택)
            limit: 최대 수집 수

        Returns:
            ExternalSignal 객체 리스트
        """
        ...

    async def fetch_tiktok_trends(
        self,
        hashtag: str = "kbeauty",
        limit: int = 20,
    ) -> list[Any]:
        """
        TikTok에서 트렌드를 수집합니다.

        Args:
            hashtag: 해시태그
            limit: 최대 수집 수

        Returns:
            ExternalSignal 객체 리스트
        """
        ...

    async def fetch_kbeauty_news(
        self,
        keywords: list[str] | None = None,
        days: int = 7,
    ) -> list[Any]:
        """
        K-Beauty 관련 뉴스를 수집합니다.

        Args:
            keywords: 필터링 키워드 (선택)
            days: 조회 기간 (일)

        Returns:
            ExternalSignal 객체 리스트
        """
        ...

    async def fetch_industry_signals(
        self,
        keywords: list[str] | None = None,
    ) -> list[Any]:
        """
        산업 신호를 수집합니다.

        Args:
            keywords: 필터링 키워드 (선택)

        Returns:
            ExternalSignal 객체 리스트
        """
        ...

    async def fetch_tavily_news(
        self,
        query: str,
        max_results: int = 5,
        days: int = 7,
    ) -> list[Any]:
        """
        Tavily API를 통해 뉴스를 수집합니다.

        Args:
            query: 검색 쿼리
            max_results: 최대 결과 수
            days: 조회 기간 (일)

        Returns:
            ExternalSignal 객체 리스트
        """
        ...

    async def fetch_all_news(
        self,
        keywords: list[str] | None = None,
        days: int = 7,
    ) -> dict[str, Any]:
        """
        모든 뉴스 소스에서 뉴스를 수집합니다.

        Args:
            keywords: 필터링 키워드 (선택)
            days: 조회 기간 (일)

        Returns:
            수집 결과 딕셔너리
            {
                "signals": [...],
                "count": int,
                "sources": [...],
                "failed_sources": [...],
            }
        """
        ...

    def add_manual_media_input(self, media_data: dict[str, Any]) -> Any:
        """
        수동 미디어 입력을 추가합니다.

        Args:
            media_data: 미디어 데이터
                {
                    "type": "news" | "social" | "trend",
                    "source": str,
                    "content": str,
                    "url": str,
                    "keywords": [...],
                    ...
                }

        Returns:
            ExternalSignal 객체
        """
        ...

    def add_weekly_trend_radar(self, radar_data: list[dict[str, Any]]) -> list[Any]:
        """
        주간 트렌드 레이더 데이터를 추가합니다.

        Args:
            radar_data: 트렌드 레이더 데이터 리스트

        Returns:
            ExternalSignal 객체 리스트
        """
        ...

    def generate_report_section(
        self,
        signals: list[Any],
        title: str = "External Signals",
    ) -> str:
        """
        리포트 섹션을 생성합니다.

        Args:
            signals: ExternalSignal 객체 리스트
            title: 섹션 제목

        Returns:
            마크다운 형식 리포트 문자열
        """
        ...

    def get_source_reliability(self, source: str) -> float:
        """
        소스의 신뢰도를 반환합니다.

        Args:
            source: 소스 이름

        Returns:
            신뢰도 (0~1)
        """
        ...

    def create_source_reference(self, signal: Any) -> dict[str, Any]:
        """
        소스 참조 정보를 생성합니다.

        Args:
            signal: ExternalSignal 객체

        Returns:
            참조 정보 딕셔너리
            {
                "type": str,
                "url": str,
                "title": str,
                "published_date": str,
                "reliability": float,
            }
        """
        ...

    def get_signals_for_kg(self) -> list[dict[str, Any]]:
        """
        Knowledge Graph에 추가할 신호를 반환합니다.

        Returns:
            신호 딕셔너리 리스트
        """
        ...

    def get_stats(self) -> dict[str, Any]:
        """
        통계를 반환합니다.

        Returns:
            통계 딕셔너리
            {
                "total_signals": int,
                "by_type": {...},
                "by_source": {...},
                "date_range": {...},
            }
        """
        ...
