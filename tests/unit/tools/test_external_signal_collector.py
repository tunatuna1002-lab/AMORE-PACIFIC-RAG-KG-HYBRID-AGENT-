"""
ExternalSignalCollector 단위 테스트

테스트 대상: src/tools/collectors/external_signal_collector.py
Coverage target: 60%+
"""

from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.tools.collectors.external_signal_collector import (
    BEAUTY_KEYWORDS,
    KBEAUTY_KEYWORDS,
    RSS_FEEDS,
    ExternalSignal,
    ExternalSignalCollector,
    SignalSource,
    SignalTier,
)


class TestSignalTier:
    """SignalTier 열거형 테스트"""

    def test_tier_values(self):
        """모든 Tier 값이 올바르게 정의"""
        assert SignalTier.TIER1_VIRAL.value == "tier1_viral"
        assert SignalTier.TIER2_VALIDATION.value == "tier2_validation"
        assert SignalTier.TIER3_AUTHORITY.value == "tier3_authority"
        assert SignalTier.TIER4_PR.value == "tier4_pr"


class TestSignalSource:
    """SignalSource 열거형 테스트"""

    def test_tier1_sources(self):
        """Tier 1 소스"""
        assert SignalSource.TIKTOK.value == "tiktok"
        assert SignalSource.INSTAGRAM.value == "instagram"

    def test_tier2_sources(self):
        """Tier 2 소스"""
        assert SignalSource.YOUTUBE.value == "youtube"
        assert SignalSource.REDDIT.value == "reddit"

    def test_tier3_sources(self):
        """Tier 3 소스"""
        assert SignalSource.ALLURE.value == "allure"
        assert SignalSource.BYRDIE.value == "byrdie"

    def test_tier4_sources(self):
        """Tier 4 소스"""
        assert SignalSource.TWITTER.value == "twitter"

    def test_manual_source(self):
        """수동 입력 소스"""
        assert SignalSource.MANUAL.value == "manual"


class TestExternalSignal:
    """ExternalSignal 데이터클래스 테스트"""

    def _make_signal(self, **kwargs):
        defaults = {
            "signal_id": "SID-TST-20260115-0001",
            "source": "allure",
            "tier": SignalTier.TIER3_AUTHORITY.value,
            "title": "Test Article",
            "content": "Test content about K-Beauty",
            "url": "https://example.com/article",
            "published_at": "2026-01-15",
            "collected_at": "2026-01-15T10:00:00+09:00",
            "keywords": ["k-beauty", "laneige"],
            "relevance_score": 0.8,
            "metadata": {"author": "Test Author"},
        }
        defaults.update(kwargs)
        return ExternalSignal(**defaults)

    def test_to_dict(self):
        """딕셔너리 변환"""
        signal = self._make_signal()
        d = signal.to_dict()

        assert d["signal_id"] == "SID-TST-20260115-0001"
        assert d["source"] == "allure"
        assert d["tier"] == "tier3_authority"
        assert d["title"] == "Test Article"
        assert isinstance(d["keywords"], list)

    def test_to_report_format_basic(self):
        """보고서 포맷 기본"""
        signal = self._make_signal()
        report = signal.to_report_format()

        assert "Allure" in report
        assert "2026-01-15" in report
        assert "Test Article" in report

    def test_to_report_format_with_views(self):
        """보고서 포맷 - 조회수 포함"""
        signal = self._make_signal(metadata={"views": 50000})
        report = signal.to_report_format()

        assert "50,000" in report

    def test_to_report_format_with_upvotes(self):
        """보고서 포맷 - 업보트 포함"""
        signal = self._make_signal(metadata={"upvotes": 2400})
        report = signal.to_report_format()

        assert "2,400" in report

    def test_to_report_format_no_date(self):
        """보고서 포맷 - 날짜 없음"""
        signal = self._make_signal(published_at="")
        report = signal.to_report_format()

        assert "날짜 미상" in report

    def test_default_values(self):
        """기본값 테스트"""
        signal = ExternalSignal(
            signal_id="SID-1",
            source="test",
            tier="tier3_authority",
            title="Test",
            content="Content",
            url="https://example.com",
            published_at="2026-01-15",
            collected_at="2026-01-15T10:00:00",
        )
        assert signal.keywords == []
        assert signal.relevance_score == 0.5
        assert signal.metadata == {}


class TestExternalSignalCollectorInit:
    """ExternalSignalCollector 초기화 테스트"""

    def test_init_default(self, tmp_path):
        """기본 초기화"""
        collector = ExternalSignalCollector(data_dir=str(tmp_path / "signals"))

        assert collector.data_dir == tmp_path / "signals"
        assert collector.data_dir.exists()
        assert collector.signals == []
        assert collector._signal_seq == 0
        assert collector._session is None

    def test_init_creates_directory(self, tmp_path):
        """데이터 디렉토리 자동 생성"""
        data_dir = tmp_path / "nested" / "signals"
        collector = ExternalSignalCollector(data_dir=str(data_dir))

        assert data_dir.exists()


class TestExternalSignalCollectorLifecycle:
    """초기화/종료 테스트"""

    @pytest.mark.asyncio
    async def test_initialize(self, tmp_path):
        """비동기 초기화"""
        collector = ExternalSignalCollector(data_dir=str(tmp_path / "signals"))

        with patch("src.tools.collectors.external_signal_collector.AIOHTTP_AVAILABLE", True):
            with patch("aiohttp.ClientSession") as mock_session_cls:
                mock_session = MagicMock()
                mock_session_cls.return_value = mock_session

                # Tavily 임포트 실패하도록 mock
                with patch.dict("sys.modules", {"src.tools.collectors.tavily_search": None}):
                    with patch(
                        "src.tools.collectors.external_signal_collector.ExternalSignalCollector._load_signals"
                    ):
                        await collector.initialize()

        assert collector._session is not None

    @pytest.mark.asyncio
    async def test_close(self, tmp_path):
        """세션 종료"""
        collector = ExternalSignalCollector(data_dir=str(tmp_path / "signals"))
        mock_session = AsyncMock()
        collector._session = mock_session

        await collector.close()

        mock_session.close.assert_called_once()
        assert collector._session is None

    @pytest.mark.asyncio
    async def test_close_no_session(self, tmp_path):
        """세션 없을 때 종료"""
        collector = ExternalSignalCollector(data_dir=str(tmp_path / "signals"))

        await collector.close()  # 에러 없이 완료


class TestExternalSignalCollectorRSS:
    """RSS 피드 수집 테스트"""

    @pytest.mark.asyncio
    async def test_fetch_rss_no_feedparser(self, tmp_path):
        """feedparser 미설치 시 빈 리스트"""
        collector = ExternalSignalCollector(data_dir=str(tmp_path / "signals"))

        with patch(
            "src.tools.collectors.external_signal_collector.FEEDPARSER_AVAILABLE",
            False,
        ):
            result = await collector.fetch_rss_articles(SignalSource.ALLURE)

        assert result == []

    @pytest.mark.asyncio
    async def test_fetch_rss_no_feed_url(self, tmp_path):
        """RSS 피드 URL이 없는 소스"""
        collector = ExternalSignalCollector(data_dir=str(tmp_path / "signals"))

        with patch(
            "src.tools.collectors.external_signal_collector.FEEDPARSER_AVAILABLE",
            True,
        ):
            with patch("src.tools.collectors.external_signal_collector.RSS_FEEDS", {}):
                result = await collector.fetch_rss_articles(SignalSource.ALLURE)

        assert result == []

    @pytest.mark.asyncio
    async def test_fetch_rss_success(self, tmp_path):
        """RSS 피드 수집 성공"""
        collector = ExternalSignalCollector(data_dir=str(tmp_path / "signals"))

        mock_entry = MagicMock()
        mock_entry.get = lambda key, default="": {
            "title": "K-Beauty LANEIGE trends",
            "summary": "LANEIGE lip sleeping mask is trending",
            "link": "https://example.com/article",
            "published_parsed": (2026, 1, 15, 10, 0, 0, 0, 0, 0),
            "author": "Test Author",
            "tags": [],
        }.get(key, default)

        mock_feed = MagicMock()
        mock_feed.bozo = False
        mock_feed.entries = [mock_entry]

        with patch(
            "src.tools.collectors.external_signal_collector.FEEDPARSER_AVAILABLE",
            True,
        ):
            with patch("feedparser.parse", return_value=mock_feed):
                result = await collector.fetch_rss_articles(
                    SignalSource.ALLURE, keywords=["laneige"]
                )

        assert len(result) == 1
        assert result[0].source == "allure"
        assert result[0].tier == SignalTier.TIER3_AUTHORITY.value

    @pytest.mark.asyncio
    async def test_fetch_rss_no_match(self, tmp_path):
        """키워드 매칭 실패"""
        collector = ExternalSignalCollector(data_dir=str(tmp_path / "signals"))

        mock_entry = MagicMock()
        mock_entry.get = lambda key, default="": {
            "title": "Completely unrelated article",
            "summary": "Nothing about beauty",
            "link": "https://example.com/unrelated",
        }.get(key, default)

        mock_feed = MagicMock()
        mock_feed.bozo = False
        mock_feed.entries = [mock_entry]

        with patch(
            "src.tools.collectors.external_signal_collector.FEEDPARSER_AVAILABLE",
            True,
        ):
            with patch("feedparser.parse", return_value=mock_feed):
                result = await collector.fetch_rss_articles(
                    SignalSource.ALLURE, keywords=["laneige"]
                )

        assert len(result) == 0

    @pytest.mark.asyncio
    async def test_fetch_rss_parse_error(self, tmp_path):
        """RSS 파싱 에러"""
        collector = ExternalSignalCollector(data_dir=str(tmp_path / "signals"))

        with patch(
            "src.tools.collectors.external_signal_collector.FEEDPARSER_AVAILABLE",
            True,
        ):
            with patch("feedparser.parse", side_effect=Exception("Parse error")):
                result = await collector.fetch_rss_articles(SignalSource.ALLURE)

        assert result == []

    @pytest.mark.asyncio
    async def test_fetch_rss_bozo_warning(self, tmp_path):
        """RSS bozo (파싱 경고) 처리"""
        collector = ExternalSignalCollector(data_dir=str(tmp_path / "signals"))

        mock_feed = MagicMock()
        mock_feed.bozo = True
        mock_feed.bozo_exception = "Some warning"
        mock_feed.entries = []

        with patch(
            "src.tools.collectors.external_signal_collector.FEEDPARSER_AVAILABLE",
            True,
        ):
            with patch("feedparser.parse", return_value=mock_feed):
                result = await collector.fetch_rss_articles(SignalSource.ALLURE)

        assert result == []


class TestExternalSignalCollectorReddit:
    """Reddit 수집 테스트"""

    @pytest.mark.asyncio
    async def test_fetch_reddit_success(self, tmp_path):
        """Reddit 수집 성공"""
        collector = ExternalSignalCollector(data_dir=str(tmp_path / "signals"))

        mock_post = {
            "data": {
                "title": "LANEIGE Lip Sleeping Mask is amazing",
                "selftext": "Best lip care product ever",
                "created_utc": 1736899200,
                "ups": 500,
                "num_comments": 50,
                "author": "testuser",
                "permalink": "/r/SkincareAddiction/comments/abc123",
            }
        }

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"data": {"children": [mock_post]}})

        mock_session = AsyncMock()
        mock_session.get = MagicMock(
            return_value=AsyncMock(
                __aenter__=AsyncMock(return_value=mock_response),
                __aexit__=AsyncMock(return_value=None),
            )
        )
        collector._session = mock_session

        with patch(
            "src.tools.collectors.external_signal_collector.ExternalSignalCollector._save_signals"
        ):
            result = await collector.fetch_reddit_trends(
                subreddits=["SkincareAddiction"],
                keywords=["laneige"],
                max_posts=5,
            )

        assert len(result) == 1
        assert result[0].source == "reddit"
        assert result[0].tier == SignalTier.TIER2_VALIDATION.value
        assert result[0].metadata["upvotes"] == 500

    @pytest.mark.asyncio
    async def test_fetch_reddit_api_error(self, tmp_path):
        """Reddit API 에러"""
        collector = ExternalSignalCollector(data_dir=str(tmp_path / "signals"))

        mock_response = AsyncMock()
        mock_response.status = 429  # Rate limited

        mock_session = AsyncMock()
        mock_session.get = MagicMock(
            return_value=AsyncMock(
                __aenter__=AsyncMock(return_value=mock_response),
                __aexit__=AsyncMock(return_value=None),
            )
        )
        collector._session = mock_session

        with patch(
            "src.tools.collectors.external_signal_collector.ExternalSignalCollector._save_signals"
        ):
            result = await collector.fetch_reddit_trends(
                subreddits=["SkincareAddiction"],
                keywords=["laneige"],
            )

        assert result == []

    @pytest.mark.asyncio
    async def test_fetch_reddit_no_session_initializes(self, tmp_path):
        """세션 없으면 자동 초기화"""
        collector = ExternalSignalCollector(data_dir=str(tmp_path / "signals"))
        collector._session = None

        with patch.object(collector, "initialize", new_callable=AsyncMock) as mock_init:
            # initialize 후에도 _session이 None이면 에러 나므로 mock 설정
            async def set_session():
                mock_session = AsyncMock()
                mock_response = AsyncMock()
                mock_response.status = 200
                mock_response.json = AsyncMock(return_value={"data": {"children": []}})
                mock_session.get = MagicMock(
                    return_value=AsyncMock(
                        __aenter__=AsyncMock(return_value=mock_response),
                        __aexit__=AsyncMock(return_value=None),
                    )
                )
                collector._session = mock_session

            mock_init.side_effect = set_session

            with patch(
                "src.tools.collectors.external_signal_collector.ExternalSignalCollector._save_signals"
            ):
                result = await collector.fetch_reddit_trends(
                    subreddits=["SkincareAddiction"],
                    keywords=["laneige"],
                )

            mock_init.assert_called_once()


class TestExternalSignalCollectorTikTok:
    """TikTok 수집 테스트"""

    @pytest.mark.asyncio
    async def test_fetch_tiktok_returns_empty(self, tmp_path):
        """TikTok은 항상 빈 리스트 반환"""
        collector = ExternalSignalCollector(data_dir=str(tmp_path / "signals"))

        result = await collector.fetch_tiktok_trends()
        assert result == []


class TestExternalSignalCollectorManualInput:
    """수동 입력 테스트"""

    def test_add_manual_media_input(self, tmp_path):
        """수동 매체 입력"""
        collector = ExternalSignalCollector(data_dir=str(tmp_path / "signals"))

        with patch.object(collector, "_save_signals"):
            signal = collector.add_manual_media_input(
                {
                    "source": "allure",
                    "date": "2026-01-10",
                    "title": "2026 Skincare Trends",
                    "url": "https://allure.com/test",
                    "quotes": ["Peptide is trending", "Lip care is hot"],
                    "keywords": ["peptide", "lip care"],
                    "views": 50000,
                }
            )

        assert signal.source == "allure"
        assert signal.tier == SignalTier.TIER3_AUTHORITY.value
        assert signal.title == "2026 Skincare Trends"
        assert signal.relevance_score == 0.9
        assert signal.metadata["views"] == 50000
        assert signal.metadata["manual_input"] is True

    def test_add_manual_tiktok_input(self, tmp_path):
        """수동 TikTok 입력 - Tier 1"""
        collector = ExternalSignalCollector(data_dir=str(tmp_path / "signals"))

        with patch.object(collector, "_save_signals"):
            signal = collector.add_manual_media_input({"source": "tiktok", "title": "Viral"})

        assert signal.tier == SignalTier.TIER1_VIRAL.value

    def test_add_manual_youtube_input(self, tmp_path):
        """수동 YouTube 입력 - Tier 2"""
        collector = ExternalSignalCollector(data_dir=str(tmp_path / "signals"))

        with patch.object(collector, "_save_signals"):
            signal = collector.add_manual_media_input({"source": "youtube", "title": "Review"})

        assert signal.tier == SignalTier.TIER2_VALIDATION.value

    def test_add_manual_twitter_input(self, tmp_path):
        """수동 Twitter 입력 - Tier 4"""
        collector = ExternalSignalCollector(data_dir=str(tmp_path / "signals"))

        with patch.object(collector, "_save_signals"):
            signal = collector.add_manual_media_input({"source": "twitter", "title": "PR Post"})

        assert signal.tier == SignalTier.TIER4_PR.value

    def test_add_weekly_trend_radar(self, tmp_path):
        """주간 트렌드 레이더 일괄 입력"""
        collector = ExternalSignalCollector(data_dir=str(tmp_path / "signals"))

        radar_data = [
            {
                "rank": 1,
                "keyword": "lip care",
                "source": "Dash Social / TikTok",
                "date": "2025-12-21",
                "consumer_need": "겨울 건조로 인한 입술 보습 관심 급증",
                "laneige_connection": "Lip Sleeping Mask 강점 강조",
                "evidence": "TikTok #lipcare(2K)",
            },
            {
                "rank": 2,
                "keyword": "glass skin",
                "source": "Reddit",
                "date": "2025-12-21",
                "consumer_need": "유리 피부 트렌드",
                "laneige_connection": "Water Bank 연결",
                "evidence": "Reddit 500 upvotes",
            },
        ]

        with patch.object(collector, "_save_signals"):
            signals = collector.add_weekly_trend_radar(radar_data)

        assert len(signals) == 2
        assert signals[0].title == "lip care"
        assert signals[0].relevance_score == 1.0
        assert signals[0].metadata["rank"] == 1
        assert "소비자 니즈" in signals[0].content


class TestExternalSignalCollectorReport:
    """보고서 생성 테스트"""

    def test_generate_report_section_empty(self, tmp_path):
        """신호가 없는 경우"""
        collector = ExternalSignalCollector(data_dir=str(tmp_path / "signals"))

        report = collector.generate_report_section()
        assert "수집된 외부 신호가 없습니다" in report

    def test_generate_report_section_with_signals(self, tmp_path):
        """신호가 있는 경우"""
        collector = ExternalSignalCollector(data_dir=str(tmp_path / "signals"))

        today = datetime.now().strftime("%Y-%m-%d")
        collector.signals = [
            ExternalSignal(
                signal_id="SID-1",
                source="allure",
                tier=SignalTier.TIER3_AUTHORITY.value,
                title="K-Beauty trends",
                content="Content",
                url="https://example.com",
                published_at=today,
                collected_at=today,
            ),
            ExternalSignal(
                signal_id="SID-2",
                source="reddit",
                tier=SignalTier.TIER2_VALIDATION.value,
                title="LANEIGE review",
                content="Content",
                url="https://reddit.com/r/test",
                published_at=today,
                collected_at=today,
                metadata={"upvotes": 500},
            ),
        ]

        report = collector.generate_report_section()

        assert "전문 매체 근거" in report
        assert "소비자 검증" in report

    def test_generate_report_section_with_tier_filter(self, tmp_path):
        """Tier 필터"""
        collector = ExternalSignalCollector(data_dir=str(tmp_path / "signals"))

        today = datetime.now().strftime("%Y-%m-%d")
        collector.signals = [
            ExternalSignal(
                signal_id="SID-1",
                source="allure",
                tier=SignalTier.TIER3_AUTHORITY.value,
                title="K-Beauty trends",
                content="Content",
                url="https://example.com",
                published_at=today,
                collected_at=today,
            ),
            ExternalSignal(
                signal_id="SID-2",
                source="reddit",
                tier=SignalTier.TIER2_VALIDATION.value,
                title="LANEIGE review",
                content="Content",
                url="https://reddit.com/r/test",
                published_at=today,
                collected_at=today,
            ),
        ]

        report = collector.generate_report_section(tier_filter=[SignalTier.TIER3_AUTHORITY])

        assert "전문 매체 근거" in report
        assert "소비자 검증" not in report

    def test_generate_report_section_old_signals_excluded(self, tmp_path):
        """오래된 신호 제외"""
        collector = ExternalSignalCollector(data_dir=str(tmp_path / "signals"))

        old_date = (datetime.now() - timedelta(days=14)).strftime("%Y-%m-%d")
        collector.signals = [
            ExternalSignal(
                signal_id="SID-1",
                source="allure",
                tier=SignalTier.TIER3_AUTHORITY.value,
                title="Old article",
                content="Content",
                url="https://example.com",
                published_at=old_date,
                collected_at=old_date,
            ),
        ]

        report = collector.generate_report_section(days=7)
        assert "수집된 외부 신호가 없습니다" in report


class TestExternalSignalCollectorKBeauty:
    """K-Beauty 전용 메서드 테스트"""

    @pytest.mark.asyncio
    async def test_fetch_kbeauty_news(self, tmp_path):
        """K-Beauty 뉴스 수집"""
        collector = ExternalSignalCollector(data_dir=str(tmp_path / "signals"))
        collector.fetch_rss_articles = AsyncMock(return_value=[])

        result = await collector.fetch_kbeauty_news(max_articles=20)

        assert isinstance(result, list)
        # fetch_rss_articles가 여러 소스에 대해 호출됨
        assert collector.fetch_rss_articles.call_count > 0

    @pytest.mark.asyncio
    async def test_fetch_industry_signals(self, tmp_path):
        """산업 신호 수집"""
        collector = ExternalSignalCollector(data_dir=str(tmp_path / "signals"))
        collector.fetch_rss_articles = AsyncMock(return_value=[])
        collector.fetch_reddit_trends = AsyncMock(return_value=[])

        result = await collector.fetch_industry_signals(keywords=["k-beauty"])

        assert isinstance(result, list)
        collector.fetch_reddit_trends.assert_called_once()


class TestExternalSignalCollectorTavily:
    """Tavily API 수집 테스트"""

    @pytest.mark.asyncio
    async def test_fetch_tavily_no_client(self, tmp_path):
        """Tavily 클라이언트 없으면 빈 리스트"""
        collector = ExternalSignalCollector(data_dir=str(tmp_path / "signals"))
        collector._tavily_client = None

        result = await collector.fetch_tavily_news(brands=["LANEIGE"])
        assert result == []

    @pytest.mark.asyncio
    async def test_fetch_tavily_success(self, tmp_path):
        """Tavily 수집 성공"""
        collector = ExternalSignalCollector(data_dir=str(tmp_path / "signals"))

        mock_result = MagicMock()
        mock_result.title = "LANEIGE News"
        mock_result.content = "K-Beauty brand LANEIGE..."
        mock_result.url = "https://example.com/news"
        mock_result.published_date = "2026-01-15"
        mock_result.score = 0.85
        mock_result.source = "example.com"
        mock_result.reliability_score = 0.9

        mock_tavily = AsyncMock()
        mock_tavily.search_beauty_news = AsyncMock(return_value=[mock_result])
        collector._tavily_client = mock_tavily

        with patch.object(collector, "_save_signals"):
            result = await collector.fetch_tavily_news(brands=["LANEIGE"], max_results=5)

        assert len(result) == 1
        assert result[0].source == "tavily_news"
        assert result[0].title == "LANEIGE News"

    @pytest.mark.asyncio
    async def test_fetch_tavily_error(self, tmp_path):
        """Tavily 수집 에러"""
        collector = ExternalSignalCollector(data_dir=str(tmp_path / "signals"))

        mock_tavily = AsyncMock()
        mock_tavily.search_beauty_news = AsyncMock(side_effect=Exception("API Error"))
        collector._tavily_client = mock_tavily

        result = await collector.fetch_tavily_news(brands=["LANEIGE"])
        assert result == []


class TestExternalSignalCollectorFetchAllNews:
    """fetch_all_news 통합 메서드 테스트"""

    @pytest.mark.asyncio
    async def test_fetch_all_news_basic(self, tmp_path):
        """전체 뉴스 수집"""
        collector = ExternalSignalCollector(data_dir=str(tmp_path / "signals"))
        collector._tavily_client = None
        collector.fetch_all_rss_feeds = AsyncMock(return_value=[])
        collector.fetch_reddit_trends = AsyncMock(return_value=[])

        result = await collector.fetch_all_news(include_tavily=False)

        assert isinstance(result, list)
        collector.fetch_all_rss_feeds.assert_called_once()
        collector.fetch_reddit_trends.assert_called_once()

    @pytest.mark.asyncio
    async def test_fetch_all_news_deduplication(self, tmp_path):
        """중복 URL 제거"""
        collector = ExternalSignalCollector(data_dir=str(tmp_path / "signals"))
        collector._tavily_client = None

        signal1 = ExternalSignal(
            signal_id="SID-1",
            source="allure",
            tier=SignalTier.TIER3_AUTHORITY.value,
            title="Article 1",
            content="Content",
            url="https://example.com/same-url",
            published_at="2026-01-15",
            collected_at="2026-01-15T10:00:00",
            relevance_score=0.8,
            metadata={"reliability_score": 0.8},
        )
        signal2 = ExternalSignal(
            signal_id="SID-2",
            source="byrdie",
            tier=SignalTier.TIER3_AUTHORITY.value,
            title="Article 2",
            content="Content",
            url="https://example.com/same-url",  # 같은 URL
            published_at="2026-01-15",
            collected_at="2026-01-15T10:00:00",
            relevance_score=0.7,
            metadata={"reliability_score": 0.7},
        )

        collector.fetch_all_rss_feeds = AsyncMock(return_value=[signal1, signal2])
        collector.fetch_reddit_trends = AsyncMock(return_value=[])

        result = await collector.fetch_all_news(include_tavily=False, include_reddit=False)

        assert len(result) == 1  # 중복 제거됨


class TestExternalSignalCollectorUtilities:
    """유틸리티 메서드 테스트"""

    def test_generate_signal_id(self, tmp_path):
        """신호 ID 생성"""
        collector = ExternalSignalCollector(data_dir=str(tmp_path / "signals"))

        id1 = collector._generate_signal_id("allure")
        id2 = collector._generate_signal_id("reddit")

        assert id1.startswith("SID-ALL-")
        assert id2.startswith("SID-RED-")
        assert id1 != id2

    def test_save_and_load_signals(self, tmp_path):
        """신호 저장 및 로드"""
        collector = ExternalSignalCollector(data_dir=str(tmp_path / "signals"))

        collector.signals = [
            ExternalSignal(
                signal_id="SID-TST-20260115-0001",
                source="allure",
                tier=SignalTier.TIER3_AUTHORITY.value,
                title="Test Article",
                content="Test content",
                url="https://example.com",
                published_at="2026-01-15",
                collected_at="2026-01-15T10:00:00",
            )
        ]

        collector._save_signals()

        # 새 인스턴스로 로드
        collector2 = ExternalSignalCollector(data_dir=str(tmp_path / "signals"))
        collector2._load_signals()

        assert len(collector2.signals) == 1
        assert collector2.signals[0].signal_id == "SID-TST-20260115-0001"

    def test_load_signals_no_file(self, tmp_path):
        """파일 없으면 빈 리스트"""
        collector = ExternalSignalCollector(data_dir=str(tmp_path / "signals"))
        collector._load_signals()
        assert collector.signals == []

    def test_load_signals_invalid_json(self, tmp_path):
        """잘못된 JSON"""
        data_dir = tmp_path / "signals"
        data_dir.mkdir(parents=True)
        (data_dir / "signals.json").write_text("invalid json{", encoding="utf-8")

        collector = ExternalSignalCollector(data_dir=str(data_dir))
        collector._load_signals()
        assert collector.signals == []

    def test_get_signals_for_kg(self, tmp_path):
        """KG용 신호 변환"""
        collector = ExternalSignalCollector(data_dir=str(tmp_path / "signals"))
        collector.signals = [
            ExternalSignal(
                signal_id="SID-1",
                source="allure",
                tier=SignalTier.TIER3_AUTHORITY.value,
                title="Test",
                content="Content",
                url="https://example.com",
                published_at="2026-01-15",
                collected_at="2026-01-15T10:00:00",
            )
        ]

        result = collector.get_signals_for_kg()

        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["signal_id"] == "SID-1"

    def test_get_stats(self, tmp_path):
        """통계 반환"""
        collector = ExternalSignalCollector(data_dir=str(tmp_path / "signals"))
        collector.signals = [
            ExternalSignal(
                signal_id="SID-1",
                source="allure",
                tier=SignalTier.TIER3_AUTHORITY.value,
                title="Test 1",
                content="",
                url="",
                published_at="2026-01-15",
                collected_at="2026-01-15T10:00:00",
            ),
            ExternalSignal(
                signal_id="SID-2",
                source="reddit",
                tier=SignalTier.TIER2_VALIDATION.value,
                title="Test 2",
                content="",
                url="",
                published_at="2026-01-15",
                collected_at="2026-01-15T11:00:00",
            ),
        ]

        stats = collector.get_stats()

        assert stats["total_signals"] == 2
        assert SignalTier.TIER3_AUTHORITY.value in stats["by_tier"]
        assert SignalTier.TIER2_VALIDATION.value in stats["by_tier"]
        assert "allure" in stats["by_source"]
        assert "reddit" in stats["by_source"]

    def test_get_stats_empty(self, tmp_path):
        """빈 통계"""
        collector = ExternalSignalCollector(data_dir=str(tmp_path / "signals"))

        stats = collector.get_stats()

        assert stats["total_signals"] == 0
        assert stats["last_updated"] is None

    def test_get_source_reliability(self, tmp_path):
        """매체별 신뢰도"""
        collector = ExternalSignalCollector(data_dir=str(tmp_path / "signals"))

        assert collector.get_source_reliability("allure") == 0.8
        assert collector.get_source_reliability("reddit") == 0.5
        assert collector.get_source_reliability("tiktok") == 0.5
        assert collector.get_source_reliability("manual") == 0.9
        assert collector.get_source_reliability("unknown_source") == 0.5

    def test_create_source_reference(self, tmp_path):
        """출처 참조 생성"""
        collector = ExternalSignalCollector(data_dir=str(tmp_path / "signals"))

        signal = ExternalSignal(
            signal_id="SID-1",
            source="allure",
            tier=SignalTier.TIER3_AUTHORITY.value,
            title="Test Article",
            content="Content",
            url="https://example.com",
            published_at="2026-01-15",
            collected_at="2026-01-15T10:00:00",
        )

        ref = collector.create_source_reference(signal)

        assert ref["id"] == "SID-1"
        assert ref["title"] == "Test Article"
        assert ref["publisher"] == "Allure"
        assert ref["source_type"] == "news"
        assert ref["reliability_score"] == 0.8


class TestRSSFeedsConfig:
    """RSS 피드 설정 테스트"""

    def test_rss_feeds_defined(self):
        """RSS 피드가 정의되어 있어야 함"""
        assert len(RSS_FEEDS) > 0

    def test_beauty_keywords_defined(self):
        """뷰티 키워드가 정의되어 있어야 함"""
        assert len(BEAUTY_KEYWORDS) > 0
        assert "laneige" in BEAUTY_KEYWORDS
        assert "k-beauty" in BEAUTY_KEYWORDS

    def test_kbeauty_keywords_defined(self):
        """K-Beauty 키워드가 정의되어 있어야 함"""
        assert len(KBEAUTY_KEYWORDS) > 0
        assert "laneige" in KBEAUTY_KEYWORDS
