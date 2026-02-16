"""
MorningBriefGenerator 단위 테스트
=================================
src/tools/intelligence/morning_brief.py 테스트

테스트 구조:
1. MorningBriefData - 데이터 클래스 테스트
2. MorningBriefGenerator 초기화
3. _get_korean_day - 요일 한글 변환
4. _analyze_crawl_data - 크롤링 데이터 분석
5. _analyze_competitors - 경쟁사 동향 분석
6. _calculate_category_stats - 카테고리별 통계
7. _add_metrics - KPI 메트릭 추가
8. _generate_ai_insights - LLM 인사이트 (mocked)
9. generate - 종합 생성
10. render_morning_brief_html - HTML 렌더링
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.tools.intelligence.morning_brief import (
    MorningBriefData,
    MorningBriefGenerator,
    render_morning_brief_html,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def generator():
    """MorningBriefGenerator 기본 인스턴스"""
    return MorningBriefGenerator()


@pytest.fixture
def sample_crawl_data():
    """샘플 크롤링 데이터"""
    return {
        "category": "Lip Care",
        "products": [
            {"brand": "LANEIGE", "rank": 3, "product_name": "Lip Sleeping Mask", "asin": "B001"},
            {"brand": "LANEIGE", "rank": 8, "product_name": "Water Bank", "asin": "B002"},
            {"brand": "AQUAPHOR", "rank": 1, "product_name": "Lip Repair", "asin": "B003"},
            {"brand": "BURT'S BEES", "rank": 2, "product_name": "Lip Balm", "asin": "B004"},
            {"brand": "COSRX", "rank": 4, "product_name": "Lip Mask", "asin": "B005"},
            {"brand": "SUMMER FRIDAYS", "rank": 5, "product_name": "Lip Balm", "asin": "B006"},
        ],
    }


@pytest.fixture
def sample_previous_data():
    """전일 크롤링 데이터"""
    return {
        "products": [
            {"brand": "LANEIGE", "rank": 5, "asin": "B001"},
            {"brand": "LANEIGE", "rank": 12, "asin": "B002"},
            {"brand": "AQUAPHOR", "rank": 1, "asin": "B003"},
            {"brand": "BURT'S BEES", "rank": 3, "asin": "B004"},
            {"brand": "COSRX", "rank": 15, "asin": "B005"},
        ],
    }


@pytest.fixture
def sample_metrics_data():
    """샘플 메트릭 데이터"""
    return {
        "sos": 15.5,
        "alerts": [
            {"severity": "critical", "message": "LANEIGE 순위 급락"},
            {"severity": "warning", "message": "경쟁사 신제품 출시"},
            {"severity": "critical", "message": "SoS 5% 이하 진입"},
        ],
    }


@pytest.fixture
def sample_brief():
    """샘플 MorningBriefData"""
    return MorningBriefData(
        date="2026.02.17",
        day_of_week="화",
        laneige_avg_rank=5.5,
        laneige_rank_change=2.0,
        laneige_top10_count=2,
        laneige_sos=12.5,
        competitor_highlights=["AQUAPHOR #1: Lip Repair"],
        market_changes=["🔺 COSRX 11등 상승 (#15→#4) Lip Mask"],
        ai_summary="LANEIGE가 안정적인 성과를 보이고 있습니다.",
        action_points=["가격 모니터링 강화", "경쟁사 신제품 분석"],
        ai_recommendations=["COSRX 급상승 모니터링 필요"],
        critical_alerts=["SoS 하락 감지"],
    )


# =============================================================================
# 1. MorningBriefData Tests
# =============================================================================


class TestMorningBriefData:
    """MorningBriefData 데이터 클래스 테스트"""

    def test_default_creation(self):
        """기본값 생성"""
        brief = MorningBriefData(date="2026.02.17", day_of_week="화")
        assert brief.date == "2026.02.17"
        assert brief.day_of_week == "화"
        assert brief.laneige_products == []
        assert brief.laneige_avg_rank == 0.0
        assert brief.laneige_rank_change == 0.0
        assert brief.laneige_top10_count == 0
        assert brief.laneige_sos == 0.0
        assert brief.competitor_highlights == []
        assert brief.market_changes == []
        assert brief.category_stats == {}
        assert brief.alerts_count == 0
        assert brief.critical_alerts == []
        assert brief.action_points == []
        assert brief.ai_summary == ""
        assert brief.ai_recommendations == []

    def test_creation_with_values(self):
        """값 설정 생성"""
        brief = MorningBriefData(
            date="2026.02.17",
            day_of_week="화",
            laneige_avg_rank=5.5,
            laneige_sos=12.5,
        )
        assert brief.laneige_avg_rank == 5.5
        assert brief.laneige_sos == 12.5


# =============================================================================
# 2. MorningBriefGenerator 초기화 Tests
# =============================================================================


class TestMorningBriefGeneratorInit:
    """MorningBriefGenerator 초기화 테스트"""

    def test_default_init(self):
        """기본 초기화"""
        gen = MorningBriefGenerator()
        assert gen.model == "gpt-4.1-mini"
        assert gen.data_source is None

    def test_custom_init(self):
        """커스텀 초기화"""
        mock_ds = MagicMock()
        gen = MorningBriefGenerator(model="gpt-4", data_source=mock_ds)
        assert gen.model == "gpt-4"
        assert gen.data_source is mock_ds

    def test_temperature_from_env(self):
        """환경 변수에서 temperature 읽기"""
        with patch.dict("os.environ", {"LLM_TEMPERATURE_INSIGHT": "0.8"}):
            gen = MorningBriefGenerator()
            assert gen.temperature == 0.8


# =============================================================================
# 3. _get_korean_day Tests
# =============================================================================


class TestGetKoreanDay:
    """요일 한글 변환 테스트"""

    def test_all_days(self, generator):
        """모든 요일 변환"""
        expected = ["월", "화", "수", "목", "금", "토", "일"]
        for i, day in enumerate(expected):
            assert generator._get_korean_day(i) == day


# =============================================================================
# 4. _analyze_crawl_data Tests
# =============================================================================


class TestAnalyzeCrawlData:
    """크롤링 데이터 분석 테스트"""

    @pytest.mark.asyncio
    async def test_laneige_avg_rank(self, generator, sample_crawl_data):
        """LANEIGE 평균 순위 계산"""
        brief = MorningBriefData(date="2026.02.17", day_of_week="화")
        await generator._analyze_crawl_data(brief, sample_crawl_data)

        # LANEIGE ranks: 3, 8 -> avg = 5.5
        assert brief.laneige_avg_rank == 5.5

    @pytest.mark.asyncio
    async def test_laneige_top10_count(self, generator, sample_crawl_data):
        """LANEIGE Top 10 진입 제품 수"""
        brief = MorningBriefData(date="2026.02.17", day_of_week="화")
        await generator._analyze_crawl_data(brief, sample_crawl_data)

        # Both rank 3 and 8 are in top 10
        assert brief.laneige_top10_count == 2

    @pytest.mark.asyncio
    async def test_laneige_sos(self, generator, sample_crawl_data):
        """LANEIGE SoS 계산"""
        brief = MorningBriefData(date="2026.02.17", day_of_week="화")
        await generator._analyze_crawl_data(brief, sample_crawl_data)

        # 2 LANEIGE out of 6 total = 33.33%
        expected_sos = (2 / 6) * 100
        assert abs(brief.laneige_sos - expected_sos) < 0.1

    @pytest.mark.asyncio
    async def test_rank_change_from_previous_data(
        self, generator, sample_crawl_data, sample_previous_data
    ):
        """전일 대비 순위 변화 계산"""
        brief = MorningBriefData(date="2026.02.17", day_of_week="화")
        await generator._analyze_crawl_data(brief, sample_crawl_data, sample_previous_data)

        # Current avg: (3+8)/2 = 5.5, Previous avg: (5+12)/2 = 8.5
        # Change = prev_avg - current_avg = 8.5 - 5.5 = 3.0 (positive = improvement)
        assert brief.laneige_rank_change == 3.0

    @pytest.mark.asyncio
    async def test_no_laneige_products(self, generator):
        """LANEIGE 제품이 없는 경우"""
        crawl_data = {
            "category": "Face Powder",
            "products": [
                {"brand": "MAYBELLINE", "rank": 1, "product_name": "Powder"},
            ],
        }
        brief = MorningBriefData(date="2026.02.17", day_of_week="화")
        await generator._analyze_crawl_data(brief, crawl_data)

        assert brief.laneige_avg_rank == 0.0
        assert brief.laneige_top10_count == 0
        assert brief.laneige_sos == 0.0

    @pytest.mark.asyncio
    async def test_empty_products(self, generator):
        """제품이 없는 경우"""
        crawl_data = {"category": "Empty", "products": []}
        brief = MorningBriefData(date="2026.02.17", day_of_week="화")
        await generator._analyze_crawl_data(brief, crawl_data)

        assert brief.laneige_products == []
        assert brief.laneige_sos == 0.0


# =============================================================================
# 5. _analyze_competitors Tests
# =============================================================================


class TestAnalyzeCompetitors:
    """경쟁사 동향 분석 테스트"""

    @pytest.mark.asyncio
    async def test_competitor_highlights_top5(self, generator, sample_crawl_data):
        """Top 5 경쟁사 하이라이트"""
        brief = MorningBriefData(date="2026.02.17", day_of_week="화")
        products = sample_crawl_data["products"]
        await generator._analyze_competitors(brief, products)

        # AQUAPHOR #1 and BURT'S BEES #2 should be highlighted (top 5, not LANEIGE)
        assert len(brief.competitor_highlights) >= 1

    @pytest.mark.asyncio
    async def test_market_changes_with_previous_data(
        self, generator, sample_crawl_data, sample_previous_data
    ):
        """전일 대비 순위 변동 감지"""
        brief = MorningBriefData(date="2026.02.17", day_of_week="화")
        products = sample_crawl_data["products"]
        await generator._analyze_competitors(brief, products, sample_previous_data)

        # COSRX changed from 15 to 4 = 11 places up (>= 10)
        cosrx_changes = [c for c in brief.market_changes if "COSRX" in c]
        assert len(cosrx_changes) >= 1

    @pytest.mark.asyncio
    async def test_no_previous_data_no_market_changes(self, generator, sample_crawl_data):
        """이전 데이터 없으면 시장 변화 없음"""
        brief = MorningBriefData(date="2026.02.17", day_of_week="화")
        products = sample_crawl_data["products"]
        await generator._analyze_competitors(brief, products, None)

        assert brief.market_changes == []


# =============================================================================
# 6. _calculate_category_stats Tests
# =============================================================================


class TestCalculateCategoryStats:
    """카테고리별 통계 테스트"""

    def test_category_stats_basic(self, generator, sample_crawl_data):
        """기본 카테고리 통계"""
        brief = MorningBriefData(date="2026.02.17", day_of_week="화")
        generator._calculate_category_stats(brief, sample_crawl_data)

        assert "Lip Care" in brief.category_stats
        stats = brief.category_stats["Lip Care"]
        assert stats["total_products"] == 6
        assert stats["laneige_count"] == 2
        assert stats["laneige_best_rank"] == 3  # rank 3 is best
        assert stats["top_brand"] == "LANEIGE"  # first product's brand

    def test_category_stats_no_laneige(self, generator):
        """LANEIGE 없는 카테고리"""
        crawl_data = {
            "category": "Face Powder",
            "products": [
                {"brand": "MAYBELLINE", "rank": 1},
            ],
        }
        brief = MorningBriefData(date="2026.02.17", day_of_week="화")
        generator._calculate_category_stats(brief, crawl_data)

        stats = brief.category_stats["Face Powder"]
        assert stats["laneige_count"] == 0
        assert stats["laneige_best_rank"] is None

    def test_category_stats_empty_products(self, generator):
        """제품 없는 카테고리"""
        crawl_data = {"category": "Empty", "products": []}
        brief = MorningBriefData(date="2026.02.17", day_of_week="화")
        generator._calculate_category_stats(brief, crawl_data)

        stats = brief.category_stats["Empty"]
        assert stats["total_products"] == 0
        assert stats["top_brand"] is None


# =============================================================================
# 7. _add_metrics Tests
# =============================================================================


class TestAddMetrics:
    """KPI 메트릭 추가 테스트"""

    def test_add_sos(self, generator, sample_metrics_data):
        """SoS 메트릭 추가"""
        brief = MorningBriefData(date="2026.02.17", day_of_week="화")
        generator._add_metrics(brief, sample_metrics_data)
        assert brief.laneige_sos == 15.5

    def test_add_alerts(self, generator, sample_metrics_data):
        """알림 메트릭 추가"""
        brief = MorningBriefData(date="2026.02.17", day_of_week="화")
        generator._add_metrics(brief, sample_metrics_data)

        assert brief.alerts_count == 3
        # critical alerts only (max 3)
        assert len(brief.critical_alerts) == 2
        assert "LANEIGE 순위 급락" in brief.critical_alerts
        assert "SoS 5% 이하 진입" in brief.critical_alerts

    def test_add_metrics_empty(self, generator):
        """빈 메트릭 데이터"""
        brief = MorningBriefData(date="2026.02.17", day_of_week="화")
        generator._add_metrics(brief, {})
        assert brief.laneige_sos == 0.0
        assert brief.alerts_count == 0

    def test_add_metrics_no_critical_alerts(self, generator):
        """critical 알림 없는 경우"""
        metrics = {
            "alerts": [
                {"severity": "warning", "message": "경고 알림"},
            ]
        }
        brief = MorningBriefData(date="2026.02.17", day_of_week="화")
        generator._add_metrics(brief, metrics)

        assert brief.alerts_count == 1
        assert brief.critical_alerts == []


# =============================================================================
# 8. _generate_ai_insights Tests (LLM mocked)
# =============================================================================


class TestGenerateAIInsights:
    """LLM 인사이트 생성 테스트 (mocked)"""

    @pytest.mark.asyncio
    async def test_ai_insights_success(self, generator):
        """LLM 인사이트 생성 성공"""
        llm_result = {
            "summary": "LANEIGE가 Lip Care에서 안정적인 성과를 보이고 있습니다.",
            "action_points": ["가격 모니터링", "경쟁사 분석", "프로모션 계획"],
            "warnings": ["COSRX 급상승 주의"],
        }
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content=json.dumps(llm_result)))]
        with patch(
            "src.tools.intelligence.morning_brief.acompletion",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            brief = MorningBriefData(date="2026.02.17", day_of_week="화")
            await generator._generate_ai_insights(brief)

        assert brief.ai_summary == llm_result["summary"]
        assert brief.action_points == llm_result["action_points"]
        assert brief.ai_recommendations == llm_result["warnings"]

    @pytest.mark.asyncio
    async def test_ai_insights_failure_fallback(self, generator):
        """LLM 실패 시 폴백 메시지"""
        with patch(
            "src.tools.intelligence.morning_brief.acompletion",
            new_callable=AsyncMock,
            side_effect=Exception("LLM API error"),
        ):
            brief = MorningBriefData(date="2026.02.17", day_of_week="화")
            await generator._generate_ai_insights(brief)

        assert "AI 분석을 생성하지 못했습니다" in brief.ai_summary
        assert len(brief.action_points) == 2


# =============================================================================
# 9. generate - 종합 생성 Tests
# =============================================================================


class TestGenerate:
    """종합 Morning Brief 생성 테스트"""

    @pytest.mark.asyncio
    async def test_generate_with_crawl_data(self, generator, sample_crawl_data):
        """크롤링 데이터로 생성"""
        llm_result = {
            "summary": "테스트 요약",
            "action_points": ["액션1"],
            "warnings": [],
        }
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content=json.dumps(llm_result)))]
        with patch(
            "src.tools.intelligence.morning_brief.acompletion",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            brief = await generator.generate(crawl_data=sample_crawl_data)

        assert isinstance(brief, MorningBriefData)
        assert brief.date  # should have a date
        assert brief.day_of_week  # should have a day of week
        assert brief.laneige_avg_rank == 5.5

    @pytest.mark.asyncio
    async def test_generate_without_data(self, generator):
        """데이터 없이 생성"""
        llm_result = {
            "summary": "데이터 없음",
            "action_points": [],
            "warnings": [],
        }
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content=json.dumps(llm_result)))]
        with patch(
            "src.tools.intelligence.morning_brief.acompletion",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            brief = await generator.generate()

        assert isinstance(brief, MorningBriefData)
        assert brief.laneige_avg_rank == 0.0

    @pytest.mark.asyncio
    async def test_generate_with_metrics(self, generator, sample_metrics_data):
        """메트릭 데이터로 생성"""
        llm_result = {
            "summary": "메트릭 기반 요약",
            "action_points": [],
            "warnings": [],
        }
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content=json.dumps(llm_result)))]
        with patch(
            "src.tools.intelligence.morning_brief.acompletion",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            brief = await generator.generate(metrics_data=sample_metrics_data)

        assert brief.laneige_sos == 15.5
        assert brief.alerts_count == 3

    @pytest.mark.asyncio
    async def test_generate_with_all_data(
        self, generator, sample_crawl_data, sample_previous_data, sample_metrics_data
    ):
        """모든 데이터로 종합 생성"""
        llm_result = {
            "summary": "종합 분석",
            "action_points": ["분석1", "분석2"],
            "warnings": ["주의1"],
        }
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content=json.dumps(llm_result)))]
        with patch(
            "src.tools.intelligence.morning_brief.acompletion",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            brief = await generator.generate(
                crawl_data=sample_crawl_data,
                metrics_data=sample_metrics_data,
                previous_data=sample_previous_data,
            )

        assert brief.laneige_avg_rank == 5.5
        # metrics_data overwrites SoS from crawl analysis
        assert brief.laneige_sos == 15.5
        assert brief.laneige_rank_change == 3.0


# =============================================================================
# 10. render_morning_brief_html Tests
# =============================================================================


class TestRenderMorningBriefHtml:
    """HTML 렌더링 테스트"""

    def test_basic_html_structure(self, sample_brief):
        """기본 HTML 구조"""
        html = render_morning_brief_html(sample_brief)
        assert "<!DOCTYPE html>" in html
        assert "AMORE Daily Brief" in html
        assert "2026.02.17" in html
        assert "(화)" in html

    def test_laneige_performance_section(self, sample_brief):
        """LANEIGE 성과 섹션"""
        html = render_morning_brief_html(sample_brief)
        assert "5.5" in html  # avg rank
        assert "12.5" in html  # sos

    def test_rank_change_positive(self, sample_brief):
        """순위 상승 (양수)"""
        sample_brief.laneige_rank_change = 2.0
        html = render_morning_brief_html(sample_brief)
        assert "▲" in html
        assert "#059669" in html  # green color

    def test_rank_change_negative(self):
        """순위 하락 (음수)"""
        brief = MorningBriefData(
            date="2026.02.17",
            day_of_week="화",
            laneige_rank_change=-3.0,
        )
        html = render_morning_brief_html(brief)
        assert "▼" in html
        assert "#dc2626" in html  # red color

    def test_rank_change_zero(self):
        """순위 변동 없음"""
        brief = MorningBriefData(
            date="2026.02.17",
            day_of_week="화",
            laneige_rank_change=0.0,
        )
        html = render_morning_brief_html(brief)
        assert "━" in html

    def test_competitor_section_with_highlights(self, sample_brief):
        """경쟁사 섹션 (하이라이트 있음)"""
        html = render_morning_brief_html(sample_brief)
        assert "AQUAPHOR" in html

    def test_competitor_section_empty(self):
        """경쟁사 섹션 (하이라이트 없음)"""
        brief = MorningBriefData(date="2026.02.17", day_of_week="화")
        html = render_morning_brief_html(brief)
        assert "특이사항 없음" in html

    def test_market_changes_section(self, sample_brief):
        """시장 변화 섹션"""
        html = render_morning_brief_html(sample_brief)
        assert "COSRX" in html
        assert "주요 순위 변동" in html

    def test_no_market_changes(self):
        """시장 변화 없음"""
        brief = MorningBriefData(date="2026.02.17", day_of_week="화")
        html = render_morning_brief_html(brief)
        # market_changes_section should be empty string
        assert "주요 순위 변동" not in html

    def test_action_points_section(self, sample_brief):
        """액션 포인트 섹션"""
        html = render_morning_brief_html(sample_brief)
        assert "가격 모니터링 강화" in html
        assert "경쟁사 신제품 분석" in html

    def test_no_action_points(self):
        """액션 포인트 없음"""
        brief = MorningBriefData(date="2026.02.17", day_of_week="화")
        html = render_morning_brief_html(brief)
        assert "액션 포인트 없음" in html

    def test_warnings_section(self, sample_brief):
        """주의사항 섹션"""
        html = render_morning_brief_html(sample_brief)
        assert "주의 사항" in html

    def test_no_warnings(self):
        """주의사항 없음"""
        brief = MorningBriefData(date="2026.02.17", day_of_week="화")
        html = render_morning_brief_html(brief)
        assert "주의 사항" not in html

    def test_ai_summary_fallback(self):
        """AI 요약 없을 때 폴백"""
        brief = MorningBriefData(date="2026.02.17", day_of_week="화")
        html = render_morning_brief_html(brief)
        assert "데이터 분석 중입니다." in html

    def test_footer_present(self, sample_brief):
        """푸터 존재"""
        html = render_morning_brief_html(sample_brief)
        assert "AMORE Market Intelligence Agent" in html
        assert "Amazon US Market Analysis" in html
