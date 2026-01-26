"""
Tests for IRReportParser

IR 보고서 파서 테스트
"""

import pytest
from datetime import datetime

from src.tools.ir_report_parser import (
    IRReportParser,
    IRReport,
    QuarterlyFinancials,
    RegionalPerformance,
    BrandHighlight,
    ReportType,
    Region,
    PREDEFINED_IR_DATA
)


class TestQuarterlyFinancials:
    """QuarterlyFinancials 데이터클래스 테스트"""

    def test_create_financials(self):
        """재무 데이터 생성 테스트"""
        financials = QuarterlyFinancials(
            year="2025",
            quarter="Q3",
            revenue_krw=1016.9,
            operating_profit_krw=91.9,
            net_income_krw=68.2,
            operating_margin=9.0,
            revenue_yoy=4.1,
            op_yoy=41.0
        )

        assert financials.year == "2025"
        assert financials.quarter == "Q3"
        assert financials.revenue_krw == 1016.9
        assert financials.operating_margin == 9.0

    def test_financials_to_insight_format(self):
        """인사이트 포맷 변환 테스트"""
        financials = QuarterlyFinancials(
            year="2025",
            quarter="Q3",
            revenue_krw=1016.9,
            revenue_yoy=4.1
        )

        insight = financials.to_insight_format()

        assert "Q3 2025" in insight
        assert "1016.9B KRW" in insight
        assert "+4.1%" in insight


class TestRegionalPerformance:
    """RegionalPerformance 데이터클래스 테스트"""

    def test_create_regional_performance(self):
        """지역별 실적 생성 테스트"""
        regional = RegionalPerformance(
            region="Americas",
            year="2025",
            quarter="Q3",
            revenue_krw=156.8,
            revenue_yoy=6.9,
            key_highlights=["Amazon Prime Day 2배 성장"]
        )

        assert regional.region == "Americas"
        assert regional.revenue_krw == 156.8
        assert regional.revenue_yoy == 6.9
        assert len(regional.key_highlights) == 1

    def test_regional_to_insight_format(self):
        """인사이트 포맷 변환 테스트"""
        regional = RegionalPerformance(
            region="Americas",
            year="2025",
            quarter="Q3",
            revenue_krw=156.8,
            revenue_yoy=6.9
        )

        insight = regional.to_insight_format()

        assert "Americas" in insight
        assert "156.8B KRW" in insight
        assert "+6.9%" in insight


class TestBrandHighlight:
    """BrandHighlight 데이터클래스 테스트"""

    def test_create_brand_highlight(self):
        """브랜드 하이라이트 생성 테스트"""
        highlight = BrandHighlight(
            brand="LANEIGE",
            year="2025",
            quarter="Q3",
            region="Americas",
            highlights=[
                "'Next-Gen Hydration' 캠페인으로 스킨케어 매출 증가",
                "Tracckr Brand Viral Index 8월 2위 기록"
            ],
            products=["Lip Sleeping Mask", "Cream Skin", "Water Bank"]
        )

        assert highlight.brand == "LANEIGE"
        assert highlight.region == "Americas"
        assert len(highlight.highlights) == 2
        assert len(highlight.products) == 3


class TestIRReport:
    """IRReport 데이터클래스 테스트"""

    def test_create_ir_report(self):
        """IR 보고서 생성 테스트"""
        financials = QuarterlyFinancials(
            year="2025",
            quarter="Q3",
            revenue_krw=1016.9
        )

        report = IRReport(
            report_id="IR-2025-Q3",
            report_type=ReportType.EARNINGS_RELEASE.value,
            year="2025",
            quarter="Q3",
            release_date="2025-11-06",
            financials=financials
        )

        assert report.report_id == "IR-2025-Q3"
        assert report.year == "2025"
        assert report.quarter == "Q3"
        assert report.reliability_score == 1.0  # IR은 최고 신뢰도

    def test_ir_report_to_dict(self):
        """to_dict 테스트"""
        report = IRReport(
            report_id="IR-2025-Q3",
            report_type=ReportType.EARNINGS_RELEASE.value,
            year="2025",
            quarter="Q3",
            release_date="2025-11-06"
        )

        data = report.to_dict()

        assert data["report_id"] == "IR-2025-Q3"
        assert data["reliability_score"] == 1.0


class TestIRReportParser:
    """IRReportParser 테스트"""

    @pytest.fixture
    def parser(self, tmp_path):
        """테스트용 parser 인스턴스"""
        return IRReportParser(data_dir=str(tmp_path / "ir_reports"))

    @pytest.mark.asyncio
    async def test_initialize(self, parser):
        """초기화 테스트"""
        await parser.initialize()

        # 미리 정의된 데이터가 로드되어야 함
        assert len(parser.reports) > 0
        assert "IR-2025-Q3" in parser.reports

        await parser.close()

    @pytest.mark.asyncio
    async def test_load_predefined_data(self, parser):
        """미리 정의된 데이터 로드 테스트"""
        await parser.initialize()

        report = parser.reports.get("IR-2025-Q3")

        assert report is not None
        assert report.year == "2025"
        assert report.quarter == "Q3"
        assert report.release_date == "2025-11-06"

        # 재무 데이터 확인
        assert report.financials is not None
        assert report.financials.revenue_krw == 1016.9
        assert report.financials.revenue_yoy == 4.1

        # 지역별 실적 확인
        assert len(report.regional_performance) > 0

        # Americas 찾기
        americas = None
        for perf in report.regional_performance:
            if perf.region == "Americas":
                americas = perf
                break

        assert americas is not None
        assert americas.revenue_krw == 156.8
        assert americas.revenue_yoy == 6.9

        # 브랜드 하이라이트 확인
        assert len(report.brand_highlights) > 0

        await parser.close()

    @pytest.mark.asyncio
    async def test_get_quarterly_data(self, parser):
        """분기 데이터 조회 테스트"""
        await parser.initialize()

        report = parser.get_quarterly_data("2025", "Q3")

        assert report is not None
        assert report.year == "2025"
        assert report.quarter == "Q3"

        await parser.close()

    @pytest.mark.asyncio
    async def test_get_quarterly_data_not_found(self, parser):
        """분기 데이터 조회 - 없는 경우"""
        await parser.initialize()

        report = parser.get_quarterly_data("2020", "Q1")

        assert report is None

        await parser.close()

    @pytest.mark.asyncio
    async def test_get_latest_report(self, parser):
        """최신 보고서 조회 테스트"""
        await parser.initialize()

        report = parser.get_latest_report()

        assert report is not None
        # 2025 Q3이 최신
        assert report.year == "2025"
        assert report.quarter == "Q3"

        await parser.close()

    @pytest.mark.asyncio
    async def test_get_regional_performance(self, parser):
        """지역별 실적 조회 테스트"""
        await parser.initialize()

        americas = parser.get_regional_performance("Americas", "2025", "Q3")

        assert americas is not None
        assert americas.region == "Americas"
        assert americas.revenue_krw == 156.8
        assert americas.revenue_yoy == 6.9

        await parser.close()

    @pytest.mark.asyncio
    async def test_get_regional_performance_case_insensitive(self, parser):
        """지역별 실적 조회 - 대소문자 무시"""
        await parser.initialize()

        americas = parser.get_regional_performance("americas", "2025", "Q3")

        assert americas is not None

        await parser.close()

    @pytest.mark.asyncio
    async def test_get_brand_highlights(self, parser):
        """브랜드 하이라이트 조회 테스트"""
        await parser.initialize()

        highlights = parser.get_brand_highlights("LANEIGE", "2025", "Q3")

        assert len(highlights) > 0
        assert highlights[0].brand == "LANEIGE"
        assert "Next-Gen Hydration" in highlights[0].highlights[0]

        await parser.close()

    @pytest.mark.asyncio
    async def test_get_americas_insights(self, parser):
        """Americas 인사이트 조회 테스트"""
        await parser.initialize()

        insights = parser.get_americas_insights("2025", "Q3")

        assert "regional_performance" in insights
        assert "brand_highlights" in insights
        assert "key_events" in insights

        # Americas 실적 확인
        assert len(insights["regional_performance"]) > 0
        perf = insights["regional_performance"][0]
        assert perf["revenue_krw"] == 156.8
        assert perf["revenue_yoy"] == 6.9

        # Americas 브랜드 확인
        assert len(insights["brand_highlights"]) > 0

        await parser.close()

    @pytest.mark.asyncio
    async def test_generate_insight_section(self, parser):
        """인사이트 섹션 생성 테스트"""
        await parser.initialize()

        section = parser.generate_insight_section("2025", "Q3")

        assert "아모레퍼시픽 IR 데이터" in section
        assert "Q3 2025" in section
        assert "매출" in section
        assert "Americas" in section

        await parser.close()

    @pytest.mark.asyncio
    async def test_generate_insight_section_latest(self, parser):
        """인사이트 섹션 생성 - 최신"""
        await parser.initialize()

        section = parser.generate_insight_section()

        assert "아모레퍼시픽 IR 데이터" in section

        await parser.close()

    @pytest.mark.asyncio
    async def test_create_source_reference(self, parser):
        """출처 참조 생성 테스트"""
        await parser.initialize()

        ref = parser.create_source_reference("2025", "Q3")

        assert ref["publisher"] == "아모레퍼시픽 IR"
        assert ref["source_type"] == "ir"
        assert ref["reliability_score"] == 1.0
        assert "2025" in ref["title"]
        assert "Q3" in ref["title"]

        await parser.close()

    @pytest.mark.asyncio
    async def test_get_stats(self, parser):
        """통계 반환 테스트"""
        await parser.initialize()

        stats = parser.get_stats()

        assert "total_reports" in stats
        assert "years_covered" in stats
        assert "latest_report" in stats
        assert stats["total_reports"] > 0

        await parser.close()


class TestPredefinedIRData:
    """미리 정의된 IR 데이터 상수 테스트"""

    def test_predefined_data_structure(self):
        """미리 정의된 데이터 구조 테스트"""
        assert "2025_Q3" in PREDEFINED_IR_DATA

        q3_data = PREDEFINED_IR_DATA["2025_Q3"]

        assert "release_date" in q3_data
        assert "financials" in q3_data
        assert "regional" in q3_data
        assert "brand_highlights" in q3_data

    def test_predefined_financials(self):
        """미리 정의된 재무 데이터 테스트"""
        financials = PREDEFINED_IR_DATA["2025_Q3"]["financials"]

        assert financials["revenue_krw"] == 1016.9
        assert financials["operating_profit_krw"] == 91.9
        assert financials["revenue_yoy"] == 4.1

    def test_predefined_regional(self):
        """미리 정의된 지역 데이터 테스트"""
        regional = PREDEFINED_IR_DATA["2025_Q3"]["regional"]

        assert "Americas" in regional
        assert regional["Americas"]["revenue_krw"] == 156.8
        assert regional["Americas"]["revenue_yoy"] == 6.9

    def test_predefined_brand_highlights(self):
        """미리 정의된 브랜드 하이라이트 테스트"""
        brands = PREDEFINED_IR_DATA["2025_Q3"]["brand_highlights"]

        assert "LANEIGE" in brands
        assert len(brands["LANEIGE"]["highlights"]) > 0
        assert len(brands["LANEIGE"]["products"]) > 0


class TestEnums:
    """Enum 테스트"""

    def test_report_type(self):
        """ReportType Enum 테스트"""
        assert ReportType.EARNINGS_RELEASE.value == "earnings_release"
        assert ReportType.ANNUAL_REPORT.value == "annual_report"
        assert ReportType.IR_PRESENTATION.value == "ir_presentation"

    def test_region(self):
        """Region Enum 테스트"""
        assert Region.DOMESTIC.value == "domestic"
        assert Region.AMERICAS.value == "americas"
        assert Region.EMEA.value == "emea"
        assert Region.GREATER_CHINA.value == "greater_china"
        assert Region.OTHER_ASIA.value == "other_asia"
