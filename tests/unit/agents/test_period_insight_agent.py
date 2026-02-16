"""
Unit tests for PeriodInsightAgent
"""

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.agents.period_insight_agent import PeriodInsightAgent, PeriodReport, SectionInsight

# Mark pytest and AsyncMock/patch as used at module level for linter
_USED = (pytest, AsyncMock, patch)

# =========================================================================
# Helper: mock analysis object
# =========================================================================


def _make_analysis():
    """Create a mock PeriodAnalysis object"""
    analysis = MagicMock()
    analysis.start_date = "2026-02-01"
    analysis.end_date = "2026-02-15"
    analysis.total_days = 14
    analysis.laneige_metrics = {
        "start_sos": 4.0,
        "end_sos": 5.0,
        "avg_sos": 4.5,
        "sos_change": 1.0,
        "sos_change_pct": 25.0,
        "avg_product_count": 3.0,
        "rising_products": [{"title": "Lip Mask"}],
        "falling_products": [],
        "top_products": [
            {
                "title": "Lip Sleeping Mask",
                "category_name": "Lip Care",
                "start_rank": 1,
                "end_rank": 2,
                "change": -1,
            }
        ],
    }
    analysis.market_metrics = {
        "avg_hhi": 800,
        "start_hhi": 750,
        "end_hhi": 850,
        "hhi_interpretation": "경쟁적 시장",
    }
    analysis.brand_performance = [
        {"brand": "LANEIGE", "avg_sos": 5.0, "sos_change": 1.0},
        {"brand": "COSRX", "avg_sos": 8.0, "sos_change": -0.5},
    ]
    analysis.competitive_shifts = {
        "new_entrants": ["BrandX"],
        "exits": ["BrandY"],
        "total_brands_start": 50,
        "total_brands_end": 51,
    }
    analysis.category_analysis = {
        "Lip Care": {"avg_sos": 6.0},
        "Skin Care": {"avg_sos": 3.0},
    }
    return analysis


class TestSectionInsight:
    """Test SectionInsight dataclass"""

    def test_to_dict(self):
        section = SectionInsight(
            section_id="exec_summary",
            section_title="Executive Summary",
            content="LANEIGE Lip Sleeping Mask maintains #1 in Lip Care.",
            key_points=["#1 rank maintained", "SoS 5.2%"],
            data_highlights={"rank": 1, "sos": 5.2},
        )
        d = section.to_dict()

        assert d["section_id"] == "exec_summary"
        assert d["section_title"] == "Executive Summary"
        assert len(d["key_points"]) == 2
        assert d["data_highlights"]["rank"] == 1

    def test_default_values(self):
        section = SectionInsight(
            section_id="test",
            section_title="Test",
            content="Test content",
        )
        assert section.key_points == []
        assert section.data_highlights == {}


class TestPeriodReport:
    """Test PeriodReport dataclass"""

    def test_to_dict_empty(self):
        report = PeriodReport(
            start_date="2026-02-01",
            end_date="2026-02-15",
            generated_at=datetime.now().isoformat(),
        )
        d = report.to_dict()

        assert d["start_date"] == "2026-02-01"
        assert d["end_date"] == "2026-02-15"
        assert d["executive_summary"] is None
        assert d["laneige_analysis"] is None

    def test_to_dict_with_sections(self):
        summary = SectionInsight(
            section_id="exec_summary",
            section_title="Executive Summary",
            content="Summary content",
        )
        report = PeriodReport(
            start_date="2026-02-01",
            end_date="2026-02-15",
            generated_at=datetime.now().isoformat(),
            executive_summary=summary,
        )
        d = report.to_dict()

        assert d["executive_summary"] is not None
        assert d["executive_summary"]["section_id"] == "exec_summary"

    def test_metadata_default(self):
        report = PeriodReport(
            start_date="2026-02-01",
            end_date="2026-02-15",
            generated_at=datetime.now().isoformat(),
        )
        assert report.metadata == {}


class TestPeriodInsightAgent:
    """Test PeriodInsightAgent functionality"""

    def test_init_default_model(self):
        agent = PeriodInsightAgent()
        assert agent.model == "gpt-4.1-mini"
        assert agent._external_signals is None

    def test_init_custom_model(self):
        agent = PeriodInsightAgent(model="gpt-4o")
        assert agent.model == "gpt-4o"

    def test_get_system_prompt_with_dates(self):
        agent = PeriodInsightAgent()
        prompt = agent._get_system_prompt(
            start_date="2026-02-01",
            end_date="2026-02-15",
        )

        assert "2026-02-01" in prompt
        assert "2026-02-15" in prompt
        assert "화장품 산업 전문 애널리스트" in prompt

    def test_get_system_prompt_without_dates(self):
        agent = PeriodInsightAgent()
        prompt = agent._get_system_prompt()

        # Should still contain template structure
        assert "애널리스트" in prompt

    def test_constants(self):
        """Test class constants"""
        assert PeriodInsightAgent.MAX_TOKENS == 2000
        assert PeriodInsightAgent.TEMPERATURE == 0.7
        assert "gpt" in PeriodInsightAgent.MODEL

    def test_system_prompt_contains_rules(self):
        """Test system prompt contains key analysis rules"""
        prompt = PeriodInsightAgent.SYSTEM_PROMPT_TEMPLATE
        assert "SoS" in prompt
        assert "HHI" in prompt
        assert "CPI" in prompt
        assert "Unknown" in prompt  # forbidden term rule
        assert "소규모/신흥 브랜드" in prompt


# =========================================================================
# _call_llm (mocked)
# =========================================================================


class TestCallLLM:
    """Test LLM call with mocked acompletion"""

    @pytest.mark.asyncio
    async def test_call_llm_success(self):
        from unittest.mock import AsyncMock, patch

        agent = PeriodInsightAgent()
        agent._analysis_start_date = "2026-02-01"
        agent._analysis_end_date = "2026-02-15"

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Test insight content"

        with patch(
            "src.agents.period_insight_agent.acompletion", new_callable=AsyncMock
        ) as mock_llm:
            mock_llm.return_value = mock_response
            result = await agent._call_llm("Test prompt")

        assert result == "Test insight content"
        mock_llm.assert_called_once()

    @pytest.mark.asyncio
    async def test_call_llm_failure(self):
        from unittest.mock import AsyncMock, patch

        agent = PeriodInsightAgent()

        with patch(
            "src.agents.period_insight_agent.acompletion", new_callable=AsyncMock
        ) as mock_llm:
            mock_llm.side_effect = RuntimeError("API Error")
            result = await agent._call_llm("Test prompt")

        assert "인사이트 생성 실패" in result


# =========================================================================
# _generate_references
# =========================================================================


class TestGenerateReferences:
    def test_no_signals(self):
        agent = PeriodInsightAgent()
        agent._external_signals = None
        result = agent._generate_references()
        assert result.section_id == "8"
        assert "수집된 외부 신호 없음" in result.content
        assert "면책사항" in result.content

    def test_with_signals(self):
        agent = PeriodInsightAgent()
        mock_signal = MagicMock()
        mock_signal.url = "https://example.com/article1"
        mock_signal.title = "K-Beauty Trends 2026"
        mock_signal.source = "tavily_news"
        mock_signal.published_at = "2026-02-10T12:00:00"
        mock_signal.metadata = {"domain": "allure.com"}

        agent._external_signals = {"signals": [mock_signal]}
        result = agent._generate_references()
        assert "K-Beauty Trends 2026" in result.content
        assert result.data_highlights["total_references"] == 1

    def test_with_duplicate_urls(self):
        agent = PeriodInsightAgent()
        mock_signal1 = MagicMock()
        mock_signal1.url = "https://example.com/same"
        mock_signal1.title = "Article 1"
        mock_signal1.source = "news"
        mock_signal1.published_at = "2026-02-10"
        mock_signal1.metadata = {"domain": "news.com"}

        mock_signal2 = MagicMock()
        mock_signal2.url = "https://example.com/same"  # same URL
        mock_signal2.title = "Article 2"
        mock_signal2.source = "news"
        mock_signal2.published_at = "2026-02-11"
        mock_signal2.metadata = {"domain": "news.com"}

        agent._external_signals = {"signals": [mock_signal1, mock_signal2]}
        result = agent._generate_references()
        # Should deduplicate
        assert result.data_highlights["total_references"] == 1


# =========================================================================
# _generate_external_signals
# =========================================================================


class TestGenerateExternalSignals:
    @pytest.mark.asyncio
    async def test_no_signals_returns_info(self):
        agent = PeriodInsightAgent()
        agent._external_signals = {}
        agent._analysis_start_date = "2026-02-01"
        agent._analysis_end_date = "2026-02-15"

        analysis = _make_analysis()
        result = await agent._generate_external_signals(analysis)
        assert result.section_id == "5"
        assert "외부 신호 수집 상태" in result.content
        assert result.data_highlights["total_signals"] == 0

    @pytest.mark.asyncio
    async def test_with_news_signals(self):
        from unittest.mock import AsyncMock, patch

        agent = PeriodInsightAgent()

        mock_signal = MagicMock()
        mock_signal.source = "tavily_news"
        mock_signal.title = "K-Beauty dominates"
        mock_signal.url = "https://example.com/article"
        mock_signal.content = "K-Beauty brands see growth"
        mock_signal.published_at = "2026-02-10"
        mock_signal.metadata = {"domain": "allure.com"}

        agent._external_signals = {"signals": [mock_signal], "report_section": ""}
        agent._analysis_start_date = "2026-02-01"
        agent._analysis_end_date = "2026-02-15"

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Analysis of external signals"

        with patch(
            "src.agents.period_insight_agent.acompletion", new_callable=AsyncMock
        ) as mock_llm:
            mock_llm.return_value = mock_response
            analysis = _make_analysis()
            result = await agent._generate_external_signals(analysis)

        assert result.section_id == "5"
        assert result.data_highlights["total_signals"] == 1


# =========================================================================
# format_report_markdown
# =========================================================================


class TestFormatReportMarkdown:
    def test_empty_report(self):
        agent = PeriodInsightAgent()
        report = PeriodReport(
            start_date="2026-02-01",
            end_date="2026-02-15",
            generated_at="2026-02-15T12:00:00",
            metadata={"total_days": 14, "model": "gpt-4.1-mini", "has_external_signals": False},
        )
        md = agent.format_report_markdown(report)
        assert "라네즈 Amazon US 기간별 분석 보고서" in md
        assert "2026-02-01" in md
        assert "2026-02-15" in md
        assert "14" in md

    def test_with_sections(self):
        agent = PeriodInsightAgent()
        report = PeriodReport(
            start_date="2026-02-01",
            end_date="2026-02-15",
            generated_at="2026-02-15T12:00:00",
            executive_summary=SectionInsight(
                section_id="1",
                section_title="Executive Summary",
                content="LANEIGE performance overview",
            ),
            market_trends=SectionInsight(
                section_id="4",
                section_title="시장 동향",
                content="Market trends analysis",
            ),
            references=SectionInsight(
                section_id="8",
                section_title="참고자료",
                content="[1] Source 1\n[2] Source 2",
            ),
            metadata={"total_days": 14, "model": "gpt-4.1-mini", "has_external_signals": True},
        )
        md = agent.format_report_markdown(report)
        assert "Executive Summary" in md
        # format_insight replaces LANEIGE → **라네즈**
        assert "라네즈" in md
        assert "performance overview" in md
        assert "[1] Source 1" in md
        assert "외부 신호 포함: Yes" in md

    def test_references_section_no_extra_heading(self):
        """References section should include content directly (no extra heading)"""
        agent = PeriodInsightAgent()
        report = PeriodReport(
            start_date="2026-02-01",
            end_date="2026-02-15",
            generated_at="2026-02-15T12:00:00",
            references=SectionInsight(
                section_id="8",
                section_title="참고자료 (References)",
                content="[1] allure.com, 'K-Beauty Trends'",
            ),
            metadata={"total_days": 14, "model": "gpt-4.1-mini"},
        )
        md = agent.format_report_markdown(report)
        assert "[1] allure.com" in md


# =========================================================================
# generate_report (integration with mocked LLM)
# =========================================================================


class TestGenerateReport:
    @pytest.mark.asyncio
    async def test_generate_report_success(self):
        from unittest.mock import AsyncMock, patch

        agent = PeriodInsightAgent()
        analysis = _make_analysis()

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Generated insight content"

        with patch(
            "src.agents.period_insight_agent.acompletion", new_callable=AsyncMock
        ) as mock_llm:
            mock_llm.return_value = mock_response
            # Disable verification to simplify
            report = await agent.generate_report(analysis, verify_report=False)

        assert report.start_date == "2026-02-01"
        assert report.end_date == "2026-02-15"
        assert report.executive_summary is not None
        assert report.laneige_analysis is not None
        assert report.competitive_analysis is not None
        assert report.market_trends is not None
        assert report.external_signals is not None
        assert report.risks_opportunities is not None
        assert report.strategic_recommendations is not None
        assert report.references is not None
        assert report.metadata["total_days"] == 14

    @pytest.mark.asyncio
    async def test_generate_report_with_external_signals(self):
        from unittest.mock import AsyncMock, patch

        agent = PeriodInsightAgent()
        analysis = _make_analysis()

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Insight"

        with patch(
            "src.agents.period_insight_agent.acompletion", new_callable=AsyncMock
        ) as mock_llm:
            mock_llm.return_value = mock_response
            report = await agent.generate_report(
                analysis,
                external_signals={"signals": [], "report_section": ""},
                verify_report=False,
            )

        assert report.metadata["has_external_signals"] is True


# =========================================================================
# PeriodReport to_dict with all sections
# =========================================================================


class TestPeriodReportFullToDict:
    def test_all_sections(self):
        sections = {}
        for attr in [
            "executive_summary",
            "laneige_analysis",
            "competitive_analysis",
            "market_trends",
            "external_signals",
            "risks_opportunities",
            "strategic_recommendations",
        ]:
            sections[attr] = SectionInsight(
                section_id=str(hash(attr))[:3],
                section_title=attr,
                content=f"Content for {attr}",
            )

        report = PeriodReport(
            start_date="2026-02-01",
            end_date="2026-02-15",
            generated_at="2026-02-15T12:00:00",
            **sections,
            metadata={"total_days": 14},
        )
        d = report.to_dict()
        for attr in sections:
            assert d[attr] is not None
            assert "content" in d[attr]
