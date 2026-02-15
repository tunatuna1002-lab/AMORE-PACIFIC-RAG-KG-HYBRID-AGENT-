"""
Unit tests for PeriodInsightAgent
"""

from datetime import datetime

from src.agents.period_insight_agent import PeriodInsightAgent, PeriodReport, SectionInsight


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
