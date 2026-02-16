"""
Unit tests for src/tools/utilities/reference_tracker.py

Tests cover:
- ReferenceType enum
- Reference dataclass creation and citation formatting
- ReferenceTracker core operations (add, get, format)
- Convenience methods (add_data_source, add_article, add_report, add_social, add_academic)
- Serialization (to_dict / from_dict)
- Auto-add helpers (Amazon sources, IR reports)
- External signals integration
- Duplicate URL detection
- format_section output
"""

from unittest.mock import MagicMock

import pytest

from src.tools.utilities.reference_tracker import (
    Reference,
    ReferenceTracker,
    ReferenceType,
)


# ---------------------------------------------------------------------------
# ReferenceType enum
# ---------------------------------------------------------------------------
class TestReferenceType:
    """ReferenceType enum 테스트"""

    def test_enum_values(self):
        assert ReferenceType.DATA.value == "data"
        assert ReferenceType.ARTICLE.value == "article"
        assert ReferenceType.REPORT.value == "report"
        assert ReferenceType.SOCIAL.value == "social"
        assert ReferenceType.ACADEMIC.value == "academic"

    def test_enum_from_value(self):
        assert ReferenceType("data") == ReferenceType.DATA
        assert ReferenceType("social") == ReferenceType.SOCIAL


# ---------------------------------------------------------------------------
# Reference dataclass
# ---------------------------------------------------------------------------
class TestReference:
    """Reference dataclass 테스트"""

    def test_auto_id_generation(self):
        """ref_id가 비어있으면 자동 생성"""
        ref = Reference(ref_type=ReferenceType.DATA, title="Test", source="Source")
        assert ref.ref_id.startswith("D")

    def test_auto_id_prefix_per_type(self):
        """유형별 접두사 검증"""
        prefixes = {
            ReferenceType.DATA: "D",
            ReferenceType.ARTICLE: "A",
            ReferenceType.REPORT: "R",
            ReferenceType.SOCIAL: "S",
            ReferenceType.ACADEMIC: "P",
        }
        for rtype, expected_prefix in prefixes.items():
            ref = Reference(ref_type=rtype, title="T", source="S")
            assert ref.ref_id.startswith(expected_prefix)

    def test_explicit_ref_id_preserved(self):
        """명시적 ref_id가 유지됨"""
        ref = Reference(ref_type=ReferenceType.DATA, title="T", source="S", ref_id="CUSTOM1")
        assert ref.ref_id == "CUSTOM1"

    # ----- format_citation per type -----
    def test_format_citation_data(self):
        ref = Reference(
            ref_type=ReferenceType.DATA,
            title="Skin Care",
            source="Amazon Best Sellers",
            date="2026-01-14 ~ 2026-01-25",
            ref_id="D1",
        )
        citation = ref.format_citation()
        assert "[D1]" in citation
        assert "Amazon Best Sellers" in citation
        assert "Skin Care" in citation
        assert "2026-01-14 ~ 2026-01-25" in citation

    def test_format_citation_data_no_date(self):
        ref = Reference(
            ref_type=ReferenceType.DATA,
            title="Lip Care",
            source="Amazon",
            ref_id="D2",
        )
        citation = ref.format_citation()
        assert "[D2]" in citation
        assert "Amazon" in citation

    def test_format_citation_article(self):
        ref = Reference(
            ref_type=ReferenceType.ARTICLE,
            title="2026 Beauty Trends",
            source="Allure",
            date="2026-01-10",
            ref_id="A1",
        )
        citation = ref.format_citation()
        assert "[A1]" in citation
        assert '"2026 Beauty Trends"' in citation
        assert "Allure" in citation
        assert "2026-01-10" in citation

    def test_format_citation_report(self):
        ref = Reference(
            ref_type=ReferenceType.REPORT,
            title="Q4 Report",
            source="Goldman Sachs",
            date="2025-Q4",
            ref_id="R1",
        )
        citation = ref.format_citation()
        assert "[R1]" in citation
        assert '"Q4 Report"' in citation
        assert "Goldman Sachs" in citation

    def test_format_citation_social_with_subreddit(self):
        ref = Reference(
            ref_type=ReferenceType.SOCIAL,
            title="Best lip products",
            source="Reddit",
            date="2026-01-15",
            ref_id="S1",
            metadata={"subreddit": "SkincareAddiction", "upvotes": 2400},
        )
        citation = ref.format_citation()
        assert "[S1]" in citation
        assert "r/SkincareAddiction" in citation
        assert "2400 upvotes" in citation

    def test_format_citation_social_with_platform(self):
        ref = Reference(
            ref_type=ReferenceType.SOCIAL,
            title="Lip Mask Review",
            source="TikTok",
            ref_id="S2",
            metadata={"platform": "TikTok", "views": 50000},
        )
        citation = ref.format_citation()
        assert "[S2]" in citation
        assert "TikTok" in citation
        assert "50000 views" in citation

    def test_format_citation_social_no_metadata(self):
        ref = Reference(
            ref_type=ReferenceType.SOCIAL,
            title="Post",
            source="X",
            ref_id="S3",
            metadata={},
        )
        citation = ref.format_citation()
        assert "[S3]" in citation
        assert '"Post"' in citation

    def test_format_citation_academic_with_author(self):
        ref = Reference(
            ref_type=ReferenceType.ACADEMIC,
            title="Beauty Market Analysis",
            source="Journal of Cosmetics",
            date="2025-06",
            author="Kim et al.",
            ref_id="P1",
        )
        citation = ref.format_citation()
        assert "[P1]" in citation
        assert "Kim et al." in citation
        assert '"Beauty Market Analysis"' in citation
        assert "Journal of Cosmetics" in citation
        # Year only for academic
        assert "2025" in citation

    def test_format_citation_academic_no_author(self):
        ref = Reference(
            ref_type=ReferenceType.ACADEMIC,
            title="Paper",
            source="Journal",
            ref_id="P2",
        )
        citation = ref.format_citation()
        assert "[P2]" in citation
        assert '"Paper"' in citation


# ---------------------------------------------------------------------------
# ReferenceTracker - Core operations
# ---------------------------------------------------------------------------
class TestReferenceTrackerCore:
    """ReferenceTracker 핵심 기능 테스트"""

    @pytest.fixture
    def tracker(self):
        return ReferenceTracker()

    def test_initial_state(self, tracker):
        assert tracker.references == []
        assert all(v == 0 for v in tracker._id_counter.values())

    def test_add_reference(self, tracker):
        ref = tracker.add_reference(ReferenceType.DATA, title="Skin Care", source="Amazon")
        assert isinstance(ref, Reference)
        assert ref.ref_id == "D1"
        assert ref.title == "Skin Care"
        assert len(tracker.references) == 1

    def test_add_reference_increments_counter(self, tracker):
        ref1 = tracker.add_reference(ReferenceType.DATA, "T1", "S1")
        ref2 = tracker.add_reference(ReferenceType.DATA, "T2", "S2")
        assert ref1.ref_id == "D1"
        assert ref2.ref_id == "D2"

    def test_add_reference_counters_per_type(self, tracker):
        d1 = tracker.add_reference(ReferenceType.DATA, "T", "S")
        a1 = tracker.add_reference(ReferenceType.ARTICLE, "T", "S")
        d2 = tracker.add_reference(ReferenceType.DATA, "T", "S")
        assert d1.ref_id == "D1"
        assert a1.ref_id == "A1"
        assert d2.ref_id == "D2"

    def test_add_reference_with_metadata_kwargs(self, tracker):
        ref = tracker.add_reference(
            ReferenceType.SOCIAL,
            title="Post",
            source="Reddit",
            subreddit="SkincareAddiction",
            upvotes=100,
        )
        assert ref.metadata["subreddit"] == "SkincareAddiction"
        assert ref.metadata["upvotes"] == 100

    def test_get_by_type(self, tracker):
        tracker.add_reference(ReferenceType.DATA, "D1", "S")
        tracker.add_reference(ReferenceType.ARTICLE, "A1", "S")
        tracker.add_reference(ReferenceType.DATA, "D2", "S")

        data_refs = tracker.get_by_type(ReferenceType.DATA)
        assert len(data_refs) == 2
        article_refs = tracker.get_by_type(ReferenceType.ARTICLE)
        assert len(article_refs) == 1

    def test_get_by_type_empty(self, tracker):
        result = tracker.get_by_type(ReferenceType.ACADEMIC)
        assert result == []

    def test_get_by_id_found(self, tracker):
        ref = tracker.add_reference(ReferenceType.DATA, "Test", "Source")
        found = tracker.get_by_id(ref.ref_id)
        assert found is ref

    def test_get_by_id_not_found(self, tracker):
        assert tracker.get_by_id("NONEXISTENT") is None


# ---------------------------------------------------------------------------
# Convenience methods
# ---------------------------------------------------------------------------
class TestReferenceTrackerConvenience:
    """편의 메서드 테스트"""

    @pytest.fixture
    def tracker(self):
        return ReferenceTracker()

    def test_add_data_source(self, tracker):
        ref = tracker.add_data_source(
            title="Skin Care",
            source="Amazon Best Sellers",
            date_range="2026-01-14 ~ 2026-01-25",
        )
        assert ref.ref_type == ReferenceType.DATA
        assert ref.date == "2026-01-14 ~ 2026-01-25"

    def test_add_article(self, tracker):
        ref = tracker.add_article(
            title="Beauty Trends",
            source="Allure",
            date="2026-01-10",
            url="https://allure.com/trends",
        )
        assert ref.ref_type == ReferenceType.ARTICLE
        assert ref.url == "https://allure.com/trends"

    def test_add_report(self, tracker):
        ref = tracker.add_report(
            title="Q4 Earnings",
            source="JP Morgan",
            date="2025-Q4",
        )
        assert ref.ref_type == ReferenceType.REPORT

    def test_add_social(self, tracker):
        ref = tracker.add_social(
            title="Lip products",
            platform="Reddit",
            date="2026-01-15",
            subreddit="SkincareAddiction",
        )
        assert ref.ref_type == ReferenceType.SOCIAL
        assert ref.source == "Reddit"

    def test_add_academic(self, tracker):
        ref = tracker.add_academic(
            title="Market Paper",
            journal="J. Cosmetics",
            author="Kim et al.",
            year="2025",
        )
        assert ref.ref_type == ReferenceType.ACADEMIC
        assert ref.author == "Kim et al."
        assert ref.date == "2025"


# ---------------------------------------------------------------------------
# Formatted output
# ---------------------------------------------------------------------------
class TestReferenceTrackerFormatting:
    """형식화 출력 테스트"""

    @pytest.fixture
    def populated_tracker(self):
        t = ReferenceTracker()
        t.add_data_source("Skin Care", "Amazon", date_range="2026-01-01 ~ 2026-01-25")
        t.add_article("Trends", "Allure", "2026-01-10", url="https://allure.com")
        t.add_social("Lip post", "Reddit", "2026-01-15", subreddit="SCA", upvotes=100)
        t.add_report("Q4", "GS", date="2025-Q4")
        t.add_academic("Paper", "Journal", author="Lee", year="2025")
        return t

    def test_get_formatted_references_all(self, populated_tracker):
        text = populated_tracker.get_formatted_references("all")
        assert "[D1]" in text
        assert "[A1]" in text
        assert "[S1]" in text

    def test_get_formatted_references_external(self, populated_tracker):
        text = populated_tracker.get_formatted_references("external")
        # External = article + report + social + academic, NOT data
        assert "[A1]" in text
        assert "[S1]" in text
        assert "[R1]" in text
        assert "[P1]" in text
        assert "[D1]" not in text

    def test_get_formatted_references_data_only(self, populated_tracker):
        text = populated_tracker.get_formatted_references("data")
        assert "[D1]" in text
        assert "[A1]" not in text

    def test_get_formatted_references_empty(self):
        t = ReferenceTracker()
        assert t.get_formatted_references("all") == ""

    def test_get_formatted_references_includes_url(self, populated_tracker):
        text = populated_tracker.get_formatted_references("all")
        assert "URL: https://allure.com" in text

    def test_format_section_all_types(self, populated_tracker):
        section = populated_tracker.format_section()
        assert "8.1" in section
        assert "8.2" in section
        assert "8.3" in section
        assert "8.4" in section
        assert "8.5" in section

    def test_format_section_partial(self):
        t = ReferenceTracker()
        t.add_data_source("X", "Y")
        section = t.format_section()
        assert "8.1" in section
        assert "8.2" not in section

    def test_format_section_empty(self):
        t = ReferenceTracker()
        assert t.format_section() == ""

    def test_format_section_includes_urls(self):
        t = ReferenceTracker()
        t.add_article("Title", "Source", "2026-01", url="https://example.com")
        section = t.format_section()
        assert "https://example.com" in section


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------
class TestReferenceTrackerSerialization:
    """직렬화/역직렬화 테스트"""

    @pytest.fixture
    def populated_tracker(self):
        t = ReferenceTracker()
        t.add_data_source("Skin Care", "Amazon", date_range="2026-01")
        t.add_article("Trends", "Allure", "2026-01-10", url="https://allure.com")
        t.add_social("Post", "Reddit", subreddit="SCA")
        return t

    def test_to_dict_structure(self, populated_tracker):
        d = populated_tracker.to_dict()
        assert "references" in d
        assert len(d["references"]) == 3

    def test_to_dict_fields(self, populated_tracker):
        d = populated_tracker.to_dict()
        first = d["references"][0]
        assert "ref_id" in first
        assert "type" in first
        assert "title" in first
        assert "source" in first
        assert "date" in first
        assert "url" in first
        assert "author" in first
        assert "metadata" in first

    def test_from_dict_roundtrip(self, populated_tracker):
        d = populated_tracker.to_dict()
        restored = ReferenceTracker.from_dict(d)

        assert len(restored.references) == len(populated_tracker.references)
        for orig, rest in zip(populated_tracker.references, restored.references, strict=False):
            assert orig.title == rest.title
            assert orig.source == rest.source
            assert orig.ref_type == rest.ref_type

    def test_from_dict_empty(self):
        restored = ReferenceTracker.from_dict({"references": []})
        assert len(restored.references) == 0

    def test_from_dict_missing_key(self):
        restored = ReferenceTracker.from_dict({})
        assert len(restored.references) == 0


# ---------------------------------------------------------------------------
# Auto-add helpers
# ---------------------------------------------------------------------------
class TestAutoAddHelpers:
    """자동 추가 헬퍼 테스트"""

    def test_auto_add_amazon_sources_default(self):
        t = ReferenceTracker()
        t.auto_add_amazon_sources("2026-01-01", "2026-01-25")
        assert len(t.references) == 5
        titles = [r.title for r in t.references]
        assert "Beauty & Personal Care" in titles
        assert "Lip Care" in titles

    def test_auto_add_amazon_sources_custom_categories(self):
        t = ReferenceTracker()
        t.auto_add_amazon_sources("2026-01-01", "2026-01-25", categories=["Cat1", "Cat2"])
        assert len(t.references) == 2

    def test_auto_add_ir_report(self):
        t = ReferenceTracker()
        t.auto_add_ir_report("Q4", year="2025")
        assert len(t.references) == 1
        assert "Q4" in t.references[0].title
        assert "2025" in t.references[0].title


# ---------------------------------------------------------------------------
# External signals
# ---------------------------------------------------------------------------
class TestExternalSignals:
    """외부 신호 통합 테스트"""

    def _make_signal(self, **kwargs):
        """MagicMock 기반 signal 생성"""
        signal = MagicMock()
        signal.title = kwargs.get("title", "Signal Title")
        signal.url = kwargs.get("url", "")
        signal.tier = kwargs.get("tier", "unknown")
        signal.source = kwargs.get("source", "unknown")
        signal.published_at = kwargs.get("published_at", None)
        signal.metadata = kwargs.get("metadata", {})
        return signal

    def test_add_external_signals_article_tier(self):
        t = ReferenceTracker()
        sig = self._make_signal(tier="tier3_authority", source="news_site")
        count = t.add_external_signals([sig])
        assert count == 1
        assert t.references[0].ref_type == ReferenceType.ARTICLE

    def test_add_external_signals_social_tier(self):
        t = ReferenceTracker()
        sig = self._make_signal(tier="tier2_validation", source="reddit_post")
        count = t.add_external_signals([sig])
        assert count == 1
        assert t.references[0].ref_type == ReferenceType.SOCIAL

    def test_add_external_signals_source_fallback(self):
        t = ReferenceTracker()
        sig = self._make_signal(tier="unknown_tier", source="youtube_video")
        count = t.add_external_signals([sig])
        assert count == 1
        assert t.references[0].ref_type == ReferenceType.SOCIAL

    def test_add_external_signals_default_article(self):
        t = ReferenceTracker()
        sig = self._make_signal(tier="unknown", source="something_new")
        count = t.add_external_signals([sig])
        assert count == 1
        assert t.references[0].ref_type == ReferenceType.ARTICLE

    def test_add_external_signals_duplicate_url_skipped(self):
        t = ReferenceTracker()
        t.add_article("Existing", "Source", "2026-01", url="https://dup.com")

        sig = self._make_signal(url="https://dup.com")
        count = t.add_external_signals([sig])
        assert count == 0
        assert len(t.references) == 1

    def test_add_external_signals_empty_url_not_duplicate(self):
        t = ReferenceTracker()
        sig = self._make_signal(url="")
        count = t.add_external_signals([sig])
        assert count == 1

    def test_add_external_signals_reddit_metadata(self):
        t = ReferenceTracker()
        sig = self._make_signal(
            tier="tier2_validation",
            source="reddit",
            metadata={"subreddit": "SCA", "reliability_score": 0.8},
        )
        count = t.add_external_signals([sig])
        assert count == 1
        ref = t.references[0]
        assert ref.metadata.get("subreddit") == "SCA"
        assert ref.metadata.get("platform") == "Reddit"
        assert ref.metadata.get("reliability_score") == 0.8

    def test_add_external_signals_youtube_metadata(self):
        t = ReferenceTracker()
        sig = self._make_signal(
            tier="tier2_validation",
            source="youtube",
            metadata={"views": 50000},
        )
        count = t.add_external_signals([sig])
        assert count == 1
        assert t.references[0].metadata.get("platform") == "YouTube"
        assert t.references[0].metadata.get("views") == 50000

    def test_add_external_signals_tiktok_metadata(self):
        t = ReferenceTracker()
        sig = self._make_signal(
            tier="tier1_viral",
            source="tiktok",
            metadata={"views": 100000},
        )
        count = t.add_external_signals([sig])
        assert count == 1
        assert t.references[0].metadata.get("platform") == "TikTok"

    def test_add_external_signals_multiple(self):
        t = ReferenceTracker()
        signals = [
            self._make_signal(tier="tier3_authority", source="news", url="https://a.com"),
            self._make_signal(tier="tier2_validation", source="reddit", url="https://b.com"),
            self._make_signal(tier="tier1_viral", source="tiktok", url="https://c.com"),
        ]
        count = t.add_external_signals(signals)
        assert count == 3
        assert len(t.references) == 3


# ---------------------------------------------------------------------------
# Duplicate detection
# ---------------------------------------------------------------------------
class TestDuplicateDetection:
    """URL 중복 감지 테스트"""

    def test_is_duplicate_true(self):
        t = ReferenceTracker()
        t.add_article("T", "S", "2026-01", url="https://test.com")
        assert t._is_duplicate("https://test.com") is True

    def test_is_duplicate_false(self):
        t = ReferenceTracker()
        t.add_article("T", "S", "2026-01", url="https://test.com")
        assert t._is_duplicate("https://other.com") is False

    def test_is_duplicate_empty_url(self):
        t = ReferenceTracker()
        assert t._is_duplicate("") is False

    def test_is_duplicate_none_url(self):
        t = ReferenceTracker()
        # None should also return False per empty check
        assert t._is_duplicate(None) is False
