"""
Tests for SourceManager

출처 관리 시스템 테스트
"""

import pytest

from src.tools.intelligence.source_manager import (
    PUBLISHER_RELIABILITY,
    RELIABILITY_SCORES,
    InsightSourceBuilder,
    Source,
    SourceManager,
    SourceType,
    create_insight_builder,
    create_source_manager,
)


class TestSource:
    """Source 데이터클래스 테스트"""

    def test_create_source(self):
        """Source 생성 테스트"""
        source = Source(
            source_id="SRC-001",
            citation_number=1,
            title="3Q 2025 Earnings Release",
            publisher="아모레퍼시픽 IR",
            date="2025-11-06",
            url="https://www.apgroup.com/...",
            source_type="ir",
        )

        assert source.source_id == "SRC-001"
        assert source.citation_number == 1
        assert source.title == "3Q 2025 Earnings Release"
        assert source.reliability_score == 1.0  # IR은 자동으로 1.0

    def test_source_auto_reliability_by_publisher(self):
        """매체별 자동 신뢰도 설정 테스트"""
        source = Source(
            source_id="SRC-002",
            citation_number=2,
            title="수출입통계",
            publisher="관세청",
            date="2025-01",
            source_type="government",
        )

        assert source.reliability_score == 0.95

    def test_source_auto_reliability_by_type(self):
        """유형별 자동 신뢰도 설정 테스트"""
        source = Source(
            source_id="SRC-003",
            citation_number=3,
            title="Test News",
            publisher="Unknown Media",
            date="2025-01-26",
            source_type="news",
        )

        assert source.reliability_score == 0.7

    def test_source_to_citation(self):
        """인용 표시 생성 테스트"""
        source = Source(
            source_id="SRC-001",
            citation_number=5,
            title="Test",
            publisher="Test",
            date="2025-01-01",
        )

        assert source.to_citation() == "[5]"

    def test_source_to_reference_line_with_url(self):
        """참고자료 라인 생성 - URL 포함"""
        source = Source(
            source_id="SRC-001",
            citation_number=1,
            title="3Q 2025 Earnings Release",
            publisher="아모레퍼시픽 IR",
            date="2025-11-06",
            url="https://www.apgroup.com/...",
        )

        line = source.to_reference_line(include_url=True)

        assert "[1]" in line
        assert "아모레퍼시픽 IR" in line
        assert "3Q 2025 Earnings Release" in line
        assert "2025.11.06" in line
        assert "https://www.apgroup.com/" in line

    def test_source_to_reference_line_without_url(self):
        """참고자료 라인 생성 - URL 제외"""
        source = Source(
            source_id="SRC-001",
            citation_number=1,
            title="Test",
            publisher="Test",
            date="2025-01-01",
            url="https://example.com",
        )

        line = source.to_reference_line(include_url=False)

        assert "https://example.com" not in line

    def test_source_to_inline_reference(self):
        """인라인 참조 생성 테스트"""
        source = Source(
            source_id="SRC-001",
            citation_number=1,
            title="Test",
            publisher="관세청",
            date="2025-01-15",
        )

        inline = source.to_inline_reference()

        assert inline == "(관세청, 2025.01)"


class TestSourceManager:
    """SourceManager 테스트"""

    @pytest.fixture
    def manager(self, tmp_path):
        """테스트용 manager 인스턴스"""
        return SourceManager(data_dir=str(tmp_path / "sources"))

    def test_manager_initialization(self, manager):
        """초기화 테스트"""
        assert len(manager.sources) == 0
        assert manager._citation_counter == 0

    def test_add_source(self, manager):
        """출처 추가 테스트"""
        source = manager.add_source(
            title="Test Report",
            publisher="Test Publisher",
            date="2025-01-26",
            url="https://example.com",
            source_type="news",
        )

        assert source.citation_number == 1
        assert source.source_id in manager.sources
        assert len(manager.sources) == 1

    def test_add_multiple_sources(self, manager):
        """여러 출처 추가 테스트"""
        source1 = manager.add_source(title="Report 1", publisher="Publisher A", date="2025-01-01")

        source2 = manager.add_source(title="Report 2", publisher="Publisher B", date="2025-01-02")

        assert source1.citation_number == 1
        assert source2.citation_number == 2
        assert len(manager.sources) == 2

    def test_add_duplicate_source(self, manager):
        """중복 출처 추가 테스트"""
        source1 = manager.add_source(
            title="Same Report", publisher="Same Publisher", date="2025-01-01"
        )

        # 동일한 정보로 다시 추가
        source2 = manager.add_source(
            title="Same Report", publisher="Same Publisher", date="2025-01-01"
        )

        # 같은 출처가 반환되어야 함
        assert source1.source_id == source2.source_id
        assert len(manager.sources) == 1

    def test_add_source_from_dict(self, manager):
        """딕셔너리에서 출처 추가 테스트"""
        data = {
            "title": "Dict Report",
            "publisher": "Dict Publisher",
            "date": "2025-01-26",
            "url": "https://example.com",
            "source_type": "research",
        }

        source = manager.add_source_from_dict(data)

        assert source.title == "Dict Report"
        assert source.publisher == "Dict Publisher"
        assert source.source_type == "research"

    def test_cite(self, manager):
        """인용 생성 테스트"""
        source = manager.add_source(title="Test", publisher="Test", date="2025-01-01")

        citation = manager.cite(source)

        assert citation == "[1]"

    def test_cite_by_id(self, manager):
        """ID로 인용 생성 테스트"""
        source = manager.add_source(title="Test", publisher="Test", date="2025-01-01")

        citation = manager.cite_by_id(source.source_id)

        assert citation == "[1]"

    def test_cite_by_id_not_found(self, manager):
        """없는 ID로 인용 생성 테스트"""
        citation = manager.cite_by_id("nonexistent")

        assert citation == ""

    def test_get_source(self, manager):
        """출처 조회 테스트"""
        source = manager.add_source(title="Test", publisher="Test", date="2025-01-01")

        retrieved = manager.get_source(source.source_id)

        assert retrieved == source

    def test_get_source_by_citation(self, manager):
        """인용 번호로 출처 조회 테스트"""
        manager.add_source(title="First", publisher="P1", date="2025-01-01")
        source2 = manager.add_source(title="Second", publisher="P2", date="2025-01-02")
        manager.add_source(title="Third", publisher="P3", date="2025-01-03")

        retrieved = manager.get_source_by_citation(2)

        assert retrieved.title == "Second"

    def test_get_all_sources_by_citation(self, manager):
        """모든 출처 조회 - 인용 번호 순"""
        manager.add_source(title="Third", publisher="P3", date="2025-01-03")
        manager.add_source(title="First", publisher="P1", date="2025-01-01")
        manager.add_source(title="Second", publisher="P2", date="2025-01-02")

        sources = manager.get_all_sources(sort_by="citation_number")

        assert sources[0].citation_number == 1
        assert sources[1].citation_number == 2
        assert sources[2].citation_number == 3

    def test_get_all_sources_by_reliability(self, manager):
        """모든 출처 조회 - 신뢰도 순"""
        manager.add_source(
            title="News",
            publisher="Unknown",
            date="2025-01-01",
            source_type="news",
            reliability_score=0.7,
        )
        manager.add_source(
            title="IR",
            publisher="아모레퍼시픽 IR",
            date="2025-01-01",
            source_type="ir",
            reliability_score=1.0,
        )

        sources = manager.get_all_sources(sort_by="reliability")

        assert sources[0].reliability_score == 1.0
        assert sources[1].reliability_score == 0.7

    def test_generate_references_section(self, manager):
        """참고자료 섹션 생성 테스트"""
        manager.add_source(
            title="Report A", publisher="Publisher A", date="2025-01-01", url="https://a.com"
        )
        manager.add_source(title="Report B", publisher="Publisher B", date="2025-01-02")

        section = manager.generate_references_section()

        assert "## 참고자료" in section
        assert "[1]" in section
        assert "[2]" in section
        assert "Publisher A" in section
        assert "Publisher B" in section

    def test_generate_references_section_empty(self, manager):
        """참고자료 섹션 생성 - 빈 경우"""
        section = manager.generate_references_section()

        assert section == ""

    def test_generate_compact_references(self, manager):
        """간략한 참고자료 생성 테스트"""
        manager.add_source(title="Test", publisher="관세청", date="2025-01-15")

        compact = manager.generate_compact_references()

        assert "참고자료" in compact
        assert "[1]" in compact
        assert "관세청" in compact
        assert "2025.01" in compact

    def test_clear(self, manager):
        """초기화 테스트"""
        manager.add_source(title="Test", publisher="Test", date="2025-01-01")
        manager.add_source(title="Test2", publisher="Test2", date="2025-01-02")

        manager.clear()

        assert len(manager.sources) == 0
        assert manager._citation_counter == 0

    def test_get_stats(self, manager):
        """통계 반환 테스트"""
        manager.add_source(title="News 1", publisher="A", date="2025-01-01", source_type="news")
        manager.add_source(title="IR 1", publisher="B", date="2025-01-02", source_type="ir")
        manager.add_source(title="News 2", publisher="C", date="2025-01-03", source_type="news")

        stats = manager.get_stats()

        assert stats["total_sources"] == 3
        assert stats["by_type"]["news"] == 2
        assert stats["by_type"]["ir"] == 1


class TestInsightSourceBuilder:
    """InsightSourceBuilder 테스트"""

    def test_builder_add_text(self):
        """텍스트 추가 테스트"""
        builder = InsightSourceBuilder()

        builder.add_text("첫 번째 문장입니다.")
        builder.add_text("두 번째 문장입니다.")

        text = builder.build(include_references=False)

        assert "첫 번째 문장입니다." in text
        assert "두 번째 문장입니다." in text

    def test_builder_add_cited_text(self):
        """인용과 함께 텍스트 추가 테스트"""
        builder = InsightSourceBuilder()

        builder.add_cited_text(
            "아모레퍼시픽 3Q 2025 매출이 +4.1% 성장했습니다",
            {
                "title": "3Q 2025 Earnings Release",
                "publisher": "아모레퍼시픽 IR",
                "date": "2025-11-06",
                "source_type": "ir",
            },
        )

        text = builder.build(include_references=False)

        assert "아모레퍼시픽 3Q 2025 매출이" in text
        assert "[1]" in text

    def test_builder_build_with_references(self):
        """참고자료 포함 빌드 테스트"""
        builder = InsightSourceBuilder()

        builder.add_cited_text(
            "테스트 인사이트",
            {"title": "Test Report", "publisher": "Test Publisher", "date": "2025-01-26"},
        )

        text = builder.build(include_references=True)

        assert "테스트 인사이트" in text
        assert "[1]" in text
        assert "## 참고자료" in text
        assert "Test Publisher" in text

    def test_builder_chaining(self):
        """체이닝 테스트"""
        builder = InsightSourceBuilder()

        result = (
            builder.add_text("첫 번째")
            .add_cited_text("두 번째", {"title": "T", "publisher": "P", "date": "2025-01-01"})
            .add_text("세 번째")
        )

        assert result is builder
        text = builder.build(include_references=False)
        assert "첫 번째" in text
        assert "두 번째" in text
        assert "세 번째" in text

    def test_builder_reset(self):
        """빌더 초기화 테스트"""
        builder = InsightSourceBuilder()

        builder.add_text("텍스트")
        builder.add_cited_text("인용", {"title": "T", "publisher": "P", "date": "2025-01-01"})

        builder.reset()

        text = builder.build(include_references=False)
        assert text == ""


class TestConstants:
    """상수 테스트"""

    def test_reliability_scores(self):
        """신뢰도 점수 상수 테스트"""
        assert RELIABILITY_SCORES["ir"] == 1.0
        assert RELIABILITY_SCORES["government"] == 0.95
        assert RELIABILITY_SCORES["research"] == 0.9
        assert RELIABILITY_SCORES["news"] == 0.7
        assert RELIABILITY_SCORES["sns"] == 0.5

    def test_publisher_reliability(self):
        """매체별 신뢰도 상수 테스트"""
        assert PUBLISHER_RELIABILITY["아모레퍼시픽 IR"] == 1.0
        assert PUBLISHER_RELIABILITY["관세청"] == 0.95
        assert PUBLISHER_RELIABILITY["KCII"] == 0.9
        assert PUBLISHER_RELIABILITY["WWD"] == 0.85
        assert PUBLISHER_RELIABILITY["Reddit"] == 0.5

    def test_source_type_enum(self):
        """SourceType Enum 테스트"""
        assert SourceType.IR.value == "ir"
        assert SourceType.GOVERNMENT.value == "government"
        assert SourceType.RESEARCH.value == "research"
        assert SourceType.ANALYST.value == "analyst"
        assert SourceType.NEWS.value == "news"
        assert SourceType.SNS.value == "sns"


class TestFactoryFunctions:
    """팩토리 함수 테스트"""

    def test_create_source_manager(self):
        """SourceManager 생성 함수 테스트"""
        manager = create_source_manager()

        assert isinstance(manager, SourceManager)

    def test_create_insight_builder(self):
        """InsightSourceBuilder 생성 함수 테스트"""
        builder = create_insight_builder()

        assert isinstance(builder, InsightSourceBuilder)
