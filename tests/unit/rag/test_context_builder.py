"""
ContextBuilder 단위 테스트
=========================
LLM 프롬프트 조립, 토큰 예산 관리, 섹션 우선순위 검증
"""

from src.rag.context_builder import (
    CompactContextBuilder,
    ContextBuilder,
    ContextPriority,
    ContextSection,
    OutputFormat,
)

# ---------------------------------------------------------------------------
# Enum 테스트
# ---------------------------------------------------------------------------


class TestEnums:
    """Enum 값 검증"""

    def test_output_format_values(self):
        assert OutputFormat.MARKDOWN == "markdown"
        assert OutputFormat.PLAIN == "plain"
        assert OutputFormat.STRUCTURED == "structured"

    def test_context_priority_values(self):
        assert ContextPriority.CRITICAL == "critical"
        assert ContextPriority.HIGH == "high"
        assert ContextPriority.MEDIUM == "medium"
        assert ContextPriority.LOW == "low"


# ---------------------------------------------------------------------------
# ContextSection dataclass
# ---------------------------------------------------------------------------


class TestContextSection:
    """ContextSection 데이터 구조"""

    def test_create_section(self):
        section = ContextSection(
            title="테스트 섹션",
            content="내용입니다.",
            priority=ContextPriority.HIGH,
        )
        assert section.title == "테스트 섹션"
        assert section.priority == ContextPriority.HIGH

    def test_token_estimate_auto(self):
        """token_estimate가 0이면 자동 계산"""
        section = ContextSection(
            title="test",
            content="a" * 100,
            priority=ContextPriority.MEDIUM,
        )
        assert section.token_estimate == 25  # 100 / 4

    def test_token_estimate_explicit(self):
        section = ContextSection(
            title="test",
            content="content",
            priority=ContextPriority.LOW,
            token_estimate=50,
        )
        assert section.token_estimate == 50

    def test_default_source(self):
        section = ContextSection(
            title="test",
            content="content",
            priority=ContextPriority.LOW,
        )
        assert section.source == "unknown"


# ---------------------------------------------------------------------------
# ContextBuilder
# ---------------------------------------------------------------------------


class TestContextBuilder:
    """ContextBuilder 테스트"""

    def test_init(self):
        builder = ContextBuilder()
        assert builder is not None

    def test_init_with_max_tokens(self):
        builder = ContextBuilder(max_tokens=2000)
        assert builder.max_tokens == 2000

    def test_build_returns_string(self):
        builder = ContextBuilder()
        context = {
            "query": "LANEIGE SoS 분석",
            "entities": {"brands": ["LANEIGE"], "categories": [], "indicators": ["sos"]},
            "ontology_facts": [],
            "inferences": [],
            "rag_chunks": [],
            "combined_context": "",
            "metadata": {},
        }
        result = builder.build(context)
        assert isinstance(result, str)

    def test_build_with_rag_chunks(self):
        builder = ContextBuilder()
        context = {
            "query": "SoS란?",
            "entities": {"brands": [], "categories": [], "indicators": ["sos"]},
            "ontology_facts": [],
            "inferences": [],
            "rag_chunks": [
                {
                    "content": "SoS는 Share of Shelf의 약자입니다.",
                    "metadata": {"title": "지표 정의"},
                },
            ],
            "combined_context": "",
            "metadata": {},
        }
        result = builder.build(context)
        assert isinstance(result, str)

    def test_build_with_inferences(self):
        builder = ContextBuilder()
        context = {
            "query": "LANEIGE 분석",
            "entities": {"brands": ["LANEIGE"], "categories": [], "indicators": []},
            "ontology_facts": ["LANEIGE is a premium brand"],
            "inferences": [{"type": "market_position", "brand": "LANEIGE", "position": "leader"}],
            "rag_chunks": [],
            "combined_context": "",
            "metadata": {},
        }
        result = builder.build(context)
        assert isinstance(result, str)

    def test_build_empty_context(self):
        builder = ContextBuilder()
        context = {
            "query": "",
            "entities": {"brands": [], "categories": [], "indicators": []},
            "ontology_facts": [],
            "inferences": [],
            "rag_chunks": [],
            "combined_context": "",
            "metadata": {},
        }
        result = builder.build(context)
        assert isinstance(result, str)

    def test_build_system_prompt(self):
        builder = ContextBuilder()
        if hasattr(builder, "build_system_prompt"):
            prompt = builder.build_system_prompt()
            assert isinstance(prompt, str)
            assert len(prompt) > 0

    def test_build_user_prompt(self):
        builder = ContextBuilder()
        if hasattr(builder, "build_user_prompt"):
            prompt = builder.build_user_prompt(
                query="LANEIGE 분석",
                context="LANEIGE는 프리미엄 브랜드입니다.",
            )
            assert isinstance(prompt, str)
            assert len(prompt) > 0


# ---------------------------------------------------------------------------
# CompactContextBuilder
# ---------------------------------------------------------------------------


class TestCompactContextBuilder:
    """CompactContextBuilder 테스트"""

    def test_init(self):
        builder = CompactContextBuilder()
        assert builder is not None

    def test_compact_build(self):
        builder = CompactContextBuilder()
        context = {
            "query": "LANEIGE 분석",
            "entities": {"brands": ["LANEIGE"], "categories": [], "indicators": []},
            "ontology_facts": [],
            "inferences": [],
            "rag_chunks": [
                {
                    "content": "LANEIGE는 프리미엄 브랜드입니다.",
                    "metadata": {"title": "브랜드 분석"},
                },
            ],
            "combined_context": "",
            "metadata": {},
        }
        result = builder.build(context)
        assert isinstance(result, str)

    def test_compact_shorter_than_full(self):
        """CompactContextBuilder 결과가 ContextBuilder보다 짧거나 같아야 함"""
        full_builder = ContextBuilder()
        compact_builder = CompactContextBuilder()

        big_chunks = [
            {"content": f"Document {i} content. " * 50, "metadata": {"title": f"Doc {i}"}}
            for i in range(10)
        ]
        context = {
            "query": "LANEIGE 분석",
            "entities": {
                "brands": ["LANEIGE"],
                "categories": ["lip_care"],
                "indicators": ["sos", "hhi"],
            },
            "ontology_facts": ["Fact 1", "Fact 2", "Fact 3"],
            "inferences": [
                {"type": "market_position", "brand": "LANEIGE", "position": "leader"},
                {"type": "competition", "brand": "LANEIGE", "competitors": ["COSRX", "TIRTIR"]},
            ],
            "rag_chunks": big_chunks,
            "combined_context": "",
            "metadata": {},
        }

        full_result = full_builder.build(context)
        compact_result = compact_builder.build(context)

        assert isinstance(full_result, str)
        assert isinstance(compact_result, str)
        # compact은 같거나 짧아야 함 (약간의 여유)
        assert len(compact_result) <= len(full_result) + 100
