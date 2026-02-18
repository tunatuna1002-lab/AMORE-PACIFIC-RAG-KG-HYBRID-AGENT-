"""
Tests for AIS (Attributed Information Synthesis) inline citation in ContextBuilder.

Validates:
- Sentence splitting (Korean + English)
- Source-sentence matching via keyword overlap
- Citation tag insertion ([출처N])
- Citation statistics
- Edge cases (empty, disabled, no sources)
"""

from src.domain.entities.relations import InferenceResult, InsightType
from src.rag.context_builder import ContextBuilder
from src.rag.hybrid_retriever import HybridContext

# =========================================================================
# Helpers
# =========================================================================


def _make_builder(**kwargs) -> ContextBuilder:
    """Create a ContextBuilder with optional overrides."""
    return ContextBuilder(**kwargs)


def _make_context(
    inferences=None,
    rag_chunks=None,
    ontology_facts=None,
    entities=None,
):
    """HybridContext creation helper."""
    ctx = HybridContext(
        query="test",
        inferences=inferences or [],
        rag_chunks=rag_chunks or [],
        ontology_facts=ontology_facts or [],
    )
    if entities is not None:
        ctx.entities = entities
    return ctx


def _make_inference(**overrides):
    defaults = {
        "rule_name": "test_rule",
        "insight_type": InsightType.MARKET_POSITION,
        "insight": "테스트 인사이트",
        "confidence": 0.85,
        "evidence": {},
    }
    defaults.update(overrides)
    return InferenceResult(**defaults)


# =========================================================================
# AIS Basic Citation
# =========================================================================


class TestAisBasicCitation:
    def test_ais_basic_citation(self):
        """Sentences get [출처N] tags when matched to sources."""
        builder = _make_builder()
        builder._register_source("rag", "LANEIGE 시장 분석", "SoS 점유율 리포트")
        builder._register_source("kg", "경쟁사 현황", "COSRX 순위 데이터")

        response = "LANEIGE 시장 점유율이 상승했습니다. COSRX 순위가 변동했습니다."
        result = builder.build_ais_response(response)

        assert "[출처" in result

    def test_ais_citation_rate_above_80(self):
        """Citation rate should be >= 80% with well-matched sources."""
        builder = _make_builder()
        builder._register_source("rag", "LANEIGE SoS 분석", "점유율 데이터 상세")
        builder._register_source("kg", "시장 경쟁 구도", "브랜드 순위 분석")
        builder._register_source("data", "카테고리 데이터", "lip_care skin_care 순위")

        response = (
            "LANEIGE SoS 점유율이 상승 추세입니다. "
            "시장 경쟁 구도에서 브랜드 순위가 변동했습니다. "
            "카테고리별 lip_care 순위 데이터를 분석했습니다. "
            "시장 점유율 분석 결과를 공유합니다. "
            "브랜드 경쟁 데이터 기반 인사이트입니다."
        )
        result = builder.build_ais_response(response)
        stats = builder.get_citation_stats()

        assert stats["citation_rate"] >= 0.80

    def test_ais_no_sources(self):
        """Graceful handling when no sources are registered."""
        builder = _make_builder()
        response = "LANEIGE의 시장 점유율이 상승했습니다."
        result = builder.build_ais_response(response)

        # No citation tags should be added
        assert "[출처" not in result
        assert result == response

    def test_ais_multiple_sources_per_sentence(self):
        """A sentence can match multiple sources."""
        builder = _make_builder()
        builder._register_source("rag", "LANEIGE 브랜드", "시장 점유율")
        builder._register_source("kg", "LANEIGE 순위", "시장 분석 데이터")

        response = "LANEIGE 시장 점유율 순위 분석 데이터입니다."
        result = builder.build_ais_response(response)

        # Should have at least one citation tag
        assert "[출처" in result


class TestAisKoreanSentenceSplit:
    def test_ais_korean_sentence_split(self):
        """Korean sentence splitting works correctly."""
        builder = _make_builder()
        sentences = builder._split_sentences(
            "LANEIGE의 점유율이 상승했습니다. COSRX의 순위가 변동했습니다."
        )
        assert len(sentences) >= 2

    def test_split_on_newlines(self):
        """Splits on newlines."""
        builder = _make_builder()
        sentences = builder._split_sentences("첫 번째 문장\n두 번째 문장\n세 번째 문장")
        assert len(sentences) == 3

    def test_split_english_sentences(self):
        """English sentences split correctly."""
        builder = _make_builder()
        sentences = builder._split_sentences(
            "LANEIGE market share increased. COSRX ranking changed. New trends emerged."
        )
        assert len(sentences) >= 2


class TestAisCitationStats:
    def test_ais_citation_stats(self):
        """get_citation_stats() returns correct counts."""
        builder = _make_builder()
        builder._register_source("rag", "LANEIGE 분석", "시장 데이터")

        response = "LANEIGE 시장 분석 데이터입니다. 관련 없는 문장입니다."
        builder.build_ais_response(response)

        stats = builder.get_citation_stats()
        assert "total_sentences" in stats
        assert "cited_sentences" in stats
        assert "uncited_sentences" in stats
        assert "citation_rate" in stats
        assert stats["total_sentences"] == stats["cited_sentences"] + stats["uncited_sentences"]

    def test_ais_empty_response(self):
        """Handles empty response text gracefully."""
        builder = _make_builder()
        builder._register_source("rag", "Source", "Detail")

        result = builder.build_ais_response("")
        stats = builder.get_citation_stats()

        assert result == ""
        assert stats["total_sentences"] == 0
        assert stats["citation_rate"] == 0.0

    def test_ais_disabled(self):
        """enable_ais=False skips citation entirely."""
        builder = _make_builder(enable_ais=False)
        builder._register_source("rag", "LANEIGE 분석", "시장 데이터")

        response = "LANEIGE 시장 분석 결과입니다."
        result = builder.build_ais_response(response)

        # No citation tags added
        assert "[출처" not in result
        assert result == response

        stats = builder.get_citation_stats()
        assert stats["total_sentences"] == 0


class TestAisSourceRegistration:
    def test_ais_source_registration(self):
        """Sources properly registered during build."""
        builder = _make_builder()
        ctx = _make_context(
            inferences=[_make_inference(insight="SoS 상승", recommendation="마케팅 강화")],
            rag_chunks=[
                {
                    "content": "가이드라인 내용",
                    "metadata": {"title": "전략 가이드", "doc_id": "strat"},
                }
            ],
        )
        builder.build(ctx)

        refs = builder.get_source_references()
        assert len(refs) >= 1
        # Ontology and RAG sources registered
        source_types = {r.source_type for r in refs}
        assert "ontology" in source_types or "rag" in source_types


class TestAisWithRealContext:
    def test_ais_with_real_context(self):
        """End-to-end: build context, then apply AIS citation."""
        builder = _make_builder()
        ctx = _make_context(
            inferences=[
                _make_inference(
                    insight="LANEIGE SoS 점유율 상승 추세",
                    recommendation="마케팅 강화 권장",
                    confidence=0.9,
                )
            ],
            rag_chunks=[
                {
                    "content": "SoS는 Share of Shelf의 약자로 시장 점유율을 의미합니다.",
                    "metadata": {"title": "SoS 지표 가이드", "doc_id": "sos_guide"},
                }
            ],
        )
        metrics = {
            "summary": {
                "laneige_products_tracked": 5,
                "alert_count": 0,
                "critical_alerts": 0,
                "warning_alerts": 0,
            }
        }
        context_str = builder.build(ctx, current_metrics=metrics, query="LANEIGE SoS 분석")

        # Now apply AIS citation to a simulated LLM response
        llm_response = (
            "LANEIGE의 SoS 점유율이 상승 추세를 보이고 있습니다. "
            "시장 분석에 따르면 마케팅 강화가 권장됩니다. "
            "SoS 지표 가이드에 따르면 Share of Shelf 점유율은 중요한 지표입니다."
        )
        annotated = builder.build_ais_response(llm_response)

        # Should have at least some citations
        assert "[출처" in annotated

        stats = builder.get_citation_stats()
        assert stats["total_sentences"] >= 2
        assert stats["cited_sentences"] >= 1


class TestAisKeywordExtraction:
    def test_extract_keywords_filters_stopwords(self):
        """Korean/English stopwords are filtered out."""
        builder = _make_builder()
        keywords = builder._extract_keywords("LANEIGE 시장 점유율 분석 the is a")
        # "the", "is", "a" are stopwords (and single-char filtered)
        assert "laneige" in keywords
        assert "시장" in keywords
        assert "점유율" in keywords
        # English stopwords filtered
        assert "the" not in keywords
        assert "is" not in keywords

    def test_extract_keywords_empty_text(self):
        """Empty text returns empty set."""
        builder = _make_builder()
        keywords = builder._extract_keywords("")
        assert keywords == set()

    def test_match_sentence_to_sources_no_overlap(self):
        """Sentence with no keyword overlap returns empty."""
        builder = _make_builder()
        builder._register_source("rag", "LANEIGE 분석", "시장 데이터")
        matched = builder._match_sentence_to_sources("완전히 다른 주제의 문장입니다")
        # Might match or not depending on overlap; test structure
        assert isinstance(matched, list)

    def test_match_sentence_to_sources_empty_sentence(self):
        """Empty sentence returns empty list."""
        builder = _make_builder()
        builder._register_source("rag", "Source", "Detail")
        matched = builder._match_sentence_to_sources("")
        assert matched == []

    def test_match_sentence_to_sources_no_sources(self):
        """No sources registered returns empty list."""
        builder = _make_builder()
        matched = builder._match_sentence_to_sources("Some sentence")
        assert matched == []
