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


# =========================================================================
# 아래: HybridContext 기반 커버리지 확장 테스트
# =========================================================================

from unittest.mock import MagicMock

from src.domain.entities.relations import InferenceResult, InsightType
from src.rag.hybrid_retriever import HybridContext


def _make_context(
    inferences=None,
    rag_chunks=None,
    ontology_facts=None,
    entities=None,
):
    """HybridContext 생성 헬퍼"""
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


# ---------------------------------------------------------------------------
# _register_source / get_source_references / build_source_appendix
# ---------------------------------------------------------------------------


class TestSourceManagement:
    """출처 관리 테스트"""

    def test_register_source_returns_incrementing_index(self):
        builder = ContextBuilder()
        idx1 = builder._register_source("rag", "Doc A", "detail A")
        idx2 = builder._register_source("kg", "Doc B", "detail B")
        assert idx1 == 1
        assert idx2 == 2

    def test_get_source_references_returns_copy(self):
        builder = ContextBuilder()
        builder._register_source("rag", "Doc", "detail")
        refs = builder.get_source_references()
        assert len(refs) == 1
        assert refs[0].source_type == "rag"
        # 복사본이므로 원본 영향 없음
        refs.clear()
        assert len(builder.get_source_references()) == 1

    def test_build_source_appendix_empty(self):
        builder = ContextBuilder()
        assert builder.build_source_appendix() == ""

    def test_build_source_appendix_with_sources(self):
        builder = ContextBuilder()
        builder._register_source("rag", "Guide", "doc_id:1")
        builder._register_source("ontology", "Reasoning", "conf:85%")
        appendix = builder.build_source_appendix()
        assert "## 출처" in appendix
        assert "[1] [rag] Guide: doc_id:1" in appendix
        assert "[2] [ontology] Reasoning: conf:85%" in appendix


# ---------------------------------------------------------------------------
# _build_inference_section
# ---------------------------------------------------------------------------


class TestBuildInferenceSection:
    """온톨로지 추론 섹션 테스트"""

    def test_basic_inference(self):
        builder = ContextBuilder()
        inf = _make_inference(
            insight="LANEIGE SoS 상승",
            confidence=0.9,
            recommendation="마케팅 강화 권장",
        )
        section = builder._build_inference_section([inf])
        assert section.priority == ContextPriority.CRITICAL
        assert "LANEIGE SoS 상승" in section.content
        assert "마케팅 강화 권장" in section.content
        assert "90%" in section.content

    def test_inference_with_evidence_conditions(self):
        builder = ContextBuilder()
        inf = _make_inference(
            evidence={"satisfied_conditions": ["SoS > 10%", "Rank < 5"]},
        )
        section = builder._build_inference_section([inf])
        assert "SoS > 10%" in section.content
        assert "Rank < 5" in section.content

    def test_inference_without_recommendation(self):
        builder = ContextBuilder()
        inf = _make_inference(recommendation=None)
        section = builder._build_inference_section([inf])
        assert "권장 액션" not in section.content

    def test_multiple_inferences(self):
        builder = ContextBuilder()
        infs = [
            _make_inference(insight="Insight 1"),
            _make_inference(insight="Insight 2"),
        ]
        section = builder._build_inference_section(infs)
        assert "인사이트 1" in section.content
        assert "인사이트 2" in section.content
        # 출처 등록 확인
        assert len(builder.get_source_references()) == 2


# ---------------------------------------------------------------------------
# _build_data_section
# ---------------------------------------------------------------------------


class TestBuildDataSection:
    """현재 데이터 섹션 테스트"""

    def test_basic_summary(self):
        builder = ContextBuilder()
        metrics = {
            "summary": {
                "laneige_products_tracked": 5,
                "alert_count": 3,
                "critical_alerts": 1,
                "warning_alerts": 2,
            }
        }
        section = builder._build_data_section(metrics, {})
        assert "5개" in section.content
        assert "3건" in section.content
        assert "Critical: 1" in section.content

    def test_sos_data(self):
        builder = ContextBuilder()
        metrics = {
            "summary": {
                "laneige_products_tracked": 0,
                "alert_count": 0,
                "critical_alerts": 0,
                "warning_alerts": 0,
                "laneige_sos_by_category": {"lip_care": 0.15, "skin_care": 0.08},
            }
        }
        section = builder._build_data_section(metrics, {})
        assert "15.0%" in section.content
        assert "8.0%" in section.content

    def test_best_product(self):
        builder = ContextBuilder()
        metrics = {
            "summary": {
                "laneige_products_tracked": 0,
                "alert_count": 0,
                "critical_alerts": 0,
                "warning_alerts": 0,
                "best_ranking_product": {
                    "title": "LANEIGE Lip Sleeping Mask Berry",
                    "rank": 3,
                    "category": "lip_care",
                },
            }
        }
        section = builder._build_data_section(metrics, {})
        assert "3위" in section.content
        assert "lip_care" in section.content

    def test_brand_metrics_detail(self):
        builder = ContextBuilder()
        metrics = {
            "summary": {
                "laneige_products_tracked": 0,
                "alert_count": 0,
                "critical_alerts": 0,
                "warning_alerts": 0,
            },
            "brand_metrics": [
                {
                    "brand_name": "LANEIGE",
                    "category_id": "lip_care",
                    "share_of_shelf": 0.12,
                    "avg_rank": 15.5,
                    "product_count": 3,
                    "top10_count": 2,
                }
            ],
        }
        entities = {"brands": ["LANEIGE"], "categories": []}
        section = builder._build_data_section(metrics, entities)
        assert "LANEIGE" in section.content
        assert "12.0%" in section.content
        assert "15.5" in section.content
        assert "Top 10: 2개" in section.content

    def test_market_metrics_detail(self):
        builder = ContextBuilder()
        metrics = {
            "summary": {
                "laneige_products_tracked": 0,
                "alert_count": 0,
                "critical_alerts": 0,
                "warning_alerts": 0,
            },
            "market_metrics": [{"category_id": "lip_care", "hhi": 0.045, "cpi": 88.5}],
        }
        entities = {"brands": [], "categories": ["lip_care"]}
        section = builder._build_data_section(metrics, entities)
        assert "0.045" in section.content
        assert "88.5" in section.content

    def test_brand_filtered_by_category(self):
        """특정 카테고리 필터링"""
        builder = ContextBuilder()
        metrics = {
            "summary": {
                "laneige_products_tracked": 0,
                "alert_count": 0,
                "critical_alerts": 0,
                "warning_alerts": 0,
            },
            "brand_metrics": [
                {
                    "brand_name": "LANEIGE",
                    "category_id": "lip_care",
                    "share_of_shelf": 0.12,
                    "product_count": 3,
                },
                {
                    "brand_name": "LANEIGE",
                    "category_id": "skin_care",
                    "share_of_shelf": 0.05,
                    "product_count": 1,
                },
            ],
        }
        entities = {"brands": ["LANEIGE"], "categories": ["lip_care"]}
        section = builder._build_data_section(metrics, entities)
        assert "12.0%" in section.content
        # skin_care는 필터 밖
        assert "5.0%" not in section.content


# ---------------------------------------------------------------------------
# _build_facts_section
# ---------------------------------------------------------------------------


class TestBuildFactsSection:
    """KG 사실 섹션 테스트"""

    def test_brand_info_fact(self):
        builder = ContextBuilder()
        facts = [
            {
                "type": "brand_info",
                "entity": "LANEIGE",
                "data": {"sos": 0.12, "avg_rank": 15.5, "product_count": 3},
            }
        ]
        section = builder._build_facts_section(facts)
        assert "LANEIGE" in section.content
        assert "12.0%" in section.content
        assert "15.5" in section.content
        assert "3개" in section.content

    def test_brand_products_fact(self):
        builder = ContextBuilder()
        facts = [
            {
                "type": "brand_products",
                "entity": "COSRX",
                "data": {"product_count": 7},
            }
        ]
        section = builder._build_facts_section(facts)
        assert "COSRX" in section.content
        assert "7개" in section.content

    def test_competitors_fact(self):
        builder = ContextBuilder()
        facts = [
            {
                "type": "competitors",
                "entity": "LANEIGE",
                "data": [
                    {"brand": "COSRX"},
                    {"brand": "TIRTIR"},
                ],
            }
        ]
        section = builder._build_facts_section(facts)
        assert "경쟁사" in section.content
        assert "COSRX" in section.content

    def test_category_brands_fact(self):
        builder = ContextBuilder()
        facts = [
            {
                "type": "category_brands",
                "entity": "lip_care",
                "data": {
                    "top_brands": [
                        {"brand": "LANEIGE"},
                        {"brand": "COSRX"},
                    ]
                },
            }
        ]
        section = builder._build_facts_section(facts)
        assert "Top 브랜드" in section.content

    def test_facts_limited_to_5(self):
        builder = ContextBuilder()
        facts = [
            {"type": "brand_products", "entity": f"Brand{i}", "data": {"product_count": i}}
            for i in range(10)
        ]
        section = builder._build_facts_section(facts)
        # 5개만 처리
        assert "Brand4" in section.content
        assert "Brand5" not in section.content

    def test_unknown_fact_type(self):
        """알 수 없는 타입은 빈 라인만"""
        builder = ContextBuilder()
        facts = [{"type": "unknown_type", "entity": "X", "data": {}}]
        section = builder._build_facts_section(facts)
        assert section.priority == ContextPriority.MEDIUM


# ---------------------------------------------------------------------------
# _build_rag_section
# ---------------------------------------------------------------------------


class TestBuildRagSection:
    """RAG 섹션 테스트"""

    def test_basic_rag_chunk(self):
        builder = ContextBuilder()
        chunks = [
            {
                "content": "SoS는 Share of Shelf입니다.",
                "metadata": {"title": "지표 정의", "doc_id": "metrics_guide"},
            }
        ]
        section = builder._build_rag_section(chunks)
        assert "지표 정의" in section.content
        assert "SoS는 Share of Shelf" in section.content
        assert len(builder.get_source_references()) == 1

    def test_rag_chunk_without_title(self):
        """제목 없는 청크는 doc_id 사용"""
        builder = ContextBuilder()
        chunks = [
            {
                "content": "Content here",
                "metadata": {"doc_id": "doc_123"},
            }
        ]
        section = builder._build_rag_section(chunks)
        assert "doc_123" in section.content

    def test_rag_content_truncation(self):
        """400자 초과 시 잘림"""
        builder = ContextBuilder()
        chunks = [
            {
                "content": "A" * 500,
                "metadata": {"title": "Long Doc", "doc_id": "long"},
            }
        ]
        section = builder._build_rag_section(chunks)
        assert "..." in section.content
        # 원본 500자가 아닌 400자+...
        assert "A" * 401 not in section.content

    def test_rag_limited_to_3_chunks(self):
        builder = ContextBuilder()
        chunks = [
            {
                "content": f"Content {i}",
                "metadata": {"title": f"Doc {i}", "doc_id": f"d{i}"},
            }
            for i in range(5)
        ]
        section = builder._build_rag_section(chunks)
        assert "Doc 2" in section.content
        assert "Doc 3" not in section.content
        assert len(builder.get_source_references()) == 3


# ---------------------------------------------------------------------------
# _select_within_limit
# ---------------------------------------------------------------------------


class TestSelectWithinLimit:
    """토큰 제한 선택 테스트"""

    def test_all_fit(self):
        builder = ContextBuilder(max_tokens=1000)
        sections = [
            ContextSection("A", "x" * 40, ContextPriority.HIGH, token_estimate=10),
            ContextSection("B", "x" * 40, ContextPriority.MEDIUM, token_estimate=10),
        ]
        selected = builder._select_within_limit(sections)
        assert len(selected) == 2

    def test_exceeds_limit_drops_lower_priority(self):
        builder = ContextBuilder(max_tokens=15)
        sections = [
            ContextSection("A", "x" * 40, ContextPriority.HIGH, token_estimate=10),
            ContextSection("B", "x" * 40, ContextPriority.MEDIUM, token_estimate=10),
        ]
        selected = builder._select_within_limit(sections)
        assert len(selected) == 1
        assert selected[0].title == "A"

    def test_critical_always_included(self):
        """CRITICAL은 토큰 초과해도 포함"""
        builder = ContextBuilder(max_tokens=5)
        sections = [
            ContextSection("A", "x" * 40, ContextPriority.HIGH, token_estimate=10),
            ContextSection("B", "x" * 40, ContextPriority.CRITICAL, token_estimate=10),
        ]
        selected = builder._select_within_limit(sections)
        titles = [s.title for s in selected]
        assert "B" in titles

    def test_priority_ordering(self):
        builder = ContextBuilder(max_tokens=100)
        sections = [
            ContextSection("Low", "x", ContextPriority.LOW, token_estimate=10),
            ContextSection("Critical", "x", ContextPriority.CRITICAL, token_estimate=10),
            ContextSection("Medium", "x", ContextPriority.MEDIUM, token_estimate=10),
        ]
        selected = builder._select_within_limit(sections)
        assert selected[0].title == "Critical"


# ---------------------------------------------------------------------------
# _assemble
# ---------------------------------------------------------------------------


class TestAssemble:
    """최종 조립 테스트"""

    def test_markdown_format(self):
        builder = ContextBuilder(output_format=OutputFormat.MARKDOWN)
        sections = [
            ContextSection("섹션1", "내용1", ContextPriority.HIGH),
        ]
        result = builder._assemble(sections, query="테스트 질문")
        assert "## 사용자 질문" in result
        assert "테스트 질문" in result
        assert "## 섹션1" in result

    def test_plain_format(self):
        builder = ContextBuilder(output_format=OutputFormat.PLAIN)
        sections = [
            ContextSection("섹션1", "내용1", ContextPriority.HIGH),
        ]
        result = builder._assemble(sections)
        assert "[섹션1]" in result

    def test_structured_format(self):
        builder = ContextBuilder(output_format=OutputFormat.STRUCTURED)
        sections = [
            ContextSection("섹션1", "내용1", ContextPriority.HIGH),
        ]
        result = builder._assemble(sections)
        assert "내용1" in result

    def test_assemble_no_query(self):
        builder = ContextBuilder()
        sections = [
            ContextSection("섹션1", "내용1", ContextPriority.HIGH),
        ]
        result = builder._assemble(sections)
        assert "사용자 질문" not in result

    def test_assemble_with_source_appendix(self):
        builder = ContextBuilder()
        builder._register_source("rag", "Doc", "detail")
        sections = [
            ContextSection("섹션1", "내용1", ContextPriority.HIGH),
        ]
        result = builder._assemble(sections)
        assert "출처" in result


# ---------------------------------------------------------------------------
# build() 통합 — HybridContext 기반
# ---------------------------------------------------------------------------


class TestBuildWithHybridContext:
    """build() 메소드 HybridContext 통합 테스트"""

    def test_build_with_inferences(self):
        builder = ContextBuilder()
        ctx = _make_context(
            inferences=[_make_inference(insight="시장 점유율 상승")],
        )
        result = builder.build(ctx, current_metrics=None, query="분석")
        assert "시장 점유율 상승" in result

    def test_build_with_rag_chunks(self):
        builder = ContextBuilder()
        ctx = _make_context(
            rag_chunks=[{"content": "가이드 내용", "metadata": {"title": "Guide", "doc_id": "g1"}}],
        )
        result = builder.build(ctx)
        assert "Guide" in result

    def test_build_with_ontology_facts(self):
        builder = ContextBuilder()
        ctx = _make_context(
            ontology_facts=[{"type": "brand_info", "entity": "LANEIGE", "data": {"sos": 0.15}}],
        )
        result = builder.build(ctx)
        assert "LANEIGE" in result

    def test_build_with_metrics(self):
        builder = ContextBuilder()
        ctx = _make_context()
        metrics = {
            "summary": {
                "laneige_products_tracked": 5,
                "alert_count": 2,
                "critical_alerts": 1,
                "warning_alerts": 1,
            }
        }
        result = builder.build(ctx, current_metrics=metrics)
        assert "5개" in result

    def test_build_full_integration(self):
        builder = ContextBuilder()
        ctx = _make_context(
            inferences=[_make_inference(insight="SoS 상승 추세", recommendation="광고 강화")],
            rag_chunks=[
                {
                    "content": "가이드라인 본문",
                    "metadata": {"title": "전략 가이드", "doc_id": "strat"},
                }
            ],
            ontology_facts=[{"type": "brand_info", "entity": "LANEIGE", "data": {"sos": 0.12}}],
        )
        metrics = {
            "summary": {
                "laneige_products_tracked": 3,
                "alert_count": 0,
                "critical_alerts": 0,
                "warning_alerts": 0,
            }
        }
        result = builder.build(ctx, current_metrics=metrics, query="LANEIGE 분석")
        assert "SoS 상승 추세" in result
        assert "광고 강화" in result
        assert "전략 가이드" in result
        assert "LANEIGE" in result

    def test_build_with_category_hierarchy_ranking_query(self):
        """순위 관련 쿼리 시 카테고리 계층 포함"""
        builder = ContextBuilder()
        mock_kg = MagicMock()
        mock_kg.get_category_hierarchy.return_value = {
            "name": "Lip Care",
            "level": 2,
            "ancestors": [{"name": "Beauty"}, {"name": "Skin Care"}],
            "descendants": [{"name": "Lip Balm"}, {"name": "Lip Treatment"}],
        }
        ctx = _make_context(entities={"categories": ["lip_care"], "products": []})
        result = builder.build(ctx, query="LANEIGE 순위 분석", knowledge_graph=mock_kg)
        assert "Lip Care" in result
        assert "Skin Care > Beauty > Lip Care" in result

    def test_build_with_product_category_context(self):
        """제품별 카테고리 순위 컨텍스트"""
        builder = ContextBuilder()
        mock_kg = MagicMock()
        mock_kg.get_category_hierarchy.return_value = {"error": "not found"}
        mock_kg.get_product_category_context.return_value = {
            "categories": [
                {
                    "category_id": "lip_care",
                    "rank": 3,
                    "hierarchy": {"name": "Lip Care", "level": 2},
                }
            ]
        }
        mock_kg.get_entity_metadata.return_value = {"product_name": "LANEIGE Lip Mask"}
        ctx = _make_context(entities={"categories": [], "products": ["B0BSHRYY1S"]})
        result = builder.build(ctx, query="순위 분석", knowledge_graph=mock_kg)
        assert "LANEIGE Lip Mask" in result
        assert "3위" in result

    def test_build_category_hierarchy_with_many_descendants(self):
        """하위 카테고리 5개 초과 시 '외 N개' 표시"""
        builder = ContextBuilder()
        mock_kg = MagicMock()
        mock_kg.get_category_hierarchy.return_value = {
            "name": "Skin Care",
            "level": 1,
            "ancestors": [],
            "descendants": [{"name": f"Sub {i}"} for i in range(8)],
        }
        ctx = _make_context(entities={"categories": ["skin_care"], "products": []})
        result = builder.build(ctx, query="순위 확인", knowledge_graph=mock_kg)
        assert "외 3개" in result


# ---------------------------------------------------------------------------
# build_system_prompt
# ---------------------------------------------------------------------------


class TestBuildSystemPrompt:
    """시스템 프롬프트 테스트"""

    def test_system_prompt_with_guardrails(self):
        builder = ContextBuilder()
        prompt = builder.build_system_prompt(include_guardrails=True)
        assert isinstance(prompt, str)
        assert len(prompt) > 100

    def test_system_prompt_without_guardrails(self):
        builder = ContextBuilder()
        prompt = builder.build_system_prompt(include_guardrails=False)
        assert isinstance(prompt, str)

    def test_system_prompt_with_date(self):
        builder = ContextBuilder()
        prompt = builder.build_system_prompt(data_date="2026-01-28")
        assert isinstance(prompt, str)


# ---------------------------------------------------------------------------
# build_user_prompt
# ---------------------------------------------------------------------------


class TestBuildUserPrompt:
    """사용자 프롬프트 테스트"""

    def test_basic_user_prompt(self):
        builder = ContextBuilder()
        prompt = builder.build_user_prompt(
            query="LANEIGE SoS?",
            context="SoS는 12%입니다.",
        )
        assert "LANEIGE SoS?" in prompt
        assert "SoS는 12%" in prompt

    def test_user_prompt_with_additional_instructions(self):
        builder = ContextBuilder()
        prompt = builder.build_user_prompt(
            query="분석",
            context="데이터",
            additional_instructions="표 형식으로 답변",
        )
        assert "추가 지시사항" in prompt
        assert "표 형식으로 답변" in prompt


# ---------------------------------------------------------------------------
# CompactContextBuilder — HybridContext 기반
# ---------------------------------------------------------------------------


class TestCompactContextBuilderExtended:
    """CompactContextBuilder HybridContext 기반 테스트"""

    def test_compact_with_inferences(self):
        builder = CompactContextBuilder()
        ctx = _make_context(
            inferences=[
                _make_inference(insight="Insight 1", recommendation="Action 1"),
                _make_inference(insight="Insight 2"),
            ],
        )
        result = builder.build(ctx)
        assert "Insight 1" in result
        assert "Action 1" in result
        assert "Insight 2" in result

    def test_compact_with_metrics(self):
        builder = CompactContextBuilder()
        ctx = _make_context()
        metrics = {
            "summary": {
                "laneige_products_tracked": 5,
                "laneige_sos_by_category": {"lip_care": 0.15},
            }
        }
        result = builder.build(ctx, current_metrics=metrics)
        assert "5개" in result
        assert "15.0%" in result

    def test_compact_with_rag_titles(self):
        builder = CompactContextBuilder()
        ctx = _make_context(
            rag_chunks=[
                {"content": "Full content", "metadata": {"title": "가이드 제목"}},
            ],
        )
        result = builder.build(ctx)
        assert "가이드 제목" in result
        # 전체 내용은 포함하지 않음 (compact)
        assert "Full content" not in result

    def test_compact_with_ranking_query_and_kg(self):
        """순위 쿼리 시 카테고리별 순위 정보"""
        builder = CompactContextBuilder()
        mock_kg = MagicMock()
        mock_kg.get_product_category_context.return_value = {
            "categories": [
                {
                    "category_id": "lip_care",
                    "rank": 5,
                    "hierarchy": {"name": "Lip Care", "level": 2},
                }
            ]
        }
        ctx = _make_context(entities={"products": ["B0BSHRYY1S"], "categories": []})
        result = builder.build(ctx, query="LANEIGE 순위", knowledge_graph=mock_kg)
        assert "Lip Care" in result
        assert "5위" in result

    def test_compact_empty_context(self):
        builder = CompactContextBuilder()
        ctx = _make_context()
        result = builder.build(ctx)
        assert isinstance(result, str)
