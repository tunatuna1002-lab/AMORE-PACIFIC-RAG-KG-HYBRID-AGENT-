"""증거 카드로 답변 프롬프트를 조립한다 (트랙 2-B, 설계 E1·E2·E8, 결함 F8).

실제 HybridRetriever(임시 KG·운영 스키마 SQLite·규칙 추론기·어댑터·렌더러)와 가짜 문서
검색기만 쓴다. v4 ``combined_context``와 v1 ``ContextBuilder.build``가 같은 카드 조립기와
렌더러를 쓰는지, DB 수치가 실리고 KG 수치 엣지는 실리지 않는지 확인한다.
"""

import re
from types import SimpleNamespace

import pytest

from src.domain.entities.evidence import Evidence, EvidenceKind, EvidenceUnit
from src.domain.value_objects.retrieval_result import UnifiedRetrievalResult
from src.rag import evidence_assembly
from src.rag.context_builder import ContextBuilder
from src.rag.evidence_adapters import EvidenceAdapter
from src.rag.evidence_assembly import (
    METRIC_GROUP_CAPS,
    METRIC_GROUP_MARKET,
    METRIC_GROUP_OTHER,
    METRIC_GROUP_QUERY_BRAND,
    METRIC_GROUP_QUERY_BRAND_PRODUCT,
    METRIC_GROUP_TOP_BRAND_SOS,
    METRIC_GROUP_TOP_PRODUCT,
    PROMPT_MAX_PER_KIND,
    assemble_evidence,
    evidence_source_labels,
    metric_group,
)
from src.rag.evidence_renderer import CITATION_INSTRUCTION, render_for_prompt

from .evidence_pipeline_fixtures import (
    AS_OF,
    CURRENT_METRICS,
    DOC_CHUNKS,
    KG_HHI_EDGE_VALUE,
    QUERY,
    make_retriever,
)

HHI_LINE = re.compile(
    r"^\[M-[0-9a-f]{6,40}\] lip_care HHI 0\.0681 \(2026-08-31, sqlite:market_metrics\)$",
    re.MULTILINE,
)


@pytest.fixture
async def retrieved(tmp_path):
    retriever = make_retriever(tmp_path)
    ctx = await retriever.retrieve(QUERY, current_metrics=CURRENT_METRICS)
    assert "retrieval_error" not in ctx.metadata
    return retriever, ctx


# ----------------------------------------------------------------------
# v4: HybridRetriever.retrieve → combined_context
# ----------------------------------------------------------------------


async def test_db_hhi_card_is_rendered_in_combined_context(retrieved):
    _, ctx = retrieved

    assert HHI_LINE.search(ctx.combined_context), ctx.combined_context


async def test_every_prompt_card_id_is_rendered_and_is_part_of_evidence(retrieved):
    _, ctx = retrieved

    assert ctx.prompt_evidence
    evidence_ids = {card.id for card in ctx.evidence}
    for card in ctx.prompt_evidence:
        assert f"[{card.id}]" in ctx.combined_context
        assert card.id in evidence_ids
    assert ctx.combined_context == render_for_prompt(ctx.prompt_evidence)


async def test_all_card_kinds_are_assembled(retrieved):
    _, ctx = retrieved

    kinds = {card.kind for card in ctx.prompt_evidence}
    assert kinds == {
        EvidenceKind.METRIC,
        EvidenceKind.RELATION,
        EvidenceKind.INFERENCE,
        EvidenceKind.DOCUMENT,
    }
    relation_texts = [c.text for c in ctx.prompt_evidence if c.kind == EvidenceKind.RELATION]
    assert any("competesWith" in text and "burt's bees" in text for text in relation_texts)
    # 추론 카드의 derived_from = 규칙 입력 카드 id (트랙 3-B), 모두 전체 카드 안에 있다
    evidence_ids = {c.id for c in ctx.evidence}
    inference_cards = [c for c in ctx.evidence if c.kind == EvidenceKind.INFERENCE]
    assert inference_cards
    assert all(c.derived_from and set(c.derived_from) <= evidence_ids for c in inference_cards)
    # 문서는 검색 결과 전부
    documents = [c for c in ctx.prompt_evidence if c.kind == EvidenceKind.DOCUMENT]
    assert [c.metadata["chunk_id"] for c in documents] == [chunk["id"] for chunk in DOC_CHUNKS]


async def test_kg_numeric_values_are_not_rendered(retrieved):
    _, ctx = retrieved
    text = ctx.combined_context

    assert "42.4" not in text  # KG 엔티티 메타데이터 SoS (날짜 없음)
    assert "33.3" not in text  # KG 엔티티 메타데이터 평균 순위
    assert KG_HHI_EDGE_VALUE not in text
    assert "hasSoS" not in text and "hasHHI" not in text
    # brand_info 사실의 날짜 없는 메타데이터(type·sos·avg_rank·product_count)가 제외된다.
    # metric_edges 사실(hasSoS·hasHHI 엣지)은 _weighted_merge의 사실 상한 5개에서 먼저 잘린다.
    assert ctx.metadata["evidence_excluded_by_reason"] == {"kg_entity_metadata_undated": 4}
    assert ctx.metadata["evidence_excluded"] == 4


def test_kg_numeric_edges_are_excluded_when_they_reach_the_assembler():
    facts = [
        {
            "type": "metric_edges",
            "entity": "laneige",
            "data": {
                "edges": [
                    {"subject": "laneige", "predicate": "hasSoS", "object": "lip_care"},
                    {"subject": "laneige", "predicate": "hasHHI", "object": KG_HHI_EDGE_VALUE},
                    {"subject": "laneige", "predicate": "ownedBy", "object": "amorepacific"},
                ]
            },
        }
    ]

    bundle = assemble_evidence(entities={"brands": ["laneige"]}, ontology_facts=facts)
    text = render_for_prompt(bundle.prompt_evidence)

    assert bundle.excluded_by_reason == {"kg_numeric_undated": 2}
    assert KG_HHI_EDGE_VALUE not in text and "hasSoS" not in text
    assert "laneige ownedBy amorepacific" in text


async def test_old_non_card_sections_are_gone(retrieved):
    _, ctx = retrieved

    for header in ("## 분석 결과", "## 관련 정보", "## 참고 가이드라인"):
        assert header not in ctx.combined_context
    assert CITATION_INSTRUCTION not in ctx.combined_context  # 인용 지시는 답변 프롬프트에만


async def test_card_ids_are_deterministic_across_retrievers(tmp_path):
    first_dir, second_dir = tmp_path / "a", tmp_path / "b"
    first_dir.mkdir()
    second_dir.mkdir()

    first = await make_retriever(first_dir).retrieve(QUERY, current_metrics=CURRENT_METRICS)
    second = await make_retriever(second_dir).retrieve(QUERY, current_metrics=CURRENT_METRICS)

    assert [c.id for c in first.evidence] == [c.id for c in second.evidence]
    assert first.combined_context == second.combined_context


async def test_adapter_failure_is_degraded_not_retrieval_error(tmp_path, monkeypatch):
    def broken(self, facts):
        raise RuntimeError("adapter broke")

    monkeypatch.setattr(EvidenceAdapter, "from_kg_facts", broken)

    ctx = await make_retriever(tmp_path).retrieve(QUERY, current_metrics=CURRENT_METRICS)

    assert "retrieval_error" not in ctx.metadata
    assert {"component": "evidence_relation", "error": "RuntimeError: adapter broke"} in (
        ctx.metadata["degraded"]
    )
    assert not [c for c in ctx.evidence if c.kind == EvidenceKind.RELATION]
    assert HHI_LINE.search(ctx.combined_context)  # 다른 종류의 카드는 그대로 실린다


async def test_self_rag_skip_has_no_cards(tmp_path):
    ctx = await make_retriever(tmp_path).retrieve("안녕")

    assert ctx.evidence == [] and ctx.prompt_evidence == []


async def test_retrieve_unified_carries_cards(tmp_path):
    retriever = make_retriever(tmp_path)

    result = await retriever.retrieve_unified(QUERY, current_metrics=CURRENT_METRICS)

    assert isinstance(result, UnifiedRetrievalResult)
    assert result.prompt_evidence and result.evidence
    assert result.combined_context == render_for_prompt(result.prompt_evidence)
    assert HHI_LINE.search(result.combined_context)


async def test_skip_path_defaults_to_empty_cards(tmp_path):
    retriever = make_retriever(tmp_path)

    skipped = await retriever.retrieve_unified("안녕")

    assert (skipped.evidence, skipped.prompt_evidence) == ([], [])


# ----------------------------------------------------------------------
# v1: ContextBuilder가 같은 카드를 렌더링한다
# ----------------------------------------------------------------------


async def test_v1_context_builder_renders_the_same_cards(retrieved):
    retriever, ctx = retrieved

    text = ContextBuilder(max_tokens=3000).build(
        ctx, current_metrics=None, query=QUERY, knowledge_graph=retriever.kg
    )

    assert render_for_prompt(ctx.prompt_evidence) in text
    for card in ctx.prompt_evidence:
        assert f"[{card.id}]" in text
    assert HHI_LINE.search(text)
    assert "42.4" not in text
    assert CITATION_INSTRUCTION not in text  # 챗봇 프롬프트 조립부가 한 번 붙인다


def test_v1_context_builder_assembles_cards_for_contexts_without_them():
    facts = [
        {"type": "category_market", "category": "lip_care", "snapshot_date": AS_OF, "hhi": 0.0681}
    ]
    plain = SimpleNamespace(
        entities={}, inferences=[], ontology_facts=[], rag_chunks=[], metric_facts=facts
    )

    text = ContextBuilder(max_tokens=3000).build(plain, None, "Lip Care HHI는?")

    assert HHI_LINE.search(text)


# ----------------------------------------------------------------------
# 조립기 단위: metric 우선순위·상한·출처 표시
# ----------------------------------------------------------------------


def _metric_facts(categories=("lip_care",), brands=("LANEIGE",)):
    facts = []
    for category in categories:
        facts.append(
            {
                "type": "category_market",
                "category": category,
                "snapshot_date": AS_OF,
                "hhi": 0.05,
                "category_avg_price": 17.2,
                "category_avg_rating": 4.5,
            }
        )
        facts.append(
            {
                "type": "category_top_brands",
                "category": category,
                "snapshot_date": AS_OF,
                "brands": [
                    {"brand": f"Top{i}", "sos": 10.0 - i, "product_count": 10 - i} for i in range(5)
                ],
            }
        )
        for brand in brands:
            facts.append(
                {
                    "type": "brand_share",
                    "brand": brand,
                    "category": category,
                    "snapshot_date": AS_OF,
                    "present": True,
                    "sos": 2.0,
                    "product_count": 2,
                    "brand_rank": 9,
                }
            )
        facts.append(
            {
                "type": "category_top_products",
                "category": category,
                "snapshot_date": AS_OF,
                "products": [
                    {
                        "rank": r,
                        "brand": f"Top{r}",
                        "name": f"{category} Top Product {r}",
                        "price": 10.0,
                        "rating": 4.5,
                        "reviews_count": 100,
                    }
                    for r in range(1, 6)
                ],
            }
        )
        for brand in brands:
            facts.append(
                {
                    "type": "brand_products",
                    "brand": brand,
                    "category": category,
                    "snapshot_date": AS_OF,
                    "products": [
                        {
                            "rank": 40 + r,
                            "brand": brand,
                            "name": f"{brand} {category} Product {r}",
                            "price": 20.0,
                            "rating": 4.6,
                            "reviews_count": 900,
                        }
                        for r in range(3)
                    ],
                }
            )
    return facts


def test_metric_cards_follow_group_priority():
    bundle = assemble_evidence(
        entities={"brands": ["laneige"], "categories": ["lip_care"]},
        metric_facts=_metric_facts(),
    )
    brands = frozenset({"laneige"})
    groups = [metric_group(c, brands) for c in bundle.prompt_evidence]

    assert groups == sorted(groups)
    assert groups[0] == METRIC_GROUP_MARKET
    assert set(groups) >= {
        METRIC_GROUP_MARKET,
        METRIC_GROUP_QUERY_BRAND,
        METRIC_GROUP_TOP_BRAND_SOS,
        METRIC_GROUP_QUERY_BRAND_PRODUCT,
        METRIC_GROUP_TOP_PRODUCT,
    }
    # 상위 브랜드의 SoS만 3그룹 — 제품 수·SoS 순위는 6그룹
    top_brand_cards = [c for c in bundle.evidence if c.subject == "top0"]
    assert {c.predicate: metric_group(c, brands) for c in top_brand_cards} == {
        "sos": METRIC_GROUP_TOP_BRAND_SOS,
        "product_count": METRIC_GROUP_OTHER,
        "sos_rank": METRIC_GROUP_OTHER,
    }


def test_metric_caps_keep_query_brand_products_for_multi_category_brand_queries():
    categories = ("lip_care", "lip_makeup", "face_powder")
    bundle = assemble_evidence(
        entities={"brands": ["laneige"]},
        metric_facts=_metric_facts(categories=categories),
    )
    brands = frozenset({"laneige"})
    metric_cards = [c for c in bundle.prompt_evidence if c.kind == EvidenceKind.METRIC]
    groups = [metric_group(c, brands) for c in metric_cards]

    assert len(metric_cards) == PROMPT_MAX_PER_KIND[EvidenceKind.METRIC]
    assert len(bundle.evidence) > len(metric_cards)  # 전체 카드는 잘리지 않는다
    assert groups.count(METRIC_GROUP_MARKET) == 3 * 3  # 카테고리 3개 × (HHI·평균가·평점)
    assert groups.count(METRIC_GROUP_QUERY_BRAND) == 3 * 3
    assert groups.count(METRIC_GROUP_TOP_BRAND_SOS) == METRIC_GROUP_CAPS[METRIC_GROUP_TOP_BRAND_SOS]
    assert groups.count(METRIC_GROUP_QUERY_BRAND_PRODUCT) > 0


def test_prompt_evidence_is_a_subsequence_of_evidence_and_render_input():
    bundle = assemble_evidence(
        entities={"brands": ["laneige"]},
        metric_facts=_metric_facts(categories=("lip_care", "lip_makeup")),
        rag_chunks=DOC_CHUNKS,
    )
    positions = {card.id: i for i, card in enumerate(bundle.evidence)}

    indexes = [positions[card.id] for card in bundle.prompt_evidence]
    assert indexes == sorted(indexes)
    assert render_for_prompt(bundle.prompt_evidence) == render_for_prompt(
        bundle.prompt_evidence, max_per_kind=PROMPT_MAX_PER_KIND
    )


def test_source_labels_are_deduplicated_strings():
    market = Evidence.create(
        kind=EvidenceKind.METRIC,
        subject="lip_care",
        predicate="hhi",
        value=0.0681,
        unit=EvidenceUnit.INDEX_0_1,
        as_of=AS_OF,
        source="sqlite:market_metrics",
        text="lip_care HHI 0.0681",
    )
    share = market.model_copy()
    bundle = assemble_evidence(
        ontology_facts=[
            {
                "type": "competitors",
                "entity": "laneige",
                "data": [{"brand": "burt's bees", "type": "competesWith", "category": "lip_care"}],
            }
        ],
        rag_chunks=DOC_CHUNKS,
    )

    labels = evidence_source_labels([market, share, *bundle.prompt_evidence])

    assert labels == [
        "sqlite:market_metrics (2026-08-31)",
        "KG",
        "HHI 해석 가이드",
        "SoS 대응 플레이북",
    ]


def test_module_exposes_caps_as_constants():
    assert evidence_assembly.PROMPT_MAX_PER_KIND[EvidenceKind.METRIC] > 0
    assert EvidenceKind.DOCUMENT not in evidence_assembly.PROMPT_MAX_PER_KIND


def test_relation_priority_keeps_ownership_when_competitors_fill_the_cap():
    """lg158 실측: 경쟁 관계가 상한을 먼저 채워 ownedBy가 프롬프트에서 잘리던 문제"""
    competitors = [{"brand": f"brand{i}", "type": "competesWith"} for i in range(25)]
    facts = [
        {"type": "competitors", "entity": "laneige", "data": competitors},
        {
            "type": "metric_edges",
            "entity": "laneige",
            "data": {
                "edges": [{"subject": "laneige", "predicate": "ownedBy", "object": "AMOREPACIFIC"}]
            },
        },
    ]

    bundle = assemble_evidence(entities={"brands": ["laneige"]}, ontology_facts=facts)
    relations = [c for c in bundle.prompt_evidence if c.kind == EvidenceKind.RELATION]

    assert len(relations) == PROMPT_MAX_PER_KIND[EvidenceKind.RELATION]
    assert relations[0].predicate == "ownedBy"
    # 같은 술어 안에서는 입력 순서를 지킨다
    competes = [c.object for c in relations if c.predicate == "competesWith"]
    assert competes == [f"brand{i}" for i in range(len(competes))]
