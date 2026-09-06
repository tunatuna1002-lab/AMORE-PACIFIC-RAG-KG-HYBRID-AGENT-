"""
Characterization: src.rag.hybrid_retriever.HybridRetriever

Real KnowledgeGraph (seeded, tmp persist path) + real OntologyReasoner with
all business rules + a fake DocumentRetriever injected through the
constructor. No ChromaDB, no LLM, no network.
"""

from __future__ import annotations

from typing import Any

import pytest

from src.rag.hybrid_retriever import HybridContext, HybridRetriever
from tests.characterization.conftest import CANNED_CHUNKS, CURRENT_METRICS, FakeDocRetriever

EXPANDED_QUERY = (
    "LANEIGE Lip Care SoS는? (Share of Shelf, 점유율, 라네즈) 시장 포지션 해석 SoS 점유율 해석"
)


@pytest.fixture
def retriever(kg, reasoner, fake_doc_retriever) -> HybridRetriever:
    return HybridRetriever(knowledge_graph=kg, reasoner=reasoner, doc_retriever=fake_doc_retriever)


def test_constructor_does_not_reregister_rules(kg, reasoner, fake_doc_retriever) -> None:
    before = len(reasoner.rules)
    hr = HybridRetriever(knowledge_graph=kg, reasoner=reasoner, doc_retriever=fake_doc_retriever)
    assert before == 37
    assert len(hr.reasoner.rules) == 37
    assert hr._initialized is False
    # config/retrieval_weights.json overrides the in-code default of 3 rag chunks
    assert hr._retrieval_weights["max_context_items"] == {
        "ontology_facts": 5,
        "inferences": 5,
        "rag_chunks": 8,
    }


# ---------------------------------------------------------------------------
# Self-RAG gate
# ---------------------------------------------------------------------------


async def test_retrieve_two_char_greeting_is_skipped_as_too_short(
    retriever: HybridRetriever, fake_doc_retriever: FakeDocRetriever, kg
) -> None:
    triples_before = len(kg.triples)
    ctx = await retriever.retrieve("안녕")

    assert isinstance(ctx, HybridContext)
    # PINS CURRENT BEHAVIOR: the len<=2 check runs before the greeting patterns,
    # so "안녕" reports query_too_short rather than greeting_or_command.
    assert ctx.metadata == {
        "self_rag_skip": True,
        "skip_reason": "query_too_short",
        "selfrag_confidence": 0.0,
    }
    assert ctx.combined_context == "[Retrieval skipped: query_too_short]"
    assert ctx.entities == {}
    assert ctx.ontology_facts == [] and ctx.inferences == [] and ctx.rag_chunks == []
    # initialize() runs *before* the gate: doc retriever initialised and the
    # category hierarchy was merged into the KG even though nothing was retrieved.
    assert fake_doc_retriever.initialized is True
    assert fake_doc_retriever.calls == []
    assert retriever._initialized is True
    assert len(kg.triples) > triples_before


async def test_retrieve_longer_greeting_is_skipped_as_greeting(retriever: HybridRetriever) -> None:
    ctx = await retriever.retrieve("안녕하세요")
    assert ctx.metadata["self_rag_skip"] is True
    assert ctx.metadata["skip_reason"] == "greeting_or_command"
    assert ctx.combined_context == "[Retrieval skipped: greeting_or_command]"


@pytest.mark.parametrize(
    ("query", "expected"),
    [
        ("hello there", (False, "greeting_or_command", 0.0)),
        ("고마워", (False, "greeting_or_command", 0.0)),
        ("abc", (False, "short_non_domain_query", 0.0)),
        ("음 그래서", (False, "short_non_domain_query", 0.0)),
        ("LANEIGE Lip Care SoS는?", (True, "domain_query_detected", 1.0)),
        ("오늘 점심 뭐 먹지?", (True, "domain_query_detected", 1.0)),  # "뭐" is a question word
        ("그냥 궁금해서요", (True, "default_retrieve", 0.8)),
        ("", (False, "query_too_short", 0.0)),
    ],
)
def test_should_retrieve(retriever: HybridRetriever, query: str, expected: tuple) -> None:
    assert retriever.should_retrieve(query) == expected


# ---------------------------------------------------------------------------
# Full retrieve() path
# ---------------------------------------------------------------------------


async def test_retrieve_laneige_lip_care_sos(
    retriever: HybridRetriever, fake_doc_retriever: FakeDocRetriever
) -> None:
    ctx = await retriever.retrieve("LANEIGE Lip Care SoS는?", current_metrics=CURRENT_METRICS)

    # Entities (EntityLinker output + "concepts")
    assert ctx.entities == {
        "brands": ["laneige"],
        "categories": ["lip_care"],
        "indicators": ["sos"],
        "time_range": [],
        "products": [],
        "sentiments": [],
        "sentiment_clusters": [],
        "concepts": ["sos"],
    }

    # Ontology facts: order after weighted merge (kg weight 0.4 * base score)
    assert [(f["type"], f["entity"]) for f in ctx.ontology_facts] == [
        ("category_brands", "lip_care"),
        ("category_hierarchy", "lip_care"),
        ("brand_products", "laneige"),
        ("metric_edges", "laneige"),
    ]
    # FIXED (brand casing): KnowledgeGraph.get_brand_products / get_competitors resolve the
    # lower-cased extracted brand "laneige" to the stored "LANEIGE" (exact match first, then
    # case-insensitive), so the seeded LANEIGE hasProduct B0LANE1 now surfaces as a fact.
    # COSRX competesWith LANEIGE has COSRX as subject, so there is still no competitors fact.
    assert not any(f["type"] in {"competitors", "brand_info"} for f in ctx.ontology_facts)

    cat_brands, hierarchy, brand_products, edges = ctx.ontology_facts
    assert brand_products["data"]["product_count"] == 1
    assert [p["asin"] for p in brand_products["data"]["products"]] == ["B0LANE1"]
    assert brand_products["_weighted_score"] == pytest.approx(0.24)
    assert cat_brands["data"] == {
        "brand_count": 1,
        "top_brands": [{"brand": "LANEIGE", "product_count": 1, "products": ["B0LANE1"]}],
    }
    assert cat_brands["_weighted_score"] == pytest.approx(0.32)
    assert hierarchy["data"]["name"] == "Lip Care"
    assert hierarchy["data"]["level"] == 2
    assert hierarchy["data"]["path"] == ["beauty", "skin_care", "lip_care"]
    assert [a["id"] for a in hierarchy["data"]["ancestors"]] == ["skin_care", "beauty"]
    assert edges["data"] == {
        "edges": [
            {"subject": "laneige", "predicate": "hasProduct", "object": "lip_sleeping_mask"},
            {"subject": "laneige", "predicate": "hasProduct", "object": "lip_sleeping"},
            {
                "subject": "lip_sleeping_mask",
                "predicate": "belongsToCategory",
                "object": "lip_care",
            },
            # seed triple ownedByGroup is re-labelled ownedBy, keeps original-case subject
            {"subject": "LANEIGE", "predicate": "ownedBy", "object": "AMOREPACIFIC"},
        ]
    }
    assert edges["_weighted_score"] == pytest.approx(0.24)

    # Inference from the metrics (avg_rank 8.5 + is_target)
    assert [(i.insight_type.value, i.rule_name, i.confidence) for i in ctx.inferences] == [
        ("market_position", "strong_avg_rank", 0.85)
    ]

    # RAG chunks: both canned chunks survive, re-scored by rag weight * score * freshness
    assert [c["id"] for c in ctx.rag_chunks] == ["c1", "c2"]
    assert ctx.rag_chunks[0]["_weighted_score"] == pytest.approx(0.4 * 0.9 * 0.8)
    assert ctx.rag_chunks[1]["_weighted_score"] == pytest.approx(0.4 * 0.7 * 0.9)

    # Search calls: filtered search, then (because <3 results) an unfiltered top-up
    assert fake_doc_retriever.calls == [
        (EXPANDED_QUERY, 5, ["metric_guide", "playbook"]),
        (EXPANDED_QUERY, 3, None),
    ]

    # Metadata
    assert set(ctx.metadata) == {
        "fusion",
        "weighted_scores",
        "retrieval_time_ms",
        "ontology_facts_count",
        "inferences_count",
        "rag_chunks_count",
        "query_expanded",
        "query_intent",
        "doc_type_filter",
        "intent_strategy",
        "intent_weights",
        "search_method",
        "selfrag_confidence",
        "bm25_available",
    }
    # FIXED (bug D25): retrieve() merges its own keys into the metadata that
    # _weighted_merge already filled, so "fusion"/"weighted_scores" reach the caller.
    assert "confidence" in ctx.metadata["fusion"]
    assert set(ctx.metadata["weighted_scores"]) == {"ontology_facts", "rag_chunks", "inferences"}
    assert ctx.metadata["weighted_scores"]["ontology_facts"] == pytest.approx(
        [0.32, 0.32, 0.24, 0.24]
    )
    assert ctx.metadata["query_intent"] == "metric"
    assert ctx.metadata["doc_type_filter"] == ["metric_guide", "playbook"]
    assert ctx.metadata["intent_strategy"] == "balanced/metric"
    assert ctx.metadata["intent_weights"] == {"kg": 0.4, "rag": 0.4, "inference": 0.2}
    assert ctx.metadata["search_method"] == "dense_only"
    assert ctx.metadata["bm25_available"] is False  # fake has no search_bm25
    assert ctx.metadata["selfrag_confidence"] == 1.0
    assert ctx.metadata["query_expanded"] is True
    assert ctx.metadata["ontology_facts_count"] == 4
    assert ctx.metadata["inferences_count"] == 1
    assert ctx.metadata["rag_chunks_count"] == 2

    # Combined context
    assert "LANEIGE" in ctx.combined_context
    assert ctx.combined_context.startswith("## 분석 결과 (Ontology Reasoning)\n")
    assert "### 인사이트 1: Market Position" in ctx.combined_context
    assert "- **신뢰도**: 85%" in ctx.combined_context
    assert "- **근거 조건**: low_avg_rank, is_target" in ctx.combined_context
    assert "- **lip_care** Top 브랜드: LANEIGE" in ctx.combined_context
    assert "- **Lip Care** 계층: beauty > skin_care > lip_care (Level 2)" in ctx.combined_context
    assert "  - 상위 카테고리: Skin Care, Beauty & Personal Care" in ctx.combined_context
    assert "- **laneige** 제품 수: 1개" in ctx.combined_context
    assert "## 참고 가이드라인 (RAG)" in ctx.combined_context
    assert "### SoS 정의\nSoS(Share of Shelf)는 점유율 지표입니다." in ctx.combined_context


async def test_retrieve_without_metrics_produces_entry_opportunity(
    retriever: HybridRetriever,
) -> None:
    ctx = await retriever.retrieve("LANEIGE Lip Care SoS는?")
    # PINS CURRENT BEHAVIOR: with no metrics the inference context has sos/hhi
    # missing, which the rule engine reads as 0 -> spurious "entry opportunity".
    assert [(i.insight_type.value, i.rule_name) for i in ctx.inferences] == [
        ("entry_opportunity", "category_entry_opportunity")
    ]
    assert "HHI: 0.000" in ctx.inferences[0].insight
    assert "(0.0%)" in ctx.inferences[0].insight


# ---------------------------------------------------------------------------
# Bug D14 (fixed): id-less BM25 items + fewer than 3 filtered results
# ---------------------------------------------------------------------------


async def test_retrieve_idless_bm25_items_with_short_filtered_results_bug_d14(kg, reasoner) -> None:
    fake = FakeDocRetriever(
        chunks=CANNED_CHUNKS[:1],  # a single dense hit (< 3 triggers the top-up search)
        bm25_results=[{"content": "bm25 only chunk", "score": 0.5}],  # no "id"
        with_rrf=True,
    )
    hr = HybridRetriever(knowledge_graph=kg, reasoner=reasoner, doc_retriever=fake)

    ctx = await hr.retrieve("LANEIGE SoS 지표 알려줘", current_metrics=CURRENT_METRICS)

    # FIXED (bug D14): the top-up dedupe keys on `.get("id")` and falls back to the
    # chunk content, so an id-less BM25 item neither raises nor gets duplicated by the
    # unfiltered top-up search. The full context is built.
    assert "error" not in ctx.metadata
    assert ctx.metadata["search_method"] == "hybrid_rrf"
    assert [c.get("id") for c in ctx.rag_chunks] == ["c1", None]
    assert ctx.metadata["rag_chunks_count"] == 2
    assert "bm25 only chunk" in ctx.combined_context
    assert ctx.entities["brands"] == ["laneige"] and ctx.entities["indicators"] == ["sos"]
    assert [f["type"] for f in ctx.ontology_facts] == ["brand_products", "metric_edges"]
    assert [i.rule_name for i in ctx.inferences] == ["strong_avg_rank"]
    # Both searches were issued (filtered, then the unfiltered top-up).
    assert [(top_k, flt) for _q, top_k, flt in fake.calls] == [
        (5, ["metric_guide", "playbook"]),
        (3, None),
    ]


async def test_retrieve_idless_bm25_items_with_enough_results_survive(kg, reasoner) -> None:
    # Same fake but 2 dense hits: 2 + 1 fused == 3 -> the dedupe branch is skipped
    # and the id-less chunk flows through to the caller untouched.
    fake = FakeDocRetriever(
        chunks=CANNED_CHUNKS,
        bm25_results=[{"content": "bm25 only chunk", "score": 0.5}],
        with_rrf=True,
    )
    hr = HybridRetriever(knowledge_graph=kg, reasoner=reasoner, doc_retriever=fake)

    ctx = await hr.retrieve("LANEIGE SoS 지표 알려줘", current_metrics=CURRENT_METRICS)

    assert "error" not in ctx.metadata
    assert ctx.metadata["search_method"] == "hybrid_rrf"
    assert ctx.metadata["bm25_available"] is True  # rank_bm25 importable + fake has search_bm25
    assert ctx.metadata["rag_chunks_count"] == 3
    assert [c.get("id") for c in ctx.rag_chunks] == ["c1", "c2", None]
    assert ctx.rag_chunks[2]["_weighted_score"] == pytest.approx(0.4 * 0.5 * 0.8)
    assert ctx.combined_context.endswith("bm25 only chunk\n")
    assert len(fake.calls) == 1


# ---------------------------------------------------------------------------
# retrieve_unified (legacy conversion path)
# ---------------------------------------------------------------------------


async def test_retrieve_unified_legacy_path(retriever: HybridRetriever) -> None:
    result = await retriever.retrieve_unified(
        "LANEIGE Lip Care SoS는?", current_metrics=CURRENT_METRICS
    )
    assert result.retriever_type == "legacy"
    # PINS CURRENT BEHAVIOR: legacy conversion hard-codes confidence 0.0 and no
    # entity_links, regardless of fusion scores.
    assert result.confidence == 0.0
    assert result.entity_links == []
    assert result.entities["brands"] == ["laneige"]
    assert isinstance(result.inferences[0], dict)
    assert result.inferences[0]["rule_name"] == "strong_avg_rank"
    assert [c["id"] for c in result.rag_chunks] == ["c1", "c2"]
    assert result.metadata["query_intent"] == "metric"
    assert "LANEIGE" in result.combined_context


async def test_retrieve_unified_greeting_skip(retriever: HybridRetriever) -> None:
    result = await retriever.retrieve_unified("안녕")
    assert result.retriever_type == "selfrag_skip"
    assert result.confidence == 0.0
    assert result.metadata == {
        "self_rag_skip": True,
        "skip_reason": "query_too_short",
        "selfrag_confidence": 0.0,
    }
    assert result.combined_context == "[Retrieval skipped: query_too_short]"


async def test_get_stats_after_retrieve(retriever: HybridRetriever) -> None:
    await retriever.retrieve("LANEIGE Lip Care SoS는?", current_metrics=CURRENT_METRICS)
    stats: dict[str, Any] = retriever.get_stats()
    assert set(stats) == {
        "knowledge_graph",
        "reasoner",
        "rules_count",
        "rag_metrics",
        "initialized",
    }
    assert stats["rules_count"] == 37
    assert stats["initialized"] is True
