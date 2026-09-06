"""HybridRetriever fixes: D2(a) SoS unit boundary, D14 id-less BM25 items, D25 metadata,
D17(b) sentiment cluster shape normalisation.

Same real-KG + real-reasoner + fake DocumentRetriever setup as the characterization suite.
"""

from __future__ import annotations

import pytest

from src.domain.entities.relations import Relation, RelationType
from src.ontology.knowledge_graph import KnowledgeGraph
from src.ontology.reasoner import OntologyReasoner
from src.ontology.rules import register_all_rules
from src.rag.hybrid_retriever import HybridRetriever
from tests.characterization.conftest import CANNED_CHUNKS, FakeDocRetriever, seed_kg


@pytest.fixture
def kg(tmp_path) -> KnowledgeGraph:
    graph = KnowledgeGraph(persist_path=str(tmp_path / "kg.json"), auto_load=False, auto_save=False)
    return seed_kg(graph)


@pytest.fixture
def reasoner(kg) -> OntologyReasoner:
    r = OntologyReasoner(kg)
    register_all_rules(r)
    return r


def make_metrics(share_of_shelf_percent: float, hhi: float = 0.10) -> dict:
    return {
        "brand_metrics": [
            {
                "brand_name": "LANEIGE",
                "is_laneige": True,
                "share_of_shelf": share_of_shelf_percent,  # PERCENT (stored contract)
                "avg_rank": 30.0,
                "product_count": 3,
                "category_id": "lip_care",
            }
        ],
        "market_metrics": [{"category_id": "lip_care", "hhi": hhi, "cpi": 100.0}],
        "alerts": [],
    }


# --- D2(a) -----------------------------------------------------------------------


async def test_percent_sos_does_not_fire_market_dominance(kg, reasoner) -> None:
    hr = HybridRetriever(knowledge_graph=kg, reasoner=reasoner, doc_retriever=FakeDocRetriever())
    ctx = await hr.retrieve("LANEIGE Lip Care SoS는?", current_metrics=make_metrics(2.0))

    rules = {i.rule_name for i in ctx.inferences}
    assert not any(r.startswith("market_dominance") for r in rules)
    # 2% share in a fragmented market is an entry opportunity, not dominance.
    assert "category_entry_opportunity" in rules


async def test_percent_sos_above_threshold_fires_dominance(kg, reasoner) -> None:
    hr = HybridRetriever(knowledge_graph=kg, reasoner=reasoner, doc_retriever=FakeDocRetriever())
    ctx = await hr.retrieve("LANEIGE Lip Care SoS는?", current_metrics=make_metrics(18.0))
    assert "market_dominance_fragmented" in {i.rule_name for i in ctx.inferences}


def test_build_inference_context_converts_percent_to_fraction(kg, reasoner) -> None:
    hr = HybridRetriever(knowledge_graph=kg, reasoner=reasoner, doc_retriever=FakeDocRetriever())
    entities = {"brands": ["laneige"], "categories": ["lip_care"]}
    context = hr._build_inference_context(entities, make_metrics(3.0))
    assert context["sos"] == pytest.approx(0.03)

    # summary.laneige_sos_by_category is derived from brand_metrics.share_of_shelf
    # (metrics_agent) and is therefore percent as well.
    summary_only = {"summary": {"laneige_sos_by_category": {"lip_care": 3.0}}}
    context = hr._build_inference_context(entities, summary_only)
    assert context["sos"] == pytest.approx(0.03)


# --- D14 -------------------------------------------------------------------------


async def test_idless_bm25_items_with_short_filtered_results_survive(kg, reasoner) -> None:
    fake = FakeDocRetriever(
        chunks=CANNED_CHUNKS[:1],
        bm25_results=[{"content": "bm25 only chunk", "score": 0.5}],  # no "id"
        with_rrf=True,
    )
    hr = HybridRetriever(knowledge_graph=kg, reasoner=reasoner, doc_retriever=fake)

    ctx = await hr.retrieve("LANEIGE SoS 지표 알려줘", current_metrics=make_metrics(12.0))

    assert "error" not in ctx.metadata
    assert ctx.metadata["search_method"] == "hybrid_rrf"
    assert [c.get("id") for c in ctx.rag_chunks] == ["c1", None]
    assert ctx.metadata["rag_chunks_count"] == 2
    assert "bm25 only chunk" in ctx.combined_context
    # filtered search then the unfiltered top-up
    assert [(top_k, flt) for _q, top_k, flt in fake.calls] == [
        (5, ["metric_guide", "playbook"]),
        (3, None),
    ]


# --- D25 -------------------------------------------------------------------------


async def test_metadata_keeps_fusion_and_weighted_scores(kg, reasoner) -> None:
    hr = HybridRetriever(knowledge_graph=kg, reasoner=reasoner, doc_retriever=FakeDocRetriever())
    ctx = await hr.retrieve("LANEIGE Lip Care SoS는?", current_metrics=make_metrics(12.0))

    assert "fusion" in ctx.metadata
    assert "weighted_scores" in ctx.metadata
    assert "confidence" in ctx.metadata["fusion"]
    # the retrieve()-level keys are still present
    assert ctx.metadata["query_intent"] == "metric"
    assert ctx.metadata["search_method"] == "dense_only"


# --- D17(b) ----------------------------------------------------------------------


def test_sentiment_cluster_counts_reach_rules_and_fire(kg, reasoner) -> None:
    for tag in ("Moisturizing", "Hydrating"):
        kg.add_relation(
            Relation(
                "B0LANE1", RelationType.HAS_SENTIMENT, tag, properties={"cluster": "Hydration"}
            )
        )
    hr = HybridRetriever(knowledge_graph=kg, reasoner=reasoner, doc_retriever=FakeDocRetriever())

    entities = {"brands": ["laneige"], "sentiment_clusters": ["Hydration"]}
    context = hr._build_inference_context(entities, {})

    # KG brand profile yields counts, not tag lists
    assert context["sentiment_clusters"] == {"Hydration": 2}
    assert "sentiment_strength_hydration" in {r.rule_name for r in reasoner.infer(context)}


def test_sentiment_clusters_list_shape_is_normalised_to_dict(kg, reasoner) -> None:
    hr = HybridRetriever(knowledge_graph=kg, reasoner=reasoner, doc_retriever=FakeDocRetriever())
    assert hr._normalize_sentiment_clusters(["Hydration", "Sensory"]) == {
        "Hydration": 1,
        "Sensory": 1,
    }
    assert hr._normalize_sentiment_clusters({"Hydration": ["a", "b"]}) == {"Hydration": ["a", "b"]}
    assert hr._normalize_sentiment_clusters({"Hydration": 2}) == {"Hydration": 2}
    assert hr._normalize_sentiment_clusters(None) == {}
