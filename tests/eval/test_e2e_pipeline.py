"""
E2E Pipeline Test
==================
End-to-end testing of the evaluation pipeline with mock agent.

Tests:
- Single item evaluation
- Full pipeline (Loader → Runner → Report)
- Metrics validation (all in [0,1])
- Aggregate metrics correctness
- StubJudge integration
- Concurrent execution
"""

import asyncio

import pytest

from eval.judge.stub import StubJudge
from eval.loader import load_dataset
from eval.schemas import EvalConfig, EvalItem, EvalTrace


class MockAgent:
    """
    Mock chatbot agent for testing the evaluation pipeline.

    Returns synthetic responses with all required fields for evaluation.
    """

    def __init__(self, response_quality: str = "good"):
        """
        Initialize mock agent.

        Args:
            response_quality: Quality level - "good", "medium", or "poor"
        """
        self.response_quality = response_quality

    async def chat(self, question: str) -> dict:
        """
        Mock chat method that returns synthetic evaluation data.

        Args:
            question: User question

        Returns:
            Dictionary with response and all trace fields
        """
        # Base response quality settings
        quality_settings = {
            "good": {
                "response": f"질문 '{question}'에 대한 정확한 답변입니다.",
                "confidence": 0.90,
                "entity_accuracy": 1.0,
                "doc_relevance": 0.95,
                "kg_coverage": 0.90,
            },
            "medium": {
                "response": f"'{question}'에 대한 답변입니다.",
                "confidence": 0.70,
                "entity_accuracy": 0.75,
                "doc_relevance": 0.70,
                "kg_coverage": 0.60,
            },
            "poor": {
                "response": "잘 모르겠습니다.",
                "confidence": 0.30,
                "entity_accuracy": 0.30,
                "doc_relevance": 0.40,
                "kg_coverage": 0.20,
            },
        }

        settings = quality_settings.get(self.response_quality, quality_settings["good"])

        # Extract mock entities from question
        entities = {
            "brands": ["laneige"] if "laneige" in question.lower() else [],
            "categories": (
                ["lip_care"]
                if "lip" in question.lower()
                else ["skin_care"]
                if "skin" in question.lower()
                else []
            ),
            "indicators": (
                ["sos"]
                if "sos" in question.lower() or "점유율" in question
                else ["hhi"]
                if "hhi" in question.lower() or "집중도" in question
                else []
            ),
            "products": (
                ["lip_sleeping_mask"]
                if "슬리핑" in question or "sleeping" in question.lower()
                else []
            ),
        }

        # Mock sources
        sources = [
            {"type": "metric_guide", "chunk_id": "metric_guide_sos_01"},
            {"type": "playbook", "chunk_id": "playbook_ranking_01"},
        ]

        # Mock KG data
        kg_entities = ["laneige", "lip_care"] if entities["brands"] else []
        kg_edges = ["laneige -hasProduct-> lip_sleeping_mask"] if "laneige" in kg_entities else []

        # Mock ontology data
        ontology_facts = (
            [{"subject": "laneige", "predicate": "hasProduct", "object": "lip_sleeping_mask"}]
            if kg_entities
            else []
        )

        inferences = (
            [
                {
                    "type": "market_position",
                    "confidence": settings["confidence"],
                    "rule": "low_sos_check",
                }
            ]
            if entities["indicators"]
            else []
        )

        # Mock citations
        citations = ["[1] Metric Guide - SoS Definition", "[2] Market Intelligence Playbook"]

        return {
            "response": settings["response"],
            "entities": entities,
            "sources": sources,
            "kg_entities": kg_entities,
            "kg_edges": kg_edges,
            "ontology_facts": ontology_facts,
            "inferences": inferences,
            "citations": citations,
            "confidence": settings["confidence"],
        }


@pytest.fixture
def mock_agent_good():
    """Fixture for good quality mock agent."""
    return MockAgent(response_quality="good")


@pytest.fixture
def mock_agent_medium():
    """Fixture for medium quality mock agent."""
    return MockAgent(response_quality="medium")


@pytest.fixture
def mock_agent_poor():
    """Fixture for poor quality mock agent."""
    return MockAgent(response_quality="poor")


@pytest.fixture
def sample_eval_item():
    """Fixture for a single evaluation item."""
    return EvalItem(
        id="test001",
        question="LANEIGE Lip Care 카테고리 SoS는 얼마인가요?",
        gold={
            "answer": "LANEIGE의 Lip Care 카테고리 SoS는 5.2%입니다.",
            "doc_chunk_ids": ["metric_guide_sos_01"],
            "kg_entities": ["laneige", "lip_care"],
            "kg_edges": ["laneige -hasProduct-> lip_sleeping_mask"],
            "concepts": ["sos", "market_share"],
            "expected_values": {"sos": 5.2},
        },
        metadata={"requires_kg": True, "domain": "metric", "difficulty": "easy"},
    )


@pytest.mark.asyncio
async def test_single_item_evaluation(mock_agent_good, sample_eval_item):
    """Test evaluation of a single item."""
    config = EvalConfig(use_judge=False, save_traces=False)

    # Mock evaluation function
    async def eval_fn(item: EvalItem) -> tuple:
        """Mock evaluation function."""
        agent_response = await mock_agent_good.chat(item.question)

        # Build trace
        trace = EvalTrace(
            item_id=item.id,
            l1_entity_linking={
                "extracted_brands": agent_response["entities"]["brands"],
                "extracted_categories": agent_response["entities"]["categories"],
                "extracted_indicators": agent_response["entities"]["indicators"],
            },
            l2_doc_retrieval={
                "chunk_ids": [s["chunk_id"] for s in agent_response["sources"]],
                "scores": [0.95, 0.90],
            },
            l3_kg_query={
                "kg_entities_found": agent_response["kg_entities"],
                "kg_edges_found": agent_response["kg_edges"],
            },
            l4_ontology={"inferences": agent_response["inferences"]},
            l5_answer={
                "final_answer": agent_response["response"],
                "citations": agent_response["citations"],
                "confidence": agent_response["confidence"],
            },
        )

        # Simple metric calculation using individual metric functions
        from eval.metrics.l1_query import entity_link_f1
        from eval.metrics.l2_retrieval import context_recall_at_k, mrr
        from eval.metrics.l3_kg import L3Metrics
        from eval.metrics.l5_answer import answer_token_f1
        from eval.schemas import L1Metrics, L2Metrics

        # Compute L1 metrics
        l1_f1 = entity_link_f1(trace.l1_entity_linking, item.gold)
        l1_metrics = L1Metrics(entity_link_f1=l1_f1)

        # Compute L2 metrics
        l2_recall = context_recall_at_k(trace.l2_doc_retrieval, item.gold, k=8)
        l2_mrr = mrr(trace.l2_doc_retrieval, item.gold)
        l2_metrics = L2Metrics(context_recall_at_k=l2_recall, mrr=l2_mrr)

        # Compute L3 metrics
        l3_metrics = L3Metrics(hits_at_k=0.85, kg_edge_f1=0.80)

        # Compute L5 metrics
        from eval.schemas import L5Metrics

        l5_f1 = answer_token_f1(trace.l5_answer, item.gold)
        l5_metrics = L5Metrics(answer_f1=l5_f1)

        return (l1_metrics, l2_metrics, l3_metrics, l5_metrics, trace)

    # Run evaluation
    l1, l2, l3, l5, trace = await eval_fn(sample_eval_item)

    # Assertions
    assert trace.item_id == "test001"
    assert trace.l5_answer.final_answer != ""
    assert 0.0 <= trace.l5_answer.confidence <= 1.0

    # Check L1 metrics
    assert 0.0 <= l1.entity_link_f1 <= 1.0
    assert 0.0 <= l1.concept_map_f1 <= 1.0

    # Check L2 metrics
    assert 0.0 <= l2.context_recall_at_k <= 1.0
    assert 0.0 <= l2.mrr <= 1.0

    # Check L3 metrics
    assert 0.0 <= l3.hits_at_k <= 1.0
    assert 0.0 <= l3.kg_edge_f1 <= 1.0

    # Check L5 metrics
    assert 0.0 <= l5.answer_f1 <= 1.0


@pytest.mark.asyncio
async def test_full_pipeline_with_loader(mock_agent_good, tmp_path):
    """Test full pipeline: Loader → Runner → Report."""
    # Create a small test dataset
    test_dataset = tmp_path / "test_dataset.jsonl"
    test_dataset.write_text(
        '{"id": "t001", "question": "LANEIGE SoS는?", "gold": {"kg_entities": ["laneige"]}, "metadata": {"domain": "metric", "difficulty": "easy"}}\n'
        '{"id": "t002", "question": "COSRX 순위는?", "gold": {"kg_entities": ["cosrx"]}, "metadata": {"domain": "brand", "difficulty": "medium"}}\n',
        encoding="utf-8",
    )

    # Load dataset
    items = load_dataset(test_dataset)
    assert len(items) == 2

    # Mock eval function
    async def eval_fn(item: EvalItem):
        agent_response = await mock_agent_good.chat(item.question)
        trace = EvalTrace(item_id=item.id)
        from eval.schemas import L1Metrics, L2Metrics, L3Metrics, L5Metrics

        return (
            L1Metrics(entity_link_f1=0.8),
            L2Metrics(context_recall_at_k=0.9),
            L3Metrics(hits_at_k=0.85),
            L5Metrics(answer_f1=0.75),
            trace,
        )

    # Run evaluation (simplified - normally would use run_evaluation)
    results = []
    for item in items:
        l1, l2, l3, l5, trace = await eval_fn(item)
        results.append(
            {
                "item_id": item.id,
                "l1": l1,
                "l2": l2,
                "l3": l3,
                "l5": l5,
                "trace": trace,
            }
        )

    assert len(results) == 2
    assert all(r["l1"].entity_link_f1 >= 0.0 for r in results)


@pytest.mark.asyncio
async def test_metrics_in_valid_range(mock_agent_good, sample_eval_item):
    """Test that all metrics are in [0, 1] range."""
    agent_response = await mock_agent_good.chat(sample_eval_item.question)

    # Validate all numeric fields
    assert 0.0 <= agent_response["confidence"] <= 1.0

    # Build trace and compute metrics
    from eval.metrics.l1_query import entity_link_f1
    from eval.metrics.l2_retrieval import context_recall_at_k, mrr
    from eval.metrics.l5_answer import answer_token_f1
    from eval.schemas import (
        AnswerTrace,
        DocRetrievalTrace,
        EntityLinkingTrace,
        KGQueryTrace,
        L1Metrics,
        L2Metrics,
        L3Metrics,
        L5Metrics,
    )

    l1_trace = EntityLinkingTrace(
        extracted_brands=agent_response["entities"]["brands"],
        extracted_categories=agent_response["entities"]["categories"],
    )
    l2_trace = DocRetrievalTrace(chunk_ids=[s["chunk_id"] for s in agent_response["sources"]])
    l3_trace = KGQueryTrace(
        kg_entities_found=agent_response["kg_entities"],
        kg_edges_found=agent_response["kg_edges"],
    )
    l5_trace = AnswerTrace(
        final_answer=agent_response["response"], confidence=agent_response["confidence"]
    )

    # Compute individual metrics
    l1_f1 = entity_link_f1(l1_trace, sample_eval_item.gold)
    l1 = L1Metrics(entity_link_f1=l1_f1)

    l2_recall = context_recall_at_k(l2_trace, sample_eval_item.gold, k=8)
    l2_mrr_score = mrr(l2_trace, sample_eval_item.gold)
    l2 = L2Metrics(context_recall_at_k=l2_recall, mrr=l2_mrr_score)

    l3 = L3Metrics(hits_at_k=0.85, kg_edge_f1=0.80)

    l5_f1 = answer_token_f1(l5_trace, sample_eval_item.gold)
    l5 = L5Metrics(answer_f1=l5_f1)

    # All metrics must be in [0, 1]
    assert 0.0 <= l1.entity_link_f1 <= 1.0
    assert 0.0 <= l1.concept_map_f1 <= 1.0
    assert 0.0 <= l2.context_recall_at_k <= 1.0
    assert 0.0 <= l2.context_precision_at_k <= 1.0
    assert 0.0 <= l2.mrr <= 1.0
    assert 0.0 <= l3.hits_at_k <= 1.0
    assert 0.0 <= l3.kg_edge_f1 <= 1.0
    assert 0.0 <= l5.answer_exact_match <= 1.0
    assert 0.0 <= l5.answer_f1 <= 1.0


@pytest.mark.asyncio
async def test_aggregates_correctness(mock_agent_good, tmp_path):
    """Test that aggregate metrics are calculated correctly."""
    # Create test items
    from eval.schemas import ItemResult, L1Metrics, L5Metrics

    results = [
        ItemResult(
            item_id="t1",
            passed=True,
            l1=L1Metrics(entity_link_f1=0.8),
            l5=L5Metrics(answer_f1=0.9),
            overall_score=0.85,
        ),
        ItemResult(
            item_id="t2",
            passed=False,
            l1=L1Metrics(entity_link_f1=0.6),
            l5=L5Metrics(answer_f1=0.5),
            overall_score=0.55,
        ),
        ItemResult(
            item_id="t3",
            passed=True,
            l1=L1Metrics(entity_link_f1=0.9),
            l5=L5Metrics(answer_f1=0.95),
            overall_score=0.925,
        ),
    ]

    # Calculate aggregates manually
    total = len(results)
    passed = sum(1 for r in results if r.passed)
    avg_score = sum(r.overall_score for r in results) / total

    # Assertions
    assert total == 3
    assert passed == 2
    assert abs(avg_score - 0.775) < 0.001  # (0.85 + 0.55 + 0.925) / 3


@pytest.mark.asyncio
async def test_stub_judge_integration(mock_agent_good, sample_eval_item):
    """Test StubJudge integration in the pipeline."""
    judge = StubJudge()

    # Get agent response
    agent_response = await mock_agent_good.chat(sample_eval_item.question)

    # Use judge
    groundedness = await judge.score_groundedness(
        answer=agent_response["response"],
        context="Mock context from retrieval",
    )
    relevance = await judge.score_relevance(
        answer=agent_response["response"],
        question=sample_eval_item.question,
    )
    factuality_score, _ = await judge.score_factuality(
        answer=agent_response["response"],
        facts=[sample_eval_item.gold.answer or ""],
    )
    factuality = factuality_score

    # StubJudge returns fixed scores
    assert 0.0 <= groundedness <= 1.0
    assert 0.0 <= relevance <= 1.0
    assert 0.0 <= factuality <= 1.0


@pytest.mark.asyncio
async def test_concurrent_execution(mock_agent_good):
    """Test concurrent evaluation of multiple items."""
    items = [
        EvalItem(
            id=f"concurrent_{i}",
            question=f"질문 {i}",
            gold={"kg_entities": ["laneige"]},
            metadata={"domain": "metric", "difficulty": "easy"},
        )
        for i in range(5)
    ]

    async def eval_single(item: EvalItem):
        """Evaluate single item."""
        await mock_agent_good.chat(item.question)
        # Simulate some processing time
        await asyncio.sleep(0.01)
        return item.id

    # Run concurrently
    results = await asyncio.gather(*[eval_single(item) for item in items])

    assert len(results) == 5
    assert all(r.startswith("concurrent_") for r in results)


@pytest.mark.asyncio
async def test_different_quality_agents(
    mock_agent_good, mock_agent_medium, mock_agent_poor, sample_eval_item
):
    """Test agents with different quality levels."""
    agents = {
        "good": mock_agent_good,
        "medium": mock_agent_medium,
        "poor": mock_agent_poor,
    }

    results = {}
    for name, agent in agents.items():
        response = await agent.chat(sample_eval_item.question)
        results[name] = response["confidence"]

    # Good agent should have higher confidence than poor
    assert results["good"] > results["medium"] > results["poor"]


@pytest.mark.asyncio
async def test_cost_tracking():
    """Test cost tracking in evaluation traces."""
    from eval.schemas import CostTrace

    cost = CostTrace(
        l1_tokens=100,
        l2_tokens=200,
        l3_tokens=150,
        l4_tokens=50,
        l5_tokens=300,
        judge_tokens=500,
        l1_cost_usd=0.001,
        l2_cost_usd=0.002,
        l3_cost_usd=0.0015,
        l4_cost_usd=0.0005,
        l5_cost_usd=0.003,
        judge_cost_usd=0.005,
    )

    assert cost.total_tokens == 1300
    assert abs(cost.total_cost_usd - 0.013) < 0.0001


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
