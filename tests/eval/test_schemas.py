"""Tests for evaluation schemas."""

from datetime import datetime

from eval.schemas import (
    AnswerTrace,
    DocRetrievalTrace,
    EntityLinkingTrace,
    EvalConfig,
    EvalItem,
    EvalTrace,
    GoldEvidence,
    ItemMetadata,
    KGQueryTrace,
    L1Metrics,
    L2Metrics,
    L3Metrics,
    L4Metrics,
    L5Metrics,
    OntologyReasoningTrace,
)


class TestGoldEvidence:
    """Tests for GoldEvidence schema."""

    def test_default_values(self):
        """Test default values are set correctly."""
        gold = GoldEvidence()
        assert gold.answer is None
        assert gold.doc_chunk_ids == []
        assert gold.kg_entities == []
        assert gold.kg_edges == []
        assert gold.concepts == []
        assert gold.constraints == []

    def test_with_values(self):
        """Test with provided values."""
        gold = GoldEvidence(
            answer="test answer",
            doc_chunk_ids=["chunk1", "chunk2"],
            kg_entities=["entity1"],
            kg_edges=["edge1"],
            concepts=["concept1"],
            constraints=["constraint1"],
        )
        assert gold.answer == "test answer"
        assert len(gold.doc_chunk_ids) == 2
        assert len(gold.kg_entities) == 1


class TestItemMetadata:
    """Tests for ItemMetadata schema."""

    def test_default_values(self):
        """Test default values."""
        meta = ItemMetadata()
        assert meta.requires_kg is True
        assert meta.domain == "general"
        assert meta.difficulty == "medium"

    def test_valid_domain_values(self):
        """Test valid domain values."""
        for domain in ["market", "brand", "product", "metric", "general"]:
            meta = ItemMetadata(domain=domain)
            assert meta.domain == domain

    def test_valid_difficulty_values(self):
        """Test valid difficulty values."""
        for difficulty in ["easy", "medium", "hard"]:
            meta = ItemMetadata(difficulty=difficulty)
            assert meta.difficulty == difficulty


class TestEvalItem:
    """Tests for EvalItem schema."""

    def test_minimal_item(self):
        """Test minimal valid item."""
        item = EvalItem(id="q001", question="Test question?")
        assert item.id == "q001"
        assert item.question == "Test question?"
        assert item.gold is not None
        assert item.metadata is not None

    def test_full_item(self):
        """Test full item with all fields."""
        item = EvalItem(
            id="q001",
            question="Test question?",
            gold=GoldEvidence(answer="test answer"),
            metadata=ItemMetadata(requires_kg=False, domain="metric"),
        )
        assert item.gold.answer == "test answer"
        assert item.metadata.requires_kg is False
        assert item.metadata.domain == "metric"

    def test_json_serialization(self):
        """Test JSON serialization."""
        item = EvalItem(id="q001", question="Test?")
        json_str = item.model_dump_json()
        assert "q001" in json_str
        assert "Test?" in json_str

    def test_json_deserialization(self):
        """Test JSON deserialization."""
        json_str = '{"id": "q002", "question": "Another test?"}'
        item = EvalItem.model_validate_json(json_str)
        assert item.id == "q002"
        assert item.question == "Another test?"


class TestTraceSchemas:
    """Tests for trace schemas."""

    def test_entity_linking_trace(self):
        """Test EntityLinkingTrace."""
        trace = EntityLinkingTrace(
            extracted_brands=["laneige"],
            extracted_categories=["lip_care"],
            extracted_indicators=["sos"],
            extracted_products=["B08XYZ"],
        )
        assert len(trace.extracted_brands) == 1
        assert trace.extracted_brands[0] == "laneige"

    def test_doc_retrieval_trace(self):
        """Test DocRetrievalTrace."""
        trace = DocRetrievalTrace(
            chunk_ids=["c1", "c2"],
            snippets=["snippet 1", "snippet 2"],
            scores=[0.9, 0.8],
        )
        assert len(trace.chunk_ids) == 2
        assert trace.scores[0] > trace.scores[1]

    def test_kg_query_trace(self):
        """Test KGQueryTrace."""
        trace = KGQueryTrace(
            kg_entities_found=["laneige", "cosrx"],
            kg_edges_found=["laneige -competesWith-> cosrx"],
            ontology_facts=[{"type": "brand", "entity": "laneige"}],
            competitor_network=[{"brand": "laneige", "competitors": ["cosrx"]}],
        )
        assert len(trace.kg_entities_found) == 2
        assert len(trace.kg_edges_found) == 1

    def test_ontology_reasoning_trace(self):
        """Test OntologyReasoningTrace."""
        trace = OntologyReasoningTrace(
            inferences=[{"rule_name": "market_dominance", "insight_type": "market_position"}],
            applied_rules=["market_dominance"],
            constraint_violations=[],
        )
        assert len(trace.inferences) == 1
        assert len(trace.applied_rules) == 1

    def test_answer_trace(self):
        """Test AnswerTrace."""
        trace = AnswerTrace(
            final_answer="LANEIGE is the top brand",
            citations=["source1", "source2"],
            confidence=0.95,
        )
        assert "LANEIGE" in trace.final_answer
        assert trace.confidence == 0.95

    def test_eval_trace_full(self):
        """Test full EvalTrace."""
        trace = EvalTrace(
            item_id="q001",
            timestamp=datetime.now(),
            l1_entity_linking=EntityLinkingTrace(
                extracted_brands=[],
                extracted_categories=[],
                extracted_indicators=[],
                extracted_products=[],
            ),
            l2_doc_retrieval=DocRetrievalTrace(chunk_ids=[], snippets=[], scores=[]),
            l3_kg_query=KGQueryTrace(
                kg_entities_found=[],
                kg_edges_found=[],
                ontology_facts=[],
                competitor_network=[],
            ),
            l4_ontology=OntologyReasoningTrace(
                inferences=[], applied_rules=[], constraint_violations=[]
            ),
            l5_answer=AnswerTrace(final_answer="", citations=[], confidence=None),
            latency_ms=150.5,
            error=None,
        )
        assert trace.item_id == "q001"
        assert trace.latency_ms == 150.5


class TestMetricSchemas:
    """Tests for metric schemas."""

    def test_l1_metrics(self):
        """Test L1Metrics."""
        metrics = L1Metrics(
            entity_link_f1=0.85,
            concept_map_f1=0.90,
            constraint_extraction_f1=0.75,
        )
        assert metrics.entity_link_f1 == 0.85

    def test_l2_metrics(self):
        """Test L2Metrics."""
        metrics = L2Metrics(
            context_recall_at_k=0.80,
            context_precision_at_k=0.75,
            mrr=0.90,
        )
        assert metrics.context_recall_at_k == 0.80

    def test_l3_metrics(self):
        """Test L3Metrics."""
        metrics = L3Metrics(
            hits_at_k=1.0,
            kg_edge_f1=0.85,
        )
        assert metrics.hits_at_k == 1.0

    def test_l4_metrics(self):
        """Test L4Metrics."""
        metrics = L4Metrics(
            constraint_violation_rate=0.02,
            type_consistency_rate=0.98,
        )
        assert metrics.constraint_violation_rate == 0.02

    def test_l5_metrics_without_judge(self):
        """Test L5Metrics without judge scores."""
        metrics = L5Metrics(
            answer_exact_match=1.0,
            answer_f1=0.95,
            groundedness_score=None,
            answer_relevance_score=None,
        )
        assert metrics.answer_exact_match == 1.0
        assert metrics.groundedness_score is None

    def test_l5_metrics_with_judge(self):
        """Test L5Metrics with judge scores."""
        metrics = L5Metrics(
            answer_exact_match=1.0,
            answer_f1=0.95,
            groundedness_score=0.85,
            answer_relevance_score=0.90,
        )
        assert metrics.groundedness_score == 0.85


class TestEvalConfig:
    """Tests for EvalConfig."""

    def test_default_config(self):
        """Test default configuration."""
        config = EvalConfig()
        assert config.top_k == 8
        assert config.use_judge is False
        assert config.judge_model == "gpt-4.1-mini"  # Has default value
        assert config.save_traces is False

    def test_custom_config(self):
        """Test custom configuration."""
        config = EvalConfig(
            top_k=5,
            use_judge=True,
            judge_model="gpt-4.1-mini",
            save_traces=True,
        )
        assert config.top_k == 5
        assert config.use_judge is True
        assert config.judge_model == "gpt-4.1-mini"
