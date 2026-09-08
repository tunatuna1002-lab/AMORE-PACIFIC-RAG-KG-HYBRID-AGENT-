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


class TestGoldSourceMetadata:
    """골드 수치의 검증 근거 층 (2026-09-06, 3단계).

    document / snapshot / domain_expectation을 구분해야 각 층에 맞는 채점을
    할 수 있다. 원자료로 검증할 수 없는 골드에 정답 일치를 요구하면 지표가
    문체 유사도를 재게 된다.
    """

    def test_defaults_keep_existing_items_valid(self):
        """기존 문항(필드 없음)은 document/None으로 읽혀야 한다."""
        meta = ItemMetadata()

        assert meta.gold_source == "document"
        assert meta.as_of is None

    def test_snapshot_items_carry_as_of(self):
        meta = ItemMetadata(gold_source="snapshot", as_of="2026-08-31")

        assert meta.gold_source == "snapshot"
        assert meta.as_of == "2026-08-31"

    def test_unknown_gold_source_is_rejected(self):
        import pytest
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            ItemMetadata(gold_source="guess")

    def test_golden_dataset_is_fully_classified(self):
        """172문항 전체에 gold_source가 있고, snapshot에는 as_of가 있어야 한다."""
        import json
        from pathlib import Path

        path = (
            Path(__file__).resolve().parents[2]
            / "eval"
            / "data"
            / "golden"
            / "laneige_golden_v2.jsonl"
        )
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

        assert rows, "골든셋이 비어 있다"
        for row in rows:
            meta = row["metadata"]
            assert meta.get("gold_source") in {
                "document",
                "snapshot",
                "domain_expectation",
            }, row["id"]
            if meta["gold_source"] == "snapshot":
                assert meta.get("as_of"), f"{row['id']}: snapshot인데 as_of가 없다"
            else:
                assert "as_of" not in meta, f"{row['id']}: {meta['gold_source']}인데 as_of가 있다"

    def test_snapshot_items_have_db_generated_values(self):
        """snapshot 문항은 DB에서 생성된 expected_values를 가져야 한다 (4단계).

        예외는 수치를 물어보지 않는 lg079(ASIN 조회)뿐이다. 이 문항은 해당 스냅샷의
        Top 100에 제품이 없다는 사실 자체가 답이라 수치 기대값이 없다.
        """
        import json
        from pathlib import Path

        path = (
            Path(__file__).resolve().parents[2]
            / "eval"
            / "data"
            / "golden"
            / "laneige_golden_v2.jsonl"
        )
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        no_numeric_answer = {"lg079"}

        missing = [
            row["id"]
            for row in rows
            if row["metadata"]["gold_source"] == "snapshot"
            and not row["gold"].get("expected_values")
            and row["id"] not in no_numeric_answer
        ]

        assert not missing, f"expected_values가 비어 있는 snapshot 문항: {missing}"
