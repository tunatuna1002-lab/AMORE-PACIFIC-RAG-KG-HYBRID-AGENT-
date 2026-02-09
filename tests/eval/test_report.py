"""Tests for report generator."""

import json
from datetime import datetime
from pathlib import Path

import pytest

from eval.report import ReportGenerator
from eval.schemas import (
    AnswerTrace,
    DocRetrievalTrace,
    EntityLinkingTrace,
    EvalConfig,
    EvalTrace,
    ItemResult,
    KGQueryTrace,
    L1Metrics,
    L2Metrics,
    L3Metrics,
    L4Metrics,
    L5Metrics,
    OntologyReasoningTrace,
)


class TestReportGenerator:
    """Tests for ReportGenerator."""

    @pytest.fixture
    def generator(self):
        """Create report generator."""
        return ReportGenerator()

    @pytest.fixture
    def sample_trace(self):
        """Create sample evaluation trace."""
        return EvalTrace(
            item_id="q001",
            timestamp=datetime.now(),
            l1_entity_linking=EntityLinkingTrace(
                extracted_brands=["laneige"],
                extracted_categories=["lip_care"],
                extracted_indicators=[],
                extracted_products=[],
            ),
            l2_doc_retrieval=DocRetrievalTrace(
                chunk_ids=["c1", "c2"],
                snippets=["snippet1", "snippet2"],
                scores=[0.9, 0.8],
            ),
            l3_kg_query=KGQueryTrace(
                kg_entities_found=["laneige"],
                kg_edges_found=["laneige -hasProduct-> B08XYZ"],
                ontology_facts=[],
                competitor_network=[],
            ),
            l4_ontology=OntologyReasoningTrace(
                inferences=[],
                applied_rules=[],
                constraint_violations=[],
            ),
            l5_answer=AnswerTrace(
                final_answer="LANEIGE is the top brand",
                citations=[],
                confidence=0.9,
            ),
            latency_ms=150.5,
            error=None,
        )

    @pytest.fixture
    def sample_results(self, sample_trace):
        """Create sample results."""
        return [
            ItemResult(
                item_id="q001",
                passed=True,
                l1=L1Metrics(entity_link_f1=1.0, concept_map_f1=1.0, constraint_extraction_f1=1.0),
                l2=L2Metrics(context_recall_at_k=1.0, context_precision_at_k=1.0, mrr=1.0),
                l3=L3Metrics(hits_at_k=1.0, kg_edge_f1=1.0),
                l4=L4Metrics(constraint_violation_rate=0.0, type_consistency_rate=1.0),
                l5=L5Metrics(
                    answer_exact_match=1.0,
                    answer_f1=1.0,
                    groundedness_score=0.9,
                    answer_relevance_score=0.95,
                ),
                overall_score=0.95,
                fail_reason_tags=[],
                trace=sample_trace,
            ),
            ItemResult(
                item_id="q002",
                passed=False,
                l1=L1Metrics(entity_link_f1=0.3, concept_map_f1=0.5, constraint_extraction_f1=0.5),
                l2=L2Metrics(context_recall_at_k=0.5, context_precision_at_k=0.5, mrr=0.5),
                l3=L3Metrics(hits_at_k=0.5, kg_edge_f1=0.5),
                l4=L4Metrics(constraint_violation_rate=0.1, type_consistency_rate=0.9),
                l5=L5Metrics(
                    answer_exact_match=0.0,
                    answer_f1=0.4,
                    groundedness_score=0.5,
                    answer_relevance_score=0.6,
                ),
                overall_score=0.45,
                fail_reason_tags=["L1_mapping_fail", "L5_wrong_answer"],
                trace=sample_trace,
            ),
        ]

    def test_compute_aggregates(self, generator, sample_results):
        """Test aggregate computation."""
        aggregates = generator._compute_aggregates(sample_results)

        assert aggregates.total == 2
        assert aggregates.passed == 1
        assert aggregates.failed == 1
        assert aggregates.pass_rate == 0.5
        assert 0.0 < aggregates.avg_overall_score < 1.0
        assert aggregates.avg_latency_ms > 0

    def test_compute_aggregates_empty(self, generator):
        """Test aggregate computation with empty results."""
        aggregates = generator._compute_aggregates([])

        assert aggregates.total == 0
        assert aggregates.passed == 0
        assert aggregates.pass_rate == 0.0

    def test_compute_layer_averages(self, generator, sample_results):
        """Test layer average computation."""
        by_layer = generator._compute_layer_averages(sample_results)

        assert "l1_entity_link_f1" in by_layer
        assert "l2_context_recall" in by_layer
        assert "l3_hits_at_k" in by_layer
        assert "l4_constraint_violation_rate" in by_layer
        assert "l5_answer_f1" in by_layer

    def test_compute_fail_reason_counts(self, generator, sample_results):
        """Test fail reason counting."""
        counts = generator._compute_fail_reason_counts(sample_results)

        assert "L1_mapping_fail" in counts
        assert "L5_wrong_answer" in counts
        assert counts["L1_mapping_fail"] == 1

    def test_generate_report(self, generator, sample_results, tmp_path: Path):
        """Test full report generation."""
        report = generator.generate_report(sample_results, tmp_path)

        # Check report structure
        assert report.timestamp is not None
        assert report.aggregates.total == 2
        assert len(report.items) == 2

        # Check files were created
        assert (tmp_path / "report.json").exists()
        assert (tmp_path / "summary.md").exists()

    def test_json_report_structure(self, generator, sample_results, tmp_path: Path):
        """Test JSON report structure."""
        generator.generate_report(sample_results, tmp_path)

        json_path = tmp_path / "report.json"
        with open(json_path) as f:
            data = json.load(f)

        assert "timestamp" in data
        assert "config" in data
        assert "aggregates" in data
        assert "items" in data
        assert len(data["items"]) == 2

    def test_markdown_summary_content(self, generator, sample_results, tmp_path: Path):
        """Test Markdown summary content."""
        generator.generate_report(sample_results, tmp_path)

        md_path = tmp_path / "summary.md"
        content = md_path.read_text()

        assert "# Evaluation Summary" in content
        assert "Pass Rate" in content
        assert "Layer Metrics" in content
        assert "Failure Reasons" in content

    def test_save_traces(self, generator, sample_results, tmp_path: Path):
        """Test trace saving."""
        generator.config.save_traces = True
        generator.generate_report(sample_results, tmp_path)

        traces_dir = tmp_path / "traces"
        assert traces_dir.exists()
        assert (traces_dir / "q001.json").exists()
        assert (traces_dir / "q002.json").exists()

    def test_recommendations(self, generator, sample_results, tmp_path: Path):
        """Test recommendations are included."""
        generator.generate_report(sample_results, tmp_path)

        md_path = tmp_path / "summary.md"
        content = md_path.read_text()

        assert "## Recommendations" in content

    def test_custom_config(self, sample_results, tmp_path: Path):
        """Test with custom config."""
        config = EvalConfig(top_k=5, use_judge=True, judge_model="gpt-4")
        generator = ReportGenerator(config=config)
        report = generator.generate_report(sample_results, tmp_path)

        assert report.config.top_k == 5
        assert report.config.use_judge is True

    def test_output_directory_creation(self, generator, sample_results, tmp_path: Path):
        """Test output directory is created."""
        out_dir = tmp_path / "nested" / "output" / "dir"
        generator.generate_report(sample_results, out_dir)

        assert out_dir.exists()
        assert (out_dir / "report.json").exists()


class TestReportContent:
    """Tests for report content quality."""

    @pytest.fixture
    def all_failing_results(self):
        """Create results where all items fail."""
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
                kg_entities_found=[], kg_edges_found=[], ontology_facts=[], competitor_network=[]
            ),
            l4_ontology=OntologyReasoningTrace(
                inferences=[], applied_rules=[], constraint_violations=[]
            ),
            l5_answer=AnswerTrace(final_answer="", citations=[], confidence=None),
            latency_ms=100.0,
            error=None,
        )

        return [
            ItemResult(
                item_id=f"q{i:03d}",
                passed=False,
                l1=L1Metrics(entity_link_f1=0.2, concept_map_f1=0.2, constraint_extraction_f1=0.2),
                l2=L2Metrics(context_recall_at_k=0.2, context_precision_at_k=0.2, mrr=0.2),
                l3=L3Metrics(hits_at_k=0.2, kg_edge_f1=0.2),
                l4=L4Metrics(constraint_violation_rate=0.5, type_consistency_rate=0.5),
                l5=L5Metrics(
                    answer_exact_match=0.0,
                    answer_f1=0.2,
                    groundedness_score=0.2,
                    answer_relevance_score=0.2,
                ),
                overall_score=0.2,
                fail_reason_tags=["L1_mapping_fail", "L5_grounding_fail"],
                trace=trace,
            )
            for i in range(5)
        ]

    def test_recommendations_for_failures(self, all_failing_results, tmp_path: Path):
        """Test recommendations are generated for failures."""
        generator = ReportGenerator()
        generator.generate_report(all_failing_results, tmp_path)

        md_path = tmp_path / "summary.md"
        content = md_path.read_text()

        # Should have recommendations for various issues
        assert "Recommendations" in content
        # Should have specific recommendations based on low scores
        assert "Improve" in content or "Consider" in content
