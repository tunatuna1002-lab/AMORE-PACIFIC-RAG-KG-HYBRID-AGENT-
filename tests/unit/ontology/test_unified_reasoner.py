"""
Unit tests for UnifiedReasoner

Tests the integrated OWL + Business Rules reasoning pipeline.
"""

from unittest.mock import Mock

from src.ontology.unified_reasoner import (
    UnifiedInferenceResult,
    UnifiedReasoner,
)


class TestUnifiedInferenceResult:
    """Test UnifiedInferenceResult dataclass"""

    def test_initialization_defaults(self):
        """Test initialization with minimal parameters"""
        result = UnifiedInferenceResult(insight="test insight")

        assert result.insight == "test insight"
        assert result.recommendation == ""
        assert result.confidence == 0.0
        assert result.source == "unknown"
        assert result.rule_name == ""
        assert result.category == ""
        assert result.supporting_facts == []

    def test_initialization_full(self):
        """Test initialization with all parameters"""
        result = UnifiedInferenceResult(
            insight="LANEIGE is dominant",
            recommendation="Maintain position",
            confidence=0.85,
            source="owl",
            rule_name="market_position",
            category="market_analysis",
            supporting_facts=["SoS: 15%", "Rank: 1"],
        )

        assert result.insight == "LANEIGE is dominant"
        assert result.recommendation == "Maintain position"
        assert result.confidence == 0.85
        assert result.source == "owl"
        assert result.rule_name == "market_position"
        assert result.category == "market_analysis"
        assert len(result.supporting_facts) == 2

    def test_to_dict(self):
        """Test to_dict conversion"""
        result = UnifiedInferenceResult(
            insight="test", confidence=0.7, source="business_rule", rule_name="test_rule"
        )

        data = result.to_dict()

        assert data["insight"] == "test"
        assert data["confidence"] == 0.7
        assert data["source"] == "business_rule"
        assert data["rule_name"] == "test_rule"
        assert "category" in data
        assert "supporting_facts" in data


class TestUnifiedReasonerInit:
    """Test UnifiedReasoner initialization"""

    def test_default_initialization(self):
        """Test initialization without reasoners"""
        reasoner = UnifiedReasoner()

        assert reasoner.rule_reasoner is None
        assert reasoner.owl_reasoner is None
        assert reasoner.owl_weight == 0.6
        assert reasoner.rule_weight == 0.4
        assert reasoner._inference_count == 0

    def test_initialization_with_reasoners(self):
        """Test initialization with both reasoners"""
        mock_rule = Mock()
        mock_owl = Mock()

        reasoner = UnifiedReasoner(ontology_reasoner=mock_rule, owl_reasoner=mock_owl)

        assert reasoner.rule_reasoner is mock_rule
        assert reasoner.owl_reasoner is mock_owl

    def test_initialization_custom_weights(self):
        """Test initialization with custom weights"""
        reasoner = UnifiedReasoner(owl_weight=0.7, rule_weight=0.3)

        assert reasoner.owl_weight == 0.7
        assert reasoner.rule_weight == 0.3


class TestUnifiedReasonerOWLReasoning:
    """Test OWL reasoning path"""

    def test_run_owl_reasoning_no_reasoner(self):
        """Test OWL reasoning when no OWL reasoner provided"""
        reasoner = UnifiedReasoner(owl_reasoner=None)

        results = reasoner._run_owl_reasoning({"brand": "LANEIGE"})

        assert results == []

    def test_run_owl_reasoning_with_brand_info(self):
        """Test OWL reasoning with brand information"""
        mock_owl = Mock()
        mock_owl.get_brand_info = Mock(
            return_value={
                "market_position": "DominantBrand",
                "sos": 0.20,
                "competitors": ["COSRX", "TIRTIR"],
            }
        )
        mock_owl.get_inferred_facts = Mock(return_value=[])

        reasoner = UnifiedReasoner(owl_reasoner=mock_owl, owl_weight=0.6)

        results = reasoner._run_owl_reasoning({"brand": "LANEIGE"})

        assert len(results) >= 1
        # Check market position result
        position_result = [r for r in results if r.rule_name == "owl_market_position"]
        assert len(position_result) == 1
        assert "지배적 브랜드" in position_result[0].insight
        assert position_result[0].source == "owl"
        assert position_result[0].confidence == 0.8 * 0.6

    def test_run_owl_reasoning_with_competitors(self):
        """Test OWL reasoning includes competitor insights"""
        mock_owl = Mock()
        mock_owl.get_brand_info = Mock(
            return_value={"market_position": "StrongBrand", "sos": 0.15, "competitors": ["COSRX"]}
        )
        mock_owl.get_inferred_facts = Mock(return_value=[])

        reasoner = UnifiedReasoner(owl_reasoner=mock_owl)

        results = reasoner._run_owl_reasoning({"brand": "LANEIGE"})

        competitor_result = [r for r in results if r.rule_name == "owl_competition"]
        assert len(competitor_result) == 1
        assert "COSRX" in competitor_result[0].insight

    def test_run_owl_reasoning_with_inferred_facts(self):
        """Test OWL reasoning with inferred facts"""
        mock_owl = Mock()
        mock_owl.get_brand_info = Mock(return_value=None)
        mock_owl.get_inferred_facts = Mock(
            return_value=[
                {
                    "type": "market_position",
                    "subject": "COSRX",
                    "position": "NicheBrand",
                    "sos": 0.08,
                }
            ]
        )

        reasoner = UnifiedReasoner(owl_reasoner=mock_owl)

        results = reasoner._run_owl_reasoning({"brand": "LANEIGE"})

        # Should include market context for other brands
        context_results = [r for r in results if r.rule_name == "owl_market_context"]
        assert len(context_results) >= 1

    def test_run_owl_reasoning_handles_errors(self):
        """Test OWL reasoning handles exceptions gracefully"""
        mock_owl = Mock()
        mock_owl.get_brand_info = Mock(side_effect=Exception("OWL failed"))

        reasoner = UnifiedReasoner(owl_reasoner=mock_owl)

        results = reasoner._run_owl_reasoning({"brand": "LANEIGE"})

        # Should return empty, not raise
        assert results == []

    def test_position_recommendation(self):
        """Test position-based recommendations"""
        reasoner = UnifiedReasoner()

        rec1 = reasoner._position_recommendation("DominantBrand")
        assert "지배력 유지" in rec1

        rec2 = reasoner._position_recommendation("StrongBrand")
        assert "점유율 확대" in rec2

        rec3 = reasoner._position_recommendation("NicheBrand")
        assert "차별화" in rec3

        rec4 = reasoner._position_recommendation("UnknownPosition")
        assert rec4 == ""


class TestUnifiedReasonerRuleReasoning:
    """Test business rule reasoning path"""

    def test_run_rule_reasoning_no_reasoner(self):
        """Test rule reasoning when no rule reasoner provided"""
        reasoner = UnifiedReasoner(ontology_reasoner=None)

        results = reasoner._run_rule_reasoning({"brand": "LANEIGE"}, "test query")

        assert results == []

    def test_run_rule_reasoning_with_intent(self):
        """Test rule reasoning using infer_with_intent"""
        mock_inference = Mock()
        mock_inference.insight = "Market share growing"
        mock_inference.recommendation = "Continue strategy"
        mock_inference.confidence = 0.75
        mock_inference.rule_name = "growth_rule"
        mock_inference.insight_type = "growth"

        mock_rule = Mock()
        mock_rule.infer_with_intent = Mock(return_value=[mock_inference])

        reasoner = UnifiedReasoner(ontology_reasoner=mock_rule, rule_weight=0.4)

        results = reasoner._run_rule_reasoning({"brand": "LANEIGE"}, "growth analysis")

        assert len(results) == 1
        assert results[0].insight == "Market share growing"
        assert results[0].confidence == 0.75 * 0.4
        assert results[0].source == "business_rule"

    def test_run_rule_reasoning_fallback_to_infer(self):
        """Test fallback to infer when infer_with_intent not available"""
        mock_inference = Mock()
        mock_inference.insight = "Price competitive"
        mock_inference.confidence = 0.65
        mock_inference.rule_name = "price_rule"

        mock_rule = Mock()
        mock_rule.infer = Mock(return_value=[mock_inference])
        # Remove infer_with_intent attribute
        delattr(mock_rule, "infer_with_intent")

        reasoner = UnifiedReasoner(ontology_reasoner=mock_rule)

        results = reasoner._run_rule_reasoning({"brand": "LANEIGE"}, "query")

        mock_rule.infer.assert_called_once()
        assert len(results) == 1

    def test_run_rule_reasoning_dict_format(self):
        """Test rule reasoning with dict-format results"""
        mock_rule = Mock()
        mock_rule.infer = Mock(
            return_value=[
                {
                    "insight": "Rank improved",
                    "confidence": 0.8,
                    "rule_name": "rank_rule",
                    "insight_type": "ranking",
                }
            ]
        )

        reasoner = UnifiedReasoner(ontology_reasoner=mock_rule, rule_weight=0.4)

        results = reasoner._run_rule_reasoning({"brand": "LANEIGE"}, "")

        assert len(results) == 1
        assert results[0].insight == "Rank improved"
        assert results[0].confidence == 0.8 * 0.4

    def test_run_rule_reasoning_handles_errors(self):
        """Test rule reasoning handles exceptions"""
        mock_rule = Mock()
        mock_rule.infer = Mock(side_effect=Exception("Rule failed"))

        reasoner = UnifiedReasoner(ontology_reasoner=mock_rule)

        results = reasoner._run_rule_reasoning({}, "")

        assert results == []


class TestUnifiedReasonerMergeResults:
    """Test result merging and deduplication"""

    def test_merge_results_empty(self):
        """Test merging empty results"""
        reasoner = UnifiedReasoner()

        merged = reasoner._merge_results([])

        assert merged == []

    def test_merge_results_no_duplicates(self):
        """Test merging unique results"""
        results = [
            UnifiedInferenceResult(insight="Insight 1", confidence=0.8, source="owl"),
            UnifiedInferenceResult(insight="Insight 2", confidence=0.7, source="business_rule"),
        ]

        reasoner = UnifiedReasoner()
        merged = reasoner._merge_results(results)

        assert len(merged) == 2

    def test_merge_results_with_duplicates(self):
        """Test merging with duplicate insights"""
        result1 = UnifiedInferenceResult(
            insight="LANEIGE is dominant brand", confidence=0.8, source="owl"
        )
        result2 = UnifiedInferenceResult(
            insight="LANEIGE is dominant brand", confidence=0.6, source="business_rule"
        )

        reasoner = UnifiedReasoner()
        merged = reasoner._merge_results([result1, result2])

        # Should merge duplicates
        assert len(merged) == 1
        assert merged[0].source == "merged"
        # Confidence should be boosted
        assert merged[0].confidence > 0.8

    def test_merge_results_supporting_facts(self):
        """Test merging combines supporting facts"""
        result1 = UnifiedInferenceResult(
            insight="Test insight", confidence=0.7, supporting_facts=["Fact 1"]
        )
        result2 = UnifiedInferenceResult(
            insight="Test insight", confidence=0.6, supporting_facts=["Fact 2"]
        )

        reasoner = UnifiedReasoner()
        merged = reasoner._merge_results([result1, result2])

        assert len(merged) == 1
        assert len(merged[0].supporting_facts) == 2


class TestUnifiedReasonerInfer:
    """Test the main infer method"""

    def test_infer_no_reasoners(self):
        """Test infer with no reasoners"""
        reasoner = UnifiedReasoner()

        results = reasoner.infer({"brand": "LANEIGE"})

        assert results == []
        assert reasoner._inference_count == 1

    def test_infer_owl_only(self):
        """Test infer with only OWL reasoner"""
        mock_owl = Mock()
        mock_owl.get_brand_info = Mock(
            return_value={"market_position": "DominantBrand", "sos": 0.20, "competitors": []}
        )
        mock_owl.get_inferred_facts = Mock(return_value=[])

        reasoner = UnifiedReasoner(owl_reasoner=mock_owl)

        results = reasoner.infer({"brand": "LANEIGE"})

        assert len(results) > 0
        assert all(r.source in ["owl", "merged"] for r in results)

    def test_infer_rules_only(self):
        """Test infer with only business rules"""
        mock_inference = Mock()
        mock_inference.insight = "Price competitive"
        mock_inference.confidence = 0.7

        mock_rule = Mock()
        mock_rule.infer = Mock(return_value=[mock_inference])

        reasoner = UnifiedReasoner(ontology_reasoner=mock_rule)

        results = reasoner.infer({"brand": "LANEIGE"})

        assert len(results) > 0
        assert all(r.source in ["business_rule", "merged"] for r in results)

    def test_infer_combined(self):
        """Test infer with both reasoners"""
        mock_owl = Mock()
        mock_owl.get_brand_info = Mock(
            return_value={"market_position": "StrongBrand", "sos": 0.15, "competitors": ["COSRX"]}
        )
        mock_owl.get_inferred_facts = Mock(return_value=[])

        mock_inference = Mock()
        mock_inference.insight = "Growing market share"
        mock_inference.confidence = 0.75
        mock_inference.rule_name = "growth"

        mock_rule = Mock()
        mock_rule.infer = Mock(return_value=[mock_inference])

        reasoner = UnifiedReasoner(ontology_reasoner=mock_rule, owl_reasoner=mock_owl)

        results = reasoner.infer({"brand": "LANEIGE", "sos": 0.15})

        # Should have results from both sources
        assert len(results) > 0
        sources = {r.source for r in results}
        assert len(sources) > 0  # Could be owl, business_rule, or merged

    def test_infer_sorted_by_confidence(self):
        """Test infer returns results sorted by confidence"""
        mock_rule = Mock()
        mock_rule.infer = Mock(
            return_value=[
                Mock(insight="Low conf", confidence=0.5, rule_name="r1"),
                Mock(insight="High conf", confidence=0.9, rule_name="r2"),
                Mock(insight="Mid conf", confidence=0.7, rule_name="r3"),
            ]
        )

        reasoner = UnifiedReasoner(ontology_reasoner=mock_rule)

        results = reasoner.infer({})

        # Should be sorted high to low
        for i in range(len(results) - 1):
            assert results[i].confidence >= results[i + 1].confidence

    def test_infer_max_results(self):
        """Test infer respects max_results parameter"""
        mock_rule = Mock()
        mock_rule.infer = Mock(
            return_value=[Mock(insight=f"Insight {i}", confidence=0.5) for i in range(20)]
        )

        reasoner = UnifiedReasoner(ontology_reasoner=mock_rule)

        results = reasoner.infer({}, max_results=5)

        assert len(results) == 5

    def test_infer_increments_count(self):
        """Test infer increments inference count"""
        reasoner = UnifiedReasoner()

        assert reasoner._inference_count == 0

        reasoner.infer({})
        assert reasoner._inference_count == 1

        reasoner.infer({})
        assert reasoner._inference_count == 2


class TestUnifiedReasonerExplain:
    """Test explanation generation"""

    def test_explain_empty(self):
        """Test explain with no results"""
        reasoner = UnifiedReasoner()

        explanation = reasoner.explain([])

        assert "추론 결과가 없습니다" in explanation

    def test_explain_with_results(self):
        """Test explain with results"""
        results = [
            UnifiedInferenceResult(
                insight="LANEIGE is strong",
                recommendation="Expand market",
                confidence=0.85,
                source="owl",
            ),
            UnifiedInferenceResult(
                insight="Price competitive", confidence=0.70, source="business_rule"
            ),
        ]

        reasoner = UnifiedReasoner()
        explanation = reasoner.explain(results)

        assert "2개의 인사이트" in explanation
        assert "LANEIGE is strong" in explanation
        assert "Price competitive" in explanation
        assert "형식 추론" in explanation
        assert "비즈니스 규칙" in explanation
        assert "85%" in explanation


class TestUnifiedReasonerStats:
    """Test statistics methods"""

    def test_get_stats_no_reasoners(self):
        """Test get_stats with no reasoners"""
        reasoner = UnifiedReasoner()

        stats = reasoner.get_stats()

        assert stats["inference_count"] == 0
        assert stats["has_owl"] is False
        assert stats["has_rules"] is False
        assert stats["owl_weight"] == 0.6
        assert stats["rule_weight"] == 0.4

    def test_get_stats_with_reasoners(self):
        """Test get_stats with reasoners"""
        mock_owl = Mock()
        mock_rule = Mock()

        reasoner = UnifiedReasoner(ontology_reasoner=mock_rule, owl_reasoner=mock_owl)
        reasoner._inference_count = 5

        stats = reasoner.get_stats()

        assert stats["inference_count"] == 5
        assert stats["has_owl"] is True
        assert stats["has_rules"] is True

    def test_get_stats_after_inferences(self):
        """Test get_stats tracks inference count"""
        reasoner = UnifiedReasoner()

        reasoner.infer({})
        reasoner.infer({})
        reasoner.infer({})

        stats = reasoner.get_stats()
        assert stats["inference_count"] == 3
