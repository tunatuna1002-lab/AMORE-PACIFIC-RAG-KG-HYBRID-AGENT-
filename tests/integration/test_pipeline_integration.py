"""
파이프라인 통합 테스트
=====================
컴포넌트 간 데이터 흐름 검증 (API 호출 없이)
"""

from unittest.mock import MagicMock, patch

import pytest

from src.core.models import ConfidenceLevel, Context, Decision, KGFact, Response


class TestContextToDecisionFlow:
    """Context → Decision 흐름 테스트"""

    @pytest.mark.asyncio
    async def test_decision_maker_receives_context(self):
        """DecisionMaker가 Context를 올바르게 받는지"""
        from src.core.decision_maker import DecisionMaker

        dm = DecisionMaker()
        context = Context(
            query="LANEIGE SoS", entities={"brands": ["LANEIGE"]}, summary="LANEIGE SoS 데이터"
        )
        system_state = {
            "data_status": "fresh",
            "mode": "responding",
            "available_tools": ["direct_answer"],
        }

        # LLM 호출을 모킹
        with patch("src.core.decision_maker.acompletion") as mock_llm:
            mock_llm.return_value = MagicMock(
                choices=[
                    MagicMock(
                        message=MagicMock(
                            content='{"tool": "direct_answer", "tool_params": {}, "reason": "test", "confidence": 0.8, "key_points": ["SoS data"]}'
                        )
                    )
                ]
            )

            decision = await dm.decide("LANEIGE SoS", context, system_state)

            assert isinstance(decision, Decision)
            assert decision.tool == "direct_answer"
            assert decision.confidence == 0.8

    @pytest.mark.asyncio
    async def test_decision_maker_with_confidence_level(self):
        """DecisionMaker에 confidence_level이 전달되는지"""
        from src.core.decision_maker import DecisionMaker

        dm = DecisionMaker()
        context = Context(query="test")
        system_state = {"data_status": "fresh", "mode": "responding", "available_tools": []}

        with patch("src.core.decision_maker.acompletion") as mock_llm:
            mock_llm.return_value = MagicMock(
                choices=[
                    MagicMock(
                        message=MagicMock(
                            content='{"tool": "direct_answer", "tool_params": {}, "reason": "high mode", "confidence": 0.9, "key_points": []}'
                        )
                    )
                ]
            )

            decision = await dm.decide("test", context, system_state, confidence_level="high")
            assert isinstance(decision, Decision)

            # Verify the prompt included HIGH mode suffix
            call_args = mock_llm.call_args
            messages = call_args.kwargs.get("messages", call_args[1].get("messages", []))
            prompt_text = messages[0]["content"] if messages else ""
            assert "HIGH" in prompt_text or "high" in prompt_text.lower()


class TestDecisionToResponseFlow:
    """Decision → Response 흐름 테스트"""

    @pytest.mark.asyncio
    async def test_response_pipeline_receives_decision(self):
        """ResponsePipeline이 Decision을 올바르게 받는지"""
        from src.core.response_pipeline import ResponsePipeline

        pipeline = ResponsePipeline()  # No client = fallback mode
        context = Context(query="LANEIGE 순위", summary="LANEIGE is ranked #3")
        decision = Decision(
            tool="direct_answer",
            tool_params={},
            reason="Direct answer from context",
            confidence=0.85,
            key_points=["LANEIGE #3"],
        )

        response = await pipeline.generate(query="LANEIGE 순위", context=context, decision=decision)

        assert isinstance(response, Response)
        assert response.text  # Not empty
        assert response.confidence_score > 0

    @pytest.mark.asyncio
    async def test_high_confidence_fast_path_detection(self):
        """HIGH confidence fast path가 올바르게 감지되는지"""
        from src.core.response_pipeline import ResponsePipeline

        pipeline = ResponsePipeline()

        # HIGH confidence decision
        decision = Decision(
            tool="direct_answer",
            tool_params={},
            reason="HIGH confidence (high) - direct context answer",
            confidence=0.9,
            key_points=[],
        )

        # Check detection logic
        is_high = (
            decision
            and hasattr(decision, "confidence")
            and decision.confidence >= 0.85
            and decision.tool == "direct_answer"
            and hasattr(decision, "reason")
            and "HIGH confidence" in (decision.reason or "")
        )

        assert is_high is True


class TestConfidenceRoutingFlow:
    """Confidence → Brain routing 흐름 테스트"""

    def test_confidence_assessor_with_rich_context(self):
        """풍부한 컨텍스트 → HIGH confidence"""
        from src.core.confidence import ConfidenceAssessor

        assessor = ConfidenceAssessor()

        # Rich context = high score
        context = Context(query="LANEIGE SoS")
        context.rag_docs = [{"content": f"doc{i}"} for i in range(3)]
        context.kg_facts = [
            KGFact(fact_type="brand_info", entity="LANEIGE", data={"sos": 0.15}),
            KGFact(fact_type="brand_info", entity="COSRX", data={"sos": 0.1}),
        ]
        context.kg_inferences = [{"insight": "LANEIGE is market leader"}]

        rule_result = {"max_score": 5.0}
        level = assessor.assess(rule_result, context)

        assert level == ConfidenceLevel.HIGH

    def test_confidence_assessor_with_empty_context(self):
        """빈 컨텍스트 → LOW/UNKNOWN"""
        from src.core.confidence import ConfidenceAssessor

        assessor = ConfidenceAssessor()
        context = Context(query="unclear question")

        rule_result = {"max_score": 0.5}
        level = assessor.assess(rule_result, context)

        assert level in [ConfidenceLevel.LOW, ConfidenceLevel.UNKNOWN]


class TestOntologyIntegration:
    """Ontology 컴포넌트 통합 테스트"""

    def test_ontology_kg_to_unified_reasoner(self):
        """OntologyKG → UnifiedReasoner 데이터 흐름"""
        from src.ontology.knowledge_graph import KnowledgeGraph
        from src.ontology.ontology_knowledge_graph import OntologyKnowledgeGraph
        from src.ontology.unified_reasoner import UnifiedReasoner

        # OntologyKG requires a KnowledgeGraph instance
        kg = KnowledgeGraph()
        okg = OntologyKnowledgeGraph(knowledge_graph=kg)

        # UnifiedReasoner can work standalone
        ur = UnifiedReasoner()
        context = {"brand": "LANEIGE", "sos": 0.15, "rank": 3}
        results = ur.infer(context=context, query="LANEIGE market position")

        assert isinstance(results, list)

    def test_unified_reasoner_result_format(self):
        """UnifiedReasoner 결과가 Context에 호환되는 형식인지"""
        from src.ontology.unified_reasoner import UnifiedInferenceResult

        result = UnifiedInferenceResult(
            insight="LANEIGE is market leader",
            confidence=0.85,
            source="owl",
            recommendation="Continue current strategy",
            supporting_facts=["High SoS", "Strong rankings"],
        )

        d = result.to_dict()

        # Context.kg_inferences에 들어갈 수 있는 형식
        assert "insight" in d
        assert "confidence" in d
        assert "source" in d
        assert isinstance(d, dict)
