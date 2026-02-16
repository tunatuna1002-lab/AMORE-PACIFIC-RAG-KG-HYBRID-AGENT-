"""
ExplainabilityEngine 단위 테스트
"""

from src.core.explainability import ExplainabilityEngine, ExplanationTrace
from src.core.models import ConfidenceLevel, Context, Decision, Response
from src.domain.entities.brain_models import KGFact, SystemState

# =============================================================================
# ExplanationTrace 테스트
# =============================================================================


class TestExplanationTraceDefaults:
    """ExplanationTrace 기본값 테스트"""

    def test_default_values(self):
        """기본값으로 생성 가능"""
        trace = ExplanationTrace()
        assert trace.sources_used == []
        assert trace.rag_doc_count == 0
        assert trace.kg_fact_count == 0
        assert trace.kg_inference_count == 0
        assert trace.confidence_level == "unknown"
        assert trace.confidence_score == 0.0
        assert trace.decision_tool == ""
        assert trace.decision_reason == ""
        assert trace.routing_path == ""
        assert trace.key_evidence == []
        assert trace.processing_time_ms == 0.0


class TestExplanationTraceToDict:
    """ExplanationTrace.to_dict 테스트"""

    def test_to_dict_returns_all_fields(self):
        """to_dict가 모든 필드를 포함"""
        trace = ExplanationTrace(
            sources_used=["RAG", "KG"],
            rag_doc_count=3,
            kg_fact_count=5,
            kg_inference_count=2,
            confidence_level="high",
            confidence_score=0.9,
            decision_tool="query_data",
            decision_reason="데이터 조회 필요",
            routing_path="HIGH -> 직접 응답",
            key_evidence=["evidence1"],
            processing_time_ms=150.0,
        )
        d = trace.to_dict()
        assert d["sources_used"] == ["RAG", "KG"]
        assert d["rag_doc_count"] == 3
        assert d["kg_fact_count"] == 5
        assert d["kg_inference_count"] == 2
        assert d["confidence_level"] == "high"
        assert d["confidence_score"] == 0.9
        assert d["decision_tool"] == "query_data"
        assert d["decision_reason"] == "데이터 조회 필요"
        assert d["routing_path"] == "HIGH -> 직접 응답"
        assert d["key_evidence"] == ["evidence1"]
        assert d["processing_time_ms"] == 150.0

    def test_to_dict_empty_trace(self):
        """빈 trace의 to_dict"""
        trace = ExplanationTrace()
        d = trace.to_dict()
        assert isinstance(d, dict)
        assert len(d) == 11


class TestExplanationTraceHumanReadable:
    """ExplanationTrace.to_human_readable 테스트"""

    def test_human_readable_with_sources(self):
        """소스 정보 포함 시 출력"""
        trace = ExplanationTrace(sources_used=["RAG", "KG"])
        result = trace.to_human_readable()
        assert "RAG" in result
        assert "KG" in result

    def test_human_readable_with_rag_docs(self):
        """RAG 문서 수 포함"""
        trace = ExplanationTrace(rag_doc_count=5)
        result = trace.to_human_readable()
        assert "5" in result

    def test_human_readable_with_kg_facts(self):
        """KG 사실 수 포함"""
        trace = ExplanationTrace(kg_fact_count=3)
        result = trace.to_human_readable()
        assert "3" in result

    def test_human_readable_with_kg_inferences(self):
        """KG 추론 수 포함"""
        trace = ExplanationTrace(kg_inference_count=2)
        result = trace.to_human_readable()
        assert "2" in result

    def test_human_readable_confidence_high(self):
        """높은 신뢰도 표시"""
        trace = ExplanationTrace(confidence_level="high", confidence_score=0.95)
        result = trace.to_human_readable()
        assert "HIGH" in result

    def test_human_readable_confidence_medium(self):
        """중간 신뢰도 표시"""
        trace = ExplanationTrace(confidence_level="medium", confidence_score=0.7)
        result = trace.to_human_readable()
        assert "MEDIUM" in result

    def test_human_readable_confidence_low(self):
        """낮은 신뢰도 표시"""
        trace = ExplanationTrace(confidence_level="low", confidence_score=0.3)
        result = trace.to_human_readable()
        assert "LOW" in result

    def test_human_readable_confidence_unknown(self):
        """알 수 없는 신뢰도 표시"""
        trace = ExplanationTrace(confidence_level="unknown", confidence_score=0.0)
        result = trace.to_human_readable()
        assert "UNKNOWN" in result

    def test_human_readable_with_routing_path(self):
        """처리 경로 포함"""
        trace = ExplanationTrace(routing_path="HIGH -> 직접 응답")
        result = trace.to_human_readable()
        assert "HIGH" in result

    def test_human_readable_with_key_evidence(self):
        """핵심 근거 포함 (최대 3개)"""
        trace = ExplanationTrace(key_evidence=["증거1", "증거2", "증거3", "증거4"])
        result = trace.to_human_readable()
        assert "증거1" in result
        assert "증거2" in result
        assert "증거3" in result
        assert "증거4" not in result

    def test_human_readable_empty_trace(self):
        """빈 trace의 human readable"""
        trace = ExplanationTrace()
        result = trace.to_human_readable()
        assert "UNKNOWN" in result

    def test_human_readable_unrecognized_confidence_level(self):
        """인식 불가 신뢰도 레벨에 대한 기본 이모지"""
        trace = ExplanationTrace(confidence_level="custom_level", confidence_score=0.5)
        result = trace.to_human_readable()
        assert "CUSTOM_LEVEL" in result


# =============================================================================
# ExplainabilityEngine 테스트
# =============================================================================


class TestExplainabilityEngineBuildTrace:
    """ExplainabilityEngine.build_trace 테스트"""

    def setup_method(self):
        self.engine = ExplainabilityEngine()

    def test_build_trace_minimal_context(self):
        """최소 컨텍스트로 trace 생성"""
        context = Context(query="test query")
        trace = self.engine.build_trace(context)
        assert isinstance(trace, ExplanationTrace)
        assert trace.sources_used == ["None"]
        assert trace.rag_doc_count == 0
        assert trace.kg_fact_count == 0

    def test_build_trace_with_rag_docs(self):
        """RAG 문서 포함 시 소스에 RAG 추가"""
        context = Context(
            query="test",
            rag_docs=[{"content": "doc1"}, {"content": "doc2"}],
        )
        trace = self.engine.build_trace(context)
        assert "RAG" in trace.sources_used
        assert trace.rag_doc_count == 2

    def test_build_trace_with_kg_facts(self):
        """KG 사실 포함 시 소스에 KG 추가"""
        facts = [KGFact(fact_type="brand_info", entity="LANEIGE", data={"rank": 1})]
        context = Context(query="test", kg_facts=facts)
        trace = self.engine.build_trace(context)
        assert "KG" in trace.sources_used
        assert trace.kg_fact_count == 1

    def test_build_trace_with_kg_inferences(self):
        """KG 추론 포함 시 소스에 Ontology 추가"""
        context = Context(
            query="test",
            kg_inferences=[{"type": "market_trend", "insight": "growing"}],
        )
        trace = self.engine.build_trace(context)
        assert "Ontology" in trace.sources_used
        assert trace.kg_inference_count == 1

    def test_build_trace_with_crawled_data(self):
        """크롤링 데이터 포함 시 소스에 Crawled 추가"""
        from datetime import datetime

        state = SystemState(last_crawl_time=datetime.now())
        context = Context(query="test", system_state=state)
        trace = self.engine.build_trace(context)
        assert "Crawled" in trace.sources_used

    def test_build_trace_without_crawl_time(self):
        """크롤링 시간 없을 시 Crawled 미포함"""
        state = SystemState(last_crawl_time=None)
        context = Context(query="test", system_state=state)
        trace = self.engine.build_trace(context)
        assert "Crawled" not in trace.sources_used

    def test_build_trace_with_decision(self):
        """Decision 포함 시 판단 경로 기록"""
        context = Context(query="test")
        decision = Decision(
            tool="query_data",
            reason="데이터 조회 필요",
            confidence=0.8,
        )
        trace = self.engine.build_trace(context, decision=decision)
        assert trace.decision_tool == "query_data"
        assert trace.decision_reason == "데이터 조회 필요"
        assert trace.confidence_score == 0.8

    def test_build_trace_with_confidence_level_high(self):
        """HIGH 신뢰도 시 routing path"""
        context = Context(query="test")
        trace = self.engine.build_trace(context, confidence_level=ConfidenceLevel.HIGH)
        assert trace.confidence_level == "high"
        assert "직접 응답" in trace.routing_path

    def test_build_trace_with_confidence_level_medium(self):
        """MEDIUM 신뢰도 시 routing path"""
        context = Context(query="test")
        decision = Decision(tool="query_data", confidence=0.7)
        trace = self.engine.build_trace(
            context, decision=decision, confidence_level=ConfidenceLevel.MEDIUM
        )
        assert trace.confidence_level == "medium"
        assert "LLM 판단" in trace.routing_path

    def test_build_trace_with_confidence_level_low(self):
        """LOW 신뢰도 시 routing path"""
        context = Context(query="test")
        decision = Decision(tool="crawl_amazon", confidence=0.3)
        trace = self.engine.build_trace(
            context, decision=decision, confidence_level=ConfidenceLevel.LOW
        )
        assert trace.confidence_level == "low"
        assert "LLM 전체 판단" in trace.routing_path

    def test_build_trace_with_confidence_level_unknown(self):
        """UNKNOWN 신뢰도 시 routing path"""
        context = Context(query="test")
        trace = self.engine.build_trace(context, confidence_level=ConfidenceLevel.UNKNOWN)
        assert trace.confidence_level == "unknown"
        assert "명확화 요청" in trace.routing_path

    def test_build_trace_decision_confidence_high_no_level(self):
        """confidence_level 없이 Decision confidence >= 0.85일 때 high"""
        context = Context(query="test")
        decision = Decision(tool="direct_answer", confidence=0.9)
        trace = self.engine.build_trace(context, decision=decision)
        assert trace.confidence_level == "high"

    def test_build_trace_decision_confidence_medium_no_level(self):
        """confidence_level 없이 Decision confidence 0.5~0.85일 때 medium"""
        context = Context(query="test")
        decision = Decision(tool="query_data", confidence=0.6)
        trace = self.engine.build_trace(context, decision=decision)
        assert trace.confidence_level == "medium"

    def test_build_trace_decision_confidence_low_no_level(self):
        """confidence_level 없이 Decision confidence < 0.5일 때 low"""
        context = Context(query="test")
        decision = Decision(tool="crawl_amazon", confidence=0.3)
        trace = self.engine.build_trace(context, decision=decision)
        assert trace.confidence_level == "low"

    def test_build_trace_with_response(self):
        """Response 포함 시 점수 및 시간 기록"""
        context = Context(query="test")
        response = Response(text="answer", confidence_score=0.85, processing_time_ms=200.0)
        trace = self.engine.build_trace(context, response=response)
        assert trace.confidence_score == 0.85
        assert trace.processing_time_ms == 200.0

    def test_build_trace_with_response_no_processing_time(self):
        """Response processing_time_ms가 None이면 0.0"""
        context = Context(query="test")
        response = Response(text="answer", processing_time_ms=None)
        trace = self.engine.build_trace(context, response=response)
        assert trace.processing_time_ms == 0.0

    def test_build_trace_medium_direct_answer(self):
        """MEDIUM + direct_answer 경로"""
        context = Context(query="test")
        decision = Decision(tool="direct_answer", confidence=0.7)
        trace = self.engine.build_trace(
            context, decision=decision, confidence_level=ConfidenceLevel.MEDIUM
        )
        assert "직접 응답" in trace.routing_path


class TestExplainabilityEngineIdentifySources:
    """ExplainabilityEngine._identify_sources 테스트"""

    def setup_method(self):
        self.engine = ExplainabilityEngine()

    def test_no_sources_returns_none_list(self):
        """소스 없을 시 ['None'] 반환"""
        context = Context(query="test")
        sources = self.engine._identify_sources(context)
        assert sources == ["None"]

    def test_all_sources(self):
        """모든 소스 식별"""
        from datetime import datetime

        context = Context(
            query="test",
            rag_docs=[{"content": "doc"}],
            kg_facts=[KGFact(fact_type="info", entity="A", data={})],
            kg_inferences=[{"insight": "test"}],
            system_state=SystemState(last_crawl_time=datetime.now()),
        )
        sources = self.engine._identify_sources(context)
        assert "RAG" in sources
        assert "KG" in sources
        assert "Ontology" in sources
        assert "Crawled" in sources


class TestExplainabilityEngineExtractEvidence:
    """ExplainabilityEngine._extract_key_evidence 테스트"""

    def setup_method(self):
        self.engine = ExplainabilityEngine()

    def test_extract_from_kg_facts(self):
        """KG 사실에서 근거 추출"""
        facts = [KGFact(fact_type="brand_info", entity="LANEIGE", data={})]
        context = Context(query="test", kg_facts=facts)
        evidence = self.engine._extract_key_evidence(context, None)
        assert any("[KG]" in e for e in evidence)

    def test_extract_from_kg_inferences_dict(self):
        """KG 추론 dict에서 근거 추출"""
        context = Context(
            query="test",
            kg_inferences=[{"insight": "시장 성장 중"}],
        )
        evidence = self.engine._extract_key_evidence(context, None)
        assert any("[Ontology]" in e for e in evidence)

    def test_extract_from_kg_inferences_type_key(self):
        """KG 추론 type 키에서 근거 추출"""
        context = Context(
            query="test",
            kg_inferences=[{"type": "market_growth"}],
        )
        evidence = self.engine._extract_key_evidence(context, None)
        assert any("[Ontology]" in e for e in evidence)

    def test_extract_from_rag_docs(self):
        """RAG 문서에서 근거 추출"""
        context = Context(
            query="test",
            rag_docs=[{"metadata": {"title": "LANEIGE Report"}}],
        )
        evidence = self.engine._extract_key_evidence(context, None)
        assert any("[RAG]" in e for e in evidence)

    def test_extract_from_decision_key_points(self):
        """Decision key_points에서 근거 추출"""
        context = Context(query="test")
        decision = Decision(key_points=["시장 점유율 상승", "가격 경쟁력 강화"])
        evidence = self.engine._extract_key_evidence(context, decision)
        assert any("[분석]" in e for e in evidence)

    def test_evidence_max_5(self):
        """최대 5개 근거 반환"""
        facts = [KGFact(fact_type=f"type{i}", entity=f"E{i}", data={}) for i in range(10)]
        context = Context(
            query="test",
            kg_facts=facts,
            kg_inferences=[{"insight": f"inf{i}"} for i in range(5)],
            rag_docs=[{"metadata": {"title": f"doc{i}"}} for i in range(3)],
        )
        decision = Decision(key_points=["p1", "p2", "p3"])
        evidence = self.engine._extract_key_evidence(context, decision)
        assert len(evidence) <= 5

    def test_extract_empty_context(self):
        """빈 컨텍스트에서 빈 근거"""
        context = Context(query="test")
        evidence = self.engine._extract_key_evidence(context, None)
        assert evidence == []


class TestExplainabilityEngineFormatForResponse:
    """ExplainabilityEngine.format_for_response 테스트"""

    def setup_method(self):
        self.engine = ExplainabilityEngine()

    def test_format_brief(self):
        """간략 포맷"""
        trace = ExplanationTrace(
            sources_used=["RAG"],
            confidence_level="high",
            confidence_score=0.9,
        )
        result = self.engine.format_for_response(trace, include_details=False)
        assert "RAG" in result
        assert "HIGH" in result

    def test_format_detailed(self):
        """상세 포맷은 to_human_readable 호출"""
        trace = ExplanationTrace(
            sources_used=["RAG", "KG"],
            confidence_level="medium",
            confidence_score=0.7,
            routing_path="MEDIUM -> LLM 판단",
        )
        result = self.engine.format_for_response(trace, include_details=True)
        assert "RAG" in result
        assert "KG" in result
        assert "MEDIUM" in result
