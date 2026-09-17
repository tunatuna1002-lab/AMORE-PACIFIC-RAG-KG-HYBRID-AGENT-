"""
Phase 4 신뢰도·가드레일 테스트 (§4.1 max→감쇠 / §4.2 사다리 통합 / §4.3 가드레일 / §4.5 출처)
"""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.core.confidence import ConfidenceAssessor
from src.core.models import ConfidenceLevel, Context
from src.core.response_pipeline import ResponsePipeline
from src.rag.templates import ResponseTemplates

SRC_ROOT = Path(__file__).resolve().parents[3] / "src"


@pytest.fixture
def pipeline():
    return ResponsePipeline()


def _ctx(rag=0, kg=0, inf=0, fresh=False):
    ctx = Context(query="LANEIGE 순위")
    ctx.rag_docs = [{"metadata": {"title": f"doc{i}"}} for i in range(rag)]
    ctx.kg_facts = [{"s": "a", "p": "b", "o": "c"} for _ in range(kg)]
    ctx.kg_inferences = [MagicMock() for _ in range(inf)]
    if fresh:
        state = MagicMock()
        state.data_freshness = "fresh"
        state.kg_initialized = True
        ctx.system_state = state
    else:
        ctx.system_state = None
    return ctx


# =============================================================================
# §4.1 근거 점수가 상한, LLM 자기보고는 감쇠만
# =============================================================================


class TestGroundingAttenuation:
    @staticmethod
    def _final(pipeline, calculated: float, decision_confidence: float | None) -> float:
        """실제 파이프라인과 같은 규칙으로 최종 신뢰도 계산"""
        if decision_confidence:
            return calculated * min(float(decision_confidence), 1.0)
        return calculated

    def test_no_evidence_stays_zero_despite_confident_llm(self, pipeline):
        """근거가 없으면 LLM이 아무리 자신해도 0 (과거 max()는 0.95로 올렸다)"""
        calculated = pipeline._calculate_confidence_score(_ctx())
        assert calculated == 0.0
        assert self._final(pipeline, calculated, 0.95) == 0.0

    def test_llm_uncertainty_lowers_grounded_score(self, pipeline):
        calculated = pipeline._calculate_confidence_score(_ctx(rag=3, kg=3))
        assert calculated == 6.0
        assert self._final(pipeline, calculated, 0.5) == 3.0

    def test_confident_llm_cannot_exceed_grounded_score(self, pipeline):
        calculated = pipeline._calculate_confidence_score(_ctx(rag=3, kg=3))
        assert self._final(pipeline, calculated, 1.0) == calculated

    def test_missing_decision_confidence_keeps_grounded_score(self, pipeline):
        calculated = pipeline._calculate_confidence_score(_ctx(rag=2))
        assert self._final(pipeline, calculated, None) == calculated

    def test_source_has_no_max_call(self):
        """max(calculated, decision.confidence) 패턴 회귀 방지"""
        text = (SRC_ROOT / "core" / "response_pipeline.py").read_text(encoding="utf-8")
        assert "max(calculated_confidence" not in text


# =============================================================================
# §4.2 신뢰도 사다리 단일화
# =============================================================================


class TestConfidenceLadderUnified:
    @pytest.mark.parametrize(
        "rag,kg,inf,expected",
        [
            (0, 0, 0, ConfidenceLevel.UNKNOWN),
            (2, 0, 0, ConfidenceLevel.LOW),
            (3, 0, 0, ConfidenceLevel.MEDIUM),
            (3, 3, 0, ConfidenceLevel.HIGH),
        ],
    )
    def test_levels_match_assessor(self, pipeline, rag, kg, inf, expected):
        ctx = _ctx(rag=rag, kg=kg, inf=inf)
        assert pipeline._assess_confidence(ctx) == expected

    def test_delegates_to_assessor(self, pipeline):
        """자체 사다리가 아니라 ConfidenceAssessor를 쓴다"""
        assert isinstance(pipeline._confidence_assessor, ConfidenceAssessor)
        score = pipeline._calculate_confidence_score(_ctx(rag=3, kg=3))
        assert pipeline._assess_confidence(_ctx(rag=3, kg=3)) == (
            pipeline._confidence_assessor.assess({"max_score": score})
        )

    def test_custom_thresholds_are_honoured(self, pipeline):
        """사다리가 한 곳에만 있으므로 임계값 변경이 그대로 반영된다"""
        pipeline._confidence_assessor = ConfidenceAssessor(threshold_high=1.0)
        assert pipeline._assess_confidence(_ctx(rag=2)) == ConfidenceLevel.HIGH

    def test_no_inline_ladder_in_source(self):
        text = (SRC_ROOT / "core" / "response_pipeline.py").read_text(encoding="utf-8")
        assert "if score >= 5.0:" not in text


# =============================================================================
# §4.3 가드레일 실동작
# =============================================================================


class TestGuardrails:
    @pytest.mark.parametrize(
        "text,forbidden",
        [
            # "원인은 X입니다" 는 단정 어미가 사라지면 된다 ("원인은"은 유지)
            ("순위 하락의 원인은 경쟁사 프로모션입니다.", "프로모션입니다"),
            ("확실히 LANEIGE가 우위입니다.", "확실히"),
            ("반드시 가격을 인하해야 합니다.", "반드시"),
            ("틀림없이 상승 추세입니다.", "틀림없이"),
            ("절대적으로 유리한 위치입니다.", "절대적으로"),
            ("100% 확실한 신호입니다.", "100%"),
        ],
    )
    def test_forbidden_phrases_are_rewritten(self, text, forbidden):
        result = ResponseTemplates.apply_guardrails(text)
        assert result != text, "가드레일이 no-op입니다"
        assert forbidden not in result

    def test_hedged_wording_is_used(self):
        result = ResponseTemplates.apply_guardrails("순위 하락의 원인은 경쟁사 프로모션입니다.")
        assert "추정됩니다" in result

    def test_idempotent(self):
        once = ResponseTemplates.apply_guardrails("확실히 상승세입니다.")
        assert ResponseTemplates.apply_guardrails(once) == once

    def test_normal_text_untouched(self):
        text = "LANEIGE의 Lip Care SoS는 2.0%로 집계됩니다."
        assert ResponseTemplates.apply_guardrails(text) == text

    def test_legitimate_percentage_untouched(self):
        """SoS 100% 같은 정상 수치는 건드리지 않는다"""
        text = "해당 카테고리 SoS는 100%입니다."
        assert ResponseTemplates.apply_guardrails(text) == text

    def test_empty_input(self):
        assert ResponseTemplates.apply_guardrails("") == ""

    def test_multiline_only_affects_matching_lines(self):
        text = "정상 문장입니다.\n확실히 우위입니다.\n또 정상입니다."
        result = ResponseTemplates.apply_guardrails(text)
        assert result.splitlines()[0] == "정상 문장입니다."
        assert result.splitlines()[2] == "또 정상입니다."
        assert "확실히" not in result


# =============================================================================
# §4.5 출처 추출 단일화
# =============================================================================


class TestSourceUnification:
    """출처는 한 경로 — 프롬프트 증거 카드(evidence_source_labels) (트랙 2-B)"""

    def test_uses_prompt_evidence_cards(self, pipeline):
        from src.rag.evidence_assembly import assemble_evidence

        ctx = _ctx(rag=1, kg=1)
        ctx.prompt_evidence = assemble_evidence(
            rag_chunks=[{"id": "d1_0", "content": "정의", "metadata": {"title": "SoS 정의"}}],
            ontology_facts=[
                {"type": "competitors", "entity": "laneige", "data": [{"brand": "cosrx"}]}
            ],
        ).prompt_evidence

        assert pipeline._extract_sources(ctx) == ["KG", "SoS 정의"]

    def test_single_source_path(self):
        """SourceProvider 위임·폴백 이중 경로가 없다"""
        import inspect

        source = inspect.getsource(ResponsePipeline._extract_sources)
        assert "evidence_source_labels" in source
        assert not hasattr(ResponsePipeline, "_extract_sources_via_provider")
        assert not hasattr(ResponsePipeline, "_extract_sources_fallback")

    def test_raw_fields_without_cards_give_no_sources(self, pipeline):
        ctx = _ctx(rag=10, kg=3, inf=2)
        ctx.rag_docs = [
            {"metadata": {"doc_id": f"d{i}", "title": f"문서 {i}"}, "score": 0.5} for i in range(10)
        ]
        assert pipeline._extract_sources(ctx) == []


# =============================================================================
# §4.4 인사이트 few-shot 가짜 수치
# =============================================================================


class TestInsightPromptNoFakeNumbers:
    FABRICATED = ["+2.1%p", "+12.3%", "+41%", "+6.9%", "+34%", "2.4M", "8.2%"]

    def test_no_fabricated_numerals_in_prompt(self):
        text = (SRC_ROOT / "agents" / "hybrid_insight_agent.py").read_text(encoding="utf-8")
        found = [n for n in self.FABRICATED if n in text]
        assert not found, f"few-shot에 가짜 수치가 남아있습니다: {found}"

    def test_uses_typed_placeholders(self):
        text = (SRC_ROOT / "agents" / "hybrid_insight_agent.py").read_text(encoding="utf-8")
        for placeholder in ("{{SOS_DELTA}}", "{{RANK}}", "[D{{n}}]"):
            assert placeholder in text, f"{placeholder} 자리표시자 누락"

    def test_placeholder_rule_is_stated(self):
        text = (SRC_ROOT / "agents" / "hybrid_insight_agent.py").read_text(encoding="utf-8")
        assert "자리표시자를 그대로 출력하거나" in text


# =============================================================================
# §4.6 brand_resolver 이름 정정
# =============================================================================


class TestBrandResolverNaming:
    def test_renamed_and_documented(self):
        from src.tools.utilities.brand_resolver import BrandResolver

        assert not hasattr(BrandResolver, "_search_brand_web")
        assert hasattr(BrandResolver, "_resolve_from_known_patterns")
        doc = BrandResolver._resolve_from_known_patterns.__doc__
        assert "웹검색을 하지 않는다" in doc
