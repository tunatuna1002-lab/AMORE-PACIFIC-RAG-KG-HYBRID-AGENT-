"""
Confidence Routing 단위 테스트
신뢰도 평가 및 라우팅 로직 검증
"""

from src.core.confidence import ConfidenceAssessor, calculate_absolute_score
from src.core.models import ConfidenceLevel


class TestConfidenceAssessor:
    """ConfidenceAssessor 테스트"""

    def setup_method(self):
        self.assessor = ConfidenceAssessor()

    def test_high_confidence_from_score(self):
        """HIGH: max_score >= 5.0"""
        result = {"max_score": 6.0, "query_type": "sos_query"}
        level = self.assessor.assess(result)
        assert level == ConfidenceLevel.HIGH

    def test_medium_confidence_from_score(self):
        """MEDIUM: 3.0 <= max_score < 5.0"""
        result = {"max_score": 4.0, "query_type": "rank_query"}
        level = self.assessor.assess(result)
        assert level == ConfidenceLevel.MEDIUM

    def test_low_confidence_from_score(self):
        """LOW: 1.5 <= max_score < 3.0"""
        result = {"max_score": 2.0, "query_type": "general"}
        level = self.assessor.assess(result)
        assert level == ConfidenceLevel.LOW

    def test_unknown_confidence_from_score(self):
        """UNKNOWN: max_score < 1.5"""
        result = {"max_score": 0.5}
        level = self.assessor.assess(result)
        assert level == ConfidenceLevel.UNKNOWN

    def test_confidence_fallback_to_confidence_field(self):
        """max_score 없을 때 confidence 필드 사용"""
        result = {"confidence": 0.9}  # 0.9 * 6.0 = 5.4 → HIGH
        level = self.assessor.assess(result)
        assert level == ConfidenceLevel.HIGH

    def test_should_skip_llm_decision(self):
        """HIGH일 때만 LLM 스킵"""
        assert self.assessor.should_skip_llm_decision(ConfidenceLevel.HIGH) is True
        assert self.assessor.should_skip_llm_decision(ConfidenceLevel.MEDIUM) is False
        assert self.assessor.should_skip_llm_decision(ConfidenceLevel.LOW) is False

    def test_should_request_clarification(self):
        """UNKNOWN일 때만 명확화 요청"""
        assert self.assessor.should_request_clarification(ConfidenceLevel.UNKNOWN) is True
        assert self.assessor.should_request_clarification(ConfidenceLevel.LOW) is False

    def test_assess_with_details(self):
        """상세 평가 결과 확인"""
        result = {"max_score": 4.5, "query_type": "sos_query"}
        level, details = self.assessor.assess_with_details(result)
        assert level == ConfidenceLevel.MEDIUM
        assert "score" in details
        assert "processing_strategy" in details

    def test_no_double_counting_with_context(self):
        """컨텍스트가 전달되어도 이중 계산하지 않음 (regression test)"""
        from src.core.models import Context

        ctx = Context(
            query="LANEIGE 분석해줘",
            rag_docs=[{"content": "doc1"}, {"content": "doc2"}, {"content": "doc3"}],
            kg_facts=[],
        )
        # max_score가 이미 컨텍스트 데이터를 반영한 점수
        result = {"max_score": 4.0, "query_type": "general"}
        level = self.assessor.assess(result, ctx)
        # context bonus가 더해지지 않으므로 4.0 → MEDIUM (3.0~4.9)
        assert level == ConfidenceLevel.MEDIUM

    def test_custom_thresholds(self):
        """커스텀 임계값"""
        custom = ConfidenceAssessor(threshold_high=8.0, threshold_medium=5.0, threshold_low=2.0)
        result = {"max_score": 6.0}
        level = custom.assess(result)
        assert level == ConfidenceLevel.MEDIUM  # 6.0 < 8.0


class TestCalculateAbsoluteScore:
    """절대 점수 계산 테스트"""

    def test_keyword_scoring(self):
        score = calculate_absolute_score(matched_keywords=2)
        assert score == 4.0  # 2 * 2.0

    def test_combined_scoring(self):
        score = calculate_absolute_score(
            matched_keywords=1, matched_indicators=1, matched_entities=1
        )
        assert score == 5.0  # 2.0 + 1.5 + 1.5

    def test_zero_score(self):
        score = calculate_absolute_score()
        assert score == 0.0
