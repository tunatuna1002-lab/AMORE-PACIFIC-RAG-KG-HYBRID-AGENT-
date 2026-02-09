"""
신뢰도 평가 모듈
================
Rule-based 분류 결과의 신뢰도를 평가하여 처리 전략 결정

신뢰도 레벨:
- HIGH (5.0+): Rule 결과 신뢰, LLM 판단 스킵
- MEDIUM (3.0~4.9): LLM에게 도구 선택 위임
- LOW (1.5~2.9): LLM에게 전체 판단 위임
- UNKNOWN (<1.5): 명확화 요청

Usage:
    assessor = ConfidenceAssessor()
    level = assessor.assess(rule_result)
"""

from typing import Any

from .models import ConfidenceLevel


class ConfidenceAssessor:
    """
    신뢰도 평가기

    RAGRouter의 분류 결과를 받아 신뢰도 레벨을 결정한다.
    절대 점수 기반으로 평가하여 일관된 판단을 보장한다.
    """

    # =========================================================================
    # 점수 임계값 설정
    # =========================================================================

    # 신뢰도 레벨별 최소 점수
    THRESHOLD_HIGH = 5.0  # 매우 확실 (키워드 2개 + 지표 1개 이상)
    THRESHOLD_MEDIUM = 3.0  # 확실 (키워드 1개 + 지표 1개)
    THRESHOLD_LOW = 1.5  # 보통 (최소 1개 매칭)

    def __init__(
        self,
        threshold_high: float = None,
        threshold_medium: float = None,
        threshold_low: float = None,
    ):
        """
        Args:
            threshold_high: HIGH 레벨 임계값 (기본 5.0)
            threshold_medium: MEDIUM 레벨 임계값 (기본 3.0)
            threshold_low: LOW 레벨 임계값 (기본 1.5)
        """
        self.threshold_high = threshold_high or self.THRESHOLD_HIGH
        self.threshold_medium = threshold_medium or self.THRESHOLD_MEDIUM
        self.threshold_low = threshold_low or self.THRESHOLD_LOW

    def assess(self, rule_result: dict[str, Any], context: Any = None) -> ConfidenceLevel:
        """
        신뢰도 레벨 평가

        Args:
            rule_result: RAGRouter.route() 결과
                - max_score: 최고 점수 (절대값)
                - confidence: 상대적 신뢰도 (0~1)
                - query_type: 분류된 유형
            context: Context 객체 (무시됨, 하위 호환용으로 유지)

        Returns:
            ConfidenceLevel

        Note:
            context 파라미터는 더 이상 점수에 반영되지 않습니다.
            호출자(brain.py, llm_orchestrator.py)가 이미 컨텍스트 데이터를
            max_score에 반영하므로, 여기서 다시 가산하면 이중 계산됩니다.
        """
        # 절대 점수 추출 (max_score 우선, 없으면 confidence로 추정)
        score = rule_result.get("max_score", 0)

        # max_score가 없으면 confidence에서 추정 (하위 호환)
        if score == 0 and "confidence" in rule_result:
            # confidence는 0~1 범위, 대략적인 점수로 변환
            score = rule_result["confidence"] * 6.0

        # NOTE: _context_bonus() 제거 — brain.py/_assess_confidence_level()에서
        # 이미 컨텍스트 데이터(kg_facts, rag_docs, kg_inferences)를 점수에 반영.
        # 여기서 다시 가산하면 이중 계산으로 거의 모든 쿼리가 HIGH로 분류됨.

        return self._score_to_level(score)

    def _context_bonus(self, context: Any) -> float:
        """
        컨텍스트 기반 점수 보너스 계산

        Args:
            context: Context 객체

        Returns:
            보너스 점수
        """
        bonus = 0.0

        # RAG 문서 있음
        if hasattr(context, "rag_docs") and context.rag_docs:
            bonus += min(len(context.rag_docs), 3) * 0.5

        # KG 사실 있음
        if hasattr(context, "kg_facts") and context.kg_facts:
            bonus += min(len(context.kg_facts), 3) * 0.5

        # KG 추론 있음
        if hasattr(context, "kg_inferences") and context.kg_inferences:
            bonus += min(len(context.kg_inferences), 2) * 0.75

        return bonus

    def _score_to_level(self, score: float) -> ConfidenceLevel:
        """
        점수를 신뢰도 레벨로 변환

        Args:
            score: 절대 점수

        Returns:
            ConfidenceLevel
        """
        if score >= self.threshold_high:
            return ConfidenceLevel.HIGH
        elif score >= self.threshold_medium:
            return ConfidenceLevel.MEDIUM
        elif score >= self.threshold_low:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.UNKNOWN

    def assess_with_details(
        self, rule_result: dict[str, Any]
    ) -> tuple[ConfidenceLevel, dict[str, Any]]:
        """
        상세 정보와 함께 신뢰도 평가

        Args:
            rule_result: RAGRouter.route() 결과

        Returns:
            (ConfidenceLevel, 상세 정보 dict)
        """
        score = rule_result.get("max_score", 0)
        if score == 0 and "confidence" in rule_result:
            score = rule_result["confidence"] * 6.0

        level = self._score_to_level(score)

        details = {
            "score": score,
            "level": level.value,
            "thresholds": {
                "high": self.threshold_high,
                "medium": self.threshold_medium,
                "low": self.threshold_low,
            },
            "processing_strategy": self._get_strategy(level),
            "matched_keywords": rule_result.get("matched_keywords", []),
            "query_type": rule_result.get("query_type"),
        }

        return level, details

    def _get_strategy(self, level: ConfidenceLevel) -> str:
        """
        신뢰도 레벨에 따른 처리 전략 설명

        Args:
            level: 신뢰도 레벨

        Returns:
            처리 전략 설명 문자열
        """
        strategies = {
            ConfidenceLevel.HIGH: "LLM 판단 스킵, Rule 결과로 바로 응답 생성",
            ConfidenceLevel.MEDIUM: "LLM에게 도구 선택 위임, 컨텍스트 기반 판단",
            ConfidenceLevel.LOW: "LLM에게 전체 판단 위임, 의도 파악부터 시작",
            ConfidenceLevel.UNKNOWN: "명확화 요청, 사용자 재입력 유도",
        }
        return strategies.get(level, "알 수 없는 전략")

    def should_skip_llm_decision(self, level: ConfidenceLevel) -> bool:
        """
        LLM 판단을 스킵해도 되는지 확인

        HIGH 레벨일 때만 LLM 판단 스킵

        Args:
            level: 신뢰도 레벨

        Returns:
            True면 LLM 판단 스킵 가능
        """
        return level == ConfidenceLevel.HIGH

    def should_request_clarification(self, level: ConfidenceLevel) -> bool:
        """
        명확화 요청이 필요한지 확인

        UNKNOWN 레벨일 때 명확화 요청

        Args:
            level: 신뢰도 레벨

        Returns:
            True면 명확화 요청 필요
        """
        return level == ConfidenceLevel.UNKNOWN


# =============================================================================
# 점수 계산 유틸리티 (RAGRouter 보완용)
# =============================================================================


def calculate_absolute_score(
    matched_keywords: int = 0,
    matched_indicators: int = 0,
    matched_entities: int = 0,
    matched_patterns: int = 0,
) -> float:
    """
    매칭된 항목 수로 절대 점수 계산

    점수 기준:
    - 키워드: +2.0점 (뭐야, 해석, 분석 등)
    - 지표명: +1.5점 (sos, hhi, cpi 등)
    - 엔티티: +1.5점 (라네즈, laneige 등)
    - 패턴: +1.0점 (높으면, 낮으면 등)

    Args:
        matched_keywords: 매칭된 키워드 수
        matched_indicators: 매칭된 지표명 수
        matched_entities: 매칭된 엔티티 수
        matched_patterns: 매칭된 패턴 수

    Returns:
        절대 점수
    """
    return (
        matched_keywords * 2.0
        + matched_indicators * 1.5
        + matched_entities * 1.5
        + matched_patterns * 1.0
    )
