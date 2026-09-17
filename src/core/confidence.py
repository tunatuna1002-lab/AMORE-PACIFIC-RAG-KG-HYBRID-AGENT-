"""
신뢰도 평가 모듈 (적합도 기반)
==============================
증거가 **이 질문에 맞는지**로 신뢰도를 매기고, 그 점수로 처리 전략을 정한다.

왜 바꿨나 (트랙 5-B)
--------------------
예전 점수는 컨텍스트 개수 합이었다 (kg_facts×1.5 + rag_docs×1.0 + inferences×2.0 + …).
검색이 질문마다 비슷한 양의 자료를 담아 오면 점수도 비슷하게 높아져서, 233문항 평가에서
230문항이 HIGH로 나왔다 (``eval_output/evidence-2026-09/s3-run1/report.json``).
HIGH는 LLM 판단을 건너뛰라는 뜻이므로 DecisionMaker·ReAct 분기가 한 번도 실행되지
않았고, 신뢰도는 아무 정보도 나르지 않았다.

이제 점수는 **적합도**다 (설계 E6). 세 성분을 0~1로 재고 가중합한다:

(a) ``entity_coverage`` — 질문이 이름을 부른 엔티티(브랜드·카테고리·제품)가 증거 카드의
    **주어·목적어**로 등장하는 비율. 본문에 단어가 스쳤는지가 아니라 카드가 그 엔티티를
    **대상으로 삼는지**를 본다. 질문이 아무 엔티티도 부르지 않으면(정의 질문 등) 닻을
    내릴 대상이 없으므로 1.0으로 둔다 — 없는 것을 벌하지 않는다.
(b) ``kind_fit`` — 질문 유형이 요구하는 카드 종류가 실제로 있는 비율. 수치 질문은 metric,
    관계 질문은 relation, 판단 질문은 inference, 정의·가정형 질문은 document를 요구한다.
    이름 붙은 엔티티가 있으면 그 엔티티에 닻을 내린 카드만 요구를 채운 것으로 센다.
(c) ``retrieval_fit`` — 문서 카드 검색 점수 분포의 뾰족함 ``(top - median) / top``.
    점수는 검색 경로마다 눈금이 달라 절대값을 쓸 수 없어서 눈금에 무관한 모양만 쓴다.
    점수가 2개 미만이면 뾰족한지 평평한지 말할 수 없으므로 중립값 0.5.

임계값은 측정 데이터로 보정했다 — ``scripts/calibrate_confidence_thresholds.py`` 참조.
사다리는 이 모듈 하나에만 있다. 개수 기반 점수를 쓰는 옛 호출자
(``response_pipeline._assess_confidence``)는 ``legacy_count_score_to_fit()``로 같은
사다리에 올린다 — 사다리를 둘로 늘리지 않으면서 옛 동작을 그대로 보존한다.

Usage:
    fit = score_evidence_fit(query, context.entities, context.prompt_evidence)
    level = ConfidenceAssessor().assess_fit(fit)
"""

from __future__ import annotations

import re
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from src.domain.entities.evidence import Evidence, EvidenceKind

from .models import ConfidenceLevel

__all__ = [
    "ConfidenceAssessor",
    "EvidenceFit",
    "FIT_WEIGHTS",
    "calculate_absolute_score",
    "legacy_count_score_to_fit",
    "required_evidence_kinds",
    "score_evidence_fit",
]


# =============================================================================
# 질문 유형 어휘
# =============================================================================
# 표현 목록은 scripts/classify_golden_types.py(트랙 1-A)가 골든셋을 유형별로 나눌 때 쓴
# 어휘를 따온 것이다. 그 스크립트는 gold 필드를 보지만 여기서는 런타임에 있는 것(질문
# 문자열 + 엔티티 링킹 결과)만 쓴다.

# 정의·산출식을 묻는 질문 → 문서 카드가 답한다
_DEFINITION_RE = re.compile(
    r"(란\s*무엇|이란|정의|계산\s*(방식|방법|공식|식)|산출식|계산에 필요한|의미하는 바|해석 방법)"
)

_METRIC_WORD = r"(SoS|HHI|CPI|이탈률|점유율|순위|리뷰)"

# 지표에 값이 이미 주어진 가정형 판정 ("SoS 5%는 좋은 수치인가요?")
_HYPOTHETICAL_VALUE_RE = re.compile(
    _METRIC_WORD + r"\s*[0-9.]+\s*%?\s*(는|은|이|가)", re.IGNORECASE
)

# 지표 조건절 ("CPI가 1.0 미만일 때", "SoS가 높은데 HHI도 높으면")
_HYPOTHETICAL_CONDITION_RE = re.compile(
    _METRIC_WORD + r"[^?]{0,20}(높으면|낮으면|높은데|낮은데|미만일 때|이상일 때|초과일 때|"
    r"하락할 때|상승할 때|하락 중|상승 중|증가 시|감소 시|동시에)",
    re.IGNORECASE,
)

# 엔티티 사이의 관계를 묻는 질문 → relation 카드가 답한다
_RELATION_RE = re.compile(
    r"(경쟁|모회사|소속|자매|포트폴리오|라인업|라인 제품|세그먼트|원산지|인수|소유|"
    r"브랜드 중|어느 브랜드|관계)"
)

# 판단·전략을 묻는 질문 → 규칙 추론(inference) 카드가 답한다
_JUDGEMENT_RE = re.compile(
    r"(전략|위협|기회|포지션|포지셔닝|시장 구조|평가|제언|판단|시사점|인사이트|"
    r"어떻게 봐야|대응)"
)

# 수치를 묻는 질문 → metric 카드가 답한다
_NUMERIC_ASK_RE = re.compile(
    r"(얼마|몇|순위|현황|top\s*\d|상위|평균|가격|점유율|비중|수치|추이|성장|규모|rank|show me)",
    re.IGNORECASE,
)

# 질문이 "이름을 부른" 엔티티로 세는 엔티티 링킹 결과 키.
# indicators(sos·hhi)는 닻 대상이 아니다 — 지표는 카드의 술어이지 주어가 아니다.
_NAMED_ENTITY_KEYS = ("brands", "categories", "products")

# 카드에서 "이 카드가 무엇을 대상으로 삼는가"를 읽는 metadata 키
_ANCHOR_METADATA_KEYS = (
    "display_name",
    "object_display_name",
    "brand",
    "brand_display_name",
    "name",
)
_ANCHOR_METADATA_LIST_KEYS = ("categories", "related_entities")

# 성분 가중치. 보정 대상은 임계값이지 가중치가 아니다 — 가중치는 설계에서 정한다.
# 엔티티 닻이 1순위(증거가 다른 대상을 말하고 있으면 나머지는 의미가 없다),
# 종류 요구가 2순위, 검색 분포가 3순위.
FIT_WEIGHTS: dict[str, float] = {
    "entity_coverage": 0.50,
    "kind_fit": 0.35,
    "retrieval_fit": 0.15,
}

_NEUTRAL_RETRIEVAL_FIT = 0.5


# =============================================================================
# 적합도 결과
# =============================================================================


@dataclass(frozen=True)
class EvidenceFit:
    """증거 적합도 측정 결과.

    Attributes:
        score: 가중합 (0~1). 신뢰도 사다리에 올리는 값.
        entity_coverage: 질문 엔티티 중 카드에 닻을 내린 비율 (0~1).
        kind_fit: 질문이 요구한 카드 종류 중 실제로 있는 비율 (0~1).
        retrieval_fit: 문서 검색 점수 분포의 뾰족함 (0~1). 알 수 없으면 0.5.
        needs: 이 질문이 요구하는 카드 종류.
        named_entities: 질문이 이름을 부른 엔티티 (정규화된 소문자).
        matched_entities: 그중 카드에 닻을 내린 엔티티.
        card_count: 평가에 쓴 카드 수.
        basis: "fit"(카드로 잼) | "empty"(카드 없음) | "legacy"(개수 점수 폴백).
    """

    score: float
    entity_coverage: float
    kind_fit: float
    retrieval_fit: float
    needs: tuple[str, ...]
    named_entities: tuple[str, ...]
    matched_entities: tuple[str, ...]
    card_count: int
    basis: str

    def to_dict(self) -> dict[str, Any]:
        """관측(route_trace)용 직렬화."""
        return {
            "entity_coverage": self.entity_coverage,
            "kind_fit": self.kind_fit,
            "retrieval_fit": self.retrieval_fit,
            "needs": list(self.needs),
            "named_entities": list(self.named_entities),
            "matched_entities": list(self.matched_entities),
            "card_count": self.card_count,
            "basis": self.basis,
        }


# =============================================================================
# 질문 유형 → 요구 카드 종류
# =============================================================================


def required_evidence_kinds(
    query: str, entities: Mapping[str, Sequence[str]] | None = None
) -> frozenset[str]:
    """이 질문에 답하려면 어떤 종류의 증거 카드가 있어야 하는지 판정한다.

    런타임에 있는 것만 본다: 질문 문자열과 엔티티 링킹 결과. 골든셋 gold 필드는 쓰지
    않는다(평가 때만 있는 정보다).

    정의 질문과 가정형 판정 질문("SoS 5%는 좋은 수치인가요?")은 DB 수치가 아니라 해석
    문서가 답하므로 metric을 요구하지 않는다 — 요구하면 답할 수 있는 질문까지 신뢰도가
    깎인다.

    Args:
        query: 사용자 질문
        entities: 엔티티 링킹 결과 (``{"brands": [...], "indicators": [...]}``)

    Returns:
        ``EvidenceKind`` 값 문자열의 집합. 아무것도 특정할 수 없으면 ``{"document"}``.
    """
    text = query or ""
    if (
        _DEFINITION_RE.search(text)
        or _HYPOTHETICAL_VALUE_RE.search(text)
        or _HYPOTHETICAL_CONDITION_RE.search(text)
    ):
        return frozenset({EvidenceKind.DOCUMENT.value})

    indicators = _normalized(entities, ("indicators",))

    needs: set[str] = set()
    if indicators or _NUMERIC_ASK_RE.search(text):
        needs.add(EvidenceKind.METRIC.value)
    if _RELATION_RE.search(text):
        needs.add(EvidenceKind.RELATION.value)
    if _JUDGEMENT_RE.search(text):
        needs.add(EvidenceKind.INFERENCE.value)

    return frozenset(needs) or frozenset({EvidenceKind.DOCUMENT.value})


# =============================================================================
# 적합도 점수
# =============================================================================


def score_evidence_fit(
    query: str,
    entities: Mapping[str, Sequence[str]] | None,
    cards: Sequence[Evidence] | None,
) -> EvidenceFit:
    """질문 대비 증거 카드의 적합도를 잰다 (LLM·검색 호출 없음, 순수 함수).

    Args:
        query: 사용자 질문
        entities: 엔티티 링킹 결과
        cards: 이번 질의의 프롬프트 증거 카드

    Returns:
        EvidenceFit
    """
    card_list = list(cards or [])
    named = _normalized(entities, _NAMED_ENTITY_KEYS)
    needs = tuple(sorted(required_evidence_kinds(query, entities)))

    if not card_list:
        # 증거가 하나도 없으면 적합도를 말할 수 없다 — 바닥 점수로 두고 LLM에 맡긴다.
        return EvidenceFit(
            score=0.0,
            entity_coverage=0.0,
            kind_fit=0.0,
            retrieval_fit=0.0,
            needs=needs,
            named_entities=tuple(sorted(named)),
            matched_entities=(),
            card_count=0,
            basis="empty",
        )

    anchors_per_card = [_card_anchors(card) for card in card_list]
    all_anchors: set[str] = set()
    for anchors in anchors_per_card:
        all_anchors |= anchors

    matched = tuple(sorted(entity for entity in named if entity in all_anchors))
    entity_coverage = (len(matched) / len(named)) if named else 1.0

    satisfied = sum(
        1 for need in needs if _need_is_satisfied(need, card_list, anchors_per_card, named)
    )
    kind_fit = satisfied / len(needs)

    retrieval_fit = _retrieval_fit(card_list)

    score = (
        FIT_WEIGHTS["entity_coverage"] * entity_coverage
        + FIT_WEIGHTS["kind_fit"] * kind_fit
        + FIT_WEIGHTS["retrieval_fit"] * retrieval_fit
    )

    return EvidenceFit(
        score=score,
        entity_coverage=entity_coverage,
        kind_fit=kind_fit,
        retrieval_fit=retrieval_fit,
        needs=needs,
        named_entities=tuple(sorted(named)),
        matched_entities=matched,
        card_count=len(card_list),
        basis="fit",
    )


def _normalized(entities: Mapping[str, Sequence[str]] | None, keys: Sequence[str]) -> set[str]:
    """엔티티 링킹 결과에서 지정한 키의 값을 소문자 집합으로 모은다."""
    out: set[str] = set()
    if not entities:
        return out
    for key in keys:
        for value in entities.get(key) or ():
            if isinstance(value, str) and value.strip():
                out.add(value.strip().lower())
    return out


def _card_anchors(card: Evidence) -> set[str]:
    """이 카드가 대상으로 삼는 엔티티 식별자 집합 (주어·목적어·metadata 표기)."""
    anchors: set[str] = set()
    for value in (card.subject, card.object):
        if isinstance(value, str) and value:
            anchors.add(value.lower())

    metadata = card.metadata or {}
    for key in _ANCHOR_METADATA_KEYS:
        value = metadata.get(key)
        if isinstance(value, str) and value:
            anchors.add(value.lower())
    for key in _ANCHOR_METADATA_LIST_KEYS:
        for value in metadata.get(key) or ():
            if isinstance(value, str) and value:
                anchors.add(value.lower())
    return anchors


def _need_is_satisfied(
    need: str,
    cards: Sequence[Evidence],
    anchors_per_card: Sequence[set[str]],
    named: set[str],
) -> bool:
    """요구한 종류의 카드가, 질문이 부른 엔티티에 닻을 내린 채로 있는지 확인한다.

    질문이 아무 엔티티도 부르지 않았으면 닻을 요구할 수 없으므로 종류만 본다.
    """
    for card, anchors in zip(cards, anchors_per_card, strict=True):
        if card.kind.value != need:
            continue
        if need == EvidenceKind.DOCUMENT.value or not named:
            return True
        if anchors & named:
            return True
    return False


def _retrieval_fit(cards: Sequence[Evidence]) -> float:
    """문서 카드 검색 점수 분포의 뾰족함 ``(top - median) / top``.

    검색 점수는 경로(BM25·벡터·RRF)마다 눈금이 달라 절대값을 비교할 수 없다. 그래서
    눈금에 무관한 모양만 쓴다. 점수가 2개 미만이면 모양을 말할 수 없으므로 중립값.
    """
    scores = sorted(
        (
            float(card.metadata["score"])
            for card in cards
            if card.kind is EvidenceKind.DOCUMENT
            and isinstance(card.metadata.get("score"), int | float)
            and not isinstance(card.metadata.get("score"), bool)
        ),
        reverse=True,
    )
    if len(scores) < 2 or scores[0] <= 0:
        return _NEUTRAL_RETRIEVAL_FIT
    gap = (scores[0] - statistics.median(scores)) / scores[0]
    return max(0.0, min(1.0, gap))


# =============================================================================
# 신뢰도 사다리
# =============================================================================


class ConfidenceAssessor:
    """
    신뢰도 평가기

    적합도 점수(0~1)를 신뢰도 레벨로 바꾼다. 임계값은 이 클래스에만 있다.
    """

    # =========================================================================
    # 임계값 (0~1 적합도 눈금) — 측정 보정값
    # =========================================================================
    # scripts/calibrate_confidence_thresholds.py 가 233문항 타입 시험지의 보정 절반
    # (item_id sha1 % 2 == 0)에서 고른 값이다. 검증 절반 수치는
    # eval_output/evidence-2026-09/notes/5b_confidence_calibration.md 에 있다.
    THRESHOLD_HIGH = 0.85  # 증거로 바로 답한다 (LLM 판단 스킵)
    THRESHOLD_MEDIUM = 0.70  # LLM에게 도구 선택 위임
    THRESHOLD_LOW = 0.50  # LLM에게 전체 판단 위임
    # THRESHOLD_LOW 미만 → UNKNOWN (명확화 요청)

    def __init__(
        self,
        threshold_high: float | None = None,
        threshold_medium: float | None = None,
        threshold_low: float | None = None,
    ):
        """
        Args:
            threshold_high: HIGH 레벨 임계값 (0~1 적합도 눈금)
            threshold_medium: MEDIUM 레벨 임계값
            threshold_low: LOW 레벨 임계값
        """
        self.threshold_high = self.THRESHOLD_HIGH if threshold_high is None else threshold_high
        self.threshold_medium = (
            self.THRESHOLD_MEDIUM if threshold_medium is None else threshold_medium
        )
        self.threshold_low = self.THRESHOLD_LOW if threshold_low is None else threshold_low

    def assess_fit(self, fit: EvidenceFit) -> ConfidenceLevel:
        """적합도 측정 결과를 신뢰도 레벨로 바꾼다."""
        return self._score_to_level(fit.score)

    def assess(self, rule_result: dict[str, Any], context: Any = None) -> ConfidenceLevel:
        """
        신뢰도 레벨 평가

        Args:
            rule_result:
                - ``fit_score``: 적합도 점수(0~1). 있으면 이것을 쓴다.
                - ``max_score``: 개수 기반 옛 점수. 있으면 같은 사다리로 옮겨 쓴다.
                - ``confidence``: 0~1 상대 신뢰도 (옛 폴백, ×6.0으로 개수 점수 환산)
            context: Context 객체 (무시됨, 하위 호환용으로 유지)

        Returns:
            ConfidenceLevel

        Note:
            context 파라미터는 점수에 반영되지 않는다. 호출자가 이미 컨텍스트를 점수에
            반영하므로 여기서 다시 가산하면 이중 계산이다.
        """
        if "fit_score" in rule_result:
            return self._score_to_level(float(rule_result["fit_score"] or 0.0))

        score = rule_result.get("max_score", 0)
        if not score and "confidence" in rule_result:
            score = (rule_result["confidence"] or 0.0) * 6.0

        return self._score_to_level(legacy_count_score_to_fit(score))

    def _score_to_level(self, score: float) -> ConfidenceLevel:
        """
        적합도 점수를 신뢰도 레벨로 변환

        Args:
            score: 0~1 적합도 점수

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
            rule_result: ``assess()``와 같은 입력

        Returns:
            (ConfidenceLevel, 상세 정보 dict)
        """
        if "fit_score" in rule_result:
            score = float(rule_result["fit_score"] or 0.0)
        else:
            raw = rule_result.get("max_score", 0)
            if not raw and "confidence" in rule_result:
                raw = (rule_result["confidence"] or 0.0) * 6.0
            score = legacy_count_score_to_fit(raw)

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
            ConfidenceLevel.HIGH: "LLM 판단 스킵, 증거 카드로 바로 응답 생성",
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
# 옛 개수 점수 → 적합도 눈금
# =============================================================================

# 옛 사다리의 분기점(0 / 1.5 / 3.0 / 5.0 / 10.0)을 새 사다리의 같은 자리에 붙인다.
# 그래서 개수 점수를 쓰는 호출자의 레벨은 하나도 바뀌지 않는다.
_LEGACY_ANCHORS: tuple[tuple[float, float], ...] = (
    (0.0, 0.0),
    (1.5, ConfidenceAssessor.THRESHOLD_LOW),
    (3.0, ConfidenceAssessor.THRESHOLD_MEDIUM),
    (5.0, ConfidenceAssessor.THRESHOLD_HIGH),
    (10.0, 1.0),
)


def legacy_count_score_to_fit(score: float) -> float:
    """개수 기반 옛 점수(0~10)를 0~1 적합도 눈금으로 옮긴다 (구간별 선형).

    옛 분기점 5.0/3.0/1.5가 새 임계값 HIGH/MEDIUM/LOW에 정확히 대응하므로, 이 함수를 거친
    옛 점수는 옛 레벨을 그대로 받는다. 사다리를 둘로 늘리지 않으려는 어댑터이지 새 점수가
    아니다 — 새 코드는 ``score_evidence_fit()``을 쓴다.

    Args:
        score: 개수 기반 점수

    Returns:
        0~1 적합도 눈금 값
    """
    value = float(score or 0.0)
    if value <= 0.0:
        return 0.0
    if value >= _LEGACY_ANCHORS[-1][0]:
        return _LEGACY_ANCHORS[-1][1]

    for (low_raw, low_fit), (high_raw, high_fit) in zip(
        _LEGACY_ANCHORS, _LEGACY_ANCHORS[1:], strict=False
    ):
        if low_raw <= value <= high_raw:
            span = high_raw - low_raw
            ratio = (value - low_raw) / span if span else 0.0
            return low_fit + ratio * (high_fit - low_fit)
    return _LEGACY_ANCHORS[-1][1]


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
    매칭된 항목 수로 절대 점수 계산 (개수 기반 옛 눈금)

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
        절대 점수 (``legacy_count_score_to_fit()``으로 적합도 눈금에 올릴 수 있다)
    """
    return (
        matched_keywords * 2.0
        + matched_indicators * 1.5
        + matched_entities * 1.5
        + matched_patterns * 1.0
    )
