"""
홉 수 기반 난이도 라우터 (트랙 5-C, 설계 E7)
=============================================

질문을 ReAct로 보낼지 파이프라인(direct/decide)으로 보낼지를 **홉 수**로 정한다.

홉이란
------
답을 만들기 위해 **앞 단계의 결과가 다음 단계의 입력이 되는** 조회 한 번이다. 단계는 네 가지다::

    entity_resolution  대상이 이름이 아니라 설명으로 지목돼 먼저 찾아야 한다
                       ("가장 강한 카테고리의", "1위 브랜드의", "그 제품이 속한")
    relation           KG 관계를 타야 대상(집합)이 정해진다
                       ("모회사", "경쟁사", "같은 그룹에 속한", "K-Beauty 브랜드 중")
    metric             크롤 DB 수치를 조회한다 (SoS·HHI·CPI·순위·가격·평점·리뷰)
    judgement          수치 위에 인과·전략 판정을 얹는다 ("왜", "이유", "달성하려면")
                       — "규칙 X를 충족하나요"는 여기 들어가지 않는다. apply_rules 한 번이
                       수치와 판정을 함께 주므로 조회가 꼬리를 물지 않는다

``hops = len(stages)``(최소 1)이고, ``hops >= HOP_THRESHOLD(2)``면 ReAct로 간다.

왜 키워드 복잡도가 아니라 홉 수인가
-----------------------------------
예전 분기(``QueryGraph._is_complex_query``)는 "비교·분석·왜·어떻게" 같은 **어조 키워드**와
"컨텍스트가 부족하다"를 섞어 복잡도를 판정했다. 그래서

* "Lip Care와 Lip Makeup HHI를 비교해주세요"처럼 **서로 의존하지 않는** 두 조회가 ReAct로
  갔고 (비교 = 키워드 히트),
* "LANEIGE 경쟁사 대비 가격 경쟁력은?"처럼 관계를 먼저 타야 하는 질문은 키워드가 없으면
  파이프라인으로 갔다.

홉 수는 "조회가 몇 번 **꼬리를 무는가**"만 본다. 비교·분석은 홉이 아니고(두 조회가 동시에
가능하다), 관계 → 수치처럼 결과가 다음 입력이 되면 홉이다.

규칙 우선, LLM은 폴백
---------------------
판정은 규칙(질문 구조·엔티티 지목 방식·관계/판정 표지)으로 한다. 규칙이 아무 신호도 찾지
못한 질문에 한해, 플래그 ``router.use_llm_fallback``(기본 OFF)이 켜졌을 때만 LLM에게 홉 수를
묻는다. LLM 판정은 질문별로 캐시하고(같은 질문에 두 번 돈을 쓰지 않는다) 근거를
``router_basis``에 ``llm``/``llm_cache``로 남긴다 — 어떤 문항이 LLM으로 갈렸는지 사후에
세어야 하기 때문이다.

라우터의 판정은 ``route_trace``에 ``hops``·``router_stages``·``router_reason``·
``router_basis``·``router_route``로 기록된다 (``QueryGraph._finalize_route_trace``).
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any

from src.shared.constants import DEFAULT_MODEL

logger = logging.getLogger(__name__)

# ── 단계 이름 ────────────────────────────────────────────────────────────
STAGE_ENTITY = "entity_resolution"
STAGE_RELATION = "relation"
STAGE_METRIC = "metric"
STAGE_JUDGEMENT = "judgement"

STAGE_ORDER: tuple[str, ...] = (STAGE_ENTITY, STAGE_RELATION, STAGE_METRIC, STAGE_JUDGEMENT)

# 이 홉 수 이상이면 ReAct. 1홉(=단일 조회)은 파이프라인이 더 싸고 더 빠르다.
HOP_THRESHOLD = 2

# LLM 폴백이 낼 수 있는 홉 수 범위 (단계가 넷뿐이므로 4를 넘을 수 없다)
MIN_HOPS = 1
MAX_HOPS = len(STAGE_ORDER)


# ── 표지 (모두 소문자 질문 문자열에 대해 검사한다) ───────────────────────

# 수치 조회 표지. 영문 약어는 **ASCII 경계**만 본다: ``\b``를 쓰면 "SoS는"처럼 한글 조사가
# 붙었을 때 한글도 단어 문자라 경계가 서지 않아 매칭이 통째로 실패한다.
_METRIC_WORDS_ASCII = re.compile(
    r"(?<![a-z0-9])(sos|hhi|cpi|bsr|rank|ranking|price|rating|reviews?|share|top)(?![a-z0-9])"
)
_METRIC_WORDS_KO: tuple[str, ...] = (
    "점유율",
    "점유",
    "집중도",
    "시장집중",
    "가격경쟁력",
    "순위",
    "랭킹",
    "가격",
    "평점",
    "리뷰",
    "평균",
    "몇 개",
    "개수",
    "수치",
    "지표",
    "상위",
    "현황",
    "매출",
    "판매량",
    "수준은",
)

# 관계(KG)를 타야 대상이 정해지는 표지
_RELATION_WORDS: tuple[str, ...] = (
    "경쟁사",
    "경쟁 제품",
    "경쟁제품",
    "경쟁 관계",
    "경쟁관계",
    "competitor",
    "모회사",
    "소속",
    "그룹",
    "계열사",
    "포트폴리오",
    "세그먼트",
    "자매",
    "인수",
    "k-beauty",
    "k뷰티",
    "케이뷰티",
    "브랜드 중",
    "카테고리 중",
    "같은 카테고리",
    "같은 그룹",
    "제품 구성",
    "속한",
)
_RELATION_PATTERNS: tuple[re.Pattern[str], ...] = (
    # "…제품의 카테고리", "…브랜드의 카테고리" — 소속 카테고리를 먼저 찾아야 한다
    re.compile(r"(제품|브랜드)의\s*카테고리"),
)

# 인과·전략 판정 표지 (수치를 받아 든 뒤에 한 단계 더 생각해야 하는 질문)
#
# 여기에 "규칙 …을 충족하나요" 같은 **명시적 규칙 판정**은 넣지 않는다. 그 판정은
# ``apply_rules`` 도구 한 번이 수치 계산과 함께 돌려주므로 조회가 꼬리를 물지 않는다.
# "…는 어떤 의미인가요" 같은 해석 질문도 마찬가지로 문서 한 번이면 끝난다.
_JUDGEMENT_WORDS: tuple[str, ...] = (
    "왜",
    "이유",
    "원인",
    "때문",
    "전략",
    "가능성",
    "하려면",
    "경로는",
    "시사",
    "대응 방안",
    "개선 방안",
)

# 대상을 설명으로 지목하는 표지 (선택자). "상위"는 넣지 않는다 —
# "SoS 상위 브랜드는?"처럼 한 번의 조회로 끝나는 목록 질문에 쓰이기 때문이다.
_SELECTORS = r"(?:가장|제일|최고|최상위|1위|대표|주력)"
_TYPE_NOUNS = r"(?:브랜드|제품|카테고리|기업|회사)"

_ENTITY_PATTERNS: tuple[re.Pattern[str], ...] = (
    # "LANEIGE 제품이 속한 …", "Neo Cushion이 속한 …" — 대상을 관계로 먼저 찾아야 한다
    re.compile(r"(?:이|가)\s*속한"),
    # "가장 강한 카테고리의", "1위 브랜드의", "대표 제품이 속한", "가장 높은 순위의 제품을 가진"
    re.compile(
        _SELECTORS + r"[^?]{0,20}?" + _TYPE_NOUNS + r"(?:의|이\s*속한|가\s*속한|[을를]\s*가진)"
    ),
    # "가장 심한 카테고리" — 브랜드가 속한 카테고리 집합을 먼저 추려야 고를 수 있다
    # (get_metrics는 카테고리별로 따로 불러야 한다)
    re.compile(_SELECTORS + r"[^?]{0,20}카테고리"),
    # 앞 단계 결과를 가리키는 지시어
    re.compile(r"해당\s*(?:카테고리|기업|브랜드|제품|지표)"),
    re.compile(r"그\s+(?:제품|브랜드|카테고리|기업|순위|가격|지표|sos|hhi|cpi)"),
)

# 선택자가 있으면 "무엇이 가장 큰가"를 재야 하므로 수치 조회가 따라온다
_SELECTOR_PATTERN = re.compile(_SELECTORS)


@dataclass(frozen=True)
class RouteDecision:
    """라우터 판정 (불변 — 기록용 값이지 상태가 아니다)."""

    hops: int
    stages: tuple[str, ...]
    reason: str
    basis: str  # "rules" | "llm" | "llm_cache"

    @property
    def use_react(self) -> bool:
        return self.hops >= HOP_THRESHOLD

    @property
    def route(self) -> str:
        return "react" if self.use_react else "pipeline"

    def to_trace(self) -> dict[str, Any]:
        """``route_trace``에 합칠 관측 필드."""
        return {
            "hops": self.hops,
            "router_stages": list(self.stages),
            "router_reason": self.reason,
            "router_basis": self.basis,
            "router_route": self.route,
        }


def _found(text: str, words: tuple[str, ...]) -> list[str]:
    return [w for w in words if w in text]


class HopRouter:
    """질문 → 홉 수 → 경로.

    Args:
        model: LLM 폴백에 쓸 모델
        llm_fallback: ``None``이면 피처 플래그(``router.use_llm_fallback``)를 읽는다.
            테스트는 명시적으로 켜고 끈다.
    """

    LLM_PROMPT = """질문에 답하려면 서로 의존하는 조회를 몇 번 해야 하는지만 판단하세요.

조회 단계는 넷입니다:
1. 엔티티 확정 (대상이 이름이 아니라 설명으로 지목된 경우)
2. 관계 조회 (모회사·경쟁사·소속 카테고리 등 지식그래프 관계)
3. 수치 조회 (SoS·HHI·CPI·순위·가격·평점)
4. 판정 (규칙 적용, 인과 설명)

앞 조회의 결과가 다음 조회의 입력이 될 때만 홉을 셉니다.
서로 독립적인 두 조회(예: 두 카테고리의 HHI 비교)는 1홉입니다.

질문: {query}

JSON만 출력하세요: {{"hops": 1-4, "reason": "한 줄 근거"}}"""

    def __init__(self, model: str = DEFAULT_MODEL, llm_fallback: bool | None = None) -> None:
        self._model = model
        self._llm_fallback = llm_fallback
        self._llm_cache: dict[str, RouteDecision] = {}

    # ── 규칙 판정 ────────────────────────────────────────────────────

    def analyze(self, query: str) -> RouteDecision:
        """규칙만으로 판정한다 (동기, LLM 호출 없음)."""
        text = (query or "").strip().lower()
        if len(text) < 2:
            return RouteDecision(
                hops=MIN_HOPS, stages=(), reason="질문이 비어 있거나 너무 짧다", basis="rules"
            )

        evidence: dict[str, list[str]] = {}
        stages: list[str] = []

        entity_hits = [p.pattern for p in _ENTITY_PATTERNS if p.search(text)]
        if entity_hits:
            stages.append(STAGE_ENTITY)
            evidence[STAGE_ENTITY] = entity_hits

        relation_hits = _found(text, _RELATION_WORDS) + [
            p.pattern for p in _RELATION_PATTERNS if p.search(text)
        ]
        if relation_hits:
            stages.append(STAGE_RELATION)
            evidence[STAGE_RELATION] = relation_hits

        metric_hits = _found(text, _METRIC_WORDS_KO)
        metric_hits += [m.group(0) for m in _METRIC_WORDS_ASCII.finditer(text)]
        has_selector = bool(_SELECTOR_PATTERN.search(text))
        if not metric_hits and has_selector:
            # "가장 강한 브랜드는?" — 무엇이 가장 큰지 재려면 수치를 봐야 한다
            metric_hits = ["선택자(수치 비교 필요)"]
        if metric_hits:
            stages.append(STAGE_METRIC)
            evidence[STAGE_METRIC] = metric_hits

        judgement_hits = _found(text, _JUDGEMENT_WORDS)
        if judgement_hits:
            stages.append(STAGE_JUDGEMENT)
            evidence[STAGE_JUDGEMENT] = judgement_hits

        ordered = tuple(s for s in STAGE_ORDER if s in stages)
        hops = max(len(ordered), MIN_HOPS)
        return RouteDecision(
            hops=hops, stages=ordered, reason=self._reason(ordered, evidence), basis="rules"
        )

    @staticmethod
    def _reason(stages: tuple[str, ...], evidence: dict[str, list[str]]) -> str:
        if not stages:
            return "단계 표지 없음 — 단일 조회로 본다"
        parts = [f"{stage}({', '.join(evidence.get(stage, [])[:3])})" for stage in stages]
        return " → ".join(parts)

    # ── LLM 폴백 ─────────────────────────────────────────────────────

    def _fallback_enabled(self) -> bool:
        if self._llm_fallback is not None:
            return self._llm_fallback
        from src.infrastructure.feature_flags import FeatureFlags

        return bool(FeatureFlags.get_instance().use_router_llm_fallback())

    async def route(self, query: str) -> RouteDecision:
        """규칙 판정. 규칙이 아무 표지도 못 찾았고 폴백이 켜져 있을 때만 LLM에 묻는다."""
        decision = self.analyze(query)
        if decision.stages or not self._fallback_enabled():
            return decision

        key = " ".join((query or "").strip().lower().split())
        cached = self._llm_cache.get(key)
        if cached is not None:
            return RouteDecision(
                hops=cached.hops, stages=cached.stages, reason=cached.reason, basis="llm_cache"
            )

        llm_decision = await self._ask_llm(query)
        if llm_decision is None:
            return decision
        self._llm_cache[key] = llm_decision
        return llm_decision

    async def _ask_llm(self, query: str) -> RouteDecision | None:
        from litellm import acompletion

        try:
            response = await acompletion(
                model=self._model,
                messages=[{"role": "user", "content": self.LLM_PROMPT.format(query=query)}],
                max_tokens=120,
                temperature=0.0,
            )
            content = response.choices[0].message.content or ""
            start, end = content.find("{"), content.rfind("}") + 1
            payload = json.loads(content[start:end]) if start >= 0 else {}
            hops = int(payload.get("hops", MIN_HOPS))
        except Exception as e:  # 폴백 실패는 규칙 판정으로 조용히 되돌린다
            logger.warning(f"router LLM fallback failed: {e}")
            return None

        hops = max(MIN_HOPS, min(MAX_HOPS, hops))
        reason = str(payload.get("reason") or "").strip() or "LLM 판정"
        return RouteDecision(hops=hops, stages=(), reason=f"LLM: {reason}", basis="llm")
