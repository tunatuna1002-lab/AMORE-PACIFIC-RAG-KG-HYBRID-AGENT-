"""
Evidence Assembly
=================
검색 결과 → 증거 카드 전체(``evidence``) → 답변 프롬프트에 싣는 카드(``prompt_evidence``).
(트랙 2-B, 설계 E1·E2·E8)

v4(``HybridRetriever._combine_contexts``)와 v1(``ContextBuilder.build``)이 이 모듈 하나로
카드를 만들고 고르고, ``evidence_renderer.render_for_prompt``로 같은 형식을 렌더링한다.

- 카드 변환은 ``EvidenceAdapter``가 한다(단위·브랜드 정규화, KG 수치 엣지 제외).
- 종류 하나의 변환이 실패해도 나머지 종류는 계속 만든다. 실패는 ``degraded``에 남긴다
  (0-B 규칙: 선택 기능 실패는 핵심 검색 실패가 아니다).
- ``prompt_evidence``는 ``select_cards`` 결과 그대로다 — 렌더 입력과 정확히 같은 목록.

선택 상한과 metric 우선순위
---------------------------
실측(원본 DB·KG 복사본, as_of 2026-08-31, 트랙 2-B 보고): 브랜드+카테고리 질의 1건에
metric 카드가 40~50장, 브랜드만 링크된 질의(카테고리 3개로 확장)는 130장 안팎이 나온다.
제품 카드는 제품 1개당 4장(순위·가격·평점·리뷰 수)이라 가장 많다.

metric 카드는 아래 그룹 순서로 정렬한 뒤 그룹 상한을 적용하고, 마지막에 종류별 상한을 건다.
그룹 안에서는 어댑터 출력 순서(= MetricFactsProvider의 카테고리 순서)를 지킨다.

1. 카테고리 시장 지표 (HHI·평균가·평균 평점) — 질의 카테고리마다 3~4장
2. 질의 브랜드 지표 (SoS·Top100 제품 수·SoS 순위·부재)
3. 상위 브랜드 SoS (질의 브랜드가 아닌 브랜드의 ``sos``만)
4. 질의 브랜드 제품 (순위·가격·평점·리뷰 수)
5. 카테고리 상위 제품
6. 그 밖의 수치 (상위 브랜드의 Top100 제품 수·SoS 순위 — 운영 DB에서 SoS가 Top100 제품 수
   비율이라 SoS 카드와 정보가 겹친다)

relation 카드는 술어 우선순위(``RELATION_PREDICATE_PRIORITY``)로 정렬한 뒤 상한을 건다.
어댑터 출력 순서(= ``_weighted_merge``가 정렬한 KG 사실 순서)대로 자르면 경쟁 관계
(브랜드 질의 1건에 최대 25장)와 ASIN 제품 관계(최대 20장)가 먼저 차서, 실측 lg158("LANEIGE
브랜드의 모회사…")에서 ``laneige ownedBy AMOREPACIFIC`` 카드가 전체 카드에는 있는데
프롬프트에서는 잘렸다. 소유·계층·카테고리 소속처럼 수가 적고 질의 특정성이 높은 관계를 앞에 둔다.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from src.domain.entities.evidence import Evidence, EvidenceKind, EvidenceSet
from src.rag.evidence_adapters import EvidenceAdapter
from src.rag.evidence_renderer import select_cards

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# 선택 상한 (근거는 모듈 docstring과 트랙 2-B 보고의 5질의 측정)
# ----------------------------------------------------------------------

# 종류별 상한. DOCUMENT는 두지 않는다 — 검색 top_k(컨텍스트 예산 8, retrieval_strategy)가
# 이미 상한이다.
PROMPT_MAX_PER_KIND: dict[EvidenceKind, int] = {
    # 한 카테고리 전체(시장 3 + 질의 브랜드 3 + 상위 브랜드 SoS 5 + 질의 브랜드 제품 12
    # + 상위 제품 5개×4 = 43)가 거의 다 들어가는 값. 카테고리 3개 질의는 앞 그룹부터 채운다.
    EvidenceKind.METRIC: 40,
    # 관계 카드 1장은 한 줄 40자 안팎이다. 브랜드 1개 질의의 경쟁·소유·계층 관계가 다 들어간다.
    EvidenceKind.RELATION: 20,
    # retrieval_weights.json max_context_items.inferences와 같다.
    EvidenceKind.INFERENCE: 5,
}

METRIC_GROUP_MARKET = 1
METRIC_GROUP_QUERY_BRAND = 2
METRIC_GROUP_TOP_BRAND_SOS = 3
METRIC_GROUP_QUERY_BRAND_PRODUCT = 4
METRIC_GROUP_TOP_PRODUCT = 5
METRIC_GROUP_OTHER = 6

# 그룹 상한 (없으면 종류별 상한만 적용). 상위 브랜드 SoS는 카테고리 2개분(5×2)까지만 —
# 카테고리 3개로 확장된 브랜드 질의에서 이 그룹이 질의 브랜드 제품(4그룹)을 밀어내지 않게.
METRIC_GROUP_CAPS: dict[int, int] = {
    METRIC_GROUP_TOP_BRAND_SOS: 10,
    METRIC_GROUP_TOP_PRODUCT: 20,  # 상위 제품 5개 × 4장
}

# relation 술어 우선순위 (작을수록 먼저, 없는 술어는 RELATION_PRIORITY_DEFAULT)
RELATION_PREDICATE_PRIORITY: dict[str, int] = {
    "ownedBy": 0,  # 소유 (시드 온톨로지, 브랜드당 1장)
    "parentCategory": 1,  # 카테고리 계층
    "hasSubcategory": 1,
    "rankedIn": 2,  # 브랜드가 카테고리 순위에 오름
    "competesWith": 3,
    "directCompetitor": 3,
    "indirectCompetitor": 3,
    "brandSentiment": 4,
    "hasSentiment": 4,
    "hasAISummary": 4,
    "hasProduct": 5,  # ASIN 코드 관계 — 수가 많고 이름이 없다
    "belongsToCategory": 5,
    "hasTrend": 6,  # 브랜드 트렌드가 없으면 시장 전체 트렌드로 대체된 값일 수 있다
}
RELATION_PRIORITY_DEFAULT = 9

_MARKET_SOURCE = "sqlite:market_metrics"
_BRAND_SOURCE = "sqlite:brand_metrics"
_PRODUCT_SOURCE = "sqlite:raw_data"


@dataclass
class EvidenceBundle:
    """카드 조립 결과.

    Attributes:
        evidence: 이번 질의의 전체 카드 (id 중복 없음, 순서 결정적).
        prompt_evidence: 답변 프롬프트에 렌더링할 카드 (``evidence``의 부분 목록).
        excluded: 어댑터가 증거에서 뺀 KG 항목 (수치 엣지·날짜 없는 메타데이터 등).
        degraded: 카드 변환 실패 ``{"component", "error"}``.
    """

    evidence: list[Evidence] = field(default_factory=list)
    prompt_evidence: list[Evidence] = field(default_factory=list)
    excluded: list[dict[str, Any]] = field(default_factory=list)
    degraded: list[dict[str, Any]] = field(default_factory=list)

    @property
    def excluded_by_reason(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for item in self.excluded:
            reason = str(item.get("reason"))
            counts[reason] = counts.get(reason, 0) + 1
        return counts


def metric_group(card: Evidence, query_brands: frozenset[str]) -> int:
    """metric 카드의 우선순위 그룹 (작을수록 먼저)."""
    if card.source == _MARKET_SOURCE:
        return METRIC_GROUP_MARKET
    if card.source == _BRAND_SOURCE:
        if card.subject in query_brands:
            return METRIC_GROUP_QUERY_BRAND
        if card.predicate == "sos":
            return METRIC_GROUP_TOP_BRAND_SOS
        return METRIC_GROUP_OTHER
    if card.source == _PRODUCT_SOURCE:
        if card.metadata.get("brand") in query_brands:
            return METRIC_GROUP_QUERY_BRAND_PRODUCT
        return METRIC_GROUP_TOP_PRODUCT
    return METRIC_GROUP_OTHER


def order_metric_cards(
    cards: Sequence[Evidence], query_brands: Iterable[str]
) -> tuple[list[Evidence], list[Evidence]]:
    """metric 카드를 그룹 순으로 정렬한다.

    Returns:
        (전체 정렬 목록, 그룹 상한을 통과한 정렬 목록). 그룹 안의 순서는 입력 순서.
    """
    brands = frozenset(query_brands)
    ordered = sorted(cards, key=lambda card: metric_group(card, brands))  # 안정 정렬
    counts: dict[int, int] = {}
    within_caps: list[Evidence] = []
    for card in ordered:
        group = metric_group(card, brands)
        cap = METRIC_GROUP_CAPS.get(group)
        if cap is not None and counts.get(group, 0) >= cap:
            continue
        counts[group] = counts.get(group, 0) + 1
        within_caps.append(card)
    return ordered, within_caps


def order_relation_cards(cards: Sequence[Evidence]) -> list[Evidence]:
    """relation 카드를 술어 우선순위로 안정 정렬한다 (같은 순위 안은 입력 순서)."""
    return sorted(
        cards,
        key=lambda card: RELATION_PREDICATE_PRIORITY.get(card.predicate, RELATION_PRIORITY_DEFAULT),
    )


def _convert(
    component: str,
    degraded: list[dict[str, Any]],
    convert: Callable[[], Any],
    default: Any,
) -> Any:
    try:
        return convert()
    except Exception as exc:  # 한 종류의 변환 실패가 다른 종류의 카드를 막지 않게
        logger.warning(f"evidence conversion failed: {component}", exc_info=True)
        degraded.append({"component": component, "error": f"{type(exc).__name__}: {exc}"})
        return default


def build_rule_input_cards(
    *,
    metric_facts: Iterable[dict[str, Any]] = (),
    ontology_facts: Iterable[dict[str, Any]] = (),
    adapter: EvidenceAdapter | None = None,
    degraded: list[dict[str, Any]] | None = None,
) -> list[Evidence]:
    """규칙 추론 입력 카드 (트랙 3-B): DB 수치 → metric 카드, KG 사실 → relation 카드.

    추론 **전에** 만든다. ``_weighted_merge``가 KG 사실을 자르기 전의 사실 전부를 쓴다 —
    규칙 입력이 프롬프트 선별 상한에 따라 달라지지 않게. 변환 실패는 ``degraded``에 남기고
    나머지 종류는 계속 만든다 (0-B).
    """
    adapter = adapter or EvidenceAdapter()
    failures = degraded if degraded is not None else []
    metric_cards = _convert(
        "rule_input_metric", failures, lambda: adapter.from_metric_facts(metric_facts), []
    )
    kg_result = _convert(
        "rule_input_relation", failures, lambda: adapter.from_kg_facts(ontology_facts), None
    )
    relation_cards = kg_result.cards if kg_result is not None else []
    return EvidenceSet([*metric_cards, *relation_cards]).to_list()


def assemble_evidence(
    *,
    entities: Mapping[str, Sequence[str]] | None = None,
    metric_facts: Iterable[dict[str, Any]] = (),
    ontology_facts: Iterable[dict[str, Any]] = (),
    inferences: Iterable[Any] = (),
    rag_chunks: Iterable[dict[str, Any]] = (),
    adapter: EvidenceAdapter | None = None,
    max_per_kind: Mapping[EvidenceKind, int] | None = None,
    input_cards: Iterable[Evidence] = (),
) -> EvidenceBundle:
    """검색 결과 필드 → 카드 전체·프롬프트 카드.

    ``evidence`` 순서: metric(그룹 우선순위 순) → relation(술어 우선순위 순) → inference → document.

    inference 카드의 ``derived_from``은 추론 결과의 ``evidence["derived_from"]``(규칙 입력
    카드 id)이다. 그 근거 카드가 ``ontology_facts``(상한으로 잘린 최종 사실)에서 다시 만들어지지
    않으면 ``input_cards``(추론 전에 만든 입력 카드)에서 가져와 relation·metric 카드에 더한다 —
    ``evidence`` 안에서 모든 ``derived_from`` id를 찾을 수 있게.
    """
    adapter = adapter or EvidenceAdapter()
    limits = PROMPT_MAX_PER_KIND if max_per_kind is None else max_per_kind
    bundle = EvidenceBundle()
    degraded = bundle.degraded

    brands = [str(brand) for brand in (entities or {}).get("brands") or [] if brand]
    query_brands = _convert(
        "evidence_query_brands", degraded, lambda: [adapter.normalize_brand(b) for b in brands], []
    )

    metric_cards = _convert(
        "evidence_metric", degraded, lambda: adapter.from_metric_facts(metric_facts), []
    )
    kg_result = _convert(
        "evidence_relation", degraded, lambda: adapter.from_kg_facts(ontology_facts), None
    )
    relation_cards = order_relation_cards(kg_result.cards) if kg_result is not None else []
    if kg_result is not None:
        bundle.excluded = list(kg_result.excluded)
    inference_cards = _convert(
        "evidence_inference", degraded, lambda: adapter.from_inferences(inferences), []
    )
    document_cards = _convert(
        "evidence_document", degraded, lambda: adapter.from_rag_chunks(rag_chunks), []
    )

    known_ids = {card.id for card in [*metric_cards, *relation_cards]}
    needed_ids = {i for card in inference_cards for i in card.derived_from} - known_ids
    basis_cards = EvidenceSet(card for card in input_cards if card.id in needed_ids).to_list()
    if basis_cards:
        metric_cards = [*metric_cards, *(c for c in basis_cards if c.kind is EvidenceKind.METRIC)]
        relation_cards = order_relation_cards(
            [*relation_cards, *(c for c in basis_cards if c.kind is EvidenceKind.RELATION)]
        )

    ordered_metrics, capped_metrics = order_metric_cards(metric_cards, query_brands)

    cards = EvidenceSet()
    stored: dict[int, Evidence] = {}  # id(원본 카드) → 실제로 담긴 카드 (id 충돌 시 늘어난 id)
    for card in [*ordered_metrics, *relation_cards, *inference_cards, *document_cards]:
        stored[id(card)] = cards.add(card)
    bundle.evidence = cards.to_list()

    candidates = [
        stored[id(card)]
        for card in [*capped_metrics, *relation_cards, *inference_cards, *document_cards]
    ]
    bundle.prompt_evidence = select_cards(candidates, limits)
    return bundle


def evidence_source_labels(cards: Iterable[Evidence]) -> list[str]:
    """카드 → 출처 표시 문자열 (중복 제거, 카드 순서).

    - document: 문서 제목 (없으면 카드 text)
    - metric: ``sqlite:<table> (as_of)``
    - relation: ``KG``
    - inference·observation: 카드 source (``rule:<이름>``, ``tool:<이름>``)
    """
    labels: list[str] = []
    for card in cards:
        if card.kind == EvidenceKind.DOCUMENT:
            label = str(card.metadata.get("title") or card.text)
        elif card.kind == EvidenceKind.METRIC:
            label = f"{card.source} ({card.as_of})" if card.as_of else card.source
        elif card.kind == EvidenceKind.RELATION:
            label = "KG"
        else:
            label = card.source
        if label not in labels:
            labels.append(label)
    return labels
