"""
Ontology Query Context (트랙 O3, 플래그 ``ontology.use_class_reasoning``) [2026-09 사후]
====================================================================================
질의 엔티티를 온톨로지(``src/ontology/ontology.py`` 단일 원본)로 해석해 조회 범위를 넓히고,
정적 정의 사실을 KG 사실 형식(``type = "ontology_static"``)으로 만든다. 플래그가 꺼져 있으면
``HybridRetriever``가 이 모듈을 부르지 않는다.

설계 (계획서 OE3·OE4·OE5·OE7, 결정 OA-6)
- **그룹·클래스 전개**: 그룹(``amorepacific``)·클래스(``LuxuryBrand`` 등 Brand 하위 클래스)·
  "같은 그룹/세그먼트" 힌트를 소속 브랜드 집합으로 넓힌다. 가짜 브랜드는 뺀다. 수치 조회용
  브랜드는 질의 브랜드 + 전개 브랜드 합계 ``MAX_EXPANDED_BRANDS``(12)개까지이고, 넘으면 뺀 브랜드를
  ``expansionTruncated`` 사실로 남긴다(순서: 크롤 DB 등장 → KG 크롤 관계 등장 → 이름).
- **정적 사실**(OE5): 질의 브랜드의 그룹·세그먼트·원산지·인수 연도, 전개 브랜드의 전개 근거
  술어 한 개. 출처는 ``ontology:registry``, 시점은 온톨로지 ``as_of``. 날짜 없는 수치 제외 규칙은
  여기에 적용하지 않는다(수치가 아니다).
- **닫힌 세계**(OE4): 그룹 소속·자매 관계를 묻는 질의에서 등록부 브랜드끼리는 "아님"을
  ``notOwnedByGroup``·``notSiblingBrand``로 판정한다. 등록부 밖 브랜드는 ``groupMembershipUnknown``
  ("모름")이며 "아님"으로 쓰지 않는다.
- **카테고리 포함**(OE3): 질의 카테고리의 하위 카테고리를 조회 범위(``scope_categories``)에만
  더한다. 수치를 상위 카테고리로 합산·환산하는 코드는 없다 — 수치 카드는 원래 카테고리를 유지한다.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

MAX_EXPANDED_BRANDS = 12
ONTOLOGY_SOURCE = "ontology:registry"
ONTOLOGY_FACT_TYPE = "ontology_static"

# 정적 정의 술어 (schema.json ``static: true``의 브랜드 술어). KG에서 오면 우선 노출 대상이다.
STATIC_BRAND_PREDICATES: frozenset[str] = frozenset(
    {"ownedByGroup", "ownsBrand", "siblingBrand", "hasSegment", "originatesFrom", "acquiredIn"}
)

# 닫힌 세계·전개 보조 술어 (온톨로지 카드 전용 — KG 트리플이 아니다)
NOT_OWNED_BY_GROUP = "notOwnedByGroup"
NOT_SIBLING_BRAND = "notSiblingBrand"
MEMBERSHIP_UNKNOWN = "groupMembershipUnknown"
EXPANSION_TRUNCATED = "expansionTruncated"

_SIBLING_HINTS = frozenset({"sibling", "siblings", "same_group", "group", "siblingbrand"})
_SEGMENT_HINTS = frozenset({"segment", "same_segment", "hassegment"})

BRAND_CLASS = "Brand"


@dataclass
class OntologyPlan:
    """질의 1건의 온톨로지 해석 결과 (결정적)."""

    mentioned_brands: list[str] = field(default_factory=list)  # 등록부 id, 언급 순서
    mentioned_raw: dict[str, str] = field(default_factory=dict)  # id → 질의 표기
    unknown_brands: list[str] = field(default_factory=list)  # 등록부 밖 표기
    groups: list[str] = field(default_factory=list)
    classes: list[str] = field(default_factory=list)
    hints: list[str] = field(default_factory=list)
    candidates: list[tuple[str, str, str]] = field(default_factory=list)  # (id, via, anchor)
    expanded_brands: list[str] = field(default_factory=list)
    dropped_brands: list[str] = field(default_factory=list)
    categories: list[str] = field(default_factory=list)  # 질의 카테고리 (정규화 id)
    descendant_categories: list[str] = field(default_factory=list)
    membership_query: bool = False

    @property
    def scope_categories(self) -> list[str]:
        """질의 카테고리 + 하위 카테고리 (조회 범위 전용, OE3)."""
        return _unique([*self.categories, *self.descendant_categories])

    @property
    def expansion_via(self) -> dict[str, tuple[str, str]]:
        """전개 브랜드 id → (전개 방식, 기준 개체)."""
        out: dict[str, tuple[str, str]] = {}
        for bid, via, anchor in self.candidates:
            out.setdefault(bid, (via, anchor))
        return out

    def summary(self) -> dict[str, Any]:
        return {
            "mentioned_brands": list(self.mentioned_brands),
            "unknown_brands": list(self.unknown_brands),
            "groups": list(self.groups),
            "classes": list(self.classes),
            "hints": list(self.hints),
            "expanded_brands": list(self.expanded_brands),
            "dropped_brands": list(self.dropped_brands),
            "scope_categories": self.scope_categories,
            "membership_query": self.membership_query,
        }


def _unique(items: Iterable[str]) -> list[str]:
    out: list[str] = []
    for item in items:
        if item and item not in out:
            out.append(item)
    return out


def _brand_classes(onto: Any) -> set[str]:
    return {c for c in onto.classes if c != BRAND_CLASS and BRAND_CLASS in onto.superclasses(c)}


def _real_brand(onto: Any, bid: str | None) -> bool:
    return bool(bid) and not onto.is_placeholder(bid) and onto.is_a(bid, BRAND_CLASS)


def plan_query(onto: Any, entities: Mapping[str, Sequence[str]] | None) -> OntologyPlan:
    """엔티티(연결기 출력 + O2의 ``classes``·``groups``·``brand_ids``·``relations_hint``) → 계획.

    전개 후보(``candidates``)까지만 만든다. 순위·상한은 ``apply_cap``이 정한다.
    """
    entities = entities or {}
    plan = OntologyPlan()

    for raw in entities.get("brands") or []:
        if not raw:
            continue
        bid = onto.normalize_brand(raw)
        gid = onto.normalize_group(raw)
        if bid is None and gid is not None:
            plan.groups = _unique([*plan.groups, gid])  # 연결기가 그룹을 브랜드로 낸 경우
        elif bid is None:
            plan.unknown_brands = _unique([*plan.unknown_brands, str(raw)])
        elif _real_brand(onto, bid) and bid not in plan.mentioned_brands:
            plan.mentioned_brands.append(bid)
            plan.mentioned_raw[bid] = str(raw)
    for raw in entities.get("brand_ids") or []:
        bid = onto.normalize_brand(raw)
        if _real_brand(onto, bid) and bid not in plan.mentioned_brands:
            plan.mentioned_brands.append(bid)
            plan.mentioned_raw[bid] = str(raw)
    for raw in entities.get("groups") or []:
        gid = onto.normalize_group(raw)
        if gid is not None:
            plan.groups = _unique([*plan.groups, gid])
    brand_classes = _brand_classes(onto)
    plan.classes = _unique(c for c in entities.get("classes") or [] if c in brand_classes)
    plan.hints = _unique(str(h).lower() for h in entities.get("relations_hint") or [] if h)

    sibling_hint = any(h in _SIBLING_HINTS for h in plan.hints)
    segment_hint = any(h in _SEGMENT_HINTS for h in plan.hints)
    group_classes = [
        c for c in plan.classes if (onto.class_spec(c).defined_by or ("", ""))[0] == "ownedByGroup"
    ]
    plan.membership_query = bool(plan.groups or group_classes or sibling_hint)

    # 전개 조건 (나열을 묻는 질의만 넓힌다):
    # - 그룹: 질의에 그 그룹 소속 브랜드가 함께 언급되지 않았을 때만. "아모레퍼시픽 포트폴리오에서
    #   COSRX의 세그먼트"의 그룹은 맥락이지 나열 요청이 아니다(rl006~rl011).
    # - 자매·세그먼트 힌트: 기준 브랜드가 하나일 때만("COSRX와 같은 그룹 브랜드", mh005).
    #   브랜드가 둘 이상이면 쌍 판정 질의다("LANEIGE와 TIRTIR는 자매 브랜드인가", rl013~rl016).
    # - 클래스(K-Beauty·Luxury 등): 항상 나열 요청으로 본다.
    single_anchor = len(plan.mentioned_brands) == 1
    candidates: list[tuple[str, str, str]] = []
    for gid in plan.groups:
        if any(onto.group_of(b) == gid for b in plan.mentioned_brands):
            continue
        candidates += [(b, "group", gid) for b in onto.brands_in_group(gid)]
    for cls in plan.classes:
        candidates += [(b, "class", cls) for b in onto.instances_of(cls)]
    if sibling_hint and single_anchor:
        for bid in plan.mentioned_brands:
            candidates += [(b, "sibling", bid) for b in onto.siblings(bid)]
    if segment_hint and single_anchor:
        allowed = {b for gid in plan.groups for b in onto.brands_in_group(gid)}
        for bid in plan.mentioned_brands:
            segment = onto.segment_of(bid)
            if segment is None:
                continue
            for other, seg in onto.relations("hasSegment"):
                if seg == segment and other != bid and (not allowed or other in allowed):
                    candidates.append((other, "segment", bid))
    seen = set(plan.mentioned_brands)
    for bid, via, anchor in candidates:
        if bid in seen or not _real_brand(onto, bid):
            continue
        seen.add(bid)
        plan.candidates.append((bid, via, anchor))

    for raw in entities.get("categories") or []:
        cid = onto.normalize_category(raw)
        if cid is not None:
            plan.categories = _unique([*plan.categories, cid])
    for cid in plan.categories:
        plan.descendant_categories = _unique(
            [*plan.descendant_categories, *onto.category_descendants(cid)]
        )
    plan.descendant_categories = [c for c in plan.descendant_categories if c not in plan.categories]
    return plan


def apply_cap(
    plan: OntologyPlan,
    rank_key: Callable[[str], tuple[Any, ...]] | None = None,
    max_brands: int = MAX_EXPANDED_BRANDS,
) -> OntologyPlan:
    """전개 후보를 순위대로 정렬해 (질의 브랜드 + 전개) ≤ ``max_brands``로 자른다.

    ``rank_key(id)``는 작을수록 먼저다(기본: 이름). 같은 키면 id 순이라 결정적이다.
    """
    ids = [bid for bid, _, _ in plan.candidates]
    ordered = sorted(ids, key=lambda b: (*(rank_key(b) if rank_key else ()), b))
    room = max(0, max_brands - len(plan.mentioned_brands))
    plan.expanded_brands = ordered[:room]
    plan.dropped_brands = ordered[room:]
    return plan


def _edge(subject: str, predicate: str, obj: str | None, **extra: Any) -> dict[str, Any]:
    edge: dict[str, Any] = {"subject": subject, "predicate": predicate, "object": obj}
    edge.update({k: v for k, v in extra.items() if v is not None})
    return edge


def _profile_edges(onto: Any, bid: str) -> list[dict[str, Any]]:
    edges = []
    group = onto.group_of(bid)
    if group:
        edges.append(_edge(bid, "ownedByGroup", group))
    segment = onto.segment_of(bid)
    if segment:
        edges.append(_edge(bid, "hasSegment", segment))
    origin = onto.origin_of(bid)
    if origin:
        edges.append(_edge(bid, "originatesFrom", origin))
    year = onto.acquired_in(bid)
    if year is not None:
        edges.append(_edge(bid, "acquiredIn", str(year)))
    return edges


def _expansion_edge(onto: Any, bid: str, via: str, anchor: str) -> dict[str, Any] | None:
    if via in ("group", "sibling"):
        group = onto.group_of(bid)
        return _edge(bid, "ownedByGroup", group) if group else None
    if via == "segment":
        segment = onto.segment_of(bid)
        return _edge(bid, "hasSegment", segment) if segment else None
    if via == "class":
        defined = onto.class_spec(anchor).defined_by
        if defined is None:
            return None
        predicate, value = defined
        return _edge(bid, predicate, value)
    return None


def static_edges(onto: Any, plan: OntologyPlan) -> list[dict[str, Any]]:
    """계획 → 정적 사실 엣지 목록 (결정적 순서, 중복 없음)."""
    edges: list[dict[str, Any]] = []
    for bid in plan.mentioned_brands:
        edges += _profile_edges(onto, bid)
    via = plan.expansion_via
    for bid in plan.expanded_brands:
        edge = _expansion_edge(onto, bid, *via[bid])
        if edge is not None:
            edges.append(edge)

    if plan.membership_query:
        groups = _unique(
            [*plan.groups, *(g for b in plan.mentioned_brands if (g := onto.group_of(b)))]
        )
        for bid in plan.mentioned_brands:
            for gid in groups:
                if onto.closed_world_member(bid, gid) is False:
                    edges.append(_edge(bid, NOT_OWNED_BY_GROUP, gid, closed_world=True))
        brands = plan.mentioned_brands
        for i, a in enumerate(brands):
            for b in brands[i + 1 :]:
                group_a, group_b = onto.group_of(a), onto.group_of(b)
                if group_a is not None and group_a == group_b:
                    edges.append(_edge(a, "siblingBrand", b))
                elif group_a is not None or group_b is not None:
                    # 둘 다 등록부 브랜드이고 적어도 한쪽 그룹이 명시됨 → 닫힌 세계로 "아님"
                    edges.append(_edge(a, NOT_SIBLING_BRAND, b, closed_world=True))
        for raw in plan.unknown_brands:
            for gid in groups:
                edges.append(_edge(raw, MEMBERSHIP_UNKNOWN, gid, unknown=True))

    if plan.dropped_brands:
        anchors = _unique(
            anchor for bid, _, anchor in plan.candidates if bid in plan.dropped_brands
        )
        edges.append(
            _edge(
                anchors[0] if anchors else "query",
                EXPANSION_TRUNCATED,
                None,
                count=len(plan.dropped_brands),
                dropped=list(plan.dropped_brands),
                kept=[*plan.mentioned_brands, *plan.expanded_brands],
            )
        )

    unique: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str | None]] = set()
    for edge in edges:
        key = (edge["subject"], edge["predicate"], edge["object"])
        if key not in seen:
            seen.add(key)
            unique.append(edge)
    return unique


def static_fact(onto: Any, plan: OntologyPlan) -> dict[str, Any] | None:
    """정적 사실 → KG 사실 1건 (``metric_edges``와 같은 ``data.edges`` 모양). 없으면 None."""
    edges = static_edges(onto, plan)
    if not edges:
        return None
    return {
        "type": ONTOLOGY_FACT_TYPE,
        "entity": ", ".join(plan.mentioned_brands or plan.groups or plan.classes) or "query",
        "data": {"edges": edges, "as_of": onto.as_of, "version": onto.version},
    }


def canonical_kg_predicate(onto: Any, enum_pred: str, original: str | None) -> str | None:
    """읽을 때 술어 정식화: ``hasPosition``은 ``original_predicate``로 나누고
    (``hasSoS``·``hasHHI``·``hasPricePosition``), 별칭은 정식 이름(``ownedBy`` → ``ownedByGroup``)으로.
    온톨로지가 모르는 술어는 None(호출자가 기존 표기를 쓴다)."""
    return onto.resolve_kg_predicate(enum_pred, original)
