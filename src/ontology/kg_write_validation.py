"""
KG 쓰기 검증 (트랙 O5, 결정 OA-7) [2026-09 사후]

KG에 트리플을 쓸 때 온톨로지(``src/ontology/ontology.py``)로 검사·정규화한다.
``KnowledgeGraph.add_relation``(쓰기 경로)과 ``scripts/migrate_kg_ontology.py``(기존 KG 정리)가
같은 규칙을 쓰도록 순수 함수로 둔다.

모드 (문자열 플래그 ``kg.write_validation``, ENV ``FF_KG_WRITE_VALIDATION``):

- ``off``: 검사하지 않는다(예전 동작 그대로).
- ``warn`` (기본): 위반을 세고 사유별 첫 예시 몇 개만 로그로 남긴다. **저장 내용은 off와 같다.**
  매일 크롤이 운영 KG에 쓰는 경로라 기본값이 저장 내용을 바꾸면 OE6(운영 데이터 불변)과 어긋난다.
- ``enforce``: 아래 정규화를 적용하고, 막아야 할 트리플은 저장하지 않고 센다.

enforce 규칙 (``normalize_triple``):

1. 술어 정식화. ``hasPosition``은 ``properties.original_predicate``로 ``hasSoS``·``hasHHI``·
   ``hasPricePosition``으로 나누고, ``belongsToCategory``(original ``rankedIn``)는 ``rankedIn``으로,
   ``ownedBy``는 ``ownedByGroup``으로 바꾼다. 원래 이름은 ``original_predicate``에 남긴다
   (이미 있으면 덮지 않는다). 나눌 수 없는 ``hasPosition``(예: ``DOMINATES_CATEGORY``)은 막는다.
   카테고리 계층 별칭 ``parentCategory``·``hasSubcategory``는 저장 이름을 유지한다 — KG의
   계층 API(``get_category_hierarchy``)가 그 enum 값으로 조회하기 때문이다. 읽기 쪽(O3)이 정식화한다.
   온톨로지 밖 술어(``hasAlert``·``hasSentiment`` 등)는 손대지 않는다.
2. 브랜드 정식 표기. 술어 정의상 Brand 자리(도메인/범위)에 오는 문자열만 바꾼다.
   **정식 KG 문자열 = 등록부 표시 이름의 소문자** (``LANEIGE``→``laneige``, ``elf``→``e.l.f.``,
   ``Beauty of Joseon``→``beauty of joseon``). 이유: 질의 경로
   ``hybrid_retriever._query_knowledge_graph``는 엔티티 링커가 낸 이름(등록부식 표기
   ``LANEIGE``·``e.l.f.``·``La Roche-Posay``)의 ``원형·lower·upper·title`` 4변형을 조회한다.
   소문자 표시 이름은 그중 ``lower`` 변형과 항상 같다. 등록부 id(``elf``·``beauty_of_joseon``)는
   ``e.l.f.``·``Beauty of Joseon``의 어느 변형과도 같지 않아 쓰지 않는다. 또 enricher가 이미
   소문자로 쓰고 있어 기존 트리플 대부분이 그대로 남는다. 등록부 밖 브랜드는 그대로 둔다.
3. 가짜 브랜드(``PlaceholderBrand``: ``unknown``·``fresh``·``chi``)가 들어간 트리플은 막는다.
4. 수치 술어(``requires_as_of``)는 ``as_of``가 있어야 한다. 쓰는 쪽이 아는 날짜(``as_of`` 인자)나
   속성의 날짜 키에서만 채운다. 없으면 막는다(마이그레이션은 ``on_missing_as_of="keep"``로
   남기고 ``undated_numeric``으로 센다 — 날짜를 지어내지 않는다).
5. 도메인·범위·리터럴 위반(개체 타입을 알 때만 판정)은 막는다.
6. 타입 기록: ``entity_metadata[entity]``에 ``type``(기존 표기 ``brand``·``category``·``product``·
   ``corporate_group``)과 ``ontology_types``(폐포 후 클래스 목록)를 **없을 때만** 넣는다.
   KG 파일 형식(``entity_metadata``는 원래 있는 키)은 바뀌지 않는다.
"""

from __future__ import annotations

import logging
import os
import re
from collections import Counter
from collections.abc import Iterable, Mapping, MutableMapping
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

MODES: tuple[str, ...] = ("off", "warn", "enforce")
DEFAULT_MODE = "warn"
ENV_NAME = "FF_KG_WRITE_VALIDATION"

# 저장 이름을 유지하는 별칭 → 정식 이름 (KG 계층 API가 저장 이름으로 조회한다)
KEPT_STORED_ALIASES: dict[str, str] = {
    "parentCategory": "subCategoryOf",
    "hasSubcategory": "hasSubCategory",
}

# as_of로 볼 수 있는 속성 키 (값이 그 날짜의 관측임을 뜻하는 것만)
AS_OF_KEYS: tuple[str, ...] = ("as_of", "snapshot_date", "collected_at")

_ASIN_RE = re.compile(r"^B0[A-Z0-9]{8}$")

_LEGACY_TYPE_NAMES: tuple[tuple[str, str], ...] = (
    ("Brand", "brand"),
    ("CorporateGroup", "corporate_group"),
    ("Category", "category"),
    ("Product", "product"),
    ("Segment", "segment"),
    ("Country", "country"),
)

BLOCKING_CODES = frozenset(
    {"placeholder_brand", "domain_violation", "range_violation", "literal_violation"}
)


# ----------------------------------------------------------------------
# Mode flag
# ----------------------------------------------------------------------


def _coerce_mode(raw: Any, source: str) -> str:
    mode = str(raw).strip().lower()
    if mode in MODES:
        return mode
    logger.warning(
        "Invalid KG write validation mode %r from %s; using %r", raw, source, DEFAULT_MODE
    )
    return DEFAULT_MODE


def get_write_validation_mode() -> str:
    """``off`` | ``warn`` | ``enforce``. ENV > ``config/feature_flags.json`` ``kg.write_validation``
    > 기본 ``warn``.

    ``FeatureFlags.kg_write_validation_mode()``가 있으면(다른 트랙이 추가) 그 값을 쓴다.
    """
    try:
        from src.infrastructure.feature_flags import FeatureFlags

        flags = FeatureFlags.get_instance()
    except Exception:  # pragma: no cover - feature flag module unavailable
        flags = None

    getter = getattr(flags, "kg_write_validation_mode", None) if flags is not None else None
    if callable(getter):
        return _coerce_mode(getter(), "FeatureFlags.kg_write_validation_mode")

    raw = os.environ.get(ENV_NAME)
    if raw is not None:
        return _coerce_mode(raw, ENV_NAME)
    config = getattr(flags, "_config", None)
    section = config.get("kg") if isinstance(config, dict) else None
    if isinstance(section, dict) and section.get("write_validation") is not None:
        return _coerce_mode(section["write_validation"], "config kg.write_validation")
    return DEFAULT_MODE


# ----------------------------------------------------------------------
# Pure helpers
# ----------------------------------------------------------------------


def canonical_brand_string(onto: Any, text: str | None) -> str | None:
    """등록부 브랜드 → 정식 KG 문자열(표시 이름 소문자). 등록부 밖이면 None."""
    if not text:
        return None
    bid = onto.normalize_brand(text)
    if bid is None:
        return None
    name = onto.brand_name(bid)
    return name.lower() if name else None


_legacy_cache: dict[str, bool] = {}


def _is_legacy_predicate(onto: Any, predicate: str) -> bool:
    cached = _legacy_cache.get(predicate)
    if cached is None:
        msgs = onto.validate_triple("_", predicate, "_")
        cached = bool(msgs) and msgs[0].startswith("legacy KG predicate")
        _legacy_cache[predicate] = cached
    return cached


def resolve_predicate(
    onto: Any, predicate: str, properties: Mapping[str, Any] | None
) -> tuple[str | None, bool]:
    """(정식 술어 또는 None, 레거시 분리 술어 여부)."""
    original = (properties or {}).get("original_predicate")
    canonical = onto.resolve_kg_predicate(predicate, original)
    if canonical is not None:
        return canonical, False
    return None, _is_legacy_predicate(onto, predicate)


def _brand_roles(spec: Any) -> tuple[bool, bool]:
    subject_is_brand = spec.domain == "Brand"
    object_is_brand = spec.kind == "object" and spec.range == "Brand"
    return subject_is_brand, object_is_brand


def _classify(message: str) -> str:
    if "placeholder brand" in message:
        return "placeholder_brand"
    if message.startswith("domain violation"):
        return "domain_violation"
    if message.startswith("range violation"):
        return "range_violation"
    if "requires an as_of" in message:
        return "missing_as_of"
    if message.startswith("unknown predicate"):
        return "outside_ontology"
    return "literal_violation"


def entity_types(onto: Any, entity: Any) -> tuple[str, ...]:
    """온톨로지가 아는 개체의 클래스 목록. ASIN은 ``Product``. 모르면 빈 튜플."""
    if not isinstance(entity, str) or not entity:
        return ()
    types = onto.types_of(entity)
    if types:
        return tuple(types)
    if _ASIN_RE.match(entity):
        return ("Product",)
    return ()


def legacy_type_name(types: Iterable[str]) -> str | None:
    tset = set(types)
    for cls, name in _LEGACY_TYPE_NAMES:
        if cls in tset:
            return name
    return None


def derive_as_of(properties: Mapping[str, Any]) -> str | None:
    for key in AS_OF_KEYS:
        value = properties.get(key)
        if value:
            return str(value)
    return None


# ----------------------------------------------------------------------
# warn: check
# ----------------------------------------------------------------------


def check_triple(
    onto: Any,
    subject: str,
    predicate: str,
    obj: Any,
    properties: Mapping[str, Any] | None = None,
) -> list[str]:
    """트리플의 위반 사유 코드(정렬, 중복 없음). 빈 목록이면 위반 없음.

    ``outside_ontology``는 정보용(온톨로지가 정의하지 않은 술어)이며 위반으로 세지 않는다.
    """
    props = properties or {}
    canonical, legacy = resolve_predicate(onto, predicate, props)
    if canonical is None:
        return ["unresolvable_legacy_predicate"] if legacy else ["outside_ontology"]

    codes: set[str] = set()
    if predicate != canonical and predicate not in KEPT_STORED_ALIASES:
        codes.add("non_canonical_predicate")

    spec = onto.predicate_spec(canonical)
    s_brand, o_brand = _brand_roles(spec)
    subject_c, obj_c = subject, obj
    for is_brand, value, role in ((s_brand, subject, "s"), (o_brand, obj, "o")):
        if not is_brand or not isinstance(value, str) or onto.is_placeholder(value):
            continue
        canon = canonical_brand_string(onto, value)
        if canon is not None and canon != value:
            codes.add("non_canonical_brand")
            if role == "s":
                subject_c = canon
            else:
                obj_c = canon

    for message in onto.validate_triple(subject_c, canonical, obj_c, props):
        codes.add(_classify(message))
    codes.discard("outside_ontology")
    return sorted(codes)


# ----------------------------------------------------------------------
# enforce: normalize
# ----------------------------------------------------------------------


@dataclass
class NormalizedTriple:
    subject: str
    predicate: str
    object: Any
    properties: dict[str, Any]
    blocked: str | None = None
    changes: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    types: dict[str, tuple[str, ...]] = field(default_factory=dict)


def normalize_triple(
    onto: Any,
    subject: str,
    predicate: str,
    obj: Any,
    properties: Mapping[str, Any] | None = None,
    *,
    as_of: str | None = None,
    on_missing_as_of: str = "block",
) -> NormalizedTriple:
    """enforce 규칙을 적용한 트리플. 입력 ``properties``는 바꾸지 않는다.

    Args:
        as_of: 쓰는 쪽이 아는 관측 날짜(크롤 날짜). 수치 술어에 ``as_of``가 없을 때만 쓴다.
        on_missing_as_of: ``block``(쓰기 경로) | ``keep``(마이그레이션: 남기고 ``undated_numeric``).
    """
    props = dict(properties or {})
    result = NormalizedTriple(subject, predicate, obj, props)

    canonical, legacy = resolve_predicate(onto, predicate, props)
    if canonical is None:
        if legacy:
            result.blocked = "unresolvable_legacy_predicate"
        else:
            result.notes.append("outside_ontology")
        return result

    if predicate != canonical and predicate not in KEPT_STORED_ALIASES:
        result.predicate = canonical
        props.setdefault("original_predicate", predicate)
        result.changes.append("predicate_canonicalized")

    spec = onto.predicate_spec(canonical)
    s_brand, o_brand = _brand_roles(spec)
    for is_brand, role in ((s_brand, "subject"), (o_brand, "object")):
        value = getattr(result, role)
        if not is_brand or not isinstance(value, str):
            continue
        if onto.is_placeholder(value):
            result.blocked = "placeholder_brand"
            return result
        canon = canonical_brand_string(onto, value)
        if canon is None:
            result.notes.append("unregistered_brand")
        elif canon != value:
            setattr(result, role, canon)
            result.changes.append("brand_canonicalized")

    if spec.requires_as_of and not props.get("as_of"):
        derived = as_of or derive_as_of(props)
        if derived:
            props["as_of"] = derived
            result.changes.append("as_of_added")
        elif on_missing_as_of == "block":
            result.blocked = "missing_as_of"
            return result
        else:
            result.notes.append("undated_numeric")

    for message in onto.validate_triple(result.subject, canonical, result.object, props):
        code = _classify(message)
        if code in BLOCKING_CODES:
            result.blocked = code
            return result

    for value, check in ((result.subject, True), (result.object, spec.kind == "object")):
        if check:
            types = entity_types(onto, value)
            if types:
                result.types[value] = types
    return result


def record_types(
    entity_metadata: MutableMapping[str, dict[str, Any]],
    types: Mapping[str, tuple[str, ...]],
) -> int:
    """``type``·``ontology_types``를 없을 때만 넣는다. 바뀐 엔티티 수를 돌려준다."""
    changed = 0
    for entity in sorted(types):
        classes = list(types[entity])
        legacy = legacy_type_name(classes)
        meta = entity_metadata.get(entity)
        additions: dict[str, Any] = {}
        if legacy and (meta is None or "type" not in meta):
            additions["type"] = legacy
        if meta is None or "ontology_types" not in meta:
            additions["ontology_types"] = classes
        if additions:
            entity_metadata.setdefault(entity, {}).update(additions)
            changed += 1
    return changed


# ----------------------------------------------------------------------
# Aggregated logging
# ----------------------------------------------------------------------


class WriteValidationStats:
    """사유별 건수 + 사유별 첫 예시 ``max_examples``개 (크롤 한 번에 로그 수천 줄 방지)."""

    def __init__(self, max_examples: int = 3) -> None:
        self.max_examples = max_examples
        self.counts: Counter[str] = Counter()
        self.examples: dict[str, list[str]] = {}

    def record(self, codes: Iterable[str], example: str) -> list[str]:
        """건수를 올리고, 예시를 새로 저장한 사유 코드 목록을 돌려준다(로그 대상)."""
        fresh: list[str] = []
        for code in codes:
            self.counts[code] += 1
            bucket = self.examples.setdefault(code, [])
            if len(bucket) < self.max_examples:
                bucket.append(example)
                fresh.append(code)
        return fresh

    def summary(self) -> dict[str, Any]:
        return {
            "counts": dict(sorted(self.counts.items())),
            "examples": {k: list(v) for k, v in sorted(self.examples.items())},
        }
