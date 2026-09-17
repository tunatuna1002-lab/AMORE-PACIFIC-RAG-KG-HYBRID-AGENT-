"""
Evidence Adapters
=================
검색·추론·도구 결과를 증거 카드(``src.domain.entities.evidence.Evidence``)로 바꾼다.

설계 E1·E2 (docs/plans/evidence-react-ontology-kickoff-prompt-2026-09-17.md §3):

- 역할: KG = 구조적 관계, SQLite = 날짜 붙은 수치의 정본, 문서 = 정의·해석.
- **단위 정규화는 여기서 한 번만 한다.** ``brand_metrics.sos``는 퍼센트(0~100)로 저장돼
  있다(운영 DB 2025-12-16~2026-09-17 전 구간 확인: min 1.0, max 18.33, 날짜·카테고리별 합
  ≤ 100.16). 카드의 SoS 정본은 0~1 ``ratio``다. ``market_metrics.hhi``는 이미 0~1이다.
- **브랜드·카테고리 canonical id도 여기서 만든다** (결함 F13). 원표기는
  ``metadata["display_name"]``에 남긴다.
- **KG 수치 엣지(hasSoS·hasHHI·hasPosition·hasRank)는 증거로 쓰지 않는다** (결함 F12).
  운영 KG(2026-09-17)의 수치 엣지 359개(hasSoS 169·가격 포지션 124·hasHHI 66)는 전부
  ``valid_from``이 없고 속성에도 스냅샷 날짜가 없다. 같은 주어에 서로 다른 값이 쌓여 있다
  (예: lip_care hasHHI 15개 값, 0~10000 스케일). 제외한 엣지는 ``KGEvidenceResult.excluded``에
  남긴다. 날짜 버전이 붙은 수치 엣지가 생기면 그때 metric 카드 변환을 추가한다.
- 값이 없으면(None·빈 문자열) 카드를 만들지 않는다. 결측을 0으로 채우지 않는다.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

from src.domain.entities.evidence import (
    Evidence,
    EvidenceKind,
    EvidenceSet,
    EvidenceUnit,
    EvidenceValue,
)
from src.domain.entities.relations import InferenceResult
from src.rag.evidence_renderer import format_metric_text

logger = logging.getLogger(__name__)

Normalizer = Callable[[str], str]

MAX_OBSERVATION_CHARS = 4000
_TEXT_SUMMARY_CHARS = 120

# 날짜 버전 없이 저장된 KG 수치 엣지 술어 (E2·F12). hybrid_retriever가 original_predicate를
# 우선해 hasSoS·hasHHI로 노출하고, 가격 포지션은 hasPosition으로 노출한다.
KG_NUMERIC_PREDICATES: frozenset[str] = frozenset({"hasSoS", "hasHHI", "hasPosition", "hasRank"})

KG_SOURCE = "kg"
EXCLUDED_NUMERIC = "kg_numeric_undated"
EXCLUDED_METADATA = "kg_entity_metadata_undated"
EXCLUDED_PLACEHOLDER = "placeholder_entity"

# 브랜드 추출에 실패한 제품을 kg_enricher가 묶어 둔 자리표시자 (운영 KG 주어 254개).
# "unknown competesWith laneige"는 어떤 브랜드에 대한 사실도 아니다.
PLACEHOLDER_ENTITIES: frozenset[str] = frozenset({"unknown"})

_ASIN_PATTERN = re.compile(r"^(?=.*\d)[A-Z0-9]{10}$")

# 술어별 (주어 역할, 목적어 역할). 역할이 정규화 함수를 고른다.
_ROLE_BRAND = "brand"
_ROLE_CATEGORY = "category"
_ROLE_ENTITY = "entity"
_ROLE_RAW = "raw"  # 감성 태그·트렌드 키워드처럼 어휘 그대로 두는 값

_PREDICATE_ROLES: dict[str, tuple[str, str]] = {
    "ownedBy": (_ROLE_BRAND, _ROLE_ENTITY),
    "ownedByGroup": (_ROLE_BRAND, _ROLE_ENTITY),
    "rankedIn": (_ROLE_BRAND, _ROLE_CATEGORY),
    "competesWith": (_ROLE_BRAND, _ROLE_BRAND),
    "directCompetitor": (_ROLE_BRAND, _ROLE_BRAND),
    "indirectCompetitor": (_ROLE_BRAND, _ROLE_BRAND),
    "hasProduct": (_ROLE_BRAND, _ROLE_ENTITY),
    "belongsToCategory": (_ROLE_ENTITY, _ROLE_CATEGORY),
    "parentCategory": (_ROLE_CATEGORY, _ROLE_CATEGORY),
    "hasSubcategory": (_ROLE_CATEGORY, _ROLE_CATEGORY),
    "hasSentiment": (_ROLE_ENTITY, _ROLE_RAW),
    "brandSentiment": (_ROLE_BRAND, _ROLE_RAW),
    "hasTrend": (_ROLE_ENTITY, _ROLE_RAW),
    "hasAISummary": (_ROLE_ENTITY, _ROLE_RAW),
}

# MetricFactsProvider의 market 필드 → (술어, 단위)
_MARKET_FIELDS: tuple[tuple[str, str, str], ...] = (
    ("hhi", "hhi", EvidenceUnit.INDEX_0_1),
    ("churn_rate", "churn_rate", EvidenceUnit.RATIO),  # MetricCalculator: 0~1 (운영 DB는 전부 NULL)
    ("category_avg_price", "avg_price", EvidenceUnit.USD),
    ("category_avg_rating", "avg_rating", EvidenceUnit.RATING_5),
)

# raw_data 제품 필드 → (술어, 단위)
_PRODUCT_FIELDS: tuple[tuple[str, str, str], ...] = (
    ("rank", "bsr_rank", EvidenceUnit.RANK),
    ("price", "price", EvidenceUnit.USD),
    ("rating", "rating", EvidenceUnit.RATING_5),
    ("reviews_count", "reviews_count", EvidenceUnit.COUNT),
)

# 추론 결론 값으로 쓸 규칙 메타데이터 키 (우선순위 순)
_CONCLUSION_KEYS = ("position", "market_structure", "market_type")


def _collapse(name: str) -> str:
    return " ".join(str(name).split())


@lru_cache(maxsize=1)
def _brand_map() -> dict[str, str]:
    """EntityLinker의 브랜드 단어사전: {소문자 이름·별칭: canonical id}.

    ``EntityExtractor.get_brand_normalization_map``(hybrid_retriever)이 감싸는 것과 같은 정본
    (config/entities.json ``known_brands`` + ``EntityLinker.KNOWN_BRANDS``)이다. hybrid_retriever를
    import하면 2-B 배선 때 순환 import가 되므로 EntityLinker를 직접 쓴다.
    """
    from src.rag.entity_linker import EntityLinker

    return dict(EntityLinker(use_spacy=False)._get_merged_brands())


@lru_cache(maxsize=1)
def _category_map() -> dict[str, str]:
    """EntityLinker의 카테고리 단어사전: {소문자 표기: category_id}."""
    from src.rag.entity_linker import EntityLinker

    return dict(EntityLinker(use_spacy=False)._get_merged_categories())


def default_brand_normalizer(name: str) -> str:
    """브랜드 canonical id: 단어사전에 있으면 그 id(``LANEIGE``·``라네즈`` → ``laneige``),
    없으면 공백을 정리한 소문자(DB·kg_enricher가 쓰는 소문자 표기와 같다)."""
    key = _collapse(name).lower()
    return _brand_map().get(key, key)


def default_category_normalizer(name: str) -> str:
    """카테고리 id: 단어사전(``Lip Care``·``립케어`` → ``lip_care``), 없으면 snake_case 소문자."""
    key = _collapse(name).lower()
    mapped = _category_map().get(key)
    if mapped:
        return mapped
    return re.sub(r"[\s\-]+", "_", key)


def _number(value: Any) -> int | float | None:
    """수치만 통과시킨다. None·빈 문자열·숫자가 아닌 문자열·bool은 결측으로 본다."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return value
    if isinstance(value, str):
        text = value.strip().replace(",", "")
        if not text:
            return None
        try:
            number = float(text)
        except ValueError:
            return None
        return int(number) if number.is_integer() else number
    return None


def _sos_ratio(percent: Any) -> float | None:
    """DB의 퍼센트 SoS(0~100) → 0~1. 범위를 벗어나면 잘못된 입력으로 보고 버린다."""
    number = _number(percent)
    if number is None:
        return None
    if not 0 <= number <= 100:
        logger.warning(f"SoS {number} outside 0..100 percent scale — card skipped")
        return None
    return round(number / 100, 6)


def _pred_value(predicate: Any) -> str:
    return str(predicate.value) if hasattr(predicate, "value") else str(predicate)


def _first_line(text: str, limit: int = _TEXT_SUMMARY_CHARS) -> str:
    for line in text.splitlines():
        stripped = line.strip().lstrip("#").strip()
        if stripped:
            return stripped[:limit]
    return ""


@dataclass(frozen=True)
class KGEvidenceResult:
    """KG 사실 변환 결과.

    Attributes:
        cards: relation 카드 (id 중복 제거, 입력 순서).
        excluded: 증거에서 뺀 항목 ``{fact_type, subject, predicate, object, reason}``.
        skipped: 변환 규칙이 없는 사실 type별 개수.
    """

    cards: list[Evidence]
    excluded: list[dict[str, Any]] = field(default_factory=list)
    skipped: dict[str, int] = field(default_factory=dict)


class EvidenceAdapter:
    """검색·추론·도구 결과 → 증거 카드.

    Args:
        brand_normalizer: 브랜드 표기 → canonical id. 기본은 EntityLinker 단어사전.
        category_normalizer: 카테고리 표기 → category_id. 기본은 EntityLinker 단어사전.
    """

    def __init__(
        self,
        brand_normalizer: Normalizer | None = None,
        category_normalizer: Normalizer | None = None,
    ) -> None:
        self._brand = brand_normalizer or default_brand_normalizer
        self._category = category_normalizer or default_category_normalizer

    # ------------------------------------------------------------------
    # 정규화
    # ------------------------------------------------------------------

    def normalize_brand(self, name: str) -> str:
        return self._brand(name)

    def normalize_category(self, name: str) -> str:
        return self._category(name)

    def _normalize(self, name: Any, role: str) -> str:
        text = _collapse(str(name))
        if role == _ROLE_BRAND:
            return self._brand(text)
        if role == _ROLE_CATEGORY:
            return self._category(text)
        if role == _ROLE_RAW or _ASIN_PATTERN.match(text):
            return text
        return text.lower()

    @staticmethod
    def _display(original: Any, canonical: str) -> dict[str, Any]:
        text = _collapse(str(original))
        return {"display_name": text} if text and text != canonical else {}

    # ------------------------------------------------------------------
    # 크롤 DB 수치 (MetricFactsProvider.collect)
    # ------------------------------------------------------------------

    def from_metric_facts(self, facts: Iterable[dict[str, Any]]) -> list[Evidence]:
        """``MetricFactsProvider.collect`` 결과 → metric 카드 (중복 제거, 입력 순서)."""
        cards = EvidenceSet()
        for fact in facts:
            kind = fact.get("type")
            category = self._category(str(fact["category"])) if fact.get("category") else None
            as_of = fact.get("snapshot_date")
            if kind == "category_market":
                cards.extend(self._market_cards(fact, category, as_of))
            elif kind == "category_top_brands":
                entries = [b for b in fact.get("brands") or [] if _number(b.get("sos")) is not None]
                for entry in entries:
                    sos = _number(entry.get("sos"))
                    rank = 1 + sum(1 for other in entries if _number(other.get("sos")) > sos)
                    cards.extend(
                        self._share_cards(
                            entry.get("brand"),
                            category,
                            as_of,
                            sos,
                            entry.get("product_count"),
                            rank,
                        )
                    )
            elif kind == "brand_share":
                if fact.get("present"):
                    cards.extend(
                        self._share_cards(
                            fact.get("brand"),
                            category,
                            as_of,
                            fact.get("sos"),
                            fact.get("product_count"),
                            fact.get("brand_rank"),
                        )
                    )
                else:
                    cards.add(self._absent_card(fact.get("brand"), category, as_of))
            elif kind in ("category_top_products", "brand_products"):
                for product in fact.get("products") or []:
                    cards.extend(self._product_cards(product, category, as_of))
            else:
                logger.debug(f"metric fact type {kind!r} has no evidence mapping")
        return cards.to_list()

    def _metric(
        self,
        subject: str,
        predicate: str,
        value: EvidenceValue,
        unit: str,
        as_of: str | None,
        table: str,
        object: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Evidence:
        meta = dict(metadata or {})
        display = meta.get("display_name") or subject
        return Evidence.create(
            kind=EvidenceKind.METRIC,
            subject=subject,
            predicate=predicate,
            object=object,
            value=value,
            unit=unit,
            as_of=as_of,
            source=f"sqlite:{table}",
            confidence=1.0,  # 크롤 DB에 기록된 관측값 그대로
            text=format_metric_text(display, predicate, object, value, unit),
            metadata=meta,
        )

    def _market_cards(
        self, fact: dict[str, Any], category: str | None, as_of: str | None
    ) -> list[Evidence]:
        if not category:
            return []  # 어느 시장의 수치인지 모르면 증거가 아니다
        cards = []
        for key, predicate, unit in _MARKET_FIELDS:
            value = _number(fact.get(key))
            if value is None:
                continue
            cards.append(self._metric(category, predicate, value, unit, as_of, "market_metrics"))
        return cards

    def _share_cards(
        self,
        brand: Any,
        category: str | None,
        as_of: str | None,
        sos: Any,
        product_count: Any,
        rank: Any,
    ) -> list[Evidence]:
        if not brand:
            return []
        subject = self._brand(str(brand))
        meta = self._display(brand, subject)
        cards = []
        values = (
            ("sos", _sos_ratio(sos), EvidenceUnit.RATIO),
            ("product_count", _number(product_count), EvidenceUnit.COUNT),
            ("sos_rank", _number(rank), EvidenceUnit.RANK),
        )
        for predicate, value, unit in values:
            if value is None:
                continue
            cards.append(
                self._metric(
                    subject, predicate, value, unit, as_of, "brand_metrics", category, meta
                )
            )
        return cards

    def _absent_card(self, brand: Any, category: str | None, as_of: str | None) -> Evidence:
        subject = self._brand(str(brand))
        return self._metric(
            subject,
            "present_in_top100",
            False,
            EvidenceUnit.BOOLEAN,
            as_of,
            "brand_metrics",
            category,
            self._display(brand, subject),
        )

    def _product_cards(
        self, product: dict[str, Any], category: str | None, as_of: str | None
    ) -> list[Evidence]:
        name = _collapse(product.get("name") or "")
        if not name:
            return []
        meta: dict[str, Any] = {}
        if product.get("brand"):
            meta["brand"] = self._brand(str(product["brand"]))
            meta["brand_display_name"] = _collapse(str(product["brand"]))
        cards = []
        for key, predicate, unit in _PRODUCT_FIELDS:
            value = _number(product.get(key))
            if value is None:
                continue
            cards.append(
                self._metric(name, predicate, value, unit, as_of, "raw_data", category, meta)
            )
        return cards

    # ------------------------------------------------------------------
    # KG 사실 (HybridRetriever._query_knowledge_graph)
    # ------------------------------------------------------------------

    def from_kg_facts(self, facts: Iterable[dict[str, Any]]) -> KGEvidenceResult:
        """``HybridRetriever._query_knowledge_graph`` 결과 → relation 카드.

        수치 엣지와 엔티티 메타데이터(날짜 없는 대시보드 수치)는 ``excluded``로 뺀다.
        """
        builder = _RelationBuilder(self)
        for fact in facts:
            kind = str(fact.get("type"))
            entity = fact.get("entity")
            data = fact.get("data")
            handler = _KG_HANDLERS.get(kind)
            if handler is None or entity is None:
                builder.skipped[kind] = builder.skipped.get(kind, 0) + 1
                continue
            handler(builder, entity, data)
        if builder.excluded:
            logger.info(
                f"KG evidence: {len(builder.excluded)} numeric/metadata items excluded (E2)"
            )
        return builder.result()

    # ------------------------------------------------------------------
    # 문서 청크
    # ------------------------------------------------------------------

    def from_rag_chunks(self, chunks: Iterable[dict[str, Any]]) -> list[Evidence]:
        """검색 결과 ``{"id", "content", "metadata", "score"}`` → document 카드.

        id는 chunk_id 기반이다(같은 청크면 점수·검색 경로가 달라도 같은 카드).
        chunk_id가 없으면 본문 sha1을 기반으로 한다.
        """
        cards = EvidenceSet()
        for chunk in chunks:
            metadata = dict(chunk.get("metadata") or {})
            content = str(chunk.get("content") or "")
            chunk_id = chunk.get("id") or metadata.get("chunk_id")
            if chunk_id:
                basis = f"chunk:{chunk_id}"
            elif content:
                basis = "content:" + hashlib.sha1(content.encode("utf-8")).hexdigest()
            else:
                logger.debug("RAG chunk without id and content skipped")
                continue
            doc_type = metadata.get("doc_type")
            title = _collapse(str(metadata.get("title") or "")) or _first_line(content)
            card_meta = {
                key: metadata[key]
                for key in ("doc_type", "title", "source_filename", "content_type")
                if metadata.get(key)
            }
            card_meta["chunk_id"] = chunk_id
            if chunk.get("score") is not None:
                card_meta["score"] = chunk["score"]
            cards.add(
                Evidence.create(
                    kind=EvidenceKind.DOCUMENT,
                    subject=str(metadata.get("doc_id") or chunk_id or "document"),
                    predicate="states",
                    source=f"rag:{doc_type}" if doc_type else "rag",
                    text=title or str(chunk_id),
                    detail=content or None,
                    metadata=card_meta,
                    id_basis=basis,
                )
            )
        return cards.to_list()

    # ------------------------------------------------------------------
    # 규칙 추론
    # ------------------------------------------------------------------

    def from_inferences(
        self,
        results: Iterable[InferenceResult],
        derived_from: Sequence[str] | None = None,
        derived_from_resolver: Callable[[InferenceResult], Sequence[str]] | None = None,
        as_of: str | None = None,
    ) -> list[Evidence]:
        """``OntologyReasoner`` 결과 → inference 카드.

        - subject: 규칙 입력의 브랜드(canonical) → 없으면 카테고리 → related_entities[0].
        - predicate: 인사이트 유형 (``market_dominance`` 등).
        - object: 적용 범위(브랜드 주어일 때 카테고리). 같은 규칙이 카테고리만 달리 발화해도
          id가 겹치지 않게 결론이 아니라 범위를 둔다.
        - value: 결론 값 (규칙 메타데이터의 position·market_structure·market_type).
        - derived_from: ``derived_from_resolver(result)``가 우선, 없으면 ``derived_from``.
          규칙 입력 계약(3단계)이 생기기 전까지는 호출자가 근거 카드 id를 넘긴다.
        """
        cards = EvidenceSet()
        for result in results:
            snapshot = dict((result.evidence or {}).get("context_snapshot") or {})
            brand = snapshot.get("brand")
            category = snapshot.get("category")
            related = [e for e in result.related_entities or [] if e]
            if brand:
                subject = self._brand(str(brand))
                display = self._display(brand, subject)
                scope = self._category(str(category)) if category else None
            elif category:
                subject, display, scope = self._category(str(category)), {}, None
            elif related:
                subject = self._normalize(related[0], _ROLE_ENTITY)
                display, scope = self._display(related[0], subject), None
            else:
                subject, display, scope = "market", {}, None

            metadata = result.metadata or {}
            value = next((metadata[k] for k in _CONCLUSION_KEYS if metadata.get(k)), None)
            if derived_from_resolver is not None:
                basis = list(derived_from_resolver(result))
            else:
                basis = list(derived_from or [])

            confidence = float(result.confidence)
            if not 0.0 <= confidence <= 1.0:
                logger.warning(f"rule {result.rule_name} confidence {confidence} clamped to 0..1")
                confidence = min(1.0, max(0.0, confidence))

            card_as_of = as_of or snapshot.get("as_of") or snapshot.get("snapshot_date")
            insight_type = _pred_value(result.insight_type)
            cards.add(
                Evidence.create(
                    kind=EvidenceKind.INFERENCE,
                    subject=subject,
                    predicate=insight_type,
                    object=scope,
                    value=str(value) if value is not None else None,
                    as_of=card_as_of,
                    source=f"rule:{result.rule_name}",
                    confidence=confidence,
                    derived_from=basis,
                    text=_collapse(result.insight) or f"{result.rule_name} 발화",
                    detail=result.recommendation or None,
                    metadata={
                        **display,
                        "rule_name": result.rule_name,
                        "insight_type": insight_type,
                        "satisfied_conditions": list(
                            (result.evidence or {}).get("satisfied_conditions") or []
                        ),
                        "related_entities": related,
                    },
                )
            )
        return cards.to_list()

    # ------------------------------------------------------------------
    # 도구 관찰
    # ------------------------------------------------------------------

    def from_tool_observation(
        self,
        tool_name: str,
        tool_input: dict[str, Any] | None,
        observation: Any,
        as_of: str | None = None,
    ) -> Evidence:
        """ReAct 도구 호출 1회 → observation 카드.

        id는 (도구, 정렬된 입력, as_of, 관찰 원문 sha1) 기반 — 같은 호출이 같은 결과를 내면
        같은 카드다. 원문은 ``MAX_OBSERVATION_CHARS``에서 자른다.
        """
        input_json = json.dumps(tool_input or {}, ensure_ascii=False, sort_keys=True, default=str)
        if isinstance(observation, str):
            raw = observation
        else:
            raw = json.dumps(observation, ensure_ascii=False, sort_keys=True, default=str)
        truncated = len(raw) > MAX_OBSERVATION_CHARS
        basis = json.dumps(
            [tool_name, input_json, as_of, hashlib.sha1(raw.encode("utf-8")).hexdigest()],
            ensure_ascii=False,
        )
        args = ", ".join(f"{k}={v}" for k, v in sorted((tool_input or {}).items()))
        summary = _first_line(raw, 80)
        return Evidence.create(
            kind=EvidenceKind.OBSERVATION,
            subject=tool_name,
            predicate="observed",
            source=f"tool:{tool_name}",
            as_of=as_of,
            text=f"{tool_name}({args}) → {summary}" if summary else f"{tool_name}({args})",
            detail=raw[:MAX_OBSERVATION_CHARS] if raw else None,
            metadata={
                "tool_input": tool_input or {},
                "truncated": truncated,
                "observation_chars": len(raw),
            },
            id_basis=basis,
        )


class _RelationBuilder:
    """KG 사실 한 묶음을 relation 카드로 모은다 (경쟁사 카테고리 병합 포함)."""

    def __init__(self, adapter: EvidenceAdapter) -> None:
        self.adapter = adapter
        self.cards = EvidenceSet()
        self.excluded: list[dict[str, Any]] = []
        self.skipped: dict[str, int] = {}
        # (subject, predicate, object) → 병합 중인 카드 인자
        self._pending: dict[tuple[str, str, str | None], dict[str, Any]] = {}
        self._order: list[tuple[str, str, str | None]] = []

    def exclude(self, fact_type: str, subject: Any, predicate: str, obj: Any, reason: str) -> None:
        self.excluded.append(
            {
                "fact_type": fact_type,
                "subject": subject,
                "predicate": predicate,
                "object": obj,
                "reason": reason,
            }
        )

    def relation(
        self,
        subject: Any,
        predicate: Any,
        obj: Any,
        fact_type: str,
        *,
        detail: str | None = None,
        metadata: dict[str, Any] | None = None,
        categories: Iterable[str] = (),
    ) -> None:
        pred = _pred_value(predicate)
        pred = {"ownedByGroup": "ownedBy"}.get(pred, pred)
        if pred in KG_NUMERIC_PREDICATES:
            self.exclude(fact_type, subject, pred, obj, EXCLUDED_NUMERIC)
            return
        subject_role, object_role = _PREDICATE_ROLES.get(pred, (_ROLE_ENTITY, _ROLE_ENTITY))
        canonical_subject = self.adapter._normalize(subject, subject_role)
        canonical_object = (
            self.adapter._normalize(obj, object_role) if obj not in (None, "") else None
        )
        if canonical_subject in PLACEHOLDER_ENTITIES or canonical_object in PLACEHOLDER_ENTITIES:
            self.exclude(fact_type, subject, pred, obj, EXCLUDED_PLACEHOLDER)
            return
        key = (canonical_subject, pred, canonical_object)
        pending = self._pending.get(key)
        if pending is None:
            meta = dict(metadata or {})
            meta.update(self.adapter._display(subject, canonical_subject))
            if canonical_object is not None:
                meta.update(
                    {
                        f"object_{k}": v
                        for k, v in self.adapter._display(obj, canonical_object).items()
                    }
                )
            pending = {"detail": detail, "metadata": meta, "categories": [], "fact_type": fact_type}
            self._pending[key] = pending
            self._order.append(key)
        for category in categories:
            if category:
                normalized = self.adapter.normalize_category(str(category))
                if normalized not in pending["categories"]:
                    pending["categories"].append(normalized)

    def result(self) -> KGEvidenceResult:
        for subject, predicate, obj in self._order:
            pending = self._pending[(subject, predicate, obj)]
            meta = dict(pending["metadata"])
            meta["fact_type"] = pending["fact_type"]
            if pending["categories"]:
                meta["categories"] = pending["categories"]
            shown_subject = meta.get("display_name") or subject
            shown_object = meta.get("object_display_name") or obj
            text = " ".join(part for part in (shown_subject, predicate, shown_object) if part)
            if pending["categories"]:
                text += f" [카테고리: {', '.join(pending['categories'])}]"
            if meta.get("scope") == "brand_or_market":
                text += " [브랜드 또는 시장 전체 트렌드]"
            self.cards.add(
                Evidence.create(
                    kind=EvidenceKind.RELATION,
                    subject=subject,
                    predicate=predicate,
                    object=obj,
                    source=KG_SOURCE,
                    text=text,
                    detail=pending["detail"],
                    metadata=meta,
                )
            )
        return KGEvidenceResult(
            cards=self.cards.to_list(), excluded=self.excluded, skipped=dict(self.skipped)
        )


# ----------------------------------------------------------------------
# fact type별 변환 (형식: src/rag/hybrid_retriever.py:_query_knowledge_graph, 2026-09-17)
# ----------------------------------------------------------------------


def _kg_brand_info(b: _RelationBuilder, entity: Any, data: Any) -> None:
    # entity_metadata: {"type", "sos", "avg_rank", "product_count", "is_target", ...}
    # (dashboard_exporter.py·brain.py·kg_updater.py가 대시보드 JSON에서 날짜 없이 기록)
    for key, value in (data or {}).items():
        b.exclude("brand_info", entity, key, value, EXCLUDED_METADATA)


def _kg_brand_products(b: _RelationBuilder, entity: Any, data: Any) -> None:
    # {"product_count", "products": [{"asin", "name", "category", "rank", **props}]}
    for product in (data or {}).get("products") or []:
        asin = product.get("asin")
        if not asin:
            continue
        title = product.get("title") or product.get("name") or ""
        category = product.get("category")
        meta = {"title": title} if title else {}
        b.relation(entity, "hasProduct", asin, "brand_products", metadata=meta)
        if category:
            b.relation(asin, "belongsToCategory", category, "brand_products", metadata=meta)


def _kg_competitors(b: _RelationBuilder, entity: Any, data: Any) -> None:
    # [{"brand", "type", "category", **props}] — 같은 경쟁사는 카테고리를 합친다
    for competitor in data or []:
        if not competitor.get("brand"):
            continue
        b.relation(
            entity,
            competitor.get("type") or "competesWith",
            competitor["brand"],
            "competitors",
            categories=[competitor.get("category") or ""],
        )


def _kg_competitor_network(b: _RelationBuilder, entity: Any, data: Any) -> None:
    # {"outgoing": [(RelationType, object)], "incoming": [(RelationType, subject)]}
    for predicate, obj in (data or {}).get("outgoing") or []:
        b.relation(entity, predicate, obj, "competitor_network")
    for predicate, subject in (data or {}).get("incoming") or []:
        b.relation(subject, predicate, entity, "competitor_network")


def _kg_metric_edges(b: _RelationBuilder, entity: Any, data: Any) -> None:
    # {"edges": [{"subject", "predicate", "object"}]} — 속성·날짜는 실리지 않는다
    for edge in (data or {}).get("edges") or []:
        b.relation(edge.get("subject"), edge.get("predicate"), edge.get("object"), "metric_edges")


def _kg_trend_keywords(b: _RelationBuilder, entity: Any, data: Any) -> None:
    # {"keywords", "count"} — 브랜드 트렌드가 없으면 MARKET 트렌드로 대체되는데 사실에는
    # 어느 쪽인지가 남지 않는다. 주어는 질의 브랜드로 두고 범위가 모호함을 표시한다.
    for keyword in (data or {}).get("keywords") or []:
        b.relation(
            entity, "hasTrend", keyword, "trend_keywords", metadata={"scope": "brand_or_market"}
        )


def _kg_category_brands(b: _RelationBuilder, entity: Any, data: Any) -> None:
    # {"brand_count", "top_brands": [{"brand", "product_count", "products"}]}
    # 제품 belongsToCategory에서 유도된 "브랜드가 카테고리 순위에 오름" — rankedIn과 같은 의미.
    # product_count는 날짜 없는 개수라 싣지 않는다.
    for entry in (data or {}).get("top_brands") or []:
        if entry.get("brand"):
            b.relation(entry["brand"], "rankedIn", entity, "category_brands")


def _kg_category_hierarchy(b: _RelationBuilder, entity: Any, data: Any) -> None:
    # {"name", "level", "path", "ancestors": [{"id","name","level"}], "descendants": [...]}
    data = data or {}
    child = entity
    child_meta = {"name": data.get("name")} if data.get("name") else {}
    for ancestor in data.get("ancestors") or []:
        if not ancestor.get("id"):
            break
        b.relation(
            child, "parentCategory", ancestor["id"], "category_hierarchy", metadata=child_meta
        )
        child = ancestor["id"]
        child_meta = {"name": ancestor.get("name")} if ancestor.get("name") else {}
    for descendant in data.get("descendants") or []:
        if descendant.get("id"):
            b.relation(entity, "hasSubcategory", descendant["id"], "category_hierarchy")


def _kg_product_sentiment(b: _RelationBuilder, entity: Any, data: Any) -> None:
    # {"asin", "ai_summary", "sentiment_tags", "sentiment_clusters": {cluster: [tags]}}
    data = data or {}
    cluster_of = {
        tag: cluster
        for cluster, tags in (data.get("sentiment_clusters") or {}).items()
        for tag in tags
    }
    asin = data.get("asin") or entity
    for tag in data.get("sentiment_tags") or []:
        meta = {"cluster": cluster_of[tag]} if tag in cluster_of else {}
        b.relation(asin, "hasSentiment", tag, "product_sentiment", metadata=meta)
    if data.get("ai_summary"):
        b.relation(asin, "hasAISummary", None, "product_sentiment", detail=str(data["ai_summary"]))


def _kg_brand_sentiment(b: _RelationBuilder, entity: Any, data: Any) -> None:
    # {"brand", "all_tags", "clusters": {cluster: count}, "dominant_sentiment", "product_count"}
    # 제품 감성 태그를 브랜드로 모은 값 — 개수·지배 감성은 날짜 없는 집계라 싣지 않는다.
    for tag in (data or {}).get("all_tags") or []:
        b.relation(
            entity,
            "brandSentiment",
            tag,
            "brand_sentiment",
            metadata={"aggregated_from": "products"},
        )


def _kg_sentiment_products(b: _RelationBuilder, entity: Any, data: Any) -> None:
    # {"sentiment_tag", "cluster", "product_count", "products": [asin]}
    data = data or {}
    tag = data.get("sentiment_tag") or entity
    meta = {"cluster": data["cluster"]} if data.get("cluster") else {}
    for asin in data.get("products") or []:
        b.relation(asin, "hasSentiment", tag, "sentiment_products", metadata=meta)


_KG_HANDLERS: dict[str, Callable[[_RelationBuilder, Any, Any], None]] = {
    "brand_info": _kg_brand_info,
    "brand_products": _kg_brand_products,
    "competitors": _kg_competitors,
    "competitor_network": _kg_competitor_network,
    "metric_edges": _kg_metric_edges,
    "trend_keywords": _kg_trend_keywords,
    "category_brands": _kg_category_brands,
    "category_hierarchy": _kg_category_hierarchy,
    "product_sentiment": _kg_product_sentiment,
    "brand_sentiment": _kg_brand_sentiment,
    "sentiment_products": _kg_sentiment_products,
}
