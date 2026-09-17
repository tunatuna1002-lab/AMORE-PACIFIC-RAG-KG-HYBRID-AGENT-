"""
읽기 전용 도구 레지스트리
=========================
에이전트가 쓸 수 있는 도구는 이 파일의 5종뿐이다 (설계 E4, 트랙 4-A).

    resolve_entity  표기 → canonical id (브랜드·카테고리·제품·지표)
    kg_neighbors    KG 구조적 관계 (소유·카테고리·경쟁·제품)
    get_metrics     크롤 DB(SQLite) 수치 — 스냅샷 날짜가 붙는 정본
    apply_rules     규칙 추론 (근거 카드 id를 derived_from에 남긴다)
    search_docs     문서 검색 (Dense + BM25 RRF)

원칙
----
- **모두 읽기 전용이다.** DB는 ``mode=ro``로 열고(MetricFactsProvider), KG·문서 색인에
  쓰지 않으며, 크롤처럼 부작용이 있는 도구는 등록하지 않는다.
- **결과는 전부 증거 카드다** (``src.domain.entities.evidence.Evidence``). 도구 결과가 답변에
  실릴 때 검색 카드와 같은 형식·같은 인용 id를 쓰게 하기 위함이다. 카드 변환은 새로 쓰지 않고
  ``EvidenceAdapter``를 재사용한다 — 단위 정규화·브랜드 canonical id·KG 수치 엣지 제외(E2)가
  검색 경로와 갈라지지 않게.
- **하나의 레지스트리만 쓴다.** DecisionMaker(function calling)·ToolCoordinator·ReAct
  실행기(``react_tools.ReActToolExecutor``)가 모두 이 객체를 본다. 예전 대시보드 JSON 도구
  5종(get_brand_status 등)은 제거됐다 — 날짜 없는 캐시 JSON이라 수치 근거가 될 수 없었다.
- **ablation 플래그를 검색 경로와 똑같이 따른다**: KG(ontology.use_ontology_kg)·DB 수치
  (retriever.use_db_metric_facts)·규칙(reasoner.*)이 꺼지면 해당 도구를 목록에서 뺀다.
  끄고 평가할 때 도구 경로로 우회되면 ablation 수치가 오염된다.

백엔드
------
``ToolRegistry(retriever)``의 retriever는 ``HybridRetriever``다. KG 조회·규칙 판정·문서
검색은 검색 경로와 **같은 메서드**(``_query_knowledge_graph``·``_evaluate_rules``·
``_hybrid_search``)를 호출한다. 도구와 검색이 다른 답을 내지 않게 하기 위함이다.

ToolResult 계약
---------------
``data``는 JSON 직렬화 가능한 dict다: ``{"evidence": [카드 dump, ...], ...메타}``.
카드 객체는 ``tool_evidence(result)``로 되살린다.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from src.domain.entities.evidence import Evidence, EvidenceKind

from .models import ToolResult

logger = logging.getLogger(__name__)

EVIDENCE_KEY = "evidence"

TOOL_RESOLVE_ENTITY = "resolve_entity"
TOOL_KG_NEIGHBORS = "kg_neighbors"
TOOL_GET_METRICS = "get_metrics"
TOOL_APPLY_RULES = "apply_rules"
TOOL_SEARCH_DOCS = "search_docs"

MAX_DOC_RESULTS = 10
DEFAULT_DOC_RESULTS = 5

_JSON_PYTHON_TYPES: dict[str, type | tuple[type, ...]] = {
    "string": str,
    "integer": int,
    "array": list,
}


# ======================================================================
# 도구 정의 (function calling 스키마의 정본)
# ======================================================================


@dataclass(frozen=True)
class ToolParam:
    """도구 파라미터 하나. JSON Schema와 파이썬 타입 검증을 함께 낸다."""

    name: str
    type: str  # string | integer | array
    description: str
    required: bool = False
    items: str | None = None  # type == "array"일 때 원소 타입
    minimum: int | None = None
    maximum: int | None = None

    def json_schema(self) -> dict[str, Any]:
        schema: dict[str, Any] = {"type": self.type, "description": self.description}
        if self.type == "array":
            schema["items"] = {"type": self.items or "string"}
        if self.minimum is not None:
            schema["minimum"] = self.minimum
        if self.maximum is not None:
            schema["maximum"] = self.maximum
        return schema

    def validate(self, value: Any) -> str | None:
        """값이 스키마에 맞지 않으면 사유 문자열, 맞으면 None."""
        expected = _JSON_PYTHON_TYPES[self.type]
        if isinstance(value, bool) or not isinstance(value, expected):
            return f"파라미터 '{self.name}'은(는) {self.type}이어야 합니다 (받은 값: {value!r})"
        if self.type == "array" and any(not isinstance(item, str) for item in value):
            return f"파라미터 '{self.name}'의 원소는 문자열이어야 합니다"
        if self.minimum is not None and value < self.minimum:
            return f"파라미터 '{self.name}'은(는) {self.minimum} 이상이어야 합니다"
        if self.maximum is not None and value > self.maximum:
            return f"파라미터 '{self.name}'은(는) {self.maximum} 이하여야 합니다"
        return None


@dataclass(frozen=True)
class ToolDefinition:
    """도구 이름·설명·파라미터. LiteLLM ``tools=`` 스키마는 여기서 생성된다."""

    name: str
    description: str
    params: tuple[ToolParam, ...] = ()

    def function_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {p.name: p.json_schema() for p in self.params},
                    "required": [p.name for p in self.params if p.required],
                    "additionalProperties": False,
                },
            },
        }

    def signature(self) -> str:
        """프롬프트용 한 줄 서명: ``get_metrics(brand?: string, category?: string)``."""
        parts = [f"{p.name}{'' if p.required else '?'}: {p.type}" for p in self.params]
        return f"{self.name}({', '.join(parts)})"

    def parameter_types(self) -> dict[str, type | tuple[type, ...]]:
        """ReAct ``ACTION_SCHEMAS``용 파라미터 타입표."""
        return {p.name: _JSON_PYTHON_TYPES[p.type] for p in self.params}

    def validate(self, params: dict[str, Any] | None) -> tuple[dict[str, Any], str | None]:
        """(정리된 인자, 오류 사유). 모르는 파라미터는 거부한다 — 조용히 무시하면
        LLM이 잘못된 호출을 고칠 신호를 못 받는다."""
        given = dict(params or {})
        known = {p.name: p for p in self.params}
        unknown = sorted(set(given) - set(known))
        if unknown:
            return {}, f"알 수 없는 파라미터: {', '.join(unknown)} (허용: {', '.join(known)})"
        cleaned: dict[str, Any] = {}
        for name, param in known.items():
            if name not in given or given[name] is None:
                if param.required:
                    return {}, f"필수 파라미터 '{name}'이(가) 없습니다"
                continue
            error = param.validate(given[name])
            if error:
                return {}, error
            cleaned[name] = given[name]
        return cleaned, None


TOOL_DEFINITIONS: tuple[ToolDefinition, ...] = (
    ToolDefinition(
        name=TOOL_RESOLVE_ENTITY,
        description=(
            "질문에 나온 브랜드·카테고리·제품·지표 표기를 정규화된 id로 바꾼다 "
            "(예: '라네즈 립케어' → 브랜드 laneige, 카테고리 lip_care). "
            "다른 도구에 넘길 id가 확실하지 않을 때 먼저 쓴다."
        ),
        params=(
            ToolParam(
                name="text",
                type="string",
                description="정규화할 표기 또는 질문 원문",
                required=True,
            ),
        ),
    ),
    ToolDefinition(
        name=TOOL_KG_NEIGHBORS,
        description=(
            "지식 그래프에서 엔티티(브랜드 또는 카테고리)의 구조적 관계를 조회한다: "
            "소유(ownedBy), 카테고리 소속·계층(rankedIn·parentCategory·hasSubcategory), "
            "경쟁(competesWith), 제품(hasProduct), 감성 태그. "
            "SoS·HHI 같은 수치는 주지 않는다 — 수치는 get_metrics를 쓴다."
        ),
        params=(
            ToolParam(
                name="entity",
                type="string",
                description="브랜드명 또는 카테고리명 (예: LANEIGE, Lip Care)",
                required=True,
            ),
            ToolParam(
                name="predicates",
                type="array",
                description="가져올 관계 술어 목록 (비우면 전부). 예: ['ownedBy','competesWith']",
                items="string",
            ),
        ),
    ),
    ToolDefinition(
        name=TOOL_GET_METRICS,
        description=(
            "크롤 DB(SQLite)에서 스냅샷 날짜가 붙은 수치를 조회한다: 카테고리 HHI·평균가·"
            "평균 평점, 브랜드 SoS·Top100 제품 수·SoS 순위·평균 순위·CPI, 상위 브랜드와 "
            "상위 제품의 순위·가격·평점·리뷰 수. brand와 category 중 하나 이상이 필요하다."
        ),
        params=(
            ToolParam(name="brand", type="string", description="브랜드명 (예: LANEIGE)"),
            ToolParam(name="category", type="string", description="카테고리명 (예: Lip Care)"),
        ),
    ),
    ToolDefinition(
        name=TOOL_APPLY_RULES,
        description=(
            "DB 수치와 KG 관계를 입력으로 비즈니스 규칙(시장 지배력·가격 포지션·경쟁 위협·"
            "성장 신호 등)을 판정하고, 발화한 추론과 그 근거 카드를 돌려준다. "
            "brand와 category 중 하나 이상이 필요하다."
        ),
        params=(
            ToolParam(name="brand", type="string", description="브랜드명 (예: LANEIGE)"),
            ToolParam(name="category", type="string", description="카테고리명 (예: Lip Care)"),
        ),
    ),
    ToolDefinition(
        name=TOOL_SEARCH_DOCS,
        description=(
            "지표 정의·해석 가이드·전략 플레이북 등 문서를 검색한다 (Dense + BM25 RRF). "
            "'SoS가 뭐야' 같은 정의·해석 질문에 쓴다."
        ),
        params=(
            ToolParam(name="query", type="string", description="검색 질의", required=True),
            ToolParam(
                name="k",
                type="integer",
                description=f"가져올 문서 수 (기본 {DEFAULT_DOC_RESULTS})",
                minimum=1,
                maximum=MAX_DOC_RESULTS,
            ),
        ),
    ),
)

TOOL_NAMES: tuple[str, ...] = tuple(d.name for d in TOOL_DEFINITIONS)
_BY_NAME: dict[str, ToolDefinition] = {d.name: d for d in TOOL_DEFINITIONS}


def function_schemas(names: Iterable[str] | None = None) -> list[dict[str, Any]]:
    """LiteLLM ``tools=``에 그대로 넣는 스키마 목록 (정의 순서)."""
    if names is None:
        wanted = set(TOOL_NAMES)
    else:
        wanted = set(names)
    return [d.function_schema() for d in TOOL_DEFINITIONS if d.name in wanted]


def tool_evidence(result: Any) -> list[Evidence]:
    """``ToolResult.data["evidence"]`` → 증거 카드. 도구 결과가 아니면 빈 목록."""
    data = getattr(result, "data", None)
    if not getattr(result, "success", False) or not isinstance(data, dict):
        return []
    cards: list[Evidence] = []
    for item in data.get(EVIDENCE_KEY) or []:
        if isinstance(item, Evidence):
            cards.append(item)
        else:
            cards.append(Evidence.model_validate(item))
    return cards


@dataclass
class ToolOutput:
    """도구 하나의 결과: 카드 + 메타데이터."""

    cards: list[Evidence] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)


# ======================================================================
# 레지스트리
# ======================================================================


class ToolRegistry:
    """읽기 전용 도구 5종의 단일 레지스트리.

    ``ToolCoordinator``/ReAct 실행기가 기대하는 인터페이스(``execute``·
    ``get_available_tools``·``is_tool_available``)를 그대로 제공한다.
    """

    def __init__(self, retriever: Any | None = None) -> None:
        self._retriever = retriever

    # ── 배선 ─────────────────────────────────────────────────────

    def bind(self, retriever: Any) -> None:
        """백엔드(HybridRetriever) 연결. 도구는 검색 경로와 같은 객체를 쓴다."""
        self._retriever = retriever

    @property
    def retriever(self) -> Any | None:
        return self._retriever

    @property
    def is_bound(self) -> bool:
        return self._retriever is not None

    # ── 목록 ─────────────────────────────────────────────────────

    @staticmethod
    def _flags() -> Any:
        from src.infrastructure.feature_flags import FeatureFlags

        return FeatureFlags.get_instance()

    def _enabled(self, name: str) -> bool:
        """ablation 플래그가 끈 하위 시스템의 도구는 내놓지 않는다 (검색 경로와 동일 기준)."""
        flags = self._flags()
        if name == TOOL_KG_NEIGHBORS:
            return bool(flags.use_ontology_kg())
        if name == TOOL_GET_METRICS:
            return bool(flags.use_db_metric_facts())
        if name == TOOL_APPLY_RULES:
            return bool(flags.use_unified_reasoner() or flags.use_owl_reasoner())
        return True

    def get_available_tools(self) -> list[str]:
        """실행 가능한 도구 이름. 백엔드가 없으면 빈 목록이다 — 못 부를 도구를 광고하지 않는다."""
        if not self.is_bound:
            return []
        return [name for name in TOOL_NAMES if self._enabled(name)]

    def is_tool_available(self, tool_name: str) -> bool:
        return tool_name in self.get_available_tools()

    # ── 실행 ─────────────────────────────────────────────────────

    async def execute(self, tool_name: str, params: dict[str, Any]) -> ToolResult:
        """도구 실행. 예외를 던지지 않고 실패도 ToolResult로 돌려준다."""
        start = datetime.now()

        def fail(message: str) -> ToolResult:
            return ToolResult(
                tool_name=tool_name,
                success=False,
                error=message,
                execution_time_ms=(datetime.now() - start).total_seconds() * 1000,
            )

        definition = _BY_NAME.get(tool_name)
        if definition is None:
            return fail(
                f"등록되지 않은 도구입니다: {tool_name} (사용 가능: {', '.join(TOOL_NAMES)})"
            )
        if not self.is_bound:
            return fail(f"도구 '{tool_name}'의 백엔드(검색기)가 연결되지 않았습니다")
        if not self._enabled(tool_name):
            return fail(f"도구 '{tool_name}'은(는) 피처 플래그로 비활성화돼 있습니다")

        arguments, error = definition.validate(params)
        if error:
            return fail(error)

        handler = getattr(self, f"_run_{tool_name}")
        try:
            output = await handler(**arguments)
        except Exception as e:  # 도구 실패가 질의 처리를 끊지 않게
            logger.warning(f"tool {tool_name} failed", exc_info=True)
            return fail(f"{type(e).__name__}: {e}")

        data: dict[str, Any] = {
            EVIDENCE_KEY: [card.model_dump(mode="json") for card in output.cards],
            **output.meta,
        }
        return ToolResult(
            tool_name=tool_name,
            success=True,
            data=data,
            execution_time_ms=(datetime.now() - start).total_seconds() * 1000,
        )

    # ── 공통 헬퍼 ────────────────────────────────────────────────

    @property
    def _adapter(self) -> Any:
        return self._retriever.evidence_adapter

    def _scope(self, brand: str | None, category: str | None) -> dict[str, list[str]]:
        """도구 인자 → canonical 엔티티 (검색 경로가 제공자·규칙에 넘기는 것과 같은 표기)."""
        adapter = self._adapter
        return {
            "brands": [adapter.normalize_brand(brand)] if brand else [],
            "categories": [adapter.normalize_category(category)] if category else [],
        }

    def _extract(self, text: str) -> dict[str, Any]:
        return self._retriever.entity_extractor.extract(text, knowledge_graph=self._retriever.kg)

    async def _metric_facts(self, entities: dict[str, list[str]]) -> list[dict[str, Any]]:
        if not self._flags().use_db_metric_facts():
            return []
        return await self._retriever.metric_facts_provider.collect(entities)

    def _kg_facts(self, entities: dict[str, list[str]]) -> list[dict[str, Any]]:
        if not self._flags().use_ontology_kg():
            return []
        return self._retriever._query_knowledge_graph(entities)

    @staticmethod
    def _select(cards: Sequence[Evidence]) -> list[Evidence]:
        """프롬프트 상한을 도구 출력에도 그대로 적용한다 (검색 카드와 같은 기준)."""
        from src.rag.evidence_assembly import PROMPT_MAX_PER_KIND
        from src.rag.evidence_renderer import select_cards

        return select_cards(cards, PROMPT_MAX_PER_KIND)

    # ── 도구 구현 ────────────────────────────────────────────────

    async def _run_resolve_entity(self, text: str) -> ToolOutput:
        adapter = self._adapter
        extracted = self._extract(text)
        resolved: list[tuple[str, str]] = []
        for kind, key, normalize in (
            ("brand", "brands", adapter.normalize_brand),
            ("category", "categories", adapter.normalize_category),
            ("product", "products", str),
            ("indicator", "indicators", str),
        ):
            for value in extracted.get(key) or []:
                item = (kind, normalize(str(value)))
                if item not in resolved:
                    resolved.append(item)

        # 어휘 판정은 "도구가 관찰한 사실"이다 — 카드 하나가 표기 하나의 canonical id를 말한다.
        # (검색·KG 사실이 아니므로 relation 카드로 만들지 않는다.)
        cards = [
            Evidence.create(
                kind=EvidenceKind.OBSERVATION,
                subject=canonical,
                predicate="resolvedAs",
                object=kind,
                source=f"tool:{TOOL_RESOLVE_ENTITY}",
                text=f"'{text}' → {kind} {canonical}",
                metadata={"entity_type": kind, "canonical_id": canonical, "surface": text},
            )
            for kind, canonical in resolved
        ]
        meta: dict[str, Any] = {
            "entities": {
                "brands": [c for k, c in resolved if k == "brand"],
                "categories": [c for k, c in resolved if k == "category"],
                "products": [c for k, c in resolved if k == "product"],
                "indicators": [c for k, c in resolved if k == "indicator"],
            }
        }
        if not cards:
            meta["message"] = "표기에서 알려진 브랜드·카테고리·제품·지표를 찾지 못했습니다."
        return ToolOutput(cards=cards, meta=meta)

    async def _run_kg_neighbors(
        self, entity: str, predicates: list[str] | None = None
    ) -> ToolOutput:
        from src.rag.evidence_assembly import order_relation_cards

        adapter = self._adapter
        extracted = self._extract(entity)
        entities = {
            "brands": [adapter.normalize_brand(str(b)) for b in extracted.get("brands") or []],
            "categories": [
                adapter.normalize_category(str(c)) for c in extracted.get("categories") or []
            ],
        }
        if not entities["brands"] and not entities["categories"]:
            # 단어사전에 없는 브랜드(KG에는 있을 수 있다)는 표기 그대로 조회한다
            entities["brands"] = [adapter.normalize_brand(entity)]

        result = adapter.from_kg_facts(self._kg_facts(entities))
        cards = order_relation_cards(result.cards)
        if predicates:
            wanted = set(predicates)
            cards = [card for card in cards if card.predicate in wanted]
        excluded_by_reason: dict[str, int] = {}
        for item in result.excluded:
            reason = str(item.get("reason"))
            excluded_by_reason[reason] = excluded_by_reason.get(reason, 0) + 1

        meta: dict[str, Any] = {
            "entity": entity,
            "resolved": entities,
            "predicates": list(predicates or []),
            "total_cards": len(cards),
            "excluded": len(result.excluded),
            "excluded_by_reason": excluded_by_reason,
        }
        if not cards:
            meta["message"] = (
                "해당 엔티티의 구조적 관계를 찾지 못했습니다 "
                "(SoS·HHI 같은 수치는 get_metrics에서 조회합니다)."
            )
        return ToolOutput(cards=self._select(cards), meta=meta)

    async def _run_get_metrics(
        self, brand: str | None = None, category: str | None = None
    ) -> ToolOutput:
        from src.rag.evidence_assembly import order_metric_cards

        if not brand and not category:
            raise ValueError("brand 또는 category 중 하나는 필요합니다")

        entities = self._scope(brand, category)
        facts = await self._metric_facts(entities)
        cards = self._adapter.from_metric_facts(facts)
        _, within_caps = order_metric_cards(cards, entities["brands"])
        selected = self._select(within_caps)

        provider = self._retriever.metric_facts_provider
        meta: dict[str, Any] = {
            "brand": entities["brands"][0] if entities["brands"] else None,
            "category": entities["categories"][0] if entities["categories"] else None,
            "as_of": getattr(provider, "as_of", None),
            "snapshot_dates": sorted({c.as_of for c in selected if c.as_of}),
            "total_cards": len(cards),
        }
        if not selected:
            meta["message"] = "해당 브랜드·카테고리의 DB 수치가 없습니다."
        return ToolOutput(cards=selected, meta=meta)

    async def _run_apply_rules(
        self, brand: str | None = None, category: str | None = None
    ) -> ToolOutput:
        from src.domain.entities.evidence import EvidenceKind, EvidenceSet
        from src.rag.evidence_assembly import PROMPT_MAX_PER_KIND, build_rule_input_cards

        if not brand and not category:
            raise ValueError("brand 또는 category 중 하나는 필요합니다")

        entities = self._scope(brand, category)
        degraded: list[dict[str, Any]] = []
        input_cards = build_rule_input_cards(
            metric_facts=await self._metric_facts(entities),
            ontology_facts=self._kg_facts(entities),
            adapter=self._adapter,
            degraded=degraded,
        )
        inferences, summary = self._retriever._evaluate_rules(entities, input_cards)
        inference_cards = self._adapter.from_inferences(inferences)[
            : PROMPT_MAX_PER_KIND[EvidenceKind.INFERENCE]
        ]

        # 추론 카드가 인용한 근거 카드는 상한과 무관하게 함께 싣는다 (derived_from 해소)
        by_id = {card.id: card for card in input_cards}
        basis = [
            by_id[card_id]
            for card in inference_cards
            for card_id in card.derived_from
            if card_id in by_id
        ]
        cards = EvidenceSet([*inference_cards, *basis]).to_list()

        meta: dict[str, Any] = {
            "brand": entities["brands"][0] if entities["brands"] else None,
            "category": entities["categories"][0] if entities["categories"] else None,
            "rule_evaluation": summary,
            "input_card_count": len(input_cards),
            "degraded": degraded,
        }
        if not inference_cards:
            meta["message"] = "입력 수치·관계로 발화한 규칙이 없습니다."
        return ToolOutput(cards=cards, meta=meta)

    async def _run_search_docs(self, query: str, k: int = DEFAULT_DOC_RESULTS) -> ToolOutput:
        degraded: list[dict[str, Any]] = []
        results, method = await self._retriever._hybrid_search(
            query, top_k=k, doc_type_filter=None, degraded=degraded
        )
        cards = self._select(self._adapter.from_rag_chunks(results))
        meta: dict[str, Any] = {
            "query": query,
            "k": k,
            "search_method": method,
            "degraded": degraded,
        }
        if not cards:
            meta["message"] = "질의에 맞는 문서를 찾지 못했습니다."
        return ToolOutput(cards=cards, meta=meta)


def render_tool_observation(result: Any) -> str:
    """도구 결과 → 사람이 읽는 관찰 문자열 (ReAct 관찰·답변 프롬프트 공용).

    카드가 있으면 프롬프트와 같은 형식(``[id] 내용 (as_of, source)``)으로, 없으면 도구가
    남긴 메시지를 쓴다. 인용 id가 관찰에도 그대로 있어야 답변이 같은 id로 인용할 수 있다.
    """
    from src.rag.evidence_renderer import render_for_prompt

    if not getattr(result, "success", False):
        return f"실행 실패: {getattr(result, 'error', '알 수 없는 오류')}"
    cards = tool_evidence(result)
    rendered = render_for_prompt(cards)
    if rendered:
        return rendered
    data = getattr(result, "data", None) or {}
    message = data.get("message") if isinstance(data, dict) else None
    return str(message or "결과 없음")


__all__ = [
    "EVIDENCE_KEY",
    "TOOL_APPLY_RULES",
    "TOOL_DEFINITIONS",
    "TOOL_GET_METRICS",
    "TOOL_KG_NEIGHBORS",
    "TOOL_NAMES",
    "TOOL_RESOLVE_ENTITY",
    "TOOL_SEARCH_DOCS",
    "ToolDefinition",
    "ToolParam",
    "ToolRegistry",
    "function_schemas",
    "render_tool_observation",
    "tool_evidence",
]
