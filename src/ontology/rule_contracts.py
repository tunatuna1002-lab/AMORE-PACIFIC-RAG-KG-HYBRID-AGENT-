"""
Rule Input Contracts
====================
규칙 입력 계약 (설계 E3, 트랙 3-A).

규칙 엔진은 ``OntologyReasoner``와 ``src/ontology/rules/*.py``의 규칙 37개 하나다. 이 모듈은
엔진을 바꾸지 않고, 규칙마다 **필요한 입력(이름·타입·단위·범위)**과 **그 입력을 채울 증거
카드**를 선언한다. 추론 컨텍스트 조립(트랙 3-B)은 ``InputSpec.binding``만 보고 카드에서 값을
찾는다.

배경: v4 평가 130문항 × 10실행에서 규칙 추론이 0건 발화했다. 추론 입력이 대시보드 JSON 키를
읽는데 그 키가 운영 데이터에 없었고, 결측이면 발화하지 않게 고친 규칙들이 이유를 남기지 않고
침묵했다. ``evaluate_rule``은 발화하지 않은 이유를 남긴다.

판정 순서 (``evaluate_rule``)
-----------------------------
1. 계약 없음 → ``no_contract``
2. 계약 검사: 결측(키 없음·None) → ``missing_input``, 타입 → ``type_mismatch``,
   범위 → ``out_of_range``. **위반이면 조건을 평가하지 않는다** — ``ctx.get("sos", 0)``처럼
   결측을 0으로 읽는 조건이 남아 있어도 그 경로에 닿지 않는다.
3. ``rule.evaluate_conditions`` 불충족 → ``conditions_not_met`` (실패한 조건 이름 전부)
4. ``rule.apply`` → 결과. 조건은 통과했는데 결과가 없으면 결론 함수가 예외를 삼킨 것이다
   (``InferenceRule.apply``가 예외를 로그만 남기고 None을 돌려준다) → ``conclusion_failed``

컨텍스트는 최종 컨텍스트다. ``OntologyReasoner._enrich_context``(KG 보강)는 부르지 않는다 —
KG에서 오던 입력(경쟁사 수 등)도 relation 카드 바인딩으로 선언돼 있다.

단위 (카드 정본과 같다)
-----------------------
SoS·churn_rate 0~1 ``ratio``, HHI 0~1 ``index_0_1`` (결정 D1), CPI 100 기준 ``index_100``,
평점 격차 ``rating_points``, 순위 ``rank``, 개수 ``count``. 카드가 없는 입력은
``ContractUnit``의 단위를 쓴다.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field, replace
from enum import Enum
from functools import lru_cache
from typing import Any

from src.domain.entities.evidence import Evidence, EvidenceKind, EvidenceUnit
from src.domain.entities.relations import InferenceResult

from .reasoner import InferenceRule

# =========================================================================
# 스키마
# =========================================================================


class InputType(str, Enum):
    """입력 값 타입. None은 타입이 아니라 결측이다."""

    NUMBER = "number"  # int·float (bool 제외, 유한값)
    INTEGER = "integer"  # 정수값 (2.0 허용, bool 제외)
    BOOLEAN = "boolean"
    STRING = "string"
    STRING_LIST = "string_list"  # list[str]
    TAG_CLUSTERS = "tag_clusters"  # {cluster: [tag, ...]}
    CLUSTER_KEYS = "cluster_keys"  # {cluster: 무엇이든} — 규칙이 키만 읽는다
    RECORD_LIST = "record_list"  # list[dict]
    OBJECT_LIST = "object_list"  # list
    ANY = "any"  # 표시용 — 결론이 str()로만 쓴다


class ContractUnit:
    """증거 카드에 없는, 규칙 입력 전용 단위."""

    RATIO_DELTA = "ratio_delta"  # 0~1 비율의 차이 (-0.02 = -2%p)
    RANK_DELTA = "rank_delta"  # 순위 변화 (음수 = 상승)
    RANK_STDDEV = "rank_stddev"  # 순위 표준편차
    RATING_SLOPE = "rating_slope"  # 일별 평점 기울기 (5점 척도)
    DAYS = "days"
    QUARTERS = "quarters"
    PERCENT = "percent"  # IR 성장률 (6.9 = 6.9%)


class Role:
    """카드의 주어·목적어가 질의에서 맡는 역할."""

    BRAND = "brand"  # 질의 브랜드 canonical id (예 ``laneige``)
    CATEGORY = "category"  # 질의 카테고리 id (예 ``lip_care``)
    PRODUCT = "product"  # 질의 브랜드의 제품 (metric: 제품명+metadata.brand, relation: ASIN)
    COMPETITOR_BRAND = "competitor_brand"  # 질의 브랜드의 경쟁 브랜드
    COMPETITOR_PRODUCT = "competitor_product"  # 경쟁 브랜드의 제품
    ENTITY = "entity"  # 그 밖의 엔티티 (그룹사 등)
    RAW = "raw"  # 어휘 그대로인 값 (감성 태그·트렌드 키워드)


class Reduce(str, Enum):
    """매칭된 카드(들)에서 입력 값을 만드는 방법."""

    VALUE = "value"  # 카드 하나의 value
    MIN_VALUE = "min_value"  # 여러 카드 value 중 최솟값 (최고 순위)
    OBJECT = "object"  # 카드 하나의 object
    OBJECT_LABEL = "object_label"  # 카드 하나의 object 표시 이름 (없으면 object)
    OBJECTS = "objects"  # object 목록
    COUNT = "count"  # 서로 다른 object 수
    OBJECTS_BY_CLUSTER = "objects_by_cluster"  # metadata["cluster"]별 object 목록
    DETAIL = "detail"  # 카드 detail 본문


@dataclass(frozen=True)
class EvidenceBinding:
    """증거 카드에서 입력 하나를 찾는 규칙.

    Attributes:
        kind: 카드 종류.
        predicate: 카드 술어 (``src/rag/evidence_adapters.py``가 만드는 이름).
        unit: 카드 단위 (``EvidenceUnit``). 관계 카드는 None.
        subject_role: 카드 주어의 역할 (``Role``).
        object_role: 카드 목적어의 역할. 목적어가 없으면 None.
        reduce: 매칭 카드에서 값을 만드는 방법.
        extra_predicates: 같은 의미로 함께 찾는 술어.
        ontology_aliases: True면 온톨로지 술어 별칭도 같은 술어로 본다
            (``get_ontology().canonical_predicate``가 같은 정식 이름을 내는 술어 — 예
            ``ownedBy``·``ownedByGroup``). 트랙 O4 [2026-09 사후].
        sources: 비어 있지 않으면 카드 ``source``가 이 중 하나일 때만 맞다.
    """

    kind: EvidenceKind
    predicate: str
    unit: str | None = None
    subject_role: str = Role.BRAND
    object_role: str | None = None
    reduce: Reduce = Reduce.VALUE
    extra_predicates: tuple[str, ...] = ()
    ontology_aliases: bool = False
    sources: tuple[str, ...] = ()

    @property
    def predicates(self) -> tuple[str, ...]:
        return (self.predicate, *self.extra_predicates)

    def _predicate_matches(self, predicate: str) -> bool:
        if predicate in self.predicates:
            return True
        if not self.ontology_aliases:
            return False
        canonical = _canonical_predicate(predicate)
        return canonical is not None and canonical in {
            _canonical_predicate(p) for p in self.predicates
        }

    def matches(self, card: Evidence) -> bool:
        """종류·술어·단위·출처가 맞는 카드인가 (주어·목적어 역할 판정은 조립기 몫)."""
        return (
            card.kind is self.kind
            and self._predicate_matches(card.predicate)
            and (self.unit is None or card.unit == self.unit)
            and (not self.sources or card.source in self.sources)
        )


@lru_cache(maxsize=256)
def _canonical_predicate(name: str) -> str | None:
    """온톨로지 정식 술어 이름. 온톨로지를 못 읽으면 None (별칭 확장 없이 기존 이름만 쓴다)."""
    try:
        from .ontology import get_ontology

        return get_ontology().canonical_predicate(name)
    except Exception:  # 원본 형식 오류 등 — 규칙 판정은 기존 술어 이름으로 계속한다
        return None


@dataclass(frozen=True)
class InputSpec:
    """규칙 입력 하나의 선언.

    공급원은 셋 중 하나다: ``binding``(증거 카드), ``derivation``(질의 엔티티·다른 입력에서
    계산), 둘 다 없음(``gap``에 이유). ``binding``이 있어도 ``gap``이 있으면 카드 모양은
    정해졌지만 현재 파이프라인·데이터가 그 카드를 만들지 않는다는 뜻이다.
    """

    name: str
    type: InputType
    unit: str | None
    min: float | None = None
    max: float | None = None
    required: bool = True
    binding: EvidenceBinding | None = None
    derivation: str | None = None
    gap: str | None = None
    note: str = ""

    @property
    def available(self) -> bool:
        """현재 파이프라인에서 공급되는가."""
        return self.gap is None and (self.binding is not None or self.derivation is not None)

    def violation(self, value: Any) -> NonFireReason | None:
        """값(None 아님)의 타입·범위 위반."""
        if not _TYPE_CHECKS[self.type](value):
            return NonFireReason.type_mismatch(self.name, value)
        if self.min is not None and value < self.min or self.max is not None and value > self.max:
            return NonFireReason.out_of_range(self.name, value, (self.min, self.max))
        return None


@dataclass(frozen=True)
class RuleContract:
    """규칙 하나의 입력 계약."""

    rule_name: str
    family: str
    inputs: tuple[InputSpec, ...]

    def __post_init__(self) -> None:
        names = [spec.name for spec in self.inputs]
        if len(names) != len(set(names)):
            raise ValueError(f"{self.rule_name}: duplicate input names {names}")

    @property
    def input_names(self) -> frozenset[str]:
        return frozenset(spec.name for spec in self.inputs)

    @property
    def required_names(self) -> frozenset[str]:
        return frozenset(spec.name for spec in self.inputs if spec.required)

    def spec(self, name: str) -> InputSpec:
        for spec in self.inputs:
            if spec.name == name:
                return spec
        raise KeyError(f"{self.rule_name} has no input {name!r}")

    def check(self, context: Mapping[str, Any]) -> NonFireReason | None:
        """결측(필수 입력 전부) → 타입·범위(선언 순서 첫 위반) 순으로 검사."""
        missing = [s.name for s in self.inputs if s.required and context.get(s.name) is None]
        if missing:
            return NonFireReason.missing_input(missing)
        for spec in self.inputs:
            value = context.get(spec.name)
            if value is not None and (reason := spec.violation(value)) is not None:
                return reason
        return None


# =========================================================================
# 판정 결과
# =========================================================================


class NonFireKind(str, Enum):
    MISSING_INPUT = "missing_input"
    OUT_OF_RANGE = "out_of_range"
    TYPE_MISMATCH = "type_mismatch"
    CONDITIONS_NOT_MET = "conditions_not_met"
    NO_CONTRACT = "no_contract"
    CONCLUSION_FAILED = "conclusion_failed"


@dataclass(frozen=True)
class NonFireReason:
    """발화하지 않은 이유.

    Attributes:
        kind: 사유 종류.
        names: 결측 입력 이름들 / 위반 입력 이름 1개 / 실패한 조건 이름들.
        value: 위반 값 (type_mismatch·out_of_range).
        range: 허용 범위 (min, max) — out_of_range.
    """

    kind: NonFireKind
    names: tuple[str, ...] = ()
    value: Any = None
    range: tuple[float | None, float | None] | None = None

    @classmethod
    def missing_input(cls, names: Iterable[str]) -> NonFireReason:
        return cls(NonFireKind.MISSING_INPUT, tuple(names))

    @classmethod
    def out_of_range(
        cls, name: str, value: Any, range: tuple[float | None, float | None]
    ) -> NonFireReason:
        return cls(NonFireKind.OUT_OF_RANGE, (name,), value, tuple(range))  # type: ignore[arg-type]

    @classmethod
    def type_mismatch(cls, name: str, value: Any) -> NonFireReason:
        return cls(NonFireKind.TYPE_MISMATCH, (name,), value)

    @classmethod
    def conditions_not_met(cls, failed_condition_names: Iterable[str]) -> NonFireReason:
        return cls(NonFireKind.CONDITIONS_NOT_MET, tuple(failed_condition_names))

    @classmethod
    def no_contract(cls) -> NonFireReason:
        return cls(NonFireKind.NO_CONTRACT)

    @classmethod
    def conclusion_failed(cls) -> NonFireReason:
        return cls(NonFireKind.CONCLUSION_FAILED)

    def labels(self) -> list[str]:
        """집계용 라벨: ``missing_input:sos``, ``conditions_not_met:sos_above_0.15`` 등."""
        if not self.names:
            return [self.kind.value]
        return [f"{self.kind.value}:{name}" for name in self.names]


@dataclass(frozen=True)
class RuleEvaluation:
    """규칙 하나의 판정.

    Attributes:
        inputs: 계약이 선언한 입력 중 컨텍스트에 값이 있던 것 (inference 카드
            ``derived_from``을 찾는 근거).
    """

    rule_name: str
    fired: bool
    result: InferenceResult | None
    non_fire_reason: NonFireReason | None
    inputs: dict[str, Any] = field(default_factory=dict)


# =========================================================================
# 판정
# =========================================================================


def evaluate_rule(
    rule: InferenceRule,
    context: Mapping[str, Any],
    contracts: Mapping[str, RuleContract] | None = None,
) -> RuleEvaluation:
    """계약 검사 → 조건 → 결론 순으로 규칙 하나를 판정한다 (모듈 설명 참고)."""
    contract = (RULE_CONTRACTS if contracts is None else contracts).get(rule.name)
    if contract is None:
        return RuleEvaluation(rule.name, False, None, NonFireReason.no_contract())

    inputs = {
        spec.name: context.get(spec.name)
        for spec in contract.inputs
        if context.get(spec.name) is not None
    }

    def not_fired(reason: NonFireReason) -> RuleEvaluation:
        return RuleEvaluation(rule.name, False, None, reason, inputs)

    violation = contract.check(context)
    if violation is not None:
        return not_fired(violation)

    ctx = dict(context)
    satisfied, satisfied_names = rule.evaluate_conditions(ctx)
    if not satisfied:
        failed = [c.name for c in rule.conditions if not c.evaluate(ctx)]
        if not failed:  # 조건이 비결정적이면 첫 실패 지점만 남긴다
            failed = [rule.conditions[len(satisfied_names)].name]
        return not_fired(NonFireReason.conditions_not_met(failed))

    result = rule.apply(ctx)
    if result is None:
        return not_fired(NonFireReason.conclusion_failed())
    return RuleEvaluation(rule.name, True, result, None, inputs)


def evaluate_all(
    rules: Iterable[InferenceRule],
    context: Mapping[str, Any],
    contracts: Mapping[str, RuleContract] | None = None,
) -> list[RuleEvaluation]:
    """규칙마다 ``evaluate_rule`` (입력 순서 유지)."""
    return [evaluate_rule(rule, context, contracts) for rule in rules]


def top_non_fire_reasons(
    evaluations: Iterable[RuleEvaluation], limit: int | None = None
) -> list[tuple[str, int]]:
    """미발화 사유 라벨별 빈도, 많은 순(동률은 라벨 순). 평가 리포트의 "미발화 사유 상위"."""
    counts: Counter[str] = Counter()
    for evaluation in evaluations:
        if evaluation.non_fire_reason is not None:
            counts.update(evaluation.non_fire_reason.labels())
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return ranked if limit is None else ranked[:limit]


def count_non_fire_kinds(evaluations: Iterable[RuleEvaluation]) -> dict[NonFireKind, int]:
    """미발화 규칙 수를 사유 종류별로 센다 (규칙 하나당 1)."""
    return dict(
        Counter(e.non_fire_reason.kind for e in evaluations if e.non_fire_reason is not None)
    )


# =========================================================================
# 타입 검사
# =========================================================================


def _is_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)


def _is_str_list(value: Any) -> bool:
    return isinstance(value, list | tuple) and all(isinstance(item, str) for item in value)


_TYPE_CHECKS: dict[InputType, Callable[[Any], bool]] = {
    InputType.NUMBER: _is_number,
    InputType.INTEGER: lambda v: _is_number(v) and float(v).is_integer(),
    InputType.BOOLEAN: lambda v: isinstance(v, bool),
    InputType.STRING: lambda v: isinstance(v, str),
    InputType.STRING_LIST: _is_str_list,
    InputType.TAG_CLUSTERS: lambda v: isinstance(v, Mapping)
    and all(isinstance(k, str) and _is_str_list(tags) for k, tags in v.items()),
    InputType.CLUSTER_KEYS: lambda v: isinstance(v, Mapping) and all(isinstance(k, str) for k in v),
    InputType.RECORD_LIST: lambda v: isinstance(v, list | tuple)
    and all(isinstance(item, Mapping) for item in v),
    InputType.OBJECT_LIST: lambda v: isinstance(v, list | tuple),
    InputType.ANY: lambda v: True,
}


# =========================================================================
# 입력 선언 (같은 이름 = 같은 의미, 모든 규칙에서 공유)
# =========================================================================

_M = EvidenceKind.METRIC
_R = EvidenceKind.RELATION
_U = EvidenceUnit
_CU = ContractUnit

_GAP_PRODUCT_METRICS = (
    "product_metrics 테이블 0행(2026-09-17)이고 카드는 단일 스냅샷만 공급 — "
    "다일자 bsr_rank 카드로 유도할 조회가 없다"
)
_GAP_IR = "IR 보고서는 document 카드(본문)로만 들어온다 — 구조화 수치·플래그 추출기가 없다"
_GAP_KG_EMPTY = "운영 KG(data/knowledge_graph.json, 2026-09-17)에 {predicate} 트리플 0건"

BRAND = InputSpec(
    "brand",
    InputType.STRING,
    None,
    derivation="질의 브랜드 canonical id (brand 역할 카드의 subject)",
)
CATEGORY = InputSpec(
    "category", InputType.STRING, None, derivation="질의 카테고리 id (category 역할 카드)"
)
IS_TARGET = InputSpec(
    "is_target",
    InputType.BOOLEAN,
    None,
    derivation="brand == 'laneige' (타겟 브랜드)",
    note=(
        "트랙 O4 [2026-09 사후] 검토: '질의가 겨냥한 브랜드'로 일반화하지 않는다 — 오프라인 "
        "rule 32문항에서 일반화하면 골드 문항 15개의 발화 규칙이 바뀐다(예: rg006 medicube "
        "top3_achievement, rg028 l'oreal strong_avg_rank). rule 골드는 LANEIGE 기준이다"
    ),
)
ASIN = InputSpec(
    "asin",
    InputType.STRING,
    None,
    derivation="대표 제품 ASIN (relation hasProduct의 object) — 어느 제품인지는 조립기가 정한다",
    note=(
        "build_rule_context는 채우지 않는다: 가격·순위 입력의 대표 제품(metric 카드, 제품명 "
        "주어)과 hasProduct 카드(ASIN)를 잇는 키가 카드에 없어 같은 제품임을 보장할 수 없다"
    ),
)

SOS = InputSpec(
    "sos",
    InputType.NUMBER,
    _U.RATIO,
    0.0,
    1.0,
    binding=EvidenceBinding(_M, "sos", _U.RATIO, Role.BRAND, Role.CATEGORY),
    note=(
        "DB brand_metrics.sos는 퍼센트 — 어댑터가 /100. Top100에 없는 브랜드는 "
        "present_in_top100=False 카드만 있다(관측된 0%로 쓸지는 조립기 결정). "
        "1% 미만 퍼센트 값(예 0.9)은 범위 검사로 못 잡는다"
    ),
)
HHI = InputSpec(
    "hhi",
    InputType.NUMBER,
    _U.INDEX_0_1,
    0.01,  # Top100 브랜드 수 N ≤ 100 → HHI ≥ 1/N ≥ 0.01. 0은 계산되지 않은 자리표시자다
    1.0,
    binding=EvidenceBinding(_M, "hhi", _U.INDEX_0_1, Role.CATEGORY),
    note=(
        "정본 0~1 (결정 D1). KG hasHHI 엣지(0~10000, 날짜 없음)는 증거가 아니다(F12). "
        "운영 DB market_metrics.hhi=0 19행(2025-12-16~25)은 brand_metrics·평균가가 없는 "
        "미계산 행이라 하한 0.01로 막는다"
    ),
)
CPI = InputSpec(
    "cpi",
    InputType.NUMBER,
    _U.INDEX_100,
    0.0,
    None,
    binding=EvidenceBinding(_M, "cpi", _U.INDEX_100, Role.BRAND, Role.CATEGORY),
    note="질의 브랜드 brand_share 사실에만 실린다 (brand_metrics.cpi non-null 7,286/13,838행)",
)
RATING_GAP = InputSpec(
    "rating_gap",
    InputType.NUMBER,
    _U.RATING_POINTS,
    -4.0,
    4.0,
    binding=EvidenceBinding(_M, "avg_rating_gap", _U.RATING_POINTS, Role.BRAND, Role.CATEGORY),
    note="규칙 키 rating_gap = 카드 술어 avg_rating_gap (질의 브랜드만)",
)
AVG_RANK = InputSpec(
    "avg_rank",
    InputType.NUMBER,
    _U.RANK,
    1.0,
    100.0,
    binding=EvidenceBinding(_M, "brand_avg_rank", _U.RANK, Role.BRAND, Role.CATEGORY),
    note="규칙 키 avg_rank = 카드 술어 brand_avg_rank (질의 브랜드만)",
)
CHURN_RATE = InputSpec(
    "churn_rate",
    InputType.NUMBER,
    _U.RATIO,
    0.0,
    1.0,
    binding=EvidenceBinding(_M, "churn_rate", _U.RATIO, Role.CATEGORY),
    gap="market_metrics.churn_rate 전부 NULL(509/509행, 2026-09-17) — 카드가 만들어지지 않는다",
)
SOS_CHANGE = InputSpec(
    "sos_change",
    InputType.NUMBER,
    _CU.RATIO_DELTA,
    -1.0,
    1.0,
    gap=(
        "카드는 단일 스냅샷만 공급 — 두 as_of의 sos 카드 차이로 유도할 수 있으나 다일자 조회가 없다"
    ),
)
_COMPETITOR_RELATION = EvidenceBinding(
    _R,
    "competesWith",
    None,
    Role.BRAND,
    Role.BRAND,
    Reduce.COUNT,
    extra_predicates=("directCompetitor", "indirectCompetitor"),
)
COMPETITOR_COUNT = InputSpec(
    "competitor_count",
    InputType.INTEGER,
    _U.COUNT,
    0,
    None,
    binding=_COMPETITOR_RELATION,
    note=(
        "OntologyReasoner._enrich_context는 KG가 연결돼 있으면 이 값을 "
        "len(kg.get_competitors(brand))로 덮어쓴다 — (경쟁사, 카테고리) 쌍의 수다. 카드는 "
        "경쟁사별로 병합되므로 서로 다른 브랜드 수 — 둘은 다를 수 있다"
    ),
)
COMPETITORS = InputSpec(
    "competitors",
    InputType.RECORD_LIST,
    None,
    binding=replace(_COMPETITOR_RELATION, reduce=Reduce.OBJECTS),
    note="규칙은 [{'brand': 이름}, ...]을 읽는다 — object를 {'brand': object}로 감싼다",
)
PRODUCTS = InputSpec(
    "products",
    InputType.OBJECT_LIST,
    None,
    binding=EvidenceBinding(_R, "hasProduct", None, Role.BRAND, Role.ENTITY, Reduce.OBJECTS),
    note="_enrich_context는 제품 dict 목록을 넣어 related_entities에 dict가 섞인다",
)

HAS_RANK_SHOCK = InputSpec("has_rank_shock", InputType.BOOLEAN, None, gap=_GAP_PRODUCT_METRICS)
RANK_CHANGE_7D = InputSpec(
    "rank_change_7d", InputType.INTEGER, _CU.RANK_DELTA, -99, 99, gap=_GAP_PRODUCT_METRICS
)
RANK_VOLATILITY = InputSpec(
    "rank_volatility", InputType.NUMBER, _CU.RANK_STDDEV, 0.0, 100.0, gap=_GAP_PRODUCT_METRICS
)
STREAK_DAYS = InputSpec(
    "streak_days", InputType.INTEGER, _CU.DAYS, 0, None, gap=_GAP_PRODUCT_METRICS
)
RATING_TREND = InputSpec(
    "rating_trend", InputType.NUMBER, _CU.RATING_SLOPE, -4.0, 4.0, gap=_GAP_PRODUCT_METRICS
)
CURRENT_RANK = InputSpec(
    "current_rank",
    InputType.INTEGER,
    _U.RANK,
    1,
    100,
    binding=EvidenceBinding(_M, "bsr_rank", _U.RANK, Role.PRODUCT, Role.CATEGORY, Reduce.MIN_VALUE),
    note="카드 subject는 제품명, metadata.brand가 브랜드 — 질의 브랜드 제품 중 최고 순위",
)
RANK = InputSpec(
    "rank",
    InputType.INTEGER,
    _U.RANK,
    1,
    100,
    binding=EvidenceBinding(_M, "bsr_rank", _U.RANK, Role.PRODUCT, Role.CATEGORY),
    note="price와 같은 제품의 순위여야 한다",
)
PRICE = InputSpec(
    "price",
    InputType.NUMBER,
    _U.USD,
    0.0,
    None,
    binding=EvidenceBinding(_M, "price", _U.USD, Role.PRODUCT, Role.CATEGORY),
)
CATEGORY_AVG_PRICE = InputSpec(
    "category_avg_price",
    InputType.NUMBER,
    _U.USD,
    0.0,
    None,
    binding=EvidenceBinding(_M, "avg_price", _U.USD, Role.CATEGORY),
    note="규칙 키 category_avg_price = 카드 술어 avg_price",
)
REVIEW_COUNT = InputSpec(
    "review_count",
    InputType.INTEGER,
    _U.COUNT,
    0,
    None,
    binding=EvidenceBinding(_M, "reviews_count", _U.COUNT, Role.PRODUCT, Role.CATEGORY),
    note="규칙 키 review_count = 카드 술어 reviews_count",
)
TREND_KEYWORDS = InputSpec(
    "trend_keywords",
    InputType.STRING_LIST,
    None,
    binding=EvidenceBinding(_R, "hasTrend", None, Role.BRAND, Role.RAW, Reduce.OBJECTS),
    gap=_GAP_KG_EMPTY.format(predicate="hasTrend"),
    note="카드 metadata.scope=brand_or_market — 브랜드 트렌드인지 시장 트렌드인지 모호",
)
PRICE_STABLE = InputSpec(
    "price_stable", InputType.BOOLEAN, None, gap="가격 이력 카드 없음(단일 스냅샷만 공급)"
)
BADGE = InputSpec(
    "badge",
    InputType.STRING,
    None,
    gap=(
        "raw_data.badge에 'Best Seller' 0건 — 값이 평점 문자열('4.6' 등)로 채워져 있다"
        "(스크레이퍼 필드 매핑 문제로 추정). MetricFactsProvider 미조회, 카드 술어 없음"
    ),
)
DISCOUNT_PERIODS = InputSpec(
    "discount_periods",
    InputType.RECORD_LIST,
    None,
    gap="raw_data.discount_percent 전부 NULL(2026-09-17), 기간 구조 카드 없음",
    note="[{'start': ISO, 'end': ISO}, ...]",
)
RANK_IMPROVEMENTS = InputSpec(
    "rank_improvements",
    InputType.RECORD_LIST,
    None,
    gap="다일자 순위 카드 없음",
    note="[{'start': ISO, 'end': ISO}, ...]",
)
PRODUCT_HISTORY = InputSpec(
    "product_history",
    InputType.RECORD_LIST,
    None,
    gap="raw_data.discount_percent 전부 NULL + 다일자 카드 없음",
    note="[{'rank', 'discount_percent', 'date'}, ...] 날짜순",
)

SENTIMENT_CLUSTERS = InputSpec(
    "sentiment_clusters",
    InputType.TAG_CLUSTERS,
    None,
    binding=EvidenceBinding(
        _R, "hasSentiment", None, Role.PRODUCT, Role.RAW, Reduce.OBJECTS_BY_CLUSTER
    ),
    gap=_GAP_KG_EMPTY.format(predicate="hasSentiment"),
    note=(
        "{cluster: [tag]}. 현 HybridRetriever는 브랜드 프로필의 {cluster: 빈도}를 넣어 "
        "타입이 다르다(Hydration·Effectiveness 개수 조건이 len(int) 예외로 침묵)"
    ),
)
SENTIMENT_TAGS = InputSpec(
    "sentiment_tags",
    InputType.STRING_LIST,
    None,
    binding=EvidenceBinding(_R, "brandSentiment", None, Role.BRAND, Role.RAW, Reduce.OBJECTS),
    gap=_GAP_KG_EMPTY.format(predicate="hasSentiment"),
)
_ABSENT_NOT_EMPTY = (
    "빈 값은 '경쟁사에 해당 감성 없음'으로 발화 근거가 된다 — 경쟁사 감성 데이터가 없으면 "
    "빈 값이 아니라 결측으로 둬야 한다"
)
COMPETITOR_SENTIMENT_TAGS = InputSpec(
    "competitor_sentiment_tags",
    InputType.STRING_LIST,
    None,
    binding=EvidenceBinding(
        _R, "brandSentiment", None, Role.COMPETITOR_BRAND, Role.RAW, Reduce.OBJECTS
    ),
    gap=_GAP_KG_EMPTY.format(predicate="hasSentiment"),
    note=_ABSENT_NOT_EMPTY,
)
COMPETITOR_SENTIMENT_CLUSTERS = InputSpec(
    "competitor_sentiment_clusters",
    InputType.CLUSTER_KEYS,
    None,
    binding=EvidenceBinding(
        _R, "hasSentiment", None, Role.COMPETITOR_PRODUCT, Role.RAW, Reduce.OBJECTS_BY_CLUSTER
    ),
    gap=_GAP_KG_EMPTY.format(predicate="hasSentiment"),
    note=_ABSENT_NOT_EMPTY,
)
AI_SUMMARY = InputSpec(
    "ai_summary",
    InputType.STRING,
    None,
    binding=EvidenceBinding(_R, "hasAISummary", None, Role.PRODUCT, None, Reduce.DETAIL),
    gap=_GAP_KG_EMPTY.format(predicate="hasAISummary"),
)

IR_MENTIONS_PRIME_DAY = InputSpec("ir_mentions_prime_day", InputType.BOOLEAN, None, gap=_GAP_IR)
RANK_CHANGE_DURING_EVENT = InputSpec(
    "rank_change_during_event", InputType.INTEGER, _CU.RANK_DELTA, -99, 99, gap=_GAP_IR
)
IR_AMERICAS_YOY = InputSpec(
    "ir_americas_yoy", InputType.NUMBER, _CU.PERCENT, -100.0, None, gap=_GAP_IR
)
IR_CONSECUTIVE_GROWTH_QUARTERS = InputSpec(
    "ir_consecutive_growth_quarters", InputType.INTEGER, _CU.QUARTERS, 0, None, gap=_GAP_IR
)
IR_PREV_QTR_GROWTH = InputSpec(
    "ir_prev_qtr_growth", InputType.NUMBER, _CU.PERCENT, -100.0, None, gap=_GAP_IR
)
IR_CURRENT_QTR_GROWTH = InputSpec(
    "ir_current_qtr_growth", InputType.NUMBER, _CU.PERCENT, -100.0, None, gap=_GAP_IR
)
IR_CAMPAIGN_MENTIONED = InputSpec("ir_campaign_mentioned", InputType.BOOLEAN, None, gap=_GAP_IR)
CAMPAIGN_NAME = InputSpec("campaign_name", InputType.STRING, None, gap=_GAP_IR)
IR_SOURCE = InputSpec("ir_source", InputType.STRING, None, gap=_GAP_IR)
PARENT_GROUP = InputSpec(
    "parent_group",
    InputType.STRING,
    None,
    binding=EvidenceBinding(
        _R, "ownedBy", None, Role.BRAND, Role.ENTITY, Reduce.OBJECT, ontology_aliases=True
    ),
    note=(
        "카드 object(소문자 canonical 'amorepacific')를 그대로 넣는다 — 규칙이 casefold로 "
        "비교한다(트랙 3-B에서 대소문자 구분 비교 결함 수정). 술어는 레거시 ownedBy와 "
        "온톨로지 정식 이름 ownedByGroup(플래그 ontology.use_class_reasoning ON 카드) 모두 — "
        "OFF 어댑터는 ownedByGroup을 ownedBy로 바꾸므로 OFF 판정은 같다(트랙 O4)"
    ),
)

# 등록부 정적 사실 카드의 출처 (= ``src/rag/evidence_adapters.ONTOLOGY_SOURCE``, 트랙 O3).
# 원산지·세그먼트·인수 입력은 이 카드만 읽는다: KG의 AP 브랜드 원산지 "Korea"는
# kg_updater 기본값이라 원본 진술이 아니다(결정 OA-5). 이 카드는 플래그 ON에서만 생긴다.
REGISTRY_SOURCE = "ontology:registry"


def _registry_profile(predicate: str, object_role: str) -> EvidenceBinding:
    return EvidenceBinding(
        _R,
        predicate,
        None,
        Role.BRAND,
        object_role,
        Reduce.OBJECT_LABEL,
        ontology_aliases=True,
        sources=(REGISTRY_SOURCE,),
    )


_REGISTRY_PROFILE_NOTE = (
    "등록부 표시 이름(카드 metadata.object_display_name, 없으면 object)을 넣는다 — 규칙은 "
    "결론 문장·metadata에만 쓴다(조건 없음). 등록부에 값이 없으면 결측"
)
COUNTRY_OF_ORIGIN = InputSpec(
    "country_of_origin",
    InputType.STRING,
    None,
    binding=_registry_profile("originatesFrom", Role.ENTITY),
    note=_REGISTRY_PROFILE_NOTE + ". 결측이면 규칙 결론이 기본값 'Korea'를 쓴다(규칙 몫)",
)
ACQUIRED = InputSpec(
    "acquired",
    InputType.ANY,
    None,
    binding=_registry_profile("acquiredIn", Role.RAW),
    note=_REGISTRY_PROFILE_NOTE + ". 카드 object는 연도 문자열('2024')",
)
SEGMENT = InputSpec(
    "segment",
    InputType.STRING,
    None,
    binding=_registry_profile("hasSegment", Role.ENTITY),
    note=_REGISTRY_PROFILE_NOTE,
)
EVIDENCE_SOURCES = InputSpec(
    "evidence", InputType.STRING_LIST, None, gap="공급원 없음 — 규칙 기본값을 쓴다"
)


def _contract(
    rule_name: str,
    family: str,
    required: Iterable[InputSpec],
    optional: Iterable[InputSpec] = (),
) -> RuleContract:
    inputs = (
        *(replace(spec, required=True) for spec in required),
        *(replace(spec, required=False) for spec in optional),
    )
    return RuleContract(rule_name=rule_name, family=family, inputs=inputs)


_CONTRACT_LIST: tuple[RuleContract, ...] = (
    # --- market_rules.py -------------------------------------------------
    _contract("market_dominance_fragmented", "market", [SOS, HHI], [BRAND]),
    _contract("market_dominance_concentrated", "market", [SOS, HHI], [BRAND]),
    _contract("challenger_position", "market", [HHI, SOS], [BRAND]),
    _contract("fragmented_market_competition", "market", [HHI, COMPETITOR_COUNT], [COMPETITORS]),
    _contract("strong_avg_rank", "market", [AVG_RANK, IS_TARGET], [BRAND]),
    _contract("competitive_pressure", "market", [SOS_CHANGE, COMPETITOR_COUNT], [COMPETITORS]),
    # --- alert_rules.py --------------------------------------------------
    _contract("price_quality_mismatch", "alert", [CPI, RATING_GAP], [BRAND, ASIN]),
    _contract("market_disruption", "alert", [HAS_RANK_SHOCK, CHURN_RATE], [PRODUCTS]),
    _contract("rank_decline_alert", "alert", [RANK_CHANGE_7D, RANK_VOLATILITY], [ASIN]),
    # --- growth_rules.py -------------------------------------------------
    _contract("stable_growth", "growth", [STREAK_DAYS, RANK_CHANGE_7D], [ASIN, BRAND]),
    _contract("trend_alignment_opportunity", "growth", [TREND_KEYWORDS, BRAND], [IS_TARGET]),
    _contract(
        "top10_stability",
        "growth",
        [CURRENT_RANK, STREAK_DAYS, RANK_VOLATILITY],
        [ASIN, BRAND],
    ),
    _contract("category_entry_opportunity", "growth", [HHI, SOS, IS_TARGET], [CATEGORY, BRAND]),
    _contract("rating_momentum_positive", "growth", [RATING_TREND, REVIEW_COUNT], [ASIN]),
    _contract("top3_achievement", "growth", [CURRENT_RANK, BRAND], [IS_TARGET, CATEGORY, ASIN]),
    _contract("strong_rating_position", "growth", [RATING_GAP, IS_TARGET], [BRAND]),
    # --- price_rules.py --------------------------------------------------
    _contract("value_position", "price", [CPI, RATING_GAP], [ASIN, BRAND]),
    _contract("premium_price_position", "price", [CPI, RATING_GAP], [BRAND]),
    _contract("discount_dependent", "price", [DISCOUNT_PERIODS, RANK_IMPROVEMENTS], [ASIN, BRAND]),
    _contract("viral_effect", "price", [PRICE_STABLE, RANK_CHANGE_7D], [ASIN, BRAND]),
    _contract("bestseller_badge_effect", "price", [BADGE, RANK_CHANGE_7D], [ASIN, BRAND]),
    _contract("high_discount_dependency_score", "price", [PRODUCT_HISTORY], [ASIN, BRAND]),
    _contract("premium_defense_success", "price", [PRICE, CATEGORY_AVG_PRICE, RANK], [ASIN, BRAND]),
    # --- sentiment_rules.py ----------------------------------------------
    _contract("sentiment_strength_hydration", "sentiment", [SENTIMENT_CLUSTERS], [ASIN, BRAND]),
    _contract(
        "sentiment_value_advantage",
        "sentiment",
        [SENTIMENT_TAGS, COMPETITOR_SENTIMENT_TAGS],
        [ASIN, BRAND],
    ),
    _contract(
        "sentiment_weakness_packaging",
        "sentiment",
        [SENTIMENT_CLUSTERS, COMPETITOR_SENTIMENT_CLUSTERS],
        [ASIN, BRAND],
    ),
    _contract("sentiment_usability_strength", "sentiment", [SENTIMENT_CLUSTERS], [ASIN, BRAND]),
    _contract("sentiment_effectiveness_strong", "sentiment", [SENTIMENT_CLUSTERS], [ASIN, BRAND]),
    _contract(
        "sentiment_gap_sensory",
        "sentiment",
        [SENTIMENT_CLUSTERS, COMPETITOR_SENTIMENT_CLUSTERS],
        [ASIN, BRAND],
    ),
    _contract("customer_perception_positive", "sentiment", [AI_SUMMARY], [ASIN, BRAND]),
    _contract("customer_perception_mixed", "sentiment", [AI_SUMMARY], [ASIN, BRAND]),
    # --- ir_rules.py -----------------------------------------------------
    _contract(
        "ir_prime_day_impact",
        "ir",
        [IR_MENTIONS_PRIME_DAY, RANK_CHANGE_DURING_EVENT],
        [BRAND, CATEGORY],
    ),
    _contract(
        "ir_americas_revenue_correlation",
        "ir",
        [IR_AMERICAS_YOY, SOS_CHANGE, IS_TARGET],
        [BRAND],
    ),
    _contract("ir_growth_momentum", "ir", [IR_CONSECUTIVE_GROWTH_QUARTERS], [BRAND]),
    _contract(
        "ir_growth_slowdown_warning", "ir", [IR_PREV_QTR_GROWTH, IR_CURRENT_QTR_GROWTH], [BRAND]
    ),
    _contract(
        "ir_brand_campaign_effect",
        "ir",
        [IR_CAMPAIGN_MENTIONED, RANK_CHANGE_7D],
        [BRAND, ASIN, CAMPAIGN_NAME, IR_SOURCE],
    ),
    _contract(
        "brand_ownership_verification",
        "ir",
        [PARENT_GROUP],
        [BRAND, COUNTRY_OF_ORIGIN, ACQUIRED, SEGMENT, EVIDENCE_SOURCES],
    ),
)

RULE_CONTRACTS: dict[str, RuleContract] = {c.rule_name: c for c in _CONTRACT_LIST}


# =========================================================================
# 증거 카드 → 규칙 컨텍스트 (트랙 3-B)
# =========================================================================
#
# 입력은 카드뿐이다. KG 수치 엣지·대시보드 JSON은 읽지 않는다(E2·F7). 입력을 채울 카드가
# 없으면 컨텍스트에 키를 넣지 않는다 — 0이나 기본값으로 채우지 않는다(계약 검사가
# missing_input으로 기록한다).

TARGET_BRAND = "laneige"  # IS_TARGET 파생 규칙: brand == 'laneige'

# 조합 상한 = MetricFactsProvider.MAX_BRANDS · MAX_CATEGORIES (src/rag/metric_facts.py).
# 제공자가 브랜드 3개·카테고리 3개까지만 조회하므로 그 밖의 조합에는 metric 카드가 없다 —
# 평가해도 missing_input만 쌓여 미발화 사유 집계를 부풀린다.
MAX_RULE_BRANDS = 3
MAX_RULE_CATEGORIES = 3

# 평가 리포트(트랙 3-C)의 "미발화 사유 상위" 개수
NON_FIRE_TOP_LIMIT = 10

_BRAND_SCOPED_ROLES = frozenset(
    {Role.BRAND, Role.PRODUCT, Role.COMPETITOR_BRAND, Role.COMPETITOR_PRODUCT}
)
_BRAND_DERIVED_INPUTS = frozenset({"brand", "is_target"})
_ABSENCE_PREDICATE = "present_in_top100"  # EvidenceAdapter._absent_card

Combination = tuple[str | None, str | None]


def _unique(values: Iterable[Any]) -> list[Any]:
    """빈 값을 빼고 순서를 지키며 중복 제거."""
    seen: list[Any] = []
    for value in values:
        if value and value not in seen:
            seen.append(value)
    return seen


def _latest_first(cards: list[Evidence]) -> Evidence:
    """여러 카드가 맞으면 as_of가 가장 늦은 카드, 같으면 먼저 온 카드."""
    return max(enumerate(cards), key=lambda item: (item[1].as_of or "", -item[0]))[1]


class _CardIndex:
    """(브랜드, 카테고리) 조합 하나에서 카드가 입력 역할에 맞는지 판정한다."""

    def __init__(self, cards: Iterable[Evidence], brand: str | None, category: str | None):
        self.cards = list(cards)
        self.brand = brand
        self.category = category
        relations = [c for c in self.cards if c.kind is EvidenceKind.RELATION]
        competitor_predicates = _COMPETITOR_RELATION.predicates
        self.competitors = frozenset(
            c.object
            for c in relations
            if brand and c.subject == brand and c.predicate in competitor_predicates and c.object
        )
        products = [c for c in relations if c.predicate == "hasProduct" and c.object]
        self.brand_asins = frozenset(c.object for c in products if brand and c.subject == brand)
        self.competitor_asins = frozenset(
            c.object for c in products if c.subject in self.competitors
        )
        # 대표 제품: 질의 브랜드의 이 카테고리 제품 중 최고 순위 (같은 순위면 먼저 온 카드).
        # 가격·순위·리뷰 수 입력은 모두 이 제품 하나에서 읽는다 — 서로 다른 제품의 값이
        # 한 규칙에 섞이지 않게 (RANK.note).
        ranked = [
            c
            for c in self.cards
            if self._is_brand_product_metric(c)
            and c.predicate == "bsr_rank"
            and _is_number(c.value)
        ]
        self.product = min(ranked, key=lambda c: c.value).subject if ranked else None

    def _is_brand_product_metric(self, card: Evidence) -> bool:
        return (
            card.kind is EvidenceKind.METRIC
            and self.brand is not None
            and card.metadata.get("brand") == self.brand
            and self.category is not None
            and card.object == self.category
        )

    def _subject_matches(self, binding: EvidenceBinding, card: Evidence) -> bool:
        role = binding.subject_role
        if role == Role.BRAND:
            return self.brand is not None and card.subject == self.brand
        if role == Role.CATEGORY:
            return self.category is not None and card.subject == self.category
        if role == Role.PRODUCT:
            if card.kind is EvidenceKind.METRIC:
                if not self._is_brand_product_metric(card):
                    return False
                return binding.reduce is Reduce.MIN_VALUE or card.subject == self.product
            return card.subject in self.brand_asins
        if role == Role.COMPETITOR_BRAND:
            return card.subject in self.competitors
        if role == Role.COMPETITOR_PRODUCT:
            return card.subject in self.competitor_asins
        return False  # ENTITY·RAW 주어를 쓰는 바인딩은 없다

    def _object_matches(self, binding: EvidenceBinding, card: Evidence) -> bool:
        role = binding.object_role
        if role is None:
            return card.object is None
        if role == Role.CATEGORY:
            return self.category is not None and card.object == self.category
        return card.object is not None

    def matching(self, binding: EvidenceBinding) -> list[Evidence]:
        return [
            card
            for card in self.cards
            if binding.matches(card)
            and self._subject_matches(binding, card)
            and self._object_matches(binding, card)
        ]


def _reduce(spec: InputSpec, cards: list[Evidence]) -> tuple[Any, list[Evidence]]:
    """매칭 카드 → (입력 값, 값을 만든 카드). 값을 만들지 못하면 (None, [])."""
    binding = spec.binding
    if binding is None:
        return None, []
    reduce = binding.reduce
    if reduce is Reduce.VALUE:
        with_value = [c for c in cards if c.value is not None]
        if not with_value:
            return None, []
        card = _latest_first(with_value)
        return card.value, [card]
    if reduce is Reduce.MIN_VALUE:
        numbers = [c for c in cards if _is_number(c.value)]
        if not numbers:
            return None, []
        card = min(numbers, key=lambda c: c.value)  # 같은 값이면 먼저 온 카드
        return card.value, [card]
    if reduce is Reduce.OBJECT:
        with_object = [c for c in cards if c.object is not None]
        if not with_object:
            return None, []
        card = _latest_first(with_object)
        return card.object, [card]
    if reduce is Reduce.OBJECT_LABEL:
        with_object = [c for c in cards if c.object is not None]
        if not with_object:
            return None, []
        card = _latest_first(with_object)
        return card.metadata.get("object_display_name") or card.object, [card]
    if reduce in (Reduce.OBJECTS, Reduce.COUNT):
        with_object = [c for c in cards if c.object is not None]
        objects = _unique(c.object for c in with_object)
        if not objects:
            return None, []  # 빈 목록이 아니라 결측 (_ABSENT_NOT_EMPTY)
        if reduce is Reduce.COUNT:
            return len(objects), with_object
        if spec.type is InputType.RECORD_LIST:  # 규칙은 [{'brand': 이름}, ...]을 읽는다
            return [{"brand": obj} for obj in objects], with_object
        return objects, with_object
    if reduce is Reduce.OBJECTS_BY_CLUSTER:
        clustered = [c for c in cards if c.object is not None and c.metadata.get("cluster")]
        if not clustered:
            return None, []
        clusters: dict[str, list[str]] = {}
        for card in clustered:
            tags = clusters.setdefault(str(card.metadata["cluster"]), [])
            if card.object not in tags:
                tags.append(card.object)
        return clusters, clustered
    if reduce is Reduce.DETAIL:
        with_detail = [c for c in cards if c.detail]
        if not with_detail:
            return None, []
        return with_detail[0].detail, [with_detail[0]]
    raise ValueError(f"unknown reduce {reduce!r}")


def _derive(name: str, brand: str | None, category: str | None) -> Any:
    """질의 엔티티에서 파생하는 입력. ``asin``은 만들지 않는다(``ASIN.note``)."""
    if name == "brand":
        return brand
    if name == "category":
        return category
    if name == "is_target":
        return None if brand is None else brand == TARGET_BRAND
    return None


def _unique_specs(contracts: Mapping[str, RuleContract]) -> list[InputSpec]:
    """입력 이름별 선언 하나 (같은 이름 = 같은 의미 — 계약 테스트가 보장)."""
    specs: dict[str, InputSpec] = {}
    for contract in contracts.values():
        for spec in contract.inputs:
            specs.setdefault(spec.name, spec)
    return list(specs.values())


def build_rule_context(
    cards: Iterable[Evidence],
    brand: str | None,
    category: str | None,
    contracts: Mapping[str, RuleContract] | None = None,
) -> tuple[dict[str, Any], dict[str, tuple[str, ...]]]:
    """증거 카드 → 규칙 컨텍스트.

    Args:
        cards: metric·relation 카드 (``EvidenceAdapter`` 출력).
        brand: 질의 브랜드 canonical id (``EvidenceAdapter.normalize_brand``) 또는 None.
        category: 카테고리 id 또는 None.

    Returns:
        (컨텍스트, 입력 이름 → 값을 만든 카드 id). 파생 입력(brand·category·is_target)은
        카드 id가 없다. 값을 만들지 못한 입력은 둘 다에 없다.
    """
    index = _CardIndex(cards, brand, category)
    context: dict[str, Any] = {}
    card_ids: dict[str, tuple[str, ...]] = {}
    for spec in _unique_specs(RULE_CONTRACTS if contracts is None else contracts):
        if spec.binding is not None:
            value, used = _reduce(spec, index.matching(spec.binding))
            if value is None:
                continue
            context[spec.name] = value
            card_ids[spec.name] = tuple(_unique(card.id for card in used))
        elif spec.derivation is not None:
            value = _derive(spec.name, brand, category)
            if value is not None:
                context[spec.name] = value
    return context, card_ids


def rule_combinations(
    cards: Iterable[Evidence], brands: Iterable[str], categories: Iterable[str]
) -> list[Combination]:
    """평가할 (브랜드, 카테고리) 조합.

    - 브랜드가 없으면 (None, 카테고리)마다.
    - 질의 카테고리가 없으면 그 브랜드가 진입한 카테고리 — 브랜드 주어 metric 카드의 object
      (카드 순서 = MetricFactsProvider의 점유율 순). 부재 카드(``present_in_top100``)는 진입이
      아니다 — 제공자는 브랜드만 링크된 질의에서 브랜드들이 진입한 카테고리의 합집합마다 각
      브랜드의 점유율·부재를 싣는다. 진입한 카테고리가 없으면 (브랜드, None).
    - 브랜드 ``MAX_RULE_BRANDS``개 × 카테고리 ``MAX_RULE_CATEGORIES``개까지.
    """
    card_list = list(cards)
    brand_list = _unique(brands)[:MAX_RULE_BRANDS]
    category_list = _unique(categories)[:MAX_RULE_CATEGORIES]
    if not brand_list:
        return [(None, category) for category in category_list]
    combinations: list[Combination] = []
    for brand in brand_list:
        entered = category_list or _unique(
            c.object
            for c in card_list
            if c.kind is EvidenceKind.METRIC
            and c.subject == brand
            and c.predicate != _ABSENCE_PREDICATE
        )
        entered = entered[:MAX_RULE_CATEGORIES]
        if entered:
            combinations.extend((brand, category) for category in entered)
        else:
            combinations.append((brand, None))
    return combinations


@dataclass(frozen=True)
class CardRuleEvaluation:
    """조합 하나에서 규칙 하나의 판정과 그 근거 카드.

    Attributes:
        derived_from: 판정에 쓰인 입력(``evaluation.inputs``)의 카드 id — 정렬·중복 제거.
        scope: 결과가 말하는 대상 (브랜드, 카테고리). 규칙이 브랜드(카테고리) 범위 입력을
            읽지 않았으면 None — 카테고리를 읽지 않은 규칙의 추론 카드가 그 카테고리의
            사실처럼 보이지 않게.
        as_of: 근거 카드 as_of 중 가장 늦은 날짜.
        identity: 중복 제거 키 (``evaluate_rules_on_cards`` 참고).
    """

    combination: Combination
    evaluation: RuleEvaluation
    derived_from: tuple[str, ...]
    scope: Combination
    as_of: str | None
    identity: tuple[Any, ...]

    def result(self) -> InferenceResult | None:
        """발화했으면 근거 카드 id·범위·시점을 담은 결과 (규칙 결과의 사본)."""
        result = self.evaluation.result
        if result is None:
            return None
        snapshot: dict[str, Any] = {
            name: value
            for name, value in self.evaluation.inputs.items()
            if isinstance(value, str | int | float | bool) and name not in ("brand", "category")
        }
        brand, category = self.scope
        if brand is not None:
            snapshot["brand"] = brand
        if category is not None:
            snapshot["category"] = category
        if self.as_of is not None:
            snapshot["as_of"] = self.as_of
        evidence = {
            **result.evidence,
            "context_snapshot": snapshot,
            "derived_from": list(self.derived_from),
        }
        return replace(result, evidence=evidence)


def _evaluate_on_context(
    rule: InferenceRule,
    combination: Combination,
    context: Mapping[str, Any],
    card_ids: Mapping[str, tuple[str, ...]],
    cards_by_id: Mapping[str, Evidence],
    contracts: Mapping[str, RuleContract],
) -> CardRuleEvaluation:
    evaluation = evaluate_rule(rule, context, contracts)
    contract = contracts.get(rule.name)
    used = list(evaluation.inputs)
    derived = sorted({card_id for name in used for card_id in card_ids.get(name, ())})

    brand_scoped = category_scoped = False
    for name in used:
        binding = contract.spec(name).binding if contract is not None else None
        if name in _BRAND_DERIVED_INPUTS or (
            binding is not None and binding.subject_role in _BRAND_SCOPED_ROLES
        ):
            brand_scoped = True
        if name == "category" or (
            binding is not None and Role.CATEGORY in (binding.subject_role, binding.object_role)
        ):
            category_scoped = True
    brand, category = combination
    scope = (brand if brand_scoped else None, category if category_scoped else None)

    dates = [cards_by_id[i].as_of for i in derived if i in cards_by_id and cards_by_id[i].as_of]
    identity = (
        rule.name,
        tuple(
            sorted(
                (name, ("cards", card_ids[name]) if name in card_ids else ("value", repr(value)))
                for name, value in evaluation.inputs.items()
            )
        ),
    )
    return CardRuleEvaluation(
        combination=combination,
        evaluation=evaluation,
        derived_from=tuple(derived),
        scope=scope,
        as_of=max(dates) if dates else None,
        identity=identity,
    )


@dataclass(frozen=True)
class RuleRun:
    """질의 하나의 규칙 평가 (중복 제거 후)."""

    combinations: list[Combination]
    evaluations: list[CardRuleEvaluation]

    def fired_results(self) -> list[InferenceResult]:
        """발화 결과 (조합 순 → 규칙 우선순위 순)."""
        results = (e.result() for e in self.evaluations if e.evaluation.fired)
        return [result for result in results if result is not None]

    def summary(self) -> dict[str, Any]:
        """``HybridContext.metadata["rule_evaluation"]`` — 키·모양은 평가 리포트(3-C)가 전제한다.

        - combinations: ``[[brand, category], ...]``
        - evaluated: 중복 제거 후 판정 수
        - fired: 발화한 규칙 이름 (중복 없음, 처음 발화한 순서)
        - non_fire_top: ``[[라벨, 개수], ...]`` 상위 ``NON_FIRE_TOP_LIMIT``개
        - non_fire_counts_by_kind: ``{사유 종류: 미발화 판정 수}``
        """
        evaluations = [e.evaluation for e in self.evaluations]
        return {
            "combinations": [[brand, category] for brand, category in self.combinations],
            "evaluated": len(evaluations),
            "fired": _unique(e.rule_name for e in evaluations if e.fired),
            "non_fire_top": [
                [label, count]
                for label, count in top_non_fire_reasons(evaluations, NON_FIRE_TOP_LIMIT)
            ],
            "non_fire_counts_by_kind": {
                kind.value: count for kind, count in count_non_fire_kinds(evaluations).items()
            },
        }


def evaluate_rules_on_cards(
    rules: Iterable[InferenceRule],
    cards: Iterable[Evidence],
    brands: Iterable[str],
    categories: Iterable[str],
    contracts: Mapping[str, RuleContract] | None = None,
) -> RuleRun:
    """질의 엔티티 조합마다 카드로 컨텍스트를 만들어 규칙을 판정한다 (``rule_combinations``).

    중복 제거: 판정의 정체는 (규칙 이름, 판정에 쓰인 입력마다 근거 카드 id 또는 파생 값)이다.
    같은 정체는 처음 조합의 판정 하나만 남긴다 — 카테고리를 읽지 않는 규칙(예:
    brand_ownership_verification)이 브랜드가 진입한 카테고리 수만큼 중복 발화하거나, 같은
    결측 사유가 조합 수만큼 집계되지 않게. 입력 카드가 다르면(카테고리별 SoS 등) 별개 판정이다.
    규칙은 입력 이름으로만 컨텍스트를 읽으므로(계약 테스트) 같은 정체는 같은 판정이다.

    Args:
        brands·categories: canonical id (``EvidenceAdapter`` 정규화와 같은 표기).
    """
    card_list = list(cards)
    rule_list = list(rules)
    contract_map = RULE_CONTRACTS if contracts is None else contracts
    cards_by_id = {card.id: card for card in card_list}
    combinations = rule_combinations(card_list, brands, categories)

    evaluations: list[CardRuleEvaluation] = []
    seen: set[tuple[Any, ...]] = set()
    for brand, category in combinations:
        context, card_ids = build_rule_context(card_list, brand, category, contract_map)
        for rule in rule_list:
            evaluated = _evaluate_on_context(
                rule, (brand, category), context, card_ids, cards_by_id, contract_map
            )
            if evaluated.identity in seen:
                continue
            seen.add(evaluated.identity)
            evaluations.append(evaluated)
    return RuleRun(combinations=combinations, evaluations=evaluations)


__all__ = [
    "MAX_RULE_BRANDS",
    "MAX_RULE_CATEGORIES",
    "NON_FIRE_TOP_LIMIT",
    "REGISTRY_SOURCE",
    "RULE_CONTRACTS",
    "TARGET_BRAND",
    "CardRuleEvaluation",
    "ContractUnit",
    "EvidenceBinding",
    "InputSpec",
    "InputType",
    "NonFireKind",
    "NonFireReason",
    "Reduce",
    "Role",
    "RuleContract",
    "RuleEvaluation",
    "RuleRun",
    "build_rule_context",
    "count_non_fire_kinds",
    "evaluate_all",
    "evaluate_rule",
    "evaluate_rules_on_cards",
    "rule_combinations",
    "top_non_fire_reasons",
]
