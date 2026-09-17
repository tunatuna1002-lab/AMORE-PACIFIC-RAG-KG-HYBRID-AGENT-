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
    """

    kind: EvidenceKind
    predicate: str
    unit: str | None = None
    subject_role: str = Role.BRAND
    object_role: str | None = None
    reduce: Reduce = Reduce.VALUE
    extra_predicates: tuple[str, ...] = ()

    @property
    def predicates(self) -> tuple[str, ...]:
        return (self.predicate, *self.extra_predicates)

    def matches(self, card: Evidence) -> bool:
        """종류·술어·단위가 맞는 카드인가 (주어·목적어 역할 판정은 조립기 몫)."""
        return (
            card.kind is self.kind
            and card.predicate in self.predicates
            and (self.unit is None or card.unit == self.unit)
        )


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

_GAP_FACTS_NOT_QUERIED = (
    "운영 DB brand_metrics.{column}에 값이 있으나({filled}/13,838행 non-null, 2026-09-17) "
    "MetricFactsProvider가 조회하지 않고 어댑터 매핑도 없어 카드가 만들어지지 않는다"
)
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
)
ASIN = InputSpec(
    "asin",
    InputType.STRING,
    None,
    derivation="대표 제품 ASIN (relation hasProduct의 object) — 어느 제품인지는 조립기가 정한다",
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
    gap=_GAP_FACTS_NOT_QUERIED.format(column="cpi", filled="7,286"),
)
RATING_GAP = InputSpec(
    "rating_gap",
    InputType.NUMBER,
    _U.RATING_POINTS,
    -4.0,
    4.0,
    binding=EvidenceBinding(_M, "avg_rating_gap", _U.RATING_POINTS, Role.BRAND, Role.CATEGORY),
    gap=_GAP_FACTS_NOT_QUERIED.format(column="avg_rating_gap", filled="13,735"),
    note="규칙 키 rating_gap = 카드 술어 avg_rating_gap",
)
AVG_RANK = InputSpec(
    "avg_rank",
    InputType.NUMBER,
    _U.RANK,
    1.0,
    100.0,
    binding=EvidenceBinding(_M, "brand_avg_rank", _U.RANK, Role.BRAND, Role.CATEGORY),
    gap=_GAP_FACTS_NOT_QUERIED.format(column="brand_avg_rank", filled="13,838"),
    note="규칙 키 avg_rank = 카드 술어 brand_avg_rank",
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
    binding=EvidenceBinding(_R, "ownedBy", None, Role.BRAND, Role.ENTITY, Reduce.OBJECT),
    note=(
        "카드 object는 소문자 canonical('amorepacific')인데 규칙은 'AMOREPACIFIC'과 "
        "대소문자 구분 비교 — 카드 metadata.object_display_name(원표기)을 써야 발화한다"
    ),
)
_GAP_BRAND_PROFILE = "표시용 — 전용 카드 변환 규칙 없음(어댑터 술어 역할 미등록, 미검증)"
COUNTRY_OF_ORIGIN = InputSpec("country_of_origin", InputType.STRING, None, gap=_GAP_BRAND_PROFILE)
ACQUIRED = InputSpec("acquired", InputType.ANY, None, gap=_GAP_BRAND_PROFILE)
SEGMENT = InputSpec("segment", InputType.STRING, None, gap=_GAP_BRAND_PROFILE)
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

__all__ = [
    "RULE_CONTRACTS",
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
    "count_non_fire_kinds",
    "evaluate_all",
    "evaluate_rule",
    "top_non_fire_reasons",
]
