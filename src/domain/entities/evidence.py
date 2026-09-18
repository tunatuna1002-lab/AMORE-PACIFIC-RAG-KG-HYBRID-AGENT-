"""
Evidence Cards
==============
검색·추론·도구 결과를 하나로 표현하는 증거 카드 (설계 E1, E2, E8).

KG 관계, DB(SQLite) 수치, 문서 청크, 규칙 추론 결과, ReAct 도구 관찰은 모두 이 모델로
바뀐 뒤에 프롬프트 조립·출처 표시·평가 트레이스·judge 컨텍스트로 간다.

id 규칙
-------
``{접두사}-{hex}`` — 접두사는 kind별 R/M/D/I/O, hex는 정체 문자열(identity key)의 sha1 앞
6자리다. 정체 문자열은 (kind, subject, predicate, object, value, unit, as_of, source)의
정규화 JSON이며, ``id_basis``가 주어지면(예: 문서 chunk_id) ``kind``와 그 값으로 만든다.
그래서 같은 사실은 실행·프로세스가 달라도 같은 id다(내장 ``hash()``는 프로세스마다 달라서
쓰지 않는다). text·detail·metadata·confidence·derived_from은 정체에 들어가지 않는다 —
같은 사실을 다른 문장으로 적어도 같은 카드다.

서로 다른 사실이 6자리에서 겹치면 ``EvidenceSet``이 뒤에 들어온 카드의 hex를 한 자리씩
늘린다(7, 8, … 40). 늘어난 id도 전체 sha1의 접두어이므로 검증을 통과한다. 어느 카드가
늘어나는지는 삽입 순서로 정해지므로, 같은 입력 순서면 결과도 같다.

값과 단위
---------
``value``는 정본 단위로 저장한다(예: SoS는 0~1 ``ratio``). 퍼센트 같은 표시 변환은
렌더러가 한다. 단위는 ``EvidenceUnit`` 상수만 허용한다 — 새 단위는 여기에 먼저 추가한다.

이 모듈은 도메인 계층이다: pydantic과 표준 라이브러리만 쓴다.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Iterable, Iterator
from datetime import date
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class EvidenceKind(str, Enum):
    """증거 카드 종류 (설계 E1)."""

    RELATION = "relation"  # KG 구조적 관계 (소유·카테고리·경쟁·제품)
    METRIC = "metric"  # 날짜 붙은 수치 (SQLite 정본)
    DOCUMENT = "document"  # 문서 청크 (정의·해석)
    INFERENCE = "inference"  # 규칙 추론 결과
    OBSERVATION = "observation"  # ReAct 도구 관찰


KIND_PREFIX: dict[EvidenceKind, str] = {
    EvidenceKind.RELATION: "R",
    EvidenceKind.METRIC: "M",
    EvidenceKind.DOCUMENT: "D",
    EvidenceKind.INFERENCE: "I",
    EvidenceKind.OBSERVATION: "O",
}

ID_HEX_LENGTH = 6
MAX_ID_HEX_LENGTH = 40  # sha1 hex 전체 길이
ID_BASIS_KEY = "id_basis"

_ID_PATTERN = re.compile(r"^([RMDIO])-([0-9a-f]{6,40})$")
_DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")


class EvidenceUnit:
    """정본 단위 상수. 표시 변환(%, $ 등)은 렌더러에서만 한다."""

    RATIO = "ratio"  # 0~1 비율 (SoS, churn_rate). 표시는 %
    INDEX_0_1 = "index_0_1"  # 0~1 지수 (HHI)
    INDEX_100 = "index_100"  # 100 = 카테고리 평균인 지수 (CPI)
    USD = "usd"  # 미국 달러
    RANK = "rank"  # 순위 (1 = 최상위). 평균 순위처럼 소수일 수 있다
    COUNT = "count"  # 개수 (제품 수, 리뷰 수)
    RATING_5 = "rating_5"  # 5점 만점 평점
    RATING_POINTS = "rating_points"  # 5점 척도 평점의 차이 (avg_rating_gap)
    BOOLEAN = "boolean"  # 참/거짓 (Top100 진입 여부)


KNOWN_UNITS: frozenset[str] = frozenset(
    value
    for name, value in vars(EvidenceUnit).items()
    if not name.startswith("_") and isinstance(value, str)
)

EvidenceValue = bool | int | float | str | None


def _normalize_value(value: EvidenceValue) -> EvidenceValue:
    """정체 문자열용 값 정규화: 2와 2.0은 같게, 부동소수 잡음은 12자리에서 자른다."""
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if isinstance(value, int):
        return value
    number = float(format(value, ".12g"))
    return int(number) if number.is_integer() else number


def identity_key_for(
    kind: EvidenceKind | str,
    subject: str,
    predicate: str,
    object: str | None,
    value: EvidenceValue,
    unit: str | None,
    as_of: str | None,
    source: str,
    id_basis: str | None = None,
) -> str:
    """카드 정체 문자열. id는 이 문자열의 sha1에서 나온다."""
    kind_value = EvidenceKind(kind).value
    if id_basis:
        parts: list[Any] = [kind_value, {ID_BASIS_KEY: id_basis}]
    else:
        parts = [
            kind_value,
            subject,
            predicate,
            object,
            _normalize_value(value),
            unit,
            as_of,
            source,
        ]
    return json.dumps(parts, ensure_ascii=False, separators=(",", ":"))


def evidence_digest(identity_key: str) -> str:
    """정체 문자열의 sha1 hex (40자리)."""
    return hashlib.sha1(identity_key.encode("utf-8")).hexdigest()


def make_evidence_id(
    kind: EvidenceKind | str, identity_key: str, length: int = ID_HEX_LENGTH
) -> str:
    """``{접두사}-{sha1 앞 length자리}``."""
    if not ID_HEX_LENGTH <= length <= MAX_ID_HEX_LENGTH:
        raise ValueError(f"id hex length must be {ID_HEX_LENGTH}..{MAX_ID_HEX_LENGTH}")
    return f"{KIND_PREFIX[EvidenceKind(kind)]}-{evidence_digest(identity_key)[:length]}"


class Evidence(BaseModel):
    """
    증거 카드.

    Attributes:
        id: ``{R|M|D|I|O}-{hex 6~40}``. 정체 문자열 sha1의 접두어여야 한다.
        kind: 카드 종류.
        subject: 주어 (정규화된 canonical id — 브랜드 ``laneige``, 카테고리 ``lip_care``,
            문서 doc_id, 도구 이름 등). 원표기는 ``metadata["display_name"]``.
        predicate: 술어 (``sos``, ``competesWith``, ``states`` 등).
        object: 목적어 또는 적용 범위 (예: 수치 카드의 카테고리). 없으면 None.
        value: 정본 단위의 값. 없으면 None.
        unit: ``EvidenceUnit`` 상수 또는 None.
        as_of: 데이터 시점 ``YYYY-MM-DD``. 시점이 없는 사실(KG 관계 등)은 None.
        source: 출처 (``sqlite:brand_metrics``, ``kg:competesWith``, ``rule:<이름>``,
            ``tool:<이름>``, ``rag:<doc_type>``).
        confidence: 0~1. 출처가 신뢰도를 주지 않으면 None (지어내지 않는다).
        derived_from: 추론 카드가 근거로 삼은 카드 id 목록.
        text: 프롬프트·출처 표시용 한 줄.
        detail: 긴 내용 (문서 본문, 도구 관찰 원문).
        metadata: 부가 정보 (chunk_id, score, category, display_name 등).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    kind: EvidenceKind
    subject: str = Field(min_length=1)
    predicate: str = Field(min_length=1)
    object: str | None = None
    value: EvidenceValue = None
    unit: str | None = None
    as_of: str | None = None
    source: str = Field(min_length=1)
    confidence: float | None = None
    derived_from: tuple[str, ...] = ()
    text: str = Field(min_length=1)
    detail: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("value", mode="before")
    @classmethod
    def _check_value(cls, value: Any) -> Any:
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError("value must be finite")
        return value

    @field_validator("unit")
    @classmethod
    def _check_unit(cls, unit: str | None) -> str | None:
        if unit is not None and unit not in KNOWN_UNITS:
            raise ValueError(f"unknown unit {unit!r}; add it to EvidenceUnit first")
        return unit

    @field_validator("as_of")
    @classmethod
    def _check_as_of(cls, as_of: str | None) -> str | None:
        if as_of is None:
            return None
        if not _DATE_PATTERN.match(as_of):
            raise ValueError("as_of must be YYYY-MM-DD")
        date.fromisoformat(as_of)  # 2026-13-01 같은 날짜는 여기서 ValueError
        return as_of

    @field_validator("confidence")
    @classmethod
    def _check_confidence(cls, confidence: float | None) -> float | None:
        if confidence is not None and not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be within 0..1")
        return confidence

    @model_validator(mode="after")
    def _check_id(self) -> Evidence:
        match = _ID_PATTERN.match(self.id)
        if not match:
            raise ValueError(f"invalid evidence id {self.id!r}")
        prefix, hex_part = match.groups()
        if prefix != KIND_PREFIX[self.kind]:
            raise ValueError(f"id prefix {prefix!r} does not match kind {self.kind.value!r}")
        if not evidence_digest(self.identity_key()).startswith(hex_part):
            raise ValueError(f"id {self.id!r} does not match card identity")
        return self

    def identity_key(self) -> str:
        """이 카드의 정체 문자열 (id 계산·충돌 판정 기준)."""
        return identity_key_for(
            self.kind,
            self.subject,
            self.predicate,
            self.object,
            self.value,
            self.unit,
            self.as_of,
            self.source,
            id_basis=self.metadata.get(ID_BASIS_KEY),
        )

    @classmethod
    def create(
        cls,
        *,
        kind: EvidenceKind | str,
        subject: str,
        predicate: str,
        source: str,
        text: str,
        object: str | None = None,
        value: EvidenceValue = None,
        unit: str | None = None,
        as_of: str | None = None,
        confidence: float | None = None,
        derived_from: Iterable[str] = (),
        detail: str | None = None,
        metadata: dict[str, Any] | None = None,
        id_basis: str | None = None,
    ) -> Evidence:
        """필드에서 결정적 id를 계산해 카드를 만든다."""
        meta = dict(metadata or {})
        if id_basis:
            meta[ID_BASIS_KEY] = id_basis
        identity = identity_key_for(
            kind, subject, predicate, object, value, unit, as_of, source, id_basis=id_basis
        )
        return cls(
            id=make_evidence_id(kind, identity),
            kind=EvidenceKind(kind),
            subject=subject,
            predicate=predicate,
            object=object,
            value=value,
            unit=unit,
            as_of=as_of,
            source=source,
            confidence=confidence,
            derived_from=tuple(derived_from),
            text=text,
            detail=detail,
            metadata=meta,
        )

    @property
    def display_subject(self) -> str:
        """표시용 주어: 원표기가 있으면 그것, 없으면 canonical id."""
        name = self.metadata.get("display_name")
        return str(name) if name else self.subject


class EvidenceSet:
    """
    순서를 보존하고 같은 사실을 한 번만 담는 카드 모음.

    - 같은 정체의 카드가 다시 오면 먼저 들어온 카드를 유지한다(first wins).
    - 다른 정체가 같은 id를 가지면 뒤에 온 카드의 id를 늘려 담는다.
    """

    def __init__(self, cards: Iterable[Evidence] = ()) -> None:
        self._cards: list[Evidence] = []
        self._by_id: dict[str, Evidence] = {}
        self._by_identity: dict[str, Evidence] = {}
        self.extend(cards)

    def add(self, card: Evidence) -> Evidence:
        """카드를 담고, 실제로 담긴 카드(기존 카드 또는 id가 늘어난 카드)를 돌려준다."""
        identity = card.identity_key()
        existing = self._by_identity.get(identity)
        if existing is not None:
            return existing

        stored = card
        if card.id in self._by_id:
            digest = evidence_digest(identity)
            prefix = card.id.split("-", 1)[0]
            hex_length = len(card.id) - 2
            while True:
                hex_length += 1
                if hex_length > MAX_ID_HEX_LENGTH:  # sha1 전체가 같다 — 사실상 불가능
                    raise ValueError(f"cannot resolve evidence id collision for {card.id}")
                candidate = f"{prefix}-{digest[:hex_length]}"
                if candidate not in self._by_id:
                    stored = card.model_copy(update={"id": candidate})
                    break

        self._cards.append(stored)
        self._by_id[stored.id] = stored
        self._by_identity[identity] = stored
        return stored

    def extend(self, cards: Iterable[Evidence]) -> list[Evidence]:
        return [self.add(card) for card in cards]

    def by_id(self, evidence_id: str) -> Evidence | None:
        return self._by_id.get(evidence_id)

    def of_kind(self, *kinds: EvidenceKind) -> list[Evidence]:
        wanted = set(kinds)
        return [card for card in self._cards if card.kind in wanted]

    @property
    def ids(self) -> list[str]:
        return [card.id for card in self._cards]

    def to_list(self) -> list[Evidence]:
        return list(self._cards)

    def __iter__(self) -> Iterator[Evidence]:
        return iter(list(self._cards))

    def __len__(self) -> int:
        return len(self._cards)

    def __contains__(self, evidence_id: object) -> bool:
        return evidence_id in self._by_id
