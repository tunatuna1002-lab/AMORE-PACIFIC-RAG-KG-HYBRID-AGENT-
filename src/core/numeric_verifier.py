"""
Numeric Verifier
================
답변 속 수치가 인용한 증거 카드의 값과 맞는지 규칙으로 검사한다 (설계 E8 뒷부분).

LLM·IO 없는 순수 함수다. ``ResponsePipeline``이 생성 직후 그 답변 프롬프트에 실린 카드
(``context.prompt_evidence``)로 호출한다.

절차
----
1. 문장 분리: 줄바꿈과 ``. ! ?`` 뒤 공백에서 자른다. 단, 공백 뒤가 인용 괄호면 자르지
   않는다("…입니다. [M-xxxxxx]"의 인용은 앞 문장 것이다).
2. 인용 추출: ``[M-xxxxxx]``, 연속 괄호 ``[M-a][D-b]``/``[M-a] [D-b]``, 괄호 안 목록
   ``[ M-a, D-b ]``. id 접두사 R/M/D/I/O, hex 6~40자리. 공백만 사이에 둔 괄호들은 한
   묶음이다.
3. 수치 추출: ``NUMBER_PATTERN``으로 퍼센트·달러·순위·개수·평점·일반 수를 찾되, 먼저
   ``EXCLUSION_PATTERNS``(날짜·연도·분기·Top N·기간·목록 번호·카드 id 등)를 가린다.
   한국어·영어 표기를 모두 다룬다(``18.0%``/``18 percent``, ``9위``/``9th``/``#9``,
   ``$21.60``/``21.60달러``/``USD 21.60``, ``1.2만``/``1.2K``, ``Aug 31``·``8/31`` 날짜 제외).
4. 수치 ↔ 인용: 같은 문장에서 그 수치 **뒤에 오는 첫 인용 묶음**이 그 수치의 인용이다.
   뒤에 인용이 없으면 인용 없음 — 앞 절의 인용을 끌어오지 않는다("SoS 2%[M-a]이고 리뷰는
   12,345건"의 12,345는 [M-a] 근거가 아니며, 끌어오면 enforce가 맞는 수를 지운다).
5. 매칭: 인용 카드(+ 인용 카드의 ``derived_from`` 중 프롬프트에 있는 카드)에서 후보 값을
   만들고 종류·허용 오차로 비교한다.
   - 수치(M) 카드: 정본 값을 렌더러 ``format_value``와 같은 표시 규칙으로 바꾼 값(비율 →
     ×100 %, 그 밖은 그대로)과 ``format_value`` 출력 문자열의 수 둘 다 후보다. 그래서
     ``18.0%``는 비율 0.18과 맞고, ``0.18%``는 맞지 않는다.
   - 문서(D)·관계(R)·추론(I)·관찰(O) 카드: ``text``와 ``detail``에서 같은 추출기로 뽑은 수.
6. 분류 우선순위: 일치 → ``verified``; 인용이 없으면 ``no_citation``; 인용 id 중 프롬프트에
   없던 것이 있으면 ``unknown_card``; 나머지 ``mismatch``.

알려진 한계: 모델이 카드 값으로 계산한 수(차이 ``0.5%p``, ``2배``, ``3계단 상승``)는 카드에
그 값이 없으므로 ``mismatch``가 된다. enforce를 기본값으로 올리기 전에 annotate 결과로 그
비율을 확인해야 한다.

허용 오차 (``ROUNDING_TOLERANCE``)
----------------------------------
답에 적힌 자릿수 기준 반올림 일치: ``|답 − 카드| ≤ 0.5 × 10^(−소수 자릿수) × 배수``
(배수는 ``만``·``K`` 같은 단위의 곱, 없으면 1).
근거: 모델은 카드 표시값을 더 적은 자릿수로 반올림해 옮길 수 있고(13.54% → 13.5%·14%),
그 이상 벌어지면 반올림으로 설명되지 않는 다른 수다. 상대 오차(예: 1%)는 쓰지 않는다 —
$21.60과 $21.80처럼 표시 자릿수에서 명백히 다른 값을 통과시키기 때문이다.
부호 없는 답 수치는 절댓값끼리 비교한다("평점 격차 0.12" ↔ 카드 −0.12).

메타데이터 스키마 (``response.metadata["numeric_verification"]``)
------------------------------------------------------------------
::

    {
      "mode": "annotate" | "enforce",
      "skipped": None | "no_evidence" | "error",
      "checked": int,        # 검사한 수치 수 (제외 규칙에 걸린 수는 세지 않는다)
      "verified": int,
      "mismatch": int,
      "no_citation": int,
      "unknown_card": int,
      "found_in_other_cards": int,  # 미검증 수치 중 인용하지 않은 프롬프트 카드에는 있던 수
      "replaced": int,       # enforce에서 치환한 수치 수 (annotate는 항상 0)
      "details": [           # 최대 MAX_DETAILS건, 미검증(mismatch→unknown_card→no_citation)
        {                    # 우선, 같은 상태 안에서는 답변 속 순서
          "status": "verified" | "no_citation" | "mismatch" | "unknown_card",
          "number": str,     # 답변 원문 스팬 (부호·$·단위 포함)
          "value": float, "kind": str,
          "sentence": str,   # 최대 MAX_SENTENCE_CHARS자
          "cited_ids": [str], "unknown_ids": [str],
          "matched_ids": [str], "found_in_ids": [str],
        }
      ],
    }

항상 ``checked == verified + mismatch + no_citation + unknown_card``이다. ``skipped``가
None이 아니면 개수는 모두 0이고 ``details``는 빈 목록이다. 플래그가 ``off``면 키 자체를
남기지 않는다.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict

from src.domain.entities.evidence import Evidence, EvidenceKind, EvidenceUnit
from src.rag.evidence_renderer import format_value

# ---------------------------------------------------------------------------
# 상수
# ---------------------------------------------------------------------------

MODE_OFF = "off"
MODE_ANNOTATE = "annotate"
MODE_ENFORCE = "enforce"
MODES: tuple[str, ...] = (MODE_OFF, MODE_ANNOTATE, MODE_ENFORCE)

UNVERIFIED_PLACEHOLDER = "확인되지 않음"
MAX_DETAILS = 10
MAX_SENTENCE_CHARS = 200

# 답에 적힌 마지막 자리 한 단위의 절반 — 표시 반올림으로 설명되는 최대 차이
ROUNDING_TOLERANCE = 0.5
_FLOAT_EPS = 1e-9

_ID = r"[RMDIO]-[0-9a-fA-F]{6,40}(?![0-9a-fA-F])"
_CITATION_GROUP = re.compile(rf"\[\s*({_ID}(?:[\s,，;/·]+{_ID})*)\s*[,，]?\s*\]")
_ID_IN_GROUP = re.compile(_ID)

# 문장 경계: 줄바꿈, 또는 . ! ? 뒤 공백 (뒤가 인용 괄호면 경계가 아니다)
_SENTENCE_BREAK = re.compile(r"\n+|(?<=[.!?。])[ \t]+(?!\[\s*[RMDIO]-)")

# 수치 토큰의 왼쪽 경계 — 다른 수의 중간(0.0681의 "81")에서 제외 패턴이 시작되지 않게 한다
_B = r"(?<![\d.,])"
_N = _B + r"\d+(?:\.\d+)?"
_MONTH = (
    r"(?i:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|june?|july?|aug(?:ust)?"
    r"|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
)
# 영문 날짜의 일(day) — 뒤에 %·소수부가 오면 날짜가 아니라 수치다 ("May 2%"는 드물지만 지킨다)
_EN_DAY = _B + r"\d{1,2}(?:st|nd|rd|th)?(?!\d|[.,]\d|\s?%|\s?(?i:percent))"

# 수치로 보지 않는 패턴 — 수치 추출 전에 가린다 (순서 무관, 겹쳐도 된다)
EXCLUSION_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("citation", _CITATION_GROUP),
    ("card_id", re.compile(rf"(?<![A-Za-z0-9]){_ID}")),
    ("iso_date", re.compile(_B + r"\d{4}[-./]\d{1,2}[-./]\d{1,2}")),
    (
        "korean_date",
        re.compile(_B + r"\d{4}\s?년(?:도|대)?(?:\s?\d{1,2}\s?월)?(?:\s?\d{1,2}\s?일)?"),
    ),
    ("month_day", re.compile(_B + r"\d{1,2}월(?:\s?\d{1,2}일)?|" + _B + r"\d{1,2}일")),
    ("english_date", re.compile(rf"\b{_MONTH}\.?\s+{_EN_DAY}|{_EN_DAY}\s+{_MONTH}\b")),
    # 월/일 (8/31). 둘째 수가 5 이하면 평점(4/5)일 수 있어 가리지 않는다
    ("slash_date", re.compile(_B + r"(?:1[0-2]|0?[1-9])/(?:3[01]|[12]\d|0?[6-9])(?![\d.])")),
    ("time", re.compile(_B + r"\d{1,2}:\d{2}")),
    (
        "bare_year",  # 단위 없는 1900~2099 네 자리 수
        re.compile(
            r"(?<![\d.,$#])(?:19|20)\d{2}"
            r"(?!\d|[.,]\d|\s?(?:%|퍼센트|위|개|건|명|종|달러|점|/|(?i:percent|dollars?)))"
        ),
    ),
    (
        "quarter",
        re.compile(
            r"(?i:\bQ[1-4]\b)|" + _B + r"[1-4]\s?분기|" + _B + r"[12]\s?반기"
            r"|(?i:\b[1-4](?:st|nd|rd|th)\s+quarter\b)"
        ),
    ),
    (
        "top_n",
        re.compile(
            r"(?i:top|bottom)[\s-]?\d+|(?:상위|하위)\s?\d+(?:\.\d+)?(?:\s?(?:위|개|%))?"
            r"|(?i:out\s+of)\s+(?!5(?![\d.]))\d+"
        ),
    ),
    ("rank_threshold", re.compile(_N + r"\s?위\s?(?:권|안|이내|내)")),
    # "주"는 붙여 쓰고 뒤에 한글이 없을 때만 ("4주," "4주 동안") — "0.15 주의"는 수치다
    ("duration", re.compile(_N + r"(?:\s?(?:개월|주간|주차|시간|일간|년간)|주(?![가-힣]))")),
    (
        "english_duration",
        re.compile(
            _N + r"(?:\s?-\s?|\s)"
            r"(?i:days?|weeks?|months?|years?|yrs?|hours?|hrs?|minutes?|mins?)(?![A-Za-z])"
        ),
    ),
    ("scale", re.compile(_N + r"\s?점\s?(?:만점|척도)|" + _N + r"[\s-]?(?i:point\s+scale)")),
    ("ordinal", re.compile(_N + r"\s?(?:단계|번째|가지|차례|순위)")),
    ("list_marker", re.compile(r"(?m)^[ \t]*(?:#{1,6}[ \t]*)?(?:\d+[.)]|\(\d+\))(?=\s)")),
    ("heading_number", re.compile(r"(?m)^[ \t]*#{1,6}[ \t]*\d+(?=\s)")),
)

# 퍼센트·달러·순위·개수·평점·일반 수. 앞이 ASCII 영숫자·소수점이거나 "영문자-"면(식별자)
# 수치가 아니고, 뒤에 영문자가 바로 붙거나(20g, 3x) 소수부가 이어지면(1.5x의 "1") 수치가 아니다.
NUMBER_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_.,])(?<![A-Za-z]-)"
    r"(?P<sign>[+\-−](?=[$\d]))?"
    r"(?P<prefix>\$\s?|US\$\s?|USD\s?|#\s?)?"
    r"(?P<num>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)"
    r"(?P<scale>\s?(?:천|만(?!큼|점)|억)|[KMB](?![A-Za-z])"
    r"|\s(?i:thousand|million|billion)(?![A-Za-z]))?"
    r"(?P<suffix>\s?(?:%p|%포인트|%|퍼센트)"
    r"|\s?(?i:percent(?:age\s+points?)?|pp)(?![A-Za-z])"
    r"|\s?(?:/|(?i:out\s+of))\s?5(?![\d.])"
    r"|\s?위|(?:st|nd|rd|th)(?![A-Za-z])"
    r"|\s?(?:개|건|명|종)"
    r"|\s?(?:달러|(?i:dollars?|usd))(?![A-Za-z])"
    r"|\s?점(?!유))?"
    r"(?![A-Za-z0-9]|[.,]\d)"
)

_SCALES: dict[str, float] = {
    "천": 1e3,
    "만": 1e4,
    "억": 1e8,
    "k": 1e3,
    "m": 1e6,
    "b": 1e9,
    "thousand": 1e3,
    "million": 1e6,
    "billion": 1e9,
}

KIND_PERCENT = "percent"
KIND_USD = "usd"
KIND_RANK = "rank"
KIND_COUNT = "count"
KIND_RATING = "rating"
KIND_NUMBER = "number"  # 단위 없는 수 — 어느 종류와도 비교한다

# 수치 카드 단위 → 비교 가능한 답 수치 종류 (KIND_NUMBER는 항상 허용)
_UNIT_KINDS: dict[str, frozenset[str]] = {
    EvidenceUnit.RATIO: frozenset({KIND_PERCENT}),
    EvidenceUnit.INDEX_0_1: frozenset(),
    EvidenceUnit.INDEX_100: frozenset(),
    EvidenceUnit.USD: frozenset({KIND_USD}),
    EvidenceUnit.RANK: frozenset({KIND_RANK}),
    EvidenceUnit.COUNT: frozenset({KIND_COUNT}),
    EvidenceUnit.RATING_5: frozenset({KIND_RATING}),
    EvidenceUnit.RATING_POINTS: frozenset({KIND_RATING}),
}


class NumericStatus(str, Enum):
    VERIFIED = "verified"
    NO_CITATION = "no_citation"
    MISMATCH = "mismatch"
    UNKNOWN_CARD = "unknown_card"


_REPLACED_STATUSES = frozenset({NumericStatus.MISMATCH, NumericStatus.UNKNOWN_CARD})
_DETAIL_ORDER = {
    NumericStatus.MISMATCH: 0,
    NumericStatus.UNKNOWN_CARD: 1,
    NumericStatus.NO_CITATION: 2,
    NumericStatus.VERIFIED: 3,
}


class ExtractedNumber(BaseModel):
    """텍스트에서 뽑은 수치 한 개. start·end는 입력 문자열 기준 위치."""

    model_config = ConfigDict(frozen=True)

    text: str
    value: float
    kind: str
    decimals: int
    signed: bool
    start: int
    end: int
    scale: float = 1.0  # 만·K 같은 배수 (value에 이미 곱했다)


class NumericClaim(BaseModel):
    """답변 속 수치 한 개의 검사 결과."""

    model_config = ConfigDict(frozen=True)

    text: str
    value: float
    kind: str
    start: int
    end: int
    sentence: str
    cited_ids: tuple[str, ...]
    status: NumericStatus
    matched_ids: tuple[str, ...] = ()
    unknown_ids: tuple[str, ...] = ()
    found_in_ids: tuple[str, ...] = ()  # 미검증일 때, 인용하지 않은 프롬프트 카드 중 일치


class NumericVerification(BaseModel):
    model_config = ConfigDict(frozen=True)

    claims: tuple[NumericClaim, ...] = ()

    def count(self, status: NumericStatus) -> int:
        return sum(1 for claim in self.claims if claim.status == status)


# ---------------------------------------------------------------------------
# 추출
# ---------------------------------------------------------------------------


def _normalize_id(raw: str) -> str:
    prefix, hex_part = raw.split("-", 1)
    return f"{prefix.upper()}-{hex_part.lower()}"


def extract_citation_ids(text: str) -> list[str]:
    """텍스트 속 인용 id를 나온 순서대로 (중복 포함) 돌려준다."""
    ids: list[str] = []
    for group in _CITATION_GROUP.finditer(text):
        ids.extend(_normalize_id(m.group(0)) for m in _ID_IN_GROUP.finditer(group.group(1)))
    return ids


def _mask_exclusions(text: str) -> str:
    chars = list(text)
    for _, pattern in EXCLUSION_PATTERNS:
        for match in pattern.finditer(text):
            for i in range(match.start(), match.end()):
                if chars[i] != "\n":
                    chars[i] = " "
    return "".join(chars)


def _kind(prefix: str | None, suffix: str | None) -> str:
    prefix = (prefix or "").strip().upper()
    suffix = (suffix or "").strip().lower()
    if prefix in ("$", "US$", "USD") or suffix in ("달러", "dollar", "dollars", "usd"):
        return KIND_USD
    if prefix == "#" or suffix in ("위", "st", "nd", "rd", "th"):
        return KIND_RANK
    if suffix.startswith(("%", "percent")) or suffix in ("퍼센트", "pp"):
        return KIND_PERCENT
    if suffix.startswith(("/", "out")) or suffix == "점":
        return KIND_RATING
    if suffix in ("개", "건", "명", "종"):
        return KIND_COUNT
    return KIND_NUMBER


def extract_numbers(text: str) -> list[ExtractedNumber]:
    """제외 규칙을 적용한 뒤 수치를 뽑는다. ``text``는 원문 스팬(단위 포함)이다."""
    masked = _mask_exclusions(text)
    numbers: list[ExtractedNumber] = []
    for match in NUMBER_PATTERN.finditer(masked):
        raw = match.group("num")
        scale = _SCALES[match.group("scale").strip().lower()] if match.group("scale") else 1.0
        value = float(raw.replace(",", "")) * scale
        if match.group("sign") in ("-", "−"):
            value = -value
        decimals = len(raw.split(".", 1)[1]) if "." in raw else 0
        start, end = match.start(), match.end()
        numbers.append(
            ExtractedNumber(
                text=text[start:end],
                value=value,
                kind=_kind(match.group("prefix"), match.group("suffix")),
                decimals=decimals,
                signed=match.group("sign") is not None,
                start=start,
                end=end,
                scale=scale,
            )
        )
    return numbers


# ---------------------------------------------------------------------------
# 매칭
# ---------------------------------------------------------------------------


def _close(claim: ExtractedNumber, candidate: float) -> bool:
    tolerance = ROUNDING_TOLERANCE * 10 ** (-claim.decimals) * claim.scale
    a, b = claim.value, candidate
    if not claim.signed:
        a, b = abs(a), abs(b)
    return abs(a - b) <= tolerance + _FLOAT_EPS * max(1.0, abs(b))


def _metric_candidates(card: Evidence) -> list[tuple[frozenset[str] | None, float]]:
    """수치 카드의 (허용 종류, 값) 후보. 허용 종류 None은 모든 종류와 비교한다는 뜻이다."""
    value = card.value
    if value is None or isinstance(value, bool):
        return []
    if isinstance(value, str) or card.unit is None:
        return [(None, n.value) for n in extract_numbers(str(value))]

    kinds = _UNIT_KINDS.get(card.unit, frozenset())
    number = float(value)
    shown = number * 100 if card.unit == EvidenceUnit.RATIO else number
    candidates = [(kinds, shown)]
    candidates.extend((kinds, n.value) for n in extract_numbers(format_value(value, card.unit)))
    if card.unit == EvidenceUnit.RATIO:
        candidates.append((frozenset(), number))  # "SoS 0.02"처럼 정본 값을 그대로 쓴 경우
    return candidates


def _card_matches(claim: ExtractedNumber, card: Evidence) -> bool:
    if card.kind == EvidenceKind.METRIC:
        for kinds, candidate in _metric_candidates(card):
            compatible = claim.kind == KIND_NUMBER or kinds is None or claim.kind in kinds
            if compatible and _close(claim, candidate):
                return True
        return False

    body = "\n".join(part for part in (card.text, card.detail) if part)
    for found in extract_numbers(body):
        compatible = KIND_NUMBER in (claim.kind, found.kind) or claim.kind == found.kind
        if compatible and _close(claim, found.value):
            return True
    return False


def _expand_derived(cited: Sequence[Evidence], by_id: dict[str, Evidence]) -> list[Evidence]:
    """인용 카드 + 그 ``derived_from`` 중 프롬프트에 있는 카드 (추론 카드의 근거 수치)."""
    expanded: dict[str, Evidence] = {}
    stack = list(cited)
    while stack:
        card = stack.pop(0)
        if card.id in expanded:
            continue
        expanded[card.id] = card
        stack.extend(by_id[i] for i in card.derived_from if i in by_id)
    return list(expanded.values())


# ---------------------------------------------------------------------------
# 검증
# ---------------------------------------------------------------------------


def _sentence_spans(answer: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = 0
    for match in _SENTENCE_BREAK.finditer(answer):
        if match.start() > start:
            spans.append((start, match.start()))
        start = match.end()
    if start < len(answer):
        spans.append((start, len(answer)))
    return spans


def _citation_clusters(sentence: str) -> list[tuple[int, int, tuple[str, ...]]]:
    """공백만 사이에 둔 인용 괄호를 한 묶음으로: (시작, 끝, ids)."""
    clusters: list[tuple[int, int, list[str]]] = []
    for group in _CITATION_GROUP.finditer(sentence):
        ids = [_normalize_id(m.group(0)) for m in _ID_IN_GROUP.finditer(group.group(1))]
        if clusters and not sentence[clusters[-1][1] : group.start()].strip():
            clusters[-1] = (clusters[-1][0], group.end(), clusters[-1][2] + ids)
        else:
            clusters.append((group.start(), group.end(), ids))
    return [(s, e, tuple(dict.fromkeys(ids))) for s, e, ids in clusters]


def _cited_for(
    number: ExtractedNumber, clusters: list[tuple[int, int, tuple[str, ...]]]
) -> tuple[str, ...]:
    following = [c for c in clusters if c[0] >= number.end]
    return following[0][2] if following else ()


def verify_numeric_claims(answer: str, cards: Iterable[Evidence]) -> NumericVerification:
    """답변의 수치를 인용 카드와 대조한다. 카드는 그 답변 프롬프트에 실린 카드여야 한다."""
    card_list = list(cards)
    by_id: dict[str, Evidence] = {}
    for card in card_list:
        by_id.setdefault(card.id, card)

    claims: list[NumericClaim] = []
    for s_start, s_end in _sentence_spans(answer):
        sentence = answer[s_start:s_end]
        numbers = extract_numbers(sentence)
        if not numbers:
            continue
        clusters = _citation_clusters(sentence)
        for number in numbers:
            cited = _cited_for(number, clusters)
            known = [by_id[i] for i in cited if i in by_id]
            unknown = tuple(i for i in cited if i not in by_id)
            matched = tuple(
                card.id for card in _expand_derived(known, by_id) if _card_matches(number, card)
            )
            if matched:
                status = NumericStatus.VERIFIED
            elif not cited:
                status = NumericStatus.NO_CITATION
            elif unknown:
                status = NumericStatus.UNKNOWN_CARD
            else:
                status = NumericStatus.MISMATCH
            found_in: tuple[str, ...] = ()
            if status != NumericStatus.VERIFIED:
                found_in = tuple(
                    card.id
                    for card in by_id.values()
                    if card.id not in cited and _card_matches(number, card)
                )
            claims.append(
                NumericClaim(
                    text=number.text,
                    value=number.value,
                    kind=number.kind,
                    start=s_start + number.start,
                    end=s_start + number.end,
                    sentence=sentence.strip(),
                    cited_ids=cited,
                    status=status,
                    matched_ids=matched,
                    unknown_ids=unknown,
                    found_in_ids=found_in,
                )
            )
    return NumericVerification(claims=tuple(claims))


# ---------------------------------------------------------------------------
# 적용
# ---------------------------------------------------------------------------


def _empty_summary(mode: str, skipped: str | None) -> dict[str, Any]:
    return {
        "mode": mode,
        "skipped": skipped,
        "checked": 0,
        "verified": 0,
        "mismatch": 0,
        "no_citation": 0,
        "unknown_card": 0,
        "found_in_other_cards": 0,
        "replaced": 0,
        "details": [],
    }


def skipped_summary(mode: str, reason: str) -> dict[str, Any]:
    """검증을 건너뛴 경우의 메타데이터 (스키마는 모듈 docstring)."""
    return _empty_summary(mode, reason)


def summarize(verification: NumericVerification, mode: str, replaced: int = 0) -> dict[str, Any]:
    """검증 결과 → 메타데이터 dict (스키마는 모듈 docstring)."""
    summary = _empty_summary(mode, None)
    claims = verification.claims
    summary.update(
        checked=len(claims),
        verified=verification.count(NumericStatus.VERIFIED),
        mismatch=verification.count(NumericStatus.MISMATCH),
        no_citation=verification.count(NumericStatus.NO_CITATION),
        unknown_card=verification.count(NumericStatus.UNKNOWN_CARD),
        found_in_other_cards=sum(
            1 for c in claims if c.status != NumericStatus.VERIFIED and c.found_in_ids
        ),
        replaced=replaced,
    )
    ordered = sorted(enumerate(claims), key=lambda item: (_DETAIL_ORDER[item[1].status], item[0]))
    summary["details"] = [
        {
            "status": claim.status.value,
            "number": claim.text,
            "value": claim.value,
            "kind": claim.kind,
            "sentence": claim.sentence[:MAX_SENTENCE_CHARS],
            "cited_ids": list(claim.cited_ids),
            "unknown_ids": list(claim.unknown_ids),
            "matched_ids": list(claim.matched_ids),
            "found_in_ids": list(claim.found_in_ids),
        }
        for _, claim in ordered[:MAX_DETAILS]
    ]
    return summary


def enforce(answer: str, verification: NumericVerification) -> tuple[str, int]:
    """``mismatch``·``unknown_card`` 수치를 자리표시어(``UNVERIFIED_PLACEHOLDER``)로 바꾼다.

    수치 스팬(부호·$·#·단위 포함)만 바꾸므로 조사·인용 괄호는 그대로다
    ("SoS 5%로" → "SoS 확인되지 않음로"). 인용 누락(no_citation)은 수치 오류의 증거가
    아니므로 바꾸지 않는다.
    """
    targets = [c for c in verification.claims if c.status in _REPLACED_STATUSES]
    text = answer
    for claim in sorted(targets, key=lambda c: c.start, reverse=True):
        text = text[: claim.start] + UNVERIFIED_PLACEHOLDER + text[claim.end :]
    return text, len(targets)


def apply_numeric_verification(
    answer: str, cards: Sequence[Evidence], mode: str
) -> tuple[str, dict[str, Any] | None]:
    """모드에 따라 검증하고 (답변, 메타데이터)를 돌려준다. ``off``면 (원문, None)."""
    if mode not in MODES:
        raise ValueError(f"unknown numeric verification mode {mode!r}; expected {MODES}")
    if mode == MODE_OFF:
        return answer, None
    if not cards:
        return answer, skipped_summary(mode, "no_evidence")

    verification = verify_numeric_claims(answer, cards)
    if mode == MODE_ENFORCE:
        text, replaced = enforce(answer, verification)
        return text, summarize(verification, mode, replaced)
    return answer, summarize(verification, mode)
