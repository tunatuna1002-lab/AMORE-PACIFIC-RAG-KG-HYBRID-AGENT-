"""
Evidence Renderer
=================
증거 카드를 답변 프롬프트와 judge 컨텍스트용 문자열로 만든다 (설계 E1, E8).

- 표시 변환은 여기서만 한다: 카드의 비율(0~1)은 %로, 달러는 $로.
- 섹션 순서는 ``SECTION_ORDER``로 고정하고, 같은 종류 안에서는 입력 순서를 지킨다.
- 같은 id는 한 번만 싣는다(먼저 온 카드).
- 인용 지시 문구는 조립기(트랙 2-B)가 붙이도록 상수 ``CITATION_INSTRUCTION``만 둔다.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping

from src.domain.entities.evidence import Evidence, EvidenceKind, EvidenceUnit, EvidenceValue

CITATION_INSTRUCTION = (
    "각 문장 끝에 근거 카드 id를 [M-xxxxxx]처럼 인용하라. 수치(M)뿐 아니라 관계(R)·추론(I)·"
    "문서(D)·관찰(O) 카드도 같은 형식으로 인용한다. 카드에 없는 수치는 쓰지 말라."
)

SECTION_ORDER: tuple[EvidenceKind, ...] = (
    EvidenceKind.METRIC,
    EvidenceKind.RELATION,
    EvidenceKind.INFERENCE,
    EvidenceKind.DOCUMENT,
    EvidenceKind.OBSERVATION,
)

SECTION_TITLES: dict[EvidenceKind, str] = {
    EvidenceKind.METRIC: "[DB 수치]",
    EvidenceKind.RELATION: "[관계]",
    EvidenceKind.INFERENCE: "[규칙 추론]",
    EvidenceKind.DOCUMENT: "[문서]",
    EvidenceKind.OBSERVATION: "[도구 관찰]",
}

METRIC_LABELS: dict[str, str] = {
    "sos": "SoS",
    "sos_rank": "SoS 브랜드 순위",
    "product_count": "Top100 제품 수",
    "present_in_top100": "Top100 진입",
    "hhi": "HHI",
    "churn_rate": "이탈률",
    "avg_price": "Top100 평균가",
    "avg_rating": "Top100 평균 평점",
    "bsr_rank": "BSR 순위",
    "price": "가격",
    "rating": "평점",
    "reviews_count": "리뷰 수",
    "cpi": "CPI",
    "avg_rating_gap": "평점 격차",
    "brand_avg_rank": "평균 순위",
}

DEFAULT_MAX_DETAIL_CHARS = 1500
_DETAIL_INDENT = "  "


def _trim(number: float, decimals: int) -> str:
    """소수 decimals자리로 반올림하고 끝의 0을 지운다 (2.00 → 2, 13.50 → 13.5)."""
    text = f"{number:.{decimals}f}"
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return "0" if text in ("-0", "") else text


def format_value(value: EvidenceValue, unit: str | None) -> str:
    """정본 단위의 값을 표시 문자열로 바꾼다."""
    if value is None:
        return ""
    if isinstance(value, bool):
        if unit == EvidenceUnit.BOOLEAN:
            return "있음" if value else "없음"
        return str(value)
    if isinstance(value, str) or unit is None:
        return str(value)

    number = float(value)
    if unit == EvidenceUnit.RATIO:
        return f"{_trim(number * 100, 2)}%"
    if unit == EvidenceUnit.INDEX_0_1:
        return f"{number:.4f}"
    if unit == EvidenceUnit.INDEX_100:
        return _trim(number, 1)
    if unit == EvidenceUnit.USD:
        return f"${number:,.2f}"
    if unit == EvidenceUnit.RANK:
        return f"{_trim(number, 2)}위"
    if unit == EvidenceUnit.COUNT:
        return f"{int(number):,}" if number.is_integer() else _trim(number, 2)
    if unit == EvidenceUnit.RATING_5:
        return f"{_trim(number, 2)}/5"
    if unit == EvidenceUnit.RATING_POINTS:
        return f"{number:+.3f}"
    return str(value)


def format_metric_text(
    subject: str, predicate: str, object: str | None, value: EvidenceValue, unit: str | None
) -> str:
    """수치 카드 한 줄 (id·날짜·출처 제외): ``LANEIGE lip_care SoS 3.2%``."""
    label = METRIC_LABELS.get(predicate, predicate)
    parts = [subject, object, label, format_value(value, unit)]
    return " ".join(part for part in parts if part)


def select_cards(
    cards: Iterable[Evidence],
    max_per_kind: int | Mapping[EvidenceKind, int] | None = None,
) -> list[Evidence]:
    """렌더링에 실릴 카드를 입력 순서대로 고른다 (id 중복 제거 → 종류별 상한).

    조립기·평가 트레이스는 이 결과를 기록하면 프롬프트에 실린 카드 집합과 정확히 같다.
    """
    seen: set[str] = set()
    counts: dict[EvidenceKind, int] = {}
    selected: list[Evidence] = []
    for card in cards:
        if card.id in seen:
            continue
        seen.add(card.id)
        if isinstance(max_per_kind, Mapping):
            limit = max_per_kind.get(card.kind)
        else:
            limit = max_per_kind
        if limit is not None and counts.get(card.kind, 0) >= limit:
            continue
        counts[card.kind] = counts.get(card.kind, 0) + 1
        selected.append(card)
    return selected


def _truncate(text: str, max_chars: int | None) -> str:
    if max_chars is None or len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip() + "…"


def _suffix(card: Evidence) -> str:
    parts = [card.as_of, card.source] if card.as_of else [card.source]
    suffix = ", ".join(parts)
    if card.derived_from:
        suffix += "; 근거: " + ", ".join(card.derived_from)
    return f"({suffix})"


def _card_lines(card: Evidence, max_detail_chars: int | None) -> list[str]:
    if card.kind == EvidenceKind.METRIC:
        head = format_metric_text(
            card.display_subject, card.predicate, card.object, card.value, card.unit
        )
    else:
        head = card.text
    lines = [f"[{card.id}] {head} {_suffix(card)}"]

    if card.detail and card.kind != EvidenceKind.METRIC:
        detail = _truncate(card.detail.strip(), max_detail_chars)
        if card.kind == EvidenceKind.INFERENCE:
            lines.append(f"{_DETAIL_INDENT}권장: {detail}")
        else:
            lines.extend(f"{_DETAIL_INDENT}{line}" for line in detail.splitlines() if line.strip())
    return lines


def render_for_prompt(
    cards: Iterable[Evidence],
    max_per_kind: int | Mapping[EvidenceKind, int] | None = None,
    max_detail_chars: int | None = DEFAULT_MAX_DETAIL_CHARS,
) -> str:
    """답변 프롬프트용 증거 블록. 카드가 없으면 빈 문자열.

    형식: 종류별 섹션(``[DB 수치]`` → ``[관계]`` → ``[규칙 추론]`` → ``[문서]`` →
    ``[도구 관찰]``), 각 줄 ``[id] 내용 (as_of, source)``. 수치 카드의 내용은 ``text``가
    아니라 필드(value·unit)에서 다시 만든다 — 표시 수치가 항상 카드 값과 일치한다.
    """
    selected = select_cards(cards, max_per_kind)
    blocks: list[str] = []
    for kind in SECTION_ORDER:
        of_kind = [card for card in selected if card.kind == kind]
        if not of_kind:
            continue
        lines = [SECTION_TITLES[kind]]
        for card in of_kind:
            lines.extend(_card_lines(card, max_detail_chars))
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def render_for_judge(
    cards: Iterable[Evidence],
    max_per_kind: int | Mapping[EvidenceKind, int] | None = None,
    max_detail_chars: int | None = DEFAULT_MAX_DETAIL_CHARS,
) -> str:
    """judge 컨텍스트용 증거 블록 — 프롬프트와 같은 카드를 같은 내용으로 싣는다.

    지금은 ``render_for_prompt``와 출력이 동일하다. judge가 답변 모델보다 더 많은
    본문을 보면 모델이 보지 못한 근거로 근거성을 채점하게 되므로, 길이 상한도 같은
    기본값을 쓴다. 둘을 다르게 할 이유가 생기면 여기서 바꾸고 그 이유를 적는다.
    """
    return render_for_prompt(cards, max_per_kind=max_per_kind, max_detail_chars=max_detail_chars)
