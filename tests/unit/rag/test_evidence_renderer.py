"""증거 카드 렌더러 — 프롬프트·judge가 같은 카드를 같은 내용으로 읽는다 (설계 E1, E8)

비율(0~1)은 여기서만 %로 바뀐다. 섹션 순서와 같은 종류 안의 순서는 결정적이어야
답변 인용과 평가 트레이스가 재현된다.
"""

import pytest

from src.domain.entities.evidence import Evidence, EvidenceKind, EvidenceUnit
from src.rag.evidence_renderer import (
    CITATION_INSTRUCTION,
    format_metric_text,
    format_value,
    render_for_judge,
    render_for_prompt,
    select_cards,
)


def _metric(subject, predicate, value, unit, obj="lip_care", display=None, as_of="2026-08-31"):
    return Evidence.create(
        kind=EvidenceKind.METRIC,
        subject=subject,
        predicate=predicate,
        object=obj,
        value=value,
        unit=unit,
        as_of=as_of,
        source="sqlite:brand_metrics",
        confidence=1.0,
        text="(렌더러가 필드에서 다시 만든다)",
        metadata={"display_name": display} if display else {},
    )


def _relation(subject, predicate, obj):
    return Evidence.create(
        kind=EvidenceKind.RELATION,
        subject=subject.lower(),
        predicate=predicate,
        object=obj.lower(),
        source="kg",
        text=f"{subject} {predicate} {obj}",
    )


class TestFormatValue:
    @pytest.mark.parametrize(
        ("value", "unit", "expected"),
        [
            (0.032, EvidenceUnit.RATIO, "3.2%"),
            (0.1354, EvidenceUnit.RATIO, "13.54%"),
            (0.02, EvidenceUnit.RATIO, "2%"),
            (0.020833, EvidenceUnit.RATIO, "2.08%"),
            (0.0, EvidenceUnit.RATIO, "0%"),
            (0.0681, EvidenceUnit.INDEX_0_1, "0.0681"),
            (0.1, EvidenceUnit.INDEX_0_1, "0.1000"),
            (111.1, EvidenceUnit.INDEX_100, "111.1"),
            (21.6, EvidenceUnit.USD, "$21.60"),
            (1234.5, EvidenceUnit.USD, "$1,234.50"),
            (7, EvidenceUnit.RANK, "7위"),
            (42.46, EvidenceUnit.RANK, "42.46위"),
            (37356, EvidenceUnit.COUNT, "37,356"),
            (4.57, EvidenceUnit.RATING_5, "4.57/5"),
            (4.6, EvidenceUnit.RATING_5, "4.6/5"),
            (-0.098, EvidenceUnit.RATING_POINTS, "-0.098"),
            (0.076, EvidenceUnit.RATING_POINTS, "+0.076"),
            (False, EvidenceUnit.BOOLEAN, "없음"),
            (True, EvidenceUnit.BOOLEAN, "있음"),
            ("dominant", None, "dominant"),
        ],
    )
    def test_format(self, value, unit, expected):
        assert format_value(value, unit) == expected

    def test_metric_text(self):
        assert (
            format_metric_text("LANEIGE", "sos", "lip_care", 0.032, EvidenceUnit.RATIO)
            == "LANEIGE lip_care SoS 3.2%"
        )
        assert (
            format_metric_text("lip_care", "hhi", None, 0.0681, EvidenceUnit.INDEX_0_1)
            == "lip_care HHI 0.0681"
        )
        assert (
            format_metric_text(
                "LANEIGE", "present_in_top100", "lip_care", False, EvidenceUnit.BOOLEAN
            )
            == "LANEIGE lip_care Top100 진입 없음"
        )


class TestRenderForPrompt:
    def test_empty(self):
        assert render_for_prompt([]) == ""
        assert render_for_judge([]) == ""

    def test_snapshot_sections_and_order(self):
        sos = _metric("laneige", "sos", 0.032, EvidenceUnit.RATIO, display="LANEIGE")
        hhi = Evidence.create(
            kind=EvidenceKind.METRIC,
            subject="lip_care",
            predicate="hhi",
            value=0.0681,
            unit=EvidenceUnit.INDEX_0_1,
            as_of="2026-08-31",
            source="sqlite:market_metrics",
            confidence=1.0,
            text="t",
        )
        rel = _relation("LANEIGE", "ownedBy", "AMOREPACIFIC")
        inference = Evidence.create(
            kind=EvidenceKind.INFERENCE,
            subject="laneige",
            predicate="market_position",
            object="lip_care",
            value="fragmented",
            as_of="2026-08-31",
            source="rule:fragmented_market_competition",
            confidence=0.85,
            derived_from=[hhi.id],
            text="시장이 분산되어 있습니다(HHI: 0.068).",
            detail="차별화 전략 강화",
        )
        doc = Evidence.create(
            kind=EvidenceKind.DOCUMENT,
            subject="metric_guide",
            predicate="states",
            source="rag:metric_guide",
            text="SoS 정의",
            detail="SoS는 Top 100 중 브랜드 제품 비율이다.\n두 번째 줄",
            id_basis="metric_guide_0",
        )
        obs = Evidence.create(
            kind=EvidenceKind.OBSERVATION,
            subject="get_metrics",
            predicate="observed",
            source="tool:get_metrics",
            text="get_metrics(brand=laneige) → 3건",
            detail='{"rows": 3}',
            id_basis="obs-1",
        )
        # 입력 순서를 섞어도 섹션 순서는 고정, 같은 종류 안에서는 입력 순서
        rendered = render_for_prompt([doc, obs, rel, inference, sos, hhi])

        expected = "\n".join(
            [
                "[DB 수치]",
                f"[{sos.id}] LANEIGE lip_care SoS 3.2% (2026-08-31, sqlite:brand_metrics)",
                f"[{hhi.id}] lip_care HHI 0.0681 (2026-08-31, sqlite:market_metrics)",
                "",
                "[관계]",
                f"[{rel.id}] LANEIGE ownedBy AMOREPACIFIC (kg)",
                "",
                "[규칙 추론]",
                f"[{inference.id}] 시장이 분산되어 있습니다(HHI: 0.068). "
                f"(2026-08-31, rule:fragmented_market_competition; 근거: {hhi.id})",
                "  권장: 차별화 전략 강화",
                "",
                "[문서]",
                f"[{doc.id}] SoS 정의 (rag:metric_guide)",
                "  SoS는 Top 100 중 브랜드 제품 비율이다.",
                "  두 번째 줄",
                "",
                "[도구 관찰]",
                f"[{obs.id}] get_metrics(brand=laneige) → 3건 (tool:get_metrics)",
                '  {"rows": 3}',
            ]
        )
        assert rendered == expected

    def test_duplicate_ids_rendered_once(self):
        sos = _metric("laneige", "sos", 0.032, EvidenceUnit.RATIO, display="LANEIGE")
        again = _metric("laneige", "sos", 0.032, EvidenceUnit.RATIO, display="laneige")
        rendered = render_for_prompt([sos, again])
        assert rendered.count(sos.id) == 1
        assert "LANEIGE lip_care" in rendered  # 먼저 온 카드의 표기

    def test_max_per_kind(self):
        cards = [_metric(f"brand{i}", "sos", 0.01 * (i + 1), EvidenceUnit.RATIO) for i in range(5)]
        rel = _relation("a", "competesWith", "b")
        selected = select_cards([*cards, rel], max_per_kind=2)
        assert selected == [cards[0], cards[1], rel]
        rendered = render_for_prompt([*cards, rel], max_per_kind={EvidenceKind.METRIC: 3})
        assert [c.id in rendered for c in cards] == [True, True, True, False, False]
        assert rel.id in rendered

    def test_detail_truncation(self):
        doc = Evidence.create(
            kind=EvidenceKind.DOCUMENT,
            subject="d",
            predicate="states",
            source="rag",
            text="제목",
            detail="가" * 50,
            id_basis="d_0",
        )
        rendered = render_for_prompt([doc], max_detail_chars=10)
        assert rendered.splitlines()[-1] == "  " + "가" * 10 + "…"

    def test_judge_sees_same_content(self):
        cards = [
            _metric("laneige", "sos", 0.032, EvidenceUnit.RATIO, display="LANEIGE"),
            _relation("LANEIGE", "competesWith", "COSRX"),
        ]
        assert render_for_judge(cards) == render_for_prompt(cards)
        assert render_for_judge(cards, max_per_kind=1) == render_for_prompt(cards, max_per_kind=1)


def test_citation_instruction_mentions_format():
    assert "[M-" in CITATION_INSTRUCTION
    assert "카드에 없는 수치" in CITATION_INSTRUCTION
