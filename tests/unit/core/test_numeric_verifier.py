"""답변 수치 검증기 — 답 속 숫자가 인용 카드의 값과 맞는지 규칙으로 본다 (설계 E8 뒷부분).

카드는 전부 실제 ``Evidence``이고, 답변은 실제 v4 답변 형식을 그대로 흉내 낸 문자열이다.
카드 id는 필드에서 계산되므로 답변 문자열에 ``card.id``를 끼워 넣는다.
"""

import pytest

from src.core.numeric_verifier import (
    UNVERIFIED_PLACEHOLDER,
    NumericStatus,
    apply_numeric_verification,
    extract_citation_ids,
    extract_numbers,
    verify_numeric_claims,
)
from src.domain.entities.evidence import Evidence, EvidenceKind, EvidenceUnit

AS_OF = "2026-08-31"


def _metric(subject, predicate, value, unit, obj="lip_care", table="brand_metrics"):
    return Evidence.create(
        kind=EvidenceKind.METRIC,
        subject=subject,
        predicate=predicate,
        object=obj,
        value=value,
        unit=unit,
        as_of=AS_OF,
        source=f"sqlite:{table}",
        confidence=1.0,
        text="(렌더러가 필드에서 다시 만든다)",
        metadata={"display_name": "LANEIGE"} if subject == "laneige" else {},
    )


HHI = _metric("lip_care", "hhi", 0.06814, EvidenceUnit.INDEX_0_1, obj=None, table="market_metrics")
SOS = _metric("laneige", "sos", 0.02, EvidenceUnit.RATIO)
SOS_RANK = _metric("laneige", "sos_rank", 9, EvidenceUnit.RANK)
PRODUCT_COUNT = _metric("laneige", "product_count", 2, EvidenceUnit.COUNT)
PRICE = _metric("lip_sleeping_mask", "price", 21.6, EvidenceUnit.USD, table="raw_data")
RATING = _metric("lip_sleeping_mask", "rating", 4.57, EvidenceUnit.RATING_5, table="raw_data")
SOS_1354 = _metric("cosrx", "sos", 0.1354, EvidenceUnit.RATIO)
HHI_DOC = Evidence.create(
    kind=EvidenceKind.DOCUMENT,
    subject="hhi_guide",
    predicate="states",
    source="rag:metric_guide",
    text="HHI 해석 가이드",
    detail="HHI가 0.15 미만이면 경쟁이 분산된 시장, 0.25 이상이면 고집중 시장으로 본다.",
    id_basis="hhi_guide_0",
)
COMPETES = Evidence.create(
    kind=EvidenceKind.RELATION,
    subject="laneige",
    predicate="competesWith",
    object="cosrx",
    source="kg:competesWith",
    text="LANEIGE competesWith COSRX (Lip Care Top 3 경쟁)",
)
INFERENCE = Evidence.create(
    kind=EvidenceKind.INFERENCE,
    subject="laneige",
    predicate="market_position",
    object="lip_care",
    value="challenger",
    as_of=AS_OF,
    source="rule:strong_avg_rank",
    confidence=0.8,
    derived_from=(SOS_RANK.id,),
    text="LANEIGE는 Lip Care 도전자 위치",
    detail="평균 순위 상위 브랜드와의 격차를 줄이는 전략",
)

CARDS = [HHI, SOS, SOS_RANK, PRODUCT_COUNT, PRICE, RATING, SOS_1354, HHI_DOC, COMPETES, INFERENCE]


def _statuses(result):
    return [(claim.text, claim.status) for claim in result.claims]


# ----------------------------------------------------------------------
# 인용 추출
# ----------------------------------------------------------------------


class TestCitationExtraction:
    def test_consecutive_brackets(self):
        assert extract_citation_ids("… [M-4dcfa7][D-d96ee3]") == ["M-4dcfa7", "D-d96ee3"]

    def test_comma_list_with_spaces(self):
        assert extract_citation_ids("… [ M-4dcfa7, D-a8396e ]") == ["M-4dcfa7", "D-a8396e"]

    def test_space_between_brackets_and_extended_hex(self):
        text = "값 [M-dce503] [I-931df1a] 끝"
        assert extract_citation_ids(text) == ["M-dce503", "I-931df1a"]

    def test_non_card_brackets_ignored(self):
        assert extract_citation_ids("[DB 수치] [인용 규칙] [M-xyz]") == []


# ----------------------------------------------------------------------
# 수치 추출·제외
# ----------------------------------------------------------------------


class TestNumberExtraction:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            ("SoS 2%로", [("2%", 2.0, "percent")]),
            ("점유율 13.54%이다", [("13.54%", 13.54, "percent")]),
            ("HHI는 0.0681로", [("0.0681", 0.0681, "number")]),
            ("SoS 순위 9위", [("9위", 9.0, "rank")]),
            ("BSR #9 제품", [("#9", 9.0, "rank")]),
            ("가격은 $21.60", [("$21.60", 21.6, "usd")]),
            ("제품 13개", [("13개", 13.0, "count")]),
            ("리뷰 12,345건", [("12,345건", 12345.0, "count")]),
            ("평점 4.57/5", [("4.57/5", 4.57, "rating")]),
        ],
    )
    def test_metric_numbers(self, text, expected):
        assert [(n.text, n.value, n.kind) for n in extract_numbers(text)] == expected

    @pytest.mark.parametrize(
        "text",
        [
            "2026-08-31 기준",
            "2026년 8월 31일 기준",
            "2026년 데이터",
            "2026 시즌",
            "Q3 실적과 3분기 흐름",
            "Amazon Top 100 안에서",
            "Top100 기준",
            "상위 10개 브랜드",
            "10위권 밖",
            "최근 3개월, 4주, 24시간",
            "5점 만점",
            "1. 요약",
            "## 2 분석",
            "근거 M-4dcfa7 카드",
            "[M-123456][D-a8396e]",
            "L2 카테고리와 GPT-4.1 모델",
            "3가지 전략, 1순위 과제, 2단계",
            "22:00 KST",
            "용량 20g",
        ],
    )
    def test_non_metric_numbers_excluded(self, text):
        assert extract_numbers(text) == []

    @pytest.mark.parametrize(
        ("text", "expected"),
        [
            # 제외 패턴이 다른 수의 중간("0.0681 일…"의 "81 일")에서 시작하지 않는다
            ("HHI 0.0681 일반적으로 낮다", [("0.0681", 0.0681)]),
            ("0.15 주의 구간", [("0.15", 0.15)]),
            ("순위 10-20위 사이", [("10", 10.0), ("20위", 20.0)]),
            ("평점 격차 -0.12점", [("-0.12점", -0.12)]),
            ("**2%** 수준", [("2%", 2.0)]),
        ],
    )
    def test_boundaries(self, text, expected):
        assert [(n.text, n.value) for n in extract_numbers(text)] == expected


# ----------------------------------------------------------------------
# 검증: 실제 답변 예시
# ----------------------------------------------------------------------


class TestRealAnswersVerified:
    def test_hhi_answer_with_document_citation(self):
        answer = (
            f"Lip Care 카테고리의 HHI는 0.0681로 경쟁이 분산된 시장입니다 [{HHI.id}][{HHI_DOC.id}]."
        )
        result = verify_numeric_claims(answer, CARDS)
        assert _statuses(result) == [("0.0681", NumericStatus.VERIFIED)]
        assert result.claims[0].matched_ids == (HHI.id,)

    def test_brand_answer_with_three_metric_citations(self):
        answer = (
            f"LANEIGE는 Lip Care에서 SoS 2%로 9위이며 Top 100에 제품 2개가 있습니다 "
            f"[{SOS.id}][{SOS_RANK.id}][{PRODUCT_COUNT.id}]."
        )
        result = verify_numeric_claims(answer, CARDS)
        assert _statuses(result) == [
            ("2%", NumericStatus.VERIFIED),
            ("9위", NumericStatus.VERIFIED),
            ("2개", NumericStatus.VERIFIED),
        ]

    def test_comma_list_citation(self):
        answer = f"HHI는 0.0681입니다 [ {HHI.id}, {HHI_DOC.id} ]."
        result = verify_numeric_claims(answer, CARDS)
        assert _statuses(result) == [("0.0681", NumericStatus.VERIFIED)]
        assert result.claims[0].cited_ids == (HHI.id, HHI_DOC.id)


# ----------------------------------------------------------------------
# 검증: 분류 규칙
# ----------------------------------------------------------------------


class TestClassification:
    def test_mismatch_against_cited_card(self):
        result = verify_numeric_claims(f"HHI는 0.12[{HHI.id}]입니다.", CARDS)
        assert _statuses(result) == [("0.12", NumericStatus.MISMATCH)]
        assert result.claims[0].cited_ids == (HHI.id,)

    def test_value_in_uncited_card_is_mismatch_but_reported(self):
        # 2%는 SoS 카드에 있지만 이 문장이 인용한 HHI 카드에는 없다
        result = verify_numeric_claims(f"SoS는 2%입니다 [{HHI.id}].", CARDS)
        claim = result.claims[0]
        assert claim.status == NumericStatus.MISMATCH
        assert claim.found_in_ids == (SOS.id,)

    def test_no_citation(self):
        result = verify_numeric_claims("LANEIGE SoS는 2%입니다.", CARDS)
        assert _statuses(result) == [("2%", NumericStatus.NO_CITATION)]

    def test_unknown_card(self):
        result = verify_numeric_claims("SoS는 5%입니다 [M-abcdef].", CARDS)
        claim = result.claims[0]
        assert claim.status == NumericStatus.UNKNOWN_CARD
        assert claim.unknown_ids == ("M-abcdef",)

    def test_known_match_wins_over_unknown_citation(self):
        result = verify_numeric_claims(f"SoS는 2%입니다 [{SOS.id}][M-abcdef].", CARDS)
        assert _statuses(result) == [("2%", NumericStatus.VERIFIED)]

    def test_each_number_uses_following_citation(self):
        answer = f"HHI는 0.0681[{HHI.id}]이고 SoS는 2%[{SOS.id}]입니다."
        result = verify_numeric_claims(answer, CARDS)
        assert _statuses(result) == [
            ("0.0681", NumericStatus.VERIFIED),
            ("2%", NumericStatus.VERIFIED),
        ]

    def test_citation_after_period_belongs_to_sentence(self):
        answer = f"SoS는 2%입니다. [{SOS.id}]\n다음 문장입니다."
        result = verify_numeric_claims(answer, CARDS)
        assert _statuses(result) == [("2%", NumericStatus.VERIFIED)]

    def test_sentences_do_not_share_citations(self):
        answer = f"HHI는 0.0681입니다 [{HHI.id}]. SoS는 2%입니다."
        result = verify_numeric_claims(answer, CARDS)
        assert _statuses(result) == [
            ("0.0681", NumericStatus.VERIFIED),
            ("2%", NumericStatus.NO_CITATION),
        ]

    def test_dates_years_quarters_top100_not_checked(self):
        answer = (
            f"2026-08-31 기준(2026년 Q3, 3분기) Amazon Top 100에서 "
            f"LANEIGE SoS는 2%입니다 [{SOS.id}]."
        )
        result = verify_numeric_claims(answer, CARDS)
        assert _statuses(result) == [("2%", NumericStatus.VERIFIED)]

    def test_document_detail_number(self):
        answer = f"HHI가 0.15 미만이면 분산 시장으로 봅니다 [{HHI_DOC.id}]."
        result = verify_numeric_claims(answer, CARDS)
        assert _statuses(result) == [("0.15", NumericStatus.VERIFIED)]

    def test_document_without_number_is_mismatch(self):
        answer = f"HHI가 0.18 미만이면 분산 시장으로 봅니다 [{HHI_DOC.id}]."
        result = verify_numeric_claims(answer, CARDS)
        assert _statuses(result) == [("0.18", NumericStatus.MISMATCH)]

    def test_relation_text_number(self):
        answer = f"COSRX는 Lip Care Top 3 경쟁사입니다 [{COMPETES.id}]."
        # "Top 3"은 비지표라 검사 대상이 아니다
        assert verify_numeric_claims(answer, CARDS).claims == ()

    def test_inference_card_reaches_derived_metric(self):
        answer = f"LANEIGE는 9위로 도전자 위치입니다 [{INFERENCE.id}]."
        result = verify_numeric_claims(answer, CARDS)
        assert _statuses(result) == [("9위", NumericStatus.VERIFIED)]

    def test_display_units(self):
        answer = f"가격은 $21.60이고 평점은 4.57/5입니다 [{PRICE.id}][{RATING.id}]."
        result = verify_numeric_claims(answer, CARDS)
        assert [c.status for c in result.claims] == [NumericStatus.VERIFIED] * 2

    def test_kind_must_be_compatible(self):
        # 9는 순위 카드 값이지만 "9%"는 순위가 아니다
        result = verify_numeric_claims(f"SoS는 9%입니다 [{SOS_RANK.id}].", CARDS)
        assert _statuses(result) == [("9%", NumericStatus.MISMATCH)]


class TestTolerance:
    @pytest.mark.parametrize("shown", ["13.54%", "13.5%", "14%"])
    def test_rounded_display_matches(self, shown):
        result = verify_numeric_claims(f"COSRX SoS는 {shown}입니다 [{SOS_1354.id}].", CARDS)
        assert result.claims[0].status == NumericStatus.VERIFIED

    @pytest.mark.parametrize("shown", ["13.6%", "13%", "13.64%"])
    def test_off_by_more_than_rounding_is_mismatch(self, shown):
        result = verify_numeric_claims(f"COSRX SoS는 {shown}입니다 [{SOS_1354.id}].", CARDS)
        assert result.claims[0].status == NumericStatus.MISMATCH

    def test_hhi_rounded(self):
        result = verify_numeric_claims(f"HHI 0.068 [{HHI.id}]", CARDS)
        assert result.claims[0].status == NumericStatus.VERIFIED

    def test_price_one_cent_off_is_mismatch(self):
        result = verify_numeric_claims(f"가격 $21.70 [{PRICE.id}]", CARDS)
        assert result.claims[0].status == NumericStatus.MISMATCH


# ----------------------------------------------------------------------
# 적용 모드
# ----------------------------------------------------------------------


class TestApply:
    def test_annotate_keeps_text_and_records_summary(self):
        answer = f"HHI는 0.12[{HHI.id}]입니다. SoS는 2%입니다 [{SOS.id}]. 순위는 9위입니다."
        text, meta = apply_numeric_verification(answer, CARDS, "annotate")
        assert text == answer
        assert meta["mode"] == "annotate"
        assert meta["skipped"] is None
        assert (meta["checked"], meta["verified"], meta["mismatch"]) == (3, 1, 1)
        assert (meta["no_citation"], meta["unknown_card"], meta["replaced"]) == (1, 0, 0)
        assert meta["details"][0]["status"] == "mismatch"
        assert meta["details"][0]["number"] == "0.12"
        assert meta["details"][0]["cited_ids"] == [HHI.id]

    def test_enforce_replaces_mismatch_and_unknown_only(self):
        answer = (
            f"LANEIGE SoS 5%로 [{SOS.id}] 부진합니다. "
            f"가격은 $30.00입니다 [M-abcdef]. 순위는 9위입니다."
        )
        text, meta = apply_numeric_verification(answer, CARDS, "enforce")
        assert text == (
            f"LANEIGE SoS {UNVERIFIED_PLACEHOLDER}로 [{SOS.id}] 부진합니다. "
            f"가격은 {UNVERIFIED_PLACEHOLDER}입니다 [M-abcdef]. 순위는 9위입니다."
        )
        assert UNVERIFIED_PLACEHOLDER == "확인되지 않음"
        assert (meta["mismatch"], meta["unknown_card"], meta["no_citation"]) == (1, 1, 1)
        assert meta["replaced"] == 2

    def test_enforce_mismatch_example(self):
        text, meta = apply_numeric_verification(f"HHI는 0.12[{HHI.id}]", CARDS, "enforce")
        assert text == f"HHI는 확인되지 않음[{HHI.id}]"
        assert meta["replaced"] == 1

    def test_off_returns_no_metadata(self):
        answer = f"HHI는 0.12[{HHI.id}]"
        assert apply_numeric_verification(answer, CARDS, "off") == (answer, None)

    def test_no_cards_is_skipped(self):
        text, meta = apply_numeric_verification("SoS는 5%입니다.", [], "enforce")
        assert text == "SoS는 5%입니다."
        assert meta["skipped"] == "no_evidence"
        assert meta["checked"] == 0 and meta["replaced"] == 0 and meta["details"] == []

    def test_details_are_capped_and_unverified_first(self):
        verified = " ".join(f"SoS 2%[{SOS.id}]." for _ in range(12))
        answer = f"{verified} HHI 0.5[{HHI.id}]."
        _, meta = apply_numeric_verification(answer, CARDS, "annotate")
        assert meta["checked"] == 13
        assert len(meta["details"]) == 10
        assert meta["details"][0]["status"] == "mismatch"

    def test_unknown_mode_rejected(self):
        with pytest.raises(ValueError):
            apply_numeric_verification("x", CARDS, "strict")


# ----------------------------------------------------------------------
# 요구 경계 사례: 퍼센트↔비율, 천 단위 구분, 음수, 통화, 영어 답변, 메타데이터 계약
# ----------------------------------------------------------------------

SOS_018 = _metric("summer_fridays", "sos", 0.18, EvidenceUnit.RATIO)
REVIEWS = _metric("lip_sleeping_mask", "reviews_count", 12345, EvidenceUnit.COUNT, table="raw_data")
RATING_GAP = _metric("laneige", "avg_rating_gap", -0.12, EvidenceUnit.RATING_POINTS)
EDGE_CARDS = [SOS_018, REVIEWS, RATING_GAP, PRICE, RATING, SOS_RANK, HHI, SOS]


def _status_of(answer: str) -> list[tuple[str, str]]:
    return [(c.text, c.status.value) for c in verify_numeric_claims(answer, EDGE_CARDS).claims]


class TestPercentVersusRatio:
    """렌더러는 비율 0.18을 ``18%``로 싣는다 — 답의 퍼센트는 ×100 값과 비교한다."""

    @pytest.mark.parametrize("shown", ["18.0%", "18%", "18 %", "18퍼센트", "18 percent"])
    def test_percent_display_of_ratio_is_verified(self, shown):
        assert _status_of(f"SoS {shown} [{SOS_018.id}]") == [(shown, "verified")]

    def test_ratio_written_as_percent_is_mismatch(self):
        # 0.18%는 0.18의 표시값(18%)이 아니다
        assert _status_of(f"SoS 0.18% [{SOS_018.id}]") == [("0.18%", "mismatch")]

    def test_canonical_ratio_without_unit_is_verified(self):
        assert _status_of(f"SoS 비율 0.18 [{SOS_018.id}]") == [("0.18", "verified")]

    def test_renderer_rounding_is_the_reference(self):
        # format_value(0.1354, ratio) == "13.54%" → 소수 둘째 자리까지 같은 값만 통과
        for shown, status in (("13.54%", "verified"), ("13.55%", "mismatch")):
            result = verify_numeric_claims(f"SoS {shown} [{SOS_1354.id}]", [SOS_1354])
            assert [(c.text, c.status.value) for c in result.claims] == [(shown, status)]


class TestSeparatorsSignsCurrency:
    @pytest.mark.parametrize(
        ("answer", "number"),
        [
            ("리뷰 12,345건", "12,345건"),
            ("리뷰 12345건", "12345건"),
            ("리뷰 1.2만 건", "1.2만 건"),
            ("12,345 reviews", "12,345"),
            ("12.3K reviews", "12.3K"),
        ],
    )
    def test_thousands_and_scale_verified(self, answer, number):
        assert _status_of(f"{answer} [{REVIEWS.id}]") == [(number, "verified")]

    @pytest.mark.parametrize("shown", ["12,354건", "1.3만 건"])
    def test_wrong_count_is_mismatch(self, shown):
        assert _status_of(f"리뷰 {shown} [{REVIEWS.id}]") == [(shown, "mismatch")]

    def test_negative_matches_signed_card(self):
        assert _status_of(f"평점 격차 -0.12점 [{RATING_GAP.id}]") == [("-0.12점", "verified")]
        assert _status_of(f"평점 격차 −0.12 [{RATING_GAP.id}]") == [("−0.12", "verified")]

    def test_explicit_wrong_sign_is_mismatch(self):
        assert _status_of(f"평점 격차 +0.12점 [{RATING_GAP.id}]") == [("+0.12점", "mismatch")]

    def test_unsigned_magnitude_is_verified(self):
        assert _status_of(f"평점이 0.12점 낮습니다 [{RATING_GAP.id}]") == [("0.12점", "verified")]

    @pytest.mark.parametrize("shown", ["$21.60", "$21.6", "21.60달러", "USD 21.60", "22달러"])
    def test_currency_forms_verified(self, shown):
        assert _status_of(f"가격 {shown} [{PRICE.id}]") == [(shown, "verified")]

    def test_currency_against_non_usd_card_is_mismatch(self):
        # 9는 순위 카드 값이지만 $9는 순위가 아니다
        assert _status_of(f"가격 $9 [{SOS_RANK.id}]") == [("$9", "mismatch")]


class TestExclusions:
    @pytest.mark.parametrize(
        "phrase",
        [
            "2026-08-31",
            "2026년 8월 31일",
            "2026년 3분기",
            "1분기",
            "Q1",
            "상반기",
            "Top 100",
            "Top100",
            "top-10",
            "상위 100개",
            "100위권",
            "Aug 31, 2026",
            "31 August",
            "8/31",
            "3rd quarter",
            "past 30 days",
            "out of 100",
        ],
    )
    def test_phrase_is_not_checked_even_with_citation(self, phrase):
        answer = f"{phrase} 기준 SoS 18% [{SOS_018.id}]"
        assert _status_of(answer) == [("18%", "verified")]


class TestEnglishAnswers:
    def test_english_sentence_mixed_statuses(self):
        answer = (
            f"As of Aug 31, 2026, LANEIGE ranked 9th in the Amazon Top 100 [{SOS_RANK.id}]. "
            f"Its SoS was 18.0% in Q3 [{SOS_018.id}], with 12,345 reviews rated 4.57 out of 5 "
            f"[{REVIEWS.id}][{RATING.id}]. The price is $25.00 [{PRICE.id}]. "
            "Share grew 3.5% over the past 30 days."
        )
        assert _status_of(answer) == [
            ("9th", "verified"),
            ("18.0%", "verified"),
            ("12,345", "verified"),
            ("4.57 out of 5", "verified"),
            ("$25.00", "mismatch"),
            ("3.5%", "no_citation"),
        ]

    def test_english_unknown_card(self):
        assert _status_of("HHI is 0.07 [M-0a1b2c].") == [("0.07", "unknown_card")]


class TestMultipleCitations:
    def test_numbers_checked_against_union_of_cluster(self):
        answer = f"LANEIGE는 9위, 리뷰 12,345건, 평점 4.57/5입니다 [{SOS_RANK.id}][{REVIEWS.id}][{RATING.id}]."
        result = verify_numeric_claims(answer, EDGE_CARDS)
        assert [(c.text, c.status.value, c.matched_ids) for c in result.claims] == [
            ("9위", "verified", (SOS_RANK.id,)),
            ("12,345건", "verified", (REVIEWS.id,)),
            ("4.57/5", "verified", (RATING.id,)),
        ]

    def test_unknown_id_in_cluster_with_wrong_number(self):
        answer = f"가격 $30 [{PRICE.id}][M-ffffff]"
        claim = verify_numeric_claims(answer, EDGE_CARDS).claims[0]
        assert claim.status == NumericStatus.UNKNOWN_CARD
        assert claim.cited_ids == (PRICE.id, "M-ffffff")
        assert claim.unknown_ids == ("M-ffffff",)


class TestMetadataContract:
    """리드가 평가 리포트에서 집계하는 ``numeric_verification`` 모양 (모듈 docstring)."""

    SUMMARY_KEYS = {
        "mode",
        "skipped",
        "checked",
        "verified",
        "mismatch",
        "no_citation",
        "unknown_card",
        "found_in_other_cards",
        "replaced",
        "details",
    }
    DETAIL_KEYS = {
        "status",
        "number",
        "value",
        "kind",
        "sentence",
        "cited_ids",
        "unknown_ids",
        "matched_ids",
        "found_in_ids",
    }

    def test_shape_and_count_invariant(self):
        answer = (
            f"SoS 18% [{SOS_018.id}]. 가격 $30 [{PRICE.id}]. 리뷰 12,345건. HHI 0.07 [M-0a1b2c]."
        )
        text, meta = apply_numeric_verification(answer, EDGE_CARDS, "annotate")
        assert text == answer
        assert set(meta) == self.SUMMARY_KEYS
        assert meta["checked"] == (
            meta["verified"] + meta["mismatch"] + meta["no_citation"] + meta["unknown_card"]
        )
        assert (meta["verified"], meta["mismatch"], meta["no_citation"], meta["unknown_card"]) == (
            1,
            1,
            1,
            1,
        )
        # 12,345는 인용 안 한 REVIEWS 카드에, 0.07은 HHI 카드(0.0681 → 0.07 반올림)에 있다
        assert meta["found_in_other_cards"] == 2
        assert [d["status"] for d in meta["details"]] == [
            "mismatch",
            "unknown_card",
            "no_citation",
            "verified",
        ]
        for detail in meta["details"]:
            assert set(detail) == self.DETAIL_KEYS
        import json

        json.dumps(meta)  # 평가 리포트에 그대로 직렬화된다

    def test_number_after_last_citation_is_not_attributed_to_it(self):
        answer = f"가격은 $21.60이고 [{PRICE.id}] 리뷰는 12,345건입니다."
        assert _status_of(answer) == [("$21.60", "verified"), ("12,345건", "no_citation")]

    def test_enforce_replacement_keeps_particles_and_citations(self):
        answer = f"가격은 $30.00이고 [{PRICE.id}] 리뷰는 12,345건입니다."
        text, meta = apply_numeric_verification(answer, EDGE_CARDS, "enforce")
        assert text == f"가격은 확인되지 않음이고 [{PRICE.id}] 리뷰는 12,345건입니다."
        assert (meta["replaced"], meta["no_citation"]) == (1, 1)
