"""적합도(fit) 기반 신뢰도 점수 테스트 (트랙 5-B).

무엇을 고정하는가
-----------------
기존 신뢰도는 컨텍스트 개수 합(kg_facts×1.5 + rag_docs×1.0 + …)이라 증거가 질문에
맞는지와 무관했고, 233문항 중 230문항이 HIGH로 나와 DecisionMaker·ReAct 분기가 한 번도
실행되지 않았다. 여기서는 점수를 **증거 적합도**로 바꾼 뒤의 계약을 고정한다:

(a) 질문 엔티티가 증거 카드에 얼마나 닻을 내렸는가 (entity_coverage)
(b) 질문 유형이 요구하는 카드 종류가 있는가 (kind_fit)
(c) 검색 점수 분포가 뾰족한가 (retrieval_fit)

가짜는 쓰지 않는다 — 실제 Evidence 카드, 실제 Context, 실제 QueryGraph를 쓴다.
"""

from __future__ import annotations

import re

import pytest

from src.core.confidence import (
    FIT_WEIGHTS,
    MAX_HIGH_THRESHOLD,
    RETRIEVAL_SWING,
    ConfidenceAssessor,
    EvidenceFit,
    legacy_count_score_to_fit,
    required_evidence_kinds,
    score_evidence_fit,
)
from src.core.models import ConfidenceLevel, Context
from src.domain.entities.evidence import Evidence, EvidenceKind

# =============================================================================
# 카드 만들기 (실제 Evidence 객체)
# =============================================================================


def metric_card(subject: str, predicate: str, value: float, scope: str | None = None) -> Evidence:
    return Evidence.create(
        kind=EvidenceKind.METRIC,
        subject=subject,
        predicate=predicate,
        object=scope,
        value=value,
        unit="ratio",
        as_of="2026-08-31",
        source="sqlite:brand_metrics",
        text=f"{subject} {predicate} {value}",
    )


def relation_card(subject: str, predicate: str, obj: str) -> Evidence:
    return Evidence.create(
        kind=EvidenceKind.RELATION,
        subject=subject,
        predicate=predicate,
        object=obj,
        source="kg",
        text=f"{subject} {predicate} {obj}",
    )


def inference_card(subject: str, predicate: str, rule: str) -> Evidence:
    return Evidence.create(
        kind=EvidenceKind.INFERENCE,
        subject=subject,
        predicate=predicate,
        value=rule,
        source=f"rule:{rule}",
        text=f"{subject} 규칙 {rule} 결론",
        metadata={"rule_name": rule, "related_entities": [subject]},
    )


def document_card(doc_id: str, score: float, text: str = "문서 청크") -> Evidence:
    return Evidence.create(
        kind=EvidenceKind.DOCUMENT,
        subject=doc_id,
        predicate="states",
        source="rag:playbook",
        text=text,
        metadata={"chunk_id": doc_id, "score": score, "doc_type": "playbook"},
        id_basis=f"chunk:{doc_id}",
    )


NUMERIC_Q = "LANEIGE의 lip_care SoS는 얼마인가요?"
NUMERIC_ENTITIES = {"brands": ["laneige"], "categories": ["lip_care"], "indicators": ["sos"]}


# =============================================================================
# (a) 엔티티 커버리지
# =============================================================================


class TestEntityCoverage:
    def test_all_question_entities_anchored_gives_full_coverage(self):
        cards = [metric_card("laneige", "sos", 0.12, "lip_care")]
        fit = score_evidence_fit(NUMERIC_Q, NUMERIC_ENTITIES, cards)
        assert fit.entity_coverage == 1.0

    def test_no_question_entity_anchored_gives_zero_coverage(self):
        """질문은 laneige/lip_care를 말하는데 카드는 전혀 다른 주어만 담고 있다."""
        cards = [metric_card("cosrx", "sos", 0.04, "face_powder")]
        fit = score_evidence_fit(NUMERIC_Q, NUMERIC_ENTITIES, cards)
        assert fit.entity_coverage == 0.0
        assert fit.matched_entities == ()

    def test_partial_coverage_is_fraction_of_named_entities(self):
        cards = [metric_card("laneige", "sos", 0.12, "face_powder")]
        fit = score_evidence_fit(NUMERIC_Q, NUMERIC_ENTITIES, cards)
        assert fit.entity_coverage == pytest.approx(0.5)
        assert fit.matched_entities == ("laneige",)

    def test_question_without_named_entities_is_not_penalised(self):
        """정의 질문처럼 이름 붙은 엔티티가 없으면 닻을 내릴 대상도 없다."""
        fit = score_evidence_fit(
            "SoS란 무엇인가요?", {"indicators": ["sos"]}, [document_card("doc_a", 9.0)]
        )
        assert fit.named_entities == ()
        assert fit.entity_coverage == 1.0

    def test_indicator_only_is_not_a_named_entity(self):
        """지표 이름(sos)은 닻 대상이 아니다 — 브랜드·카테고리·제품만 센다."""
        fit = score_evidence_fit(NUMERIC_Q, NUMERIC_ENTITIES, [])
        assert set(fit.named_entities) == {"laneige", "lip_care"}


# =============================================================================
# (b) 질문 유형별 카드 종류 요구
# =============================================================================


class TestRequiredEvidenceKinds:
    def test_numeric_question_requires_metric_cards(self):
        assert required_evidence_kinds(NUMERIC_Q, NUMERIC_ENTITIES) == frozenset({"metric"})

    def test_relation_question_requires_relation_cards(self):
        kinds = required_evidence_kinds("LANEIGE의 모회사는 어디인가요?", {"brands": ["laneige"]})
        assert "relation" in kinds
        assert "metric" not in kinds

    def test_judgement_question_requires_inference_cards(self):
        kinds = required_evidence_kinds(
            "LANEIGE의 시장 포지션 전략은 무엇인가요?", {"brands": ["laneige"]}
        )
        assert "inference" in kinds

    def test_definition_question_requires_document_cards(self):
        assert required_evidence_kinds("SoS란 무엇인가요?", {"indicators": ["sos"]}) == frozenset(
            {"document"}
        )

    def test_hypothetical_metric_question_requires_documents_not_metrics(self):
        """'SoS 5%는 좋은 수치인가요?'는 조회가 아니라 해석 질문이다."""
        kinds = required_evidence_kinds("SoS 5%는 좋은 수치인가요?", {"indicators": ["sos"]})
        assert kinds == frozenset({"document"})

    def test_empty_query_requires_nothing_resolvable(self):
        assert required_evidence_kinds("", {}) == frozenset({"document"})


class TestKindFit:
    def test_numeric_question_without_metric_cards_has_zero_kind_fit(self):
        cards = [document_card("doc_a", 9.0), relation_card("laneige", "ownedBy", "amorepacific")]
        fit = score_evidence_fit(NUMERIC_Q, NUMERIC_ENTITIES, cards)
        assert fit.needs == ("metric",)
        assert fit.kind_fit == 0.0

    def test_numeric_question_with_matching_metric_card_has_full_kind_fit(self):
        cards = [metric_card("laneige", "sos", 0.12, "lip_care")]
        fit = score_evidence_fit(NUMERIC_Q, NUMERIC_ENTITIES, cards)
        assert fit.kind_fit == 1.0

    def test_metric_card_about_another_brand_does_not_satisfy_the_need(self):
        """질문이 부른 엔티티(laneige·lip_care) 어디에도 닻을 내리지 않은 수치 카드."""
        cards = [metric_card("cosrx", "sos", 0.04, "face_powder")]
        fit = score_evidence_fit(NUMERIC_Q, NUMERIC_ENTITIES, cards)
        assert fit.kind_fit == 0.0

    def test_metric_card_scoped_to_the_named_category_does_satisfy_the_need(self):
        """다른 브랜드라도 질문이 부른 카테고리를 대상으로 삼으면 요구를 채운다."""
        cards = [metric_card("cosrx", "sos", 0.04, "lip_care")]
        fit = score_evidence_fit(NUMERIC_Q, NUMERIC_ENTITIES, cards)
        assert fit.kind_fit == 1.0
        assert fit.entity_coverage == pytest.approx(0.5)

    def test_relation_question_needs_a_relation_card_about_the_named_brand(self):
        query = "LANEIGE의 모회사는 어디인가요?"
        entities = {"brands": ["laneige"]}
        without = score_evidence_fit(query, entities, [document_card("doc_a", 9.0)])
        with_card = score_evidence_fit(
            query, entities, [relation_card("laneige", "ownedBy", "amorepacific")]
        )
        assert without.kind_fit == 0.0
        assert with_card.kind_fit == 1.0

    def test_judgement_question_needs_an_inference_card(self):
        query = "LANEIGE의 시장 포지션 전략은 무엇인가요?"
        entities = {"brands": ["laneige"]}
        with_card = score_evidence_fit(
            query, entities, [inference_card("laneige", "market_position", "challenger")]
        )
        assert with_card.kind_fit == 1.0

    def test_partial_kind_fit_when_only_one_of_two_needs_is_met(self):
        query = "LANEIGE의 경쟁 브랜드 대비 시장 포지션 전략은?"
        entities = {"brands": ["laneige"]}
        needs = required_evidence_kinds(query, entities)
        assert needs == frozenset({"relation", "inference"})
        fit = score_evidence_fit(
            query, entities, [relation_card("laneige", "competesWith", "cosrx")]
        )
        assert fit.kind_fit == pytest.approx(0.5)


# =============================================================================
# (c) 검색 점수 분포 + 경계 사례
# =============================================================================


class TestScoreDistributionEdgeCases:
    def test_no_cards_gives_zero_score(self):
        fit = score_evidence_fit(NUMERIC_Q, NUMERIC_ENTITIES, [])
        assert fit.card_count == 0
        assert fit.score == 0.0
        assert fit.basis == "empty"

    def test_single_document_card_has_neutral_retrieval_fit(self):
        """카드가 하나면 '뾰족하다/평평하다'를 말할 수 없다 — 중립값."""
        fit = score_evidence_fit(
            "SoS란 무엇인가요?", {"indicators": ["sos"]}, [document_card("d", 9.0)]
        )
        assert fit.retrieval_fit == pytest.approx(0.5)

    def test_peaked_score_distribution_beats_flat_one(self):
        query = "SoS란 무엇인가요?"
        entities = {"indicators": ["sos"]}
        peaked = score_evidence_fit(
            query,
            entities,
            [document_card(f"d{i}", s) for i, s in enumerate([20.0, 1.0, 1.0, 1.0])],
        )
        flat = score_evidence_fit(
            query,
            entities,
            [document_card(f"d{i}", s) for i, s in enumerate([20.0, 20.0, 20.0, 20.0])],
        )
        assert peaked.retrieval_fit > flat.retrieval_fit
        assert flat.retrieval_fit == pytest.approx(0.0)

    def test_all_matching_evidence_reaches_the_top_of_the_scale(self):
        cards = [
            metric_card("laneige", "sos", 0.12, "lip_care"),
            metric_card("lip_care", "hhi", 0.07),
            document_card("d0", 20.0),
            document_card("d1", 1.0),
            document_card("d2", 1.0),
        ]
        fit = score_evidence_fit(NUMERIC_Q, NUMERIC_ENTITIES, cards)
        assert fit.entity_coverage == 1.0
        assert fit.kind_fit == 1.0
        assert fit.score > 0.9

    def test_components_are_combined_as_base_plus_bounded_retrieval_swing(self):
        cards = [metric_card("laneige", "sos", 0.12, "lip_care")]
        fit = score_evidence_fit(NUMERIC_Q, NUMERIC_ENTITIES, cards)
        base = (
            FIT_WEIGHTS["entity_coverage"] * fit.entity_coverage
            + FIT_WEIGHTS["kind_fit"] * fit.kind_fit
        )
        expected = base + RETRIEVAL_SWING * (fit.retrieval_fit - 0.5)
        assert fit.score == pytest.approx(max(0.0, min(1.0, expected)))
        assert sum(FIT_WEIGHTS.values()) == pytest.approx(1.0)


# =============================================================================
# 회귀 방지: (c)가 HIGH를 혼자 막지 못한다
# =============================================================================


class TestRetrievalShapeCannotBlockHigh:
    """첫 판의 결함 회귀 방지.

    0.50/0.35/0.15 가중합 + HIGH 0.99였을 때는 entity_coverage=1.0·kind_fit=1.0이어도
    retrieval_fit >= 0.933이어야 HIGH였고, 점수가 붙은 문서 카드가 0~1건이면 총점 상한이
    0.925라 **어떤 질의도 HIGH가 될 수 없었다**. 즉 (c) 혼자 레벨을 갈랐다.
    """

    @staticmethod
    def _cards(*documents: Evidence) -> list[Evidence]:
        return [
            metric_card("laneige", "sos", 0.12, "lip_care"),
            metric_card("lip_care", "hhi", 0.07),
            *documents,
        ]

    @pytest.mark.parametrize(
        ("label", "documents"),
        [
            ("문서 0건", ()),
            ("문서 1건", (document_card("d0", 9.0),)),
            ("문서 2건 평평", (document_card("d0", 9.0), document_card("d1", 9.0))),
            ("문서 3건 평평", tuple(document_card(f"d{i}", 9.0) for i in range(3))),
        ],
    )
    def test_requirements_met_is_high_whatever_the_document_scores(self, label, documents):
        fit = score_evidence_fit(NUMERIC_Q, NUMERIC_ENTITIES, self._cards(*documents))
        assert fit.entity_coverage == 1.0, label
        assert fit.kind_fit == 1.0, label
        assert ConfidenceAssessor().assess_fit(fit) == ConfidenceLevel.HIGH, label

    def test_only_unrelated_brand_cards_is_not_high(self):
        cards = [
            metric_card("cosrx", "sos", 0.04, "face_powder"),
            document_card("d0", 20.0),
            document_card("d1", 1.0),
            document_card("d2", 1.0),
        ]
        fit = score_evidence_fit(NUMERIC_Q, NUMERIC_ENTITIES, cards)
        assert fit.entity_coverage == 0.0
        assert ConfidenceAssessor().assess_fit(fit) != ConfidenceLevel.HIGH

    def test_high_threshold_cannot_exceed_the_structural_cap(self):
        """임계값이 상한을 넘으면 (c)가 다시 HIGH를 막을 수 있다."""
        assert ConfidenceAssessor.THRESHOLD_HIGH <= MAX_HIGH_THRESHOLD
        assert MAX_HIGH_THRESHOLD == pytest.approx(1.0 - RETRIEVAL_SWING / 2)

    def test_retrieval_swing_is_bounded_by_construction(self):
        """base가 같으면 retrieval_fit이 0이든 1이든 총점 차이는 swing을 넘지 않는다."""
        query = "SoS란 무엇인가요?"  # 문서를 요구하는 질문이라 (c)가 실제로 적용된다
        entities = {"indicators": ["sos"]}
        flat = score_evidence_fit(query, entities, [document_card(f"d{i}", 9.0) for i in range(3)])
        peaked = score_evidence_fit(
            query,
            entities,
            [document_card("d0", 20.0), document_card("d1", 1.0), document_card("d2", 1.0)],
        )
        assert flat.retrieval_fit == pytest.approx(0.0)
        assert peaked.retrieval_fit > 0.9
        assert peaked.score - flat.score <= RETRIEVAL_SWING + 1e-9

    def test_retrieval_fit_is_neutral_when_the_question_does_not_need_documents(self):
        """수치 질문의 답은 metric 카드가 한다 — 문서 점수 모양은 무관하다."""
        fit = score_evidence_fit(
            NUMERIC_Q,
            NUMERIC_ENTITIES,
            self._cards(document_card("d0", 20.0), document_card("d1", 1.0)),
        )
        assert "document" not in fit.needs
        assert fit.retrieval_fit == pytest.approx(0.5)


class TestMonotonicity:
    def test_adding_a_matching_metric_card_never_lowers_a_numeric_score(self):
        base = [document_card("d0", 20.0), document_card("d1", 1.0)]
        before = score_evidence_fit(NUMERIC_Q, NUMERIC_ENTITIES, base)
        after = score_evidence_fit(
            NUMERIC_Q, NUMERIC_ENTITIES, [*base, metric_card("laneige", "sos", 0.12, "lip_care")]
        )
        assert after.score >= before.score
        assert after.kind_fit >= before.kind_fit
        assert after.entity_coverage >= before.entity_coverage

    def test_adding_matching_cards_one_by_one_is_monotone(self):
        added = [
            metric_card("laneige", "sos", 0.12, "lip_care"),
            metric_card("lip_care", "hhi", 0.07),
            relation_card("laneige", "ownedBy", "amorepacific"),
        ]
        cards: list[Evidence] = [document_card("d0", 20.0), document_card("d1", 1.0)]
        previous = score_evidence_fit(NUMERIC_Q, NUMERIC_ENTITIES, cards).score
        for card in added:
            cards = [*cards, card]
            current = score_evidence_fit(NUMERIC_Q, NUMERIC_ENTITIES, cards).score
            assert current >= previous
            previous = current


# =============================================================================
# 임계값 단일 출처
# =============================================================================


class TestSingleThresholdLadder:
    def test_assessor_defaults_come_from_the_module_constants(self):
        assessor = ConfidenceAssessor()
        assert assessor.threshold_high == ConfidenceAssessor.THRESHOLD_HIGH
        assert assessor.threshold_medium == ConfidenceAssessor.THRESHOLD_MEDIUM
        assert assessor.threshold_low == ConfidenceAssessor.THRESHOLD_LOW

    def test_thresholds_live_on_the_0_1_fit_scale_and_are_ordered(self):
        assert 0.0 < ConfidenceAssessor.THRESHOLD_LOW < ConfidenceAssessor.THRESHOLD_MEDIUM
        assert ConfidenceAssessor.THRESHOLD_MEDIUM < ConfidenceAssessor.THRESHOLD_HIGH <= 1.0

    def test_assess_fit_uses_the_same_ladder_as_assess(self):
        assessor = ConfidenceAssessor()
        for score in (0.0, 0.2, 0.45, 0.6, 0.75, 0.9, 1.0):
            fit = EvidenceFit(
                score=score,
                entity_coverage=score,
                kind_fit=score,
                retrieval_fit=score,
                needs=("metric",),
                named_entities=(),
                matched_entities=(),
                card_count=1,
                basis="fit",
            )
            assert assessor.assess_fit(fit) == assessor.assess({"fit_score": score})

    def test_only_confidence_module_declares_the_ladder(self):
        """다른 모듈이 자기만의 사다리를 다시 선언하지 않았는지 확인한다."""
        from pathlib import Path

        root = Path(__file__).resolve().parents[3] / "src"
        offenders = []
        for path in root.rglob("*.py"):
            if path.name == "confidence.py" and path.parent.name == "core":
                continue
            # iCloud 충돌 사본(`* 2.py`)은 추적되지 않는 백업이라 검사 대상이 아니다
            if re.search(r" \d+\.py$", path.name):
                continue
            text = path.read_text(encoding="utf-8")
            if "THRESHOLD_HIGH" in text or "THRESHOLD_MEDIUM" in text:
                offenders.append(str(path))
        assert offenders == []

    def test_legacy_count_score_keeps_its_old_level(self):
        """개수 기반 점수를 쓰는 호출자(response_pipeline)의 레벨이 바뀌지 않는다."""
        assessor = ConfidenceAssessor()
        assert assessor.assess({"max_score": 6.0}) == ConfidenceLevel.HIGH
        assert assessor.assess({"max_score": 5.0}) == ConfidenceLevel.HIGH
        assert assessor.assess({"max_score": 4.0}) == ConfidenceLevel.MEDIUM
        assert assessor.assess({"max_score": 2.0}) == ConfidenceLevel.LOW
        assert assessor.assess({"max_score": 0.5}) == ConfidenceLevel.UNKNOWN

    def test_legacy_mapping_is_monotone(self):
        previous = -1.0
        for raw in (0.0, 1.0, 1.5, 2.9, 3.0, 4.9, 5.0, 8.0, 10.0, 20.0):
            value = legacy_count_score_to_fit(raw)
            assert value >= previous
            previous = value
        assert 0.0 <= legacy_count_score_to_fit(0.0) <= 1.0
        assert legacy_count_score_to_fit(1000.0) <= 1.0


# =============================================================================
# 실제 QueryGraph 노드
# =============================================================================


def _make_graph():
    from src.core.cache import ResponseCache
    from src.core.query_graph import QueryGraph

    return QueryGraph(
        cache=ResponseCache(),
        context_gatherer=None,
        confidence_assessor=ConfidenceAssessor(),
        decision_maker=None,
        tool_coordinator=None,
        response_pipeline=None,
    )


class TestAssessConfidenceNode:
    def test_numeric_question_without_metric_cards_is_not_high(self):
        from src.core.graph_state import QueryState

        graph = _make_graph()
        state = QueryState(query=NUMERIC_Q)
        state.context = Context(
            query=NUMERIC_Q,
            entities=NUMERIC_ENTITIES,
            prompt_evidence=[document_card("d0", 20.0), document_card("d1", 1.0)],
        )

        result = graph._node_assess_confidence(state)

        assert result.confidence_level != ConfidenceLevel.HIGH
        assert result.metadata["confidence_components"]["kind_fit"] == 0.0

    def test_numeric_question_with_matching_metric_cards_is_high(self):
        from src.core.graph_state import QueryState

        graph = _make_graph()
        state = QueryState(query=NUMERIC_Q)
        state.context = Context(
            query=NUMERIC_Q,
            entities=NUMERIC_ENTITIES,
            prompt_evidence=[
                metric_card("laneige", "sos", 0.12, "lip_care"),
                metric_card("lip_care", "hhi", 0.07),
                document_card("d0", 20.0),
                document_card("d1", 1.0),
                document_card("d2", 1.0),
            ],
        )

        result = graph._node_assess_confidence(state)

        assert result.confidence_level == ConfidenceLevel.HIGH
        assert result.metadata["confidence_components"]["entity_coverage"] == 1.0
        assert result.metadata["confidence_score"] <= 1.0

    def test_node_falls_back_to_legacy_counts_when_no_cards_exist(self):
        """카드가 없는 구형 경로(v1 retriever)는 기존 개수 점수로 평가한다."""
        from src.core.graph_state import QueryState

        graph = _make_graph()
        query = "라네즈 립케어 카테고리 분석해줘"
        state = QueryState(query=query)
        state.context = Context(
            query=query,
            entities={"brands": ["laneige"]},
            rag_docs=[{"content": "d1"}, {"content": "d2"}, {"content": "d3"}],
            kg_facts=[{"fact": "f1"}, {"fact": "f2"}, {"fact": "f3"}],
            kg_inferences=[{"insight": "i1"}],
        )

        result = graph._node_assess_confidence(state)

        assert result.metadata["confidence_components"]["basis"] == "legacy"
        assert result.confidence_level == ConfidenceLevel.HIGH

    def test_missing_context_stays_unknown(self):
        from src.core.graph_state import QueryState

        graph = _make_graph()
        state = QueryState(query=NUMERIC_Q)
        state.context = None

        result = graph._node_assess_confidence(state)

        assert result.confidence_level == ConfidenceLevel.UNKNOWN


# =============================================================================
# 보정 스크립트가 프로덕션 사다리에서 떨어져 나가지 않게 고정
# =============================================================================


class TestCalibrationScriptStaysInSync:
    @staticmethod
    def _module():
        import importlib

        return importlib.import_module("scripts.calibrate_confidence_thresholds")

    def test_split_rule_is_deterministic_and_documented(self):
        """보정/검증 분할은 id의 sha1 최하위 비트로만 정해진다 (실행·순서와 무관)."""
        import hashlib

        module = self._module()
        for item_id in ("lg041", "lg201", "rg032", "rl017"):
            expected = (
                "calibration"
                if int(hashlib.sha1(item_id.encode("utf-8")).hexdigest(), 16) % 2 == 0
                else "validation"
            )
            assert module.split_half(item_id) == expected
            assert module.split_half(item_id) == module.split_half(item_id)

    def test_script_level_mapping_matches_the_assessor(self):
        """스크립트가 자기만의 사다리를 쓰지 않는다 — 같은 점수면 같은 레벨."""
        module = self._module()
        assessor = ConfidenceAssessor()
        high = ConfidenceAssessor.THRESHOLD_HIGH
        medium = ConfidenceAssessor.THRESHOLD_MEDIUM
        low = ConfidenceAssessor.THRESHOLD_LOW
        for score in (0.0, 0.3, low, 0.8, medium, 0.95, high, 1.0):
            assert module.level_of(score, high, medium, low) == (
                assessor.assess({"fit_score": score}).value.upper()
            )
