"""[2026-09 사후] O0-A: L3 골드 엣지 있는 문항만의 recall과 술어별 recall."""

from eval.metrics.l3_kg import L3KGMetrics, aggregate_l3_extended
from eval.schemas import GoldEvidence, KGQueryTrace


def _trace(edges: list[str]) -> KGQueryTrace:
    return KGQueryTrace(kg_entities_found=[], kg_edges_found=edges)


class TestGoldOnlyRecall:
    def test_no_gold_edge_is_none_not_one(self):
        # 기존 kg_edge_recall은 1.0(찾을 것 없음)으로 채운다 — 새 필드는 None으로 빠진다
        m = L3KGMetrics().compute(_trace(["a -competesWith-> b"]), GoldEvidence(kg_edges=[]))
        assert m.kg_edge_recall == 1.0
        assert m.kg_edge_recall_gold_only is None
        assert m.gold_edge_count == 0
        assert m.edge_recall_by_predicate == {}

    def test_gold_only_equals_recall_when_gold_exists(self):
        gold = GoldEvidence(
            kg_edges=["COSRX -ownedByGroup-> amorepacific", "cosrx -competesWith-> laneige"]
        )
        m = L3KGMetrics().compute(_trace(["cosrx -competesWith-> laneige"]), gold)
        assert m.kg_edge_recall == 0.5
        assert m.kg_edge_recall_gold_only == 0.5
        assert (m.gold_edge_count, m.gold_edge_matched) == (2, 1)

    def test_per_predicate_uses_gold_spelling_and_no_alias(self):
        # 런타임은 ownedBy로 방출한다 — 별칭을 맞춰 주지 않으므로 ownedByGroup은 0/1
        gold = GoldEvidence(
            kg_edges=[
                "cosrx -ownedByGroup-> amorepacific",
                "laneige -hasSegment-> premium",
                "laneige -competesWith-> nivea",
            ]
        )
        trace = _trace(["cosrx -ownedBy-> amorepacific", "LANEIGE -competesWith-> nivea"])
        m = L3KGMetrics().compute(trace, gold)
        assert m.edge_recall_by_predicate == {
            "ownedByGroup": {"matched": 0, "total": 1},
            "hasSegment": {"matched": 0, "total": 1},
            "competesWith": {"matched": 1, "total": 1},
        }


class TestCanonicalPredicateRecall:
    """[2026-09 사후] O0-추가: 술어 별칭(ownedBy↔ownedByGroup)·브랜드/그룹 표기를
    온톨로지 로더로 맞춘 뒤 다시 잰 recall. raw 필드는 그대로 비교 대상으로 남는다."""

    def test_alias_predicate_and_case_are_merged(self):
        # 골드는 ownedByGroup·대문자 COSRX, 방출은 ownedBy(별칭)·소문자 cosrx — raw는 놓치지만
        # canonical은 같은 엣지로 본다
        gold = GoldEvidence(kg_edges=["COSRX -ownedByGroup-> AMOREPACIFIC"])
        trace = _trace(["cosrx -ownedBy-> amorepacific"])
        m = L3KGMetrics().compute(trace, gold)
        assert m.edge_recall_by_predicate == {"ownedByGroup": {"matched": 0, "total": 1}}
        assert m.kg_edge_recall_gold_only == 0.0
        assert m.edge_recall_by_predicate_canonical == {"ownedByGroup": {"matched": 1, "total": 1}}
        assert m.kg_edge_recall_gold_only_canonical == 1.0
        assert (m.gold_edge_count_canonical, m.gold_edge_matched_canonical) == (1, 1)

    def test_country_values_are_not_aliased(self):
        # 등록부에 나라 정규화가 없으므로 korea와 south_korea는 canonical에서도 다른 값이다
        gold = GoldEvidence(kg_edges=["laneige -originatesFrom-> south_korea"])
        trace = _trace(["laneige -originatesFrom-> korea"])
        m = L3KGMetrics().compute(trace, gold)
        assert m.kg_edge_recall_gold_only_canonical == 0.0

    def test_no_gold_edge_canonical_is_none(self):
        m = L3KGMetrics().compute(_trace([]), GoldEvidence(kg_edges=[]))
        assert m.kg_edge_recall_gold_only_canonical is None
        assert m.gold_edge_count_canonical == 0
        assert m.edge_recall_by_predicate_canonical == {}


class TestAggregate:
    def test_macro_micro_and_predicate_totals(self):
        calc = L3KGMetrics()
        items = [
            calc.compute(_trace([]), GoldEvidence(kg_edges=[])),  # 골드 없음 → 제외
            calc.compute(
                _trace(["a -competesWith-> b"]),
                GoldEvidence(kg_edges=["a -competesWith-> b", "a -hasSegment-> premium"]),
            ),
            calc.compute(_trace([]), GoldEvidence(kg_edges=["c -competesWith-> d"])),
        ]
        agg = aggregate_l3_extended(items)
        assert agg["items"] == 3
        assert agg["kg_edge_recall_all"] == (1.0 + 0.5 + 0.0) / 3
        assert agg["gold_edge_items"] == 2
        assert agg["kg_edge_recall_gold_only"] == 0.25
        assert agg["kg_edge_recall_micro"] == 1 / 3
        assert agg["recall_by_predicate"]["competesWith"] == {
            "matched": 1,
            "total": 2,
            "recall": 0.5,
        }
        assert agg["recall_by_predicate"]["hasSegment"]["recall"] == 0.0
        # 이 픽스처는 이미 정식 표기(competesWith·hasSegment는 별칭이 없다)라 canonical == raw
        assert agg["gold_edge_items_canonical"] == 2
        assert agg["kg_edge_recall_gold_only_canonical"] == 0.25
        assert agg["recall_by_predicate_canonical"]["competesWith"] == {
            "matched": 1,
            "total": 2,
            "recall": 0.5,
        }

    def test_empty(self):
        agg = aggregate_l3_extended([])
        assert agg["kg_edge_recall_gold_only"] is None
        assert agg["kg_edge_recall_all"] is None
        assert agg["kg_edge_recall_gold_only_canonical"] is None
