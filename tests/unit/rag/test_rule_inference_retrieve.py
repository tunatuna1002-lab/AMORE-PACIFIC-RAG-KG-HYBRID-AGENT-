"""v4 검색에서 규칙 추론을 증거 카드로 살린다 (트랙 3-B, 설계 E3, 결함 F7)

실제 HybridRetriever(임시 KG·운영 스키마 SQLite·규칙 추론기·어댑터·계약)와 가짜 문서
검색기만 쓴다. 규칙 입력은 DB 수치(metric 카드)·KG 관계(relation 카드)이고,
``current_metrics``(대시보드 JSON)는 추론 입력이 아니다.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from src.domain.entities.evidence import EvidenceKind
from src.rag.hybrid_retriever import HybridRetriever
from src.rag.metric_facts import MetricFactsProvider

from .evidence_pipeline_fixtures import (
    AS_OF,
    OPERATIONAL_SCHEMA,
    FakeDocRetriever,
    make_kg,
)

RULE_EVALUATION_KEYS = {
    "combinations",
    "evaluated",
    "fired",
    "non_fire_top",
    "non_fire_counts_by_kind",
}


def _db(
    tmp_path: Path,
    brands: list[tuple],
    markets: list[tuple],
    products: list[tuple] = (),
) -> Path:
    """brands: (category, brand, sos%, product_count, brand_avg_rank, cpi, avg_rating_gap)
    markets: (category, hhi, category_avg_price, category_avg_rating)
    products: (category, rank, asin, name, brand, price, rating, reviews)"""
    path = tmp_path / "amore_data.db"
    conn = sqlite3.connect(path)
    conn.executescript(OPERATIONAL_SCHEMA)
    conn.executemany(
        "INSERT INTO brand_metrics (snapshot_date, category_id, brand, sos, product_count,"
        " brand_avg_rank, cpi, avg_rating_gap) VALUES (?,?,?,?,?,?,?,?)",
        [(AS_OF, *row) for row in brands],
    )
    conn.executemany(
        "INSERT INTO market_metrics (snapshot_date, category_id, hhi, category_avg_price,"
        " category_avg_rating) VALUES (?,?,?,?,?)",
        [(AS_OF, *row) for row in markets],
    )
    conn.executemany(
        "INSERT INTO raw_data (snapshot_date, category_id, rank, asin, product_name, brand,"
        " price, rating, reviews_count) VALUES (?,?,?,?,?,?,?,?,?)",
        [(AS_OF, *row) for row in products],
    )
    conn.commit()
    conn.close()
    return path


def _retriever(tmp_path: Path, db_path: Path) -> HybridRetriever:
    return HybridRetriever(
        knowledge_graph=make_kg(tmp_path),
        doc_retriever=FakeDocRetriever(),
        metric_facts_provider=MetricFactsProvider(db_path, as_of=AS_OF),
    )


def _metric(ctx, predicate: str, subject: str, obj: str | None = None):
    (card,) = [
        c
        for c in ctx.evidence
        if c.kind == EvidenceKind.METRIC
        and c.predicate == predicate
        and c.subject == subject
        and c.object == obj
    ]
    return card


def _inference_cards(ctx, rule_name: str):
    return [c for c in ctx.evidence if c.source == f"rule:{rule_name}"]


@pytest.fixture
def dominant_db(tmp_path):
    return _db(
        tmp_path,
        brands=[
            ("lip_care", "LANEIGE", 18.0, 18, 30.5, None, None),
            ("lip_care", "eos", 9.0, 9, 40.0, None, None),
        ],
        markets=[("lip_care", 0.10, 12.0, 4.5)],
    )


# ----------------------------------------------------------------------
# 게이트: DB SoS 18% · HHI 0.10 → market_dominance_fragmented
# ----------------------------------------------------------------------


async def test_gate_market_dominance_fires_from_db_cards(tmp_path, dominant_db):
    retriever = _retriever(tmp_path, dominant_db)

    ctx = await retriever.retrieve("LANEIGE Lip Care 시장 포지션은?")

    assert "retrieval_error" not in ctx.metadata
    assert "market_dominance_fragmented" in [r.rule_name for r in ctx.inferences]

    sos = _metric(ctx, "sos", "laneige", "lip_care")
    hhi = _metric(ctx, "hhi", "lip_care")
    assert sos.value == pytest.approx(0.18)
    (card,) = _inference_cards(ctx, "market_dominance_fragmented")
    assert card.kind == EvidenceKind.INFERENCE
    assert sos.id in card.derived_from and hhi.id in card.derived_from
    assert list(card.derived_from) == sorted(set(card.derived_from))
    assert card.value == "dominant_in_fragmented"
    assert card.id in [c.id for c in ctx.prompt_evidence]
    assert f"근거: {', '.join(card.derived_from)}" in ctx.combined_context

    rule_evaluation = ctx.metadata["rule_evaluation"]
    assert set(rule_evaluation) == RULE_EVALUATION_KEYS
    assert "market_dominance_fragmented" in rule_evaluation["fired"]
    assert rule_evaluation["combinations"] == [["laneige", "lip_care"]]


async def test_every_inference_basis_card_is_in_evidence(tmp_path, dominant_db):
    retriever = _retriever(tmp_path, dominant_db)

    ctx = await retriever.retrieve("LANEIGE Lip Care 시장 포지션은?")

    evidence_ids = {c.id for c in ctx.evidence}
    inference_cards = [c for c in ctx.evidence if c.kind == EvidenceKind.INFERENCE]
    assert inference_cards
    for card in inference_cards:
        assert card.derived_from, card
        assert set(card.derived_from) <= evidence_ids, card


async def test_rule_name_reaches_inference_results(tmp_path, dominant_db):
    # eval/runner.py _extract_l4_trace는 HybridContext.inferences[*].to_dict()["rule_name"]을 읽는다
    retriever = _retriever(tmp_path, dominant_db)

    ctx = await retriever.retrieve("LANEIGE Lip Care 시장 포지션은?")

    rule_names = [inf.to_dict()["rule_name"] for inf in ctx.inferences]
    assert "market_dominance_fragmented" in rule_names
    (result,) = [r for r in ctx.inferences if r.rule_name == "market_dominance_fragmented"]
    assert result.evidence["derived_from"] == sorted(result.evidence["derived_from"])


async def test_current_metrics_is_not_an_inference_input(tmp_path, dominant_db):
    retriever = _retriever(tmp_path, dominant_db)
    dashboard = {
        "summary": {"laneige_sos_by_category": {"lip_care": 0.01}},
        "brand_metrics": [{"brand_name": "laneige", "share_of_shelf": 0.01, "avg_rank": 3}],
        "market_metrics": [{"category_id": "lip_care", "hhi": 0.9}],
    }

    with_json = await retriever.retrieve(
        "LANEIGE Lip Care 시장 포지션은?", current_metrics=dashboard
    )
    without = await _retriever(tmp_path / "b", dominant_db).retrieve(
        "LANEIGE Lip Care 시장 포지션은?"
    )

    assert [r.rule_name for r in with_json.inferences] == [r.rule_name for r in without.inferences]
    assert "strong_avg_rank" not in with_json.metadata["rule_evaluation"]["fired"]


# ----------------------------------------------------------------------
# 입력 결측: 브랜드가 Top100에 없다
# ----------------------------------------------------------------------


async def test_absent_brand_does_not_fire_and_records_missing_sos(tmp_path):
    db = _db(
        tmp_path,
        brands=[("lip_care", "eos", 20.0, 20, 30.0, None, None)],
        markets=[("lip_care", 0.10, 12.0, 4.5)],
    )
    retriever = _retriever(tmp_path, db)

    ctx = await retriever.retrieve("LANEIGE Lip Care 시장 포지션은?")

    assert _metric(ctx, "present_in_top100", "laneige", "lip_care").value is False
    rule_evaluation = ctx.metadata["rule_evaluation"]
    assert "market_dominance_fragmented" not in rule_evaluation["fired"]
    assert not _inference_cards(ctx, "market_dominance_fragmented")
    assert dict(rule_evaluation["non_fire_top"]).get("missing_input:sos", 0) >= 1
    assert rule_evaluation["non_fire_counts_by_kind"]["missing_input"] >= 1


# ----------------------------------------------------------------------
# cpi·avg_rating_gap 전달 → price_quality_mismatch (rg022와 같은 값)
# ----------------------------------------------------------------------


async def test_cpi_and_rating_gap_fire_price_quality_mismatch(tmp_path):
    db = _db(
        tmp_path,
        brands=[
            ("face_powder", "COVERGIRL", 8.0, 8, 49.75, 59.6, -0.043),
            ("face_powder", "LANEIGE", 1.0, 1, 94.0, 226.7, -0.281),
        ],
        markets=[("face_powder", 0.0527, 17.2, 4.4)],
    )
    retriever = _retriever(tmp_path, db)

    ctx = await retriever.retrieve("LANEIGE Face Powder 가격 경쟁력은?")

    cpi = _metric(ctx, "cpi", "laneige", "face_powder")
    gap = _metric(ctx, "avg_rating_gap", "laneige", "face_powder")
    rank = _metric(ctx, "brand_avg_rank", "laneige", "face_powder")
    assert (cpi.value, gap.value, rank.value) == (226.7, -0.281, 94.0)
    (card,) = _inference_cards(ctx, "price_quality_mismatch")
    assert {cpi.id, gap.id} <= set(card.derived_from)
    assert "price_quality_mismatch" in ctx.metadata["rule_evaluation"]["fired"]
    # 새 수치 카드는 질의 브랜드 그룹으로 프롬프트에 실린다
    prompt_ids = {c.id for c in ctx.prompt_evidence}
    assert {cpi.id, gap.id, rank.id} <= prompt_ids
    # 상위 브랜드(질의 브랜드 아님)의 cpi는 조회하지 않는다
    assert not [c for c in ctx.evidence if c.predicate == "cpi" and c.subject == "covergirl"]


# ----------------------------------------------------------------------
# 발화가 프롬프트 상한(5)보다 많다 — 발화 전부가 inferences·카드에 남는다
# ----------------------------------------------------------------------


@pytest.fixture
def many_fire_db(tmp_path):
    # 스냅샷 rg014(LANEIGE Face Powder)와 같은 모양: 진입 기회·평균 순위·평점 우위·프리미엄 가격·
    # 프리미엄 방어·Top3 + KG 소유 관계(make_kg) → 발화 6개 이상
    return _db(
        tmp_path,
        brands=[
            ("face_powder", "LANEIGE", 1.0, 1, 12.0, 170.0, 0.1),
            ("face_powder", "COVERGIRL", 8.0, 8, 49.75, 59.6, -0.043),
        ],
        markets=[("face_powder", 0.0527, 17.2, 4.4)],
        products=[
            (
                "face_powder",
                2,
                "B0NEO00001",
                "LANEIGE Neo Blurring Powder",
                "LANEIGE",
                25.0,
                4.5,
                900,
            )
        ],
    )


async def test_every_fired_rule_is_an_inference_and_card_beyond_prompt_cap(tmp_path, many_fire_db):
    retriever = _retriever(tmp_path, many_fire_db)

    ctx = await retriever.retrieve("LANEIGE Face Powder 경쟁력은?")

    fired = ctx.metadata["rule_evaluation"]["fired"]
    assert len(fired) > 5  # 프롬프트 추론 카드 상한(PROMPT_MAX_PER_KIND[INFERENCE])보다 많다
    assert "category_entry_opportunity" in fired
    # 평가 applied_rules(eval/runner.py _extract_l4_trace)는 inferences의 rule_name을 읽는다
    assert sorted({r.rule_name for r in ctx.inferences}) == sorted(fired)
    assert ctx.metadata["inferences_count"] == len(ctx.inferences)
    card_rules = sorted(
        c.metadata["rule_name"] for c in ctx.evidence if c.kind == EvidenceKind.INFERENCE
    )
    assert card_rules == sorted(fired)
    prompt_inferences = [c for c in ctx.prompt_evidence if c.kind == EvidenceKind.INFERENCE]
    assert len(prompt_inferences) == 5


async def test_fusion_gets_string_insight_for_rule_results(tmp_path, many_fire_db, monkeypatch):
    # FusionInferenceResult.insight는 str이다. InferenceResult.conclusion(결론 dict)이 생긴 뒤
    # getattr(inf, "conclusion", ...)이 dict를 넘기던 회귀를 막는다
    from src.rag.confidence_fusion import ConfidenceFusion

    seen: list = []
    real_fuse = ConfidenceFusion.fuse

    def spy(self, *args, **kwargs):
        seen.extend(kwargs.get("ontology_results") or [])
        return real_fuse(self, *args, **kwargs)

    monkeypatch.setattr(ConfidenceFusion, "fuse", spy)
    retriever = _retriever(tmp_path, many_fire_db)

    ctx = await retriever.retrieve("LANEIGE Face Powder 경쟁력은?")

    assert len(seen) == len(ctx.inferences) > 0
    assert all(isinstance(result.insight, str) for result in seen)


# ----------------------------------------------------------------------
# 추론 off (ablation no-ontology)
# ----------------------------------------------------------------------


async def test_reasoner_flags_off_skip_rule_evaluation(tmp_path, dominant_db, monkeypatch):
    monkeypatch.setenv("FF_REASONER_USE_UNIFIED_REASONER", "false")
    monkeypatch.setenv("FF_REASONER_USE_OWL_REASONER", "false")
    retriever = _retriever(tmp_path, dominant_db)

    ctx = await retriever.retrieve("LANEIGE Lip Care 시장 포지션은?")

    assert "retrieval_error" not in ctx.metadata
    assert ctx.inferences == []
    assert not [c for c in ctx.evidence if c.kind == EvidenceKind.INFERENCE]
    assert "rule_evaluation" not in ctx.metadata
    # 수치 카드는 그대로 실린다
    assert _metric(ctx, "sos", "laneige", "lip_care")
