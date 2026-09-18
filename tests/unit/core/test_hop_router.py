"""
홉 수 기반 난이도 라우터 (트랙 5-C, 설계 E7)
=============================================

라우터는 "이 질문에 답하려면 서로 의존하는 조회를 몇 번 해야 하는가"(홉 수)로 경로를
정한다. 1홉 이하면 파이프라인(direct/decide), 2홉 이상이면 ReAct다.

홉 = 앞 단계의 결과가 다음 단계의 입력이 되는 조회 한 번. 단계는 네 가지다:

    entity_resolution  대상 엔티티가 이름이 아니라 설명으로 지목돼 먼저 찾아야 한다
    relation           KG 관계를 타야 대상 집합/대상이 정해진다
    metric             DB 수치를 조회한다
    judgement          수치·관계 위에 규칙/인과 판정을 얹는다

라벨
----
아래 표의 ``hops``는 사람이 직접 매긴 정답이다. 출처는 ``eval/data/golden/typed/``의
multihop.jsonl(2홉 이상이어야 하는 질문)과 numeric.jsonl(대부분 1홉)이며, 질문 원문이
골든셋과 같은지 자체 검증한다(``test_labelled_questions_match_the_golden_files``).

``split``:
    ``design``  규칙을 만들 때 참고한 질문 (in-sample, 40문항)
    ``holdout`` 규칙을 다 쓴 뒤에 처음 돌려본 질문 (out-of-sample, 18문항)

두 집합 모두 정확도를 따로 보고한다 — in-sample 정확도만으로는 규칙이 일반화되는지 알 수 없다.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest

from src.core.router import (
    HOP_THRESHOLD,
    STAGE_ENTITY,
    STAGE_JUDGEMENT,
    STAGE_METRIC,
    STAGE_RELATION,
    HopRouter,
    RouteDecision,
)


@dataclass(frozen=True)
class Labelled:
    qid: str
    source: str  # multihop | numeric
    question: str
    hops: int  # 사람이 매긴 정답 홉 수
    split: str  # design | holdout

    @property
    def expects_react(self) -> bool:
        return self.hops >= HOP_THRESHOLD


LABELS: tuple[Labelled, ...] = (
    # ── multihop.jsonl (설계에 참고, 20문항) ────────────────────────────
    Labelled(
        "lg099", "multihop", "K-Beauty 브랜드 중 Amazon US에서 가장 강한 브랜드는?", 2, "design"
    ),
    Labelled("lg148", "multihop", "LANEIGE 제품이 속한 카테고리의 HHI는?", 3, "design"),
    Labelled("lg149", "multihop", "LANEIGE가 가장 강한 카테고리의 시장집중도는?", 2, "design"),
    Labelled(
        "lg150", "multihop", "LANEIGE의 대표 제품이 속한 카테고리에서 1위 브랜드는?", 3, "design"
    ),
    Labelled("lg151", "multihop", "K-Beauty 1위 브랜드의 주력 카테고리 HHI는?", 3, "design"),
    Labelled(
        "lg152",
        "multihop",
        "LANEIGE Lip Sleeping Mask와 같은 카테고리 경쟁사의 평균 가격은?",
        2,
        "design",
    ),
    Labelled(
        "lg153",
        "multihop",
        "LANEIGE Neo Cushion이 속한 카테고리의 K-Beauty 브랜드 SoS 합계는?",
        3,
        "design",
    ),
    Labelled(
        "lg154",
        "multihop",
        "COSRX의 가장 인기 있는 제품의 카테고리와 해당 카테고리 SoS는?",
        3,
        "design",
    ),
    Labelled(
        "lg155", "multihop", "LANEIGE가 Skin Care에서 COSRX 수준의 SoS를 달성하려면?", 2, "design"
    ),
    Labelled(
        "lg156", "multihop", "Lip Sleeping Mask의 리뷰 수가 SoS에 기여하는 경로는?", 2, "design"
    ),
    Labelled(
        "lg158",
        "multihop",
        "LANEIGE 브랜드의 모회사와 해당 기업의 다른 브랜드 Amazon 현황은?",
        3,
        "design",
    ),
    Labelled(
        "lg160",
        "multihop",
        "LANEIGE의 가장 비싼 제품과 그 제품이 속한 카테고리 평균 가격 대비 CPI는?",
        3,
        "design",
    ),
    Labelled(
        "lg161",
        "multihop",
        "LANEIGE 카테고리에서 1위 브랜드의 SoS와 LANEIGE의 격차는?",
        2,
        "design",
    ),
    Labelled("lg192", "multihop", "LANEIGE Lip Care 점유율이 높은 이유는?", 2, "design"),
    Labelled("lg193", "multihop", "COSRX가 잘 팔리는 이유와 카테고리 집중도는?", 2, "design"),
    Labelled(
        "lg195", "multihop", "TIRTIR가 Face Powder에서 LANEIGE보다 SoS가 높은 이유는?", 2, "design"
    ),
    Labelled("lg196", "multihop", "LANEIGE가 경쟁이 가장 심한 카테고리는?", 2, "design"),
    Labelled(
        "mh001",
        "multihop",
        "아모레퍼시픽 그룹 소속 브랜드 중 2026-08-31 Face Powder Top 100에 제품이 있는 브랜드와 각 브랜드의 SoS는?",
        2,
        "design",
    ),
    Labelled(
        "mh003",
        "multihop",
        "2026-08-31 기준 LANEIGE 제품이 Top 100에 있는 카테고리 중 시장 집중도(HHI)가 가장 높은 카테고리와 그 HHI는?",
        3,
        "design",
    ),
    Labelled(
        "mh005",
        "multihop",
        "COSRX와 같은 그룹에 속한 브랜드(COSRX 포함) 중 2026-08-31 Face Powder Top 100에 제품이 있는 브랜드는 몇 개인가요?",
        2,
        "design",
    ),
    # ── numeric.jsonl (설계에 참고, 20문항) ─────────────────────────────
    Labelled("lg048", "numeric", "현재 LANEIGE의 Lip Care 카테고리 SoS는 얼마인가요?", 1, "design"),
    Labelled("lg049", "numeric", "Lip Care 카테고리 HHI 현황은?", 1, "design"),
    Labelled("lg050", "numeric", "LANEIGE 시장 지표 종합 분석을 해주세요", 1, "design"),
    # 경쟁사를 먼저 찾아야 가격을 비교할 수 있다 — numeric.jsonl이지만 2홉이다
    Labelled("lg051", "numeric", "LANEIGE 경쟁사 대비 가격 경쟁력은?", 2, "design"),
    Labelled("lg055", "numeric", "Lip Care와 Lip Makeup 카테고리 HHI를 비교해주세요", 1, "design"),
    Labelled("lg057", "numeric", "BIODANCE의 Lip Care 점유율은 얼마인가요?", 1, "design"),
    Labelled(
        "lg060", "numeric", "Beauty & Personal Care 전체 카테고리 SoS 상위 브랜드는?", 1, "design"
    ),
    Labelled("lg066", "numeric", "Beauty of Joseon의 Skin Care SoS는?", 1, "design"),
    Labelled("lg068", "numeric", "Face Powder 카테고리 SoS 현황은?", 1, "design"),
    Labelled("lg071", "numeric", "LANEIGE Lip Sleeping Mask 현재 순위는?", 1, "design"),
    Labelled("lg072", "numeric", "Lip Care 카테고리 Top 10 제품 목록은?", 1, "design"),
    Labelled("lg073", "numeric", "LANEIGE 제품 평균 가격은?", 1, "design"),
    Labelled("lg075", "numeric", "평점 4.5 이상인 LANEIGE 제품은?", 1, "design"),
    Labelled("lg076", "numeric", "리뷰 수가 가장 많은 LANEIGE 제품은?", 1, "design"),
    Labelled("lg083", "numeric", "LANEIGE 전체 제품 중 가장 높은 BSR은?", 1, "design"),
    Labelled(
        "lg086", "numeric", "Rare Beauty의 Lip Makeup 카테고리 주요 제품 순위는?", 1, "design"
    ),
    Labelled("lg095", "numeric", "Face Powder 카테고리 Top 5 제품은?", 1, "design"),
    Labelled("lg113", "numeric", "Burt's Bees와 LANEIGE의 Lip Care 시장 비교는?", 1, "design"),
    Labelled("lg162", "numeric", "라에니즈 립케어 순위 알려줘", 1, "design"),
    Labelled("lg164", "numeric", "show me laneige sos", 1, "design"),
    # ── holdout: 규칙을 다 쓴 뒤 처음 돌린 질문 (18문항) ────────────────
    Labelled(
        "mh002",
        "multihop",
        "아모레퍼시픽 그룹 소속 브랜드 중 2026-08-31 Lip Makeup Top 100에 제품이 있는 브랜드와 각 브랜드의 SoS는?",
        2,
        "holdout",
    ),
    Labelled(
        "mh004",
        "multihop",
        "2026-08-31 기준 LANEIGE 제품의 평균 순위가 가장 좋은 카테고리에서 SoS 1위 브랜드와 그 SoS는?",
        3,
        "holdout",
    ),
    Labelled(
        "mh006",
        "multihop",
        "Sulwhasoo와 같은 그룹에 속한 브랜드(Sulwhasoo 포함) 중 2026-08-31 Lip Makeup Top 100에 제품이 있는 브랜드는 몇 개인가요?",
        2,
        "holdout",
    ),
    Labelled(
        "mh007",
        "multihop",
        "2026-08-31 기준 Face Powder Top 100에 제품이 있는 아모레퍼시픽 그룹 브랜드 중 가장 높은 순위의 제품을 가진 브랜드와 그 순위는?",
        3,
        "holdout",
    ),
    Labelled(
        "mh008",
        "multihop",
        "2026-08-31 기준 innisfree 제품이 Top 100에 있는 카테고리에서 SoS 1위 브랜드와 그 SoS는?",
        3,
        "holdout",
    ),
    Labelled(
        "mh009",
        "multihop",
        "2026-08-31 기준 Face Powder SoS 1위 브랜드의 최고 순위 제품은 무엇이고 가격은 얼마인가요?",
        2,
        "holdout",
    ),
    Labelled(
        "mh010",
        "multihop",
        "2026-08-31 기준 Skin Care SoS 1위 브랜드의 최고 순위 제품은 무엇이고 가격은 얼마인가요?",
        2,
        "holdout",
    ),
    Labelled("lg070", "numeric", "MEDICUBE의 Skin Care SoS와 HHI 영향은?", 1, "holdout"),
    Labelled("lg074", "numeric", "경쟁 제품 대비 Lip Sleeping Mask 가격은?", 2, "holdout"),
    Labelled("lg078", "numeric", "최근 Lip Care 카테고리 신규 진입 제품은?", 1, "holdout"),
    Labelled("lg079", "numeric", "LANEIGE Water Bank Blue Hyaluronic 제품 ASIN은?", 1, "holdout"),
    Labelled("lg087", "numeric", "e.l.f. Cosmetics의 Face Powder 주요 제품 순위는?", 1, "holdout"),
    Labelled("lg089", "numeric", "COSRX Snail Mucin 96% Essence 순위와 특징은?", 1, "holdout"),
    Labelled("lg091", "numeric", "LANEIGE Lip Sleeping Mask 향별 순위 비교는?", 1, "holdout"),
    Labelled("lg134", "numeric", "Amazon US 뷰티 시장 전체 HHI 수준은?", 1, "holdout"),
    Labelled("lg169", "numeric", "LANEIGE SoS가 100%인가요?", 1, "holdout"),
    Labelled("lg170", "numeric", "라네즈 SoS가 어제랑 똑같아?", 1, "holdout"),
    Labelled("lg171", "numeric", "랜지 립스틱 순위가 왜 이렇게 낮아?", 2, "holdout"),
)

# 라우팅 정확도 하한 (스테이지 게이트). 라벨은 사람이 매긴 것이고, 규칙은 design 집합만
# 보고 만들었다. holdout에서도 이 하한을 넘어야 규칙이 일반화된 것으로 본다.
MIN_ROUTE_ACCURACY = 0.9
MIN_HOP_ACCURACY = 0.8

GOLDEN_DIR = Path(__file__).resolve().parents[3] / "eval" / "data" / "golden" / "typed"


@pytest.fixture
def router() -> HopRouter:
    return HopRouter()


# =============================================================================
# 라벨 자체 검증: 질문 원문이 골든셋과 같아야 한다
# =============================================================================


def test_labelled_questions_match_the_golden_files():
    """라벨의 질문은 eval/data/golden/typed/에서 그대로 가져온 것이다."""
    by_source: dict[str, dict[str, str]] = {}
    for source in ("multihop", "numeric"):
        path = GOLDEN_DIR / f"{source}.jsonl"
        if not path.exists():  # 평가 데이터가 없는 체크아웃에서는 건너뛴다
            pytest.skip(f"golden file missing: {path}")
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
        by_source[source] = {row["id"]: row["question"] for row in rows}

    for label in LABELS:
        assert label.qid in by_source[label.source], label.qid
        assert by_source[label.source][label.qid] == label.question, label.qid


def test_labels_cover_both_splits_and_enough_questions():
    design = [x for x in LABELS if x.split == "design"]
    holdout = [x for x in LABELS if x.split == "holdout"]
    assert len(design) >= 20
    assert len(holdout) >= 10
    assert len({x.qid for x in LABELS}) == len(LABELS)


# =============================================================================
# 게이트 지표: 라우팅 정확도 / 홉 수 정확도
# =============================================================================


def _score(router: HopRouter, labels) -> tuple[int, int, int, list[str]]:
    route_hits = 0
    hop_hits = 0
    misses: list[str] = []
    for label in labels:
        decision = router.analyze(label.question)
        if decision.use_react == label.expects_react:
            route_hits += 1
        else:
            misses.append(
                f"{label.qid}: 정답 {label.hops}홉 → {'react' if label.expects_react else 'pipeline'}, "
                f"라우터 {decision.hops}홉 → {'react' if decision.use_react else 'pipeline'} "
                f"({decision.reason})"
            )
        if decision.hops == label.hops:
            hop_hits += 1
    return route_hits, hop_hits, len(list(labels)), misses


@pytest.mark.parametrize("split", ["design", "holdout"])
def test_router_route_accuracy_against_manual_labels(router, split, capsys):
    labels = [x for x in LABELS if x.split == split]
    route_hits, hop_hits, total, misses = _score(router, labels)

    with capsys.disabled():
        print(
            f"\n[router/{split}] 라우팅 {route_hits}/{total} "
            f"({route_hits / total:.0%}), 홉 정확 일치 {hop_hits}/{total} ({hop_hits / total:.0%})"
        )
        for miss in misses:
            print(f"  miss {miss}")

    assert route_hits / total >= MIN_ROUTE_ACCURACY, misses
    assert hop_hits / total >= MIN_HOP_ACCURACY


def test_every_multihop_label_routes_to_react(router):
    """multihop.jsonl 질문은 전부 2홉 이상으로 판정돼야 한다."""
    wrong = [
        x.qid for x in LABELS if x.source == "multihop" and not router.analyze(x.question).use_react
    ]
    assert wrong == []


# =============================================================================
# 규칙 단위 테스트
# =============================================================================


def test_single_metric_lookup_is_one_hop(router):
    decision = router.analyze("현재 LANEIGE의 Lip Care 카테고리 SoS는 얼마인가요?")
    assert decision.hops == 1
    assert decision.stages == (STAGE_METRIC,)
    assert decision.use_react is False
    assert decision.basis == "rules"


def test_relation_then_metric_is_two_hops(router):
    decision = router.analyze("LANEIGE 경쟁사 대비 가격 경쟁력은?")
    assert decision.stages == (STAGE_RELATION, STAGE_METRIC)
    assert decision.hops == 2
    assert decision.use_react is True


def test_judgement_on_top_of_metric_is_two_hops(router):
    decision = router.analyze("LANEIGE Lip Care 점유율이 높은 이유는?")
    assert STAGE_JUDGEMENT in decision.stages
    assert decision.hops == 2
    assert decision.use_react is True


def test_indirect_entity_adds_a_hop(router):
    decision = router.analyze("LANEIGE 제품이 속한 카테고리의 HHI는?")
    assert decision.stages == (STAGE_ENTITY, STAGE_RELATION, STAGE_METRIC)
    assert decision.hops == 3


def test_comparison_of_two_named_entities_stays_one_hop(router):
    """'비교'는 홉이 아니다 — 두 조회가 서로 의존하지 않는다.

    옛 ``_is_complex_query``는 '비교/분석' 키워드만으로 ReAct로 보냈다.
    """
    decision = router.analyze("Lip Care와 Lip Makeup 카테고리 HHI를 비교해주세요")
    assert decision.hops == 1
    assert decision.use_react is False


def test_empty_query_is_one_hop_and_never_react(router):
    for query in ("", "   ", "?"):
        decision = router.analyze(query)
        assert decision.hops == 1
        assert decision.use_react is False


def test_decision_exposes_trace_fields(router):
    decision = router.analyze("LANEIGE 제품이 속한 카테고리의 HHI는?")
    trace = decision.to_trace()
    assert trace["hops"] == 3
    assert trace["router_basis"] == "rules"
    assert trace["router_stages"] == list(decision.stages)
    assert isinstance(trace["router_reason"], str) and trace["router_reason"]
    assert trace["router_route"] == "react"


def test_route_decision_is_immutable(router):
    decision = router.analyze("LANEIGE SoS는?")
    assert isinstance(decision, RouteDecision)
    with pytest.raises(AttributeError):
        decision.hops = 9  # type: ignore[misc]


# =============================================================================
# LLM 폴백: 규칙이 아무 표지도 못 찾은 질문에만, 캐시하고, 근거를 남긴다
# =============================================================================


def _llm_reply(payload: dict) -> object:
    from types import SimpleNamespace

    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(payload)))]
    )


@pytest.mark.asyncio
async def test_llm_fallback_is_off_by_default_even_for_unmarked_queries(monkeypatch):
    """폴백이 꺼져 있으면 LLM을 부르지 않는다 (기본값 — 평가에서 비용이 새지 않게)."""
    from unittest.mock import AsyncMock

    llm = AsyncMock()
    monkeypatch.setattr("litellm.acompletion", llm)

    decision = await HopRouter(llm_fallback=False).route("그거 어떻게 됐어")

    assert decision.basis == "rules"
    assert decision.hops == 1
    llm.assert_not_awaited()


@pytest.mark.asyncio
async def test_llm_fallback_runs_only_when_rules_find_nothing(monkeypatch):
    from unittest.mock import AsyncMock

    llm = AsyncMock(return_value=_llm_reply({"hops": 3, "reason": "설명이 필요"}))
    monkeypatch.setattr("litellm.acompletion", llm)
    router = HopRouter(llm_fallback=True)

    unmarked = await router.route("그거 어떻게 됐어")
    assert unmarked.basis == "llm"
    assert unmarked.hops == 3
    assert unmarked.use_react is True
    assert "설명이 필요" in unmarked.reason
    assert llm.await_count == 1

    # 표지가 있는 질문은 규칙으로 끝난다 — LLM을 더 부르지 않는다
    marked = await router.route("LANEIGE Lip Care SoS는?")
    assert marked.basis == "rules"
    assert llm.await_count == 1


@pytest.mark.asyncio
async def test_llm_fallback_is_cached_per_question(monkeypatch):
    from unittest.mock import AsyncMock

    llm = AsyncMock(return_value=_llm_reply({"hops": 2, "reason": "두 단계"}))
    monkeypatch.setattr("litellm.acompletion", llm)
    router = HopRouter(llm_fallback=True)

    first = await router.route("그거 어떻게 됐어")
    second = await router.route("  그거   어떻게 됐어  ")

    assert first.basis == "llm"
    assert second.basis == "llm_cache"
    assert second.hops == first.hops
    assert llm.await_count == 1


@pytest.mark.asyncio
async def test_llm_fallback_failure_falls_back_to_rules(monkeypatch):
    from unittest.mock import AsyncMock

    llm = AsyncMock(side_effect=RuntimeError("no network"))
    monkeypatch.setattr("litellm.acompletion", llm)

    decision = await HopRouter(llm_fallback=True).route("그거 어떻게 됐어")

    assert decision.basis == "rules"
    assert decision.hops == 1


@pytest.mark.asyncio
async def test_llm_fallback_clamps_out_of_range_hops(monkeypatch):
    from unittest.mock import AsyncMock

    llm = AsyncMock(return_value=_llm_reply({"hops": 99, "reason": "과장"}))
    monkeypatch.setattr("litellm.acompletion", llm)

    decision = await HopRouter(llm_fallback=True).route("그거 어떻게 됐어")

    assert decision.hops == 4


@pytest.mark.asyncio
async def test_router_flag_default_is_off(monkeypatch):
    """플래그 기본값이 OFF여야 서비스 경로에서 LLM 호출이 늘지 않는다."""
    from src.infrastructure.feature_flags import FeatureFlags

    monkeypatch.delenv("FF_ROUTER_USE_LLM_FALLBACK", raising=False)
    FeatureFlags.reset_instance()
    try:
        assert FeatureFlags.get_instance().use_router_llm_fallback() is False
    finally:
        FeatureFlags.reset_instance()


# =============================================================================
# 표본 밖 분포 확인: 라벨이 없는 typed 셋에서 경로가 어떻게 갈리는지
# =============================================================================


def test_router_distribution_on_unlabelled_typed_sets(router, capsys):
    """relation/rule 셋은 라벨이 없다 — 분포만 기록한다 (회귀 감시용 상한/하한만 건다)."""
    summary: dict[str, tuple[int, int]] = {}
    for name in ("relation", "rule", "numeric", "multihop"):
        path = GOLDEN_DIR / f"{name}.jsonl"
        if not path.exists():
            pytest.skip(f"golden file missing: {path}")
        questions = [
            json.loads(line)["question"]
            for line in path.read_text(encoding="utf-8").splitlines()
            if line
        ]
        react = sum(1 for q in questions if router.analyze(q).use_react)
        summary[name] = (react, len(questions))

    with capsys.disabled():
        print("\n[router] typed 셋 ReAct 비율")
        for name, (react, total) in summary.items():
            print(f"  {name}: {react}/{total} ({react / total:.0%})")

    # multihop은 전부 ReAct여야 하고, 나머지 유형은 대부분 파이프라인이어야 한다.
    # rule 셋("규칙 X를 충족하나요")은 apply_rules 한 번이면 끝나므로 ReAct가 아니다.
    assert summary["multihop"][0] == summary["multihop"][1]
    for name in ("numeric", "relation", "rule"):
        assert summary[name][0] / summary[name][1] <= 0.25, name
