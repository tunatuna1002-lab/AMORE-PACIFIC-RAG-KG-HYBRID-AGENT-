"""질의 경로 온톨로지 추론 (트랙 O3, 플래그 ``ontology.use_class_reasoning`` ON) [2026-09 사후].

- 그룹·클래스·자매·세그먼트 전개와 12개 상한·잘림 카드 (결정적 순서)
- 정적 정의 사실 카드 (출처 ``ontology:registry``, ``as_of`` = 온톨로지 as_of, 정식 술어)
- 닫힌 세계 부정(OE4): 등록부 브랜드만 "아님", 등록부 밖은 "모름"
- 읽을 때 술어 정식화 (``ownedBy`` → ``ownedByGroup``, ``hasPosition`` 분리)
- 카테고리 포함(OE3): 조회 범위만 넓히고 수치는 자기 카테고리 그대로 (합산·환산 없음)
- 카드 수 추적 (``metadata["ontology"]``)

OFF 동작 불변은 ``test_class_reasoning_off_characterization.py``가 고정한다.
"""

from __future__ import annotations

import sqlite3
from typing import Any

import pytest

from src.domain.entities.evidence import EvidenceKind
from src.ontology.ontology import get_ontology
from src.rag.evidence_adapters import (
    EXCLUDED_METADATA,
    EXCLUDED_NUMERIC,
    ONTOLOGY_SOURCE,
    EvidenceAdapter,
)
from src.rag.metric_facts import MetricFactsProvider
from src.rag.ontology_context import (
    EXPANSION_TRUNCATED,
    MAX_EXPANDED_BRANDS,
    MEMBERSHIP_UNKNOWN,
    NOT_OWNED_BY_GROUP,
    NOT_SIBLING_BRAND,
    apply_cap,
    plan_query,
    static_edges,
)

from .evidence_pipeline_fixtures import AS_OF
from .ontology_query_fixtures import QUERIES, make_ontology_db, make_ontology_retriever

FLAG = "FF_ONTOLOGY_USE_CLASS_REASONING"


@pytest.fixture
def onto():
    return get_ontology()


@pytest.fixture
def flag_on(monkeypatch):
    monkeypatch.setenv(FLAG, "true")


def _triples(edges: list[dict[str, Any]]) -> set[tuple[str, str, str | None]]:
    return {(e["subject"], e["predicate"], e["object"]) for e in edges}


def _retriever(tmp_path, entities: dict[str, Any]):
    retriever = make_ontology_retriever(tmp_path)
    retriever.entity_extractor.extract = lambda query, knowledge_graph=None: {
        k: list(v) for k, v in entities.items()
    }
    return retriever


def _onto_cards(cards) -> list:
    return [c for c in cards if c.source == ONTOLOGY_SOURCE]


# ----------------------------------------------------------------------
# 계획 (순수 함수)
# ----------------------------------------------------------------------


class TestPlanExpansion:
    def test_group_mentioned_as_brand_expands_to_real_members(self, onto):
        plan = apply_cap(plan_query(onto, {"brands": ["amorepacific"]}))
        assert plan.groups == ["amorepacific"]
        assert plan.mentioned_brands == []
        members = [b for b in onto.brands_in_group("amorepacific") if not onto.is_placeholder(b)]
        assert len(plan.expanded_brands) == MAX_EXPANDED_BRANDS
        assert sorted(plan.expanded_brands + plan.dropped_brands) == sorted(members)
        # 기본 순위는 이름 순 (결정적)
        assert plan.expanded_brands == sorted(plan.expanded_brands)

    def test_o2_groups_and_classes_keys(self, onto):
        plan = apply_cap(plan_query(onto, {"groups": ["아모레퍼시픽"], "classes": ["LuxuryBrand"]}))
        assert plan.groups == ["amorepacific"]
        assert plan.classes == ["LuxuryBrand"]
        assert plan.membership_query is True

    def test_class_expansion_uses_defining_predicate(self, onto):
        plan = apply_cap(plan_query(onto, {"classes": ["LuxuryBrand"]}))
        expected = set(onto.instances_of("LuxuryBrand"))
        assert set(plan.expanded_brands) == expected
        edges = _triples(static_edges(onto, plan))
        assert {(b, "hasSegment", "luxury") for b in expected} <= edges

    def test_non_brand_class_is_ignored(self, onto):
        plan = plan_query(onto, {"classes": ["Category", "Brand", "NoSuchClass"]})
        assert plan.classes == [] and plan.candidates == []

    def test_placeholders_never_expand_or_count_as_brands(self, onto):
        plan = apply_cap(plan_query(onto, {"brands": ["unknown", "fresh", "chi"]}))
        assert plan.mentioned_brands == [] and plan.expanded_brands == []
        assert plan.unknown_brands == []  # 가짜 브랜드는 "모름"도 아니다 — 브랜드가 아니다

    def test_group_with_member_brand_is_context_not_enumeration(self, onto):
        # "아모레퍼시픽 포트폴리오에서 COSRX의 세그먼트는?" (rl007)
        plan = apply_cap(plan_query(onto, {"brands": ["cosrx"], "groups": ["amorepacific"]}))
        assert plan.expanded_brands == [] and plan.dropped_brands == []

    def test_sibling_hint_with_single_anchor_expands_siblings(self, onto):
        # "COSRX와 같은 그룹 브랜드(COSRX 포함)" (mh005)
        plan = apply_cap(plan_query(onto, {"brands": ["COSRX"], "relations_hint": ["sibling"]}))
        assert plan.mentioned_brands == ["cosrx"]
        assert set(plan.expanded_brands) | set(plan.dropped_brands) == set(onto.siblings("cosrx"))
        assert len(plan.mentioned_brands) + len(plan.expanded_brands) == MAX_EXPANDED_BRANDS

    def test_sibling_hint_with_two_brands_is_pair_check(self, onto):
        plan = apply_cap(
            plan_query(onto, {"brands": ["LANEIGE", "Sulwhasoo"], "relations_hint": ["sibling"]})
        )
        assert plan.expanded_brands == []
        assert ("laneige", "siblingBrand", "sulwhasoo") in _triples(static_edges(onto, plan))

    def test_segment_hint_within_group(self, onto):
        # "아모레퍼시픽 포트폴리오에서 LANEIGE와 같은 세그먼트 브랜드" (rl018)
        plan = apply_cap(
            plan_query(
                onto,
                {"brands": ["laneige"], "groups": ["amorepacific"], "relations_hint": ["segment"]},
            )
        )
        premium_ap = {
            b
            for b, seg in onto.relations("hasSegment")
            if seg == "premium" and onto.group_of(b) == "amorepacific" and b != "laneige"
        }
        assert set(plan.expanded_brands) == premium_ap
        edges = _triples(static_edges(onto, plan))
        assert {(b, "hasSegment", "premium") for b in premium_ap} <= edges

    def test_cap_rank_key_orders_then_truncates_deterministically(self, onto):
        preferred = {"sulwhasoo", "tata_harper"}
        plan = apply_cap(
            plan_query(onto, {"groups": ["amorepacific"]}),
            rank_key=lambda b: (0 if b in preferred else 1,),
        )
        assert plan.expanded_brands[:2] == ["sulwhasoo", "tata_harper"]
        again = apply_cap(
            plan_query(onto, {"groups": ["amorepacific"]}),
            rank_key=lambda b: (0 if b in preferred else 1,),
        )
        assert (again.expanded_brands, again.dropped_brands) == (
            plan.expanded_brands,
            plan.dropped_brands,
        )

    def test_truncation_edge_lists_dropped_brands(self, onto):
        plan = apply_cap(plan_query(onto, {"groups": ["amorepacific"]}))
        truncated = [e for e in static_edges(onto, plan) if e["predicate"] == EXPANSION_TRUNCATED]
        assert len(truncated) == 1
        assert truncated[0]["subject"] == "amorepacific"
        assert truncated[0]["dropped"] == plan.dropped_brands
        assert truncated[0]["count"] == len(plan.dropped_brands)

    def test_no_truncation_edge_under_cap(self, onto):
        plan = apply_cap(plan_query(onto, {"classes": ["LuxuryBrand"]}))
        assert not [e for e in static_edges(onto, plan) if e["predicate"] == EXPANSION_TRUNCATED]


class TestStaticFacts:
    def test_mentioned_brand_profile(self, onto):
        plan = apply_cap(plan_query(onto, {"brands": ["COSRX"]}))
        assert _triples(static_edges(onto, plan)) == {
            ("cosrx", "ownedByGroup", "amorepacific"),
            ("cosrx", "hasSegment", "k_beauty"),
            ("cosrx", "originatesFrom", "south_korea"),
            ("cosrx", "acquiredIn", "2024"),
        }

    def test_unknown_origin_is_not_invented(self, onto):
        # 등록부에 원산지가 없는 브랜드(AESTURA)는 원산지 사실을 만들지 않는다 (§8)
        plan = apply_cap(plan_query(onto, {"brands": ["AESTURA"]}))
        assert not [e for e in static_edges(onto, plan) if e["predicate"] == "originatesFrom"]

    @pytest.mark.parametrize("other", ["TIRTIR", "Beauty of Joseon"])
    def test_closed_world_negative_for_registry_brands(self, onto, other):
        # rl015·rl016: "LANEIGE와 X는 같은 그룹 자매 브랜드인가?"
        plan = apply_cap(
            plan_query(onto, {"brands": ["LANEIGE", other], "relations_hint": ["sibling"]})
        )
        bid = onto.normalize_brand(other)
        edges = _triples(static_edges(onto, plan))
        assert (bid, NOT_OWNED_BY_GROUP, "amorepacific") in edges
        assert ("laneige", NOT_SIBLING_BRAND, bid) in edges
        assert ("laneige", "siblingBrand", bid) not in edges

    def test_negative_with_group_entity_instead_of_hint(self, onto):
        # 골드 엔티티 형태 (laneige, tirtir, amorepacific)
        plan = apply_cap(plan_query(onto, {"brands": ["laneige", "tirtir", "amorepacific"]}))
        edges = _triples(static_edges(onto, plan))
        assert ("tirtir", NOT_OWNED_BY_GROUP, "amorepacific") in edges

    def test_unknown_brand_is_unknown_never_not(self, onto):
        plan = apply_cap(
            plan_query(onto, {"brands": ["LANEIGE", "Zzyzx Beauty"], "relations_hint": ["sibling"]})
        )
        edges = static_edges(onto, plan)
        assert ("Zzyzx Beauty", MEMBERSHIP_UNKNOWN, "amorepacific") in _triples(edges)
        assert not [
            e for e in edges if e["subject"] == "Zzyzx Beauty" and e["predicate"].startswith("not")
        ]

    def test_no_negative_cards_without_membership_question(self, onto):
        plan = apply_cap(plan_query(onto, {"brands": ["LANEIGE", "TIRTIR"]}))
        predicates = {e["predicate"] for e in static_edges(onto, plan)}
        assert not predicates & {NOT_OWNED_BY_GROUP, NOT_SIBLING_BRAND, MEMBERSHIP_UNKNOWN}

    def test_category_descendants_are_scope_only(self, onto):
        plan = plan_query(onto, {"categories": ["Skin Care"]})
        assert plan.categories == ["skin_care"]
        assert "lip_care" in plan.descendant_categories
        assert plan.scope_categories[0] == "skin_care"


# ----------------------------------------------------------------------
# 어댑터 정식 술어 모드
# ----------------------------------------------------------------------


class TestAdapterCanonicalMode:
    def _edges_fact(self, *edges):
        return [{"type": "metric_edges", "entity": "laneige", "data": {"edges": list(edges)}}]

    def test_legacy_mode_keeps_ownedby_display(self):
        fact = self._edges_fact(
            {"subject": "LANEIGE", "predicate": "ownedByGroup", "object": "AMOREPACIFIC"}
        )
        cards = EvidenceAdapter().from_kg_facts(fact).cards
        assert [c.predicate for c in cards] == ["ownedBy"]

    def test_canonical_mode_uses_canonical_predicate_and_registry_ids(self, onto):
        fact = self._edges_fact(
            {"subject": "Tata Harper", "predicate": "ownedBy", "object": "AMOREPACIFIC"},
            {"subject": "laneige", "predicate": "hasPricePosition", "object": "premium"},
        )
        result = EvidenceAdapter(ontology=onto).from_kg_facts(fact)
        assert [(c.subject, c.predicate, c.object) for c in result.cards] == [
            ("tata_harper", "ownedByGroup", "amorepacific")
        ]
        assert [e["reason"] for e in result.excluded] == [EXCLUDED_NUMERIC]

    def test_canonical_mode_drops_registry_placeholder_brands_only_as_brands(self, onto):
        fact = self._edges_fact(
            {"subject": "laneige", "predicate": "competesWith", "object": "fresh"},
            {"subject": "laneige", "predicate": "hasTrend", "object": "fresh"},
        )
        legacy = EvidenceAdapter().from_kg_facts(fact).cards
        canonical = EvidenceAdapter(ontology=onto).from_kg_facts(fact).cards
        assert {c.predicate for c in legacy} == {"competesWith", "hasTrend"}
        assert {c.predicate for c in canonical} == {"hasTrend"}  # 트렌드 단어 "fresh"는 유지

    def test_static_metadata_becomes_cards_only_in_canonical_mode(self, onto):
        fact = [
            {
                "type": "brand_info",
                "entity": "laneige",
                "data": {"segment": "Premium", "sos": 0.42, "avg_rank": 3.3},
            }
        ]
        legacy = EvidenceAdapter().from_kg_facts(fact)
        assert legacy.cards == [] and len(legacy.excluded) == 3
        canonical = EvidenceAdapter(ontology=onto).from_kg_facts(fact)
        assert [(c.subject, c.predicate, c.object) for c in canonical.cards] == [
            ("laneige", "hasSegment", "premium")
        ]
        assert {e["predicate"] for e in canonical.excluded} == {"sos", "avg_rank"}
        assert {e["reason"] for e in canonical.excluded} == {EXCLUDED_METADATA}

    def test_ontology_static_fact_cards(self, onto):
        fact = [
            {
                "type": "ontology_static",
                "entity": "cosrx",
                "data": {
                    "edges": [
                        {"subject": "cosrx", "predicate": "acquiredIn", "object": "2024"},
                        {
                            "subject": "tirtir",
                            "predicate": NOT_OWNED_BY_GROUP,
                            "object": "amorepacific",
                            "closed_world": True,
                        },
                    ],
                    "as_of": onto.as_of,
                    "version": onto.version,
                },
            }
        ]
        cards = EvidenceAdapter(ontology=onto).from_kg_facts(fact).cards
        assert all(c.source == ONTOLOGY_SOURCE and c.as_of == onto.as_of for c in cards)
        assert [(c.subject, c.predicate, c.object) for c in cards] == [
            ("cosrx", "acquiredIn", "2024"),
            ("tirtir", NOT_OWNED_BY_GROUP, "amorepacific"),
        ]
        assert cards[1].metadata["closed_world"] is True
        assert "TIRTIR" in cards[1].text


# ----------------------------------------------------------------------
# DB 수치 제공자
# ----------------------------------------------------------------------


class TestMetricFactsScope:
    async def test_default_brand_cap_unchanged(self, tmp_path):
        provider = MetricFactsProvider(make_ontology_db(tmp_path), as_of=AS_OF)
        facts = await provider.collect(
            {"brands": ["LANEIGE", "TIRTIR", "innisfree", "ETUDE"], "categories": ["face_powder"]}
        )
        shares = [f["brand"] for f in facts if f["type"] == "brand_share"]
        assert shares == ["LANEIGE", "TIRTIR", "innisfree"]

    async def test_raised_brand_cap(self, tmp_path):
        provider = MetricFactsProvider(make_ontology_db(tmp_path), as_of=AS_OF)
        facts = await provider.collect(
            {"brands": ["LANEIGE", "TIRTIR", "innisfree", "ETUDE"], "categories": ["face_powder"]},
            max_brands=12,
        )
        shares = [f["brand"] for f in facts if f["type"] == "brand_share"]
        assert shares == ["LANEIGE", "TIRTIR", "innisfree", "ETUDE"]

    async def test_scope_categories_only_those_with_data_and_keep_own_category(self, tmp_path):
        provider = MetricFactsProvider(make_ontology_db(tmp_path), as_of=AS_OF)
        facts = await provider.collect(
            {"brands": ["LANEIGE"], "categories": ["skin_care"]},
            scope_categories=["body_skincare", "lip_care", "face_skincare"],
        )
        assert [f["category"] for f in facts if f["type"] == "category_market"] == [
            "skin_care",
            "lip_care",
        ]
        laneige = {f["category"]: f for f in facts if f["type"] == "brand_share"}
        assert laneige["skin_care"]["present"] is False  # lip_care SoS가 skin_care로 오지 않는다
        assert laneige["lip_care"]["sos"] == 2.0

    async def test_present_brands(self, tmp_path):
        provider = MetricFactsProvider(make_ontology_db(tmp_path), as_of=AS_OF)
        assert "tirtir" in await provider.present_brands()
        assert await provider.present_brands(["skin_care"]) == {"medicube", "cosrx"}


# ----------------------------------------------------------------------
# 검색기 (실제 KG·SQLite, 가짜 문서 검색기)
# ----------------------------------------------------------------------


class TestRetrieverFlagOn:
    async def test_group_query_expands_and_cards_reach_prompt(self, tmp_path, flag_on, onto):
        retriever = _retriever(
            tmp_path, {"brands": ["amorepacific"], "categories": ["face_powder"]}
        )
        ctx = await retriever.retrieve(QUERIES["group"])

        trace = ctx.metadata["ontology"]
        assert trace["groups"] == ["amorepacific"]
        # 크롤 DB(face_powder)에 있는 소속 브랜드가 먼저 남는다
        assert trace["expanded_brands"][:3] == ["laneige", "etude", "innisfree"]
        assert len(trace["expanded_brands"]) == MAX_EXPANDED_BRANDS
        prompt_onto = _onto_cards(ctx.prompt_evidence)
        assert trace["prompt_ontology_cards"] == len(prompt_onto) == trace["ontology_cards"]
        owned = {c.subject for c in prompt_onto if c.predicate == "ownedByGroup"}
        assert {"laneige", "etude", "innisfree"} <= owned
        truncated = [c for c in prompt_onto if c.predicate == EXPANSION_TRUNCATED]
        assert len(truncated) == 1 and truncated[0].value == len(trace["dropped_brands"])
        # 전개 브랜드의 DB 수치가 실제로 조회된다 (브랜드 3개 상한은 전개에서만 12로)
        shares = {
            c.subject
            for c in ctx.evidence
            if c.kind is EvidenceKind.METRIC and c.predicate == "sos"
        }
        assert {"laneige", "etude", "innisfree"} <= shares
        assert trace["ontology_cards"] >= 13

    async def test_static_cards_survive_fact_cap_and_are_canonical(self, tmp_path, flag_on, onto):
        retriever = _retriever(
            tmp_path, {"brands": ["laneige", "tirtir"], "relations_hint": ["sibling"]}
        )
        ctx = await retriever.retrieve(QUERIES["negative"])

        static = [f for f in ctx.ontology_facts if f["type"] == "ontology_static"]
        assert len(static) == 1  # _weighted_merge의 사실 5개 상한에서 빠지지 않는다
        prompt = {(c.subject, c.predicate, c.object) for c in _onto_cards(ctx.prompt_evidence)}
        assert ("laneige", "ownedByGroup", "amorepacific") in prompt
        assert ("tirtir", NOT_OWNED_BY_GROUP, "amorepacific") in prompt
        assert ("laneige", NOT_SIBLING_BRAND, "tirtir") in prompt
        assert all(c.as_of == onto.as_of for c in _onto_cards(ctx.evidence))
        # KG 사본(ownedBy)은 중복으로 싣지 않는다, 옛 술어 이름도 없다
        predicates = {c.predicate for c in ctx.evidence if c.kind is EvidenceKind.RELATION}
        assert "ownedBy" not in predicates
        assert "[관계]" in ctx.combined_context and "notSiblingBrand" in ctx.combined_context

    async def test_price_position_is_split_and_excluded_as_numeric(self, tmp_path, flag_on):
        retriever = _retriever(tmp_path, {"brands": ["laneige"]})
        ctx = await retriever.retrieve("LANEIGE 가격 포지션은?")
        edges = [
            e for f in ctx.ontology_facts if f["type"] == "metric_edges" for e in f["data"]["edges"]
        ]
        assert "hasPosition" not in {e["predicate"] for e in edges}
        assert ctx.metadata["evidence_excluded_by_reason"].get(EXCLUDED_NUMERIC, 0) >= 1
        assert not [c for c in ctx.evidence if c.predicate in ("hasPricePosition", "hasSoS")]

    async def test_inclusion_widens_scope_but_never_rescales_numbers(self, tmp_path, flag_on):
        retriever = _retriever(
            tmp_path, {"brands": ["laneige", "cosrx"], "categories": ["skin_care"]}
        )
        ctx = await retriever.retrieve(QUERIES["inclusion"])

        assert "lip_care" in ctx.metadata["ontology"]["scope_categories"]
        metric = [c for c in ctx.evidence if c.kind is EvidenceKind.METRIC]
        # lip_care 수치가 조회 범위에 들어왔지만 카테고리는 lip_care 그대로다
        assert any(c.object == "lip_care" and c.subject == "laneige" for c in metric)
        # skin_care에서 LANEIGE는 여전히 Top 100 부재 — lip_care SoS를 올려 담지 않는다
        laneige_skin = [c for c in metric if c.subject == "laneige" and c.object == "skin_care"]
        assert [(c.predicate, c.value) for c in laneige_skin] == [("present_in_top100", False)]
        # 모든 SoS 카드는 DB의 (브랜드, 자기 카테고리) 값 그대로 (합산·환산 없음)
        conn = sqlite3.connect(tmp_path / "amore_data.db")
        db = {
            (b.lower(), c): s
            for b, c, s in conn.execute("SELECT brand, category_id, sos FROM brand_metrics")
        }
        conn.close()
        for card in metric:
            if card.predicate == "sos":
                assert card.value == pytest.approx(db[(card.subject, card.object)] / 100)
        # 시장 지표도 카테고리별 그대로 (skin_care HHI ≠ lip_care HHI)
        hhi = {c.subject: c.value for c in metric if c.predicate == "hhi"}
        assert hhi == {"skin_care": 0.067, "lip_care": 0.0681}

    async def test_inclusion_extends_rule_combinations(self, tmp_path, flag_on):
        retriever = _retriever(tmp_path, {"brands": ["laneige"], "categories": ["skin_care"]})
        ctx = await retriever.retrieve(QUERIES["inclusion"])
        combos = ctx.metadata["rule_evaluation"]["combinations"]
        assert any("lip_care" in str(combo) for combo in combos)

    async def test_flag_is_evaluated_per_query(self, tmp_path, monkeypatch):
        retriever = _retriever(tmp_path, {"brands": ["laneige", "tirtir"]})
        monkeypatch.setenv(FLAG, "true")
        on = await retriever.retrieve(QUERIES["negative"])
        monkeypatch.setenv(FLAG, "false")
        off = await retriever.retrieve(QUERIES["negative"])
        assert "ontology" in on.metadata and "ontology" not in off.metadata
        assert not _onto_cards(off.evidence)
        assert "ownedBy" in {c.predicate for c in off.evidence}

    async def test_plan_failure_degrades_to_legacy_path(self, tmp_path, flag_on, monkeypatch):
        retriever = _retriever(tmp_path, {"brands": ["laneige"]})

        async def boom(entities):
            raise RuntimeError("registry unavailable")

        monkeypatch.setattr(retriever, "_plan_ontology", boom)
        ctx = await retriever.retrieve(QUERIES["negative"])
        assert "ontology" not in ctx.metadata
        assert [d["component"] for d in ctx.metadata["degraded"]] == ["ontology_plan"]
        assert ctx.evidence
