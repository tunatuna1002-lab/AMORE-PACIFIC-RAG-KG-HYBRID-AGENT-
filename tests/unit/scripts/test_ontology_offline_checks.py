"""[2026-09 사후] O0-B: scripts/ontology_offline_checks.py 작은 픽스처 검증."""

import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module")
def checks():
    spec = importlib.util.spec_from_file_location(
        "ontology_offline_checks", REPO / "scripts" / "ontology_offline_checks.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def registry(tmp_path):
    from eval.validators.ontology_validator import EntityTypeRegistry

    (tmp_path / "brands.json").write_text(
        json.dumps(
            {
                "target_brand": {"name": "LANEIGE", "parent": "AMOREPACIFIC", "aliases": []},
                "amorepacific_brands": [{"name": "COSRX"}],
                "segments": {"Premium": []},
            }
        )
    )
    (tmp_path / "category_hierarchy.json").write_text(
        json.dumps({"categories": {"lip_care": {}, "beauty": {}}})
    )
    return EntityTypeRegistry.from_config(tmp_path)


def _t(s, p, o, **props):
    return {"subject": s, "predicate": p, "object": o, "properties": props, "valid_from": None}


TRIPLES = [
    _t("LANEIGE", "ownedByGroup", "AMOREPACIFIC"),
    _t("COSRX", "ownedByGroup", "AMOREPACIFIC"),
    _t("LANEIGE", "hasSegment", "Premium"),
    _t("LANEIGE", "siblingBrand", "COSRX"),
    _t("COSRX", "siblingBrand", "LANEIGE"),
    _t("laneige", "competesWith", "nivea"),
    _t("nivea", "competesWith", "laneige"),
    _t("laneige", "competesWith", "unknown"),
    _t("laneige", "hasProduct", "B000000001", title="LANEIGE Lip Sleeping Mask", rank=1),
    _t("B000000001", "belongsToCategory", "lip_care", rank=1),
    _t("laneige", "belongsToCategory", "lip_care", original_predicate="rankedIn"),
    _t("laneige", "hasPosition", "lip_care", original_predicate="hasSoS", share=0.05),
    _t("lip_care", "hasPosition", "0.07", original_predicate="hasHHI", hhi=0.07),
    _t("unknown", "hasProduct", "B000000002", rank=3),
]


def test_effective_predicate(checks):
    assert checks.effective_predicate(TRIPLES[10]) == "rankedIn"
    assert checks.effective_predicate(
        _t("a", "hasProduct", "b", original_predicate="HAS_PRODUCT")
    ) == ("hasProduct")


def test_brand_surfaces_skip_categories_and_merge_case(checks, registry):
    surfaces = checks.kg_brand_surfaces(TRIPLES, registry)
    assert surfaces["laneige"] == ["LANEIGE", "laneige"]
    assert "lip_care" not in surfaces  # hasHHI 주어(카테고리)는 브랜드가 아니다
    assert "unknown" in surfaces  # 가짜 브랜드는 호출 측에서 분리


def test_brand_recognition_splits_implicit(checks, registry):
    class FakeLinker:
        def extract_entities(self, text, knowledge_graph=None):
            return {"brands": ["laneige"] if "laneige" in text.lower() else []}

    gold = [
        {"question": "LANEIGE 순위는?", "gold": {"kg_entities": ["laneige", "lip_care"]}},
        {"question": "COSRX 순위는?", "gold": {"kg_entities": ["cosrx"]}},
        {"question": "K-Beauty 1위는?", "gold": {"kg_entities": ["laneige"]}},
    ]
    surfaces = checks.kg_brand_surfaces(TRIPLES, registry)
    result = checks.brand_recognition(surfaces, gold, FakeLinker(), registry)
    assert result["kg_brands"] == 3  # laneige, cosrx, nivea (unknown 제외)
    assert result["kg_brands_recognized"] == 1  # FakeLinker는 laneige만 안다
    assert result["kg_unrecognized"] == ["cosrx", "nivea"]
    assert result["gold_brand_mentions"] == 2
    assert result["gold_brand_linked"] == 1
    assert result["gold_unlinked"] == {"cosrx": 1}
    assert result["gold_implicit_by_brand"] == {"laneige": 1}


def test_gold_edge_reachability(checks):
    gold = [
        {
            "gold": {
                "kg_edges": [
                    "cosrx -ownedByGroup-> amorepacific",
                    "cosrx -ownedBy-> amorepacific",
                    "laneige -hasSegment-> premium",
                    "laneige -rankedIn-> lip_care",
                    "laneige -hasProduct-> lip_sleeping_mask",
                    "lip_sleeping_mask -belongsToCategory-> lip_care",
                    "laneige -competesWith-> tirtir",
                ]
            }
        }
    ]
    emitted = ["belongsToCategory", "competesWith", "hasProduct", "ownedBy", "rankedIn"]
    r = checks.gold_edge_reachability(gold, TRIPLES, emitted)
    assert r["ownedByGroup"] == {
        "total": 1,
        "in_kg_same_name": 1,
        "in_kg_alias": 1,
        "runtime_emits_name": 0,
        "reachable_now": 0,
        "reachable_if_normalized": 1,
    }
    assert r["ownedBy"]["in_kg_same_name"] == 0
    assert r["ownedBy"]["reachable_now"] == 1
    assert r["hasSegment"]["in_kg_same_name"] == 1
    assert r["hasSegment"]["reachable_now"] == 0
    assert r["rankedIn"]["reachable_now"] == 1
    assert r["hasProduct"]["reachable_now"] == 1  # 제목 슬러그로 ASIN에 닿는다
    assert r["belongsToCategory"]["reachable_now"] == 1
    assert r["competesWith"]["in_kg_same_name"] == 0


def test_kg_consistency(checks, registry):
    triples = TRIPLES + [_t("Laneige", "competesWith", "cosrx")]
    r = checks.kg_consistency(triples, registry)
    assert r["case_duplicates"] == 2  # laneige(3표기), cosrx(2표기)
    assert r["placeholder_triples"]["unknown"] == {"competesWith": 1, "hasProduct": 1}
    asym = r["asymmetric_symmetric_relations"]
    assert asym["siblingBrand"]["asymmetric"] == 0
    assert asym["competesWith"]["asymmetric"] == 2  # laneige→unknown, laneige→cosrx
    assert r["numeric_edges_without_valid_from"] == {
        "belongsToCategory": 1,
        "hasHHI": 1,
        "hasProduct": 2,
        "hasSoS": 1,
    }
    assert r["hasPosition_by_original_predicate"] == {"hasHHI": 1, "hasSoS": 1}
    assert r["domain_range_violations"]["competesWith:object=placeholder"] == 1
    assert r["domain_range_violations"]["hasProduct:subject=placeholder"] == 1


def test_read_source_constants_parses_nested_and_frozenset(checks, tmp_path):
    src = tmp_path / "m.py"
    src.write_text(
        "X: frozenset[str] = frozenset({'a', 'b'})\n"
        "def f():\n    priority_preds = {'hasSoS', 'ownedBy'}\n    return priority_preds\n"
    )
    found = checks.read_source_constants(src, {"X", "priority_preds"})
    assert found == {"X": {"a", "b"}, "priority_preds": {"hasSoS", "ownedBy"}}


def test_real_retriever_filters_are_found(checks):
    filters = checks.retriever_predicate_filters(REPO / "src")
    # 하위 호환 키 = 플래그 OFF(레거시)
    assert "ownedBy" in filters["priority_preds"]
    assert "hasSegment" not in filters["runtime_emitted_predicates"]
    assert "hasSoS" in filters["kg_numeric_predicates_excluded_from_cards"]
    # O3: OFF/ON을 따로 낸다 — ON은 정식 술어(ownedByGroup)와 정적 카드 술어를 담는다
    assert "ownedBy" in filters["off"]["priority_preds"]
    assert "ownedByGroup" not in filters["off"]["priority_preds"]
    assert "ownedByGroup" in filters["on"]["priority_preds"]
    assert "ownedBy" not in filters["on"]["priority_preds"]
    assert "hasSegment" in filters["on"]["runtime_emitted_predicates"]
    assert "hasSegment" in filters["on"]["static_brand_predicates_ontology_card"]
