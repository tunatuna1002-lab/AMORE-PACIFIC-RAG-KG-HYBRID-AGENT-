"""EntityLinker 온톨로지 등록부 연결 — 플래그 ON 동작 (트랙 O2, 결정 OA-6)

플래그 `ontology.use_class_reasoning`이 켜지면:
1. 등록부(`config/ontology/brands.json`) 이름·별칭으로 브랜드를 더 인식한다. 기존 사전이 인식하던
   브랜드는 같은 문자열로 그대로 나온다.
2. 가짜 브랜드(`unknown`·`fresh`·`chi`)는 내보내지 않는다.
3. 클래스·그룹 언급 → ``classes``·``groups``, 자매 브랜드 의도 → ``relations_hint``.
OFF 동작은 `test_entity_linker_characterization.py`가 고정한다.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from src.infrastructure.feature_flags import FeatureFlags
from src.rag import entity_tags
from src.rag.entity_linker import AMBIGUOUS_REGISTRY_KEYS, EntityLinker, class_reasoning_enabled

FLAG_ENV = "FF_ONTOLOGY_USE_CLASS_REASONING"


@pytest.fixture(autouse=True)
def flag_on(monkeypatch):
    monkeypatch.setenv(FLAG_ENV, "true")
    FeatureFlags.reset_instance()
    EntityLinker._config_cache = None
    EntityLinker._config_loaded_at = None
    yield
    FeatureFlags.reset_instance()


@pytest.fixture
def linker() -> EntityLinker:
    return EntityLinker(use_spacy=False)


def _extract(linker: EntityLinker, query: str, kg: Any = None) -> dict[str, Any]:
    return linker.extract_entities(query, knowledge_graph=kg)


# ── 플래그 ───────────────────────────────────────────────────────


def test_flag_is_read_per_call(monkeypatch, linker):
    assert class_reasoning_enabled()
    assert "brand_ids" in _extract(linker, "LANEIGE")
    monkeypatch.setenv(FLAG_ENV, "false")
    FeatureFlags.reset_instance()
    assert not class_reasoning_enabled()
    assert "brand_ids" not in _extract(linker, "LANEIGE")


# ── 1. 등록부 브랜드 ─────────────────────────────────────────────


@pytest.mark.parametrize(
    ("query", "brand", "brand_id"),
    [
        ("2026-08-31 기준 IT Cosmetics는 Face Powder에서?", "it cosmetics", "it_cosmetics"),
        ("Jouer는 Face Powder에서 가격-품질 불일치인가요?", "jouer", "jouer"),
        ("Almay는 Face Powder에서 가성비 포지션인가요?", "almay", "almay"),
        ("Charlotte Tilbury는 프리미엄 가격인가요?", "charlotte tilbury", "charlotte_tilbury"),
        ("COVERGIRL는 Face Powder에서?", "covergirl", "covergirl"),
        ("IOPE 매출은?", "iope", "iope"),
        ("primera와 한율 비교", "primera", "primera"),
        ("한율 세그먼트는?", "hanyul", "hanyul"),
        ("Mise-en-scène 샴푸", "mise-en-scène", "mise_en_scene"),
        ("TATA HARPER의 원산지는?", "tata harper", "tata_harper"),
    ],
)
def test_registry_brands_are_recognized(linker, query, brand, brand_id):
    entities = _extract(linker, query)
    assert brand in entities["brands"]
    assert brand_id in entities["brand_ids"]


@pytest.mark.parametrize(
    ("query", "brand"),
    [
        ("LANEIGE 라네즈 Lip Care", "laneige"),
        ("e.l.f. 립 오일", "e.l.f."),
        ("La Roche-Posay 비교", "la roche-posay"),
        ("L'Oreal Face Powder", "l'oreal"),
        ("COSRX 코스알엑스", "cosrx"),
    ],
)
def test_known_brands_keep_existing_surface_string(monkeypatch, linker, query, brand):
    on = _extract(linker, query)
    monkeypatch.setenv(FLAG_ENV, "false")
    FeatureFlags.reset_instance()
    off = _extract(linker, query)
    assert brand in on["brands"]
    # ON은 OFF 결과를 순서까지 앞부분에 그대로 포함한다
    assert on["brands"][: len(off["brands"])] == off["brands"]
    for key in ("categories", "indicators", "time_range", "products", "sentiments"):
        assert on[key] == off[key]


def test_case_and_symbol_variants_resolve_to_one_brand(linker):
    entities = _extract(linker, "E.L.F. vs elf vs Elf, la roche posay vs LA ROCHE-POSAY")
    assert entities["brands"].count("e.l.f.") == 1
    assert entities["brands"].count("la roche-posay") == 1
    assert sorted(entities["brand_ids"]) == ["elf", "la_roche_posay"]


@pytest.mark.parametrize(
    "query",
    [
        "LANEIGE가 Face Makeup 시장에서 성공하려면?",  # '려'(RYO 한글 별칭)
        "고려해야 할 요소는?",
        "COSRX Snail Mucin 96% Essence 순위는?",  # essence
        "median price in Lip Care",
        "competitive matrix of lip brands",
        "share of shelf 기준 shelf 전체",  # 'elf' in shelf
        "chief marketing, chips, archive",  # 'chi' 부분 문자열
        "the ryokan and bojangles",  # ryo·boj 부분 문자열
        "apple and APAC markets",  # ap
    ],
)
def test_short_or_common_aliases_do_not_false_positive(linker, query):
    entities = _extract(linker, query)
    assert not {
        "ryo",
        "essence",
        "median",
        "matrix",
        "e.l.f.",
        "chi",
        "boj",
        "amore pacific",
    } & set(entities["brands"]), entities["brands"]


def test_short_aliases_still_match_as_whole_words(linker):
    entities = _extract(linker, "RYO 샴푸와 BOJ 선크림, elf 립")
    assert {"ryo", "beauty of joseon", "e.l.f."} <= set(entities["brands"])


def test_ambiguous_keys_are_documented():
    assert {"려", "ap", "essence"} <= AMBIGUOUS_REGISTRY_KEYS


# ── 2. 가짜 브랜드 ───────────────────────────────────────────────


class _PlaceholderKG:
    def query(self, predicate: Any = None, object_: Any = None) -> list[Any]:
        rel = SimpleNamespace(
            subject="unknown",
            predicate=SimpleNamespace(value="hasProduct"),
            object="B0UNK00001",
            properties={"title": "Hydrating Lip Oil Serum Treatment", "category": "lip_care"},
        )
        return [rel] if object_ is None else []


def test_placeholder_brands_are_never_emitted(linker, monkeypatch):
    query = "Hydrating Lip Oil 제품과 fresh, unknown, chi 브랜드"
    on = _extract(linker, query, kg=_PlaceholderKG())
    assert not {"unknown", "fresh", "chi"} & set(on["brands"])
    assert not {"unknown", "fresh", "chi"} & set(on["brand_ids"])
    # 제품 역링크 자체는 유지된다
    assert "hydrating_lip_oil" in on["products"]

    monkeypatch.setenv(FLAG_ENV, "false")
    FeatureFlags.reset_instance()
    off = _extract(linker, query, kg=_PlaceholderKG())
    assert "unknown" in off["brands"]  # OFF는 기존 동작(가짜 브랜드 포함) 그대로


# ── 3. 클래스·그룹 언급 ─────────────────────────────────────────


@pytest.mark.parametrize(
    ("query", "classes", "groups"),
    [
        ("아모레퍼시픽 브랜드들의 Lip Care 순위는?", ["AmorepacificBrand"], ["amorepacific"]),
        ("아모레퍼시픽 계열 브랜드 SoS", ["AmorepacificBrand"], ["amorepacific"]),
        ("아모레퍼시픽 그룹 소속 브랜드 중 Face Powder", ["AmorepacificBrand"], ["amorepacific"]),
        ("Amorepacific brands in Skin Care", ["AmorepacificBrand"], ["amorepacific"]),
        ("AP 그룹 브랜드들의 순위", ["AmorepacificBrand"], ["amorepacific"]),
        ("AP 그룹 매출", [], ["amorepacific"]),
        ("아모레퍼시픽 그룹 브랜드 COSRX의 원산지는?", [], ["amorepacific"]),
        ("K-Beauty 브랜드 중 1위는?", ["KBeautyBrand"], []),
        ("K뷰티 트렌드", ["KBeautyBrand"], []),
        ("한국 브랜드와 미국 브랜드", ["KBeautyBrand"], []),
        ("Korean brands on Amazon", ["KBeautyBrand"], []),
        ("럭셔리 브랜드 가격대", ["LuxuryBrand"], []),
        ("프리미엄 브랜드 SoS", ["PremiumBrand"], []),
        ("premium brands in lip care", ["PremiumBrand"], []),
        ("매스 브랜드 비중", ["MassBrand"], []),
        ("중저가 브랜드", ["MidTierBrand", "AffordableBrand"], []),
        ("Skin Care 카테고리 프리미엄화 트렌드는?", [], []),
        ("가격 프리미엄이 붙은 제품", [], []),
    ],
)
def test_class_and_group_mentions(linker, query, classes, groups):
    entities = _extract(linker, query)
    assert entities["classes"] == classes
    assert entities["groups"] == groups


def test_group_is_distinguished_from_amore_pacific_brand_line(linker):
    group = _extract(linker, "아모레퍼시픽 그룹 매출")
    assert group["groups"] == ["amorepacific"]
    assert "amore_pacific" not in group["brand_ids"]
    # 기존 사전이 'amorepacific'을 브랜드로 내보내던 동작은 유지 (등록부 브랜드는 아니다)
    assert "amorepacific" in group["brands"]

    line = _extract(linker, "Amore Pacific 라인 제품")
    assert line["groups"] == []
    assert line["brand_ids"] == ["amore_pacific"]
    assert line["brands"] == ["amore pacific"]


@pytest.mark.parametrize(
    "query",
    ["LANEIGE와 같은 그룹 브랜드는?", "LANEIGE의 자매 브랜드는?", "sister brands of LANEIGE"],
)
def test_sibling_intent(linker, query):
    assert _extract(linker, query)["relations_hint"] == ["sibling"]


def test_no_sibling_intent_by_default(linker):
    assert _extract(linker, "LANEIGE Lip Care SoS")["relations_hint"] == []


def test_emitted_classes_exist_in_schema(linker):
    from src.ontology.ontology import get_ontology

    classes = set(get_ontology().classes)
    entities = _extract(linker, "아모레퍼시픽 브랜드들 중 K-Beauty 럭셔리 브랜드와 중저가 브랜드")
    assert entities["classes"]
    assert set(entities["classes"]) <= classes


# ── link() ───────────────────────────────────────────────────────


def test_link_adds_registry_brands_with_ids(linker):
    linked = linker.link("IT Cosmetics와 LANEIGE")
    by_label = {e.concept_label: e for e in linked if e.entity_type == "brand"}
    assert by_label["it cosmetics"].context["registry_id"] == "it_cosmetics"
    assert by_label["LANEIGE"].context["registry_id"] == "laneige"


# ── entity_tags: 태그 문자열 형식 불변 ─────────────────────────


def test_entity_tags_keep_format_and_match_existing_snapshot_tags(monkeypatch):
    entity_tags._linker.cache_clear()
    tags = entity_tags.extract_tags("LANEIGE Lip Sleeping Mask와 IT Cosmetics Face Powder")
    assert "laneige" in tags["brands"]  # 기존 색인 태그 "|laneige|"와 그대로 일치
    assert "it cosmetics" in tags["brands"]
    encoded = entity_tags.encode_tags(tags["brands"])
    assert "|laneige|" in encoded
    targets = entity_tags.query_tag_targets(
        EntityLinker(use_spacy=False).extract_entities("라네즈 립케어")
    )
    assert targets["brands"] == {"laneige"}
    assert targets["categories"] == {"lip_care"}
