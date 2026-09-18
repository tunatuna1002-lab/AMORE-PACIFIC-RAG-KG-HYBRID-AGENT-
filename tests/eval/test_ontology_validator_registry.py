"""[2026-09 사후] O0-A: ontology_validator 타입 등록부·엣지 파싱·골드 엣지 타입."""

from eval.validators.ontology_validator import (
    EntityTypeRegistry,
    canonical_predicate,
    gold_edge_types,
    normalize_entity_key,
    parse_edge,
)


class TestParsing:
    def test_parse_edge(self):
        assert parse_edge("beauty of joseon -competesWith-> laneige") == (
            "beauty of joseon",
            "competesWith",
            "laneige",
        )
        assert parse_edge("not an edge") is None

    def test_normalize_and_alias(self):
        assert normalize_entity_key(" Beauty of-Joseon ") == "beauty_of_joseon"
        assert canonical_predicate("ownedBy") == "ownedByGroup"
        assert canonical_predicate("hasSoS") == "hasSoS"


class TestRealConfigRegistry:
    """실제 config(읽기 전용)에서 만든 등록부의 대표 타입."""

    def test_types(self):
        reg = EntityTypeRegistry(
            EntityTypeRegistry.from_config()._types, EntityTypeRegistry.from_config()._groups
        )
        assert reg.type_of("LANEIGE") == ("brand", "registry")
        assert reg.type_of("laneige")[0] == "brand"
        assert reg.type_of("elf")[0] == "brand"  # e.l.f. 점 제거 변형
        assert reg.type_of("beauty_of_joseon")[0] == "brand"
        assert reg.type_of("amorepacific")[0] == "corporate_group"
        assert reg.type_of("lip_care")[0] == "category"
        assert reg.type_of("3761351")[0] == "category"
        assert reg.type_of("Premium")[0] == "segment"
        # 'Makeup'은 brands.json 세그먼트이기도 하지만 카테고리 ID가 우선한다
        assert reg.type_of("makeup")[0] == "category"
        assert EntityTypeRegistry.from_config().alt_types("Makeup") == {"segment"}
        assert reg.type_of("unknown")[0] == "placeholder"
        assert reg.type_of("fresh")[0] == "placeholder"
        assert reg.type_of("B0BZGRCBY4") == ("product", "pattern")
        assert reg.type_of("nivea") == (None, None)  # 등록부에 없다 — 검사 불가
        assert reg.group_of("cosrx") == "amorepacific"
        assert reg.group_of("tirtir") is None

    def test_segment_name_shadowed_by_category_still_satisfies_has_segment(self):
        from eval.validators.ontology_validator import OntologyValidator

        base = EntityTypeRegistry.from_config()
        reg = EntityTypeRegistry(base._types, base._groups, alt_types=base._alt_types)
        result = OntologyValidator().check_trace_types(
            ["espoir -hasSegment-> Makeup"], [], registry=reg
        )
        assert result["violations"] == 0
        assert result["checks"] == 2


class TestOntologyPrecedence:
    def test_ontology_entity_type_wins_when_available(self):
        class FakeOntology:
            def entity_type(self, name: str) -> str | None:
                return "brand" if name == "nivea" else None

        reg = EntityTypeRegistry({"laneige": "brand"}, ontology=FakeOntology())
        assert reg.type_of("nivea") == ("brand", "ontology")
        assert reg.type_of("laneige") == ("brand", "registry")  # 온톨로지가 모르면 등록부
        assert reg.source_label == "ontology"


class TestGoldEdgeTypes:
    def test_unique_signature_only(self):
        types = gold_edge_types(
            [
                "cosrx -ownedByGroup-> amorepacific",
                "laneige -rankedIn-> lip_care",  # 도메인 {brand, product} → 주어 타입 안 정함
                "laneige -hasSegment-> Premium",
                "garbage",
            ]
        )
        assert types == {
            "cosrx": "brand",
            "amorepacific": "corporate_group",
            "lip_care": "category",
            "laneige": "brand",
            "premium": "segment",
        }
