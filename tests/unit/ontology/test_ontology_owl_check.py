"""OWL export + Pellet cross-check of the JSON ontology (track O1, OE2-추가).

The Pellet test needs Java >= 25 (owlready2 0.50 bundled Pellet). Without it the test is
skipped with an explicit reason — never silently passed.
"""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

REPO = Path(__file__).resolve().parents[3]


def _load_script(name: str) -> ModuleType:
    path = REPO / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_o1_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


owlready2 = pytest.importorskip("owlready2", reason="owlready2 not installed (dev-only dependency)")
export_mod = _load_script("export_ontology_owl")
check_mod = _load_script("check_ontology_owl")


class TestExport:
    def test_export_writes_owl_file(self, tmp_path: Path) -> None:
        out = export_mod.export_owl(tmp_path / "onto.owl")
        assert out.exists()
        text = out.read_text(encoding="utf-8")
        assert "AmorepacificBrand" in text
        assert "owl:AllDifferent" in text
        assert "owl:hasValue" in text

    def test_export_round_trip_classes(self, tmp_path: Path) -> None:
        from src.ontology.ontology import load_ontology

        onto = load_ontology()
        out = export_mod.export_owl(tmp_path / "onto.owl", onto)
        world = owlready2.World()
        owl = world.get_ontology(out.as_uri()).load()
        names = {c.name for c in owl.classes()}
        assert set(onto.classes) <= names
        assert len(list(owl.individuals())) == len(onto.individuals)

    def test_export_refuses_data_dir(self) -> None:
        with pytest.raises(ValueError, match="data"):
            export_mod.export_owl(REPO / "data" / "onto.owl")


class TestCompare:
    def test_diff_reports_missing_and_extra(self) -> None:
        diff = check_mod.diff_maps(
            "class",
            expected={"Brand": {"a", "b"}, "Group": {"g"}},
            actual={"Brand": {"a", "c"}, "Group": {"g"}},
        )
        assert any("Brand" in d and "'b'" in d for d in diff)
        assert any("Brand" in d and "'c'" in d for d in diff)
        assert not any("Group" in d for d in diff)

    def test_diff_empty_when_equal(self) -> None:
        assert check_mod.diff_maps("p", {"x": {("a", "b")}}, {"x": {("a", "b")}}) == []

    def test_java_version_parse(self) -> None:
        assert check_mod.parse_java_major('openjdk version "25.0.4.1" 2026-08-18 LTS') == 25
        assert check_mod.parse_java_major('java version "1.8.0_402"') == 8
        assert check_mod.parse_java_major('openjdk version "17.0.2" 2022-01-18') == 17
        assert check_mod.parse_java_major("garbage") is None

    def test_main_exit_3_without_java(self, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
        monkeypatch.setattr(check_mod, "java_major_version", lambda: None)
        assert check_mod.main([]) == 3
        assert "SKIPPED: Pellet requires Java >= 25" in capsys.readouterr().out

    def test_main_exit_3_with_old_java(self, monkeypatch: pytest.MonkeyPatch, capsys) -> None:
        monkeypatch.setattr(check_mod, "java_major_version", lambda: 8)
        assert check_mod.main([]) == 3
        assert "SKIPPED" in capsys.readouterr().out


_JAVA = check_mod.java_major_version()


@pytest.mark.slow
@pytest.mark.skipif(
    _JAVA is None or _JAVA < 25,
    reason=f"SKIPPED: Pellet requires Java >= 25 (found: {_JAVA})",
)
class TestPellet:
    def test_pellet_consistent_and_matches_closure(self, tmp_path: Path) -> None:
        result = check_mod.run_check(tmp_path)
        assert result.consistent, result.inconsistent_classes
        assert result.inconsistent_classes == []
        assert result.mismatches == [], "\n".join(result.mismatches)
        assert result.class_count >= 10
        assert result.individual_count > 100
