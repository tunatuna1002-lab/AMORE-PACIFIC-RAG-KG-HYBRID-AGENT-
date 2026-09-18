"""[2026-09 사후] OE10: scripts/start.py가 uvicorn 전에 온톨로지 원본을 로드하고, 오류면 종료."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from src.ontology import ontology as ontology_module

REPO = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module")
def start():
    spec = importlib.util.spec_from_file_location("start_script", REPO / "scripts" / "start.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_loads_real_source(start, capsys: pytest.CaptureFixture[str]) -> None:
    start._load_ontology_or_exit()
    out = capsys.readouterr().out
    onto = ontology_module.get_ontology()
    assert f"classes={onto.class_count}" in out
    assert f"brands={onto.brand_count}" in out


def test_malformed_source_exits_nonzero(start, tmp_path: Path) -> None:
    (tmp_path / "schema.json").write_text("{ not json", encoding="utf-8")
    (tmp_path / "brands.json").write_text(json.dumps({"brands": []}), encoding="utf-8")

    def broken() -> ontology_module.Ontology:
        return ontology_module.load_ontology(tmp_path)

    with patch.object(ontology_module, "get_ontology", broken):
        with pytest.raises(SystemExit) as exc:
            start._load_ontology_or_exit()
    assert exc.value.code == 1


def test_loader_runs_before_uvicorn() -> None:
    source = (REPO / "scripts" / "start.py").read_text(encoding="utf-8")
    main = source[source.index('if __name__ == "__main__":') :]
    assert main.index("_load_ontology_or_exit()") < main.index("uvicorn.run(")
