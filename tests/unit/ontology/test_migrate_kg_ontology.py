"""scripts/migrate_kg_ontology.py (track O5, OE6).

The migration applies the enforce rules of ``kg_write_validation`` to an existing KG JSON copy.
It never writes in dry-run, refuses to write into the repository ``data/`` or over its input,
and is deterministic.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

from src.domain.entities.relations import RelationType
from src.ontology.knowledge_graph import KnowledgeGraph

REPO = Path(__file__).resolve().parents[3]


def _load_script(name: str) -> ModuleType:
    path = REPO / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_o5_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


mig = _load_script("migrate_kg_ontology")

TS = "2026-08-30T09:53:24"


def _t(s, p, o, props=None, source="kg_enricher"):
    return {
        "subject": s,
        "predicate": p,
        "object": o,
        "properties": props or {},
        "confidence": 0.9,
        "source": source,
        "created_at": TS,
        "valid_from": None,
        "valid_to": None,
    }


FIXTURE = {
    "version": "2.0",
    "triples": [
        _t("LANEIGE", "ownedByGroup", "AMOREPACIFIC", source="config/brands.json"),
        _t("laneige", "hasProduct", "B0AAAAAAA1", {"original_predicate": "HAS_PRODUCT"}),
        _t("laneige", "hasPosition", "lip_care", {"original_predicate": "hasSoS", "sos_pct": 5.0}),
        _t("lip_care", "hasPosition", "812.5", {"original_predicate": "hasHHI", "hhi": 812.5}),
        _t("cosrx", "hasPosition", "budget", {"original_predicate": "PRICE_POSITION"}),
        _t("unknown", "hasPosition", "lip_care", {"original_predicate": "hasSoS"}),
        _t("laneige", "competesWith", "cosrx", {"original_predicate": "COMPETES_WITH"}),
        _t("fresh", "competesWith", "laneige", {"original_predicate": "COMPETES_WITH"}),
        _t("laneige", "belongsToCategory", "lip_care", {"original_predicate": "rankedIn"}),
        _t("LANEIGE", "ownedBy", "AMOREPACIFIC", source="system"),
        _t("elf", "hasProduct", "B0AAAAAAA2", {"original_predicate": "HAS_PRODUCT"}),
        _t(
            "cosrx",
            "hasPosition",
            "lip_care",
            {"original_predicate": "hasSoS", "snapshot_date": "2026-09-01"},
        ),
        _t("lip_care", "parentCategory", "skin_care", source="config"),
        _t("B0AAAAAAA1", "hasAlert", "rank_drop", source="metrics"),
    ],
    "entity_metadata": {"LANEIGE": {"type": "brand", "segment": "Premium"}},
    "stats": {},
    "saved_at": "2026-09-17T21:12:02",
}


@pytest.fixture
def kg_in(tmp_path: Path) -> Path:
    path = tmp_path / "in.json"
    path.write_text(json.dumps(FIXTURE, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _keys(data):
    return {(t["subject"], t["predicate"], t["object"]) for t in data["triples"]}


class TestMigrate:
    def test_counts_by_reason(self):
        result = mig.migrate(json.loads(json.dumps(FIXTURE)))
        c = result.counts
        assert c["removed"]["placeholder_brand"] == 2
        assert c["removed"]["merged_duplicate"] == 1
        assert c["added"]["symmetric_closure"] == 1
        assert c["modified"]["predicate_canonicalized"] == 5  # survivors only (#10 is merged)
        assert c["modified"]["brand_canonicalized"] == 2
        assert c["modified"]["as_of_added"] == 1
        assert c["info"]["undated_numeric"] == 3

    def test_result_triples(self):
        out = mig.migrate(json.loads(json.dumps(FIXTURE))).data
        keys = _keys(out)
        assert ("laneige", "hasSoS", "lip_care") in keys
        assert ("lip_care", "hasHHI", "812.5") in keys
        assert ("cosrx", "hasPricePosition", "budget") in keys
        assert ("laneige", "rankedIn", "lip_care") in keys
        assert ("laneige", "ownedByGroup", "AMOREPACIFIC") in keys
        assert ("e.l.f.", "hasProduct", "B0AAAAAAA2") in keys
        assert ("cosrx", "competesWith", "laneige") in keys
        assert ("lip_care", "parentCategory", "skin_care") in keys
        assert ("B0AAAAAAA1", "hasAlert", "rank_drop") in keys
        assert not any("unknown" in k or "fresh" in k for k in keys)
        assert not any(k[1] in {"hasPosition", "ownedBy"} for k in keys)
        sos = next(
            t for t in out["triples"] if t["predicate"] == "hasSoS" and t["subject"] == "cosrx"
        )
        assert sos["properties"]["as_of"] == "2026-09-01"
        assert sos["properties"]["original_predicate"] == "hasSoS"

    def test_metadata(self):
        meta = mig.migrate(json.loads(json.dumps(FIXTURE))).data["entity_metadata"]
        assert "LANEIGE" not in meta
        assert meta["laneige"]["segment"] == "Premium"  # merged from the old key
        assert meta["laneige"]["type"] == "brand"
        assert "Brand" in meta["laneige"]["ontology_types"]
        assert meta["lip_care"]["type"] == "category"
        assert meta["B0AAAAAAA1"]["type"] == "product"

    def test_consistency_after(self):
        after = mig.consistency_report(mig.migrate(json.loads(json.dumps(FIXTURE))).data)
        assert after["case_duplicate_brands"] == 0
        assert after["asymmetric_symmetric_edges"] == 0
        assert after["violation:placeholder_brand"] == 0
        assert after["violation:non_canonical_predicate"] == 0
        assert after["violation:non_canonical_brand"] == 0
        assert after["violation:missing_as_of"] == 3  # undated, never invented
        assert after["untyped_entities"] == 0

    def test_consistency_before(self):
        before = mig.consistency_report(json.loads(json.dumps(FIXTURE)))
        assert before["case_duplicate_brands"] == 1  # LANEIGE / laneige
        assert before["violation:placeholder_brand"] == 2
        assert before["asymmetric_symmetric_edges"] >= 1


class TestCli:
    def test_dry_run_writes_nothing(self, kg_in, tmp_path, capsys):
        out = tmp_path / "out.json"
        assert mig.main(["--in", str(kg_in), "--out", str(out)]) == 0
        assert not out.exists()
        assert not list(tmp_path.glob("*.changes.json"))
        assert "placeholder_brand" in capsys.readouterr().out

    def test_apply_is_deterministic_and_input_untouched(self, kg_in, tmp_path):
        before = kg_in.read_bytes()
        a, b = tmp_path / "a.json", tmp_path / "b.json"
        assert mig.main(["--in", str(kg_in), "--out", str(a), "--apply"]) == 0
        assert mig.main(["--in", str(kg_in), "--out", str(b), "--apply"]) == 0
        assert a.read_bytes() == b.read_bytes()
        assert kg_in.read_bytes() == before
        log = json.loads((tmp_path / "a.json.changes.json").read_text(encoding="utf-8"))
        assert log["counts"]["removed"]["placeholder_brand"] == 2
        assert log["input_sha256"] and log["output_sha256"]
        assert log["changes"]

    def test_refuses_same_path(self, kg_in):
        before = kg_in.read_bytes()
        assert mig.main(["--in", str(kg_in), "--out", str(kg_in), "--apply"]) != 0
        assert kg_in.read_bytes() == before

    def test_refuses_repo_data(self, kg_in):
        target = REPO / "data" / "o5_migration_should_not_exist.json"
        assert mig.main(["--in", str(kg_in), "--out", str(target), "--apply"]) != 0
        assert not target.exists()

    def test_output_loads_in_knowledge_graph(self, kg_in, tmp_path):
        out = tmp_path / "out.json"
        assert mig.main(["--in", str(kg_in), "--out", str(out), "--apply"]) == 0
        kg = KnowledgeGraph(persist_path=str(out), auto_save=False, auto_load=True)
        preds = {r.predicate for r in kg.triples}
        assert RelationType.HAS_SOS in preds
        assert RelationType.HAS_POSITION not in preds
        assert len(kg.triples) == len(json.loads(out.read_text(encoding="utf-8"))["triples"])
