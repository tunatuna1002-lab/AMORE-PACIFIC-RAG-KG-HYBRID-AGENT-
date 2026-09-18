"""[2026-09 사후] O0-A: scripts/rescore_l3_l4.py — 저장된 trace만으로 새 지표를 재채점."""

import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module")
def rescore():
    spec = importlib.util.spec_from_file_location(
        "rescore_l3_l4", REPO / "scripts" / "rescore_l3_l4.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _item(item_id, edges, inferences, fired, stored_recall, error=None):
    return {
        "item_id": item_id,
        "l3": {"kg_edge_recall": stored_recall},
        "trace": {
            "item_id": item_id,
            "error": error,
            "l3_kg_query": {"kg_entities_found": [], "kg_edges_found": edges, "ontology_facts": []},
            "l4_ontology": {"inferences": inferences},
            "rule_evaluation": {"fired": fired},
        },
    }


INF_BAD = {
    "rule_name": "r1",
    "insight_type": "market_position",
    "insight": "x",
    "confidence": 0.9,
    "evidence": {"context_snapshot": {"brand": "laneige", "as_of": "2026-08-31"}},
    "related_entities": [""],
}


@pytest.fixture
def workspace(tmp_path):
    gold = tmp_path / "gold.jsonl"
    gold.write_text(
        "\n".join(
            json.dumps(row)
            for row in [
                {"id": "a", "gold": {"kg_edges": ["cosrx -ownedByGroup-> amorepacific"]}},
                {"id": "b", "gold": {"kg_edges": []}},
                {"id": "c", "gold": {"kg_edges": ["laneige -competesWith-> nivea"]}},
            ]
        )
        + "\n"
    )
    for run in ("x-run1", "x-run2"):
        (tmp_path / run).mkdir()
        items = [
            _item("a", ["cosrx -ownedBy-> amorepacific"], [INF_BAD], ["r1"], 0.0),
            _item("b", [], [], [], 1.0),
            _item("c", ["laneige -competesWith-> nivea"], [], [], 1.0 if run == "x-run1" else 0.5),
            _item("d", [], [], [], 1.0, error="agent_timeout"),
        ]
        (tmp_path / run / "report.json").write_text(
            json.dumps({"config": {"git_commit": "abc"}, "items": items})
        )
    return tmp_path, gold


def test_rescore_report_counts_and_checks_legacy_recall(rescore, workspace):
    base, gold = workspace
    gold_by_id = rescore.load_jsonl_by_id(gold)
    report = json.loads((base / "x-run2" / "report.json").read_text())
    result = rescore.rescore_report(
        report, gold_by_id, rescore.L3KGMetrics(), rescore.L4OntologyMetrics()
    )
    assert result["scored"] == 3
    assert result["errored"] == ["d"]
    # x-run2의 c는 저장값 0.5가 재계산 1.0과 달라 드러난다
    assert result["legacy_recall_mismatch"] == ["c"]
    assert result["l3"]["gold_edge_items"] == 2
    assert result["l3"]["kg_edge_recall_gold_only"] == 0.5
    assert result["l3"]["recall_by_predicate"]["ownedByGroup"]["recall"] == 0.0
    assert result["l4"]["rule_constraint_violation_rate"] == 1.0
    assert result["l4"]["rule_violation_kinds"] == {"related_entity_invalid": 1}


def test_main_writes_outputs_and_subset(rescore, workspace, tmp_path, capsys):
    base, gold = workspace
    ids = tmp_path / "ids.jsonl"
    ids.write_text(json.dumps({"id": "c"}) + "\n")
    out = tmp_path / "out"
    code = rescore.main(
        [
            "--base",
            str(base),
            "--out",
            str(out),
            "--gold",
            str(gold),
            "--subset",
            f"x-sub:x={ids}",
            "x-run1",
            "x-run2",
        ]
    )
    assert code == 0
    summary = json.loads((out / "summary.json").read_text())["summary"]
    assert summary["x"]["runs"] == 2
    assert summary["x"]["l3.kg_edge_recall_gold_only"]["mean"] == 0.5
    assert summary["x-sub"]["l3.kg_edge_recall_gold_only"]["mean"] == 1.0
    assert summary["x-sub"]["l4.rule_constraint_violation_rate"]["mean"] is None
    assert (out / "x-run1.json").exists()
    assert "| x |" in capsys.readouterr().out
