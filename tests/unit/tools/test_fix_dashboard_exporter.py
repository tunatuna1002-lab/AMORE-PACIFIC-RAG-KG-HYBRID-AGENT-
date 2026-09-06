"""Bug D18: DashboardExporter inference context must not fabricate rank/streak/rating data,
and must convert percent SoS through the shared unit helper.

``_generate_ontology_insights`` is the closest entry short of ``export_dashboard_data``
(which needs SQLite); the KG/reasoner are swapped for tmp-path instances after construction.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.ontology.knowledge_graph import KnowledgeGraph
from src.ontology.reasoner import OntologyReasoner
from src.ontology.rules import register_all_rules
from src.tools.exporters.dashboard_exporter import DashboardExporter


@pytest.fixture
def exporter(tmp_path) -> DashboardExporter:
    with (
        patch("src.tools.exporters.dashboard_exporter.SheetsWriter") as sheets,
        patch("src.tools.exporters.dashboard_exporter.SQLiteStorage") as sqlite,
        patch.object(DashboardExporter, "_init_ontology"),
    ):
        sheets.return_value = MagicMock(initialize=AsyncMock(return_value=True))
        sqlite.return_value = MagicMock(initialize=AsyncMock(return_value=True))
        exp = DashboardExporter(spreadsheet_id="x", enable_ontology=True)
    kg = KnowledgeGraph(persist_path=str(tmp_path / "kg.json"), auto_load=False, auto_save=False)
    exp._knowledge_graph = kg
    exp._reasoner = OntologyReasoner(kg)
    register_all_rules(exp._reasoner)
    return exp


DASHBOARD_DATA = {
    "brand": {
        "kpis": {"sos": 3.0, "avg_rank": 8.5, "hhi": 0.10},  # sos is PERCENT
        "competitors": [
            {"brand": "COSRX", "sos": 10.0, "avg_rank": 4.0, "product_count": 10},
            {"brand": "LANEIGE", "sos": 3.0, "avg_rank": 8.5, "product_count": 3},
        ],
    },
    "categories": {"lip_care": {"cpi": 100.0, "best_rank": 1}},
    "products": {
        "B0LANE1": {"asin": "B0LANE1", "rank": 1, "category": "lip_care"},
        "B0LANE2": {"asin": "B0LANE2", "rank": 5, "category": "lip_care"},
    },
    "home": {"action_items": [{"asin": "B0LANE1", "rank": 1, "rank_change_7d": -2}]},
}


def test_inference_context_has_no_fabricated_defaults(exporter):
    result = exporter._generate_ontology_insights([], DASHBOARD_DATA)
    ctx = result["context"]

    assert "streak_days" not in ctx
    assert "rating_gap" not in ctx
    # rank_change_7d comes from the computed action items, not a hard-coded 0
    assert ctx["rank_change_7d"] == -2
    # unit boundary: percent -> fraction via the shared helper
    assert ctx["sos"] == pytest.approx(0.03)
    assert ctx["top1_sos"] == pytest.approx(0.10)

    fired = {i["rule_name"] for i in result["inferences"]}
    assert "strong_rating_position" not in fired
    assert "market_dominance_fragmented" not in fired


def test_rank_change_7d_omitted_when_unknown(exporter):
    data = {**DASHBOARD_DATA, "home": {}}
    ctx = exporter._generate_ontology_insights([], data)["context"]
    assert "rank_change_7d" not in ctx


def test_missing_sos_is_zero_fraction(exporter):
    data = {**DASHBOARD_DATA, "brand": {"kpis": {}, "competitors": []}}
    ctx = exporter._generate_ontology_insights([], data)["context"]
    assert ctx["sos"] == 0.0
    assert ctx["top1_sos"] == 0.0
