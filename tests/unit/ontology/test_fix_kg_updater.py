"""Bug D2(b): KGUpdaterMixin.load_from_metrics_data must store SoS as a FRACTION.

brain.py and dashboard_exporter.py both write ``sos`` = percent/100 into the same
entity-metadata key; kg_updater wrote the raw percent.
"""

import pytest

from src.ontology.knowledge_graph import KnowledgeGraph


@pytest.fixture
def kg(tmp_path) -> KnowledgeGraph:
    return KnowledgeGraph(persist_path=str(tmp_path / "kg.json"), auto_load=False, auto_save=False)


def test_brand_metadata_sos_is_fraction(kg):
    kg.load_from_metrics_data(
        {"brand_metrics": [{"brand_name": "LANEIGE", "share_of_shelf": 3.0, "is_laneige": True}]}
    )
    assert kg.get_entity_metadata("LANEIGE")["sos"] == pytest.approx(0.03)


def test_brand_metadata_sos_none_stays_none(kg):
    kg.load_from_metrics_data({"brand_metrics": [{"brand_name": "X", "share_of_shelf": None}]})
    assert kg.get_entity_metadata("X")["sos"] is None
