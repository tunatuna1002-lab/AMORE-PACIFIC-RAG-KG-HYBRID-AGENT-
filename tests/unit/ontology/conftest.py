"""Shared snapshot fixture for the F9 ontology tests (builder / materializer / CQs).

~30 rank records across three categories (beauty > skin_care > lip_care):
- lip_care: LANEIGE 3 products (ranks 1/3/8), COSRX 2 (2/15), 5 fillers -> LANEIGE SoS 0.30
- skin_care: "laneige" (lower-case spelling) 1 product, COSRX 3, fillers
- beauty: "Laneige" (title-case spelling) 1 product, fillers
Brand spellings deliberately vary so canonicalisation is exercised.
"""

from __future__ import annotations

from typing import Any

import pytest

from src.ontology.knowledge_graph import KnowledgeGraph

SNAPSHOT_DATE = "2026-09-10"
OLD = "2026-01-01"
RECENT = "2026-09-07"  # 3 days before the snapshot -> NewEntrant


def _rec(cat: str, brand: str, asin: str, rank: int, first_seen: str = OLD, **kw) -> dict:
    return {
        "category_id": cat,
        "brand": brand,
        "asin": asin,
        "product_name": f"{brand} product {asin}",
        "rank": rank,
        "price": kw.get("price", 20.0),
        "rating": kw.get("rating", 4.5),
        "first_seen": first_seen,
    }


LIP_CARE = [
    _rec("lip_care", "LANEIGE", "B0LAN001", 1),
    _rec("lip_care", "COSRX", "B0COS001", 2),
    _rec("lip_care", "LANEIGE", "B0LAN002", 3, first_seen=RECENT),
    _rec("lip_care", "Summer Fridays", "B0SUM001", 4),
    _rec("lip_care", "Burt's Bees", "B0BUR001", 5),
    _rec("lip_care", "eos", "B0EOS001", 6),
    _rec("lip_care", "Aquaphor", "B0AQU001", 7),
    _rec("lip_care", "LANEIGE", "B0LAN003", 8),
    _rec("lip_care", "Burt's Bees", "B0BUR002", 12),
    _rec("lip_care", "COSRX", "B0COS002", 15),
]
SKIN_CARE = [
    _rec("skin_care", "COSRX", "B0COS010", 1),
    _rec("skin_care", "CeraVe", "B0CER001", 2),
    _rec("skin_care", "COSRX", "B0COS011", 3),
    _rec("skin_care", "laneige", "B0LAN010", 5),
    _rec("skin_care", "Anua", "B0ANU001", 6),
    _rec("skin_care", "COSRX", "B0COS012", 9),
    _rec("skin_care", "CeraVe", "B0CER002", 11),
    _rec("skin_care", "innisfree", "B0INN001", 14),
    _rec("skin_care", "Tatcha", "B0TAT001", 20),
    _rec("skin_care", "Anua", "B0ANU002", 33),
]
BEAUTY = [
    _rec("beauty", "CeraVe", "B0CER001", 1),
    _rec("beauty", "e.l.f.", "B0ELF001", 2),
    _rec("beauty", "Laneige", "B0LAN001", 4),
    _rec("beauty", "Maybelline", "B0MAY001", 7),
    _rec("beauty", "NYX", "B0NYX001", 9),
    _rec("beauty", "COSRX", "B0COS010", 13),
    _rec("beauty", "e.l.f.", "B0ELF002", 18),
    _rec("beauty", "CeraVe", "B0CER003", 25),
    _rec("beauty", "Rhode", "B0RHO001", 40),
    _rec("beauty", "Maybelline", "B0MAY002", 55),
]

RECORDS: list[dict[str, Any]] = [*LIP_CARE, *SKIN_CARE, *BEAUTY]

HIERARCHY = {
    "categories": {
        "beauty": {"name": "Beauty & Personal Care", "level": 0, "parent_id": None},
        "skin_care": {"name": "Skin Care", "level": 1, "parent_id": "beauty"},
        "lip_care": {"name": "Lip Care", "level": 2, "parent_id": "skin_care"},
    }
}

GROUPS = {"AMOREPACIFIC": ["LANEIGE", "INNISFREE"]}

# brand_metrics share_of_shelf is PERCENT (MetricCalculator contract)
METRICS = {
    "brand_metrics": [
        {
            "brand_name": "LANEIGE",
            "category_id": "lip_care",
            "share_of_shelf": 30.0,
            "avg_rank": 4.0,
            "product_count": 3,
            "is_laneige": True,
        },
        {
            "brand_name": "COSRX",
            "category_id": "lip_care",
            "share_of_shelf": 20.0,
            "avg_rank": 8.5,
            "product_count": 2,
        },
    ],
    "market_metrics": [{"category_id": "lip_care", "hhi": 0.18, "cpi": 105.0}],
}


@pytest.fixture
def snapshot_records() -> list[dict[str, Any]]:
    return [dict(r) for r in RECORDS]


@pytest.fixture
def snapshot_metrics() -> dict[str, Any]:
    return {k: [dict(x) for x in v] for k, v in METRICS.items()}


@pytest.fixture
def hierarchy() -> dict[str, Any]:
    return HIERARCHY


@pytest.fixture
def groups() -> dict[str, list[str]]:
    return {k: list(v) for k, v in GROUPS.items()}


@pytest.fixture
def fresh_kg() -> KnowledgeGraph:
    return KnowledgeGraph(persist_path=None, auto_load=False, auto_save=False)
