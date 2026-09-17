"""
ReAct 읽기 전용 도구 실행기 검증 (결정 D2)

실제 SQLite(임시 파일, 운영 스키마)와 실제 KnowledgeGraph(임시 경로)를 쓴다.
"""

import sqlite3

import pytest

from src.core.react_tools import build_react_tool_executor
from src.domain.entities.relations import Relation, RelationType
from src.ontology.knowledge_graph import KnowledgeGraph
from src.tools.storage.sqlite_storage import SQLiteStorage


@pytest.fixture
def db_path(tmp_path):
    path = tmp_path / "amore.db"
    conn = sqlite3.connect(path)
    conn.executescript(SQLiteStorage.SCHEMA)
    rows = [
        ("2026-09-10", "lip_care", 1, "A1", "Lip Sleeping Mask", "LANEIGE", 24.0),
        ("2026-09-10", "lip_care", 2, "A2", "Aquaphor Lip Repair", "Aquaphor", 5.0),
        ("2026-09-10", "lip_care", 3, "A3", "Aquaphor Lip Balm", "Aquaphor", 6.0),
        ("2026-09-10", "lip_care", 4, "A4", "Burt's Bees Balm", "Burt's Bees", 4.0),
    ]
    conn.executemany(
        "INSERT INTO raw_data (snapshot_date, category_id, rank, asin, product_name, brand, price)"
        " VALUES (?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
    conn.execute(
        "INSERT INTO brand_metrics (snapshot_date, category_id, brand, sos, product_count)"
        " VALUES ('2026-09-10', 'lip_care', 'LANEIGE', 25.0, 1)"
    )
    conn.execute(
        "INSERT INTO market_metrics (snapshot_date, category_id, hhi)"
        " VALUES ('2026-09-10', 'lip_care', 0.375)"
    )
    conn.commit()
    conn.close()
    return path


@pytest.fixture
def kg(tmp_path):
    graph = KnowledgeGraph(persist_path=str(tmp_path / "kg.json"), auto_save=False)
    graph.add_relation(
        Relation(subject="laneige", predicate=RelationType.COMPETES_WITH, object="aquaphor")
    )
    graph.add_relation(
        Relation(
            subject="laneige",
            predicate=RelationType.HAS_PRODUCT,
            object="A1",
            properties={"product_name": "Lip Sleeping Mask", "category": "lip_care"},
        )
    )
    return graph


@pytest.fixture
def executor(db_path, kg):
    return build_react_tool_executor(knowledge_graph=kg, db_path=db_path)


@pytest.mark.asyncio
async def test_registers_exactly_the_read_only_tools(executor):
    assert set(executor.get_available_tools()) == {
        "query_data",
        "query_knowledge_graph",
        "calculate_metrics",
        "direct_answer",
    }


@pytest.mark.asyncio
async def test_query_data_returns_db_facts_for_free_text(executor):
    result = await executor.execute("query_data", {"category": "Lip Care", "brand": "LANEIGE"})

    assert result.success, result.error
    facts = result.data["facts"]
    assert any(f.get("hhi") == 0.375 for f in facts)
    assert result.data["entities"]["categories"] == ["lip_care"]


@pytest.mark.asyncio
async def test_query_data_without_entities_reports_nothing_found(executor):
    result = await executor.execute("query_data", {"category": "날씨"})

    assert result.success
    assert result.data["facts"] == []
    assert "message" in result.data


@pytest.mark.asyncio
async def test_query_knowledge_graph_competitors_is_case_insensitive(executor):
    result = await executor.execute(
        "query_knowledge_graph", {"entity": "LANEIGE", "relation": "competitors"}
    )

    assert result.success, result.error
    assert [c["brand"] for c in result.data["competitors"]] == ["aquaphor"]


@pytest.mark.asyncio
async def test_query_knowledge_graph_products(executor):
    result = await executor.execute(
        "query_knowledge_graph", {"entity": "laneige", "relation": "products"}
    )

    assert result.data["products"][0]["asin"] == "A1"


@pytest.mark.asyncio
async def test_calculate_metrics_uses_metric_calculator_on_latest_snapshot(executor):
    result = await executor.execute(
        "calculate_metrics",
        {"metric_type": "sos", "brands": ["LANEIGE"], "category_id": "lip_care"},
    )

    assert result.success, result.error
    data = result.data
    assert data["snapshot_date"] == "2026-09-10"
    assert data["categories"]["lip_care"]["brands"]["LANEIGE"]["sos"] == pytest.approx(25.0)
    # HHI는 브랜드 점유율 제곱합: (1/4)^2 + (2/4)^2 + (1/4)^2
    assert data["categories"]["lip_care"]["hhi"] == pytest.approx(0.375)


@pytest.mark.asyncio
async def test_tools_never_create_missing_db(tmp_path, kg):
    missing = tmp_path / "nope.db"
    executor = build_react_tool_executor(knowledge_graph=kg, db_path=missing)

    result = await executor.execute("calculate_metrics", {"category_id": "lip_care"})

    assert result.success
    assert "message" in result.data
    assert not missing.exists()
