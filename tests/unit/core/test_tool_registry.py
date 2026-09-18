"""
읽기 전용 도구 레지스트리 검증 (트랙 4-A, 설계 E4)

실제 객체: SQLite(임시 파일, 운영 스키마)·KnowledgeGraph(임시 경로, auto_save=False)·
HybridRetriever·MetricFactsProvider·EvidenceAdapter·규칙 추론기. 가짜는 문서 검색기
(색인 I/O·임베딩)뿐이다 — ``evidence_pipeline_fixtures.FakeDocRetriever``.
"""

from __future__ import annotations

import sqlite3

import jsonschema
import pytest

from src.core.tool_registry import (
    TOOL_DEFINITIONS,
    TOOL_NAMES,
    ToolRegistry,
    function_schemas,
    tool_evidence,
)
from src.domain.entities.evidence import EvidenceKind
from src.infrastructure.feature_flags import FeatureFlags
from src.rag.evidence_adapters import KG_NUMERIC_PREDICATES
from src.rag.hybrid_retriever import HybridRetriever
from src.rag.metric_facts import AS_OF_ENV, MetricFactsProvider
from tests.unit.rag.evidence_pipeline_fixtures import (
    AS_OF,
    KG_HHI_EDGE_VALUE,
    OPERATIONAL_SCHEMA,
    FakeDocRetriever,
    make_kg,
    make_metrics_db,
)

EXPECTED_NAMES = ("resolve_entity", "kg_neighbors", "get_metrics", "apply_rules", "search_docs")


@pytest.fixture(autouse=True)
def default_flags(monkeypatch):
    for name in (
        "FF_ONTOLOGY_USE_ONTOLOGY_KG",
        "FF_RETRIEVER_USE_DB_METRIC_FACTS",
        "FF_REASONER_USE_UNIFIED_REASONER",
        "FF_REASONER_USE_OWL_REASONER",
    ):
        monkeypatch.setenv(name, "true")
    monkeypatch.delenv(AS_OF_ENV, raising=False)
    FeatureFlags.reset_instance()
    yield
    FeatureFlags.reset_instance()


def _registry(tmp_path, provider: MetricFactsProvider | None = None) -> ToolRegistry:
    retriever = HybridRetriever(
        knowledge_graph=make_kg(tmp_path),
        doc_retriever=FakeDocRetriever(),
        metric_facts_provider=provider
        or MetricFactsProvider(make_metrics_db(tmp_path), as_of=AS_OF),
    )
    return ToolRegistry(retriever)


def _dominance_db(tmp_path):
    """LANEIGE Lip Care SoS 18%(DB 퍼센트 저장), HHI 0.10 — market_dominance_fragmented 조건."""
    path = tmp_path / "dominance.db"
    conn = sqlite3.connect(path)
    conn.executescript(OPERATIONAL_SCHEMA)
    conn.executemany(
        "INSERT INTO brand_metrics (snapshot_date, category_id, brand, sos, product_count)"
        " VALUES (?,?,?,?,?)",
        [(AS_OF, "lip_care", "LANEIGE", 18.0, 18), (AS_OF, "lip_care", "eos", 9.0, 9)],
    )
    conn.execute(
        "INSERT INTO market_metrics (snapshot_date, category_id, hhi) VALUES (?,?,?)",
        (AS_OF, "lip_care", 0.10),
    )
    conn.commit()
    conn.close()
    return path


# ── 정의·스키마 ─────────────────────────────────────────────────────


def test_registry_defines_exactly_the_five_read_only_tools():
    assert TOOL_NAMES == EXPECTED_NAMES
    assert tuple(d.name for d in TOOL_DEFINITIONS) == EXPECTED_NAMES


def test_function_schemas_are_valid_litellm_tools():
    schemas = function_schemas()

    assert [s["function"]["name"] for s in schemas] == list(EXPECTED_NAMES)
    for schema in schemas:
        assert schema["type"] == "function"
        function = schema["function"]
        assert function["description"]
        parameters = function["parameters"]
        jsonschema.Draft202012Validator.check_schema(parameters)
        assert parameters["type"] == "object"
        assert parameters["additionalProperties"] is False
        assert set(parameters["required"]) <= set(parameters["properties"])


def test_function_schemas_can_be_limited_to_available_names():
    schemas = function_schemas(["search_docs", "get_metrics", "not_a_tool"])

    assert [s["function"]["name"] for s in schemas] == ["get_metrics", "search_docs"]


def test_unbound_registry_offers_no_tools():
    registry = ToolRegistry()

    assert registry.get_available_tools() == []


@pytest.mark.asyncio
async def test_unbound_registry_execution_fails_without_raising():
    result = await ToolRegistry().execute("get_metrics", {"brand": "LANEIGE"})

    assert result.success is False
    assert "연결" in result.error


def test_bound_registry_lists_all_tools(tmp_path):
    assert _registry(tmp_path).get_available_tools() == list(EXPECTED_NAMES)


@pytest.mark.parametrize(
    ("env", "hidden"),
    [
        ("FF_ONTOLOGY_USE_ONTOLOGY_KG", "kg_neighbors"),
        ("FF_RETRIEVER_USE_DB_METRIC_FACTS", "get_metrics"),
    ],
)
def test_ablation_flags_hide_the_matching_tool(tmp_path, monkeypatch, env, hidden):
    monkeypatch.setenv(env, "false")
    FeatureFlags.reset_instance()

    tools = _registry(tmp_path).get_available_tools()

    assert hidden not in tools
    assert len(tools) == 4


def test_rule_flags_off_hide_apply_rules(tmp_path, monkeypatch):
    monkeypatch.setenv("FF_REASONER_USE_UNIFIED_REASONER", "false")
    monkeypatch.setenv("FF_REASONER_USE_OWL_REASONER", "false")
    FeatureFlags.reset_instance()

    assert "apply_rules" not in _registry(tmp_path).get_available_tools()


# ── 인자 검증 ─────────────────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("tool", "params", "fragment"),
    [
        ("crawl_amazon", {}, "등록되지 않은 도구"),
        ("get_metrics", {"brand": "LANEIGE", "sql": "SELECT 1"}, "sql"),
        ("search_docs", {}, "query"),
        ("search_docs", {"query": "HHI", "k": 0}, "k"),
        ("search_docs", {"query": "HHI", "k": "3"}, "k"),
        ("kg_neighbors", {"entity": "LANEIGE", "predicates": "ownedBy"}, "predicates"),
        ("get_metrics", {}, "brand"),
        ("apply_rules", {}, "brand"),
    ],
)
async def test_invalid_arguments_fail_with_reason(tmp_path, tool, params, fragment):
    result = await _registry(tmp_path).execute(tool, params)

    assert result.success is False
    assert fragment in result.error


# ── resolve_entity ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_resolve_entity_returns_canonical_ids_as_cards(tmp_path):
    result = await _registry(tmp_path).execute("resolve_entity", {"text": "라네즈 립케어 HHI"})

    assert result.success, result.error
    cards = tool_evidence(result)
    assert {c.kind for c in cards} == {EvidenceKind.OBSERVATION}
    assert all(c.source == "tool:resolve_entity" for c in cards)
    resolved = {(c.metadata["entity_type"], c.metadata["canonical_id"]) for c in cards}
    assert {("brand", "laneige"), ("category", "lip_care"), ("indicator", "hhi")} <= resolved
    assert result.data["entities"]["brands"] == ["laneige"]


@pytest.mark.asyncio
async def test_resolve_entity_without_match_reports_nothing(tmp_path):
    result = await _registry(tmp_path).execute("resolve_entity", {"text": "오늘 날씨"})

    assert result.success
    assert tool_evidence(result) == []
    assert result.data["message"]


# ── kg_neighbors ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_kg_neighbors_returns_structural_relations_without_numeric_edges(tmp_path):
    result = await _registry(tmp_path).execute("kg_neighbors", {"entity": "LANEIGE"})

    assert result.success, result.error
    cards = tool_evidence(result)
    assert cards and {c.kind for c in cards} == {EvidenceKind.RELATION}
    triples = {(c.subject, c.predicate, c.object) for c in cards}
    assert ("laneige", "ownedBy", "amorepacific") in triples
    assert ("laneige", "competesWith", "burt's bees") in triples
    assert ("laneige", "hasProduct", "B0LSM00001") in triples
    # 날짜 없는 KG 수치 엣지는 증거가 아니다 (E2)
    assert not {c.predicate for c in cards} & KG_NUMERIC_PREDICATES
    assert all(KG_HHI_EDGE_VALUE not in (c.object or "") for c in cards)
    assert result.data["excluded_by_reason"]["kg_numeric_undated"] >= 2


@pytest.mark.asyncio
async def test_kg_neighbors_filters_predicates(tmp_path):
    result = await _registry(tmp_path).execute(
        "kg_neighbors", {"entity": "laneige", "predicates": ["ownedBy"]}
    )

    assert [(c.subject, c.predicate, c.object) for c in tool_evidence(result)] == [
        ("laneige", "ownedBy", "amorepacific")
    ]


@pytest.mark.asyncio
async def test_kg_neighbors_numeric_predicate_request_returns_no_cards(tmp_path):
    result = await _registry(tmp_path).execute(
        "kg_neighbors", {"entity": "laneige", "predicates": ["hasSoS", "hasHHI"]}
    )

    assert result.success
    assert tool_evidence(result) == []


# ── get_metrics ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_get_metrics_reads_sqlite_at_provider_as_of(tmp_path):
    result = await _registry(tmp_path).execute(
        "get_metrics", {"brand": "LANEIGE", "category": "Lip Care"}
    )

    assert result.success, result.error
    cards = tool_evidence(result)
    assert cards and {c.kind for c in cards} == {EvidenceKind.METRIC}
    # 2026-09-11 행(LANEIGE SoS 3%)이 DB에 있지만 as_of 상한 2026-08-31을 넘는다
    assert {c.as_of for c in cards} == {AS_OF}
    sos = [c for c in cards if c.subject == "laneige" and c.predicate == "sos"]
    assert [(c.value, c.unit, c.object) for c in sos] == [(0.02, "ratio", "lip_care")]
    hhi = [c for c in cards if c.predicate == "hhi"]
    assert [c.value for c in hhi] == [0.0681]
    assert result.data["as_of"] == AS_OF


@pytest.mark.asyncio
async def test_get_metrics_respects_as_of_environment(tmp_path, monkeypatch):
    monkeypatch.setenv(AS_OF_ENV, AS_OF)
    registry = _registry(tmp_path, provider=MetricFactsProvider(make_metrics_db(tmp_path)))

    result = await registry.execute("get_metrics", {"category": "lip_care"})

    cards = tool_evidence(result)
    assert cards and {c.as_of for c in cards} == {AS_OF}
    assert [c.value for c in cards if c.predicate == "hhi"] == [0.0681]


@pytest.mark.asyncio
async def test_get_metrics_latest_snapshot_without_as_of(tmp_path):
    registry = _registry(tmp_path, provider=MetricFactsProvider(make_metrics_db(tmp_path)))

    result = await registry.execute("get_metrics", {"category": "lip_care"})

    assert [c.value for c in tool_evidence(result) if c.predicate == "hhi"] == [0.07]


# ── apply_rules ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_apply_rules_fires_market_dominance_with_derived_from(tmp_path):
    provider = MetricFactsProvider(_dominance_db(tmp_path), as_of=AS_OF)
    registry = _registry(tmp_path, provider=provider)

    result = await registry.execute("apply_rules", {"brand": "LANEIGE", "category": "Lip Care"})

    assert result.success, result.error
    cards = tool_evidence(result)
    inference = [c for c in cards if c.source == "rule:market_dominance_fragmented"]
    assert len(inference) == 1
    card = inference[0]
    assert card.kind == EvidenceKind.INFERENCE
    assert card.subject == "laneige"
    assert card.derived_from
    by_id = {c.id: c for c in cards}
    basis = [by_id[i] for i in card.derived_from]  # 근거 카드가 모두 결과에 있다
    assert {(b.predicate, b.value) for b in basis} >= {("sos", 0.18), ("hhi", 0.1)}
    assert "market_dominance_fragmented" in result.data["rule_evaluation"]["fired"]


@pytest.mark.asyncio
async def test_apply_rules_does_not_fire_dominance_for_low_share(tmp_path):
    result = await _registry(tmp_path).execute(
        "apply_rules", {"brand": "LANEIGE", "category": "lip_care"}
    )

    assert result.success, result.error
    sources = {c.source for c in tool_evidence(result)}
    assert "rule:market_dominance_fragmented" not in sources


# ── search_docs ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_search_docs_returns_document_cards(tmp_path):
    result = await _registry(tmp_path).execute("search_docs", {"query": "HHI 해석", "k": 1})

    assert result.success, result.error
    cards = tool_evidence(result)
    assert [(c.kind, c.metadata["chunk_id"]) for c in cards] == [
        (EvidenceKind.DOCUMENT, "metric_guide_hhi_0")
    ]
    assert result.data["search_method"] in ("dense_only", "hybrid_rrf")


def test_tool_result_data_is_json_serializable(tmp_path):
    import asyncio
    import json

    result = asyncio.run(_registry(tmp_path).execute("get_metrics", {"brand": "LANEIGE"}))

    json.dumps(result.to_dict())
