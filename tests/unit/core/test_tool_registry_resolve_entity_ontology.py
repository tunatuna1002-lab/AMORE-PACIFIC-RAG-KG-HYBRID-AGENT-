"""resolve_entity 도구 — 온톨로지 등록부 연결 (트랙 O2, 결정 OA-6)

- 플래그 `ontology.use_class_reasoning` OFF: 결과가 O2 이전 코드(0b56e4b)와 같다
  (`fixtures/resolve_entity_off_snapshot.json`, O2 이전 코드로 기록).
- ON: 브랜드 카드 metadata에 등록부 id·클래스 소속이 실리고, 그룹·클래스 언급도 카드가 된다.

기록(O2 이전 코드에서만): ``.venv/bin/python -m tests.unit.core.test_tool_registry_resolve_entity_ontology --regen``
"""

from __future__ import annotations

import asyncio
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import pytest

from src.core.tool_registry import ToolRegistry, tool_evidence
from src.infrastructure.feature_flags import FeatureFlags
from src.rag.hybrid_retriever import HybridRetriever
from src.rag.metric_facts import AS_OF_ENV, MetricFactsProvider
from tests.unit.rag.evidence_pipeline_fixtures import (
    AS_OF,
    FakeDocRetriever,
    make_kg,
    make_metrics_db,
)

SNAPSHOT_PATH = Path(__file__).parent / "fixtures" / "resolve_entity_off_snapshot.json"
FLAG_ENV = "FF_ONTOLOGY_USE_CLASS_REASONING"
TEXTS = (
    "라네즈 립케어 HHI",
    "IT Cosmetics Face Powder CPI",
    "아모레퍼시픽 브랜드",
    "COSRX 코스알엑스 SoS",
    "오늘 날씨",
    "unknown chi fresh",
    "Lip Sleeping Mask 순위",
)
_ENABLE = (
    "FF_ONTOLOGY_USE_ONTOLOGY_KG",
    "FF_RETRIEVER_USE_DB_METRIC_FACTS",
    "FF_REASONER_USE_UNIFIED_REASONER",
    "FF_REASONER_USE_OWL_REASONER",
)


def _registry(tmp_path: Path) -> ToolRegistry:
    retriever = HybridRetriever(
        knowledge_graph=make_kg(tmp_path),
        doc_retriever=FakeDocRetriever(),
        metric_facts_provider=MetricFactsProvider(make_metrics_db(tmp_path), as_of=AS_OF),
    )
    return ToolRegistry(retriever)


async def _run_all(tmp_path: Path) -> dict[str, Any]:
    registry = _registry(tmp_path)
    out: dict[str, Any] = {}
    for text in TEXTS:
        result = await registry.execute("resolve_entity", {"text": text})
        assert result.success, result.error
        out[text] = result.data
    return out


def _dump(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, sort_keys=True, indent=1)


@pytest.fixture(autouse=True)
def flags(monkeypatch):
    for name in _ENABLE:
        monkeypatch.setenv(name, "true")
    monkeypatch.setenv(FLAG_ENV, "false")
    monkeypatch.delenv(AS_OF_ENV, raising=False)
    FeatureFlags.reset_instance()
    yield
    FeatureFlags.reset_instance()


def _flag_on(monkeypatch) -> None:
    monkeypatch.setenv(FLAG_ENV, "true")
    FeatureFlags.reset_instance()


# ── OFF: 변경 없음 ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_resolve_entity_flag_off_is_unchanged(tmp_path):
    expected = json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8"))
    got = await _run_all(tmp_path)
    for text in TEXTS:
        assert _dump(got[text]) == _dump(expected[text]), text


# ── ON: 등록부 id·클래스 소속 ────────────────────────────────────


@pytest.mark.asyncio
async def test_resolve_entity_on_attaches_registry_id_and_classes(tmp_path, monkeypatch):
    _flag_on(monkeypatch)
    result = await _registry(tmp_path).execute("resolve_entity", {"text": "라네즈 립케어 HHI"})
    assert result.success, result.error
    cards = {c.metadata["entity_type"]: c for c in tool_evidence(result)}
    brand = cards["brand"].metadata
    assert brand["canonical_id"] == "laneige"  # canonical 표기는 그대로
    assert brand["registry_id"] == "laneige"
    assert brand["registry"] == "ontology:registry"
    assert {"Brand", "AmorepacificBrand", "KBeautyBrand", "PremiumBrand"} <= set(brand["classes"])
    assert brand["group"] == "amorepacific"
    assert cards["category"].metadata["registry_id"] == "lip_care"
    assert result.data["entities"]["brand_ids"] == ["laneige"]


@pytest.mark.asyncio
async def test_resolve_entity_on_recognizes_registry_only_brand(tmp_path, monkeypatch):
    _flag_on(monkeypatch)
    result = await _registry(tmp_path).execute(
        "resolve_entity", {"text": "IT Cosmetics Face Powder CPI"}
    )
    brand_cards = [c for c in tool_evidence(result) if c.metadata["entity_type"] == "brand"]
    assert [c.metadata["canonical_id"] for c in brand_cards] == ["it cosmetics"]
    assert brand_cards[0].metadata["registry_id"] == "it_cosmetics"
    # 등록부에 그룹이 없는 브랜드: 소속 클래스는 Brand뿐 (추정으로 채우지 않는다)
    assert brand_cards[0].metadata["classes"] == ["Brand"]
    assert brand_cards[0].metadata["group"] is None


@pytest.mark.asyncio
async def test_resolve_entity_on_group_and_class_cards(tmp_path, monkeypatch):
    _flag_on(monkeypatch)
    result = await _registry(tmp_path).execute(
        "resolve_entity", {"text": "아모레퍼시픽 브랜드들 중 K-Beauty 브랜드"}
    )
    cards = tool_evidence(result)
    by_kind = {(c.metadata["entity_type"], c.metadata["canonical_id"]): c for c in cards}
    group = by_kind[("group", "amorepacific")].metadata
    assert "laneige" in group["members"] and "cosrx" in group["members"]
    ap_class = by_kind[("class", "AmorepacificBrand")].metadata
    assert "sulwhasoo" in ap_class["instances"]
    assert ("class", "KBeautyBrand") in by_kind
    assert result.data["entities"]["groups"] == ["amorepacific"]
    assert result.data["entities"]["classes"] == ["AmorepacificBrand", "KBeautyBrand"]


@pytest.mark.asyncio
async def test_resolve_entity_on_drops_placeholder_brands(tmp_path, monkeypatch):
    _flag_on(monkeypatch)
    result = await _registry(tmp_path).execute("resolve_entity", {"text": "unknown chi fresh"})
    assert not [c for c in tool_evidence(result) if c.metadata["entity_type"] == "brand"]


if __name__ == "__main__" and "--regen" in sys.argv:
    import os

    for _name in _ENABLE:
        os.environ[_name] = "true"
    os.environ[FLAG_ENV] = "false"
    os.environ.pop(AS_OF_ENV, None)
    _data = asyncio.run(_run_all(Path(tempfile.mkdtemp())))
    SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    SNAPSHOT_PATH.write_text(_dump(_data) + "\n", encoding="utf-8")
    print(f"wrote {SNAPSHOT_PATH}")
