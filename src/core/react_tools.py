"""
ReAct 읽기 전용 도구
====================
ReActAgent의 ALLOWED_ACTIONS 중 실행기가 한 번도 등록되지 않았던 3종을 구현한다
(결정 D2, docs/plans/risk-remediation-decisions-2026-09-17.md).

- query_data: 크롤 DB(SQLite) 수치 사실 — MetricFactsProvider 재사용
- query_knowledge_graph: KG 관계 조회 — 경쟁사·제품·카테고리 브랜드
- calculate_metrics: 최신 스냅샷 raw_data로 SoS·HHI·CPI 계산 — MetricCalculator 재사용

원칙
----
- 모두 읽기 전용이다. DB는 `mode=ro`로 열고 파일이 없으면 만들지 않는다. KG에 쓰지 않는다.
- 크롤처럼 부작용이 있는 도구는 등록하지 않는다.
- DecisionMaker가 쓰는 대시보드 도구 실행기와 섞지 않도록 별도 ToolExecutor를 만든다.
  섞으면 ReAct를 끈 상태에서도 DecisionMaker의 도구 목록이 바뀐다.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from .tools import ToolExecutor

logger = logging.getLogger(__name__)

MAX_ITEMS = 20
MAX_CATEGORIES = 5


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [v.strip() for v in value.split(",") if v.strip()]
    if isinstance(value, list | tuple):
        return [str(v).strip() for v in value if str(v).strip()]
    return [str(value)]


class ReActReadOnlyTools:
    """ReAct 읽기 전용 도구 모음."""

    def __init__(
        self,
        knowledge_graph: Any | None = None,
        db_path: str | Path | None = None,
    ):
        self.kg = knowledge_graph
        self._db_path = Path(db_path) if db_path else None
        self._linker: Any | None = None

    # ── 공통 ─────────────────────────────────────────────────────────

    def _resolve_db_path(self) -> Path:
        if self._db_path is not None:
            return self._db_path
        from src.tools.storage.sqlite_storage import get_sqlite_storage

        return Path(get_sqlite_storage().db_path)

    def _extract_entities(self, *texts: Any) -> dict[str, list[str]]:
        if self._linker is None:
            from src.rag.entity_linker import EntityLinker

            self._linker = EntityLinker(use_spacy=False)
        text = " ".join(str(t) for t in texts if t)
        extracted = self._linker.extract_entities(text) if text else {}
        return {
            "brands": list(extracted.get("brands") or []),
            "categories": list(extracted.get("categories") or []),
        }

    # ── query_data ───────────────────────────────────────────────────

    async def query_data(self, **params: Any) -> dict[str, Any]:
        """브랜드·카테고리(자유 텍스트 허용)의 크롤 DB 수치 사실."""
        from src.rag.metric_facts import MetricFactsProvider

        category_text = params.get("category") or params.get("category_id")
        brand_values = _as_list(params.get("brand"))
        entities = self._extract_entities(category_text, *brand_values)
        for brand in brand_values:
            if brand.lower() not in entities["brands"]:
                entities["brands"].append(brand.lower())

        limit = params.get("limit")
        limit = limit if isinstance(limit, int) and 0 < limit <= MAX_ITEMS else MAX_ITEMS

        provider = MetricFactsProvider(db_path=self._resolve_db_path())
        result: dict[str, Any] = {"entities": entities, "as_of": provider.as_of}
        if not entities["brands"] and not entities["categories"]:
            result["facts"] = []
            result["message"] = "질의에서 브랜드·카테고리를 찾지 못했습니다."
            return result

        facts = await provider.collect(entities)
        result["facts"] = facts[:limit]
        if not facts:
            result["message"] = "해당 브랜드·카테고리의 DB 수치가 없습니다."
        return result

    # ── query_knowledge_graph ────────────────────────────────────────

    def _entity_variants(self, entity: str) -> list[str]:
        variants = [entity, entity.lower(), entity.upper(), entity.title()]
        return list(dict.fromkeys(v for v in variants if v))

    async def query_knowledge_graph(self, **params: Any) -> dict[str, Any]:
        """엔티티의 경쟁사·제품·카테고리 브랜드·메타데이터."""
        entity = str(params.get("entity") or "").strip()
        relation = str(params.get("relation") or params.get("relation_type") or "all").lower()
        if self.kg is None:
            return {"entity": entity, "message": "지식 그래프가 연결되지 않았습니다."}
        if not entity:
            return {"entity": entity, "message": "entity 파라미터가 필요합니다."}

        result: dict[str, Any] = {"entity": entity, "relation": relation}
        variants = self._entity_variants(entity)

        def _first_nonempty(fn: Any) -> list[dict[str, Any]]:
            for name in variants:
                found = fn(name)
                if found:
                    return found[:MAX_ITEMS]
            return []

        if relation in ("competitors", "competes_with", "all"):
            result["competitors"] = _first_nonempty(self.kg.get_competitors)
        if relation in ("products", "has_product", "all"):
            result["products"] = _first_nonempty(self.kg.get_brand_products)
        if relation in ("category", "category_brands", "all"):
            category = self._extract_entities(entity)["categories"]
            names = category or variants
            result["category_brands"] = []
            for name in names:
                brands = self.kg.get_category_brands(name)
                if brands:
                    result["category_brands"] = brands[:MAX_ITEMS]
                    break
        if relation == "all":
            result["metadata"] = next(
                (m for m in (self.kg.get_entity_metadata(v) for v in variants) if m), {}
            )

        if not any(result.get(k) for k in ("competitors", "products", "category_brands")):
            result["message"] = "지식 그래프에서 해당 관계를 찾지 못했습니다."
        return result

    # ── calculate_metrics ────────────────────────────────────────────

    async def calculate_metrics(self, **params: Any) -> dict[str, Any]:
        """최신 스냅샷(또는 AMORE_DATA_AS_OF 이하 최신)의 raw_data로 SoS·HHI·CPI 계산."""
        import aiosqlite

        from src.rag.metric_facts import AS_OF_ENV
        from src.tools.calculators.metric_calculator import MetricCalculator

        brands = _as_list(params.get("brands") or params.get("brand"))
        category_text = params.get("category_id") or params.get("category")
        categories = self._extract_entities(category_text, *brands)["categories"]
        if not categories and category_text:
            categories = [str(category_text).strip().lower().replace(" ", "_")]

        db_path = self._resolve_db_path()
        if not db_path.exists():
            return {"message": "크롤 DB가 없어 지표를 계산할 수 없습니다."}

        as_of = os.environ.get(AS_OF_ENV) or None
        async with aiosqlite.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
            conn.row_factory = aiosqlite.Row
            if as_of:
                cursor = await conn.execute(
                    "SELECT MAX(snapshot_date) FROM raw_data WHERE snapshot_date <= ?", (as_of,)
                )
            else:
                cursor = await conn.execute("SELECT MAX(snapshot_date) FROM raw_data")
            row = await cursor.fetchone()
            snapshot_date = row[0] if row else None
            if not snapshot_date:
                return {"message": "raw_data에 스냅샷이 없습니다."}

            if not categories:
                cursor = await conn.execute(
                    "SELECT DISTINCT category_id FROM raw_data WHERE snapshot_date = ?",
                    (snapshot_date,),
                )
                categories = [r["category_id"] for r in await cursor.fetchall()]

            calculator = MetricCalculator()
            out: dict[str, Any] = {}
            for category in categories[:MAX_CATEGORIES]:
                cursor = await conn.execute(
                    "SELECT rank, brand, price FROM raw_data"
                    " WHERE snapshot_date = ? AND category_id = ?",
                    (snapshot_date, category),
                )
                records = [dict(r) for r in await cursor.fetchall()]
                if not records:
                    continue
                for record in records:
                    record["brand"] = record.get("brand") or ""
                brand_metrics = {
                    brand: {
                        "sos": calculator.calculate_sos(records, brand),
                        "avg_rank": calculator.calculate_brand_avg_rank(records, brand),
                        "cpi": calculator.calculate_cpi(records, brand),
                    }
                    for brand in brands
                }
                out[category] = {
                    "product_count": len(records),
                    "hhi": calculator.calculate_hhi(records),
                    "brands": brand_metrics,
                }

        result: dict[str, Any] = {
            "snapshot_date": snapshot_date,
            "metric_type": params.get("metric_type"),
            "categories": out,
            "note": "SoS·CPI는 %·100 기준, HHI는 0-1 스케일",
        }
        if not out:
            result["message"] = "해당 카테고리의 스냅샷 데이터가 없습니다."
        return result


def build_react_tool_executor(
    knowledge_graph: Any | None = None,
    db_path: str | Path | None = None,
) -> ToolExecutor:
    """ReAct 전용 ToolExecutor를 만들고 읽기 전용 도구 3종을 등록한다."""
    tools = ReActReadOnlyTools(knowledge_graph=knowledge_graph, db_path=db_path)
    executor = ToolExecutor()
    executor.register_executor("query_data", tools.query_data)
    executor.register_executor("query_knowledge_graph", tools.query_knowledge_graph)
    executor.register_executor("calculate_metrics", tools.calculate_metrics)
    return executor
