"""증거 카드 프롬프트 조립 테스트용 실제 객체 픽스처 (트랙 2-B).

가짜는 문서 검색기(색인 I/O·임베딩)뿐이다. KnowledgeGraph·MetricFactsProvider(SQLite)·
HybridRetriever·규칙 추론기는 실제 객체를 임시 경로로 만든다. KG는 반드시 임시
``persist_path``와 ``auto_save=False``로 연다 — 기본 경로는 원본 KG를 읽고 쓸 수 있다.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

from src.domain.entities.relations import Relation, RelationType
from src.ontology.knowledge_graph import KnowledgeGraph
from src.rag.hybrid_retriever import HybridRetriever
from src.rag.metric_facts import MetricFactsProvider

AS_OF = "2026-08-31"
QUERY = "LANEIGE Lip Care HHI 현황은?"

# KG에만 있는 날짜 없는 수치 — 증거로 쓰이면 안 된다 (E2)
KG_METADATA_SOS = 0.4242  # 렌더되면 "42.4%"
KG_HHI_EDGE_VALUE = "777.7"
KG_SOS_EDGE_PCT = 42.42

# 운영 DB(data/amore_data.db, 2026-09-17)의 세 테이블 스키마 그대로
OPERATIONAL_SCHEMA = """
CREATE TABLE brand_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    snapshot_date TEXT NOT NULL,
    category_id TEXT NOT NULL,
    brand TEXT NOT NULL,
    sos REAL,
    brand_avg_rank REAL,
    product_count INTEGER,
    cpi REAL,
    avg_rating_gap REAL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(snapshot_date, category_id, brand)
);
CREATE TABLE market_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    snapshot_date TEXT NOT NULL,
    category_id TEXT NOT NULL,
    hhi REAL,
    churn_rate REAL,
    category_avg_price REAL,
    category_avg_rating REAL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(snapshot_date, category_id)
);
CREATE TABLE raw_data (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    snapshot_date TEXT NOT NULL,
    category_id TEXT NOT NULL,
    rank INTEGER NOT NULL,
    asin TEXT NOT NULL,
    product_name TEXT,
    brand TEXT,
    price REAL,
    list_price REAL,
    discount_percent REAL,
    rating REAL,
    reviews_count INTEGER,
    badge TEXT,
    coupon_text TEXT,
    is_subscribe_save INTEGER DEFAULT 0,
    promo_badges TEXT,
    product_url TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    price_currency TEXT DEFAULT 'USD',
    price_original REAL,
    list_price_original REAL,
    exchange_rate REAL,
    UNIQUE(snapshot_date, category_id, rank)
);
"""

# 규칙 추론이 발화하도록 주는 current_metrics (strong_avg_rank·category_entry_opportunity)
CURRENT_METRICS: dict[str, Any] = {
    "brand_metrics": [
        {"brand_name": "laneige", "share_of_shelf": 0.02, "avg_rank": 8.5, "product_count": 2}
    ],
    "market_metrics": [{"category_id": "lip_care", "hhi": 0.0681}],
}

DOC_CHUNKS: list[dict[str, Any]] = [
    {
        "id": "metric_guide_hhi_0",
        "content": "# HHI 해석\nHHI는 시장 집중도 지표다. 0.15 미만이면 분산 시장으로 본다.",
        "metadata": {
            "doc_id": "metric_guide",
            "title": "HHI 해석 가이드",
            "doc_type": "metric_guide",
        },
        "score": 0.91,
    },
    {
        "id": "playbook_sos_1",
        "content": "# SoS 대응\nSoS가 낮은 카테고리는 진입 전략을 검토한다.",
        "metadata": {"doc_id": "playbook", "title": "SoS 대응 플레이북", "doc_type": "playbook"},
        "score": 0.72,
    },
]


class FakeDocRetriever:
    """색인 I/O·임베딩 없이 고정 청크를 돌려주는 문서 검색기."""

    def __init__(self, chunks: list[dict[str, Any]] | None = None) -> None:
        self.chunks = chunks if chunks is not None else DOC_CHUNKS
        self.calls: list[dict[str, Any]] = []

    async def initialize(self) -> None:
        return None

    async def search(
        self, query: str, top_k: int = 5, doc_type_filter: list[str] | None = None, **_: Any
    ) -> list[dict[str, Any]]:
        self.calls.append({"query": query, "top_k": top_k, "doc_type_filter": doc_type_filter})
        return [dict(chunk, metadata=dict(chunk["metadata"])) for chunk in self.chunks[:top_k]]


def make_metrics_db(tmp_path: Path) -> Path:
    path = tmp_path / "amore_data.db"
    conn = sqlite3.connect(path)
    conn.executescript(OPERATIONAL_SCHEMA)
    conn.executemany(
        "INSERT INTO brand_metrics (snapshot_date, category_id, brand, sos, product_count)"
        " VALUES (?,?,?,?,?)",
        [
            (AS_OF, "lip_care", "eos", 9.0, 9),
            (AS_OF, "lip_care", "Burt's Bees", 8.0, 8),
            (AS_OF, "lip_care", "LANEIGE", 2.0, 2),
            ("2026-09-11", "lip_care", "LANEIGE", 3.0, 3),
        ],
    )
    conn.executemany(
        "INSERT INTO market_metrics (snapshot_date, category_id, hhi, category_avg_rating)"
        " VALUES (?,?,?,?)",
        [(AS_OF, "lip_care", 0.0681, 4.57), ("2026-09-11", "lip_care", 0.07, 4.6)],
    )
    conn.executemany(
        "INSERT INTO raw_data (snapshot_date, category_id, rank, asin, product_name, brand,"
        " price, rating, reviews_count) VALUES (?,?,?,?,?,?,?,?,?)",
        [
            (
                AS_OF,
                "lip_care",
                1,
                "B0BURT0001",
                "Burt's Bees Lip Balm",
                "Burt's Bees",
                10.48,
                4.8,
                121467,
            ),
            (
                AS_OF,
                "lip_care",
                7,
                "B0LSM00001",
                "LANEIGE Lip Sleeping Mask: Berry",
                "LANEIGE",
                21.6,
                4.6,
                37356,
            ),
        ],
    )
    conn.commit()
    conn.close()
    return path


def make_kg(tmp_path: Path) -> KnowledgeGraph:
    kg = KnowledgeGraph(persist_path=str(tmp_path / "kg.json"), auto_save=False)
    relations = [
        Relation(
            "laneige",
            RelationType.HAS_PRODUCT,
            "B0LSM00001",
            properties={"product_name": "Lip Sleeping Mask", "category": "lip_care", "rank": 7},
        ),
        Relation("B0LSM00001", RelationType.BELONGS_TO_CATEGORY, "lip_care"),
        Relation("burt's bees", RelationType.HAS_PRODUCT, "B0BURT0001", properties={"rank": 1}),
        Relation("B0BURT0001", RelationType.BELONGS_TO_CATEGORY, "lip_care"),
        Relation(
            "laneige",
            RelationType.COMPETES_WITH,
            "burt's bees",
            properties={"category": "lip_care"},
        ),
        Relation("laneige", RelationType.OWNED_BY, "amorepacific"),
        # 날짜 없는 KG 수치 엣지 (kg_enricher 형식) — 증거에서 제외돼야 한다
        Relation(
            "laneige",
            RelationType.HAS_POSITION,
            "lip_care",
            properties={"original_predicate": "hasSoS", "sos_pct": KG_SOS_EDGE_PCT},
        ),
        Relation(
            "laneige",
            RelationType.HAS_POSITION,
            KG_HHI_EDGE_VALUE,
            properties={"original_predicate": "hasHHI"},
        ),
    ]
    for relation in relations:
        kg.add_relation(relation)
    kg.set_entity_metadata(
        "laneige", {"type": "brand", "sos": KG_METADATA_SOS, "avg_rank": 33.3, "product_count": 12}
    )
    return kg


def make_retriever(
    tmp_path: Path, doc_retriever: FakeDocRetriever | None = None
) -> HybridRetriever:
    return HybridRetriever(
        knowledge_graph=make_kg(tmp_path),
        doc_retriever=doc_retriever or FakeDocRetriever(),
        metric_facts_provider=MetricFactsProvider(make_metrics_db(tmp_path), as_of=AS_OF),
    )


def markdown_headers(text: str) -> list[str]:
    """``## `` 머리글을 등장 순서대로."""
    return [line.strip() for line in text.splitlines() if line.startswith("## ")]
