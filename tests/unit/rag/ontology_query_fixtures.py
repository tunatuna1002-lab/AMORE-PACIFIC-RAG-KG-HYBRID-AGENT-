"""온톨로지 질의 경로(트랙 O3) 테스트용 실제 객체 픽스처 [2026-09 사후].

KG는 운영 시드(``config/brands.json`` → ``kg_updater``) 표기를 그대로 흉내 낸다: 대문자 주어
(``LANEIGE``·``COSRX``), 그룹 ``AMOREPACIFIC``, 자매·세그먼트·원산지·인수 연도 트리플, 합쳐진
수치 술어(``hasPosition`` + ``original_predicate``). DB는 운영 스키마(``evidence_pipeline_fixtures``)
그대로이고, 문서 검색기만 가짜다. KG는 임시 ``persist_path``·``auto_save=False``로만 연다.

``QUERIES``는 그룹·세그먼트·원산지·카테고리 포함·부정 판정 질의 다섯 개다(골든 mh001·rl007·rl011·
lg155·rl015와 같은 형태). ``snapshot_of``는 ``HybridContext``를 비교 가능한 JSON으로 바꾼다
(실행 시간처럼 매번 달라지는 값은 뺀다).
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from src.domain.entities.relations import Relation, RelationType
from src.ontology.knowledge_graph import KnowledgeGraph
from src.rag.hybrid_retriever import HybridRetriever
from src.rag.metric_facts import MetricFactsProvider

from .evidence_pipeline_fixtures import AS_OF, OPERATIONAL_SCHEMA, FakeDocRetriever

SNAPSHOT_PATH = Path(__file__).parent / "fixtures" / "o3_flag_off_snapshot.json"

QUERIES: dict[str, str] = {
    "group": "아모레퍼시픽 그룹 소속 브랜드 중 Face Powder Top 100에 제품이 있는 브랜드는?",
    "segment": "아모레퍼시픽 브랜드 포트폴리오에서 COSRX의 세그먼트는 무엇인가요?",
    "origin": "아모레퍼시픽 그룹 브랜드 COSRX의 원산지(출신 국가)는 어디인가요?",
    "inclusion": "LANEIGE가 Skin Care에서 COSRX 수준의 SoS를 달성하려면?",
    "negative": "LANEIGE와 TIRTIR는 같은 그룹에 속한 자매 브랜드인가요?",
}

_AP_SEED: tuple[tuple[str, str, str], ...] = (
    # (브랜드 표기, 세그먼트 표기, 원산지) — 운영 시드 값
    ("LANEIGE", "Premium", "Korea"),
    ("COSRX", "K-Beauty", "Korea"),
    ("Sulwhasoo", "Luxury", "Korea"),
    ("innisfree", "Premium", "Korea"),
    ("ETUDE", "Mass", "Korea"),
)


def make_ontology_kg(tmp_path: Path) -> KnowledgeGraph:
    kg = KnowledgeGraph(persist_path=str(tmp_path / "kg.json"), auto_save=False)
    relations: list[Relation] = []
    names = [name for name, _, _ in _AP_SEED]
    for name, segment, origin in _AP_SEED:
        relations += [
            Relation(
                name, RelationType.OWNED_BY_GROUP, "AMOREPACIFIC", properties={"segment": segment}
            ),
            Relation("AMOREPACIFIC", RelationType.OWNS_BRAND, name),
            Relation(name, RelationType.HAS_SEGMENT, segment),
            Relation(name, RelationType.ORIGINATES_FROM, origin),
        ]
        relations += [
            Relation(
                name, RelationType.SIBLING_BRAND, other, properties={"parent_group": "AMOREPACIFIC"}
            )
            for other in names
            if other != name
        ]
    relations += [
        Relation(
            "COSRX", RelationType.ACQUIRED_IN, "2024", properties={"original_country": "Korea"}
        ),
        # enricher 형식 (소문자 주어, 합쳐진 술어)
        Relation(
            "laneige",
            RelationType.HAS_PRODUCT,
            "B0LSM00001",
            properties={"title": "LANEIGE Lip Sleeping Mask", "category": "lip_care", "rank": 7},
        ),
        Relation(
            "B0LSM00001",
            RelationType.BELONGS_TO_CATEGORY,
            "lip_care",
            properties={"original_predicate": "BELONGS_TO_CATEGORY"},
        ),
        Relation(
            "laneige",
            RelationType.BELONGS_TO_CATEGORY,
            "lip_care",
            properties={"original_predicate": "rankedIn"},
        ),
        Relation(
            "cosrx",
            RelationType.BELONGS_TO_CATEGORY,
            "skin_care",
            properties={"original_predicate": "rankedIn"},
        ),
        Relation(
            "laneige", RelationType.COMPETES_WITH, "tirtir", properties={"category": "face_powder"}
        ),
        Relation(
            "laneige",
            RelationType.HAS_POSITION,
            "lip_care",
            properties={"original_predicate": "hasSoS", "sos_pct": 2.0},
        ),
        Relation(
            "laneige",
            RelationType.HAS_POSITION,
            "premium",
            properties={"original_predicate": "PRICE_POSITION"},
        ),
        Relation(
            "tirtir",
            RelationType.BELONGS_TO_CATEGORY,
            "face_powder",
            properties={"original_predicate": "rankedIn"},
        ),
    ]
    for relation in relations:
        kg.add_relation(relation)
    return kg


def make_ontology_db(tmp_path: Path) -> Path:
    path = tmp_path / "amore_data.db"
    conn = sqlite3.connect(path)
    conn.executescript(OPERATIONAL_SCHEMA)
    conn.executemany(
        "INSERT INTO brand_metrics (snapshot_date, category_id, brand, sos, product_count)"
        " VALUES (?,?,?,?,?)",
        [
            (AS_OF, "lip_care", "eos", 9.0, 9),
            (AS_OF, "lip_care", "LANEIGE", 2.0, 2),
            (AS_OF, "skin_care", "medicube", 13.54, 13),
            (AS_OF, "skin_care", "COSRX", 1.04, 1),
            (AS_OF, "face_powder", "Maybelline", 12.0, 12),
            (AS_OF, "face_powder", "TIRTIR", 3.0, 3),
            (AS_OF, "face_powder", "LANEIGE", 2.0, 2),
            (AS_OF, "face_powder", "innisfree", 1.0, 1),
            (AS_OF, "face_powder", "ETUDE", 1.0, 1),
        ],
    )
    conn.executemany(
        "INSERT INTO market_metrics (snapshot_date, category_id, hhi, category_avg_price)"
        " VALUES (?,?,?,?)",
        [
            (AS_OF, "lip_care", 0.0681, 14.2),
            (AS_OF, "skin_care", 0.0670, 21.0),
            (AS_OF, "face_powder", 0.0527, 17.7),
        ],
    )
    conn.executemany(
        "INSERT INTO raw_data (snapshot_date, category_id, rank, asin, product_name, brand,"
        " price, rating, reviews_count) VALUES (?,?,?,?,?,?,?,?,?)",
        [
            (
                AS_OF,
                "lip_care",
                7,
                "B0LSM00001",
                "LANEIGE Lip Sleeping Mask",
                "LANEIGE",
                21.6,
                4.6,
                37356,
            ),
            (AS_OF, "skin_care", 99, "B0CSX00001", "COSRX Snail Mucin", "COSRX", 17.0, 4.6, 106338),
            (
                AS_OF,
                "face_powder",
                9,
                "B0LNP00001",
                "LANEIGE Neo Blurring Powder",
                "LANEIGE",
                25.0,
                4.4,
                1200,
            ),
            (
                AS_OF,
                "face_powder",
                12,
                "B0TIR00001",
                "TIRTIR Mask Fit Powder",
                "TIRTIR",
                18.0,
                4.3,
                900,
            ),
        ],
    )
    conn.commit()
    conn.close()
    return path


def make_ontology_retriever(tmp_path: Path) -> HybridRetriever:
    return HybridRetriever(
        knowledge_graph=make_ontology_kg(tmp_path),
        doc_retriever=FakeDocRetriever(),
        metric_facts_provider=MetricFactsProvider(make_ontology_db(tmp_path), as_of=AS_OF),
    )


_VOLATILE_METADATA = ("retrieval_time_ms",)


def snapshot_of(ctx: Any) -> dict[str, Any]:
    """``HybridContext`` → 비교용 JSON (실행 시간 제외)."""
    data = json.loads(json.dumps(ctx.to_dict(), ensure_ascii=False, sort_keys=True, default=str))
    for key in _VOLATILE_METADATA:
        data.get("metadata", {}).pop(key, None)
    return data
