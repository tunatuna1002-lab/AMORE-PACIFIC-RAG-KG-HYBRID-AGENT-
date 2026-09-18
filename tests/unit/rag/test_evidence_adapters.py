"""증거 카드 어댑터 — 설계 E1·E2, 결함 F12(KG 수치 엣지)·F13(브랜드 이중 표기)

입력은 실제 객체에서 얻는다: 크롤 DB 사실은 임시 SQLite + 실제 MetricFactsProvider, KG 사실은
임시 경로의 실제 KnowledgeGraph + 실제 HybridRetriever._query_knowledge_graph, 추론은 실제
OntologyReasoner + 등록된 비즈니스 규칙. 가짜는 없다(LLM·네트워크를 부르는 경로가 없다).

고정하는 것
- SoS는 DB에 퍼센트(0~100)로 저장돼 있다 → 카드 정본은 0~1(ratio). 변환은 어댑터에서 한 번.
- 값이 없는(NULL·빈 문자열) 필드는 카드를 만들지 않는다 — 결측을 0으로 채우지 않는다.
- KG의 수치 엣지(hasSoS·hasHHI·hasPosition)는 날짜가 없어 증거에서 제외하고 제외 목록에 남긴다.
- LANEIGE/laneige/Laneige는 같은 subject(canonical id)가 되고 원표기는 display_name에 남는다.
"""

import sqlite3

import pytest

from src.domain.entities.evidence import Evidence, EvidenceKind, EvidenceSet, EvidenceUnit
from src.domain.entities.relations import (
    InferenceResult,
    InsightType,
    Relation,
    RelationType,
    create_ai_summary_relation,
    create_sentiment_relation,
)
from src.ontology.knowledge_graph import KnowledgeGraph
from src.ontology.reasoner import OntologyReasoner
from src.ontology.rules import register_all_rules
from src.rag.evidence_adapters import (
    KG_NUMERIC_PREDICATES,
    MAX_OBSERVATION_CHARS,
    EvidenceAdapter,
    default_brand_normalizer,
    default_category_normalizer,
)
from src.rag.hybrid_retriever import HybridRetriever
from src.rag.metric_facts import MetricFactsProvider

AS_OF = "2026-08-31"

# 운영 DB(data/amore_data.db)의 실제 스키마 — sqlite_master에서 복사 (2026-09-17)
_SCHEMA = """
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


@pytest.fixture
def db_path(tmp_path):
    path = tmp_path / "amore.db"
    conn = sqlite3.connect(path)
    conn.executescript(_SCHEMA)
    conn.executemany(
        "INSERT INTO brand_metrics (snapshot_date, category_id, brand, sos, brand_avg_rank,"
        " product_count, cpi, avg_rating_gap) VALUES (?,?,?,?,?,?,?,?)",
        [
            # 퍼센트 저장 (운영 DB 2026-08-31 skin_care MEDICUBE 13.54 / 13개와 같은 스케일)
            (AS_OF, "lip_care", "Burt's Bees", 13.54, 20.1, 13, None, 0.05),
            (AS_OF, "lip_care", "eos", 9.0, 30.0, 9, None, -0.1),
            (AS_OF, "lip_care", "LANEIGE", 2.0, 8.5, 2, None, 0.076),
            (AS_OF, "skin_care", "MEDICUBE", 13.54, 42.46, 13, None, -0.098),
            ("2026-09-11", "lip_care", "LANEIGE", 3.0, 7.0, 3, None, 0.08),
        ],
    )
    conn.executemany(
        "INSERT INTO market_metrics (snapshot_date, category_id, hhi, churn_rate,"
        " category_avg_price, category_avg_rating) VALUES (?,?,?,?,?,?)",
        [
            (AS_OF, "lip_care", 0.0681, None, None, 4.57),
            ("2026-09-11", "lip_care", 0.0700, None, None, 4.6),
        ],
    )
    conn.executemany(
        "INSERT INTO raw_data (snapshot_date, category_id, rank, asin, product_name, brand,"
        " price, rating, reviews_count) VALUES (?,?,?,?,?,?,?,?,?)",
        [
            (
                AS_OF,
                "lip_care",
                1,
                "B000000001",
                "Burt's Bees Lip Balm",
                "Burt's Bees",
                10.48,
                4.7,
                90000,
            ),
            # 운영 DB 2025-12~2026-01 행처럼 reviews_count가 빈 문자열, rating은 NULL
            (AS_OF, "lip_care", 2, "B000000002", "eos Lip Balm", "eos", 4.99, None, ""),
            (
                AS_OF,
                "lip_care",
                7,
                "B000000007",
                "LANEIGE Lip Sleeping Mask: Korean",
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


@pytest.fixture
def adapter():
    return EvidenceAdapter()


def _find(cards, **fields):
    return [c for c in cards if all(getattr(c, k) == v for k, v in fields.items())]


def _one(cards, **fields):
    found = _find(cards, **fields)
    assert len(found) == 1, f"{fields} → {[(c.subject, c.predicate, c.object) for c in found]}"
    return found[0]


# =============================================================================
# 브랜드·카테고리 정규화 (F13)
# =============================================================================


class TestNormalization:
    @pytest.mark.parametrize("name", ["LANEIGE", "laneige", "Laneige", "  laneige ", "라네즈"])
    def test_laneige_variants_share_canonical_id(self, name):
        assert default_brand_normalizer(name) == "laneige"

    def test_unknown_brand_lowercased_whitespace_collapsed(self):
        assert default_brand_normalizer("Burt's  Bees") == "burt's bees"

    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("lip_care", "lip_care"),
            ("Lip Care", "lip_care"),
            ("립케어", "lip_care"),
            ("Face-Powder", "face_powder"),
            ("BEAUTY", "beauty"),
        ],
    )
    def test_category_ids(self, name, expected):
        assert default_category_normalizer(name) == expected

    def test_injected_normalizers_are_used(self):
        adapter = EvidenceAdapter(
            brand_normalizer=lambda name: "brand::" + name.lower(),
            category_normalizer=lambda name: "cat::" + name,
        )
        cards = adapter.from_metric_facts(
            [
                {
                    "type": "brand_share",
                    "brand": "LANEIGE",
                    "category": "lip_care",
                    "snapshot_date": AS_OF,
                    "present": False,
                }
            ]
        )
        assert cards[0].subject == "brand::laneige"
        assert cards[0].object == "cat::lip_care"


# =============================================================================
# 크롤 DB 수치 → metric 카드
# =============================================================================


class TestFromMetricFacts:
    async def _cards(self, db_path, adapter, brands=("laneige", "cosrx")):
        provider = MetricFactsProvider(db_path=db_path, as_of=AS_OF)
        facts = await provider.collect({"brands": list(brands), "categories": ["lip_care"]})
        assert facts, "픽스처 DB에서 사실을 얻지 못함"
        return adapter.from_metric_facts(facts)

    async def test_all_cards_are_dated_metrics_with_values(self, db_path, adapter):
        cards = await self._cards(db_path, adapter)
        assert cards
        for card in cards:
            assert card.kind == EvidenceKind.METRIC
            assert card.as_of == AS_OF
            assert card.value is not None
            assert card.source.startswith("sqlite:")
            assert card.id.startswith("M-")

    async def test_sos_converted_to_ratio(self, db_path, adapter):
        cards = await self._cards(db_path, adapter)
        laneige = _one(cards, subject="laneige", predicate="sos")
        assert laneige.value == pytest.approx(0.02)
        assert laneige.unit == EvidenceUnit.RATIO
        assert laneige.object == "lip_care"
        assert laneige.source == "sqlite:brand_metrics"
        assert laneige.metadata["display_name"] == "LANEIGE"
        burts = _one(cards, subject="burt's bees", predicate="sos")
        assert burts.value == pytest.approx(0.1354)
        assert all(0 <= c.value <= 1 for c in _find(cards, predicate="sos"))

    async def test_rank_and_count(self, db_path, adapter):
        cards = await self._cards(db_path, adapter)
        assert _one(cards, subject="laneige", predicate="sos_rank").value == 3
        assert _one(cards, subject="laneige", predicate="product_count").value == 2
        assert _one(cards, subject="eos", predicate="sos_rank").value == 2
        assert _one(cards, subject="laneige", predicate="sos_rank").unit == EvidenceUnit.RANK

    async def test_top_brands_and_brand_share_dedupe_to_same_ids(self, db_path, adapter):
        """top_brands와 brand_share가 같은 사실을 두 번 내도 카드는 한 장이다."""
        cards = await self._cards(db_path, adapter)
        assert len(cards) == len({c.id for c in cards})
        assert len(_find(cards, subject="laneige", predicate="sos")) == 1

    async def test_market_fields_without_null(self, db_path, adapter):
        cards = await self._cards(db_path, adapter)
        hhi = _one(cards, subject="lip_care", predicate="hhi")
        assert hhi.value == pytest.approx(0.0681)
        assert hhi.unit == EvidenceUnit.INDEX_0_1
        assert hhi.source == "sqlite:market_metrics"
        assert _one(cards, subject="lip_care", predicate="avg_rating").value == pytest.approx(4.57)
        # churn_rate·category_avg_price는 NULL → 카드 없음
        assert not _find(cards, subject="lip_care", predicate="churn_rate")
        assert not _find(cards, subject="lip_care", predicate="avg_price")

    async def test_absent_brand_card(self, db_path, adapter):
        cards = await self._cards(db_path, adapter)
        absent = _one(cards, subject="cosrx", predicate="present_in_top100")
        assert absent.value is False
        assert absent.unit == EvidenceUnit.BOOLEAN
        assert absent.object == "lip_care"
        assert not _find(cards, subject="cosrx", predicate="sos")  # 0%로 채우지 않는다

    async def test_product_cards_skip_missing_values(self, db_path, adapter):
        cards = await self._cards(db_path, adapter)
        mask = "LANEIGE Lip Sleeping Mask"
        assert _one(cards, subject=mask, predicate="bsr_rank").value == 7
        assert _one(cards, subject=mask, predicate="price").value == pytest.approx(21.6)
        assert _one(cards, subject=mask, predicate="price").unit == EvidenceUnit.USD
        assert _one(cards, subject=mask, predicate="reviews_count").value == 37356
        assert _one(cards, subject=mask, predicate="bsr_rank").metadata["brand"] == "laneige"
        # eos: rating NULL, reviews_count '' → 해당 카드 없음, 순위·가격은 있음
        assert _find(cards, subject="eos Lip Balm", predicate="bsr_rank")
        assert not _find(cards, subject="eos Lip Balm", predicate="rating")
        assert not _find(cards, subject="eos Lip Balm", predicate="reviews_count")

    async def test_as_of_respected(self, db_path, adapter):
        cards = await self._cards(db_path, adapter)
        assert not _find(cards, subject="laneige", predicate="sos", value=0.03)

    def test_out_of_range_sos_is_dropped(self, adapter):
        cards = adapter.from_metric_facts(
            [
                {
                    "type": "brand_share",
                    "brand": "X",
                    "category": "lip_care",
                    "snapshot_date": AS_OF,
                    "present": True,
                    "sos": 150.0,
                    "product_count": 1,
                    "brand_rank": 1,
                }
            ]
        )
        assert not _find(cards, predicate="sos")
        assert _one(cards, predicate="product_count").value == 1

    def test_same_fact_same_id_across_calls(self, adapter):
        fact = {
            "type": "category_market",
            "category": "lip_care",
            "snapshot_date": AS_OF,
            "hhi": 0.0681,
        }
        assert (
            adapter.from_metric_facts([fact])[0].id
            == EvidenceAdapter().from_metric_facts([dict(fact)])[0].id
        )


# =============================================================================
# KG 사실 → relation 카드 (수치 엣지 제외)
# =============================================================================


@pytest.fixture
def kg(tmp_path):
    graph = KnowledgeGraph(persist_path=str(tmp_path / "kg.json"), auto_save=False, auto_load=False)
    graph.load_category_hierarchy()  # config/category_hierarchy.json (저장소 실제 설정)
    graph.add_relations(
        [
            # 시드 온톨로지 표기 (대문자, config/brands.json)
            Relation("LANEIGE", RelationType.OWNED_BY_GROUP, "AMOREPACIFIC", source="config"),
            # kg_enricher 표기 (소문자, original_predicate 보존 — kg_enricher.py:393-401)
            Relation(
                "laneige",
                RelationType.HAS_POSITION,
                "lip_care",
                properties={
                    "sos_pct": 2.0,
                    "product_count": 2,
                    "total": 100,
                    "original_predicate": "hasSoS",
                },
                source="kg_enricher",
            ),
            Relation(
                "laneige",
                RelationType.BELONGS_TO_CATEGORY,
                "lip_care",
                properties={"product_count": 2, "original_predicate": "rankedIn"},
                source="kg_enricher",
            ),
            Relation(
                "laneige",
                RelationType.HAS_POSITION,
                "premium",
                properties={
                    "category": "lip_care",
                    "avg_price": 21.6,
                    "market_avg": 10.4,
                    "original_predicate": "PRICE_POSITION",
                },
                source="kg_enricher",
            ),
            Relation(
                "laneige",
                RelationType.COMPETES_WITH,
                "cosrx",
                properties={"category": "lip_care", "original_predicate": "COMPETES_WITH"},
                source="kg_enricher",
            ),
            Relation(
                "laneige",
                RelationType.HAS_PRODUCT,
                "B000000007",
                properties={
                    "category": "lip_care",
                    "rank": 7,
                    "title": "LANEIGE Lip Sleeping Mask: Korean",
                    "original_predicate": "HAS_PRODUCT",
                },
                source="kg_enricher",
            ),
            Relation(
                "B000000007",
                RelationType.BELONGS_TO_CATEGORY,
                "lip_care",
                properties={"rank": 7, "original_predicate": "BELONGS_TO_CATEGORY"},
                source="kg_enricher",
            ),
            Relation(
                "lip_care",
                RelationType.HAS_POSITION,
                "1012",
                properties={"hhi": 1012, "brand_count": 29, "original_predicate": "hasHHI"},
                source="kg_enricher",
            ),
            create_sentiment_relation("B000000007", "Moisturizing", "Hydration"),
            create_ai_summary_relation("B000000007", "Customers like the hydration."),
        ]
    )
    return graph


@pytest.fixture
def kg_facts(kg, db_path):
    retriever = HybridRetriever(
        knowledge_graph=kg,
        reasoner=OntologyReasoner(kg),
        doc_retriever=object(),  # _query_knowledge_graph는 문서 검색기를 쓰지 않는다
        auto_init_rules=False,
        metric_facts_provider=MetricFactsProvider(db_path=db_path, as_of=AS_OF),
    )
    facts = retriever._query_knowledge_graph(
        {
            "brands": ["laneige"],
            "categories": ["lip_care"],
            "products": ["B000000007"],
            "sentiments": ["Moisturizing"],
            "sentiment_clusters": [],
        }
    )
    assert facts, "실제 KG 조회 결과가 비었다"
    return facts


class TestFromKgFacts:
    def test_fact_types_come_from_real_query(self, kg_facts):
        types = {f["type"] for f in kg_facts}
        assert {
            "brand_products",
            "competitors",
            "metric_edges",
            "category_brands",
            "category_hierarchy",
            "product_sentiment",
            "brand_sentiment",
        } <= types

    def test_numeric_edges_excluded(self, adapter, kg_facts):
        result = adapter.from_kg_facts(kg_facts)
        assert result.cards
        assert all(c.kind == EvidenceKind.RELATION for c in result.cards)
        assert not [c for c in result.cards if c.predicate in KG_NUMERIC_PREDICATES]
        excluded = {(e["predicate"], e["object"]) for e in result.excluded}
        assert ("hasSoS", "lip_care") in excluded
        assert ("hasPosition", "premium") in excluded
        assert all(e["reason"] == "kg_numeric_undated" for e in result.excluded)
        assert all(c.value is None and c.as_of is None for c in result.cards)

    def test_structural_relations_kept_with_canonical_subjects(self, adapter, kg_facts):
        cards = adapter.from_kg_facts(kg_facts).cards
        owned = _one(cards, subject="laneige", predicate="ownedBy")
        assert owned.object == "amorepacific"
        assert owned.metadata["display_name"] == "LANEIGE"
        _one(cards, subject="laneige", predicate="competesWith", object="cosrx")
        _one(cards, subject="laneige", predicate="rankedIn", object="lip_care")
        _one(cards, subject="laneige", predicate="hasProduct", object="B000000007")
        _one(cards, subject="B000000007", predicate="belongsToCategory", object="lip_care")
        _one(cards, subject="lip_care", predicate="parentCategory", object="skin_care")
        assert not _find(cards, subject="LANEIGE")
        assert len(cards) == len({c.id for c in cards})

    def test_sentiment_relations(self, adapter, kg_facts):
        cards = adapter.from_kg_facts(kg_facts).cards
        tag = _one(cards, subject="B000000007", predicate="hasSentiment", object="Moisturizing")
        assert tag.metadata["cluster"] == "Hydration"
        summary = _one(cards, subject="B000000007", predicate="hasAISummary")
        assert summary.detail == "Customers like the hydration."
        _one(cards, subject="laneige", predicate="brandSentiment", object="Moisturizing")

    def test_competitor_categories_merged(self, adapter):
        facts = [
            {
                "type": "competitors",
                "entity": "laneige",
                "data": [
                    {"brand": "cosrx", "type": "competesWith", "category": "lip_care"},
                    {"brand": "COSRX", "type": "competesWith", "category": "skin_care"},
                ],
            }
        ]
        cards = adapter.from_kg_facts(facts).cards
        card = _one(cards, subject="laneige", predicate="competesWith", object="cosrx")
        assert card.metadata["categories"] == ["lip_care", "skin_care"]

    def test_brand_info_metadata_is_not_evidence(self, adapter):
        """entity_metadata의 sos·avg_rank는 대시보드 JSON에서 온 날짜 없는 수치다."""
        facts = [
            {
                "type": "brand_info",
                "entity": "LANEIGE",
                "data": {"type": "brand", "sos": 0.08, "avg_rank": 12.0, "is_target": True},
            }
        ]
        result = adapter.from_kg_facts(facts)
        assert result.cards == []
        assert {e["predicate"] for e in result.excluded} == {"sos", "avg_rank", "type", "is_target"}

    def test_placeholder_brand_excluded(self, adapter):
        """kg_enricher가 브랜드 미상 제품을 묶은 'unknown'은 브랜드가 아니다."""
        facts = [
            {
                "type": "competitor_network",
                "entity": "laneige",
                "data": {
                    "outgoing": [(RelationType.COMPETES_WITH, "cosrx")],
                    "incoming": [(RelationType.COMPETES_WITH, "unknown")],
                },
            }
        ]
        result = adapter.from_kg_facts(facts)
        assert [(c.subject, c.object) for c in result.cards] == [("laneige", "cosrx")]
        assert result.excluded == [
            {
                "fact_type": "competitor_network",
                "subject": "unknown",
                "predicate": "competesWith",
                "object": "laneige",
                "reason": "placeholder_entity",
            }
        ]

    def test_unknown_fact_type_counted(self, adapter):
        result = adapter.from_kg_facts([{"type": "mystery", "entity": "x", "data": {}}])
        assert result.cards == []
        assert result.skipped == {"mystery": 1}


# =============================================================================
# 문서 청크 → document 카드
# =============================================================================


class TestFromRagChunks:
    # retriever.py:1241-1257 (BM25 경로)이 내는 형식
    CHUNK = {
        "id": "metric_guide_2",
        "content": "## SoS (Share of Shelf)\nTop 100 중 브랜드 제품 비율.\n계산식: ...",
        "metadata": {
            "doc_id": "metric_guide",
            "doc_type": "metric_guide",
            "title": "SoS (Share of Shelf)",
            "chunk_id": "metric_guide_2",
            "source_filename": "metric_guide.md",
        },
        "score": 0.82,
    }

    def test_document_card(self, adapter):
        (card,) = adapter.from_rag_chunks([self.CHUNK])
        assert card.kind == EvidenceKind.DOCUMENT
        assert card.id.startswith("D-")
        assert card.subject == "metric_guide"
        assert card.predicate == "states"
        assert card.text == "SoS (Share of Shelf)"
        assert card.detail == self.CHUNK["content"]
        assert card.metadata["chunk_id"] == "metric_guide_2"
        assert card.metadata["score"] == 0.82
        assert card.metadata["doc_type"] == "metric_guide"
        assert card.source == "rag:metric_guide"

    def test_id_depends_on_chunk_id_only(self, adapter):
        rescored = {**self.CHUNK, "score": 0.1, "content": self.CHUNK["content"] + " (재검색)"}
        other = {
            **self.CHUNK,
            "id": "metric_guide_3",
            "metadata": {**self.CHUNK["metadata"], "chunk_id": "metric_guide_3"},
        }
        (original,) = adapter.from_rag_chunks([self.CHUNK])
        (same_chunk,) = adapter.from_rag_chunks([rescored])
        (different_chunk,) = adapter.from_rag_chunks([other])
        assert original.id == same_chunk.id
        assert original.id != different_chunk.id
        # 한 번에 넣으면 같은 청크는 한 장으로 합쳐진다 (먼저 온 카드)
        merged = adapter.from_rag_chunks([self.CHUNK, rescored, other])
        assert [c.id for c in merged] == [original.id, different_chunk.id]

    def test_title_falls_back_to_first_line(self, adapter):
        chunk = {
            "id": "x_0",
            "content": "\n\n첫 줄 요약입니다\n둘째 줄",
            "metadata": {},
            "score": 1,
        }
        (card,) = adapter.from_rag_chunks([chunk])
        assert card.text == "첫 줄 요약입니다"
        assert card.subject == "x_0"


# =============================================================================
# 규칙 추론 → inference 카드
# =============================================================================


class TestFromInferences:
    @pytest.fixture
    def results(self, kg):
        reasoner = OntologyReasoner(kg)
        register_all_rules(reasoner)
        results = reasoner.infer(
            {"brand": "laneige", "category": "lip_care", "sos": 0.2, "hhi": 0.1}
        )
        assert any(r.rule_name == "market_dominance_fragmented" for r in results)
        return results

    def test_inference_card(self, adapter, results):
        cards = adapter.from_inferences(results, derived_from=["M-aaaaaa", "M-bbbbbb"], as_of=AS_OF)
        card = _one(cards, source="rule:market_dominance_fragmented")
        assert card.kind == EvidenceKind.INFERENCE
        assert card.id.startswith("I-")
        assert card.subject == "laneige"
        assert card.predicate == InsightType.MARKET_DOMINANCE.value
        assert card.object == "lip_care"
        assert card.value == "dominant_in_fragmented"  # 결론 position (metadata market_type 아님)
        assert card.confidence == pytest.approx(0.9)
        assert card.as_of == AS_OF
        assert card.derived_from == ("M-aaaaaa", "M-bbbbbb")
        assert "20.0%" in card.text
        assert card.detail  # 권장 액션
        assert "sos_above_0.15" in card.metadata["satisfied_conditions"]

    def test_resolver_takes_precedence(self, adapter, results):
        cards = adapter.from_inferences(
            results,
            derived_from=["M-aaaaaa"],
            derived_from_resolver=lambda r: [f"M-{len(r.rule_name):06x}"],
        )
        for card in cards:
            assert card.derived_from != ("M-aaaaaa",)

    def test_confidence_clamped(self, adapter):
        result = InferenceResult(
            rule_name="r",
            insight_type=InsightType.RISK_ALERT,
            insight="위험",
            confidence=1.2,
            evidence={"context_snapshot": {"brand": "LANEIGE"}},
        )
        (card,) = adapter.from_inferences([result])
        assert card.confidence == 1.0
        assert card.subject == "laneige"


# =============================================================================
# 도구 관찰 → observation 카드
# =============================================================================


class TestFromToolObservation:
    def test_observation_card(self, adapter):
        card = adapter.from_tool_observation(
            "get_metrics", {"brand": "laneige", "as_of": AS_OF}, {"sos": 2.0, "rows": 1}
        )
        assert card.kind == EvidenceKind.OBSERVATION
        assert card.id.startswith("O-")
        assert card.source == "tool:get_metrics"
        assert card.subject == "get_metrics"
        assert card.detail == '{"rows": 1, "sos": 2.0}'
        assert card.text.startswith("get_metrics(")
        again = adapter.from_tool_observation(
            "get_metrics", {"as_of": AS_OF, "brand": "laneige"}, {"rows": 1, "sos": 2.0}
        )
        assert again.id == card.id
        different = adapter.from_tool_observation("get_metrics", {"brand": "laneige"}, "other")
        assert different.id != card.id

    def test_long_observation_truncated(self, adapter):
        card = adapter.from_tool_observation(
            "search_docs", {"q": "x"}, "가" * (MAX_OBSERVATION_CHARS + 10)
        )
        assert len(card.detail) == MAX_OBSERVATION_CHARS
        assert card.metadata["truncated"] is True
        assert card.metadata["observation_chars"] == MAX_OBSERVATION_CHARS + 10


def test_adapters_compose_into_one_set(adapter):
    """어댑터 결과는 하나의 EvidenceSet으로 합쳐진다 (2-B 조립기의 사용 형태)."""
    metric = adapter.from_metric_facts(
        [{"type": "category_market", "category": "lip_care", "snapshot_date": AS_OF, "hhi": 0.07}]
    )
    docs = adapter.from_rag_chunks([TestFromRagChunks.CHUNK])
    cards = EvidenceSet([*metric, *docs, *metric])
    assert len(cards) == 2
    assert isinstance(cards.to_list()[0], Evidence)
