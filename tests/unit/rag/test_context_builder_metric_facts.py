"""컨텍스트 빌더가 크롤 DB 수치를 날짜·단위와 함께 프롬프트에 싣는다 (사이클 10)

이전에는 KG의 metric_edges 사실이 조회만 되고 렌더링되지 않아, 평가의 L3 엣지 recall은
올라가는데 답변은 그 수치를 몰랐다.
"""

from types import SimpleNamespace

from src.rag.context_builder import ContextBuilder

FACTS = [
    {
        "type": "category_market",
        "category": "lip_care",
        "snapshot_date": "2026-08-31",
        "hhi": 0.0681,
    },
    {
        "type": "category_top_brands",
        "category": "lip_care",
        "snapshot_date": "2026-08-31",
        "brands": [{"brand": "eos", "sos": 9.0, "product_count": 9}],
    },
    {
        "type": "brand_share",
        "brand": "LANEIGE",
        "category": "lip_care",
        "snapshot_date": "2026-08-31",
        "present": True,
        "sos": 2.0,
        "product_count": 2,
        "brand_rank": 9,
    },
    {
        "type": "brand_share",
        "brand": "tirtir",
        "category": "lip_care",
        "snapshot_date": "2026-08-31",
        "present": False,
    },
    {
        "type": "brand_products",
        "brand": "LANEIGE",
        "category": "lip_care",
        "snapshot_date": "2026-08-31",
        "products": [
            {"rank": 7, "name": "LANEIGE Lip Sleeping Mask", "price": 21.6, "reviews_count": 37356}
        ],
    },
]


def _context(metric_facts):
    return SimpleNamespace(
        entities={}, inferences=[], ontology_facts=[], rag_chunks=[], metric_facts=metric_facts
    )


def test_metric_facts_are_rendered_with_units_and_dates():
    text = ContextBuilder(max_tokens=4000).build(_context(FACTS), None, "Lip Care HHI는?")

    for expected in (
        "HHI 0.0681",
        "eos 9.0%",
        "LANEIGE lip_care SoS (2026-08-31): 2.0%",
        "브랜드 중 9위",
        "tirtir lip_care (2026-08-31): Top 100 내 제품 없음",
        "7위 LANEIGE Lip Sleeping Mask $21.60",
        "리뷰 37,356건",
    ):
        assert expected in text, expected


def test_missing_values_are_not_rendered_as_zero():
    facts = [{"type": "category_market", "category": "skin_care", "snapshot_date": "2026-08-31"}]

    text = ContextBuilder(max_tokens=4000).build(_context(facts), None, "q")

    assert "HHI" not in text


def test_contexts_without_metric_facts_are_unchanged():
    builder = ContextBuilder(max_tokens=4000)

    assert builder.build(_context([]), None, "q") == builder.build(
        SimpleNamespace(entities={}, inferences=[], ontology_facts=[], rag_chunks=[]), None, "q"
    )
