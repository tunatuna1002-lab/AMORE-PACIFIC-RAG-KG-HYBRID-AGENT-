"""컨텍스트 빌더가 크롤 DB 수치를 날짜·단위와 함께 프롬프트에 싣는다 (사이클 10 → 트랙 2-B)

이전에는 KG의 metric_edges 사실이 조회만 되고 렌더링되지 않아, 평가의 L3 엣지 recall은
올라가는데 답변은 그 수치를 몰랐다. 트랙 2-B부터 수치는 [DB 수치] 카드 한 줄씩
``[M-id] 주어 범위 지표 값 (스냅샷 날짜, sqlite:<table>)``으로 싣는다.
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


def _context(metric_facts, brands=("laneige", "tirtir")):
    return SimpleNamespace(
        entities={"brands": list(brands), "categories": ["lip_care"]},
        inferences=[],
        ontology_facts=[],
        rag_chunks=[],
        metric_facts=metric_facts,
    )


def test_metric_facts_are_rendered_with_units_and_dates():
    text = ContextBuilder(max_tokens=4000).build(_context(FACTS), None, "Lip Care HHI는?")

    for expected in (
        "lip_care HHI 0.0681 (2026-08-31, sqlite:market_metrics)",
        "eos lip_care SoS 9% (2026-08-31, sqlite:brand_metrics)",
        "LANEIGE lip_care SoS 2% (2026-08-31, sqlite:brand_metrics)",
        "LANEIGE lip_care SoS 브랜드 순위 9위 (2026-08-31, sqlite:brand_metrics)",
        "tirtir lip_care Top100 진입 없음 (2026-08-31, sqlite:brand_metrics)",
        "LANEIGE Lip Sleeping Mask lip_care BSR 순위 7위 (2026-08-31, sqlite:raw_data)",
        "LANEIGE Lip Sleeping Mask lip_care 가격 $21.60 (2026-08-31, sqlite:raw_data)",
        "LANEIGE Lip Sleeping Mask lip_care 리뷰 수 37,356 (2026-08-31, sqlite:raw_data)",
    ):
        assert expected in text, expected


def test_missing_values_are_not_rendered_as_zero():
    facts = [{"type": "category_market", "category": "skin_care", "snapshot_date": "2026-08-31"}]

    text = ContextBuilder(max_tokens=4000).build(_context(facts), None, "q")

    assert "HHI" not in text
    assert "[DB 수치]" not in text


def test_contexts_without_metric_facts_are_unchanged():
    builder = ContextBuilder(max_tokens=4000)

    assert builder.build(_context([]), None, "q") == builder.build(
        SimpleNamespace(entities={}, inferences=[], ontology_facts=[], rag_chunks=[]), None, "q"
    )
