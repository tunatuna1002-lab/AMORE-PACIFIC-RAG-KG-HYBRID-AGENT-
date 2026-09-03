"""Shared inline fixtures for characterization tests (no real data files)."""

from __future__ import annotations

OTHER_BRANDS = [
    "Burt's Bees",
    "eos",
    "Aquaphor",
    "Vaseline",
    "Carmex",
    "Summer Fridays",
    "Glossier",
    "Blistex",
]


def build_lip_care_snapshot() -> list[dict]:
    """100-record lip_care snapshot.

    - ranks 1..100
    - LANEIGE at ranks 1, 5, 20 (3 products)
    - COSRX at ranks 2,3,4,6,7,8,9,10,11,12 (10 products)
    - remaining 87 records round-robin across 8 other brands
      (7 brands x 11 products, Blistex x 10)
    - price 24.0 + (rank % 5), rating 4.0 + (rank % 10) / 10
    """
    brand_by_rank = {1: "LANEIGE", 5: "LANEIGE", 20: "LANEIGE"}
    for r in (2, 3, 4, 6, 7, 8, 9, 10, 11, 12):
        brand_by_rank[r] = "COSRX"

    records: list[dict] = []
    other_idx = 0
    for rank in range(1, 101):
        brand = brand_by_rank.get(rank)
        if brand is None:
            brand = OTHER_BRANDS[other_idx % len(OTHER_BRANDS)]
            other_idx += 1
        records.append(
            {
                "asin": f"B{rank:09d}",
                "rank": rank,
                "brand": brand,
                "title": f"{brand} product {rank}",
                "price": 24.0 + (rank % 5),
                "rating": 4.0 + (rank % 10) / 10,
            }
        )
    return records
