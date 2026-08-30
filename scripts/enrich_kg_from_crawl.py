"""SQLite 최신 크롤 스냅샷으로 KG 메트릭/경쟁 관계 엔리치먼트

배경 (2026-08-30):
    kg_enricher(hasSoS·rankedIn·hasHHI·competesWith 추출)는 구현돼 있었으나
    파이프라인 어디에서도 호출되지 않아, KG에는 시드 브랜드 온톨로지
    (siblingBrand 등)만 존재하고 크롤 기반 메트릭 엣지가 전혀 없었다.

동작:
    data/amore_data.db의 raw_data에서 카테고리별 최신 snapshot_date의
    Top100을 읽어 KGEnricher.enrich_and_store()로 KG에 저장하고 파일로 flush.

사용법:
    python3 scripts/enrich_kg_from_crawl.py            # 적용
    python3 scripts/enrich_kg_from_crawl.py --dry-run  # 추출 수만 출력
"""

import argparse
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ontology.kg_enricher import KGEnricher  # noqa: E402
from src.ontology.knowledge_graph import KnowledgeGraph  # noqa: E402

MONITORED_CATEGORIES = ["beauty", "skin_care", "lip_care", "lip_makeup", "face_powder"]
DB_PATH = "data/amore_data.db"


def load_latest_snapshot(con: sqlite3.Connection, category: str) -> list[dict]:
    latest = con.execute(
        "SELECT MAX(snapshot_date) FROM raw_data WHERE category_id = ?", (category,)
    ).fetchone()[0]
    if not latest:
        return []
    rows = con.execute(
        """SELECT asin, brand, rank, price, product_name
           FROM raw_data WHERE category_id = ? AND snapshot_date = ?
           ORDER BY rank""",
        (category, latest),
    ).fetchall()
    return [
        {"asin": r[0], "brand": r[1] or "", "rank": r[2], "price": r[3], "title": r[4] or ""}
        for r in rows
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    con = sqlite3.connect(DB_PATH)
    kg = None if args.dry_run else KnowledgeGraph()
    enricher = KGEnricher(knowledge_graph=kg)

    total = {"extracted": 0, "stored": 0}
    for category in MONITORED_CATEGORIES:
        products = load_latest_snapshot(con, category)
        if not products:
            print(f"{category}: no data, skipped")
            continue
        crawl_data = {"category": category, "products": products}
        if args.dry_run:
            triples = enricher.enrich_from_crawl(crawl_data)
            print(f"{category}: {len(products)} products → {len(triples)} triples (dry-run)")
            total["extracted"] += len(triples)
        else:
            result = enricher.enrich_and_store(crawl_data)
            print(f"{category}: {len(products)} products → {result}")
            total["extracted"] += result["extracted"]
            total["stored"] += result["stored"]

    if kg is not None:
        kg.save(force=True)
        stats = kg.get_stats()
        print(f"saved. total={total}, kg_stats={ {k: stats.get(k) for k in list(stats)[:5]} }")
    else:
        print(f"dry-run total={total}")


if __name__ == "__main__":
    main()
