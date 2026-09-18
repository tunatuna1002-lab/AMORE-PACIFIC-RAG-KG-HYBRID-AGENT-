"""골든 보충 문항 생성기의 공용 사실 조회 (읽기 전용, API $0).

사용처: scripts/generate_rule_questions.py, generate_multihop_questions.py,
generate_relation_questions.py

순환 검증을 피하는 방법
----------------------
정답을 챗봇이 읽는 경로(`src/rag/metric_facts.py` → `brand_metrics`·`market_metrics`)에서
가져오면 "시스템이 읽는 값 = 정답"이 된다. 그래서 정답은 **원자료 `raw_data`에서 정본 공식을
SQL로 다시 계산**하고, 지표 테이블과 `MetricFactsProvider` 결과는 교차 확인에만 쓴다.

정본 공식 (코드 인용, 2026-09-17 기준)
------------------------------------
- 표본: `raw_data`의 (snapshot_date, category_id) 전체 행.
- SoS (%, 0~100): `src/tools/calculators/metric_snapshot.py:build_metric_rows`
  → `calculate_sos_pct(len(brand_rows), total)`
  분모 total = 해당 날짜·카테고리의 **전체** 행 수(Unknown 브랜드 포함), total < 50이면 None,
  값 = round(count / total * 100, 2). 브랜드 키는 `TRIM(brand)` 정확 일치(대소문자 구분,
  `count_brands`가 `strip()`한 문자열로 센다).
- HHI (0~1): `calculate_hhi_from_counts(count_brands(rows))` — 소문자 기준 '', unknown, n/a,
  none 브랜드를 분자·분모에서 모두 제외, Σ(n_i / N)², round 4.
- 카테고리 평균가: 0.5 ≤ price ≤ 500 인 price의 평균 (저장 시 round 2).
- 브랜드 CPI (100 기준): 브랜드 평균가(같은 필터) / 카테고리 평균가(반올림 전) × 100, round 1.
- 평점 갭: 브랜드 평균 평점 − 카테고리 평균 평점(NULL 제외, 반올림 전), round 3.
- 브랜드 평균 순위: round(mean(rank), 2). 최고 순위: MIN(rank).

반올림은 코드와 같게 파이썬 `round`로 한다(SQL은 합계·개수만 돌려준다).

브랜드 귀속 검증
----------------
`raw_data.brand`에는 부분 문자열 오귀속이 있다(2026-08-31: lip_care "Hera" 8행이 전부
Vaseline·Jack Black 제품 — "Therapy" 안의 "hera", face_powder "CHI" = KimChiChic 등).
SoS·HHI 정의는 이 필드를 그대로 쓰므로 정답 계산은 정의를 따르되, **브랜드를 주어로 삼는
문항**은 그 브랜드의 모든 행에서 제품명에 브랜드명이 단어로 들어 있을 때만 만든다
(`attribution_failures`). 카테고리 수준 HHI 문항에는 오귀속 행 수를 함께 기록한다.

KG
--
`data/knowledge_graph.json`에서 **큐레이션 출처**(`config/brands.json`) 트리플만
정답 근거로 쓴다. `system` 출처 competesWith 14개는 `competitor_sos` 속성을 단 데이터 파생
엣지라 제외한다. `kg_enricher`가 크롤에서 만든 트리플(hasPosition=옛 hasSoS 등)은 날짜
버전이 없어 as_of 정답 근거가 될 수 없다(kickoff 지시서 F12).
"""

from __future__ import annotations

import asyncio
import json
import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = REPO_ROOT / "data" / "amore_data.db"
DEFAULT_KG = REPO_ROOT / "data" / "knowledge_graph.json"
TYPED_DIR = REPO_ROOT / "eval" / "data" / "golden" / "typed"

AS_OF = "2026-08-31"
CATEGORIES = ("beauty", "skin_care", "lip_care", "lip_makeup", "face_powder")
CATEGORY_LABEL = {
    "beauty": "Beauty & Personal Care",
    "skin_care": "Skin Care",
    "lip_care": "Lip Care",
    "lip_makeup": "Lip Makeup",
    "face_powder": "Face Powder",
}
SOS_MIN_SAMPLE = 50  # metric_calculator.SOS_MIN_SAMPLE
UNKNOWN_BRANDS = ("", "unknown", "n/a", "none")  # metric_calculator.UNKNOWN_BRAND_LABELS
PRICE_MIN, PRICE_MAX = 0.5, 500  # metric_snapshot.build_metric_rows 가격 필터
NUMERIC_TOLERANCE = 0.10  # eval/metrics/l5_answer.py NUMERIC_TOLERANCE와 같은 허용오차
# `system` 출처 competesWith는 competitor_sos 속성을 단 크롤 파생 엣지라 제외한다
CURATED_KG_SOURCES = ("config/brands.json",)

# ---------------------------------------------------------------------------
# SQL (정답 근거로 문항 metadata에 그대로 저장한다)
# ---------------------------------------------------------------------------

SQL_TOTAL = "SELECT COUNT(*) FROM raw_data WHERE snapshot_date = :d AND category_id = :c"

SQL_BRAND_COUNTS = """
SELECT TRIM(brand) AS brand, COUNT(*) AS n
  FROM raw_data
 WHERE snapshot_date = :d AND category_id = :c
   AND LOWER(TRIM(COALESCE(brand, ''))) NOT IN ('', 'unknown', 'n/a', 'none')
 GROUP BY TRIM(brand)
""".strip()

SQL_CATEGORY_PRICE_RATING = """
SELECT SUM(CASE WHEN price BETWEEN 0.5 AND 500 THEN price END) AS price_sum,
       COUNT(CASE WHEN price BETWEEN 0.5 AND 500 THEN 1 END)  AS price_n,
       SUM(rating) AS rating_sum, COUNT(rating) AS rating_n
  FROM raw_data
 WHERE snapshot_date = :d AND category_id = :c
""".strip()

SQL_BRAND_DETAIL = """
SELECT COUNT(*) AS n, SUM(rank) AS rank_sum, MIN(rank) AS best_rank,
       SUM(CASE WHEN price BETWEEN 0.5 AND 500 THEN price END) AS price_sum,
       COUNT(CASE WHEN price BETWEEN 0.5 AND 500 THEN 1 END)  AS price_n,
       SUM(rating) AS rating_sum, COUNT(rating) AS rating_n
  FROM raw_data
 WHERE snapshot_date = :d AND category_id = :c AND TRIM(brand) = :b
""".strip()

SQL_PRODUCTS = """
SELECT rank, asin, TRIM(brand) AS brand, product_name, price, rating, reviews_count
  FROM raw_data
 WHERE snapshot_date = :d AND category_id = :c
 ORDER BY rank
""".strip()

SQL_METRIC_TABLE_BRAND = """
SELECT brand, sos, brand_avg_rank, product_count, cpi, avg_rating_gap
  FROM brand_metrics WHERE snapshot_date = :d AND category_id = :c
""".strip()

SQL_METRIC_TABLE_MARKET = """
SELECT hhi, category_avg_price, category_avg_rating
  FROM market_metrics WHERE snapshot_date = :d AND category_id = :c
""".strip()


# ---------------------------------------------------------------------------
# 독립 계산 (raw_data)
# ---------------------------------------------------------------------------


@dataclass
class BrandFacts:
    brand: str
    count: int
    sos: float | None  # 0~100
    avg_rank: float
    best_rank: int
    cpi: float | None  # 100 기준
    rating_gap: float | None


@dataclass
class CategoryFacts:
    date: str
    category: str
    total: int
    known_total: int
    hhi: float | None  # 0~1
    avg_price: float | None  # 반올림 전
    avg_rating: float | None  # 반올림 전
    brands: dict[str, BrandFacts] = field(default_factory=dict)
    products: list[dict[str, Any]] = field(default_factory=list)

    def attribution_failures(self) -> dict[str, list[str]]:
        """제품명에 브랜드명이 단어로 들어 있지 않은 행 {브랜드: [제품명 앞 60자]}."""
        failures: dict[str, list[str]] = {}
        for p in self.products:
            brand = (p["brand"] or "").strip()
            if brand.lower() in UNKNOWN_BRANDS:
                continue
            if not brand_in_name(brand, p["product_name"]):
                failures.setdefault(brand, []).append((p["product_name"] or "")[:60])
        return failures

    def ranked_brands(self) -> list[BrandFacts]:
        """SoS 내림차순, 동률은 브랜드명 오름차순 (metric_facts와 같은 정렬 기준)."""
        return sorted(self.brands.values(), key=lambda b: (-(b.sos or 0), b.brand))


def brand_in_name(brand: str, product_name: str | None) -> bool:
    """브랜드명이 제품명에 앞뒤가 영숫자가 아닌 단어로 들어 있는가 (대소문자 무시)."""
    pattern = r"(?<![a-z0-9])" + re.escape(brand.lower()) + r"(?![a-z0-9])"
    return re.search(pattern, (product_name or "").lower()) is not None


def connect_ro(db_path: Path) -> sqlite3.Connection:
    """읽기 전용 연결. 파일이 없으면 만들지 않고 실패한다."""
    if not Path(db_path).exists():
        raise FileNotFoundError(f"DB 없음: {db_path}")
    conn = sqlite3.connect(f"file:{Path(db_path).resolve()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def category_facts(conn: sqlite3.Connection, category: str, date: str = AS_OF) -> CategoryFacts:
    params = {"d": date, "c": category}
    total = conn.execute(SQL_TOTAL, params).fetchone()[0]
    counts = {r["brand"]: r["n"] for r in conn.execute(SQL_BRAND_COUNTS, params)}
    known = sum(counts.values())
    hhi = round(sum((n / known) ** 2 for n in counts.values()), 4) if known else None

    cat = conn.execute(SQL_CATEGORY_PRICE_RATING, params).fetchone()
    avg_price = cat["price_sum"] / cat["price_n"] if cat["price_n"] else None
    avg_rating = cat["rating_sum"] / cat["rating_n"] if cat["rating_n"] else None

    facts = CategoryFacts(
        date=date,
        category=category,
        total=total,
        known_total=known,
        hhi=hhi,
        avg_price=avg_price,
        avg_rating=avg_rating,
    )
    for brand in sorted(counts):
        row = conn.execute(SQL_BRAND_DETAIL, {**params, "b": brand}).fetchone()
        cpi = None
        if row["price_n"] and avg_price:
            cpi = round(row["price_sum"] / row["price_n"] / avg_price * 100, 1)
        gap = None
        if row["rating_n"] and avg_rating is not None:
            gap = round(row["rating_sum"] / row["rating_n"] - avg_rating, 3)
        facts.brands[brand] = BrandFacts(
            brand=brand,
            count=row["n"],
            sos=round(row["n"] / total * 100, 2) if total >= SOS_MIN_SAMPLE else None,
            avg_rank=round(row["rank_sum"] / row["n"], 2),
            best_rank=row["best_rank"],
            cpi=cpi,
            rating_gap=gap,
        )
    facts.products = [dict(r) for r in conn.execute(SQL_PRODUCTS, params)]
    return facts


def find_brand(facts: CategoryFacts, name: str) -> BrandFacts | None:
    """대소문자 무시 이름 일치. 같은 이름이 대소문자만 달리 둘 이상이면 모호하므로 None."""
    hits = [b for key, b in facts.brands.items() if key.lower() == name.lower()]
    return hits[0] if len(hits) == 1 else None


# ---------------------------------------------------------------------------
# 교차 확인 (지표 테이블 · MetricFactsProvider) — 정답 계산에는 쓰지 않는다
# ---------------------------------------------------------------------------


def metric_table_facts(conn: sqlite3.Connection, category: str, date: str = AS_OF) -> dict:
    params = {"d": date, "c": category}
    market = conn.execute(SQL_METRIC_TABLE_MARKET, params).fetchone()
    brands = {r["brand"]: dict(r) for r in conn.execute(SQL_METRIC_TABLE_BRAND, params)}
    return {"market": dict(market) if market else None, "brands": brands}


def metric_facts_provider_view(
    db_path: Path, brands: list[str], categories: list[str], as_of: str = AS_OF
) -> list[dict[str, Any]]:
    """챗봇이 실제로 받는 DB 사실(`MetricFactsProvider.collect`). 교차 확인 전용."""
    from src.rag.metric_facts import MetricFactsProvider

    provider = MetricFactsProvider(db_path=db_path, as_of=as_of)
    return asyncio.run(provider.collect({"brands": brands, "categories": categories}))


def rel_diff(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    if a == b:
        return 0.0
    denom = max(abs(a), abs(b))
    return abs(a - b) / denom if denom else 0.0


def relative_diff(gold: float | None, other: float | None) -> float | None:
    """|other − gold| / |gold| (정답 기준 상대 차이, 소수 4자리). gold가 0이고 다르면 None."""
    if gold is None or other is None:
        return None
    if gold == other:
        return 0.0
    if gold == 0:
        return None
    return round(abs(other - gold) / abs(gold), 4)


def distribution(values: list[float]) -> dict[str, float | int | None]:
    """개수·최대·중앙값."""
    if not values:
        return {"n": 0, "max": None, "median": None}
    ordered = sorted(values)
    mid = len(ordered) // 2
    median = ordered[mid] if len(ordered) % 2 else (ordered[mid - 1] + ordered[mid]) / 2
    return {"n": len(ordered), "max": ordered[-1], "median": round(median, 4)}


def within_tolerance(gold: float | None, other: float | None) -> bool:
    """numeric_accuracy와 같은 판정(상대 오차 10%, 0은 절대 비교)을 두 출처 사이에 적용."""
    if gold is None or other is None:
        return gold is None and other is None
    if gold == 0:
        return abs(other) < 1e-9
    return abs(other - gold) / abs(gold) <= NUMERIC_TOLERANCE


# ---------------------------------------------------------------------------
# KG (큐레이션 트리플만)
# ---------------------------------------------------------------------------


def load_curated_triples(kg_path: Path) -> list[dict[str, Any]]:
    with open(kg_path, encoding="utf-8") as f:
        triples = json.load(f)["triples"]
    return [t for t in triples if t.get("source") in CURATED_KG_SOURCES]


def objects(triples: list[dict], subject: str, predicate: str) -> list[str]:
    return sorted(
        {t["object"] for t in triples if t["subject"] == subject and t["predicate"] == predicate}
    )


def subjects(triples: list[dict], predicate: str, obj: str) -> list[str]:
    return sorted(
        {t["subject"] for t in triples if t["predicate"] == predicate and t["object"] == obj}
    )


def norm_id(name: str) -> str:
    """골드셋 엔티티·엣지 표기. eval/runner.py `_normalize_edge_node`와 같은 규칙."""
    s = str(name).lower().replace("'", "")
    return re.sub(r"[^a-z0-9]+", "_", s).strip("_")


def edge(subject: str, predicate: str, obj: str) -> str:
    return f"{norm_id(subject)} -{predicate}-> {norm_id(obj)}"


# ---------------------------------------------------------------------------
# 레코드 · 파일
# ---------------------------------------------------------------------------


def build_record(
    *,
    item_id: str,
    question: str,
    answer: str,
    question_type: str,
    generator: str,
    domain: str,
    difficulty: str,
    requires_kg: bool,
    gold_source: str,
    expected_values: dict[str, float] | None = None,
    kg_entities: list[str] | None = None,
    kg_edges: list[str] | None = None,
    concepts: list[str] | None = None,
    as_of: str | None = AS_OF,
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """기존 골든 레코드(laneige_golden_v2.jsonl)와 같은 모양의 문항."""
    metadata: dict[str, Any] = {
        "requires_kg": requires_kg,
        "domain": domain,
        "difficulty": difficulty,
        "gold_source": gold_source,
    }
    if as_of:
        metadata["as_of"] = as_of
    metadata.update({"generated": True, "generator": generator, "question_type": question_type})
    metadata.update(extra_metadata or {})
    return {
        "id": item_id,
        "question": question,
        "gold": {
            "answer": answer,
            "doc_chunk_ids": [],
            "kg_entities": kg_entities or [],
            "kg_edges": kg_edges or [],
            "concepts": concepts or [],
            "constraints": [],
            "expected_values": expected_values or {},
            "doc_chunk_groups": [],
        },
        "metadata": metadata,
    }


def dumps_jsonl(records: list[dict[str, Any]]) -> str:
    return "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records)


def dumps_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2) + "\n"


def write_or_check(outputs: dict[Path, str], check: bool) -> int:
    """check=True면 디스크 파일과 비교(다르면 1), 아니면 쓴다."""
    if check:
        stale = [
            str(p)
            for p, content in outputs.items()
            if not p.exists() or p.read_text(encoding="utf-8") != content
        ]
        if stale:
            print(f"재생성 결과와 다른 파일: {stale}")
            return 1
        print("OK — 재생성 결과가 디스크 파일과 같다")
        return 0
    for path, content in outputs.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        print(f"저장: {path}")
    return 0


def fmt_pct(v: float | None) -> str:
    return "없음" if v is None else f"{v:.2f}%".replace(".00%", ".0%")


def fmt_num(v: float | None, digits: int = 4) -> str:
    return "없음" if v is None else f"{v:.{digits}f}"
