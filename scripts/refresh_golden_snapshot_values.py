#!/usr/bin/env python3
"""snapshot 문항의 골드 수치를 크롤 DB에서 생성한다 (멱등, --dry-run 지원).

배경 (docs/eval/rag-eval-review-2026-09-06.md §2-c)
--------------------------------------------------
데이터형 문항의 골드는 도메인 지식으로 작성된 추정치였고 크롤 데이터와 대조된
적이 없다. 골드 5.2% vs 실측 2.0%처럼 어긋난 문항이 12건 이상 확인됐다.
이 스크립트는 3단계에서 `gold_source="snapshot"`으로 분류된 56문항에 대해
`metadata.as_of` 시점의 DB에서 수치를 조회해 `gold.expected_values`를 채우고
`gold.answer`의 수치도 같은 값으로 갱신한다.

원칙
----
- **근거는 DB뿐이다.** 시스템의 현재 답변을 보고 골드를 맞추지 않는다.
- **없는 값은 지어내지 않는다.** as_of 시점에 값이 없으면
  (a) DB가 "없음"을 말할 수 있는 경우(브랜드가 Top 100에 없음)는 0으로 단정하고
      답변도 "해당 스냅샷 Top 100에 없음"으로 바꾼다 — 이것도 DB의 답이다.
  (b) 애초에 관측이 없는 경우(해당 월 스냅샷 부재, 미수집 컬럼)는
      domain_expectation으로 강등하고 이유를 남긴다.
- 문항마다 근거 SQL을 이 파일에 적는다. 재현은 스크립트를 읽으면 된다.

사용법:
    python3 scripts/refresh_golden_snapshot_values.py --dry-run   # 변경 전후 diff
    python3 scripts/refresh_golden_snapshot_values.py
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASET = REPO_ROOT / "eval" / "data" / "golden" / "laneige_golden_v2.jsonl"
DB_PATH = REPO_ROOT / "data" / "amore_data.db"


@dataclass
class Result:
    """문항 갱신 결과."""

    answer: str | None = None
    values: dict[str, float] = field(default_factory=dict)
    demote_reason: str | None = None  # 채워지면 domain_expectation으로 강등


# =============================================================================
# 공통 조회 (모든 SQL은 as_of 스냅샷 하나만 본다)
# =============================================================================


class DB:
    def __init__(self, conn: sqlite3.Connection, as_of: str):
        self.conn = conn
        self.as_of = as_of

    def one(self, sql: str, params: tuple = ()) -> sqlite3.Row | None:
        return self.conn.execute(sql, params).fetchone()

    def all(self, sql: str, params: tuple = ()) -> list[sqlite3.Row]:
        return self.conn.execute(sql, params).fetchall()

    # --- brand_metrics --------------------------------------------------
    def brand(self, category: str, brand: str) -> sqlite3.Row | None:
        """브랜드의 카테고리 SoS·제품수·CPI와 브랜드 순위.

        SQL: select sos, product_count, cpi,
                    (select count(*)+1 from brand_metrics b2
                      where b2.snapshot_date=? and b2.category_id=? and b2.sos > b1.sos) rank
               from brand_metrics b1
              where snapshot_date=? and category_id=? and brand=? collate nocase
        """
        return self.one(
            """
            select b1.sos, b1.product_count, b1.cpi,
                   (select count(*) + 1 from brand_metrics b2
                     where b2.snapshot_date = ? and b2.category_id = ?
                       and b2.sos > b1.sos) as brand_rank
              from brand_metrics b1
             where b1.snapshot_date = ? and b1.category_id = ?
               and lower(b1.brand) = lower(?)
            """,
            (self.as_of, category, self.as_of, category, brand),
        )

    def top_brands(self, category: str, n: int) -> list[sqlite3.Row]:
        """카테고리 SoS 상위 n개 브랜드.

        SQL: select brand, sos, product_count from brand_metrics
              where snapshot_date=? and category_id=? order by sos desc limit ?
        """
        return self.all(
            """
            select brand, sos, product_count from brand_metrics
             where snapshot_date = ? and category_id = ?
             order by sos desc, brand limit ?
            """,
            (self.as_of, category, n),
        )

    # --- market_metrics -------------------------------------------------
    def hhi(self, category: str) -> float | None:
        """SQL: select hhi from market_metrics where snapshot_date=? and category_id=?"""
        row = self.one(
            "select hhi from market_metrics where snapshot_date = ? and category_id = ?",
            (self.as_of, category),
        )
        return row["hhi"] if row and row["hhi"] is not None else None

    # --- raw_data -------------------------------------------------------
    def products(self, category: str, brand: str) -> list[sqlite3.Row]:
        """SQL: select rank, product_name, price, rating, reviews_count from raw_data
        where snapshot_date=? and category_id=? and brand=? collate nocase
        order by rank"""
        return self.all(
            """
            select rank, product_name, price, rating, reviews_count
              from raw_data
             where snapshot_date = ? and category_id = ? and lower(brand) = lower(?)
             order by rank
            """,
            (self.as_of, category, brand),
        )

    def brand_products_all(self, brand: str) -> list[sqlite3.Row]:
        """SQL: 모든 카테고리에서 해당 브랜드의 제품 (rank 오름차순)"""
        return self.all(
            """
            select category_id, rank, product_name, price, rating, reviews_count
              from raw_data
             where snapshot_date = ? and lower(brand) = lower(?)
             order by rank
            """,
            (self.as_of, brand),
        )

    def top_products(self, category: str, n: int) -> list[sqlite3.Row]:
        """SQL: select rank, brand, product_name, price from raw_data
        where snapshot_date=? and category_id=? order by rank limit ?"""
        return self.all(
            """
            select rank, brand, product_name, price from raw_data
             where snapshot_date = ? and category_id = ? order by rank limit ?
            """,
            (self.as_of, category, n),
        )

    def category_avg_price(self, category: str) -> float | None:
        """카테고리 Top 100 평균 가격.

        market_metrics.category_avg_price는 대부분 NULL이라 raw_data에서 직접 낸다.
        SQL: select avg(price) from raw_data
              where snapshot_date=? and category_id=? and price is not null
        """
        row = self.one(
            """
            select avg(price) as p, count(price) as n from raw_data
             where snapshot_date = ? and category_id = ? and price is not null
            """,
            (self.as_of, category),
        )
        return round(row["p"], 2) if row and row["n"] else None

    def snapshot_months(self) -> set[str]:
        """SQL: select distinct substr(snapshot_date,1,7) from brand_metrics"""
        return {
            r[0] for r in self.all("select distinct substr(snapshot_date, 1, 7) from brand_metrics")
        }


def _pct(value: float) -> str:
    """수치 추출기가 읽기 쉽도록 소수 1자리로 고정한다."""
    return f"{value:.1f}%"


def _name(row: sqlite3.Row, width: int = 34) -> str:
    """제품명을 다듬는다. 이름이 이미 브랜드로 시작하면 브랜드를 덧붙이지 않는다."""
    name = (row["product_name"] or "").split(":")[0].strip()
    brand = (row["brand"] or "").strip() if "brand" in row.keys() else ""
    if brand and not name.lower().startswith(brand.lower()):
        name = f"{brand} {name}"
    return name[:width].rstrip()


def _brand_list(rows: list[sqlite3.Row]) -> str:
    return ", ".join(f"{r['brand']} {_pct(r['sos'])}" for r in rows)


# =============================================================================
# 문항별 규칙
# =============================================================================

HANDLERS: dict[str, Callable[[DB], Result]] = {}


def handler(item_id: str):
    def wrap(fn):
        HANDLERS[item_id] = fn
        return fn

    return wrap


# --- LANEIGE Lip Care SoS 계열 ------------------------------------------
def _laneige_lip_care(db: DB) -> sqlite3.Row:
    return db.brand("lip_care", "LANEIGE")


@handler("lg048")
def lg048(db: DB) -> Result:
    b = _laneige_lip_care(db)
    top = db.top_brands("lip_care", 1)[0]
    return Result(
        answer=(
            f"현재 LANEIGE의 Lip Care 카테고리 SoS는 {_pct(b['sos'])}입니다. "
            f"Top 100 기준 {b['product_count']}개 제품이 랭크되어 있으며, "
            f"이는 Lip Care 브랜드 중 {b['brand_rank']}위 수준입니다"
            f"(1위 {top['brand']} {_pct(top['sos'])})."
        ),
        values={"sos": b["sos"]},
    )


@handler("lg164")
def lg164(db: DB) -> Result:
    b = _laneige_lip_care(db)
    return Result(
        answer=(
            "영어로 질문하셨군요. LANEIGE의 SoS(Share of Shelf) 정보를 안내해 드립니다. "
            f"LANEIGE Lip Care 카테고리 SoS는 약 {_pct(b['sos'])}로, "
            f"Top 100 기준 {b['product_count']}개 제품이 랭크되어 있습니다. "
            "더 자세한 카테고리별 SoS가 필요하시면 알려주세요."
        ),
        values={"sos": b["sos"]},
    )


@handler("lg169")
def lg169(db: DB) -> Result:
    b = _laneige_lip_care(db)
    return Result(
        answer=(
            "아니요, LANEIGE의 SoS가 100%일 수는 없습니다. SoS 100%는 해당 카테고리 "
            "Top 100 전부가 LANEIGE 제품인 경우를 의미하는데, 현실적으로 불가능합니다. "
            f"현재 LANEIGE의 Lip Care 카테고리 SoS는 약 {_pct(b['sos'])}"
            f"(Top 100 중 {b['product_count']}개 제품)입니다."
        ),
        values={"sos": b["sos"]},
    )


@handler("lg170")
def lg170(db: DB) -> Result:
    b = _laneige_lip_care(db)
    return Result(
        answer=(
            "SoS는 일별로 소폭 변동이 있을 수 있으나, 크롤링 주기(일 1회)에 따라 최신 "
            f"데이터를 기준으로 합니다. 최근 LANEIGE Lip Care SoS는 약 {_pct(b['sos'])} "
            "수준입니다. 일별 정확한 비교는 시스템의 크롤링 이력을 확인해 주세요."
        ),
        values={"sos": b["sos"]},
    )


# --- HHI 계열 ------------------------------------------------------------
@handler("lg049")
def lg049(db: DB) -> Result:
    hhi = db.hhi("lip_care")
    top = db.top_brands("lip_care", 5)
    return Result(
        answer=(
            f"Lip Care 카테고리 HHI는 {hhi:.4f}로 매우 낮은 집중도(고도 분산) 시장입니다. "
            f"상위 브랜드는 {_brand_list(top)}로, 어느 브랜드도 두 자릿수 점유율을 "
            "갖지 못하는 구조입니다."
        ),
        values={"hhi": round(hhi, 4)},
    )


@handler("lg055")
def lg055(db: DB) -> Result:
    lip_care, lip_makeup = db.hhi("lip_care"), db.hhi("lip_makeup")
    lc_top, lm_top = db.top_brands("lip_care", 2), db.top_brands("lip_makeup", 2)
    return Result(
        answer=(
            f"Lip Care HHI: {lip_care:.4f}, Lip Makeup HHI: {lip_makeup:.4f}. "
            f"Lip Makeup이 더 집중된 시장으로, 상위 브랜드({_brand_list(lm_top)})가 "
            f"두 자릿수 점유율을 갖습니다. Lip Care는 상위 브랜드({_brand_list(lc_top)})조차 "
            "한 자릿수에 그치는 고도 분산 구조입니다."
        ),
        values={"lip_care_hhi": round(lip_care, 4), "lip_makeup_hhi": round(lip_makeup, 4)},
    )


@handler("lg134")
def lg134(db: DB) -> Result:
    beauty, lip_care, skin_care = db.hhi("beauty"), db.hhi("lip_care"), db.hhi("skin_care")
    return Result(
        answer=(
            f"Amazon US Beauty & Personal Care 전체 HHI는 {beauty:.4f}로 매우 낮은 집중도를 "
            "보입니다. 수천 개 브랜드가 경쟁하는 고도 분산 시장입니다. 카테고리별로는 "
            f"Lip Care({lip_care:.4f})와 Skin Care({skin_care:.4f})가 전체 평균과 비슷한 "
            "수준입니다."
        ),
        values={"beauty_hhi": round(beauty, 4)},
    )


@handler("lg148")
def lg148(db: DB) -> Result:
    lc, fp, sc = db.hhi("lip_care"), db.hhi("face_powder"), db.hhi("skin_care")
    return Result(
        answer=(
            f"LANEIGE Lip Sleeping Mask → Lip Care 카테고리 → HHI {lc:.4f}. "
            f"LANEIGE Neo 라인 → Face Powder 카테고리 → HHI {fp:.4f}. "
            f"Water Bank 계열이 속한 Skin Care 카테고리 → HHI {sc:.4f}. "
            "세 카테고리 모두 0.1 미만의 고도 분산 시장입니다."
        ),
        values={
            "lip_care_hhi": round(lc, 4),
            "face_powder_hhi": round(fp, 4),
            "skin_care_hhi": round(sc, 4),
        },
    )


@handler("lg149")
def lg149(db: DB) -> Result:
    hhi = db.hhi("lip_care")
    b = _laneige_lip_care(db)
    top = db.top_brands("lip_care", 2)
    return Result(
        answer=(
            f"LANEIGE가 가장 강한 카테고리는 Lip Care입니다. Lip Care HHI는 {hhi:.4f}로 "
            f"고도 분산 시장이며, LANEIGE SoS {_pct(b['sos'])}로 브랜드 중 {b['brand_rank']}위입니다. "
            f"1위 {top[0]['brand']}({_pct(top[0]['sos'])}), 2위 {top[1]['brand']}"
            f"({_pct(top[1]['sos'])})와의 격차가 커 상위권 진입 여지가 남아 있습니다."
        ),
        values={"hhi": round(hhi, 4), "laneige_sos": b["sos"]},
    )


@handler("lg196")
def lg196(db: DB) -> Result:
    rows = [(c, db.hhi(c)) for c in ("face_powder", "skin_care", "lip_care", "lip_makeup")]
    hardest = max(rows, key=lambda r: r[1])
    names = {
        "face_powder": "Face Powder",
        "skin_care": "Skin Care",
        "lip_care": "Lip Care",
        "lip_makeup": "Lip Makeup",
    }
    listed = ", ".join(f"{names[c]} HHI {h:.4f}" for c, h in rows)
    return Result(
        answer=(
            f"LANEIGE가 진입한 카테고리의 집중도: {listed}. HHI가 가장 높은(=상위 브랜드 "
            f"집중이 강한) 카테고리는 {names[hardest[0]]}이며, LANEIGE가 상위권과 겨루기에 "
            "가장 불리한 구조입니다. 나머지는 모두 0.07 이하의 고도 분산 시장입니다."
        ),
        values={f"{c}_hhi": round(h, 4) for c, h in rows},
    )


@handler("lg193")
def lg193(db: DB) -> Result:
    cosrx = db.brand("skin_care", "COSRX")
    hhi = db.hhi("skin_care")
    top = db.top_brands("skin_care", 1)[0]
    return Result(
        answer=(
            "COSRX의 강점은 달팽이 뮤신 등 효능 입증 성분과 가성비, 10만 건대 리뷰 축적입니다. "
            f"다만 {db.as_of} 스냅샷 기준 Skin Care Top 100 내 COSRX 제품은 "
            f"{cosrx['product_count']}개로 SoS {_pct(cosrx['sos'])}에 그칩니다. "
            f"Skin Care HHI는 {hhi:.4f}(고도 분산)이며 1위는 {top['brand']}"
            f"({_pct(top['sos'])})입니다."
        ),
        values={"cosrx_sos": cosrx["sos"], "skin_care_hhi": round(hhi, 4)},
    )


# --- 브랜드별 SoS ---------------------------------------------------------
def _absent_brand_answer(db: DB, brand: str, category_label: str) -> str:
    return (
        f"{db.as_of} 스냅샷 기준 {category_label} Top 100에 {brand} 제품은 없습니다. "
        f"따라서 해당 카테고리 SoS는 0%입니다. Top 100 밖의 판매는 이 시스템이 "
        "수집하지 않으므로 브랜드의 전체 실적을 뜻하지는 않습니다."
    )


@handler("lg057")
def lg057(db: DB) -> Result:
    # BIODANCE는 skin_care·beauty에만 있고 lip_care Top 100에는 없다
    b = db.brand("lip_care", "BIODANCE")
    if b is None:
        sc = db.brand("skin_care", "BIODANCE")
        extra = f" 같은 스냅샷에서 Skin Care SoS는 {_pct(sc['sos'])}입니다." if sc else ""
        return Result(
            answer=_absent_brand_answer(db, "BIODANCE", "Lip Care") + extra, values={"sos": 0.0}
        )
    return Result(
        answer=f"BIODANCE의 Lip Care SoS는 {_pct(b['sos'])}입니다.", values={"sos": b["sos"]}
    )


@handler("lg066")
def lg066(db: DB) -> Result:
    b = db.brand("skin_care", "Beauty of Joseon")
    if b is None:
        return Result(
            answer=_absent_brand_answer(db, "Beauty of Joseon", "Skin Care"), values={"sos": 0.0}
        )
    return Result(
        answer=f"Beauty of Joseon의 Skin Care SoS는 {_pct(b['sos'])}입니다.",
        values={"sos": b["sos"]},
    )


@handler("lg070")
def lg070(db: DB) -> Result:
    b = db.brand("skin_care", "MEDICUBE")
    hhi = db.hhi("skin_care")
    return Result(
        answer=(
            "MEDICUBE는 AGE-R 디바이스와 Zero 라인으로 Skin Care에서 빠르게 성장한 K-Beauty "
            f"브랜드입니다. {db.as_of} 스냅샷 기준 Skin Care SoS는 {_pct(b['sos'])}"
            f"(Top 100 내 {b['product_count']}개)로 카테고리 {b['brand_rank']}위입니다. "
            f"단일 브랜드로는 가장 높은 점유율이지만 카테고리 HHI는 {hhi:.4f}로 여전히 "
            "고도 분산 상태입니다."
        ),
        values={"sos": b["sos"]},
    )


@handler("lg060")
def lg060(db: DB) -> Result:
    top = db.top_brands("beauty", 5)
    laneige = db.brand("beauty", "LANEIGE")
    listed = ", ".join(f"{i}) {r['brand']} {_pct(r['sos'])}" for i, r in enumerate(top, 1))
    tail = (
        f" LANEIGE는 {_pct(laneige['sos'])}입니다."
        if laneige
        else " LANEIGE는 이 스냅샷의 Beauty & Personal Care Top 100에 없습니다(0%)."
    )
    return Result(
        answer=f"Beauty & Personal Care 전체 카테고리 SoS 상위 브랜드: {listed}.{tail}",
        values={f"top{i}_sos": r["sos"] for i, r in enumerate(top, 1)},
    )


@handler("lg068")
def lg068(db: DB) -> Result:
    top = db.top_brands("face_powder", 5)
    laneige = db.brand("face_powder", "LANEIGE")
    listed = ", ".join(f"{r['brand']} {_pct(r['sos'])}" for r in top)
    return Result(
        answer=(
            f"Face Powder 카테고리 주요 브랜드 SoS: {listed}. "
            f"LANEIGE는 {_pct(laneige['sos'])}(Top 100 내 {laneige['product_count']}개)로 "
            "틈새 포지션입니다."
        ),
        values={"laneige_sos": laneige["sos"], "top1_sos": top[0]["sos"]},
    )


@handler("lg109")
def lg109(db: DB) -> Result:
    aq = db.brand("lip_care", "Aquaphor")
    ln = _laneige_lip_care(db)
    lc_products = db.products("lip_care", "LANEIGE")
    price = lc_products[0]["price"] if lc_products else None
    return Result(
        answer=(
            f"Aquaphor vs LANEIGE Lip Care: {db.as_of} 스냅샷 기준 Aquaphor SoS "
            f"{_pct(aq['sos'])}(카테고리 {aq['brand_rank']}위), LANEIGE SoS {_pct(ln['sos'])}"
            f"(카테고리 {ln['brand_rank']}위)입니다. LANEIGE Lip Sleeping Mask는 ${price:.2f}로 "
            "대중 브랜드 대비 3~5배 높은 프리미엄 포지션이며, 직접 가격 경쟁보다 "
            "슬리핑 마스크라는 차별화된 용도로 공존합니다."
        ),
        values={"aquaphor_sos": aq["sos"], "laneige_sos": ln["sos"]},
    )


@handler("lg113")
def lg113(db: DB) -> Result:
    bb = db.brand("lip_care", "Burt's Bees")
    ln = _laneige_lip_care(db)
    top = db.top_brands("lip_care", 1)[0]
    return Result(
        answer=(
            f"Burt's Bees vs LANEIGE Lip Care: Burt's Bees는 천연/유기농 포지션으로 SoS "
            f"{_pct(bb['sos'])}(카테고리 {bb['brand_rank']}위, 1위는 {top['brand']} "
            f"{_pct(top['sos'])}), LANEIGE는 프리미엄 K-Beauty 포지션으로 SoS {_pct(ln['sos'])}"
            f"({ln['brand_rank']}위)입니다. 두 브랜드는 다른 사용 상황을 타깃하여 공존합니다."
        ),
        values={"burts_bees_sos": bb["sos"], "laneige_sos": ln["sos"]},
    )


@handler("lg150")
def lg150(db: DB) -> Result:
    top = db.top_brands("lip_care", 1)[0]
    ln = _laneige_lip_care(db)
    return Result(
        answer=(
            f"LANEIGE 대표 제품 Lip Sleeping Mask → Lip Care 카테고리 → {db.as_of} 스냅샷 "
            f"기준 1위 브랜드는 {top['brand']}(SoS {_pct(top['sos'])}, Top 100 내 "
            f"{top['product_count']}개)입니다. LANEIGE는 SoS {_pct(ln['sos'])}로 "
            f"{ln['brand_rank']}위입니다."
        ),
        values={"top_brand_sos": top["sos"], "laneige_sos": ln["sos"]},
    )


@handler("lg161")
def lg161(db: DB) -> Result:
    top = db.top_brands("lip_care", 1)[0]
    ln = _laneige_lip_care(db)
    gap = round(top["sos"] - ln["sos"], 2)
    return Result(
        answer=(
            f"LANEIGE 주력 카테고리 Lip Care: 1위 {top['brand']} SoS {_pct(top['sos'])} vs "
            f"LANEIGE {_pct(ln['sos'])} → 격차 약 {gap:.1f}%p입니다. 격차를 좁히려면 "
            f"LANEIGE가 Top 100 진입 제품을 현재 {ln['product_count']}개에서 "
            f"{top['product_count']}개 수준으로 늘려야 합니다."
        ),
        values={"gap": gap, "top_brand_sos": top["sos"], "laneige_sos": ln["sos"]},
    )


@handler("lg151")
def lg151(db: DB) -> Result:
    # "K-Beauty 1위 = COSRX"라는 전제 자체가 스냅샷과 다르다 (MEDICUBE가 최상위)
    hhi = db.hhi("skin_care")
    top = db.top_brands("skin_care", 1)[0]
    cosrx = db.brand("skin_care", "COSRX")
    return Result(
        answer=(
            f"{db.as_of} 스냅샷에서 Skin Care SoS가 가장 높은 K-Beauty 브랜드는 "
            f"{top['brand']}({_pct(top['sos'])})이며, COSRX는 {_pct(cosrx['sos'])}에 그칩니다. "
            f"주력 카테고리인 Skin Care의 HHI는 {hhi:.4f}로 수천 개 브랜드가 경쟁하는 "
            "고도 분산 시장입니다."
        ),
        values={
            "skin_care_hhi": round(hhi, 4),
            "top_kbeauty_sos": top["sos"],
            "cosrx_sos": cosrx["sos"],
        },
    )


@handler("lg154")
def lg154(db: DB) -> Result:
    cosrx = db.brand("skin_care", "COSRX")
    prods = db.products("skin_care", "COSRX")
    p = prods[0] if prods else None
    detail = (
        f" 해당 제품은 Skin Care {p['rank']}위, 리뷰 {p['reviews_count']:,}건입니다." if p else ""
    )
    return Result(
        answer=(
            "COSRX 최인기 제품 Snail Mucin 계열 → Skin Care 카테고리 → COSRX의 Skin Care "
            f"SoS는 {_pct(cosrx['sos'])}(Top 100 내 {cosrx['product_count']}개)입니다.{detail} "
            "단일 제품 의존도가 높은 구조입니다."
        ),
        values={"cosrx_sos": cosrx["sos"]},
    )


@handler("lg155")
def lg155(db: DB) -> Result:
    cosrx = db.brand("skin_care", "COSRX")
    ln = db.brand("skin_care", "LANEIGE")
    ln_sos = ln["sos"] if ln else 0.0
    top = db.top_brands("skin_care", 1)[0]
    return Result(
        answer=(
            f"{db.as_of} 스냅샷 기준 COSRX의 Skin Care SoS는 {_pct(cosrx['sos'])}이고 "
            f"LANEIGE는 Skin Care Top 100에 제품이 없어 SoS {_pct(ln_sos)}입니다. "
            f"COSRX 수준에 이르려면 Top 100 진입 제품을 최소 {cosrx['product_count']}개 "
            f"확보해야 하며, 카테고리 선두({top['brand']} {_pct(top['sos'])})까지는 "
            "훨씬 큰 격차가 있습니다."
        ),
        values={"cosrx_sos": cosrx["sos"], "laneige_sos": ln_sos},
    )


@handler("lg158")
def lg158(db: DB) -> Result:
    portfolio = []
    values: dict[str, float] = {}
    for brand in ("LANEIGE", "Innisfree", "Sulwhasoo", "Etude"):
        rows = [
            (c, db.brand(c, brand))
            for c in ("lip_care", "lip_makeup", "face_powder", "skin_care", "beauty")
        ]
        present = [(c, r) for c, r in rows if r]
        if present:
            detail = ", ".join(f"{c} {_pct(r['sos'])}" for c, r in present)
            portfolio.append(f"{brand}({detail})")
            values[f"{brand.lower()}_max_sos"] = max(r["sos"] for _, r in present)
        else:
            portfolio.append(f"{brand}(Top 100 내 없음)")
            values[f"{brand.lower()}_max_sos"] = 0.0
    return Result(
        answer=(
            f"LANEIGE 모회사 아모레퍼시픽의 {db.as_of} 스냅샷 Amazon 포트폴리오: "
            + ", ".join(portfolio)
            + ". 모니터링 카테고리 안에서는 LANEIGE가 가장 넓게 진입해 있습니다."
        ),
        values=values,
    )


@handler("lg195")
def lg195(db: DB) -> Result:
    tirtir = db.brand("face_powder", "TIRTIR")
    ln = db.brand("face_powder", "LANEIGE")
    top = db.top_brands("face_powder", 1)[0]
    if tirtir is None:
        return Result(
            answer=(
                f"{db.as_of} 스냅샷 기준 Face Powder Top 100에 TIRTIR 제품은 없어 SoS는 "
                f"0%입니다. LANEIGE는 {_pct(ln['sos'])}(Top 100 내 {ln['product_count']}개)로 "
                f"TIRTIR보다 앞서 있으며, 카테고리 1위는 {top['brand']}({_pct(top['sos'])})입니다. "
                "TIRTIR의 바이럴 성과는 이 스냅샷의 Top 100에서는 확인되지 않습니다."
            ),
            values={"tirtir_sos": 0.0, "laneige_sos": ln["sos"]},
        )
    return Result(
        answer=(
            f"Face Powder SoS: TIRTIR {_pct(tirtir['sos'])} vs LANEIGE {_pct(ln['sos'])}. "
            "TIRTIR는 해당 카테고리 전문화, LANEIGE는 Lip Care 주력이라는 차이가 있습니다."
        ),
        values={"tirtir_sos": tirtir["sos"], "laneige_sos": ln["sos"]},
    )


# --- 제품 순위·가격·리뷰 ---------------------------------------------------
def _lip_sleeping_mask(db: DB) -> sqlite3.Row | None:
    """SQL: raw_data에서 lip_care의 LANEIGE 제품 중 이름에 'Sleeping'이 든 최상위 행"""
    return db.one(
        """
        select rank, product_name, price, rating, reviews_count from raw_data
         where snapshot_date = ? and category_id = 'lip_care'
           and lower(brand) = 'laneige' and product_name like '%Sleeping%'
         order by rank limit 1
        """,
        (db.as_of,),
    )


@handler("lg071")
def lg071(db: DB) -> Result:
    p = _lip_sleeping_mask(db)
    others = db.products("lip_care", "LANEIGE")
    return Result(
        answer=(
            f"LANEIGE Lip Sleeping Mask는 {db.as_of} 스냅샷 기준 Lip Care 카테고리 "
            f"{p['rank']}위입니다. 같은 스냅샷에서 LANEIGE는 Lip Care Top 100에 "
            f"{len(others)}개 제품(최상위 {p['rank']}위)을 올려 두고 있습니다."
        ),
        values={"rank": float(p["rank"])},
    )


@handler("lg162")
def lg162(db: DB) -> Result:
    p = _lip_sleeping_mask(db)
    return Result(
        answer=(
            "'라에니즈'는 'LANEIGE(라네즈)'의 오타로 보입니다. LANEIGE Lip Sleeping Mask의 "
            f"Lip Care 카테고리 순위를 안내해 드리겠습니다. {db.as_of} 스냅샷 기준 "
            f"Lip Care {p['rank']}위입니다."
        ),
        values={"rank": float(p["rank"])},
    )


@handler("lg083")
def lg083(db: DB) -> Result:
    rows = db.brand_products_all(db_brand := "LANEIGE")
    best = rows[0]
    return Result(
        answer=(
            f"{db.as_of} 스냅샷에서 {db_brand}의 최상위 순위 제품은 "
            f"{_name(best, 40)}({best['category_id']} {best['rank']}위)입니다. "
            f"LANEIGE는 모니터링 카테고리 전체에 {len(rows)}개 제품이 Top 100 안에 있습니다."
        ),
        values={"best_rank": float(best["rank"])},
    )


@handler("lg091")
def lg091(db: DB) -> Result:
    rows = db.all(
        """
        select rank, product_name from raw_data
         where snapshot_date = ? and category_id = 'lip_care'
           and lower(brand) = 'laneige' and product_name like '%Sleeping%'
         order by rank
        """,
        (db.as_of,),
    )
    return Result(
        answer=(
            f"{db.as_of} 스냅샷의 Lip Care Top 100에는 LANEIGE Lip Sleeping Mask가 "
            f"{len(rows)}개 항목({', '.join(f'{r["rank"]}위' for r in rows)}) 올라 있습니다. "
            "향별로 분리된 리스팅은 이 스냅샷에서 확인되지 않아, 향별 순위 비교는 "
            "현재 수집 범위로는 답할 수 없습니다."
        ),
        values={"sleeping_mask_listings": float(len(rows))},
    )


@handler("lg073")
def lg073(db: DB) -> Result:
    rows = [r for r in db.brand_products_all("LANEIGE") if r["price"] is not None]
    avg = round(sum(r["price"] for r in rows) / len(rows), 2)
    listed = ", ".join(f"{_name(r, 28)} ${r['price']:.2f}" for r in rows)
    return Result(
        answer=(
            f"{db.as_of} 스냅샷에서 가격이 수집된 LANEIGE 제품: {listed}. "
            f"평균 가격은 약 ${avg:.2f}입니다(가격 미수집 제품 제외)."
        ),
        values={"avg_price": avg},
    )


@handler("lg074")
def lg074(db: DB) -> Result:
    p = _lip_sleeping_mask(db)
    cat_avg = db.category_avg_price("lip_care")
    competitors = db.all(
        """
        select brand, product_name, price from raw_data
         where snapshot_date = ? and category_id = 'lip_care'
           and lower(brand) != 'laneige' and price is not null
         order by rank limit 4
        """,
        (db.as_of,),
    )
    listed = ", ".join(f"{r['brand']} ${r['price']:.2f}" for r in competitors)
    return Result(
        answer=(
            f"LANEIGE Lip Sleeping Mask(${p['price']:.2f})와 같은 스냅샷의 Lip Care 상위 "
            f"경쟁 제품 가격: {listed}. Lip Care Top 100 평균 가격은 ${cat_avg:.2f}로, "
            f"Lip Sleeping Mask는 카테고리 평균의 약 {p['price'] / cat_avg:.1f}배입니다."
        ),
        values={"lip_sleeping_mask_price": p["price"], "category_avg_price": cat_avg},
    )


@handler("lg152")
def lg152(db: DB) -> Result:
    cat_avg = db.category_avg_price("lip_care")
    p = _lip_sleeping_mask(db)
    ratio = round(p["price"] / cat_avg, 2)
    return Result(
        answer=(
            f"LANEIGE Lip Sleeping Mask → Lip Care 카테고리 → {db.as_of} 스냅샷의 "
            f"Lip Care Top 100 평균 가격은 ${cat_avg:.2f}입니다. Lip Sleeping Mask "
            f"${p['price']:.2f}는 카테고리 평균 대비 약 {ratio:.2f}배로, 카테고리 안에서 "
            "높은 프리미엄 포지션입니다."
        ),
        values={"category_avg_price": cat_avg, "price_ratio": ratio},
    )


@handler("lg160")
def lg160(db: DB) -> Result:
    rows = [r for r in db.brand_products_all("LANEIGE") if r["price"] is not None]
    top = max(rows, key=lambda r: r["price"])
    cat_avg = db.category_avg_price(top["category_id"])
    ratio = round(top["price"] / cat_avg, 2)
    return Result(
        answer=(
            f"{db.as_of} 스냅샷에서 가장 비싼 LANEIGE 제품은 {_name(top, 38)}"
            f"(${top['price']:.2f}, {top['category_id']})입니다. 해당 카테고리 Top 100 평균 "
            f"가격 ${cat_avg:.2f} 대비 약 {ratio:.2f}배로, 카테고리 평균보다 높은 "
            "프리미엄 포지션입니다."
        ),
        values={"product_price": top["price"], "price_ratio": ratio},
    )


@handler("lg051")
def lg051(db: DB) -> Result:
    # brand_metrics.cpi는 as_of의 lip_care에 NULL이라 raw_data 가격으로 비율을 낸다.
    # (face_powder에는 CPI가 있으나 지수 스케일이 ×100이라 골드의 1.2와 단위가 다르다)
    ln = [r for r in db.products("lip_care", "LANEIGE") if r["price"] is not None]
    cat_avg = db.category_avg_price("lip_care")
    ln_avg = round(sum(r["price"] for r in ln) / len(ln), 2)
    ratio = round(ln_avg / cat_avg, 2)
    fp = db.brand("face_powder", "LANEIGE")
    fp_note = (
        f" Face Powder에서는 지표 테이블의 CPI가 {fp['cpi']:.1f}(카테고리 평균=100 기준)로 "
        "기록돼 있습니다."
        if fp and fp["cpi"] is not None
        else ""
    )
    return Result(
        answer=(
            f"{db.as_of} 스냅샷 기준 LANEIGE의 Lip Care 평균 가격은 ${ln_avg:.2f}, 카테고리 "
            f"Top 100 평균은 ${cat_avg:.2f}로 약 {ratio:.2f}배 프리미엄입니다."
            f"{fp_note} 해당 스냅샷의 Lip Care CPI 컬럼은 비어 있어 원자료 가격에서 "
            "직접 산출했습니다."
        ),
        values={"price_ratio": ratio, "laneige_avg_price": ln_avg},
    )


@handler("lg075")
def lg075(db: DB) -> Result:
    rows = [r for r in db.brand_products_all("LANEIGE") if r["rating"] and r["rating"] >= 4.5]
    listed = ", ".join(
        f"{_name(r, 30)} {r['rating']}점(리뷰 {r['reviews_count']:,}건)" for r in rows
    )
    return Result(
        answer=(
            f"{db.as_of} 스냅샷에서 평점 4.5 이상인 LANEIGE 제품은 {len(rows)}개입니다: "
            f"{listed}."
        ),
        values={"count_rating_4_5_plus": float(len(rows))},
    )


@handler("lg076")
def lg076(db: DB) -> Result:
    rows = sorted(db.brand_products_all("LANEIGE"), key=lambda r: -(r["reviews_count"] or 0))
    top = rows[0]
    runner = rows[1]
    return Result(
        answer=(
            f"{db.as_of} 스냅샷 기준 리뷰 수가 가장 많은 LANEIGE 제품은 "
            f"{_name(top, 34)}로 {top['reviews_count']:,}건입니다. 다음은 "
            f"{_name(runner, 30)}({runner['reviews_count']:,}건)입니다. "
            "높은 리뷰 수는 Amazon 검색 노출에 기여합니다."
        ),
        values={"top_product_reviews": float(top["reviews_count"])},
    )


@handler("lg156")
def lg156(db: DB) -> Result:
    p = _lip_sleeping_mask(db)
    ln = _laneige_lip_care(db)
    return Result(
        answer=(
            f"Lip Sleeping Mask 리뷰 수({p['reviews_count']:,}건) → Amazon 검색 알고리즘 "
            "가중치 → 검색 노출 증가 → BSR 유지 → Top 100 지속 유지 → LANEIGE Lip Care "
            f"SoS 유지의 경로입니다. {db.as_of} 스냅샷에서 이 제품은 Lip Care {p['rank']}위, "
            f"LANEIGE의 Lip Care SoS는 {_pct(ln['sos'])}입니다."
        ),
        values={"reviews": float(p["reviews_count"]), "sos": ln["sos"]},
    )


@handler("lg192")
def lg192(db: DB) -> Result:
    ln = _laneige_lip_care(db)
    p = _lip_sleeping_mask(db)
    return Result(
        answer=(
            f"{db.as_of} 스냅샷 기준 LANEIGE의 Lip Care SoS는 {_pct(ln['sos'])}"
            f"(Top 100 내 {ln['product_count']}개)입니다. 근거: 1) Lip Sleeping Mask가 "
            f"{p['rank']}위·리뷰 {p['reviews_count']:,}건·평점 {p['rating']}점으로 "
            "슬리핑 마스크 세그먼트의 기준 제품, 2) Lip Glowy Balm이 틴티드 립밤 수요를 "
            "흡수, 3) 프리미엄 가격대에도 재구매가 유지되는 구조입니다."
        ),
        values={"sos": ln["sos"]},
    )


@handler("lg194")
def lg194(db: DB) -> Result:
    ln = _laneige_lip_care(db)
    hhi = db.hhi("lip_care")
    top = db.top_brands("lip_care", 1)[0]
    return Result(
        answer=(
            f"LANEIGE Lip Care 성장 가능성: {db.as_of} 스냅샷 기준 SoS {_pct(ln['sos'])}"
            f"({ln['product_count']}개 제품)로, 1위 {top['brand']}({_pct(top['sos'])}) 대비 "
            f"여력이 큽니다. HHI {hhi:.4f}의 고도 분산 시장이라 신규 SKU 진입으로 점유율을 "
            "늘릴 여지가 있습니다. 방법: 신규 향·기능성 Lip Sleeping Mask SKU 추가, "
            "Lip Glowy Balm 라인업 확대, SPF 립케어 진입."
        ),
        values={"current_sos": ln["sos"]},
    )


@handler("lg171")
def lg171(db: DB) -> Result:
    makeup = db.products("lip_makeup", "LANEIGE")
    care = db.products("lip_care", "LANEIGE")
    mk = ", ".join(f"{r['rank']}위" for r in makeup) or "없음"
    ca = ", ".join(f"{r['rank']}위" for r in care) or "없음"
    return Result(
        answer=(
            "'랜지'는 'LANEIGE(라네즈)'의 오타로, '립스틱'은 Lip Makeup 카테고리 제품입니다. "
            f"{db.as_of} 스냅샷 기준 LANEIGE는 Lip Makeup에서 {mk}로 하위권이지만, "
            f"주력인 Lip Care에서는 {ca}로 상위권입니다. LANEIGE는 색조보다 립케어에 "
            "집중하는 브랜드이기 때문입니다."
        ),
        values={
            "lip_makeup_best_rank": float(makeup[0]["rank"]) if makeup else 0.0,
            "lip_care_best_rank": float(care[0]["rank"]) if care else 0.0,
        },
    )


@handler("lg082")
def lg082(db: DB) -> Result:
    rows = db.products("face_powder", "LANEIGE")
    top = db.top_products("face_powder", 3)
    listed = ", ".join(f"{r['rank']}위 {r['brand']}" for r in top)
    if not rows:
        return Result(
            answer=_absent_brand_answer(db, "LANEIGE", "Face Powder"), values={"rank": 0.0}
        )
    p = rows[0]
    return Result(
        answer=(
            f"{db.as_of} 스냅샷 기준 LANEIGE의 Face Powder 최상위 제품은 "
            f"{_name(p, 38)}로 카테고리 {p['rank']}위입니다. 상위 경쟁 제품은 "
            f"{listed} 순입니다."
        ),
        values={"rank": float(p["rank"])},
    )


@handler("lg087")
def lg087(db: DB) -> Result:
    b = db.brand("face_powder", "e.l.f.")
    rows = db.products("face_powder", "e.l.f.")[:2]
    listed = ", ".join(f"{_name(r, 32)}({r['rank']}위)" for r in rows)
    return Result(
        answer=(
            f"e.l.f. Cosmetics는 {db.as_of} 스냅샷 기준 Face Powder SoS {_pct(b['sos'])}"
            f"(Top 100 내 {b['product_count']}개)로 카테고리 {b['brand_rank']}위입니다. "
            f"주요 제품: {listed}. 저가 전략으로 LANEIGE와 다른 가격대에서 경쟁합니다."
        ),
        values={"elf_sos": b["sos"], "best_rank": float(rows[0]["rank"])},
    )


@handler("lg089")
def lg089(db: DB) -> Result:
    rows = db.products("skin_care", "COSRX")
    p = rows[0]
    return Result(
        answer=(
            f"{db.as_of} 스냅샷 기준 COSRX Snail Mucin 계열은 Skin Care 카테고리 "
            f"{p['rank']}위입니다. 가격 ${p['price']:.2f}, 평점 {p['rating']}점, 리뷰 "
            f"{p['reviews_count']:,}건으로 검증된 제품이며, LANEIGE Water Bank 라인의 "
            "가성비 경쟁 제품입니다."
        ),
        values={"rank": float(p["rank"]), "reviews": float(p["reviews_count"])},
    )


@handler("lg072")
def lg072(db: DB) -> Result:
    rows = db.top_products("lip_care", 10)
    listed = ", ".join(f"{r['rank']}) {_name(r)}" for r in rows)
    laneige_ranks = [r["rank"] for r in rows if (r["brand"] or "").lower() == "laneige"]
    return Result(
        answer=(
            f"{db.as_of} 스냅샷 기준 Lip Care Top 10: {listed}. "
            + (
                f"LANEIGE는 {', '.join(f'{r}위' for r in laneige_ranks)}로 "
                f"{len(laneige_ranks)}개 제품이 Top 10에 있습니다."
                if laneige_ranks
                else "LANEIGE 제품은 Top 10에 없습니다."
            )
        ),
        values={"laneige_in_top10": float(len(laneige_ranks))},
    )


@handler("lg095")
def lg095(db: DB) -> Result:
    rows = db.top_products("face_powder", 5)
    listed = ", ".join(
        f"{r['rank']}) {_name(r)}" + (f"(${r['price']:.2f})" if r["price"] else "") for r in rows
    )
    return Result(
        answer=f"{db.as_of} 스냅샷 기준 Face Powder Top 5 제품: {listed}.",
        values={"top1_rank": 1.0},
    )


@handler("lg050")
def lg050(db: DB) -> Result:
    ln = _laneige_lip_care(db)
    hhi = db.hhi("lip_care")
    prices = [r["price"] for r in db.products("lip_care", "LANEIGE") if r["price"]]
    cat_avg = db.category_avg_price("lip_care")
    ratio = round((sum(prices) / len(prices)) / cat_avg, 2)
    top = db.top_brands("lip_care", 1)[0]
    return Result(
        answer=(
            f"LANEIGE 시장 지표 종합({db.as_of} 스냅샷): SoS(Lip Care) {_pct(ln['sos'])} — "
            f"브랜드 {ln['brand_rank']}위, 1위 {top['brand']} {_pct(top['sos'])}. "
            f"HHI(Lip Care) {hhi:.4f} — 고도 분산 시장. 가격은 카테고리 평균의 약 "
            f"{ratio:.2f}배로 프리미엄 포지션입니다. 분산된 시장에서 프리미엄 가격을 "
            "유지하되 진열 점유율은 아직 상위권과 거리가 있습니다."
        ),
        values={"sos": ln["sos"], "hhi": round(hhi, 4), "price_ratio": ratio},
    )


# --- DB가 "없음"을 말할 수 있는 문항 --------------------------------------
@handler("lg079")
def lg079(db: DB) -> Result:
    rows = db.all(
        """
        select category_id, rank, product_name from raw_data
         where snapshot_date = ? and lower(brand) = 'laneige'
           and product_name like '%Water Bank%'
        """,
        (db.as_of,),
    )
    if not rows:
        return Result(
            answer=(
                f"{db.as_of} 스냅샷의 모니터링 카테고리 Top 100에는 LANEIGE Water Bank "
                "제품이 없어 ASIN을 제시할 수 없습니다. 이 시스템은 카테고리 Top 100에 "
                "오른 제품만 수집하므로, Top 100 밖 제품의 ASIN은 보유하지 않습니다."
            )
        )
    r = rows[0]
    return Result(answer=f"Water Bank 계열은 {r['category_id']} {r['rank']}위에 있습니다.")


@handler("lg086")
def lg086(db: DB) -> Result:
    b = db.brand("lip_makeup", "Rare Beauty")
    if b is None:
        return Result(
            answer=_absent_brand_answer(db, "Rare Beauty", "Lip Makeup"), values={"sos": 0.0}
        )
    rows = db.products("lip_makeup", "Rare Beauty")
    listed = ", ".join(f"{_name(r, 28)}({r['rank']}위)" for r in rows[:3])
    return Result(
        answer=(
            f"Rare Beauty는 {db.as_of} 스냅샷 기준 Lip Makeup SoS {_pct(b['sos'])}입니다. "
            f"주요 제품: {listed}."
        ),
        values={"sos": b["sos"]},
    )


# --- 관측 자체가 없어 강등하는 문항 ---------------------------------------
@handler("lg077")
def lg077(db: DB) -> Result:
    rows = db.all(
        """
        select count(*) as n from product_metrics p
          join raw_data r on r.asin = p.asin and r.snapshot_date = p.snapshot_date
         where p.snapshot_date = ? and lower(r.brand) = 'laneige'
           and p.rank_change is not null
        """,
        (db.as_of,),
    )
    if rows[0]["n"]:
        return Result(demote_reason=None)
    return Result(
        demote_reason=(
            f"{db.as_of} 스냅샷의 product_metrics에 LANEIGE 제품의 rank_change가 한 건도 "
            "없다. 순위 상승 여부를 원자료로 판정할 수 없다."
        )
    )


@handler("lg078")
def lg078(db: DB) -> Result:
    """SQL: raw_data ⋈ products 로 as_of 30일 이내 first_seen 제품을 찾는다."""
    rows = db.all(
        """
        select r.rank, r.brand, r.product_name, p.first_seen_date
          from raw_data r join products p on p.asin = r.asin
         where r.snapshot_date = ? and r.category_id = 'lip_care'
           and p.first_seen_date >= date(?, '-30 day')
         order by r.rank limit 5
        """,
        (db.as_of, db.as_of),
    )
    if not rows:
        return Result(
            demote_reason=(
                f"{db.as_of} 기준 30일 이내에 처음 관측된 Lip Care Top 100 제품이 없다. "
                "스냅샷 간격이 커서 '신규 진입'을 원자료로 판정할 수 없다."
            )
        )
    listed = ", ".join(f"{_name(r)}({r['rank']}위)" for r in rows)
    return Result(
        answer=(f"{db.as_of} 기준 최근 30일 내 처음 관측된 Lip Care Top 100 제품: {listed}."),
        values={"new_entrants": float(len(rows))},
    )


@handler("lg093")
def lg093(db: DB) -> Result:
    row = db.one(
        """
        select sum(is_subscribe_save) as n from raw_data
         where snapshot_date = ? and category_id = 'lip_care'
        """,
        (db.as_of,),
    )
    if row and (row["n"] or 0) > 0:
        rows = db.all(
            """
            select rank, brand, product_name from raw_data
             where snapshot_date = ? and category_id = 'lip_care' and is_subscribe_save = 1
             order by rank limit 3
            """,
            (db.as_of,),
        )
        listed = ", ".join(f"{_name(r)}({r['rank']}위)" for r in rows)
        return Result(
            answer=f"{db.as_of} 스냅샷의 Lip Care Subscribe & Save 제품: {listed}.",
            values={"subscribe_save_count": float(row["n"])},
        )
    return Result(
        demote_reason=(
            f"{db.as_of} 스냅샷의 lip_care raw_data에 is_subscribe_save=1인 행이 없다. "
            "구독 여부가 수집되지 않아 원자료로 판정할 수 없다."
        )
    )


def _period_demote(db: DB, item_id: str, months_back: int, label: str) -> Result:
    """기간 문항: 필요한 월 스냅샷이 있는지 확인하고 없으면 강등."""
    available = db.snapshot_months()
    year, month = int(db.as_of[:4]), int(db.as_of[5:7])
    needed = []
    for i in range(months_back + 1):
        m = month - i
        y = year
        while m <= 0:
            m += 12
            y -= 1
        needed.append(f"{y:04d}-{m:02d}")
    missing = [m for m in needed if m not in available]
    if not missing:
        return Result(demote_reason=None)
    return Result(
        demote_reason=(
            f"{label}에 필요한 월 스냅샷 {', '.join(missing)}가 brand_metrics에 없다"
            f"(보유: {', '.join(sorted(available))}). 추이를 원자료로 산출할 수 없다."
        )
    )


@handler("lg175")
def lg175(db: DB) -> Result:
    return _period_demote(db, "lg175", 2, "지난 3개월 SoS 추이")


@handler("lg176")
def lg176(db: DB) -> Result:
    return _period_demote(db, "lg176", 12, "작년 대비 순위 변화")


@handler("lg178")
def lg178(db: DB) -> Result:
    return _period_demote(db, "lg178", 12, "최근 1년 SoS 성장 추이")


@handler("lg181")
def lg181(db: DB) -> Result:
    period = _period_demote(db, "lg181", 5, "최근 6개월 CPI 추이")
    if period.demote_reason:
        return period
    row = db.brand("lip_care", "LANEIGE")
    if row is None or row["cpi"] is None:
        return Result(
            demote_reason=(
                f"{db.as_of} 스냅샷의 lip_care brand_metrics.cpi가 NULL이라 CPI 추이를 "
                "산출할 수 없다."
            )
        )
    return Result(
        answer=f"LANEIGE Lip Care CPI는 {row['cpi']:.1f}입니다.", values={"cpi": row["cpi"]}
    )


@handler("lg153")
def lg153(db: DB) -> Result:
    """face_powder의 K-Beauty 브랜드 SoS 합계.

    K-Beauty 브랜드 목록은 config/brands.json이 아니라 이 스크립트에 고정한다 —
    골드가 어떤 브랜드를 K-Beauty로 셌는지가 재현 가능해야 하기 때문이다.
    SQL: select brand, sos from brand_metrics
          where snapshot_date=? and category_id='face_powder' and brand in (...)
    """
    kbeauty = ("LANEIGE", "TIRTIR", "Innisfree", "COSRX", "MEDICUBE", "ANUA", "Beauty of Joseon")
    present = [(b, db.brand("face_powder", b)) for b in kbeauty]
    found = [(b, r["sos"]) for b, r in present if r]
    missing = [b for b, r in present if r is None]
    total = round(sum(sos for _, sos in found), 2)
    listed = ", ".join(f"{b} {_pct(sos)}" for b, sos in found) or "없음"
    return Result(
        answer=(
            f"LANEIGE Neo 라인 → Face Powder 카테고리 → {db.as_of} 스냅샷 기준 Top 100에 "
            f"오른 K-Beauty 브랜드는 {listed}로 합산 SoS는 {_pct(total)}입니다. "
            f"{', '.join(missing)}는 이 스냅샷의 Face Powder Top 100에 없습니다."
        ),
        values={"k_beauty_sos_total": total},
    )


# =============================================================================
# 실행
# =============================================================================


def run(rows: list[dict], conn: sqlite3.Connection) -> tuple[list[dict], list[str], list[str]]:
    """분류가 snapshot인 문항을 갱신한다. (갱신된 행, 변경 로그, 강등 로그)"""
    changes: list[str] = []
    demotions: list[str] = []

    for row in rows:
        meta = row["metadata"]
        if meta.get("gold_source") != "snapshot":
            continue
        item_id = row["id"]
        if item_id not in HANDLERS:
            raise SystemExit(f"{item_id}: snapshot인데 갱신 규칙이 없다")

        db = DB(conn, meta["as_of"])
        result = HANDLERS[item_id](db)

        if result.demote_reason:
            meta["gold_source"] = "domain_expectation"
            meta.pop("as_of", None)
            row["gold"]["expected_values"] = {}
            demotions.append(f"{item_id}: {result.demote_reason}")
            continue

        gold = row["gold"]
        if result.answer and gold.get("answer") != result.answer:
            changes.append(f"{item_id} answer\n    - {gold.get('answer')}\n    + {result.answer}")
            gold["answer"] = result.answer
        if gold.get("expected_values") != result.values:
            changes.append(
                f"{item_id} expected_values\n    - {gold.get('expected_values')}\n"
                f"    + {result.values}"
            )
            gold["expected_values"] = result.values

    return rows, changes, demotions


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="변경 전후 diff만 출력")
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--db", type=Path, default=DB_PATH)
    args = parser.parse_args()

    if not args.db.exists():
        print(f"DB를 찾을 수 없다: {args.db}")
        return 1

    rows = [
        json.loads(line)
        for line in args.dataset.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    conn = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        rows, changes, demotions = run(rows, conn)
    finally:
        conn.close()

    print(f"\n변경 {len(changes)}건, 강등 {len(demotions)}건")
    if demotions:
        print("\n--- domain_expectation으로 강등 (원자료 부재) ---")
        for line in demotions:
            print(f"  {line}")
    if changes:
        print("\n--- 변경 전후 ---")
        for line in changes:
            print(f"  {line}")

    if args.dry_run:
        print("\n[dry-run] 파일은 그대로")
        return 0

    with args.dataset.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"\n기록: {args.dataset}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
