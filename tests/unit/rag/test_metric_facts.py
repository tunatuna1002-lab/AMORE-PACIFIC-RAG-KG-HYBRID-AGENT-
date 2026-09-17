"""크롤 DB 수치 사실 제공자 (사이클 10, docs/experiments/eval_cycle10_2026-09-12.md §2)

챗봇이 DB에 있는 SoS·HHI·순위·리뷰 수를 "데이터에 없다"고 답했다. 수치를 SQLite에서
스냅샷 날짜와 함께 가져와 컨텍스트에 싣는다. 여기서는 시점 고정, 결측 처리, 부재 단정의
조건, DB 부재 시 부작용 없음을 고정한다.
"""

import sqlite3

import pytest

from src.rag.metric_facts import AS_OF_ENV, MetricFactsProvider


@pytest.fixture
def db_path(tmp_path):
    path = tmp_path / "amore.db"
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE brand_metrics (snapshot_date TEXT, category_id TEXT, brand TEXT,
                                    sos REAL, product_count INTEGER, brand_avg_rank REAL,
                                    cpi REAL, avg_rating_gap REAL);
        CREATE TABLE market_metrics (snapshot_date TEXT, category_id TEXT, hhi REAL,
                                     churn_rate REAL, category_avg_price REAL,
                                     category_avg_rating REAL);
        CREATE TABLE raw_data (snapshot_date TEXT, category_id TEXT, rank INTEGER, brand TEXT,
                               product_name TEXT, price REAL, rating REAL, reviews_count INTEGER);
        """
    )
    conn.executemany(
        "INSERT INTO brand_metrics VALUES (?,?,?,?,?,?,?,?)",
        [
            ("2026-08-31", "lip_care", "eos", 9.0, 9, 52.89, 80.0, 0.126),
            ("2026-08-31", "lip_care", "LANEIGE", 2.0, 2, 8.5, None, 0.076),
            ("2026-09-11", "lip_care", "eos", 8.0, 8, None, None, None),
            ("2026-09-11", "lip_care", "LANEIGE", 3.0, 3, None, None, None),
        ],
    )
    conn.executemany(
        "INSERT INTO market_metrics VALUES (?,?,?,?,?,?)",
        [
            ("2026-08-31", "lip_care", 0.0681, None, None, 4.57),
            ("2026-09-11", "lip_care", 0.0700, None, None, 4.6),
            ("2026-08-31", "skin_care", None, None, None, None),
        ],
    )
    conn.executemany(
        "INSERT INTO raw_data VALUES (?,?,?,?,?,?,?,?)",
        [
            ("2026-08-31", "lip_care", 1, "Burt's Bees", "Burt's Bees Lip Balm", 10.48, 4.7, 90000),
            (
                "2026-08-31",
                "lip_care",
                7,
                "LANEIGE",
                "LANEIGE Lip Sleeping Mask: Korean",
                21.6,
                4.6,
                37356,
            ),
            (
                "2026-09-11",
                "lip_care",
                8,
                "LANEIGE",
                "LANEIGE Lip Sleeping Mask: Korean",
                22.0,
                4.6,
                37435,
            ),
        ],
    )
    conn.commit()
    conn.close()
    return path


def _by_type(facts, kind):
    return [f for f in facts if f["type"] == kind]


async def test_as_of_pins_every_table_to_that_snapshot(db_path):
    facts = await MetricFactsProvider(db_path, as_of="2026-08-31").collect(
        {"brands": ["laneige"], "categories": ["lip_care"]}
    )

    assert {f["snapshot_date"] for f in facts} == {"2026-08-31"}
    share = _by_type(facts, "brand_share")[0]
    assert share["present"] is True and share["sos"] == 2.0 and share["brand_rank"] == 2
    assert _by_type(facts, "category_market")[0]["hhi"] == 0.0681
    product = _by_type(facts, "brand_products")[0]["products"][0]
    assert product == {
        "rank": 7,
        "brand": "LANEIGE",
        "name": "LANEIGE Lip Sleeping Mask",
        "price": 21.6,
        "rating": 4.6,
        "reviews_count": 37356,
    }


async def test_query_brand_share_carries_rank_cpi_and_rating_gap_only_when_present(db_path):
    # 규칙 입력(평균 순위·CPI·평점 격차)은 질의 브랜드에만, NULL은 키 자체를 넣지 않는다
    facts = await MetricFactsProvider(db_path, as_of="2026-08-31").collect(
        {"brands": ["laneige"], "categories": ["lip_care"]}
    )

    (share,) = _by_type(facts, "brand_share")
    assert share["brand_avg_rank"] == 8.5
    assert share["avg_rating_gap"] == 0.076
    assert "cpi" not in share
    top = _by_type(facts, "category_top_brands")[0]["brands"]
    assert all(set(entry) == {"brand", "sos", "product_count"} for entry in top)

    later = await MetricFactsProvider(db_path, as_of="2026-09-11").collect(
        {"brands": ["laneige"], "categories": ["lip_care"]}
    )
    (later_share,) = _by_type(later, "brand_share")
    assert not {"brand_avg_rank", "cpi", "avg_rating_gap"} & set(later_share)


async def test_without_as_of_latest_snapshot_is_used(db_path, monkeypatch):
    monkeypatch.delenv(AS_OF_ENV, raising=False)

    facts = await MetricFactsProvider(db_path).collect({"categories": ["lip_care"]})

    assert _by_type(facts, "category_market")[0]["hhi"] == 0.0700


async def test_env_var_sets_as_of(db_path, monkeypatch):
    monkeypatch.setenv(AS_OF_ENV, "2026-08-31")

    facts = await MetricFactsProvider(db_path).collect({"categories": ["lip_care"]})

    assert _by_type(facts, "category_market")[0]["snapshot_date"] == "2026-08-31"


async def test_null_market_fields_are_omitted_not_zeroed(db_path):
    facts = await MetricFactsProvider(db_path, as_of="2026-08-31").collect(
        {"categories": ["lip_care", "skin_care"]}
    )

    market = _by_type(facts, "category_market")
    assert [m["category"] for m in market] == ["lip_care"]  # skin_care는 전부 NULL
    assert "churn_rate" not in market[0] and "category_avg_price" not in market[0]


async def test_absence_is_stated_only_where_the_category_has_share_data(db_path):
    facts = await MetricFactsProvider(db_path, as_of="2026-08-31").collect(
        {"brands": ["tirtir"], "categories": ["lip_care", "skin_care"]}
    )

    shares = _by_type(facts, "brand_share")
    assert shares == [
        {
            "type": "brand_share",
            "brand": "tirtir",
            "category": "lip_care",
            "snapshot_date": "2026-08-31",
            "present": False,
        }
    ]  # skin_care에는 그날 점유율 데이터가 없으므로 부재도 단정하지 않는다


async def test_brand_only_query_uses_categories_the_brand_is_in(db_path):
    facts = await MetricFactsProvider(db_path, as_of="2026-08-31").collect({"brands": ["LANEIGE"]})

    assert {f["category"] for f in facts} == {"lip_care"}


async def test_no_entities_means_no_query(tmp_path):
    missing = tmp_path / "nope.db"

    assert await MetricFactsProvider(missing).collect({}) == []


async def test_missing_db_is_not_created(tmp_path):
    missing = tmp_path / "nope.db"

    facts = await MetricFactsProvider(missing).collect({"categories": ["lip_care"]})

    assert facts == []
    assert not missing.exists()
