"""유형별 시험지 분류(1-A)와 보충 문항 생성기(1-B) 테스트.

- 분류: 결정성(디스크 파일 = 재계산), 게이트(유형별 20문항 이상), 원본 레코드 무변경, 로더 호환.
- 생성기: 픽스처 SQLite·KG(실제 파일, mock 없음)에서 결정성, 독립 계산 공식 = 운영 계산
  (`build_metric_rows`), 규칙 엔진 재판정 일치, 오귀속 브랜드 제외, 로더 호환.
- 실데이터 재생성 비교는 원본 DB가 있을 때만 돈다(CI에는 data/가 없다).
"""

from __future__ import annotations

import importlib.util
import json
import sqlite3
import sys
from pathlib import Path

import pytest

from eval.loader import load_dataset
from eval.metrics.l5_answer import numeric_accuracy
from eval.schemas import EvalItem

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = PROJECT_ROOT / "scripts"
TYPED_DIR = PROJECT_ROOT / "eval" / "data" / "golden" / "typed"


def _load(name: str):
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location(f"{name}_under_test", SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclass가 모듈을 sys.modules에서 찾는다
    spec.loader.exec_module(module)
    return module


classify = _load("classify_golden_types")
rule_gen = _load("generate_rule_questions")
gf = rule_gen.gf  # 생성기와 같은 모듈 인스턴스
mh_gen = _load("generate_multihop_questions")
rel_gen = _load("generate_relation_questions")


# =============================================================================
# 1-A 분류
# =============================================================================


def _source_items() -> list[dict]:
    return classify.read_jsonl(classify.DATASET)


class TestClassification:
    def test_committed_files_equal_recomputation(self):
        """--check와 같은 비교: 분류 규칙·생성 문항이 바뀌면 파일도 다시 만들어야 한다."""
        outputs = classify.build_outputs(_source_items(), classify.load_generated(TYPED_DIR))
        for name, content in outputs.items():
            assert (TYPED_DIR / name).read_text(encoding="utf-8") == content, name

    def test_classification_is_deterministic(self):
        items = _source_items()
        generated = classify.load_generated(TYPED_DIR)
        assert classify.build_outputs(items, generated) == classify.build_outputs(items, generated)

    @pytest.mark.parametrize("qtype", classify.TYPES)
    def test_each_sheet_meets_gate_and_loads(self, qtype):
        path = TYPED_DIR / f"{qtype}.jsonl"
        lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        items = load_dataset(path)
        assert len(items) == len(lines) >= 20  # 로더가 한 줄도 버리지 않았다
        assert {r["metadata"]["question_type"] for r in lines} == {qtype}
        assert len({r["id"] for r in lines}) == len(lines)

    def test_original_records_copied_unchanged(self):
        source = {r["id"]: r for r in _source_items()}
        for qtype in classify.TYPES:
            for line in (TYPED_DIR / f"{qtype}.jsonl").read_text(encoding="utf-8").splitlines():
                record = json.loads(line)
                if not record["id"].startswith("lg"):
                    continue
                assert record["metadata"].pop("question_type") == qtype
                assert record == source[record["id"]]

    def test_every_source_item_classified_once(self):
        rows = [
            json.loads(line)
            for line in (TYPED_DIR / "classification.jsonl").read_text().splitlines()
        ]
        original = [r["id"] for r in rows if r["id"].startswith("lg")]
        assert sorted(original) == sorted(r["id"] for r in _source_items())
        assert all(r["reasons"] for r in rows)

    @pytest.mark.parametrize(
        ("item_id", "expected_type", "subkind"),
        [
            ("lg041", "other", "definition"),  # SoS란?
            ("lg048", "numeric", None),  # LANEIGE Lip Care SoS
            ("lg103", "relation", None),  # 경쟁 관계
            ("lg045", "rule", None),  # HHI 0.15는 어떤 시장 구조
            ("lg194", "rule", None),  # 성장 가능성 (category_entry_opportunity)
            ("lg150", "multihop", None),  # 대표 제품이 속한 카테고리의 1위
            ("lg166", "other", "out_of_scope"),
            ("lg201", "other", "ir_document"),
        ],
    )
    def test_known_examples(self, item_id, expected_type, subkind):
        item = next(r for r in _source_items() if r["id"] == item_id)
        row = classify.classify_item(item)
        assert (row["type"], row["subkind"]) == (expected_type, subkind)

    def test_domain_expectation_is_held_out(self):
        item = next(r for r in _source_items() if r["id"] == "lg175")  # 지난 3개월 SoS 추이
        row = classify.classify_item(item)
        assert row["type"] == "numeric"
        assert row["in_sheet"] is False and row["holdout_reason"]

    def test_loader_ignores_generator_metadata_keys(self):
        record = json.loads((TYPED_DIR / "generated_rule.jsonl").read_text().splitlines()[0])
        assert "rule_gold" in record["metadata"]
        item = EvalItem.model_validate(record)
        assert item.id == record["id"]
        assert item.metadata.gold_source == record["metadata"]["gold_source"]


# =============================================================================
# 1-B 생성기 — 픽스처
# =============================================================================

AS_OF = gf.AS_OF

SCHEMA = """
CREATE TABLE raw_data (
    id INTEGER PRIMARY KEY AUTOINCREMENT, snapshot_date TEXT NOT NULL, category_id TEXT NOT NULL,
    rank INTEGER NOT NULL, asin TEXT NOT NULL, product_name TEXT, brand TEXT, price REAL,
    rating REAL, reviews_count INTEGER, UNIQUE(snapshot_date, category_id, rank));
CREATE TABLE brand_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT, snapshot_date TEXT NOT NULL, category_id TEXT NOT NULL,
    brand TEXT NOT NULL, sos REAL, brand_avg_rank REAL, product_count INTEGER, cpi REAL,
    avg_rating_gap REAL, UNIQUE(snapshot_date, category_id, brand));
CREATE TABLE market_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT, snapshot_date TEXT NOT NULL, category_id TEXT NOT NULL,
    hhi REAL, churn_rate REAL, category_avg_price REAL, category_avg_rating REAL,
    UNIQUE(snapshot_date, category_id));
"""

# 카테고리별 (브랜드, 제품 수, 제품명 접두어). 순서대로 순위를 채운다.
FIXTURE_BRANDS = {
    "lip_care": [
        ("BigBalm", 20, "BigBalm"),
        ("LANEIGE", 2, "LANEIGE"),
        ("Hera", 3, "Vaseline Lip Therapy"),  # 오귀속: 제품명에 Hera 없음
        ("Unknown", 15, "Generic"),
        *[(f"Brand{i}", 4, f"Brand{i}") for i in range(15)],
    ],
    "face_powder": [
        ("PowderCo", 12, "PowderCo"),
        ("LANEIGE", 1, "LANEIGE"),
        ("innisfree", 1, "innisfree"),
        ("CHI", 2, "KimChiChic"),  # 오귀속: 부분 문자열 "chi"
        ("Unknown", 24, "Generic"),
        *[(f"Face{i}", 3, f"Face{i}") for i in range(20)],
    ],
}


def _rows() -> list[dict]:
    rows = []
    for category, brands in FIXTURE_BRANDS.items():
        rank = 1
        for brand, count, prefix in brands:
            for _ in range(count):
                rows.append(
                    {
                        "snapshot_date": AS_OF,
                        "category_id": category,
                        "rank": rank,
                        "asin": f"{category[:2].upper()}{rank:04d}",
                        "product_name": f"{prefix} Product {rank}",
                        "brand": brand,
                        "price": 30.0 if brand == "LANEIGE" else 8.0 + rank % 9,
                        "rating": 4.0 + (rank % 10) / 10,
                        "reviews_count": 100 * rank,
                    }
                )
                rank += 1
        assert rank == 101, category
    return rows


@pytest.fixture
def fixture_db(tmp_path: Path) -> Path:
    from src.tools.calculators.metric_snapshot import build_metric_rows

    db = tmp_path / "fixture.db"
    rows = _rows()
    brand_rows, market_rows = build_metric_rows(rows)  # 운영 계산으로 지표 테이블을 채운다
    with sqlite3.connect(db) as conn:
        conn.executescript(SCHEMA)
        conn.executemany(
            "INSERT INTO raw_data (snapshot_date, category_id, rank, asin, product_name, brand,"
            " price, rating, reviews_count) VALUES (:snapshot_date, :category_id, :rank, :asin,"
            " :product_name, :brand, :price, :rating, :reviews_count)",
            rows,
        )
        conn.executemany(
            "INSERT INTO brand_metrics (snapshot_date, category_id, brand, sos, brand_avg_rank,"
            " product_count, cpi, avg_rating_gap) VALUES (:snapshot_date, :category_id, :brand,"
            " :sos, :brand_avg_rank, :product_count, :cpi, :avg_rating_gap)",
            brand_rows,
        )
        conn.executemany(
            "INSERT INTO market_metrics (snapshot_date, category_id, hhi, churn_rate,"
            " category_avg_price, category_avg_rating) VALUES (:snapshot_date, :category_id,"
            " :hhi, :churn_rate, :category_avg_price, :category_avg_rating)",
            market_rows,
        )
        # 지표 테이블을 raw_data와 일부러 어긋나게 한다(실데이터의 08-30 크롤 기준 테이블 재현):
        # (1) 수치만 다르고 결론은 같은 경우, (2) 결론이 뒤집히는 경우, (3) 테이블 값이 없는 경우
        conn.execute(
            "UPDATE market_metrics SET hhi = ROUND(hhi * 1.05, 4) WHERE category_id = 'lip_care'"
        )
        conn.execute(
            "UPDATE brand_metrics SET sos = 3.5 WHERE category_id = 'lip_care' AND brand = 'LANEIGE'"
        )
        conn.execute(
            "UPDATE brand_metrics SET cpi = NULL WHERE category_id = 'face_powder' AND brand = 'Face0'"
        )
    return db


def _table_value(db: Path, sql: str) -> float:
    with sqlite3.connect(db) as conn:
        return conn.execute(sql).fetchone()[0]


@pytest.fixture
def fixture_kg(tmp_path: Path) -> Path:
    def triple(s, p, o, source="config/brands.json"):
        return {"subject": s, "predicate": p, "object": o, "properties": {}, "source": source}

    brands = ["LANEIGE", "COSRX", "innisfree", "Sulwhasoo", "HERA"]
    triples = [triple(b, "ownedByGroup", "AMOREPACIFIC") for b in brands]
    triples += [triple("AMOREPACIFIC", "ownsBrand", b) for b in brands]
    triples += [triple("LANEIGE", "siblingBrand", b) for b in brands if b != "LANEIGE"]
    triples.append(triple("LANEIGE", "competesWith", "BigBalm", source="kg_enricher"))
    path = tmp_path / "kg.json"
    path.write_text(json.dumps({"triples": triples}), encoding="utf-8")
    return path


class TestIndependentFacts:
    def test_raw_sql_formulas_match_production_metric_rows(self, fixture_db):
        """독립 SQL 계산 = 운영 build_metric_rows. 공식을 잘못 옮겼으면 여기서 깨진다."""
        from src.tools.calculators.metric_snapshot import build_metric_rows

        brand_rows, market_rows = build_metric_rows(_rows())
        conn = gf.connect_ro(fixture_db)
        for market in market_rows:
            facts = gf.category_facts(conn, market["category_id"])
            assert facts.hhi == market["hhi"]
            assert round(facts.avg_price, 2) == market["category_avg_price"]
            for row in (r for r in brand_rows if r["category_id"] == market["category_id"]):
                b = facts.brands[row["brand"]]
                assert (b.sos, b.avg_rank, b.count, b.cpi, b.rating_gap) == (
                    row["sos"],
                    row["brand_avg_rank"],
                    row["product_count"],
                    row["cpi"],
                    row["avg_rating_gap"],
                )

    def test_read_only_connection_cannot_write(self, fixture_db):
        conn = gf.connect_ro(fixture_db)
        with pytest.raises(sqlite3.OperationalError):
            conn.execute("DELETE FROM raw_data")

    def test_attribution_failure_detected(self, fixture_db):
        facts = gf.category_facts(gf.connect_ro(fixture_db), "lip_care")
        assert set(facts.attribution_failures()) == {"Hera"}
        powder = gf.category_facts(gf.connect_ro(fixture_db), "face_powder")
        assert set(powder.attribution_failures()) == {"CHI"}

    def test_only_curated_kg_triples_loaded(self, fixture_kg):
        triples = gf.load_curated_triples(fixture_kg)
        assert triples and all(t["source"] == "config/brands.json" for t in triples)


class TestRuleGenerator:
    def test_rule_inputs_table_covers_all_rules(self):
        from src.ontology.rules import ALL_BUSINESS_RULES

        names = {r.name for r in ALL_BUSINESS_RULES}
        assert len(names) == 37
        assert set(rule_gen.RULE_INPUTS) == names

    def test_deterministic_and_loadable(self, fixture_db, fixture_kg, tmp_path):
        first, log1 = rule_gen.build(fixture_db, fixture_kg)
        second, log2 = rule_gen.build(fixture_db, fixture_kg)
        assert gf.dumps_jsonl(first) == gf.dumps_jsonl(second)
        assert gf.dumps_json(log1) == gf.dumps_json(log2)
        assert first

        out = tmp_path / "generated_rule.jsonl"
        out.write_text(gf.dumps_jsonl(first), encoding="utf-8")
        assert len(load_dataset(out)) == len(first)

    def test_conclusions_reproduced_by_rule_engine(self, fixture_db, fixture_kg):
        rules = rule_gen.rules_by_name()
        records, _ = rule_gen.build(fixture_db, fixture_kg)
        for record in records:
            gold = record["metadata"]["rule_gold"]
            ctx = gold["rule_context"]
            if gold["condition"]:
                rule = rules[gold["rule_ids"][0]]
                cond = next(c for c in rule.conditions if c.name == gold["condition"])
                fires = cond.evaluate(ctx)
            else:
                fires = all(rules[r].evaluate_conditions(ctx)[0] for r in gold["rule_ids"])
            assert fires == gold["expected_conclusion"]["fires"], record["id"]
            if record["gold"]["expected_values"]:
                assert (
                    numeric_accuracy(record["gold"]["answer"], record["gold"]["expected_values"])
                    == 1.0
                )

    def test_sos_scale_conversion_recorded(self, fixture_db, fixture_kg):
        """BigBalm SoS 20%·HHI < 0.15 → 규칙 의도(0~1)로는 분산 시장 지배, 변환 없이도 참."""
        records, _ = rule_gen.build(fixture_db, fixture_kg)
        dominance = [
            r
            for r in records
            if r["metadata"]["rule_gold"]["rule_ids"] == ["market_dominance_fragmented"]
        ]
        assert dominance
        for r in dominance:
            gold = r["metadata"]["rule_gold"]
            assert gold["rule_context"]["sos"] == pytest.approx(gold["inputs"]["sos_ratio"])
            assert gold["inputs"]["sos_ratio"] == pytest.approx(
                r["gold"]["expected_values"]["sos"] / 100
            )
            assert gold["units"]["sos_ratio"].startswith("0~1")
            assert "conclusion_if_sos_unconverted" in gold

    def test_gold_is_metric_table_value_raw_kept_as_cross_check(self, fixture_db, fixture_kg):
        records, _ = rule_gen.build(fixture_db, fixture_kg)
        table_hhi = _table_value(
            fixture_db, "SELECT hhi FROM market_metrics WHERE category_id = 'lip_care'"
        )
        raw_hhi = gf.category_facts(gf.connect_ro(fixture_db), "lip_care").hhi
        assert table_hhi != raw_hhi
        rec = next(
            r
            for r in records
            if r["metadata"]["rule_gold"]["condition"] == "hhi_below_0.15"
            and "lip_care" in r["gold"]["kg_entities"]
        )
        assert rec["gold"]["expected_values"] == {"hhi": table_hhi}
        assert f"{table_hhi:.4f}" in rec["gold"]["answer"]
        check = rec["metadata"]["rule_gold"]["cross_check"]
        assert check["raw_recomputed"] == {"hhi_ratio": raw_hhi}
        assert check["conclusion_agrees"] is True
        assert check["relative_diff"]["hhi_ratio"] == pytest.approx(
            abs(raw_hhi - table_hhi) / table_hhi, abs=1e-4
        )

    def test_conclusion_disagreement_excluded(self, fixture_db, fixture_kg):
        """LANEIGE lip_care SoS: 테이블 3.5%(기회 아님) vs raw 2%(기회) → 싣지 않는다."""
        records, log = rule_gen.build(fixture_db, fixture_kg)
        assert not any(
            r["metadata"]["rule_gold"]["rule_ids"] == ["category_entry_opportunity"]
            and "lip_care" in r["gold"]["kg_entities"]
            for r in records
        )
        excluded = [
            e
            for e in log["excluded"]
            if e["kind"] == "F_category_entry_opportunity" and e["category"] == "lip_care"
        ]
        assert excluded and "결론 불일치" in excluded[0]["reason"]

    def test_missing_table_value_excluded_and_system_access_marked(self, fixture_db, fixture_kg):
        records, log = rule_gen.build(fixture_db, fixture_kg)
        face0 = [e for e in log["excluded"] if e["subject"] == "Face0"]
        assert face0 and all("정답 값 없음" in e["reason"] for e in face0)
        for r in records:
            if "cpi" in r["gold"]["expected_values"]:
                assert r["metadata"]["system_access"] == "not_in_metric_facts"

    def test_misattributed_brand_never_a_subject(self, fixture_db, fixture_kg):
        records, log = rule_gen.build(fixture_db, fixture_kg)
        assert all("chi" not in r["gold"]["kg_entities"] for r in records)
        chi = [e for e in log["excluded"] if e["subject"] == "CHI"]
        assert chi and all("브랜드 귀속 검증 실패" in e["reason"] for e in chi)


class TestMultihopAndRelationGenerators:
    def test_multihop_deterministic_and_loadable(self, fixture_db, fixture_kg, tmp_path):
        first, _ = mh_gen.build(fixture_db, fixture_kg)
        second, _ = mh_gen.build(fixture_db, fixture_kg)
        assert gf.dumps_jsonl(first) == gf.dumps_jsonl(second)
        assert first
        out = tmp_path / "generated_multihop.jsonl"
        out.write_text(gf.dumps_jsonl(first), encoding="utf-8")
        assert len(load_dataset(out)) == len(first)
        for record in first:
            assert record["metadata"]["hop_plan"]
            assert (
                numeric_accuracy(record["gold"]["answer"], record["gold"]["expected_values"]) == 1.0
            )

    def test_multihop_skips_group_match_with_misattribution(self, fixture_db, fixture_kg):
        """lip_care 'Hera'(제품명 Vaseline)가 KG HERA와 매칭되면 그 카테고리 문항은 만들지 않는다."""
        _, log = mh_gen.build(fixture_db, fixture_kg)
        reasons = {e["candidate"]: e["reason"] for e in log["excluded"]}
        assert "오귀속" in reasons["group_brands_sos/lip_care"]

    def test_relation_deterministic_and_loadable(self, fixture_kg, tmp_path):
        first, _ = rel_gen.build(fixture_kg)
        second, _ = rel_gen.build(fixture_kg)
        assert gf.dumps_jsonl(first) == gf.dumps_jsonl(second)
        assert first
        out = tmp_path / "generated_relation.jsonl"
        out.write_text(gf.dumps_jsonl(first), encoding="utf-8")
        assert len(load_dataset(out)) == len(first)
        assert all(r["gold"]["kg_edges"] or r["gold"]["expected_values"] for r in first)


# =============================================================================
# 실데이터 재생성 비교 (원본 DB·KG가 있을 때만)
# =============================================================================

REAL_DATA = gf.DEFAULT_DB.exists() and gf.DEFAULT_KG.exists()


@pytest.mark.skipif(not REAL_DATA, reason="data/amore_data.db·knowledge_graph.json 없음")
class TestCommittedGeneratedFiles:
    """DB 2026-08-31 스냅샷이나 config/brands.json이 바뀌면 깨진다 — 생성기를 다시 돌리고 diff를 검토할 것."""

    def test_rule_file_matches_regeneration(self):
        records, log = rule_gen.build(gf.DEFAULT_DB, gf.DEFAULT_KG)
        assert (TYPED_DIR / "generated_rule.jsonl").read_text() == gf.dumps_jsonl(records)
        assert (TYPED_DIR / "generation_log_rule.json").read_text() == gf.dumps_json(log)

    def test_multihop_file_matches_regeneration(self):
        records, _ = mh_gen.build(gf.DEFAULT_DB, gf.DEFAULT_KG)
        assert (TYPED_DIR / "generated_multihop.jsonl").read_text() == gf.dumps_jsonl(records)

    def test_relation_file_matches_regeneration(self):
        records, _ = rel_gen.build(gf.DEFAULT_KG)
        assert (TYPED_DIR / "generated_relation.jsonl").read_text() == gf.dumps_jsonl(records)
