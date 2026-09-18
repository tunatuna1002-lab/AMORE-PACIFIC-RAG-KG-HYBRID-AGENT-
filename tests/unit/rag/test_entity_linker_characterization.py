"""EntityLinker 특성화 테스트 — 플래그 OFF 출력 고정 (트랙 O2, 결정 OA-6)

`ontology.use_class_reasoning`이 꺼져 있으면 ``extract_entities``·``link``의 출력은 O2 이전
코드(0b56e4b)와 바이트 단위로 같아야 한다. 기대값은 O2 코드를 넣기 **전** 코드로 기록했다
(`tests/unit/rag/fixtures/entity_linker_off_snapshot.json`).

질의: 골든 typed multihop·relation 전 문항 + rule 일부 + 경계 사례(짧은 별칭, 그룹 표현,
미인식 브랜드). 다시 기록할 일은 없어야 한다 — 기록하려면 O2 이전 코드에서:

    .venv/bin/python -m tests.unit.rag.test_entity_linker_characterization --regen
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from src.infrastructure.feature_flags import FeatureFlags
from src.rag.entity_linker import EntityLinker

_REPO_ROOT = Path(__file__).resolve().parents[3]
SNAPSHOT_PATH = Path(__file__).parent / "fixtures" / "entity_linker_off_snapshot.json"
GOLDEN_DIR = _REPO_ROOT / "eval" / "data" / "golden" / "typed"
FLAG_ENV = "FF_ONTOLOGY_USE_CLASS_REASONING"

RULE_IDS = (
    "lg114",
    "rg001",
    "rg005",
    "rg007",
    "rg020",
    "rg021",
    "rg022",
    "rg026",
    "rg027",
    "rg028",
    "rg030",
    "rg031",
    "rg032",
)

EXTRA_QUERIES = (
    "share of shelf 기준으로 shelf 전체를 보면?",
    "e.l.f. 와 ELF, elf 비교",
    "IT Cosmetics와 Jouer, Almay, Charlotte Tilbury, COVERGIRL 가격 비교",
    "IOPE, primera, 설화수, 한율 매출은?",
    "아모레퍼시픽 브랜드들의 Lip Care 순위는?",
    "Amorepacific brands in Skin Care",
    "Amore Pacific 라인과 AP 그룹 비교",
    "K-Beauty 브랜드 중 SoS 1위는?",
    "K뷰티 한국 브랜드 트렌드",
    "럭셔리 브랜드와 프리미엄 브랜드, 매스 브랜드 비교",
    "중저가 브랜드의 Lip Makeup 점유율",
    "LANEIGE와 같은 그룹의 자매 브랜드는?",
    "chi와 ryo, boj, eos, opi, verb, matrix 언급",
    "fresh unknown 브랜드 순위",
    "Lip Sleeping Mask 3위 제품 순위는? rank 3",
    "고려해야 할 어려운 점은 로드맵에 있다",
    "La Roche-Posay vs la roche posay vs LRP",
    "Burt's Bees, L'Oreal, Mise-en-scène, B.Ready",
    "B0LIPXYZ01 제품 리뷰 텍스처 향 가성비",
    "최근 30일 HHI 변동성 급변",
)


def _load_golden(name: str) -> list[dict[str, Any]]:
    path = GOLDEN_DIR / f"{name}.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def characterization_queries() -> list[str]:
    queries: list[str] = []
    for name in ("multihop", "relation"):
        queries.extend(row["question"] for row in _load_golden(name))
    rule_rows = {row["id"]: row for row in _load_golden("rule")}
    queries.extend(rule_rows[i]["question"] for i in RULE_IDS)
    queries.extend(EXTRA_QUERIES)
    # 순서 유지 중복 제거
    return list(dict.fromkeys(queries))


class FakeKG:
    """제품 슬러그 역링크·순위 기반 제품 추출 경로를 태우는 최소 KG."""

    def __init__(self) -> None:
        self._rels = [
            SimpleNamespace(
                subject="LANEIGE",
                predicate=SimpleNamespace(value="hasProduct"),
                object="B0LSM00001",
                properties={"title": "LANEIGE Lip Sleeping Mask - Berry", "category": "lip_care"},
            ),
            SimpleNamespace(
                subject="unknown",
                predicate=SimpleNamespace(value="hasProduct"),
                object="B0UNK00001",
                properties={"title": "Hydrating Lip Oil Serum Treatment", "category": "lip_care"},
            ),
            SimpleNamespace(
                subject="B0LSM00001",
                predicate=SimpleNamespace(value="rankedIn"),
                object="lip_care",
                properties={"rank": 3},
            ),
        ]

    def query(self, predicate: Any = None, object_: Any = None) -> list[Any]:
        if object_ is None:
            return list(self._rels)
        return [r for r in self._rels if r.object == object_]


FAKE_KG_QUERIES = (
    "Lip Sleeping Mask 3위 제품 순위는?",
    "lip care 3위 제품은 뭐야",
    "Hydrating Lip Oil 제품 리뷰",
)


def _fresh_linker() -> EntityLinker:
    EntityLinker._config_cache = None
    EntityLinker._config_loaded_at = None
    return EntityLinker(use_spacy=False)


def compute_snapshot() -> dict[str, Any]:
    linker = _fresh_linker()
    extract = {q: linker.extract_entities(q) for q in characterization_queries()}
    link = {q: [e.to_dict() for e in linker.link(q)] for q in characterization_queries()}
    with_kg = {
        q: _fresh_linker().extract_entities(q, knowledge_graph=FakeKG()) for q in FAKE_KG_QUERIES
    }
    return {"extract_entities": extract, "link": link, "extract_entities_with_kg": with_kg}


def _dump(data: Any) -> str:
    return json.dumps(data, ensure_ascii=False, sort_keys=True, indent=1)


@pytest.fixture(autouse=True)
def flag_off(monkeypatch):
    monkeypatch.setenv(FLAG_ENV, "false")
    FeatureFlags.reset_instance()
    EntityLinker._config_cache = None
    EntityLinker._config_loaded_at = None
    yield
    FeatureFlags.reset_instance()


@pytest.fixture(scope="module")
def snapshot() -> dict[str, Any]:
    return json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8"))


def test_snapshot_covers_enough_queries(snapshot):
    assert len(snapshot["extract_entities"]) >= 40
    assert set(snapshot["extract_entities"]) == set(characterization_queries())


@pytest.mark.parametrize("query", characterization_queries())
def test_extract_entities_flag_off_is_unchanged(snapshot, query):
    got = _fresh_linker().extract_entities(query)
    assert _dump(got) == _dump(snapshot["extract_entities"][query])


@pytest.mark.parametrize("query", characterization_queries())
def test_link_flag_off_is_unchanged(snapshot, query):
    got = [e.to_dict() for e in _fresh_linker().link(query)]
    assert _dump(got) == _dump(snapshot["link"][query])


@pytest.mark.parametrize("query", FAKE_KG_QUERIES)
def test_extract_entities_with_kg_flag_off_is_unchanged(snapshot, query):
    got = _fresh_linker().extract_entities(query, knowledge_graph=FakeKG())
    assert _dump(got) == _dump(snapshot["extract_entities_with_kg"][query])


def test_flag_unset_behaves_like_off(snapshot, monkeypatch):
    """env도 JSON 키도 없을 때 기본값은 OFF다 (다른 트랙이 JSON 키를 false로 넣어도 같다)."""
    monkeypatch.delenv(FLAG_ENV, raising=False)
    FeatureFlags.reset_instance()
    if FeatureFlags.get_instance().get_flag("ontology", "use_class_reasoning", default=False):
        pytest.skip("config/feature_flags.json turns the flag on")
    for query in list(snapshot["extract_entities"])[:10]:
        got = _fresh_linker().extract_entities(query)
        assert _dump(got) == _dump(snapshot["extract_entities"][query])


if __name__ == "__main__" and "--regen" in sys.argv:
    SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    SNAPSHOT_PATH.write_text(_dump(compute_snapshot()) + "\n", encoding="utf-8")
    print(f"wrote {SNAPSHOT_PATH}")
