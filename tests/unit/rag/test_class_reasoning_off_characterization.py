"""플래그 OFF 특성화: ``ontology.use_class_reasoning``이 꺼져 있으면 출력이 O3 이전과 같다.

스냅샷(``fixtures/o3_flag_off_snapshot.json``)은 O3 코드 변경 **전** 커밋(``0b56e4b``)에서
``ontology_query_fixtures.QUERIES`` 다섯 질의(그룹·세그먼트·원산지·포함·부정)로 만든
``HybridContext`` 전체(엔티티·KG 사실·DB 사실·증거 카드·프롬프트 카드·렌더 문자열·메타데이터)다.
실행 시간만 뺐다. 이 테스트가 깨지면 OFF 동작이 바뀐 것이다(OE7, 결정 OA-6).
"""

from __future__ import annotations

import json

import pytest

from .ontology_query_fixtures import QUERIES, SNAPSHOT_PATH, make_ontology_retriever, snapshot_of


@pytest.fixture(scope="module")
def expected() -> dict:
    return json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8"))


@pytest.mark.parametrize("name", sorted(QUERIES))
@pytest.mark.parametrize("env_value", [None, "false"])
async def test_flag_off_output_matches_pre_o3_snapshot(
    tmp_path, monkeypatch, expected, name, env_value
):
    if env_value is None:
        monkeypatch.delenv("FF_ONTOLOGY_USE_CLASS_REASONING", raising=False)
    else:
        monkeypatch.setenv("FF_ONTOLOGY_USE_CLASS_REASONING", env_value)
    retriever = make_ontology_retriever(tmp_path)

    ctx = await retriever.retrieve(QUERIES[name])

    assert snapshot_of(ctx) == expected[name]
    assert "ontology" not in ctx.metadata
