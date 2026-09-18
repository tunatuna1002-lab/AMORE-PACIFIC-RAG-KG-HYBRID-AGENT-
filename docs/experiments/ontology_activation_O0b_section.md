# §O0-추가 (2026-09-18, O0 이어가기)

> `docs/experiments/ontology_activation_2026-09.md`의 §O0(O0-A·O0-B) 뒤에 이어 붙일 절 초안이다.
> 리드가 검토 후 그 문서로 옮긴다. 계획: [`docs/plans/ontology-activation-plan-2026-09-18.md`](../plans/ontology-activation-plan-2026-09-18.md).
> API 비용 $0 (LLM·임베딩 호출 0건). worktree 기준 `git log -1`을 `e0ffac7`로 맞추고 시작했다.

## 무엇을 왜 고쳤나

O3(`f4e995d` 플래그 `ontology.use_class_reasoning`)가 `hybrid_retriever.py`의 `priority_preds`를
모듈 상수 하나에서 `_PRIORITY_PREDS_CANONICAL if canonical else _PRIORITY_PREDS`(지역 변수, 조건식)로
바꾸면서 O0-B가 ast로 읽던 `retriever_predicate_filters`가 빈 집합을 돌려주게 됐다
(`tests/unit/scripts/test_ontology_offline_checks.py::test_real_retriever_filters_are_found` 실패).
또한 O3가 정적 정의 술어(그룹·세그먼트·원산지·인수·자매)를 `metric_edges`가 아니라 새 fact
타입 `ontology_static`(`src/rag/ontology_context.py:static_fact`)으로 내면서, 평가 L3 트레이스
추출(`eval/runner.py:_extract_l3_trace`)이 `metric_edges`만 보고 있어 그 카드들이 통째로
안 잡히고 있었다.

## 1. `scripts/ontology_offline_checks.py` 감지 수리 (플래그 OFF/ON 분리)

`retriever_predicate_filters`가 이제 `_PRIORITY_PREDS`·`_PRIORITY_PREDS_CANONICAL`(둘 다 모듈
상수라 ast로 잡힌다)과 `ontology_context.py`의 `STATIC_BRAND_PREDICATES`를 따로 읽어 `off`/`on`
키로 나눠 낸다. 최상위 `priority_preds`·`runtime_emitted_predicates`는 하위 호환을 위해 OFF 값을
그대로 유지한다. `run_checks`도 `gold_edge_reachability`(OFF)와 `gold_edge_reachability_on`(ON)을
각각 계산하고, `render_markdown`이 "2a. 플래그 OFF"·"2b. 플래그 ON" 두 절로 나눠 출력한다.

**부수로 잡은 버그**: `gold_edge_reachability` 안에 "검색기가 KG의 ownedByGroup을 ownedBy로 바꿔
방출한다"는 가정이 하드코딩돼 있었다. OFF에서는 맞는 가정이지만(OFF는 실제로 `ownedBy`만 낸다),
그대로 두면 ON(정식 이름 `ownedByGroup`을 그대로 낸다)에서 `ownedByGroup`이 emitted_names에서
통째로 사라져 "런타임 이름 방출"이 항상 0으로 나왔다. 호출자가 넘기는 `runtime_emitted_predicates`가
이미 모드별 실제 방출 이름이므로 이 하드코딩을 지우고 그대로 썼다 — 이제 모드별 값이 정확하다.

**재현**(태그 스냅샷 KG, 골든 233문항):

```bash
.venv/bin/python scripts/ontology_offline_checks.py \
  --kg $ROOT/eval_output/evidence-2026-09/eval_data_snapshot_tagged/knowledge_graph.json
```

| 술어 | 골드 | OFF 지금 도달 가능 | ON 지금 도달 가능 |
|---|---|---|---|
| ownedByGroup | 29 | 0 | **29** |
| hasSegment | 14 | 0 | **14** |
| originatesFrom | 2 | 0 | **2** |
| siblingBrand | 2 | 0 | **2** |
| acquiredIn | 1 | 0 | **1** |
| ownedBy(별칭 표기) | 11 | 11 | 0 (ON은 이 이름을 안 씀) |
| hasSoS·rankedIn·competesWith·hasProduct·belongsToCategory | — | 변화 없음 | 변화 없음 |

정적 정의 5개 술어(ownedByGroup·hasSegment·originatesFrom·siblingBrand·acquiredIn)는 KG에 사실이
있는 만큼(같은 이름 기준) ON에서 전부 "지금 도달 가능"으로 바뀐다 — `ontology_static` 카드가
KG 조회 경로 밖에서 이 술어들을 정식 이름 그대로 내기 때문이다(O3 결과, `docs/experiments/ontology_activation_2026-09.md` §O3 절 참고). `ownedBy`(레거시 별칭 표기)는 ON에서 아예 안 쓰이므로
0으로 바뀌는 게 맞다 — 골드가 `ownedBy`로 쓰였으면 오히려 ON에서 놓친다(별칭 문제, 아래 §3).

## 2. `eval/runner.py` L3 추출에 `ontology_static` 포함

`_extract_l3_trace`의 `elif fact_type == "metric_edges":` 분기를
`elif fact_type in ("metric_edges", "ontology_static"):`로 넓혔다. `ontology_static` 카드 중
`object`가 없는 것(`expansionTruncated` — 엣지가 아니라 "몇 개 잘랐다"는 집계 카드)은 건너뛴다.
플래그 OFF에서는 `ontology_static` fact 자체가 생기지 않으므로(O3, `hybrid_retriever.py`) 이
변경은 OFF 트레이스에 전혀 영향이 없다 — `tests/eval/test_runner_trace.py`의
`test_off_flag_trace_unaffected_no_ontology_static_fact`로 고정했다. `eval/brain_adapter.py`는
`ontology_facts`를 그대로 전달만 하므로 고칠 곳이 없었다(확인만 함).

## 3. 정식 술어(canonical) per-predicate recall 추가

`eval/metrics/l3_kg.py`에 `canonicalize_edge()`를 추가했다: 술어는
`src.ontology.ontology.get_ontology().canonical_predicate`(모르면 원래 이름), 끝점은
`normalize_brand` 우선 → 안 되면 `normalize_group` → 둘 다 안 되면 원문(소문자, 즉 항등)으로
맞춘다. **나라는 등록부에 정규화 함수가 없어 의도적으로 별칭 처리하지 않는다** —
`korea`와 `south_korea`는 canonical에서도 계속 다른 값이다. 온톨로지 로더를 못 불러오는 극단적
환경에서는 항등 함수로 대체해(`_load_ontology_normalizers`) canonical 값이 raw와 같아지도록
방어했다.

기존 raw 필드(`edge_recall_by_predicate` 등)는 그대로 두고, `L3Metrics`에
`*_canonical` 4필드(`kg_edge_recall_gold_only_canonical`·`gold_edge_count_canonical`·
`gold_edge_matched_canonical`·`edge_recall_by_predicate_canonical`)를 추가했다(additive).
`aggregate_l3_extended`와 `eval/report.py`의 `by_layer`(`l3_kg_edge_recall_gold_only_canonical`)에도
같은 원칙으로 더했다. `scripts/rescore_l3_l4.py`에 `KEY_PREDICATES_CANONICAL`(raw 목록에서
별칭으로 흡수되는 `ownedBy`를 뺀 것)과 canonical 표를 추가하고, 메인 표에 canonical
골드-엣지-문항 recall·micro 두 열을 더했다.

### 재채점 (재사용 기준선, 새 파일만 — 기존 `rescore/*.json`은 그대로 둠)

```bash
.venv/bin/python scripts/rescore_l3_l4.py \
  --base $ROOT/eval_output/evidence-2026-09 \
  --out $ROOT/eval_output/ontology-2026-09/rescore/canonical \
  --subset s3-rule42:s3=eval/data/golden/typed/rule.jsonl \
  s6a-run1 s6a-run2 s6a-run3 s5-run1 s3-run1 s3-run2 s3-run3 s3-roff-run1 s3-roff-run2 s3-roff-run3
```

**canonical per-predicate recall — mean (min~max) [골드 엣지 수]** (raw는 §O0-B 표 참고, 전부 태그
스냅샷·OFF 코드 경로 기준. rule on/off·s6a/s5 구성은 §O0-A와 동일)

| 구성 | ownedByGroup | hasSegment | originatesFrom | siblingBrand | acquiredIn | belongsToCategory |
|---|---|---|---|---|---|---|
| s6a (multihop+relation 54, 3회) | 0.424 [33] | 0.000 [14] | 0.000 [2] | 0.000 [2] | 0.000 [1] | 0.333 [12] |
| s5 (233, 1회) | 0.475 [40] | 0.000 [14] | 0.000 [2] | 0.000 [2] | 0.000 [1] | 0.375 [16] |
| s3 규칙 on (233, 3회) | 0.475 (0.475~0.475) [40] | 0.000 [14] | 0.000 [2] | 0.000 [2] | 0.000 [1] | 0.375 [16] |
| s3-roff / s3-rule42 (rule 42, 3회) | 1.000 [2] | — [0] | — [0] | — [0] | — [0] | — [0] |

- **`ownedByGroup`만 바뀐다.** raw 0.000 → canonical 0.424~0.475(s6a·s5·s3), 골드 엣지 수도
  29(raw, `ownedByGroup` 표기만) → 40(canonical, 골드의 `ownedBy`(11) + `ownedByGroup`(29)을 합침)로
  는다. 이 시점 KG·검색기는 아직 **레거시(플래그 OFF)** 경로만 담겨 있다 — OFF는 `ownedBy`로
  방출하므로, 골드가 `ownedByGroup`으로 쓴 문항 중 방출이 `ownedBy`로 표기가 달라 raw가
  놓치던 것을 canonical이 잡는다. rule 42문항(s3-roff)은 골드 2건이 전부 canonical에서
  일치(1.000)로 바뀐다.
- **`hasSegment`·`originatesFrom`·`siblingBrand`는 안 바뀐다(0.000 그대로).** 이 술어들은
  OFF `priority_preds`에 아예 없어(§O0-B 표) 이름이 뭐든 KG 조회 경로에서 통째로 버려진다 —
  별칭·표기 정규화로 고칠 수 있는 문제가 아니라 **누락**이라 canonical도 0이다. (ON에서는
  §1의 도달률처럼 완전히 살아난다 — 그 효과는 실제 답변 recall로는 O7 플래그 on/off 측정에서
  잰다. 이 rescore는 아직 저장된 리포트의 trace만 다시 채점한 것이라 코드 경로가 안 바뀐다.)
- `belongsToCategory`는 등록부에 정규화 대상(브랜드·그룹)이 아니라 canonical == raw다(0.375·0.333).
- `s3`는 3회 모두 0.475로 완전히 같다 — 이 술어는 실행 간 흔들림이 없었다.

### 가져야 할 결론

- **O0-B 무비용 도달률(§1)은 이름 정규화(canonical_predicate)만으로 `ownedByGroup`이 이미
  29/29까지 도달 가능함을 보였는데(플래그 OFF에서도!), 재사용 기준선 L3 recall 재채점(§3)은
  실제로 그 절반 정도(0.42~0.48)만 잡는다.** 나머지는 정규화로 못 채우는 구조적 한계다 —
  질의 브랜드 인식(O2 범위)·상한 12개 잘림·평가 골드가 `ownedBy`/`ownedByGroup` 어느 쪽으로도
  안 쓰인 문항 등. §1의 "도달 가능(상한)"과 §3의 "실측 canonical recall" 차이를 O2·O3·O7에서
  구분해 봐야 한다.
- `hasSegment`·`originatesFrom`·`siblingBrand`·`acquiredIn`은 canonical로 못 고친다 — O3의
  플래그 ON(정적 사실 카드)이 유일한 해법이고, §1에서 확인했듯 이미 도달률은 100%로 올라간다.

## Owned 파일 변경 요약

- `scripts/ontology_offline_checks.py`: OFF/ON 분리 감지 + `ownedByGroup` rename 버그 수정.
- `eval/runner.py`: `_extract_l3_trace`가 `ontology_static` 카드도 담음.
- `eval/schemas.py` / `eval/metrics/l3_kg.py` / `eval/report.py`: canonical per-predicate 필드
  additive 추가.
- `scripts/rescore_l3_l4.py`: canonical 집계·표 추가, `eval_output/ontology-2026-09/rescore/canonical/`에
  새로 재채점.
- `scripts/typed_eval_summary.py`: `by_layer`에는 있었지만 `TRACKED`에 빠져 있던
  `l3_kg_edge_recall_gold_only`·`l3_kg_edge_recall_gold_only_canonical`·
  `l4_rule_constraint_violation_rate`·`l4_typed_consistency_rate`를 추가(구형 리포트는 "—").
  `eval_output/evidence-2026-09/notes/s6_compare.py`는 `T.TRACKED`를 그대로 참조해서(제외 목록에
  안 걸림) 따로 고치지 않아도 새 필드를 그대로 물려받는다(파일 자체는 `eval_output/` 아래라
  수정 금지 대상이기도 하다).
- 테스트: `tests/unit/scripts/test_ontology_offline_checks.py`,
  `tests/unit/scripts/test_rescore_l3_l4.py`, `tests/eval/test_runner_trace.py`,
  `tests/eval/test_metrics_l3_gold_only.py`에 새 동작 고정.
- `tests/eval tests/unit/scripts -q` 562 passed, 3 skipped(기존 스킵, 무관), 0 failed.
