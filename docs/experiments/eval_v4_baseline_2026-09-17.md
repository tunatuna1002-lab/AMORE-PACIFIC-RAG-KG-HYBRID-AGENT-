# v4 Brain 경로 기준선 — brain-v4-1.0 (2026-09-17)

> 2026-09 공모전 이후 사후 분석(docs/portfolio/amore_architecture_evidence.md §7)에서 "평가가
> 대시보드 경로가 아니라 v1을 측정했다"는 문제를 확인하고, 결정 D3
> (docs/plans/risk-remediation-decisions-2026-09-17.md)에 따라 새로 시작한 기준선이다.
> **v1 기준선(v1.0~v9.1)과 직접 비교하지 않는다.**

## 조건

| 항목 | 값 |
|---|---|
| 평가일 | 2026-09-17 16:22~16:35 KST |
| 대상 경로 | `--target v4` → `UnifiedBrain.process_query(skip_cache=True)` (eval/brain_adapter.py) |
| 대상 커밋 | `3c5af51` (report.json `config.git_commit`, 미커밋 변경 없음) |
| 데이터셋 | `eval/data/golden/laneige_golden_v2.jsonl` 172문항 (document 78 · snapshot 50 · domain_expectation 44) |
| 데이터 시점 | `--data-as-of 2026-08-31` |
| 판정 | `--judge llm --judge-model gpt-4.1-mini --semantic-similarity --concurrency 4` |
| 피처 플래그 | 저장소 기본값: ReAct OFF, OWL 전략 OFF, reranker OFF, KG·규칙 추론·DB 지표 ON |
| 실행 횟수 | 1회 |
| 저장 | `eval/baselines/brain-v4-1.0-2026-09-17/` (원본 `eval_output/v4-full-run1/`) |

v1 평가 명령에 붙던 `FF_AGENTS_USE_EXTERNAL_SIGNALS=false`, `LLM_TEMPERATURE=0.1`은 v4에 작용하지 않는다.
v4는 외부 신호를 쓰지 않고, 답변 온도는 `ResponsePipeline` 0.3·`DecisionMaker` 0.1로 코드에 고정돼 있다.

## 결과

| 지표 | 값 |
|---|---|
| 채점 / 인프라 실패 | 172 / 0 |
| 종합 점수 | 0.626 |
| 통과 | 16 (9.3%) — document 5, domain_expectation 11, snapshot 0 |
| L1 엔티티 F1 / 개념 F1 | 0.654 / 0.365 |
| L2 개념 Recall / MRR | 0.548 / 0.370 |
| L3 Hits@8 / 엣지 Recall | 0.826 / 0.592 |
| L5 근거성 / 관련성 / 토큰 F1 | 0.705 / 0.847 / 0.128 |
| L5 수치 정확도 (68문항) | 0.029 |
| 평균 지연 | 5.7초 |
| 추적 비용 | $0.314 (답변·결정·환각 점검·judge. 질의 확장·임베딩은 미집계) |

실행 간 분산은 이 경로에서 따로 재지 않았다. KG 실험(docs/experiments/kg_ablation_2026-09.md)의
full 구성 3회가 requires_kg=true 130문항에 대한 분산을 준다.

## 한계 — 이 수치를 읽을 때

1. **대시보드 스트림 경로와 완전히 같은 코드가 아니다.** 대시보드는 `process_query_stream`을 쓰고,
   신뢰도 분기 규칙이 QueryGraph와 별도로 그 메서드 안에 중복 구현돼 있다. 어댑터는 QueryGraph 쪽을 잰다.
   캐시는 껐다.
2. **v4 답변 프롬프트에는 크롤 DB 수치 사실이 실리지 않는다.** `3fce8e8`에서 추가한 `metric_facts`는
   v1 `ContextBuilder`만 렌더링하고 v4의 `combined_context`에는 없다. 반면 judge 컨텍스트(`data_facts`)에는
   두 경로 모두 들어간다. snapshot 통과 0건·수치 정확도 0.029의 원인 후보다(미검증, FUTURE_WORK 9.8).
   172문항 중 72개 답변이 "제공된 데이터"·"확인할 수 없"·"포함되어 있지 않"·"데이터에 없" 중 하나를
   포함한다(문자열 검색, 문맥 미확인).
3. **v1 기준선과 비교할 수 없다.** 경로·프롬프트·온도·비용 집계 범위가 모두 다르다. v1의 최신 측정(v9.1)은
   `96673d1`·`3fce8e8` 이전 코드이기도 하다.
4. 1회 실행이다. 통과 수는 사이클 9 기준 폭 4건, 근거성 0.023 정도 흔들릴 수 있다.
