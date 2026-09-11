# 평가 인프라 개선 진행 상황 (2026-09-06 시작 → 2026-09-10)

지시서: `docs/eval/opus5-eval-remediation-prompt-2026-09-06.md`
근거 검토: `docs/eval/rag-eval-review-2026-09-06.md`

| 단계 | 내용 | 상태 | 커밋 | 메모 |
|---|---|---|---|---|
| 1 | 미커밋 top_k 수정 마무리 + 사이클 8 기록 | done | 05be0ba | 검색 전용 측정 개념 recall 0.576→0.671. 파이프라인 확인은 6단계에서 +0.030으로 확정 |
| 2 | 하네스 기술 부채 4건 (타임아웃·requires_kg·비용·종합점수) | done | 1ac4e2e | 6단계 실행에서 4건 모두 실동작 확인 (lg106 타임아웃 분리, cost $0.378, requires_kg 130/42) |
| 3 | 골든셋 3층 분리 + as_of | done | 5f8d29b | document 78 / snapshot 56 / domain_expectation 38 |
| 4 | snapshot 문항 expected_values DB 자동 생성 | done | 33390c0 | 99건 갱신·6건 강등 → snapshot 50 / domain_expectation 44 |
| 5 | numeric_accuracy 지표 배선 | done | 8679c1f | snapshot만 게이트(0.5). 실행에서 v9.0 0.465 기록 |
| 6 | 3회 분산 측정 + baseline v9.0 | done | (이 커밋) | 노이즈 폭: 종합 0.003 / 검색 0.001 / 근거성 0.023 / 수치 0.044. 통과 수는 폭 4건. baseline v9.0-2026-09-10 저장. 3회 $1.13 |
| 7 | Judge 교차 채점 (claude-sonnet-5) | blocked | — | `.env`에 ANTHROPIC_API_KEY가 없어 실행 불가. 키가 준비되면 172문항×3콜 ≈ $2.8. 가격표는 eval/cost_tracker.py·eval/judge/llm.py에 추가해 둠 |

## 이어받는 방법

- 사이클 기록: `docs/experiments/eval_cycle8_2026-08-30.md`, `docs/experiments/eval_cycle9_2026-09-10.md`
- 종합 점수 정의 변경과 baseline 연속성: `docs/eval/overall-score-formula-2026-09-06.md`
- 회귀 기준: `eval/baselines/v9.0-2026-09-10/`. **v8.1 이하와 종합 점수를 직접 비교하지 말 것**
  (공식·골드·게이트가 모두 바뀜).
- 델타 판정 기준(사이클 9 §2): 검색 계열 0.01, 종합 0.01, 근거성 0.03, 수치 정확도 0.05,
  통과 문항 수 5건.

## 7단계 실행 절차 (키 확보 후)

1. `.env`에 `ANTHROPIC_API_KEY` 추가.
2. `eval_output/v9-run3/report.json`의 답변 172개를 `claude-sonnet-5` judge로 재채점.
3. gpt-4.1-mini 점수와의 상관·평균 차이, groundedness가 임계 0.70 기준으로 갈리는 문항 수를 기록.
4. 교체 여부는 결정하지 않는다 — 수치만 남긴다.

## 다음 사이클 1순위 (사이클 9 §5·§7)

KG에 시장 지표 트리플이 0건이라(HHI·리뷰 수) snapshot 문항의 절반이 구조적으로 답할 수
없다. 수치 정확도와 근거성 양쪽에 가장 큰 레버다.

---

# 사이클 10 (2026-09-12 시작)

기록: `docs/experiments/eval_cycle10_2026-09-12.md`

| 단계 | 내용 | 상태 | 커밋 | 메모 |
|---|---|---|---|---|
| 10-1 | 수치 정확도 거짓양성 교정 + v9 재채점 + baseline v9.1 + 사이클 9 정정 | done | 41d75d7 | 0.465 → 0.040, snapshot 통과 1 → 0. v9.0 보존 |
| 10-2 | 결측 지표로 규칙이 발동하지 않게 ("HHI: 0.000" 날조 제거) | done | (이 커밋) | SoS·HHI·CPI 조건 6개를 기존 `_present` 규약으로, 인라인 조건 3곳, 추론 컨텍스트 기본값 주입 제거 |
| 10-3 | SQLite 지표를 질의 엔티티에 맞춰 컨텍스트에 싣기 (평가는 as_of 고정) | in_progress | — | 경로 단절 4곳: current_metrics 키 불일치, metric_edges 미렌더링, 카테고리 주어 미조회, (d)는 10-2 |
| 10-4 | 1회 측정, v9.1과 비교 (**API 비용, 승인 필요**) | not_started | — | 판정 기준: 사이클 9 §2 (수치 정확도 행은 폐기, 0.000 폭) |

분리한 작업: `brand_metrics`·`market_metrics`가 08-31 이후 미기록 (task_93bf902c).
