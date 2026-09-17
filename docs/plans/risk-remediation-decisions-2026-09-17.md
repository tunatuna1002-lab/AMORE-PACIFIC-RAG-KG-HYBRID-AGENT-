# 위험 지점 보완 — 결정표 (2026-09-17)

> 대상: `docs/portfolio/amore_architecture_evidence.md`가 지적한 R1~R4.
> 지시서: `docs/plans/risk-remediation-kickoff-prompt-2026-09-17.md`
> 상태: **확정** (아래 §확정 기록)

## 과거 기록을 찾은 범위

- 저장소: `docs/plans/`, `docs/eval/`, `docs/experiments/`, `docs/dev/FUTURE_WORK.md`, `docs/portfolio/`, `PORTFOLIO_FACTS.md`, `.sisyphus/`, `.omc/`, 미추적 `refactoring-plan-2026-08-31.md`·`refactoring-kickoff-prompt.md`.
  → ReAct·OWL·v4 평가·KG ablation 처리 방향을 **정한** 기록은 없음. `refactoring-plan-2026-08-31.md`의 D1~D4는 HHI 스케일·이벤트 버스 등 다른 쟁점.
- 이전 Claude 세션 기록 11개(2026-08-29 ~ 09-17)와 `memory/`.
  → 사용자가 직접 답한 결정은 **한 세션(`b8589097`, 2026-09-17 15:36 KST)의 AskUserQuestion 응답 1건**뿐이다. 그 세션이 지시서를 만들었고 D1~D5 권고안은 그때 어시스턴트가 쓴 것이다. 이 결정표 파일은 그동안 만들어진 적이 없다.

## 결정표

| 결정 ID | 쟁점 | 과거에 정한 것 (출처) | 출처 못 찾음 | 이번 권고안 | 근거 |
|---|---|---|---|---|---|
| D0 (R1 방향) | ReAct·OWL을 삭제 / 서술만 교정 / 연결·수리 | **연결·수리** — 사용자 답 "실제로 연결·수리" (세션 `b8589097` 2026-09-17T06:36:17Z, 선택지 4개 중). 같은 답에서 보완 범위 = R1~R4 전부, 제약 = "코드 수정·커밋 허용, 유료 API 실행 허용" | — | 과거 결정대로 (지시서에서도 확정 항목) | 사용자 직접 응답 |
| D1 (R1) | 수리 후 ReAct·OWL 기본 ON/OFF | 기록 없음. 지시서 권고(같은 세션, 어시스턴트 작성·미확정): 플래그 뒤, 테스트 통과 + 평가 회귀 없음이면 ON, 아니면 OFF + "연결됨, 기본 비활성" | **못 찾음** (권고만 있음) | 지시서 권고 + 보강 3가지: ① **`retriever.use_owl_strategy`는 지금 `config/feature_flags.json`에 `true`다.** 생성자만 고치면 대시보드가 검증 없이 곧바로 OWL 경로로 바뀌므로, 수리 커밋에서 기본값을 `false`로 내린다. ② ReAct용 플래그 `agents.use_react_agent`(기본 `false`)를 새로 둔다. ③ ON 판정은 Phase 3의 ReAct·OWL 구성(1회)이 같은 130문항 full 3회의 범위 대비 기존 노이즈 기준(종합 −0.01, 근거성 −0.03, 통과 수 −5, 수치 정확도 −0.05)을 넘게 떨어지지 않고, ReAct가 실제로 1문항 이상 발동했을 때만. 1회 측정이라는 한계는 문서에 적는다 | 플래그가 이미 true라 "고치면 켜짐"이 됨. OWL 전략은 `use_reranking=True`가 기본이라 reranker 플래그(OFF, `7d8725b`)도 따르게 해야 함 |
| D2 (R1) | 실행기 없는 `query_data`·`query_knowledge_graph`·`calculate_metrics` | 기록 없음. 지시서 권고: 앞의 둘은 구현(SQLite 읽기 전용, KG 조회), `calculate_metrics`는 `MetricCalculator` 얇게 감싸기, 부작용 도구 추가 안 함 | **못 찾음** (권고만 있음) | 지시서 권고대로 + 보강: `refine_search`는 내부적으로 `query_data`를 호출하므로(`react_agent.py:327`) 같은 수리로 함께 살아난다. `final_answer`는 `action_input.answer`에서 답을 취하도록 고친다. 초기화 시 `ALLOWED_ACTIONS`와 등록된 실행기가 어긋나면 warning을 남기고, 미등록 도구 선택 시 관찰값에 오류를 넣고 루프를 계속한다 | `react_agent.py:244-245, 321-327`, `brain.py:903-907` |
| D3 (R2) | 평가 대상 경로 | 기록 없음. v1/v4가 다르다는 **관찰**만 있음 (세션 `53b49023` 2026-08-29, `b8589097` 2026-09-17). 지시서 권고: `--target v1\|v4` 어댑터, 이후 기준선은 v4로 새로 시작, v1과 직접 비교 안 함, 이중 경로 통합은 범위 밖 | **못 찾음** (권고만 있음) | 지시서 권고대로 + 한계 명시: 대시보드는 `process_query_stream`(라우팅이 `brain.py:614-704`에 인라인)을, 어댑터는 `process_query`(`QueryGraph`)를 탄다. **같은 분기 규칙을 두 곳에 중복 구현한 것이라 완전히 같은 경로라고 할 수 없다.** 어댑터는 `skip_cache=True`로 호출하고, ReAct 수리는 두 곳(`brain._process_with_react`, `query_graph._node_react`) 모두에 적용한다. 이 차이는 v4 기준선 문서에 한계로 적는다 | `src/core/query_graph.py:476-527` vs `brain.py:560-704`, 스트림은 SSE라 평가에 부적합 |
| D4 (R3) | KG 효과 실험 설계 | 기록 없음. 과거 ablation은 30문항(requires_kg=false)·6구성·1회(`docs/experiments/ablation_2026-08-30.md`). 지시서 권고: requires_kg=true 전체, full / KG off / 규칙 추론 off, 구성당 3회, D3 경로, 무효과·음의 효과도 기록 | **못 찾음** (권고만 있음) | 지시서 권고대로. 문항 수 = **130** (172 중 requires_kg=true). 비용 절감: v4 기준선(172문항, 같은 커밋·같은 구성)의 130문항 부분을 full 1회차로 재사용 — 실행 조건이 같으므로. 비용 초과 시 축소 순서(미리 고정): ① ReAct·OWL 구성 생략 → ② 규칙 추론 off를 3회→2회. full·KG off 3회는 유지 | 예상 비용은 아래 표. 분산 기준은 `11698b7`(통과 수 폭 4, 근거성 0.023) |
| D5 (R4) | PORTFOLIO_FACTS.md 처리 + 문서 교정 | 과거에는 PORTFOLIO_FACTS.md를 사이클마다 갱신하는 살아 있는 문서로 운용(세션 `53b49023`·`7219dd4c`·`c795424d`, 2026-08-29~31). 동결·대체 방향은 지시서가 처음 제안 | 동결 결정은 **못 찾음** | 지시서 권고대로(상단 고지 + 본문 보존, 나머지 문서는 현재 코드 기준 교정). 추가 교정 대상 2건: PORTFOLIO_FACTS.md:113이 `use_owl_strategy`를 "live"로 적었으나 당시에도 생성 실패였음(고지에 한 줄), 근거 문서 §3.3의 "`9c32aba`에서 유입"은 부정확 — `9c32aba`(02-08) 당시 `docs_path`는 `get_true_hybrid_retriever`의 유효 인자였고 `f049cb8`(02-15) 리팩터가 `OWLRetrievalStrategy`로 바꾸며 인자만 남겨 깨졌다 | `git log -S docs_path`, `git show 9c32aba:src/core/brain.py:309-311` |
| D6 (선행 확인) | `retrieval_strategy.py` "다른 세션 작업 중" | 완료됨 — `05be0ba`(2026-09-06, 인텐트 top_k 예산 정렬)로 커밋. 세션 `c795424d` 기록과 일치 | — | 충돌 없음. 미추적 `src/rag/retrieval_strategy 2.py`는 HEAD 파일과 바이트 동일한 Finder 복제본이며 읽기만 하고 건드리지 않는다 | `diff` 결과 없음, `git log -- src/rag/retrieval_strategy.py` |

## 예상 비용 (D4 포함 전체, 확정 시 상한 $5 → $10으로 변경)

v1 실측 172문항 1회 ≈ $0.38 → 문항당 약 $0.0022 (답변 + judge). v4는 DecisionMaker·HallucinationDetector 호출이 더해질 수 있어 문항당 비용을 5문항 스모크에서 실측한 뒤 확정한다.

| 실행 | 문항×회 | v4 비용 = v1과 같을 때 | v1의 1.5배일 때 |
|---|---|---|---|
| 스모크 | 5 | $0.01 | $0.02 |
| v4 기준선 (= full 1회차 재사용) | 172 | $0.38 | $0.57 |
| full 추가 2회 | 130×2 | $0.57 | $0.86 |
| KG off 3회 | 130×3 | $0.86 | $1.29 |
| 규칙 추론 off 3회 | 130×3 | $0.86 | $1.29 |
| ReAct·OWL ON 1회 | 130 | $0.29 (ReAct 반복 호출로 더 큼) | $0.43+ |
| **합계** | | **≈ $2.97** | **≈ $4.46** |

확정 시 상한이 $10으로 올라 80% 선은 $8.00이다. 두 시나리오 모두 선 아래이므로 축소 없이 전량 실행하되, 스모크 실측값으로 재계산해 $8을 넘을 것 같으면 D4의 축소 순서를 적용하고 보고서에 적는다.

## 확정 기록

- 확정 일시: 2026-09-17 16:04 KST (사용자 응답 "한도는 10달러로 늘려줘. 그리고 하던대로 진행하도록 해")
- 최종값: D0~D6 모두 위 표의 **이번 권고안** 그대로 확정.
- 변경 1건: 유료 API 총 상한 **$5 → $10** (80% 경보선 $8).
