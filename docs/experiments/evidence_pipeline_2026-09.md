# 증거 카드 · 규칙 추론 · ReAct 통합 — 단계별 결과 (2026-09)

> 공모전 이후 사후 작업(2026-09-17~). 지시서 `docs/plans/evidence-react-ontology-kickoff-prompt-2026-09-17.md`,
> 결정표 `docs/plans/evidence-react-ontology-decisions-2026-09.md`.
> 판정 규칙(지시서 §2): 평균 차이가 노이즈 기준(종합·검색 0.01, 근거성·관련성 0.03, 수치 정확도 0.05, 통과 수 5건)과
> 실제 실행 간 폭 중 큰 값 이상이고 실행 값 범위가 겹치지 않을 때만 "차이 있음".

## 0단계 — 정리와 측정 도구 수리

### 시작 상태 (2026-09-17)

- 브랜치 `feat/evidence-react-ontology-2026-09`를 `de90c05`(지시서 커밋)에서 생성.
- `data/chroma`: 백업 `data/chroma_backup_2026-09-17-pre-restore/`(1,145청크) → 787청크 삭제 → `amore_docs` **358**.

### 트랙별 파일 소유

| 트랙 | 소유 파일 | 비고 |
|---|---|---|
| 0-A 색인·조회 분리 | `src/rag/retriever.py`(초기화·색인부), `src/rag/retrieval_strategy.py`(`OWLRetrievalStrategy.__init__`, `create_owl_strategy`), 새 `src/rag/build_index.py`, `scripts/start.py`, 상태 API 라우트 | |
| 0-B 오류 가시화 | `src/rag/hybrid_retriever.py`(예외 처리부), `eval/runner.py`, `eval/schemas.py`, `eval/report.py`, `eval/brain_adapter.py`(`V4RetrievalTrace`·`_build_trace`) | |
| 0-C 비용 | `eval/cost_tracker.py`, `eval/schemas.py`·`eval/report.py`의 비용부 | 0-B와 같은 파일의 다른 영역 |
| 0-D 경로 관측 | `src/core/query_graph.py`, `src/core/graph_state.py`, `src/core/models.py`(Response), `src/core/brain.py:process_query` | 평가 리포트 연결은 0-B 병합 후 |
| 0-E 테스트 결함 | `tests/conftest.py`, `tests/unit/core/test_react_agent.py`, `tests/unit/core/test_cache.py` | |
| 1-A/1-B 시험지 | 새 `scripts/classify_golden_types.py`, `scripts/generate_rule_questions.py`, `eval/data/golden/typed/` | 코드 무관, 0단계와 병렬 |

### 시작 시 전체 테스트

- `de90c05` + chroma 복구 직후: **5,488 passed / 7 skipped / 0 failed** (403초). 이 실행에는 `test_react_run`의 실제 OpenAI 호출(F15)이 포함돼 있다.

### 병합 기록

| 트랙 | 커밋 | 내용 | 병합 전 리드 확인 |
|---|---|---|---|
| 0-D | `31040bf` | QueryGraph가 문항별 `route_trace`(route·confidence_level·score·구성 요소·도구) 기록 | diff 리뷰, 관련 테스트 72 passed |
| 0-C | `f7d9bcd` | 단가를 litellm `model_cost` 우선 조회(gpt-4.1-mini $0.40/$1.60), 리포트에 계층별 토큰·사용 단가 기록 | 1M+1M 토큰 → $2.00 확인 |
| 0-A | `3000d42` `49bc9f0` `368434a` `c349da9` | `DocumentRetriever.initialize()` 읽기 전용(`get_collection`만), 어긋남은 경고+`get_index_status()`, 색인 전용 `python -m src.rag.build_index`(`--dry-run`/`--prune`), `OWLRetrievalStrategy` `doc_retriever` 필수(F2), `scripts/start.py`가 서버 시작 전 색인(S0-2), `/api/v4/brain/status`에 `index_status` | 원본 복사본 dry-run: indexed 358 / expected 358 / in_sync True |
| 0-E | `c9db4e9` | `test_react_run` LLM 가짜화, 캐시 TTL 경계 버그(`>`→`>=`, 코드 결함) 수정, `tests/conftest.py` API 가드(더미 키 + 닫힌 base URL) | 가드 하에서 `test_llm_integration.py` 포함 50 passed |
| 2-A | `0ed7594` | 증거 카드 모델·어댑터·렌더러(새 파일만, 이 시점엔 호출처 없음 → 기준선 동작 불변) | 102 passed |
| 0-B | `024ad89` `f083519` | 검색 실패를 핵심(`retrieval_error` → 평가 인프라 실패)과 선택(`degraded`)으로 구분 기록, 19곳 판정 | 33 passed |
| 0-D2 | `f5eba7a` | route_trace를 평가 트레이스·리포트 집계(route·confidence 분포, 규칙 발화 문항)에 연결 | 변경 전 코드로 새 테스트 실행 → 8 failed(RED 확인) 후 병합 |
| 1-A/1-B | `d2a57a6` `3d39bd9` | 유형 분류·생성 문항(§1) | 생성 문항 3개 정답을 원본 DB로 재조회해 일치 확인 |

### 게이트 (2026-09-17 21:36)

- 전체 테스트(`9325a74`, `data/` 복사본이 있는 별도 worktree에서 — S0-7): **5,673 passed / 7 skipped / 0 failed** (369초). 실행 전후 원본 `data/` 파일 sha256 불변.
- `amore_docs` = 358 (0-A dry-run, 스모크·기준선 실행 후 복사본도 358).
- 오류 주입(`tests/unit/rag/test_retrieval_error_visibility.py`, `tests/eval/test_runner_retrieval_error.py`) · 비용(`tests/eval/test_cost_pricing.py`) · 경로 기록(`tests/unit/core/test_query_route_observation.py`, `tests/eval/test_route_trace_report.py`) 통과.
- 스모크 5문항(`0b39e02`): 채점 5 / 인프라 실패 0, route 전부 direct·high, 규칙 발화 0 — 알려진 결함이 그대로 관측됨.

### 비용 재계산 (0-C, 토큰 기반 — 기존 baseline 파일은 수정하지 않음)

입력·출력 단가가 같은 배율(8/3)로 바뀌므로 `재계산 = 기록 × 2.667`이 토큰 분할과 무관하게 정확하다. 리포트에는 L5(답변)+judge 토큰만 기록돼 있고 L2 임베딩·질의 확장(`DocumentRetriever.expand_query`, openai 클라이언트 직접 호출)은 어느 리포트에도 집계되지 않았다.

| 리포트 | 토큰 | 기록 비용 | 재계산 |
|---|---:|---:|---:|
| `eval/baselines/brain-v4-1.0-2026-09-17` | 1,203,879 | $0.3142 | $0.8379 |
| kg-full-run1/2/3 | 997,249 / 1,001,752 / 1,006,761 | $0.2547 / 0.2558 / 0.2581 | $0.6793 / 0.6821 / 0.6882 |
| kg-nokg-run1/2/3 | 808,865 / 808,112 / 824,234 | $0.2252 / 0.2262 / 0.2312 | $0.6005 / 0.6033 / 0.6164 |
| kg-norules-run1/2/3 | 1,008,482 / 1,009,674 / 999,796 | $0.2589 / 0.2594 / 0.2554 | $0.6904 / 0.6918 / 0.6810 |
| kg-reactowl-run1 | 4,226,054 | $0.6949 | $1.8531 |

## 1단계 — 유형별 시험지와 기준선

### 시험지 (1-A/1-B, `d2a57a6` `3d39bd9`)

- 분류: `scripts/classify_golden_types.py`(결정적 규칙 C1~C11, 문항별 근거 `eval/data/golden/typed/classification.jsonl`). 172문항 → numeric 53 · relation 8 · rule 10 · multihop 18(주 유형) · other 83(definition 9, document_explanation 34, document_general 18, ir_document 12, out_of_scope 7, forecast 3).
- 시험지에서 제외 23문항: 주 유형은 numeric·multihop이지만 `gold_source=domain_expectation`(추정 골드)이라 정답 채점 불가.
- 생성(`metadata.generated=true`, 근거 SQL·값·교차 확인 포함): rule 32(`scripts/generate_rule_questions.py`), relation 19(`generate_relation_questions.py`, `config/brands.json` 출처 트리플만), multihop 10(`generate_multihop_questions.py`, KG 관계 + DB 수치, 날짜 없는 KG 수치 엣지 미사용).
- 정답 수치 출처는 지표 테이블(결정 S1-1). 결론이 raw 재계산과 다른 후보는 제외(규칙 후보 결론 불일치 2, 브랜드 오귀속 4, multihop 3). 포함 문항의 두 출처 수치 차이: 39문항 중 최대 0.25(rg011 평균 순위 12 vs 9), 중앙값 0.029.
- 규칙 입력 공급 가능: 37개 중 13개 규칙만 DB·KG 입력이 있어 문항 생성에 12개 사용(sentiment 8·IR 5·이력 필요 규칙 불가).

| 유형 | 원본 | 생성 | 합계 |
|---|---:|---:|---:|
| ① numeric 수치 조회 | 31 | 0 | **31** |
| ② relation 관계 | 8 | 19 | **27** |
| ③ rule 규칙 판단 | 10 | 32 | **42** |
| ④ multihop 다단계 | 17 | 10 | **27** |

경계 문항 24개(목록은 `classification.jsonl`의 `boundary`): rule로 분류했지만 골드가 문서 서술인 9개(lg044·045·046·047·056·059·094·097·114), domain 라벨만으로 multihop 5개(lg155·156·159·192·195), 두 유형 기준 동시 충족 7개(lg051·074·082·109·194·196·099), other인데 rule 탐지기가 걸린 3개(lg111·121·122).

**측정 방식**: 통합 시험지 `eval/data/golden/typed/combined_v1.jsonl`(원본 172 + 생성 61 = 233)을 한 번 실행하고 `scripts/typed_eval_summary.py`로 유형별 부분집합을 같은 채점 코드로 재집계한다. 유형에 속하지 않는 106문항(other + 제외 23)은 회귀 확인용이다.

### 1-C 기준선 (`0b39e02`, 2026-09-17 21:37~21:59, 3회, 동시 3개·45초 간격, 실행별 Chroma 복사본)

조건: `--target v4 --data-as-of 2026-08-31 --judge llm --judge-model gpt-4.1-mini --semantic-similarity --concurrency 4`, 저장소 기본 플래그(ReAct OFF, OWL 전략 OFF, reranker OFF). 3회 모두 채점 233 / 인프라 실패 0 / degraded 0, 실행 후 Chroma 358. 저장: `eval/baselines/brain-v4-typed-1.0-2026-09-17`(종합 점수 중앙값인 1회차), 3회 리포트 `eval_output/evidence-2026-09/base-run{1,2,3}/`.

| 지표 (평균 [최소, 최대]) | ① numeric 31 | ② relation 27 | ③ rule 42 | ④ multihop 27 | 그 밖 106 | 전체 233 |
|---|---|---|---|---|---|---|
| 종합 점수 | 0.663 [0.658, 0.668] | 0.687 [0.679, 0.695] | 0.759 [0.754, 0.761] | 0.655 [0.650, 0.660] | 0.606 [0.604, 0.608] | 0.656 [0.655, 0.658] |
| 통과 수 | 0 [0, 0] | 0 [0, 0] | 2 [2, 2] | 0 [0, 0] | 8.7 [7, 10] | 10.7 [9, 12] |
| L2 개념 Recall | 0.597 | 0.981 | 0.905 | 0.716 | 0.504 [0.498, 0.509] | 0.668 [0.666, 0.671] |
| L3 엣지 Recall | 0.473 | 0.062 | 0.881 | 0.407 | 0.693 | 0.592 |
| L5 근거성 | 0.774 [0.755, 0.811] | 0.686 [0.610, 0.732] | 0.701 [0.696, 0.708] | 0.728 [0.711, 0.747] | 0.665 [0.660, 0.672] | 0.696 [0.687, 0.705] |
| L5 관련성 | 0.742 [0.737, 0.748] | 0.884 [0.856, 0.900] | 0.916 [0.902, 0.924] | 0.802 [0.785, 0.826] | 0.865 [0.860, 0.867] | 0.853 [0.842, 0.860] |
| L5 토큰 F1 | 0.142 [0.134, 0.147] | 0.150 [0.144, 0.155] | 0.216 [0.211, 0.219] | 0.184 [0.182, 0.187] | 0.123 [0.120, 0.125] | 0.152 [0.151, 0.154] |
| L5 수치 정확도 | 0.000 [0.000, 0.000] | 0.250 | 0.032 | 0.026 [0.019, 0.038] | 0.028 [0.000, 0.056] | 0.029 [0.028, 0.032] |
| 규칙 발화 문항 비율 | 0 | 0 | 0 | 0 | 0 | 0 |
| 규칙 정답 일치율 | — | — | 0.469 (생성 32문항 중 "발화하지 않음"이 정답인 15문항만 일치) | — | — | — |

| 분포 (3회 모두 동일) | 값 |
|---|---|
| route | direct 232, decide 1(relation 유형 1문항) — clarify·react 0 |
| confidence_level | high 232, medium 1 |
| 규칙 추론 | 발화 문항 0, 추론 0건 |
| ReAct | 0 (플래그 OFF) |

비용: 리포트 합 $3.232(답변+judge, 실제 단가) → 질의 확장·임베딩 미집계 보정 ×1.1 ≈ **$3.56**.

해석(추측 없이 말할 수 있는 것만): 네 유형 모두에서 경로가 사실상 한 가지(direct/high)라 신뢰도 분기·DecisionMaker·ReAct는 이 기준선에서 측정되지 않았다. 규칙은 한 번도 발화하지 않았고, 수치 조회 유형의 수치 정확도는 3회 모두 0이다(F8: v4 답변 프롬프트에 DB 수치가 없음). ③ rule의 종합 점수·관련성이 높은 것은 생성 문항에 골드 엣지·문서가 없어 L3·L2 게이트가 기계적으로 통과하는 영향이 섞여 있으므로 유형 간 비교에 쓰지 않는다.
