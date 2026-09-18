# 증거 카드 · 규칙 추론 · ReAct 통합 — 실행 지시서 (0~6단계)

> 새 Claude Code 세션에 `---` 아래 블록을 그대로 붙여넣기.
> 배경: 2026-09-17 R1~R4 보완(브랜치 `fix/wire-react-owl-2026-09`)에서 ReAct·OWL은 연결됐지만 기본 OFF이고, 평가에서 규칙 추론 0건·ReAct 발동 0건·OWL 검색 결함이 확인됐다. 이 지시서는 원래 의도한 "KG + 온톨로지 규칙 + ReAct가 함께 동작하는 에이전트"를 설계부터 다시 맞춰 실제로 동작·측정되게 만든다.
>
> **붙여넣기 전에 확인·수정할 곳 (3곳)**
> 1. 유료 API 총 상한: 기본 **$40**
> 2. `data/chroma`에 실험 중 잘못 추가된 청크 787개 제거 승인: 기본 **승인**(백업 후 제거). 승인하지 않으면 해당 줄을 "미승인"으로 바꿀 것
> 3. 작업 브랜치 이름: 기본 `feat/evidence-react-ontology-2026-09`

---

## 0. 원칙과 권한

**첫째 원칙은 정확하게, 둘째 원칙은 빠르게다.** 둘이 부딪히면 언제나 정확함을 택해라.
- 정확하게: 모든 주장은 코드·테스트·측정으로 확인한다. 추측은 추측이라고 적는다. mock으로 버그를 가리는 테스트를 만들지 않는다(이 저장소는 `1cd4307`에서 "테스트가 버그를 정상으로 박제"한 전례가 있다). 결과를 유리하게 다듬지 않는다 — 산출물은 면접에서 검증당할 근거다.
- 빠르게: 독립적인 작업은 병렬로 돌리고, 기다리는 동안 다른 트랙을 진행한다. 토큰은 아끼지 않아도 된다.

**병렬 오케스트레이션을 허용한다.** 서브에이전트(`Agent` 도구, 코드 수정 작업은 `isolation: "worktree"`), 백그라운드 실행, 필요하면 `Workflow` 도구를 써서 여러 에이전트를 동시에 돌려라. 너(리드)는 설계 결정·작업 분배·리뷰·병합·게이트 판정을 맡는다.

**`data/chroma` 청크 787개 제거: 승인.** 먼저 `data/chroma` 전체를 `data/chroma_backup_2026-09-17-pre-restore/`로 복사한 뒤, `eval_output/risk-remediation-2026-09-17/chroma_added_ids.json`의 ID만 삭제하고 `amore_docs`가 358개인지 확인해라. 이 외에 `data/`의 DB·KG·Chroma를 수정하지 마라.

**유료 API 총 상한: $40**(경보선 80% = $32). 장부 `eval_output/risk-remediation-2026-09-17/cost_ledger.md`에 이어 적어라(직전 작업 누적 ≈ $11.1은 이 상한에 포함하지 않는다). 0단계에서 단가표를 고치기 전까지는 리포트 비용 × 2.667 × 1.1로 계산해라.

멈추고 질문할 곳은 두 가지뿐이다: (1) 위 승인 범위를 넘는 데이터 삭제·수정이 필요할 때, (2) 경보선에 닿아 남은 측정을 줄여야 할 때(줄이는 순서는 §3에 정해 두었으니 적용 후 보고만 해도 된다). 그 밖의 모호함은 신중한 동료처럼 스스로 판단하고, 판단 근거를 커밋 메시지나 문서에 한 줄로 남겨라. 계획이 틀렸거나 더 나은 방법이 있으면 한 문장으로 밝히고 진행해라 — 조용히 범위를 줄이거나 늘리지 마라. 끝낼 수 없는 항목은 나머지를 다 한 뒤 무엇이 왜 빠졌는지 명시해라.

## 1. 먼저 읽을 것 (병렬로 읽어라)

- `docs/portfolio/amore_architecture_evidence.md` — 특히 §3.3, §5.3, §5.5, §7의 **[2026-09 사후]** 표시
- `docs/experiments/kg_ablation_2026-09.md` — 규칙 0건·신뢰도 전부 HIGH·OWL 필터 결함의 근거
- `docs/experiments/eval_v4_baseline_2026-09-17.md`
- `docs/plans/risk-remediation-decisions-2026-09-17.md`
- `docs/dev/FUTURE_WORK.md` §9.8
- `CLAUDE.md` (구조·명령어·컨벤션)

## 2. 현재 상태 (2026-09-17 기준, 수정 전 반드시 재확인)

브랜치 `fix/wire-react-owl-2026-09` HEAD `244cb1c`(push 안 함). 여기서 새 브랜치를 만든다. 전체 테스트 5,488 passed / 7 skipped / 0 failed.

확인된 결함과 위치:

| # | 결함 | 위치 | 근거 |
|---|---|---|---|
| F1 | 검색기 초기화가 Chroma에 **쓰기(증분 색인)**까지 한다. 여러 프로세스·전략이 같은 컬렉션을 오염시킨 원인 | `src/rag/retriever.py` `initialize`(504~) → `_initialize_vector_search`(744~) → `collection.add`(~944) | 358→1,145 사고 |
| F2 | `OWLRetrievalStrategy` 생성자가 `doc_retriever` 미지정 시 시맨틱 청킹 검색기를 새로 만든다 | `src/rag/retrieval_strategy.py` `__init__` | `create_owl_strategy`만 `0669b75`에서 우회 |
| F3 | 검색 예외를 삼키고 빈 컨텍스트로 답을 계속 만든다. `except Exception` 18곳 | `src/rag/hybrid_retriever.py` (예: `retrieve` 638~) | 평가가 인프라 실패로 분류 못 함 |
| F4 | gpt-4.1-mini 단가가 공시가의 1/2.67 | `eval/cost_tracker.py:34` | $0.15/$0.60 vs $0.40/$1.60 per 1M |
| F5 | 신뢰도가 개수 기반이라 평가 172/172문항이 HIGH → DecisionMaker·ReAct 미실행 | `src/core/query_graph.py:_node_assess_confidence`(116~), `src/core/brain.py:_assess_confidence_level`(1059~, **중복 구현**), `src/core/confidence.py` 임계 5.0/3.0/1.5 | 로그 `HIGH confidence - skipping` 172/172 |
| F6 | 분기 규칙이 두 벌: 대시보드 `process_query_stream`(인라인)과 `process_query`(QueryGraph). `_is_complex_query`도 두 벌 | `brain.py:990`, `query_graph.py:430` | 평가가 대시보드와 같은 코드를 잰다고 말할 수 없음 |
| F7 | 규칙 추론 입력을 `current_metrics["summary"]`·`brand_metrics`(대시보드 JSON 형식)에서 읽는다. 바로 위에서 DB 수치(`metric_facts`)를 가져오지만 쓰지 않는다 → 추론 0건 | `hybrid_retriever.py:_build_inference_context`(1091~), `metric_facts_provider.collect`(~526) | 130문항 × 10실행 추론 0건 |
| F8 | v4 답변 프롬프트에 DB 수치 사실이 없다(v1 `ContextBuilder`만 렌더링) | `hybrid_retriever.py:_combine_contexts`(1640), `src/core/response_pipeline.py:_format_context`(312), `src/rag/context_builder.py` | v4 snapshot 통과 0 |
| F9 | OWL 검색 필터: `$or` Chroma where 형식을 `_matches_filters`가 처리 못 하고 문서 메타데이터에 brand·category 키가 없다 → 엔티티가 연결되면 문서 0건 | `retrieval_strategy.py:_matches_filters`, `src/rag/entity_linker.py:get_ontology_filters`(1088) | 124/130문항 청크 0, 오프라인 재현 130/130 일치 |
| F10 | OWL 경로가 레거시 경로의 BM25/RRF·DB 사실·KG 사실 조회를 버린다 | `retrieval_strategy.py:retrieve` | 엣지 Recall 0.200 |
| F11 | 도구가 두 벌: DecisionMaker용 5종(`dashboard_data.json` 읽기, `brain.py:_register_dashboard_tools` 943~)과 ReAct용 3종(`src/core/react_tools.py`). LLM 호출은 JSON 문자열 파싱 | `src/core/decision_maker.py`, `src/core/react_agent.py` | 네이티브 function calling 미사용 |
| F12 | KG의 수치 엣지(`hasSoS` 등)는 날짜 버전 없이 낡은 값이 섞여 DB 수치와 충돌 | `src/ontology/kg_enricher.py`, `kg_query.py` | 근거 문서 §5.4 |
| F13 | 브랜드 표기 이중화(LANEIGE/laneige)를 조회 시 변형 4종으로 우회 | `hybrid_retriever.py:884-893` | 근거 문서 §5.5 |
| F14 | 호출되지 않는 코드: `src/ontology/unified_reasoner.py`(인스턴스화 0), `src/core/llm_orchestrator.py`·`src/core/query_processor.py`(서비스 import 0), SPARQL(`kg_query.py`·`owl_reasoner.py` 내부에서만) | — | 삭제 전 호출처 재확인 필수 |
| F15 | 테스트 결함: `tests/unit/core/test_react_agent.py::test_react_run`이 실제 OpenAI 호출, `tests/unit/core/test_cache.py::test_cleanup_expired_removes_old` 간헐 실패 | — | FUTURE_WORK 9.8 |

골든셋 `eval/data/golden/laneige_golden_v2.jsonl` 172문항: domain metric 30 · product 30 · brand 25 · market 25 · multi_hop 20 · edge 15 · time 15 · ir 12. gold 필드 비어 있지 않은 수: kg_edges 105 · expected_values 68 · doc_chunk_ids 149. 규칙 판단 전용 gold 필드는 없다.

평가 명령(v4): `.venv/bin/python -m eval.cli run --dataset <jsonl> --out <dir> --target v4 --data-as-of 2026-08-31 --judge llm --judge-model gpt-4.1-mini --semantic-similarity --concurrency 4`. 비교: `scripts/kg_ablation_compare.py`, 교차 채점: `scripts/judge_context_swap.py`. 노이즈 기준(사이클 9): 종합·검색 0.01, 근거성·관련성 0.03, 수치 정확도 0.05, 통과 수 5건 — 판정은 이 값과 실제 실행 간 폭 중 큰 값, 그리고 범위 비겹침.

## 3. 확정된 설계 (다시 묻지 말고 따를 것)

**E1. 증거 카드(Evidence) 단일 모델.** KG 사실, DB 수치, 문서 청크, 규칙 추론 결과, ReAct 도구 관찰을 모두 하나의 도메인 모델로 표현한다. 최소 필드: `id`, `kind`(relation|metric|document|inference|observation), `subject`, `predicate`, `object`, `value`, `unit`, `as_of`, `source`, `confidence`, `derived_from`(추론이면 근거 카드 id 목록), `text`(프롬프트용 한 줄). 위치는 `src/domain/entities/evidence.py`(Clean Architecture: domain은 외부 의존 없음). 프롬프트 조립·출처 표시·평가 트레이스·judge 컨텍스트가 **이 모델만** 읽는다.

**E2. 역할 분담.** KG = 구조적 관계(소유·카테고리 소속·경쟁·제품). DB(SQLite) = 날짜 붙은 수치의 정본. 문서(RAG) = 정의·해석. 온톨로지 = 단어 정규화·계층·규칙의 입력 계약. KG의 수치 엣지는 검색 증거로 쓰지 않는다(삭제하지 말고 조회에서 제외, 날짜 버전이 붙은 경우만 허용).

**E3. 추론 엔진은 규칙 엔진 하나.** 기존 Python 규칙 엔진(`src/ontology/reasoner.py`, 규칙 37개)을 유지하고 입력을 증거 카드로 바꾼다. 규칙마다 필요한 입력(이름·타입·단위·범위)을 선언형 스키마로 둔다. 입력이 없으면 발화하지 않고 "미발화 사유"를 기록한다. 추론 결과는 `kind=inference` 카드로 `derived_from`을 채운다. OWL(owlready2)은 카테고리 계층·일관성 검사 같은 "단어사전" 역할로만 남기고 검색 전략으로 쓰지 않는다.

**E4. 도구 레지스트리 하나.** 도구 5종 — `resolve_entity`(단어사전), `kg_neighbors`(관계), `get_metrics`(DB, as_of 준수), `apply_rules`(규칙 엔진), `search_docs`(Dense+BM25 RRF) — 전부 읽기 전용, 반환은 증거 카드. 정해진 파이프라인(쉬운 질문)과 ReAct(어려운 질문)가 **같은 레지스트리**를 쓴다. LLM의 도구 선택은 네이티브 function calling(`tools=`, `tool_choice`)으로 한다. `dashboard_data.json` 읽기 도구 5종은 새 도구로 대체한 뒤 호출처를 옮기고 제거한다.

**E5. 분기 규칙 하나.** QueryGraph를 유일한 구현으로 남기고, 스트리밍은 QueryGraph 노드가 이벤트 콜백(status/tool_call/text/done)을 내보내는 방식으로 바꾼다. `brain.py`의 중복 분기·신뢰도·복잡도 함수는 제거한다. 합치기 전에 **특성화 테스트**(같은 입력에 두 경로가 같은 route를 내는지)를 먼저 만든다.

**E6. 신뢰도는 충족도 기반, 임계값은 골든셋으로 보정.** 개수 합산 대신 (a) 질문 엔티티가 증거에 등장하는 비율, (b) 질문 유형이 요구하는 증거 종류의 존재(수치 질문 → metric 카드, 관계 질문 → relation 카드, 판단 질문 → inference 카드), (c) 검색 점수 분포를 쓴다. 임계값은 유형별 시험지에서 "HIGH인데 실패"가 최소가 되도록 정하고 근거 표를 남긴다.

**E7. 난이도 라우터.** ReAct 발동은 키워드가 아니라 **필요 홉 수**로 판단한다(엔티티 해석 → 관계 → 수치 → 판단 중 몇 단계가 필요한가). 1홉 이하는 파이프라인, 2홉 이상은 ReAct. ReAct에는 최대 단계 수·문항당 토큰 예산·읽기 전용 도구만 준다. 켜기 전 **그림자 모드**(ReAct를 함께 돌려 결과만 기록하고 답은 파이프라인 것을 반환)로 먼저 측정한다.

**E8. 답변은 카드 인용 + 수치 검증.** 답변 프롬프트는 문장마다 카드 id 인용을 요구한다. 생성 후 답 속 숫자가 인용 카드의 `value`와 일치하는지 규칙 기반으로 검사하고, 불일치 수치는 제거하거나 "확인되지 않음"으로 바꾼다(검사 결과는 응답 메타데이터에 남긴다).

**E9. 색인과 조회 분리.** 색인은 전용 명령 `python -m src.rag.build_index`에서만 한다. 서버·평가·검색기의 `initialize()`는 읽기 전용이며, 색인이 문서와 어긋나면 경고와 상태 API 노출만 한다. 색인 시 청크 메타데이터에 엔티티 링커로 `brands`·`categories`·`metrics`를 태깅한다(문자열 목록은 Chroma 제약에 맞게 구분자 문자열로 저장).

**E10. 온톨로지 신호는 거르기가 아니라 가산점.** 엔티티·카테고리 일치는 재정렬 가산점으로만 쓴다. 필터로 결과가 k개 미만이 되면 필터 없는 결과로 채운다.

**E11. 죽은 코드 정리.** F14 목록은 호출처(import·인스턴스화·문자열 참조·테스트)를 다시 확인한 뒤, 서비스·평가 경로에서 도달하지 않으면 삭제하고 관련 테스트·피처 플래그도 함께 정리한다. 확신이 없으면 `docs/dev/FUTURE_WORK.md`에 남기고 건드리지 않는다. v1 챗 경로(`/api/chat`, `HybridChatbotAgent`)는 삭제하지 않고 증거 카드 조립기만 공유하게 한다(v1 제거는 범위 밖).

**경보선 도달 시 측정 축소 순서**(고정): ① 6단계 ReAct on/off 반복을 3회→2회 ② 4단계 OWL 제거 확인 측정을 1회로 ③ 2·3단계 재측정을 전체 172문항 대신 해당 유형 시험지로만. 기준선(1단계)과 최종 비교(6단계 on/off 각 최소 2회)는 줄이지 않는다.

## 4. 병렬 실행 규칙

- 리드는 단계 시작 때 **트랙별 파일 소유 표**를 먼저 만든다. 같은 파일을 두 트랙이 동시에 고치지 않게 나누고, 겹치면 순차로 돌린다.
- 코드 트랙은 서브에이전트 + `isolation: "worktree"`로 돌린다. 서브에이전트 지시에는 목표·소유 파일·금지 파일·테스트 방식(§5)·완료 조건을 모두 적는다.
- 병합은 리드가 **한 번에 하나씩** 한다: diff 리뷰 → 작업 브랜치에 병합 → 관련 테스트 → 다음 병합. 단계 게이트에서 전체 테스트 1회.
- 서브에이전트 결과는 믿지 말고 검증해라: diff를 직접 읽고, 테스트를 리드가 다시 돌린다.
- **평가 실행 격리**: 0-A(E9, 초기화 읽기 전용) 병합 이전에는 평가 프로세스마다 `data/chroma`를 스크래치 디렉터리로 복사해 `CHROMA_PERSIST_DIR`로 지정한다. 이후에도 동시 평가는 최대 3개, 시작 간격 45초 이상. 평가는 `run_in_background`로 띄우고 기다리는 동안 다른 트랙을 진행한다.
- 평가 대상 커밋은 `-dirty` 없이 기록되게 한다(평가 전 커밋, 평가 중 추적 파일 수정 금지 — 문서 작업은 별도 worktree에서).

## 5. 작업 규칙

- 테스트: `.venv/bin/python -m pytest`. 동작이 바뀌는 수정은 **실패하는 테스트를 먼저** 쓰고(RED 확인 기록), 구현 후 통과(GREEN). 가짜로 두는 것은 LLM 호출과 네트워크·색인 I/O뿐이고, 배선·KG·SQLite·규칙 엔진·QueryGraph는 실제 객체로 검증한다. 테스트가 실제 API를 부르지 않게 LLM 호출 경로에 가드를 둔다.
- 리팩터링은 특성화 테스트 → 변경 → 같은 테스트 통과 순서.
- 커밋: 단계 안에서도 논리 단위로, 컨벤션 `type(scope): 한글 요약` + 본문(무엇을·왜·검증). 끝에 `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>`.
- 전체 테스트: 시작 시 1회, 각 단계 게이트에서 1회.
- Ruff(line-length 100). pre-commit이 파일을 고치면 다시 스테이징해 커밋.
- 문서는 필요한 만큼만. 이 작업의 결과·결정은 `docs/experiments/evidence_pipeline_2026-09.md` 하나에 단계별로 누적하고, 결정은 `docs/plans/evidence-react-ontology-decisions-2026-09.md`에 표로 남긴다.
- 범위 밖 발견은 고치지 말고 `docs/dev/FUTURE_WORK.md`에 한 줄.

## 6. 단계별 작업

각 단계는 **게이트**(끝났다는 증거)를 통과해야 다음 단계로 간다. 괄호 안 트랙은 병렬 가능.

### 0단계 — 정리와 측정 도구 수리 (API $0)

병렬 트랙:
- **0-A 데이터·색인 분리** (소유: `src/rag/retriever.py`, `src/rag/retrieval_strategy.py` 생성자, 새 `src/rag/build_index.py`): `data/chroma` 백업 후 787개 제거·358 확인. E9 구현(초기화 읽기 전용, 색인 전용 명령, 어긋남 경고). F2 제거(`doc_retriever` 필수). 테스트: 초기화 두 번·프로세스 두 개가 컬렉션 개수를 바꾸지 않음(임시 Chroma 경로 사용).
- **0-B 오류 가시화** (소유: `src/rag/hybrid_retriever.py`의 예외 처리부, `eval/runner.py`, `eval/schemas.py`): F3 — 선택 기능 실패는 `degraded` 목록, 핵심 검색 실패는 `retrieval_error`로 기록하고 평가 러너가 인프라 실패로 분류. 테스트: 검색기에 예외 주입 → 리포트 `errored` 1.
- **0-C 비용** (소유: `eval/cost_tracker.py`): F4 — `litellm` 모델 가격(`litellm.model_cost`/`completion_cost`)으로 계산, 리포트에 prompt/completion 토큰 저장. 기존 기준선 파일은 수정하지 말고 토큰 기반 재계산표를 문서에 추가. 테스트: 알려진 토큰 수 → 공시가 일치.
- **0-D 경로 관측** (소유: `src/core/query_graph.py` 상태 메타데이터, `src/core/graph_state.py`, `eval/brain_adapter.py`): 문항별 `confidence_level`·`confidence_score`·`route`(direct|clarify|decide|react)·사용 도구를 응답 메타데이터와 평가 리포트에 기록.
- **0-E 테스트 결함** (소유: 해당 테스트 파일): F15 두 건.

게이트: 전체 테스트 통과, `amore_docs`=358, 오류 주입 테스트·비용 테스트·경로 기록 테스트 통과. 커밋 후 문서 §0 기록.

### 1단계 — 유형별 시험지와 기준선 (API ≈ $3~5)

- **1-A 시험지 분류** (0단계와 병렬 가능, 코드 무관): 172문항을 유형 ①수치 조회 ②관계 ③규칙 판단 ④다단계로 **gold 필드와 질문 구조에 근거해** 분류하는 스크립트(`scripts/classify_golden_types.py`)를 만든다. 분류 근거를 문항별로 남기고, 경계 문항 목록을 문서에 싣는다. `laneige_golden_v2.jsonl`은 수정하지 않고 `eval/data/golden/typed/{numeric,relation,rule,multihop}.jsonl`로 저장.
- **1-B 부족 유형 보충**: ③규칙 판단은 현재 전용 gold가 없다. 규칙 엔진의 규칙과 `--data-as-of 2026-08-31` DB 값으로 **기계적으로** gold를 만드는 스크립트로 25~40문항을 생성한다(예: "2026-08-31 기준 lip_care는 분산 시장인가?" → HHI 값과 규칙 임계로 정답·근거 수치 생성). ④다단계가 20문항 미만이면 KG 관계 + DB 수치 조합으로 같은 방식으로 보충한다. 생성 문항은 사람이 검토할 수 있게 근거 쿼리와 값을 함께 저장하고 `metadata.generated=true`를 단다.
- **1-C 기준선**: 0단계 병합 후, 현재 코드로 유형별 시험지 전체를 3회 측정(동시 최대 3, Chroma 격리). 유형별 평균·범위·route 분포·신뢰도 분포·규칙 발화 수·비용을 기록하고 기준선 `brain-v4-typed-1.0-<날짜>`로 저장.

게이트: 네 유형 각각 20문항 이상, 유형별 기준선 3회 수치와 분포표가 문서에 있음.

### 2단계 — 증거 카드와 프롬프트 통합 (API ≈ $2~3)

- **2-A 모델·어댑터** (소유: `src/domain/entities/evidence.py`, 새 `src/rag/evidence_adapters.py`): E1 모델, KG 사실·`MetricFactsProvider` 결과·검색 청크·규칙 결과·도구 관찰 → 카드 어댑터. 단위 정규화(SoS 0~1 정본, 표시용 % 변환은 렌더러에서)와 브랜드 정규화(F13, 단어사전 기반 canonical id)를 여기서 한 번만 한다. KG 수치 엣지 제외(E2, F12).
- **2-B 조립기 통합** (소유: `src/rag/context_builder.py`, `src/core/response_pipeline.py`, `hybrid_retriever.py:_combine_contexts`): v1·v4가 같은 카드 렌더러를 쓰게 하고(F8) 카드 id 인용 지시를 넣는다(E8의 앞부분). 특성화 테스트로 v1 프롬프트 섹션 구성이 의도치 않게 사라지지 않게 고정.
- **2-C 평가 연결** (소유: `eval/brain_adapter.py`, `eval/runner.py:_build_context_string`): judge 컨텍스트와 트레이스를 카드에서 만든다. 답변 프롬프트에 실린 카드와 judge 컨텍스트가 같은 집합이 되게 한다(현재는 judge에만 DB 사실이 들어가는 불일치).

게이트: 유형별 시험지 3회 재측정에서 ①수치 조회의 수치 정확도·통과 수가 기준선 대비 노이즈 이상 개선되거나, 개선이 없으면 원인 분석을 문서에 남김. 다른 유형의 근거성 회귀 없음.

### 3단계 — 규칙 추론 살리기 (API ≈ $2~3, 2단계 2-A 병합 후 시작, 2-B·2-C와 병렬 가능)

- **3-A 입력 계약** (소유: `src/ontology/rules/*`, 새 `src/ontology/rule_contracts.py`): 37개 규칙의 필요 입력을 선언형으로 정리하고, 계약 위반·결측 시 "미발화 사유"를 반환하는 래퍼. 규칙 임계값 단위가 SoS 0~1·HHI 0~1과 맞는지 전수 확인(HHI 스케일 결정 D1 `refactoring-plan-2026-08-31.md` 참조).
- **3-B 카드 입력 추론** (소유: `hybrid_retriever.py:_build_inference_context`, `src/ontology/reasoner.py`): F7 — 추론 입력을 metric/relation 카드에서 만든다(`current_metrics` 의존 제거). 결과는 inference 카드 + `derived_from`.
- **3-C 관측**: 평가 리포트에 문항별 발화 규칙·미발화 사유 상위 목록, 유형 ③의 규칙 정답 일치율 지표 추가.

게이트: DB 픽스처 통합 테스트에서 시장 포지션 규칙 발화, ③규칙 판단 시험지 3회에서 추론 발화율 > 0과 규칙 정답 일치율 기록, "규칙 off" 대비 차이를 §2 판정 규칙으로 판정(무효과여도 그대로 기록).

### 4단계 — 도구 레지스트리와 OWL 역할 정리 (API ≈ $2)

병렬 트랙:
- **4-A 레지스트리** (소유: 새 `src/core/tool_registry.py`, `src/core/react_tools.py`, `src/core/tools.py`, `brain.py:_register_dashboard_tools`): E4 도구 5종, function calling 스키마 자동 생성, DecisionMaker·ReAct 공용. `dashboard_data.json` 도구 제거 전 호출처 이전. 도구별 실제 SQLite·KG 픽스처 테스트.
- **4-B 색인 태깅·가산점** (소유: `src/rag/build_index.py`, `retrieval_strategy.py`의 검색 부분, `entity_linker.py:get_ontology_filters`): E9 태깅, E10 가산점·폴백. F9 재현 테스트(엔티티 연결 질의가 문서 0건이 되지 않음)를 먼저 RED로.
- **4-C OWL 역할 축소·죽은 코드** (소유: `retrieval_strategy.py:OWLRetrievalStrategy`, `container.py`, `brain.py` OWL 부분, F14 파일): E3에 따라 OWL 검색 전략을 제거하거나 "레거시 결과 위 온톨로지 가산점"으로 흡수(F10 해결 방식은 리드가 코드 확인 후 택일하고 근거 기록). `retriever.use_owl_strategy` 플래그 정리. E11.

게이트: 전체 테스트, 엔티티 연결 질의 청크 0건 문항 = 0(②관계 시험지 1회), 도구 레지스트리 단일화 확인(`get_available_tools` 호출처가 한 레지스트리만 봄).

### 5단계 — 분기 통합·신뢰도 보정·라우터 (API ≈ $3~4)

- **5-A 분기 통합** (소유: `src/core/brain.py` 질의 처리부, `query_graph.py`, `src/api/routes/chat.py`): E5 — 특성화 테스트 먼저, 스트림을 QueryGraph 콜백으로. 대시보드 SSE 이벤트 형식 유지 테스트.
- **5-B 신뢰도** (소유: `src/core/confidence.py`, `query_graph.py:_node_assess_confidence`): E6 — 1단계 route·신뢰도 분포와 유형별 성공/실패를 이용해 임계값 보정. 보정 근거 표를 문서에. 과적합 방지를 위해 유형별 시험지를 보정용/검증용으로 나눠(문항 id 해시로 고정 분할) 검증용에서 확인.
- **5-C 난이도 라우터** (소유: 새 `src/core/router.py`, `query_graph.py` 라우팅, `react_agent.py`): E7 — 홉 수 판정기(규칙 기반 우선, LLM 판정은 쓰더라도 결과 캐시·기록), ReAct function calling 전환, 단계·토큰 예산, 그림자 모드 플래그 `agents.react_shadow_mode`.

게이트: 전체 테스트, 172+보충 문항 1회에서 route 분포에 direct 외 경로가 존재하고 ④다단계 문항의 라우터 판정 정확도(수동 라벨 대비) 기록, 신뢰도 HIGH 구간의 실패율이 LOW 구간보다 낮음.

### 6단계 — ReAct 비교와 켜기 판정 (API ≈ $4~6)

- ④다단계 시험지(+②관계 일부)에서 (a) 파이프라인만, (b) 그림자 모드 기록, (c) ReAct ON을 각 3회 측정. ReAct 도구 관찰은 judge 컨텍스트에 포함(카드).
- 판정: (c)가 (a) 대비 근거성·정답 지표(토큰 F1·수치 정확도·규칙 일치율)에서 §2 기준 이상 개선되고 다른 유형 회귀가 없으면 `agents.use_react_agent` 기본 ON, 아니면 OFF 유지하고 원인(도구 선택 실수·단계 초과·인용 누락 등 로그 분류)을 기록. 비용·지연 증가도 함께 적는다.
- E8 수치 검증기의 제거·수정 건수를 유형별로 보고.

게이트: 판정 결과와 근거가 결정 문서에 있음.

## 7. 마무리

- 문서 갱신: `docs/experiments/evidence_pipeline_2026-09.md`(단계별 결과), `docs/plans/evidence-react-ontology-decisions-2026-09.md`(결정표), `docs/portfolio/amore_architecture_evidence.md`의 §3.3·§5.5·§7·§10에 **[2026-09 사후]** 표시로 반영(공모전 이후 작업임을 명시, 당시 동작처럼 읽히는 문장 금지), CLAUDE.md·README·AGENTS.md의 아키텍처 서술을 새 구조에 맞게.
- 전체 테스트 최종 1회.
- 최종 보고에는 다음만: 단계별 커밋 해시와 게이트 통과 근거, 유형별 기준선 → 최종 수치 표(평균·범위·판정), 규칙 발화율·ReAct 판정, 제거한 코드 목록, 누적 API 비용, 남긴 항목과 이유.

## 8. 하지 말 것

push, 배포, Railway 설정 변경, 외부 크롤링, 기존 baseline 파일 수정·삭제, `laneige_golden_v2.jsonl` 수정, §0 승인 범위 밖의 `data/` 수정, 비밀키·스프레드시트 ID 출력, `* 2.py` 파일 수정·삭제, 평가 결과를 유리하게 다듬기.
