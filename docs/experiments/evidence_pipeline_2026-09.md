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
