# 개선 사이클 계획서 — v4.1 이후 (2026-08-30 작성)

> 새 세션(Claude Opus 5)에 인수인계하기 위한 실행 계획서.
> 실행 프롬프트: `docs/plans/opus5-handoff-prompt-2026-08-30.md`
> 반드시 함께 읽을 것: `PORTFOLIO_FACTS.md`, `docs/experiments/eval_cycle3_2026-08-30.md`, `CLAUDE.md`

---

## 0. 현재 상태 스냅샷

### 확정된 측정 기준 (baseline v4.1-2026-08-30, 160문항, 트레이스 오염 수정 후 클린 측정)

| 지표 | 값 | 상태 |
|---|---|---|
| Overall (가중) | 0.507 | v1.0(0.428) 대비 +18% |
| Groundedness (LLM Judge) | 0.650 | 강점으로 전환됨 (v1.0 0.365) |
| Answer Relevance | 0.786 | 안정적 강점 |
| L4 Type Consistency | ~0.95 | 강점 |
| L3 KG Hits@8 | 0.725 | 양호 |
| L1 Entity F1 | 0.487 | **개선 대상 (mapping fail 72문항)** |
| L1 Concept F1 | 0.334 | 게이트 0.3 기준 fail 64문항 |
| L2 Context Recall | 0.222 | **개선 대상 (코퍼스 부족)** |
| L3 Edge F1 | 0.211 | **최다 fail 태그 (125문항)** |
| 평균 지연 | 8.3초 | reranker off로 확보 |
| 게이트 통과 | 3/160 | L3 edge·grounding 게이트가 지배 |

- 회귀 비교 기준: `eval/baselines/v4.1-2026-08-30/` (v1.0~v3.0은 **트레이스 오염으로 하향 왜곡** — 레이어 지표 절대 비교 금지, eval_cycle3 문서 참조)
- 전체 테스트: 5,207 passed / 0 failed. KG: 2,700 트리플. RAG 코퍼스: ChromaDB `amore_docs` 271청크.
- 활성 구성: reranker OFF(ablation 근거), 프롬프트 v1b(근거 인용 강화), 컨텍스트 상한 rag_chunks=8.

### 작업 이력 요약 (2026-08-30, 4개 사이클 완료)

사실 검증 감사 → P0~P2 수정 → baseline v1.0 → 골든셋 재매핑·의미 게이트·프롬프트 A/B·ablation(reranker off) → v2.0/v3.0 → **동시 실행 트레이스 오염 발견·수정**(중대) → 어휘 정합·제품명 엣지·게이트 재보정 → v4.0/v4.1. 전 과정 `docs/experiments/` 3개 문서와 커밋 메시지에 기록됨.

---

## 1. 작업 환경·운영 규칙 (모든 사이클 공통)

### 환경
- Python: `.venv/bin/python` 사용 (홈브류 python3는 pytest 없음). 테스트: `.venv/bin/python -m pytest tests/ -q --no-cov`
- API 키: `.env`의 OPENAI_API_KEY (유효 확인됨). 셸에서 `source .env` 금지(CRLF로 깨짐) — 파이썬에서 `from dotenv import load_dotenv; load_dotenv('.env')` 패턴 사용.
- 시크릿: `.env` 값을 출력·커밋·문서화하지 말 것. 문서 저장 전 `sk-|tvly-` 패턴 grep.

### 평가 실행 표준 (조건 고정 — baseline 비교 유효성의 전제)
```bash
FF_AGENTS_USE_EXTERNAL_SIGNALS=false LLM_TEMPERATURE=0.1 .venv/bin/python -c "
from dotenv import load_dotenv; load_dotenv('.env')
from eval.cli import main
main(['run', '--dataset', 'eval/data/golden/laneige_golden_v2.jsonl',
      '--out', 'eval_output/<이름>', '--judge', 'llm', '--judge-model', 'gpt-4.1-mini',
      '--semantic-similarity', '--concurrency', '4'])
"
```
- 외부신호 OFF는 Tavily 월 쿼터(1,000건) 보호 목적 — 절대 켜지 말 것 (챗 1회당 Tavily 6쿼리).
- 비용 규율: **코드 수정 전 반드시 v4.1 리포트 로컬 분석으로 원인 확정** → 수정 → 단위테스트 → (선택) subset_nokg 30문항 스모크 → 사이클당 160문항 본실행 1~2회. 진단 없는 재실행 금지.
- 회귀 비교·저장:
```bash
.venv/bin/python -m eval.cli compare --baseline-name v4.1-2026-08-30 --report <report.json>
.venv/bin/python -m eval.cli set-baseline --name v<다음>-2026-08-30 --report <report.json>
```

### 지표·데이터 무결성 (게이밍 금지)
- 골든셋·게이트·지표를 결과가 좋아 보이도록 고치지 말 것. 데이터/지표의 **구조 결함을 입증**한 경우에만 수정하고, 근거(진단 수치)와 함께 `docs/experiments/`에 기록 (선례: 청크 ID 재매핑, semantic 게이트, concept 게이트 0.3).
- 개선 주장은 반드시 compare 출력 또는 리포트 수치로 뒷받침.

### git·커밋
- 논리 단위별 conventional commit(한국어 본문), 말미에 `Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>`.
- pre-commit 특성: ①베이스라인 JSON은 EOF-fixer가 1회 수정 → **재-add 후 재커밋**하면 됨 ②`ruff format`에 .json 파일을 절대 넘기지 말 것(trailing comma 삽입으로 JSON 파손 이력) ③부분 스테이징 + 훅 자동수정이 충돌하면 `pre-commit run --files <files>`로 워크트리를 먼저 고정.
- 사이클 종료 시: 전체 pytest 0 failed 확인 → baseline 커밋 → `git push`.

### 문서화 관행
- 사이클마다 `docs/experiments/eval_cycle<N>_<날짜>.md` (배경→진단→수정→결과표→한계→재현 명령).
- `PORTFOLIO_FACTS.md`에 사이클 요약 섹션 추가, 메모리 파일은 건드리지 않아도 됨.

---

## 2. 작업 패키지 (우선순위순)

### P1. L3 Edge F1 잔여 미스 재진단·개선 (fail 125문항, 0.211)

**배경**: 제품명 슬러그 엣지 도입으로 0.194→0.211에 그침. 이전 진단은 오염 데이터 기반이었으므로 **v4.1 클린 리포트로 재진단이 선행**.

**작업 단계**:
1. 진단 스크립트(패턴은 기존 세션에서 검증됨): v4.1 report.json에서 문항별 gold `kg_edges` vs trace `kg_edges_found`를 대조해 미스를 유형화 — (a) 골드 제품이 rank>10이라 슬러그 엣지 미방출, (b) 슬러그 변형 불일치(예: water_bank vs water_bank_blue), (c) 질의 브랜드가 entity 미추출(P2와 연동), (d) competesWith 특정 경쟁사 불일치(현재 삽입순 3개 — SoS순 정렬 검토), (e) 지표 설계 문제(gold 1~3개 vs found ~12개 set-F1).
2. 유형별 비중이 큰 것부터 수정. 관련 코드: `src/rag/hybrid_retriever.py`의 `_query_knowledge_graph`(metric_edges 블록)·`_product_name_slugs`, `eval/runner.py`의 `_extract_l3_trace`.
3. (e)가 지배적이면 지표 개선 검토: L2와 동일하게 **edge recall**을 병기하고 게이트를 recall 기반으로 전환 — 반드시 근거와 함께 문서화 (`eval/metrics/l3_kg.py`, `eval/metrics/aggregator.py`, `eval/README.md`, `tests/eval/` 동반 수정).

**완료 기준**: L3 Edge(또는 대체 지표) fail 문항이 유의미하게 감소하고, compare에서 다른 지표 회귀 없음. 진단·결정이 experiments 문서에 기록됨.

### P2. L1 Entity 매핑 커버리지 (mapping fail 72문항, 0.487)

**작업 단계**:
1. v4.1 리포트에서 gold `kg_entities` vs `extracted_brands`를 대조해 미추출 브랜드 빈도표 작성 (이전 오염 데이터에서 tirtir·biodance 미추출 정황).
2. 브랜드 사전 확인: `config/brands.json`(+`src/rag/entity_linker.py`의 `_get_merged_brands`) — 누락 브랜드·별칭(한글 표기, 대소문자, 축약형) 추가. KG에 존재하는 브랜드(엔리치먼트로 543 subject)와의 정합도 점검.
3. 골드 표기(`gold.kg_entities`)와 추출 표기의 정규화 불일치면 `eval/metrics/base.py`의 BRAND_ALIASES 확장으로 해소.

**완료 기준**: L1 entity F1 상승(목표 0.6+), mapping fail 문항 감소, 기존 테스트 그린.

### P3. RAG 코퍼스 확충 (L2 recall 0.222, grounding fail 83문항)

**배경**: 클린 측정에서도 낮음 — 코퍼스 271청크가 골든셋 질의 범위를 못 덮음. groundedness 잔여 실패와 같은 뿌리.

**작업 단계**:
1. **인덱싱 파이프라인 먼저 파악**: ChromaDB `amore_docs`가 어떤 소스에서 어떻게 구축되는지 확인 (`src/rag/retriever.py`의 initialize/인덱싱 경로, `_chunk_index`, 소스 문서 위치). 재인덱싱 실행 방법을 확보한 뒤에 문서를 추가할 것.
2. v4.1 리포트에서 L2 미스가 큰 도메인(time 0.265 최약, market, multi_hop)의 질문을 표본으로, 부족한 문서 유형을 특정.
3. 확충 소스는 **이미 리포에 있는 자료 우선**: `docs/analysis/rag-ontology-kg-deep-analysis.md`, `docs/reports/`, IR 자료(`docs/ir/`), 지표 가이드 원문 등. 새 문서를 창작할 경우 사실 검증 가능한 내용만.
4. 임베딩 비용: 신규 청크 수 × text-embedding-3-small — 저렴하지만 재인덱싱 횟수는 최소화.
5. 골드 `doc_chunk_ids`는 기존 7종 유지 — 새 청크를 골드로 추가하려면 재매핑 스크립트(`scripts/remap_golden_chunk_ids.py`) 관례를 따라 문서화.

**완료 기준**: L2 recall·groundedness 동반 상승, grounding fail 감소. 인덱싱 절차가 재현 가능하게 스크립트/문서화됨.

### P4. 시스템 정리 (독립 항목 — P1~P3 사이 틈에 처리 가능)

| 항목 | 내용 | 완료 기준 |
|---|---|---|
| v4 프롬프트 통합 | `src/core/response_pipeline.py`의 하드코딩 `SYSTEM_PROMPT`를 `prompts/registry.py` 경유로 통합 (v1b 근거 인용 규칙이 v4 경로에도 적용되게) | 대시보드 챗이 registry 프롬프트 사용, 관련 테스트 통과 |
| kg_backup 경로 버그 | `src/tools/utilities/kg_backup.py`가 `src/data/knowledge_graph.json`(오경로)을 봄 — 실제는 `data/` | `python -m src.tools.utilities.kg_backup backup` 성공 |
| 소셜 수집기 4종 | tiktok/instagram/youtube/reddit_collector: 파이프라인 연결 or 삭제 결정. 권장: 삭제(전용 테스트 포함)하고 README 표에서 제거 — Reddit은 external_signal_collector 내장 구현이 이미 담당 | 결정·실행·문서 일치 |
| RSS 파싱 복구 | external_signal_collector의 RSS 6종이 "not well-formed" 실패 — 피드 URL 갱신 또는 lenient 파싱 | 최소 3개 소스에서 기사 수집 확인 |
| README 커버리지 갱신 | 72.76%는 2026-02 수치 — `--cov`로 재측정 후 갱신 (전체 실행 수 분 소요) | README 수치 = 실측 |
| ablation 클린 재실행(선택) | 오염 수정 후 6구성 재실행으로 reranker-off 결정 재확인 | ablation 문서 갱신 |

---

## 3. 사이클 운영 루프 (표준)

각 작업 패키지는 이 루프로 진행한다:

```
① v4.1 리포트 로컬 진단 (무비용) → 원인 유형화·비중 수치화
② 수정 구현 + 단위/회귀 테스트 추가
③ 관련 테스트 스위트 그린 확인
④ (불확실하면) subset 30문항 스모크
⑤ 160문항 본실행 → compare vs v4.1 → 개선 확인
⑥ set-baseline v<다음> → experiments 문서 + PORTFOLIO_FACTS 갱신
⑦ 논리 단위 커밋 → 전체 pytest → push
```

**중단 기준**: OpenAI 401/키 문제, 골든셋 대규모 재설계가 필요한 발견, 판단이 갈리는 파괴적 결정(파일 대량 삭제, 지표 체계 전면 교체)은 작업을 멈추고 사용자에게 보고.
