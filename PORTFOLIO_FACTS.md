# PORTFOLIO_FACTS.md — 사실 검증 결과

> 검증일: 2026-08-30 / 검증 기준 커밋: 2b04aa1 (main)
> 목적: 채용 포트폴리오에 기재 가능한 사실만 추림. 모든 항목에 증거 라벨 부착.
> 라벨: [실행 확인] 명령 실행으로 검증 / [코드 확인] 코드 판독으로 검증 / [문서 주장] 문서에만 존재 / [불명] 확인 불가

---

## Q1-a. 평가 체계 구조

### 골든셋 파일별 문항 수 [실행 확인]
Python으로 JSONL 파싱·집계 (`wc -l` 아닌 실제 JSON 파싱 기준):

| 파일 | 문항 수 |
|------|--------|
| `eval/data/golden/laneige_golden_v1.jsonl` | 40 |
| `eval/data/golden/laneige_golden_v2.jsonl` | **160** |
| `eval/data/golden/subset_nokg.jsonl` | 30 |
| `eval/data/examples/chatbot_eval.jsonl` | 10 |
| `tests/golden/chatbot_golden.jsonl` | 20 (별도 스키마: query/expected_facts/expected_source_types) |
| `tests/golden/report_golden.jsonl` | 3 (리포트용: expected_sections/expected_kpis) |

### 문항 분류 분포 (실데이터 필드 집계) [실행 확인]

**laneige_golden_v2.jsonl (160문항, 메인셋):**
- `requires_kg`: true 130 / false 30
- `domain`: metric 30, product 30, brand 25, market 25, multi_hop 20, edge 15, time 15
- `difficulty`: medium 89, hard 43, easy 28

**laneige_golden_v1.jsonl (40문항):** requires_kg 20/20, domain product 19·metric 8·brand 7·market 4·general 2, difficulty easy 32·medium 8

**subset_nokg.jsonl (30문항, 전부 requires_kg=false):** domain metric 15·edge 8·market 6·multi_hop 1, difficulty medium 14·easy 10·hard 6

### 채점 방식 [코드 확인]
2단 구조: **결정적 메트릭(L1~L5) + 선택적 Judge**.

- 결정적 메트릭: `eval/metrics/` — L1 Entity/Concept/Constraint set-F1, L2 Recall@k·Precision@k·MRR, L3 Hits@k·KG Edge F1, L4 Constraint Violation·Type Consistency, L5 Exact Match·Token F1 (eval/README.md:9-16, `eval/metrics/aggregator.py`에서 가중 합산·게이팅)
- Judge 3종 구현체가 모두 실재:
  - **Stub**: 네트워크 없는 휴리스틱 (`eval/judge/stub.py`)
  - **LLM Judge**: LiteLLM 기반, RAGAS 스타일 groundedness/relevance 프롬프트 + 비용 추적 내장 (`eval/judge/llm.py:1-33`, GROUNDEDNESS_PROMPT :40~)
  - **NLI Judge**: cross-encoder/nli-deberta-v3 entailment 점수, 무비용·오프라인 (`eval/judge/nli.py:1-49`)
- 게이팅 임계값: Entity F1<0.5, Recall<0.8, Groundedness<0.7 등 10개 (eval/README.md:122-137)

### 회귀 감지 (regression.py) [코드 확인]
`eval/regression.py:1-88` — 베이스라인 스냅샷(`eval/baselines/{name}/report.json` + 문항별 items/)을 저장해 두고, 새 실행의 **집계 메트릭을 임계값 기반으로 비교**한다. 비교 대상과 허용 폭(`RegressionThresholds`, regression.py:55-88): pass_rate -3%p, avg_overall_score -5%p, 지연 +500ms, 레이어별 L1~L5 F1 각 -5%p (L4 위반율은 +2%p), **비용 +20%**. 초과 시 회귀 플래그 + diff 리포트 생성. CLI로 `run --baseline`, `compare`, `set-baseline` 서브커맨드 지원 (`eval/cli.py:9-27`).

---

## Q1-b. 평가 실행 (1회, 최소 비용 서브셋)

- 사용법 [코드 확인]: `eval/cli.py` — 서브커맨드 `run`(--dataset/--out/--top-k/--judge {stub,nli,llm}/--baseline/--dry-run 등), `compare`, `set-baseline` (eval/cli.py:55-240). 주의: docstring의 `python -m eval run ...`은 오류(`eval/__main__.py` 없음) — 실제로는 `python -m eval.cli run ...`으로 실행해야 함 [실행 확인].
- 실행 명령 [실행 확인]:
  ```
  .venv/bin/python -m eval.cli run --dataset eval/data/golden/subset_nokg.jsonl --out <scratch>/eval_out
  ```
  (requires_kg=false 30문항, judge는 기본 stub — LLM judge 추가 비용 없음. 에이전트 응답 생성에만 GPT-4.1-mini + text-embedding-3-small 호출)
- 소요 시간 [실행 확인]: **11분 4초** (30문항, 문항당 평균 지연 21,850ms — 대부분 LLM 재시도 대기)
- 형식적 결과 [실행 확인]: Total 30 / Passed 0 (0.0%) / Avg Score 0.254 / 실패 태그: L5_wrong_answer 30, L2_doc_retrieval_fail 24, L1_concept_fail 10, L1_mapping_fail 5
- **⚠️ 이 수치는 품질 지표로 무효** — report.json 검증 결과 **30문항 중 24문항의 최종 답변이 LLM 호출 실패 폴백 문구**("죄송합니다. 현재 응답을 생성할 수 없습니다...")였고, RAG 스니펫은 30문항 전량 공백. **[원인 확정 — 후속 검증]** 별도 테스트 실행에서 OpenAI 서버가 정상 응답하며 **401 `invalid_api_key`("Incorrect API key provided")를 반환**하는 것을 확인 → 근본 원인은 네트워크가 아니라 **`.env`의 OPENAI_API_KEY가 유효하지 않음** (키 값은 본 문서에 기록하지 않음). 유효한 키로 교체 후 재실행하면 정상 측정 가능. 사용자 지시("재시도로 비용 낭비 금지")에 따라 재실행하지 않고 중단.
- 따라서 **골든셋 통과율/점수는 [불명]** — 이번 세션에서 유효 측정 불가. 단, **평가 하네스 자체의 end-to-end 동작은 [실행 확인]**: 데이터셋 30문항 로드 → 에이전트 호출 → L1~L5 트레이스 캡처 → 게이팅 채점 → report.json + summary.md 생성까지 전 파이프라인이 오류 없이 완주됨.

---

## Q2. 오답 개선 사례 — "평가 → 개선" 루프의 흔적

**존재함. 커밋·문서로 명확히 추적 가능** [실행 확인(git log/show) + 코드 확인]

1. **Ablation 인프라 구축** — 커밋 `24a8fbf` "feat(eval): ablation 스터디 + 프롬프트 실험 파이프라인": `eval/ablation.py`(339줄) 신규 — 피처 플래그 조합 6종(full / no-kg / no-ontology / no-reranker / no-query-rewrite / no-fusion)으로 컴포넌트별 기여도 측정 (eval/ablation.py:38-56). 프롬프트 변형 v0~v4(`prompts/agents/variants/`) + 실행 스크립트 포함.
2. **Ablation 결과를 반영한 개선** — 커밋 `9474e82` "feat(rag): ablation 스터디 반영 품질 개선": 커밋 메시지에 근거 명시 — "CompactBuilder 분기 제거 (제목만 전달되어 Answer F1 저하 유발 — ablation P0-a)", kg_enricher에 SoS·HHI·rankedIn 메트릭 트리플 추출 추가(P1), entity_linker 개념 링킹 추가.
3. **하네스 자체 결함 발견·수정 기록** — `docs/experiments/prompt_exp_2026-08.md`: dry-run 중 골든셋 160문항 중 51건이 스키마 불일치로 탈락하는 문제 발견 → `eval/schemas.py` 3건 수정(judge_model None 크래시=커밋 `5b90cd4`, domain Literal 확장, constraints dict 허용) 과정이 표로 기록됨.
4. **결과 보고서** — 커밋 `2b04aa1`: `docs/reports/ablation_study_report.docx`(579KB) + 차트 6종(overall_score, delta, radar, latency, f1, failures).

---

## Q3. 가드레일

### 실제 작동하는 거절/제한 사례 3건 [코드 확인]
런타임 진입점은 `brain.process_query(_stream)` (src/api/routes/chat.py:183, :254)이므로 실효 가드레일은 `query_graph.py`/`brain.py` 경유분이다.

**사례 1 — 프롬프트 인젝션/시스템 명령 차단 (하드 거절):** NFKC 정규화 후 64개 인젝션 패턴 + 4개 시스템 명령 패턴 정규식 매칭 (src/core/prompt_guard.py:158-166) → `query_graph.py:81-88`에서 `is_blocked=True`로 즉시 거절 응답(`PromptGuard.get_rejection_message`, 거절 문구 prompt_guard.py:235, :249) 반환하고 라우팅 종료 (query_graph.py:278). 스트리밍 경로 동일 (brain.py:578-594).

**사례 2 — 저신뢰 시 명확화 요청:** 컨텍스트 점수 < 1.5 → `ConfidenceLevel.UNKNOWN` (src/core/confidence.py:35-37, :124) → `query_graph.py:302`에서 clarification 분기 → 응답 "질문을 더 구체적으로 해주시겠어요? 예를 들어 특정 브랜드나 카테고리, 분석 지표(SoS, HHI 등)를 포함해주세요." (query_graph.py:249, confidence_score=0.2).

**사례 3 — 출력측 시스템 정보 유출 차단:** 응답 텍스트에 내부 도구 시그니처("namespace functions" 등) 포함 시 "시스템 정보는 공개할 수 없습니다. LANEIGE 마켓 분석에 관해 질문해 주세요." 로 대체 (src/core/prompt_guard.py:216, 배선 query_graph.py:266 / brain.py:699).

**주의(포트폴리오에 과장 금지):**
- 오프도메인 질문은 **거절이 아니라 경고 후 계속 처리** — `out_of_scope_warning`은 `is_safe=True`로 반환되고 처리 계속 (prompt_guard.py:193, query_graph.py:92 주석 "out_of_scope 경고 시에도 처리는 계속"). "전문 영역이 아닙니다" 거절 문구는 런타임 도달 불가.
- 환각 감지기(`hallucination_detector.py`)는 **로깅만 하고 응답을 바꾸지 않음** (response_pipeline.py:150-176 — 경고 로그 후 원문 그대로 반환).
- `src/rag/templates.py:268-276` `apply_guardrails()`는 루프 내부가 `pass`인 **no-op**인데 hybrid_chatbot_agent.py:591에서 호출됨.

### Self-RAG 게이트 [코드 확인]
**실재하며 런타임 배선됨, 단 범위 제한적.** `HybridRetriever.should_retrieve()` (src/rag/hybrid_retriever.py:358) — 정규식 기반(LLM 아님) SKIP_PATTERNS(인사/도움말, :272-280)·RETRIEVE_PATTERNS(브랜드/지표, :283-292)로 검색 필요성 판단, `retrieve()` 진입부에서 호출되어 skip 시 검색 생략 (:411-421). 반성(reflection) 절반은 `relevance_grader.py:20`의 LLM 기반 관련성 채점 + 1회 재작성 재시도 (hybrid_retriever.py:510-533). **단, 대시보드 기본 v4 경로는 `retrieve_unified()`→OWL 전략으로 위임되어 이 게이트를 우회** (hybrid_retriever.py:605-611). 게이트는 `HybridChatbotAgent.chat()` 경로(hybrid_chatbot_agent.py:292)에서 작동.

### 7-type 출처 표시 [코드 확인]
`src/agents/source_provider.py:17` docstring "Supports 7 source types" — **7개 추출 블록은 실재**하나 enum/상수 정의는 없고 인라인 문자열이며, 실제 방출되는 type 값은 **10종**(crawled_data, knowledge_graph, ontology_inference, rag_document, category_hierarchy, external_news, social_media, rss_feed, external_source, ai_model — source_provider.py:68~224). `external_source` 타입은 표시 포매터에 분기가 없어 인용 번호가 건너뛰어지는 결함 있음(:264-376). 포트폴리오에는 "7-type"보다 "다중 소스 유형 출처 표시"로 기재 권장.

---

## Q4. RAG 실체

**결론 한 문장:** 지금 상태에서 검색은 **dense-only 벡터 검색**이다 — OpenAI `text-embedding-3-small` 임베딩으로 ChromaDB `amore_docs` 컬렉션(persist: data/chroma, 5.1MB 실데이터 존재)을 질의하는 경로가 필수(키워드 폴백 없음, 실패 시 RuntimeError)이며, BM25/RRF 코드는 완비돼 있으나 `rank_bm25` 패키지가 .venv에도 requirements.txt에도 없어 **완전히 비활성**이다. [코드 확인 + 실행 확인]

근거:
- 벡터 검색 필수: `src/rag/retriever.py:984-992` — collection/openai_client 없으면 `RuntimeError("Vector search is required but not available")`, 폴백 없음. ChromaDB 인스턴스화 `retriever.py:741` (`chromadb.PersistentClient`), OpenAI 임베딩 `retriever.py:430, 780-781`. sentence-transformers는 검색 임베딩이 아니라 CrossEncoder 리랭커에만 사용 (src/rag/reranker.py:96,130).
- BM25 비활성: `retriever.py:41-45` `from rank_bm25 import BM25Okapi` ImportError → `BM25_AVAILABLE=False` → `:996` 분기에서 `bm25_results=[]`, RRF 미수행. `.venv`에 rank_bm25 미설치·requirements.txt에 부재 [실행 확인]. `hybrid_retriever.py:571`의 `bm25_available` 메타데이터는 hasattr 기반이라 항상 True로 **오보고**.
- 검색 실패 시: 예외가 삼켜져 빈 컨텍스트로 LLM이 답변 (retrieval_strategy.py:407-410, hybrid_retriever.py:574-576) — 사용자에게 실패 신호 없음.

### 피처 플래그 → 분기 추적 [코드 확인]
해석 우선순위: **ENV(`FF_{SECTION}_{KEY}`) > config/feature_flags.json > 하드코딩 기본값** (src/infrastructure/feature_flags.py:80-91).

| 플래그 (현재값) | 상태 | 제어 분기 |
|---|---|---|
| `retriever.use_owl_strategy` (true) | **live** | container.py:161-185 / brain.py:313-329 — true면 `OWLRetrievalStrategy` 주입, false면 legacy 변환 경로 |
| `cache.use_sqlite_embedding_cache` (false) | **live** | retriever.py:447-457 — false=InMemory(1000개, 재시작 시 소실), true=SQLite |
| `prompts.use_centralized_prompts` (true) | **live** | context_builder.py:723-729 (아래) |
| `retriever.use_unified_retriever` (true) | **dead** — 읽는 코드 없음 | 없음 |
| `reasoner.use_unified_reasoner` (true) | **dead** | 없음 |
| `reasoner.use_owl_reasoner` (true) | **dead** | 없음 (OWL은 use_owl_strategy로만 게이트) |
| `ontology.use_ontology_kg` (true) | **dead** | 없음 |
| `agents.use_decomposed_chatbot` (true) | **dead** | 없음 (챗봇 구현체 단일) |

추가로 JSON에 없지만 live인 플래그: `use_reranker`(기본 true, hybrid_retriever.py:506), `use_query_rewriter`(hybrid_chatbot_agent.py:254), `use_confidence_fusion`, `use_external_signals`(external_signal_manager.py:79-82).

### 프롬프트 위치 [코드 확인]
- `use_centralized_prompts=true`(현재): `prompts/registry.py:110-119`가 **`prompts/agents/chatbot_system.txt`** 로드 + 가드레일/날짜 치환 (registry.py:78-127). 호출처는 hybrid_chatbot_agent.py:478, hybrid_insight_agent.py:420 두 곳.
- `false`일 때: `src/rag/context_builder.py:738-752`의 인라인 Python 문자열("당신은 Amazon 베스트셀러 순위 분석 전문가입니다...") — 내용은 centralized 버전과 거의 동일.
- **주의**: 대시보드가 실제 쓰는 v4 경로(`/api/v4/chat/stream`)는 이 플래그와 무관하게 `src/core/response_pipeline.py:44`의 하드코딩 `SYSTEM_PROMPT`를 사용. 또한 `prompts/chat_system.txt`·`query_router.txt`·`insight_generation.txt`는 어떤 .py에서도 참조되지 않는 **orphan 파일**.
- **주의**: `/api/v3/chat`은 존재하지 않는 엔드포인트(문서 잔재). 실제 라우트는 `POST /api/chat`, `/api/v4/chat`, `/api/v4/chat/stream` (src/api/routes/chat.py:39, 146, 219).

---

## Q5. 소셜·외부 데이터 파이프라인 연결 여부

[코드 확인]

| 수집기 | 판정 | 호출 경로 |
|---|---|---|
| `external_signal_collector.py` | **연결됨** (자동+수동) | ①챗 런타임: hybrid_chatbot_agent.py:303 → external_signal_manager.py:53→95 (매 질의, 피처 플래그 게이트) ②배치: batch_workflow.py:730 → hybrid_insight_agent.py:948 ③수동 API: routes/signals.py:27, export.py:230 |
| `tavily_search.py` | **연결됨** (간접) | external_signal_collector.py:372-374에서만 생성 |
| `public_data_collector.py` | **연결됨** (배치·수동) | market_intelligence.py:126 → batch 경로. 단 **스케줄러 자동 경로는 brain.py:1297의 깨진 import**(`src.tools.market_intelligence` — 실제는 `src.tools.intelligence.market_intelligence`)로 ModuleNotFoundError가 침묵 처리되어 미작동 |
| `google_trends_collector.py` | **연결됨** (배치) | hybrid_insight_agent.py:1074 (trendspyg/pytrends 옵션 의존) |
| `reddit_collector.py` | **미연결** | `__init__.py` re-export만. 실제 Reddit 수집은 external_signal_collector.py:499-525의 자체 JSON API 구현이 담당 |
| `tiktok_collector.py` | **미연결** | re-export만. external_signal_collector의 `fetch_tiktok_trends`(:590-607)는 항상 `[]` 반환 스텁이며 호출처도 없음 |
| `instagram_collector.py` | **미연결** | re-export만, 인스턴스화 0건 |
| `youtube_collector.py` | **미연결** | re-export만. docs/plans/NEWS_REPORT_SEARCH_IMPLEMENTATION_PLAN.md:16에도 "구현됨/미통합"으로 자인 |

→ **판정: TikTok/Instagram/YouTube/Reddit 전용 수집기 4종은 "코드는 있으나 파이프라인 미연결".** 포트폴리오에 "소셜 미디어 수집" 기재 시 "뉴스(Tavily/GNews/RSS)·Reddit JSON·공공데이터·Google Trends 연동, SNS 전용 수집기는 모듈 구현 단계"로 한정할 것.

**Grok: 리포 전체 grep 0건** [실행 확인] (`grep -ri grok src/ scripts/ config/ docs/` — 무일치. LLM 스택은 litellm 단일).

---

## Q6. 수치 검증

### 테스트 [실행 확인]
```
.venv/bin/python -m pytest tests/ -q --no-cov
→ 2 failed, 5180 passed, 19 skipped in 419.20s (6분 59초)
```
- 총 수집 5,201개, 통과율 **99.96%** (5,180/5,182 실행분).
- 실패 2건: `tests/eval/test_judges.py::TestStubJudge::test_stats_tracking`(이벤트루프 RuntimeError — docs/experiments/prompt_exp_2026-08.md에 이미 "환경 요인"으로 기록된 알려진 실패), `tests/integration/test_pipeline_integration.py::TestDecisionToResponseFlow::test_response_pipeline_receives_decision`.
- README의 "5,195개 100% 통과(0 failed)"는 현재 시점 기준 부정확 → 기재 시 "5,180+ 통과 / 99.9%+"로 수정 필요.
- 커버리지 72.76%: 이번 실행에서 `--no-cov`로 생략 → **[문서 주장]** 유지 (README:290).

### OWL/추론 규칙 수 [실행 확인]
```
from src.ontology.rules import ALL_BUSINESS_RULES → 37개 (전부 InferenceRule)
```
파일별: alert 3, growth 7, ir 6, market 6, price 7, sentiment 8. README "29+ 규칙"은 사실(보수적 표현). **단 이는 rule-based reasoner 규칙이며 OWL DL 공리 개수가 아님** — "OWL 온톨로지 29+ 규칙"이 아니라 "규칙 기반 추론 엔진 37개 비즈니스 규칙"으로 기재해야 정확.

### KG 트리플 "50K+" [실행 확인 → 반증]
`data/knowledge_graph.json`의 `triples` 배열 = **1,028개**. README:62 "50K+ 트리플"은 로컬 실데이터 기준 약 50배 과장. (Railway 볼륨의 프로덕션 KG는 미확인이나, 동기화 스크립트로 받은 로컬본이 1K대인 이상 50K+ 기재 불가.)

### 임베딩 캐시 "API 비용 33%↓" [문서 주장]
근거는 docs/embedding_cache_guide.md:84의 **가상 예시 계산**("9 requests → 6 API calls (33% reduction)")뿐. 실측 로그·통계 없음. 캐시 자체는 실재하며 hit_rate 통계 코드 있음 (src/rag/embedding_cache.py:107, InMemory 1000개 FIFO — 현재 SQLite 캐시는 플래그 off). 기재 시 "캐시 도입(적중 시 임베딩 API 호출 절감)"까지만.

### 리팩토링 "97K→70.7K 라인 (-27%)" [실행 확인 → 스코프 불일치 판정]
git 히스토리 실측 (2026-02-09 커밋 80ececb vs 현재):
- 02-09 전체 .py: 98,908줄/278파일 → "97K"는 **테스트 포함 전체** 집계로 추정
- 02-09 프로덕션(테스트·스크립트 제외): 73,275줄/170파일
- 현재 src/: 75,727줄/214파일
→ "97K→70.7K"는 Before(전체)와 After(src만)의 **비대칭 비교**. 동일 기준 프로덕션 코드는 73.3K→75.7K로 오히려 소폭 증가(기능 추가 병행). 라인 감소 주장 대신 "5,634줄 monolith를 12개 라우트 모듈로 분리(-43%)" 같은 **파일 단위 분해 수치**(REFACTORING_RESULTS.md, god object 분해: business_rules 1540→54줄 등)를 쓰는 것이 방어 가능.

---

## Q7. 아모레퍼시픽 CI 적용

[코드 확인] — 문서와 코드 값이 일치:
- 디자인 문서: docs/AMOREPACIFIC_DESIGN_SYSTEM.md:23-24 — Pacific Blue `#001C58`, Amore Blue `#1F5795`; :59-62 폰트 폴백 체인 `'Arita Dotum' → 'Noto Sans KR' → 시스템`.
- 리포트 생성기: src/tools/exporters/report_generator.py:52-58 —
  ```python
  PACIFIC_BLUE = RGBColor(0, 28, 88)   # #001C58
  AMORE_BLUE   = RGBColor(31, 87, 149) # #1F5795
  GRAY = RGBColor(125,125,125), ACCENT_RED #E53935, ACCENT_GREEN #43A047
  ```
  :61-72 폰트명 상수: `"Arita Dotum KR"`(+Medium/SemiBold/Bold/Light), `"Arita Buri KR"`(본문 세리프 계열).

---

## Q8. 트러블슈팅 자산

### (1) Amazon 크롤러 Top 100 수집 복구 (lazy loading) [코드 확인]
- **문제**: 카테고리당 100개가 아닌 60개만 수집 (README:420-433에 상세 기록).
- **원인**: Amazon 베스트셀러 페이지 lazy loading — 초기 로드 시 30개만 렌더, 스크롤해야 페이지당 50개 완성. 기존 크롤러는 스크롤 없이 파싱해 30×2페이지=60개.
- **해결(코드 실재)**: `_scroll_to_load_all()` — 최대 10회 점진 스크롤하며 `[data-asin]` 카드 수가 50 도달/정체 시 중단 (src/tools/scrapers/amazon_scraper.py:289-309, 호출 :424,:439) + `span.zg-bdg-text` 순위 배지 기반 파싱(:525) + `#zg-right-col` 컨테이너 한정으로 광고 제외(:511-512).

### (2) 대시보드 상태 소실 2건
**(2a) 대시보드 데이터 미표시 (Railway)** [코드 확인 + 문서]
- **문제**: 배포 대시보드가 무한 로딩 — 차트·액션 보드 빈 상태 (docs/analysis/dashboard-data-fix.md, dashboard-loading-analysis.md).
- **원인**: `DATA_PATH="./data/dashboard_data.json"` 상대경로 하드코딩 → Railway 볼륨(`/data/`)에서 파일 미발견 + FileNotFoundError 시 빈 dict 반환으로 프론트가 에러 감지 못함 (dashboard-loading-analysis.md:22-30).
- **해결**: `RESOLVED_DATA_DIR = "/data" if Path("/data").exists() else "./data"` 경로 자동 감지 + 로깅 추가 (dashboard-data-fix.md의 diff, 커밋 7ab97c4·56fa042·c88f9ef).

**(2b) KG 데이터 소실 — last-writer-wins** [실행 확인(git show c244cea)]
- **문제**: 크롤링이 갱신한 knowledge_graph.json을 장시간 상주한 서버가 stale 메모리 상태로 덮어씀 (커밋 메시지: "실제 발생 확인: 크롤 저장 11초 후 서버가 구버전으로 덮어씀").
- **원인**: 서버(FastAPI)와 daily_crawl이 같은 파일에 각자 전체 저장 수행.
- **해결**: 정식 기록자를 daily_crawl로 단일화, 서버 상주 KG 인스턴스 6곳(brain.py, container.py, bootstrap.py, get_knowledge_graph 싱글턴, base_hybrid_agent, hybrid_retriever, telegram_bot)을 `auto_save=False` 읽기 전용으로 전환 (커밋 c244cea).

---

## 최종 표: 포트폴리오 기재 가능 수치

| 포트폴리오 기재 가능 수치 | 값 | 증거 라벨 |
|---|---|---|
| 골든셋 평가 문항 (메인 v2) | 160문항 (domain 7종, difficulty 3단계, requires_kg 태깅) | [실행 확인] |
| 골든셋 전체 (v1+v2+서브셋 등) | 240+ 문항 | [실행 확인] |
| 평가 레이어 | L1~L5 (쿼리이해→검색→KG→온톨로지→답변) + Judge 3종(Stub/LLM/NLI) | [코드 확인] |
| 회귀 감지 | 베이스라인 대비 12개 지표 임계 비교 (pass_rate -3%p, 비용 +20% 등) | [코드 확인] |
| Ablation 구성 | 6종 (full/no-kg/no-ontology/no-reranker/no-query-rewrite/no-fusion) | [코드 확인] |
| 평가→개선 사이클 | ablation P0-a 결과로 CompactBuilder 제거 등 (커밋 9474e82) | [실행 확인] |
| 테스트 수 | 5,201개 수집 / 5,180 통과 (99.96%) / 6분 59초 | [실행 확인] |
| 테스트 커버리지 | 72.76% (branch coverage) | [문서 주장] |
| 추론 규칙 | 37개 비즈니스 규칙 (rule-based reasoner, 6개 도메인 파일) | [실행 확인] |
| KG 트리플 (로컬 실데이터) | 1,028개 | [실행 확인] |
| 크롤링 규모 | 5 카테고리 × Top 100 = 500 제품/일 (lazy-loading 대응 코드 실재) | [코드 확인] |
| dashboard_api 분해 | 5,634줄 → 12개 라우트 모듈 (README·REFACTORING_RESULTS 기록) | [문서 주장] |
| 가드레일 | 인젝션 64+4 패턴 차단, 저신뢰 명확화, 출력측 시스템정보 차단 | [코드 확인] |
| CI 디자인 시스템 | #001C58/#1F5795 + Arita Dotum/Buri, 코드-문서 일치 | [코드 확인] |
| 프롬프트 실험 인프라 | 시스템 프롬프트 v0~v4 변형 + 실험 프로토콜 문서 | [코드 확인] |
| 평가 하네스 e2e 동작 | 30문항 로드→채점→report.json/summary.md 생성 완주 | [실행 확인] |

## 기재 금지 / 수정 필요 주장

1. ~~"KG 50K+ 트리플"~~ (README:62) — 실측 1,028개. **기재 금지** 또는 실측치로 교체.
2. ~~"테스트 5,195개 100% 통과 (0 failed)"~~ (README:288-289) — 실측 5,180 통과/2 실패. "5,180+ 통과(99.9%+)"로 수정.
3. ~~"코드 97K→70.7K 라인 -27% 감량"~~ — Before(전체 .py)와 After(src만)의 스코프 불일치 비교. 동일 기준으로는 감소 없음. **모듈 분해 수치로 대체**.
4. ~~"임베딩 캐시로 API 비용 33% 절감"~~ — 문서의 가상 예시일 뿐 실측 없음. "임베딩 캐시 도입"까지만.
5. ~~"BM25/RRF 하이브리드 검색 운영"~~ (README:191) — 코드는 있으나 rank_bm25 미설치로 비활성. "dense 벡터 검색 + CrossEncoder 리랭킹 운영, BM25/RRF 구현(활성화 대기)"으로 한정.
6. ~~"7-type 출처 표시"~~ — 실제 방출 타입 10종, "7"은 docstring 관습. "다중 소스 유형 출처 표시(크롤링/KG/추론/RAG/외부신호 등)"로 기재.
7. ~~"TikTok/Instagram/YouTube 소셜 수집 운영"~~ (README:160-168) — 전용 수집기 4종 파이프라인 미연결. "모듈 구현 완료, 통합 예정"으로 한정.
8. ~~"POST /api/v3/chat"~~ (README:208, CLAUDE.md) — 존재하지 않는 엔드포인트. 실제는 /api/chat, /api/v4/chat(+/stream).
9. "환각 감지로 응답 차단" 류 표현 금지 — 감지기는 로깅만 수행, 응답 미변경.
10. "피처 플래그로 전 컴포넌트 제어" — 8개 중 5개 dead flag. live 플래그(owl_strategy, 캐시, 프롬프트, reranker 등)만 언급.
11. "OWL 온톨로지 29+ 규칙 추론" — 37개는 rule-based 규칙이며 OWL 공리 아님. "규칙 기반 추론 37개 + OWL 스키마(owlready2)"로 분리 기재.
12. ~~골든셋 평가 수치 기재 금지(무효 키)~~ → **해소됨** (2026-08-30 키 교체 후 160문항 실측 완료 — 하단 "실측 Baseline" 섹션의 기재 가이드를 따를 것. 단 L2 recall·answer token-F1은 여전히 구조 결함으로 인용 금지).
13. **ablation 스터디의 `no-kg`/`no-ontology` arm 수치 인용 금지** — 해당 arm이 설정하는 피처 플래그(`FF_ONTOLOGY_USE_ONTOLOGY_KG`, `FF_REASONER_USE_*`)를 읽는 코드가 당시 존재하지 않아 **두 arm은 사실상 no-op**(full과 동일 구성)이었음 [코드 확인]. 2026-08-30 플래그 배선 수정 후 재실행한 결과만 인용 가능. (`no-reranker`/`no-query-rewrite`/`no-fusion` arm의 플래그는 당시에도 live였음.)

## 부록 — 2026-08-30 수정 반영 (검증 이후 상태 변화)

본 문서의 검증 결과를 바탕으로 같은 날 P0~P2 수정을 적용했다. 아래 항목은 **수정 후 더 이상 유효하지 않은 지적**이므로, 이 문서를 읽을 때 참고:

| 원래 지적 | 수정 내용 | 현재 상태 |
|---|---|---|
| BM25/RRF 비활성 (rank_bm25 미설치) | requirements.txt 추가 + .venv 설치 | **활성** (BM25_AVAILABLE=True 확인) |
| Self-RAG 게이트 v4 경로 우회 | `retrieve_unified()` 진입부에 게이트 적용 + 테스트 3건 | **v4 포함 전 경로 적용** |
| ablation no-kg/no-ontology no-op | 플래그를 retrieve/OWL 전략 초크포인트에 배선 + 게이팅 테스트 3건 | **실효** (단, 과거 결과는 여전히 인용 금지) |
| brain.py:1297 깨진 import (스케줄러 시장정보 침묵 실패) | `src.tools.intelligence.market_intelligence`로 수정 | **복구** |
| chat_workflow 시그니처 불일치 (v1 경로 사망) | positional 호출 + `set_data_context` 경유로 수정, Protocol 정합화 | **복구** (테스트 9/9) |
| 환각 감지 로깅만 수행 | 미근거 시 confidence×0.6 + `grounding_warning` 필드 노출 | **응답 반영** |
| `external_source` 인용 번호 누락 | 포매터 폴백 분기 추가 | **수정** |
| dead flag 5종 | 3종(ontology/reasoner)은 배선, 2종(unified_retriever/decomposed_chatbot)은 제거 | **정리 완료** |
| orphan 프롬프트 3종 | 삭제 + prompts/AGENTS.md 실구조 기준 재작성 | **정리 완료** |
| `python -m eval` 실행 불가 | `eval/__main__.py` 추가 | **동작** (dry-run 확인) |
| 테스트 2건 실패 | 이벤트루프 방식 교체 + 환경변수 격리(monkeypatch) | **전체 통과** (5,199 passed / 7 skipped, 0 failed) |
| README 과장 수치 (50K 트리플, 100% 통과, -27% 등) | 실측값·한정 표현으로 전면 교정 | **정합** |

**여전히 남은 항목 (사용자 액션 필요):**
1. ~~OPENAI_API_KEY 교체~~ → **완료** (2026-08-30, ping으로 유효성 확인)
2. ~~골든셋 평가 + baseline 저장~~ → **완료** (아래 "실측 baseline" 섹션)
3. ablation 재실행 (no-kg/no-ontology arm이 이제 실효하므로 보고서 수치 갱신).
4. 소셜 수집기 4종(TikTok/IG/YT/Reddit 전용 모듈): 파이프라인 연결 또는 제거 결정.
5. RSS 소스 6종 파싱 실패 복구 (외부 피드 포맷 변경 대응).
6. v4 경로 시스템 프롬프트의 PromptRegistry 통합 (현재 3벌 존재 → 2벌로 줄었으나 v4는 여전히 하드코딩).
7. **골든셋 v2의 `gold.doc_chunk_ids` 재매핑** — 골드 8종 ID가 실제 ChromaDB 코퍼스(271청크)와 겹침 0 (가상 ID 체계). L2 지표가 유효해지려면 실제 청크 ID로 교체 필요.
8. **`gold.answer` 스타일 정합화 또는 L5 지표 보완** — 골드가 1~2문장 정의문인데 에이전트는 장문 마크다운을 내므로 token-F1이 구조적으로 낮음(전 문항 <0.3). semantic similarity 지표 활용 또는 골드 확장 필요.

---

## 실측 Baseline — 2026-08-30 (v1.0-2026-08-30)

**실행**: `laneige_golden_v2.jsonl` 160문항 전체, gpt-4.1-mini + LLM Judge(gpt-4.1-mini), top-k 8, concurrency 4, 외부신호 OFF(`FF_AGENTS_USE_EXTERNAL_SIGNALS=false`, Tavily 쿼터 보호). 소요 약 21분, 평균 지연 12.7초/문항. **폴백 응답 0건 — 160문항 전부 실 응답** [실행 확인]. 스냅샷: `eval/baselines/v1.0-2026-08-30/` (git 추적 대상, 회귀 비교 기준점).

### 레이어별 실측 (평균)

| 지표 | 값 | 해석 |
|---|---|---|
| L4 Type Consistency | **0.956** | 강점 — 온톨로지 타입 정합성 높음 |
| L5 Answer Relevance (Judge) | **0.768** (84/160이 ≥0.8) | 강점 — 질문 적합성 양호 |
| L3 KG Hits@8 | 0.650 | 보통 — KG 엔티티 히트 |
| L5 Groundedness (Judge) | 0.365 (23/160만 ≥0.7) | **약점 — 컨텍스트 근거 부족(개선 1순위)** |
| L1 Entity Link F1 | 0.357 | 약점 — 엔티티 추출/링킹 |
| L3 KG Edge F1 | 0.250 | 약점 — 관계(엣지) 검색 |
| L2 Context Recall | 0.138 | **해석 불가** — 골드 청크 ID가 코퍼스와 불일치(구조 결함, 남은 항목 7) |
| L1 Concept Map F1 | 0.127 | 약점/부분 구조 — 개념 어휘 체계 불일치 의심 |
| L5 Answer F1 (token) | 0.100 (전 문항 <0.3) | **해석 불가** — 골드 답변 스타일 불일치(남은 항목 8) |
| **Overall (가중)** | **0.428** | |
| 게이트 통과율 | 0% (160 fail) | 10개 게이트 중 하나라도 위반 시 fail — L5 token-F1 게이트(<0.5)가 전 문항에서 걸림 |

### 분해 (overall 기준)

- domain: edge 0.546 > brand 0.480 > product 0.466 > multi_hop 0.418 > metric 0.403 > market 0.395 > **time 0.265 (최약)**
- difficulty: easy 0.467 > medium 0.428 > hard 0.403 / requires_kg 유무 차이 없음 (0.423 vs 0.429)

### 포트폴리오 기재 가이드

- **기재 가능**: "160문항 골든셋 L1~L5 자동 평가 체계 구축·실측, baseline 스냅샷 기반 회귀 감지 운영. 실측에서 강점(타입 정합성 0.96, 응답 적합성 0.77)과 개선 과제(근거성 0.37, 엔티티 링킹 0.36)를 정량 진단" — **측정→진단→개선 사이클의 증거**로 사용.
- **기재 금지**: "통과율 0%"를 맥락 없이 쓰는 것(엄격 게이팅+지표 보정 이슈 미설명 시 오해), 그리고 L2/answer-F1 수치를 품질 지표로 인용하는 것(구조 결함).

## 개선 사이클 결과 — 2026-08-30 (측정→진단→개선→재측정 1회전 완주)

v1.0 baseline의 진단(근거성 최약, L2/F1 구조 결함)을 바탕으로 같은 날 개선 사이클을 완주했다. 전 과정 [실행 확인].

### 수행한 개선

1. **골든셋 구조 결함 해소**: doc_chunk_ids 재매핑(137문항) + 의미 유사도 게이트(다국어 모델, 임계 0.65) — L2·L5가 처음으로 유효 측정 가능
2. **프롬프트 A/B (n=30, 조건 고정)**: 근거 인용 강화(v1b)가 groundedness 0.286→0.326(+14% 상대), 부작용 없음 → 채택. 기록: `docs/experiments/prompt_exp_2026-08.md`
3. **Ablation 재실행 (6구성, 플래그 실효 확인)**: **reranker(LLM 관련성 채점)가 groundedness를 0.414→0.237로 악화시키고 지연 +7.4초** — 컨텍스트를 얇게 만드는 순손실 컴포넌트로 판정, 기본 비활성화. 기록: `docs/experiments/ablation_2026-08-30.md`

### v1.0 → v2.0 (160문항 재측정, 폴백 0건)

| 지표 | v1.0 | v2.0 | Δ |
|---|---|---|---|
| Groundedness | 0.365 | **0.392** (≥0.7: 23→28) | +0.027 |
| 평균 지연 | 12.7s | **8.1s** | **-36%** |
| Overall | 0.428 | 0.442 | +0.014 |
| L2 Recall | (측정 불가) | 0.175 | 최초 유효 측정 |
| Relevance | 0.768 | 0.767 | 유지 |
| L3 KG Hits | 0.650 | 0.575 | -0.075 (회귀 도구가 [minor]로 자동 감지) |
| 게이트 통과 | 0 | 2 | +2 |

- **회귀 감지 실증**: `compare --baseline-name v1.0-2026-08-30`이 L3 하락을 [minor] 회귀로, 통과 전환 2건을 Fixed로 자동 리포트 — 도구가 실전에서 작동함을 확인.
- 주의: v1.0↔v2.0은 데이터셋 재매핑·게이트 변경이 겹쳐 있어 순수 모델 개선분과 지표 개선분이 혼재. **이후 회귀 비교의 기준은 v2.0** (`eval/baselines/v2.0-2026-08-30/`).
- 다음 병목(측정 근거): L2 recall 0.175·groundedness 0.39 → **RAG 코퍼스 확충(현 271청크)이 최우선**, L1 concept(0.13)·L3 edge(0.25)는 어휘/포맷 정합화 필요.

## 개선 사이클 2 — 2026-08-30 (L2·L1·L3 병목 해소, v3.0)

v2.0 진단에서 지목한 병목 3개를 로컬 원인 분석 → 수정 → 160문항 재측정으로 검증. 상세: `docs/experiments/eval_cycle2_2026-08-30.md`. 전 과정 [실행 확인].

핵심 발견 (또 다른 "구현만 있고 미배선" 2건): ① `extract_concepts()`(개념 추출) 호출처 0건 → 배선, ② `KGEnricher`(hasSoS·competesWith 트리플 추출) 호출처 0건 → daily crawl 배선 + 백필로 **KG 1,028→2,700 트리플**. ③ 컨텍스트 상한이 3청크라 top-8 검색이 무의미했던 병목 해소(3→8).

**v2.0 → v3.0**: Groundedness **0.392→0.528 (+35%, ≥0.7 문항 28→49)**, L2 recall +16%, relevance 0.79, overall 0.455, 지연 유지. 단 **L3 edge F1은 0.212→0.132 회귀**(compare 도구가 [minor]로 자동 감지) — 엣지 표면화는 성공했으나 골드 기대 경쟁사와 노출 순서 불일치 + set-F1 지표 설계 한계. 다음 사이클 1순위 과제로 기록.

이후 회귀 기준: `eval/baselines/v3.0-2026-08-30/`.

## 개선 사이클 3 — 2026-08-30 (측정 인프라 결함 발견, v4.1)

**이 사이클의 핵심은 "측정이 틀려 있었다"는 발견이다.** HHI 문항 트레이스에 sos 개념이 찍힌 스팟 체크를 단서로, 동시 실행 시 공유 상태 경쟁으로 **문항 트레이스가 다른 문항의 컨텍스트를 캡처**하던 결함을 확인·수정(커밋 5b64536). v1.0~v3.0의 트레이스 지표는 전부 하향 왜곡이었음이 확정. 상세: `docs/experiments/eval_cycle3_2026-08-30.md`. [실행 확인]

- 오염 수정만으로(동일 코드): L1 concept +130%, L1 entity +59%, L3 hits +35%, groundedness 0.53→0.63
- 추가 정합화(generic 오탐 제거·제품명 엣지·게이트 재보정) 후 **v4.1: groundedness 0.650, overall 0.507**
- 하루 누적(v1.0→v4.1): overall +18%, groundedness +78%, 지연 -35% — 단 레이어 지표 절대 비교는 클린 측정(v4.0 이후)만 유효
- **포트폴리오 관점**: "평가 인프라 자체의 신뢰성을 검증하고 회귀 테스트로 고정했다"는 스토리 — 측정→개선 루프의 성숙도를 보여주는 가장 강한 증거 중 하나
- 이후 회귀 기준: `eval/baselines/v4.1-2026-08-30/`

## 개선 사이클 4 — 2026-08-30 (채점 크래시 발견 + 엣지 지표 재설계, v5.0)

사이클 3과 같은 계열의 **측정 인프라 결함을 한 번 더** 잡아냈다. 상세:
`docs/experiments/eval_cycle4_2026-08-30.md`. 전 과정 [실행 확인]

- **채점 크래시 7문항(4.4%)**: 골든셋의 dict 형태 constraint가 `set()`에 들어가
  TypeError를 내면서 해당 문항의 채점 파이프라인 전체가 중단, 응답이 정상
  생성됐음에도 L1~L5 전부 0점 + judge 미수행 상태였다. 7문항 중 6문항이 time
  도메인 — **time이 최약 도메인(0.265)이던 원인의 상당 부분이 시스템 품질이
  아니라 채점 크래시**였다. 수정 후 7문항 모두 실측(overall 0.41~0.62).
- **L3 Edge F1의 판별력 부재를 수치로 입증**: 골드는 문항당 1~3개(중앙값 1),
  시스템 방출은 최대 12개(중앙값 10) → 완벽 회수해도 F1 0.18, F1 0.5 게이트는
  총 방출 3개 이하일 때만 도달 가능(requires_kg 130문항 중 125문항 상시 fail).
  `kg_edge_recall`·`kg_edge_precision`을 신설해 게이트를 recall로 옮기고 F1은
  연속성을 위해 계속 보고. overall 공식은 v4.1과 동일하게 유지해 비교 유효성 확보.
- **골드 엣지의 46%(164건 중 75건)가 KG에 부재** — 엣지 recall의 구조적 상한은
  약 0.54다. 골드를 시스템 출력에 맞춰 고치는 대신 상한을 명시하는 쪽을 택했다.
- 시스템 개선: 대문자 subject 미조회(ownedBy 11건), 12개 상한이 ownedBy·rankedIn을
  밀어내던 선택 순서, 제품명→브랜드 역링크, 브랜드 substring 오탐(shelf→e.l.f.).
- **기각한 변경도 기록**: 브랜드 미추출 시 LANEIGE 기본 주입은 엣지 recall +0.034를
  주지만 precision이 0.377→0.074로 붕괴 — 골드의 암묵적 관례에 맞추는 것에 가까워
  채택하지 않았다.

### v4.1 → v5.0 (160문항, 조건 고정)

| 지표 | v4.1 | v5.0 |
|---|---|---|
| Overall (가중) | 0.507 | **0.533** |
| 게이트 통과 | 3 | 7 |
| L1 Entity F1 | 0.487 | **0.606** |
| L3 Hits@8 | 0.725 | 0.775 |
| L3 Edge Recall | — | 0.577 (신설) |
| L3 Edge Precision | — | 0.258 (신설) |
| L5 Groundedness | 0.650 | 0.638 |
| 평균 지연 | 8.29s | 7.67s |

fail 태그: L3_edge_fail 125→67, L1_mapping_fail 72→50, L4 위반 7→0.
`compare` 결과 회귀 0건 / Fixed 5건 / 신규 실패 1건.

**기재 시 주의(개선분의 출처 분리)**: 크래시 7문항을 제외한 동일 153문항 기준으로는
overall 0.530→0.535, **groundedness는 0.650→0.634로 소폭 하락**했다. 즉 overall
상승분의 대부분은 채점 크래시 수정이고, 레이어 지표 개선(L1 entity +0.119,
L3 hits +0.05)은 별개로 실재한다. groundedness 하락은 LLM judge 편차 범위와
겹치므로 단정하지 않고 다음 사이클 관찰 대상으로 둔다.

이후 회귀 기준: `eval/baselines/v5.0-2026-08-30/`.

### 함께 처리한 시스템 정리 (P4)
- kg_backup 경로 버그 수정 → CLI 백업 정상 동작 (이전엔 `src/data/`를 조회)
- v4 경로 시스템 프롬프트를 PromptRegistry로 통합 (프롬프트 3벌 → 1벌)
- RSS 12개 피드 전수 실측 후 3종 URL 갱신·5종 제거 + 브라우저 UA fetch →
  **7개 소스 중 5개에서 기사 수집 확인**

## 개선 사이클 5 — 2026-08-30 (RAG 코퍼스 침묵 누락, v6.0)

"코퍼스가 작다"는 가설을 검증하다 **코퍼스의 88%가 색인조차 되지 않은 상태**를
찾아냈다. 상세: `docs/experiments/eval_cycle5_2026-08-30.md`. 전 과정 [실행 확인]

- **IR 분기보고서 3종 1,971청크가 검색 불가였다**: 문서 로드·청킹은 2,242청크를
  만드는데 ChromaDB에는 271청크만 있었다. 원인은 `if collection.count() == 0`으로
  최초 1회만 색인하던 조건 — IR 문서가 나중에 등록돼 영원히 색인 대상이 되지
  못했다. 예외도 경고도 없는 침묵 누락. 증분 색인으로 수정(271 → 2,242).
- **L2의 진짜 병목은 코퍼스 크기가 아니라 골드 라벨 입도**: 160문항이 가리키는
  서로 다른 청크는 **7개**뿐이고, 65문항이 한 문서(38청크)의 291자짜리 문단
  하나를 정답 근거로 요구한다(사이클 2 재매핑에서 문서 수준 가상 ID를 대표 청크
  1개로 1:1 치환한 결과). 같은 트레이스로 재면 청크 단위 recall 0.102 vs
  **출처 문서 단위 0.555** — 검색기는 대개 맞는 문서를 가져오고 있었다.
  `context_recall_at_k_doc`을 신설해 게이트를 문서 단위로 옮기고, 청크 단위는
  연속성을 위해 계속 보고한다.

### v5.0 → v6.0 (160문항, 조건 고정)

| 지표 | v5.0 | v6.0 |
|---|---|---|
| Overall (가중) | 0.533 | 0.536 |
| 게이트 통과 | 7 | 9 |
| L2 Context Recall (청크) | 0.231 | 0.234 |
| L2 Context Recall (문서) | — | 0.619 (신설) |
| L5 Groundedness | 0.638 | 0.643 |
| 평균 지연 | 7.67s | **8.25s** |

L1·L3 지표는 소수점까지 동일 — 이번 사이클이 검색 코퍼스만 건드렸다는 증거다.

**정직한 결론**: 코퍼스를 8배로 늘렸는데 골든셋 지표는 사실상 그대로다.
160문항의 검색 컨텍스트 870청크 중 **IR 청크는 0건** — 영문 재무보고서와 한국어
시장·지표 질문이 임베딩 공간에서 경합하지 않기 때문이다. 오염도 이득도 없고,
**골든셋이 측정하지 않는 능력을 복원한 것**이다. 복원 자체는 확인했다:
`AMOREPACIFIC 1Q25 revenue and operating profit` 질의는 top-5가 전부 IR 청크.
다만 `아모레퍼시픽 2025년 2분기 실적은?`(한국어)은 IR에 도달하지 못한다 —
교차언어 검색 갭이 새 과제로 남는다. 지연 +0.58초는 코퍼스 8배의 대가이며,
골든셋 점수를 위해 IR을 다시 빼는 것은 벤치마크에 맞춰 제품 기능을 삭제하는
것이므로 하지 않았다.

이후 회귀 기준: `eval/baselines/v6.0-2026-08-30/`.

## 개선 사이클 6 — 2026-08-30 (교차언어 갭의 진짜 원인 + L2 라벨 재설계, v7.0)

사이클 5가 남긴 "한국어 질의가 영문 IR 청크에 도달하지 못한다"를 파고든 결과,
**원인은 임베딩 모델이 아니라 영어 전용 CrossEncoder 리랭커**였다. 상세:
`docs/experiments/eval_cycle6_2026-08-30.md`. 전 과정 [실행 확인]

- **단계별 분해로 원인 특정**: 한국어 IR 질의 10개의 top-8 중 IR 청크 비중이
  벡터검색만 0.600 → BM25 융합 0.500 → CrossEncoder 적용 0.263 → 현행 전체 0.138로
  떨어졌다. `text-embedding-3-small`은 한국어→영문 매칭을 잘 하고 있었고,
  `cross-encoder/ms-marco-MiniLM-L-6-v2`(영어 전용)가 코퍼스의 88%인 영문 청크를
  체계적으로 강등시키고 있었다. **다국어 임베딩 모델 교체와 전체 재색인(2,242청크)은
  불필요하다고 판정** — 비용이 큰 파괴적 변경을 수치로 기각한 사례.
- 이 손실은 IR에 국한되지 않았다: 골든셋 137문항의 골드 문서 recall@8도
  CrossEncoder 적용 시 0.766 → 0.620. 사이클 1 ablation이 껐던 것은 다른 리랭커
  (hybrid_retriever의 LLM 관련성 채점기)였고, 이 CrossEncoder는 그 사각지대에서
  계속 돌고 있었다.
- **질의 확장이 효과 0인 채로 비용만 쓰고 있었다**: 확장 질의별 결과를 순차 append 후
  자르는 구조라 첫 질의(원본) 결과만 top-k를 채웠다. RRF 융합으로 교체하고, 확장
  프롬프트가 반대 언어 변형을 반드시 포함하도록 바꿨다.
- 수정 후 한국어 IR 질의: IR 청크 비중 0.138 → **0.725**, 1건 이상 회수 4/10 → **10/10**.

### 덤으로 잡힌 것 — 코퍼스의 84%가 base64 이미지 덩어리였다

베이스라인 커밋 중 secret 스캐너가 IR 청크의 고엔트로피 문자열을 잡은 것이 단서였다.
IR 분기보고서 3종은 PDF→마크다운 변환물이라 이미지가 data URI로 통째로 박혀 있고
**분량이 파일의 97%**다(AP_2Q25_EN.md: 510,358자 중 494,960자). 그대로 청킹한 결과
**2,242청크 중 1,877청크(84%)가 base64 덩어리**였다 — 즉 사이클 5가 "복원"한 IR
1,971청크 중 실제 본문은 87청크뿐이었다. 문서 로드 시 제거하도록 고치고 기존 잡음
청크를 삭제·재색인해 **코퍼스 2,242 → 358청크**(한국어 271 + IR 본문 87).
지연이 추가로 0.53초 줄었고, 컨텍스트에 들어간 잡음 청크는 1건 → 0건이 됐다.
사이클 5의 "IR이 코퍼스의 88%"는 이 잡음을 포함한 수치였다 — 정제 후 24%다.

### 골든셋 L2 라벨 재설계 — 개선이 아니라 측정 변경

사이클 2 재매핑이 문서 수준 가상 ID를 대표 청크 1개로 치환한 결과를 바로잡아,
개념을 "그 개념이 서술된 절 전체의 청크 집합"으로 매핑했다
(`scripts/remap_golden_chunk_groups.py`). `context_recall_at_k_concept` 신설,
L2 게이트 이전. **동일 트레이스에 라벨만 갈아 끼우면 청크 recall은 0.234 → 0.221로
오히려 내려간다** — 점수가 오르도록 고른 라벨이 아니라는 증거다.
골드가 바뀌었으므로 v6.0 이전 baseline과의 L2 비교는 단절된다.

### v6.0 → v7.1 (160문항, 조건 고정)

v7.0은 리랭커·RRF·라벨 재설계만 적용한 중간 측정, v7.1은 base64 정리까지 더한 최종.

| 지표 | v6.0 | v7.0 (중간) | **v7.1 (기준)** |
|---|---|---|---|
| Overall (가중) | 0.536 | 0.560 | **0.562** |
| 게이트 통과 | 9 | 17 | **17** |
| L2 Context Recall (concept) | — | 0.589 | **0.603** (신설) |
| L2 MRR | 0.209 | 0.545 | **0.537** |
| L5 Groundedness | 0.643 | 0.681 | 0.678 |
| L1 Entity / L3 Hits | 0.606 / 0.775 | 동일 | 동일 |
| 평균 지연 | 8.25s | 7.82s | **7.29s (-12%)** |
| 코퍼스 청크 | 2,242 | 2,242 | **358** |

동일 라벨 기준으로 분리하면 개념 recall 0.457 → 0.589, overall 0.541 → 0.560이
시스템 개선분이고, 라벨 변경분은 overall +0.005에 그친다. L1·L3가 소수점까지
동일한 것은 이번 사이클이 문서 검색만 건드렸다는 증거다.
groundedness는 v7.0 → v7.1에서 0.681 → 0.678로 소폭 내렸다(judge 편차 범위).
`compare --baseline-name v6.0`: 회귀 0건 / Fixed 11건 / 신규 실패 3건.

### 함께 처리
- 미연결 소셜 수집기 4종(1,531줄) + 전용 테스트 삭제, instaloader·yt-dlp 의존성 제거
- IR 도메인 골든셋 문항(10~12개)은 **설계만 하고 보류** — 문항 추가는 overall·
  pass_rate의 분모를 바꿔 기존 baseline 비교를 끊는다. L2 라벨 단절과 겹치면
  분리 불가이므로 다음 사이클에서 단독 실행한다.

이후 회귀 기준: `eval/baselines/v7.1-2026-08-30/`.

## 한 줄 총평

이 리포에서 가장 강력한 '기획 역량 증거'는 **측정→진단→개선의 폐쇄 루프를 실제로 완주한 ablation 스터디**다: 컴포넌트별 기여도를 정량 측정하는 인프라(eval/ablation.py, 6구성)를 직접 만들고, 그 결과(P0-a "제목만 전달되어 Answer F1 저하")를 특정해 코드 개선 커밋(9474e82)으로 연결했으며, 과정에서 발견한 평가 하네스 자체의 결함 3건까지 문서(docs/experiments/prompt_exp_2026-08.md)로 남겼다 — 수치를 부풀리는 대신 실험으로 시스템을 교정한 기록이라는 점이 채용 관점에서 가장 설득력 있다.
