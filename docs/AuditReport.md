# AMORE Pacific RAG-KG Hybrid Agent: End-to-End Integration Audit Report

**Audit Date**: 2026-01-27
**Auditor Role**: RAG/IR + Ontology/SemanticWeb + Graph + MLOps/SRE + Web Crawling/Apify + Security/Compliance + LLM App Architect
**Status**: STATIC AUDIT (No Execution - Pre-Approval)

---

## A. EXECUTIVE SUMMARY (핵심 결론 5줄)

1. **E2E 데이터 흐름 검증됨**: Apify/Playwright → Storage → KG/RAG → Report/Chatbot 파이프라인이 코드 레벨에서 연결되어 있으나, **ID 추적(ASIN→KG→Citation)의 명시적 provenance chain이 불완전**
2. **Apify 통합 양호**: Actor 호출/폴백/브랜드 검증 패턴 구현, 단 **webhook 서명검증 미구현**, **run_id 기반 E2E 추적 부재**
3. **KG/Ontology 성숙도 양호**: 50K 트리플 지원, smart eviction, 23개 RelationType, OWL 온톨로지 존재. **SHACL 제약 검증 미구현**
4. **Report/Chatbot Citation 시스템 존재**: 7-type 출처 추출, 참고자료 섹션 생성. **문서 ID/chunk ID 기반 정밀 인용 부족**
5. **골든셋/회귀테스트 부재**: 챗봇 QA 및 리포트 검증용 evaluation harness 미구현. **재현성 검증 불가**

---

## B. END-TO-END ARCHITECTURE (데이터 흐름 + 모듈/파일 매핑 + ID 추적)

### B.1 통합 아키텍처 다이어그램

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              TRIGGER LAYER                                       │
├─────────────────────────────────────────────────────────────────────────────────┤
│  [1] Manual: POST /api/crawl/start (API Key)                                    │
│  [2] Scheduler: UnifiedBrain (src/core/brain.py) - 22:00 KST Daily              │
│  ID: session_id (UUID) generated at workflow start                              │
└───────────────────────────────────┬─────────────────────────────────────────────┘
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              COLLECTION LAYER                                    │
├─────────────────────────────────────────────────────────────────────────────────┤
│  ApifyAmazonScraper (src/tools/apify_amazon_scraper.py)                        │
│  ├── Actor: junglee/amazon-bestsellers                                          │
│  ├── Fallback: AmazonScraper (src/tools/amazon_scraper.py) - Playwright        │
│  ├── Brand Recognition: ≥90% threshold for Apify acceptance                     │
│  └── Output: List[RankRecord] per category (5 categories × 100 products)        │
│                                                                                  │
│  YouTube/RSS Collectors (src/tools/youtube_collector.py, external_signal_*.py) │
│  └── Actor: streamers/youtube-scraper, RSS feeds                                │
│                                                                                  │
│  ID Tracking: ASIN (Amazon Standard Identification Number) per product          │
│  Missing: run_id correlation between Apify Actor run and local workflow         │
└───────────────────────────────────┬─────────────────────────────────────────────┘
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                            NORMALIZATION & STORAGE                               │
├─────────────────────────────────────────────────────────────────────────────────┤
│  CrawlerAgent (src/agents/crawler_agent.py)                                     │
│  ├── Deduplication: By ASIN                                                     │
│  ├── Brand Normalization: BRAND_NORMALIZATION dict + LLM verification           │
│  └── Output Schema: RankRecord (asin, brand, title, rank, price, category, date)│
│                                                                                  │
│  StorageAgent (src/agents/storage_agent.py)                                     │
│  ├── Google Sheets: Primary backup + human sharing                              │
│  ├── SQLite: Railway production (data/amore_data.db)                            │
│  └── JSON: data/latest_crawl_result.json (dashboard export용)                    │
│                                                                                  │
│  ID Continuity: ASIN preserved through normalization                            │
│  Missing: canonical_url deduplication for non-Amazon sources                    │
└───────────────────────────────────┬─────────────────────────────────────────────┘
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              INDEXING LAYER                                      │
├─────────────────────────────────────────────────────────────────────────────────┤
│  KnowledgeGraph (src/ontology/knowledge_graph.py)                               │
│  ├── load_from_crawl_data(): ASIN → Brand/Category/Product relations            │
│  ├── load_from_metrics_data(): Brand → SoS/HHI/CPI metadata                     │
│  ├── load_category_hierarchy(): Category parent-child relations                 │
│  ├── load_brand_ownership(): Brand → AMOREPACIFIC ownership                     │
│  └── 23 RelationTypes, 50K max triples, smart eviction                          │
│                                                                                  │
│  DocumentRetriever/RAG (src/rag/retriever.py)                                   │
│  ├── Document Types: A(Playbook), B(Intelligence), C(Crisis), D(Metric), E(IR) │
│  ├── Chunking: Basic (no semantic chunking enabled)                             │
│  ├── Vector Search: Disabled (config: vector_search_enabled=false)              │
│  └── Keyword Search: BM25-style with TTL caching (5 min)                        │
│                                                                                  │
│  OntologyReasoner (src/ontology/reasoner.py)                                    │
│  ├── Business Rules: src/ontology/business_rules.py (registered at init)        │
│  ├── OWL File: src/ontology/cosmetics_ontology.owl                              │
│  └── Inference: Rule-based (no SPARQL/Cypher, no SHACL validation)              │
│                                                                                  │
│  ID Mapping:                                                                     │
│  ├── ASIN → KG Subject (Product entity)                                         │
│  ├── Brand Name → KG Subject (Brand entity)                                     │
│  └── Document → doc_path (no persistent doc_id/chunk_id)                        │
└───────────────────────────────────┬─────────────────────────────────────────────┘
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                             RETRIEVAL LAYER                                      │
├─────────────────────────────────────────────────────────────────────────────────┤
│  HybridRetriever (src/rag/hybrid_retriever.py)                                  │
│  ├── EntityExtractor: brands, categories, indicators, sentiments                │
│  ├── Intent Classification: 7 types (diagnosis, trend, crisis, metric, etc.)   │
│  ├── KG Query: get_brand_products(), get_competitors(), get_sentiments()        │
│  ├── Reasoner Inference: infer_with_intent() → InferenceResult[]                │
│  └── RAG Document Search: doc_type_filter based on intent                       │
│                                                                                  │
│  Output: HybridContext {query, entities, ontology_facts, inferences, rag_chunks}│
│                                                                                  │
│  EntityLinker (src/rag/entity_linker.py)                                        │
│  └── Text → OWL URI mapping with confidence scores                              │
│                                                                                  │
│  ID Tracking: Entity names (not URIs) used in context building                  │
│  Missing: Persistent chunk_id for precise citation                              │
└───────────────────────────────────┬─────────────────────────────────────────────┘
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                            OUTPUT LAYER (7A + 7B)                                │
├─────────────────────────────────────────────────────────────────────────────────┤
│  [7A] HybridInsightAgent (src/agents/hybrid_insight_agent.py)                   │
│  ├── Input: metrics_data, crawl_data, HybridContext                             │
│  ├── External Signals: Tavily, RSS, Reddit, YouTube, Google Trends              │
│  ├── Market Intelligence: 4-layer (Macro→Industry→Consumer→Amazon)              │
│  ├── LLM Generation: daily_insight (1200 tokens max)                            │
│  ├── Output: action_items, highlights, warnings, inferences, explanations       │
│  └── References: Numbered citations [1], [2], ... in markdown                   │
│                                                                                  │
│  [7B] HybridChatbotAgent (src/agents/hybrid_chatbot_agent.py)                   │
│  ├── Input: user_message, session_id, HybridContext                             │
│  ├── Query Rewriting: Context-aware (QueryRewriter)                             │
│  ├── External Signals: Tavily + RSS + Reddit (max 8)                            │
│  ├── Response Generation: LLM with temperature 0.7                              │
│  ├── Source Extraction: 7 types (Crawled, KG, Ontology, RAG, Hierarchy, etc.)   │
│  └── Output: response, sources, suggestions, entities, stats                    │
│                                                                                  │
│  Citation System:                                                                │
│  ├── Report: [N] numbered references with source type + date                    │
│  ├── Chatbot: "📚 출처 및 참고자료" section with 7 source types                  │
│  └── Missing: doc_id/chunk_id based precise linking                             │
│                                                                                  │
│  Session/Trace IDs:                                                              │
│  ├── session_id: Per workflow/user session                                      │
│  ├── chat_trace: Via ExecutionTracer (span-based)                               │
│  └── Missing: report_id for versioned report tracking                           │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### B.2 ID 추적 매트릭스 (Entity Traceability)

| Entity Type | Source ID | Normalized ID | KG ID | Report/Chat Reference |
|-------------|-----------|---------------|-------|----------------------|
| Product | Amazon ASIN | ASIN | Subject (ASIN) | "[Crawled Data]" (no ASIN shown) |
| Brand | title parsing | BRAND_NORMALIZATION | Subject (brand_name) | brand_name in text |
| Category | category_id (config) | category_id | Subject/Object | category_name |
| Document | file_path | doc_path | N/A | "[RAG Document] {filename}" |
| Inference | N/A | rule_name | N/A | "[Ontology Rule] {name} ({conf}%)" |
| External Signal | source_url | source_url | N/A | "[Signal] {source}" |

**Gap**: 동일 엔티티가 파이프라인 전체에서 **ASIN/Brand Name**으로 추적 가능하나, **KG→Report/Chatbot 출력에서 ASIN 기반 정밀 인용 없음**

---

## C. ISSUE LIST TABLE (이슈 리스트)

| # | Severity | Evidence | Problem Summary | Impact | Fix Suggestion | Verification Test | Confidence |
|---|----------|----------|-----------------|--------|----------------|-------------------|------------|
| **C.1** | **High** | `apify_amazon_scraper.py:131-144` | Webhook 서명검증 미구현 | Apify webhook spoofing 가능, 위조 데이터 주입 위험 | Webhook signature verification (HMAC-SHA256) 추가 | Mock webhook 호출로 signature 불일치 시 reject 확인 | High |
| **C.2** | **High** | `hybrid_chatbot_agent.py` 전체, `templates.py` | Prompt injection 방어 미흡 | 시스템 프롬프트/API 키 노출, 내부 경로 유출 가능 | Input sanitization + output filtering 레이어 추가 | Injection 시도 문자열 테스트 (`{{system}}`, `ignore above`) | High |
| **C.3** | **High** | 전체 파이프라인 | 골든셋/회귀테스트 부재 | Report/Chatbot 품질 회귀 감지 불가, 재현성 검증 불가 | `tests/golden/` 디렉토리 + JSONL 형식 테스트 데이터 생성 | 골든셋 10개 질의 실행 후 expected vs actual 비교 | High |
| **C.4** | **Med** | `knowledge_graph.py:186-191` | KG auto_load 시 버전/무결성 검증 없음 | 손상된 JSON 로드 시 silent corruption | JSON schema validation + checksum 추가 | 손상 JSON 로드 시 명시적 에러 발생 확인 | High |
| **C.5** | **Med** | `batch_workflow.py:647`, `hybrid_retriever.py` | 크롤링 실패 시 Report/Chatbot graceful degradation 미흡 | 부분 실패 시 stale data 사용 여부 불명확 | Explicit data freshness check + stale warning 표시 | 1개 카테고리 실패 시뮬레이션 후 output 검증 | Med |
| **C.6** | **Med** | `retriever.py`, `hybrid_retriever.py` | Document chunk_id 미생성, 정밀 인용 불가 | 같은 문서의 다른 위치 인용 구분 불가 | UUID-based chunk_id 생성 및 메타데이터 저장 | 동일 문서 다중 청크 검색 시 chunk_id 유니크 확인 | High |
| **C.7** | **Med** | `apify_amazon_scraper.py` | Apify run_id E2E 추적 없음 | Actor 실행 → 결과 데이터 연결 감사 불가 | run_id를 crawl_result 메타데이터에 포함 | run_id로 Apify 콘솔 vs 로컬 데이터 매칭 확인 | High |
| **C.8** | **Med** | `reasoner.py`, `cosmetics_ontology.owl` | SHACL 제약 검증 미구현 | OWL 스키마 위반 데이터 silent 적재 | SHACL shapes 정의 + validation hook 추가 | 스키마 위반 트리플 삽입 시도 후 reject 확인 | Med |
| **C.9** | **Low** | `config/thresholds.json` | API 키/토큰이 환경변수로만 관리 | 키 로테이션 추적 어려움, 실수로 로그 노출 가능 | Secrets manager 연동 + 로그 마스킹 강화 | 로그에서 `sk-`, `apify_api_` 패턴 스캔 | Med |
| **C.10** | **Low** | `hybrid_insight_agent.py:100-128` | Google Trends/YouTube Collector optional import | Feature flag 누락 시 silent failure | Explicit feature toggle + health check endpoint | GOOGLE_TRENDS_AVAILABLE=False 시 명시적 로그 확인 | High |
| **C.11** | **Low** | `batch_workflow.py:507-509` | KG save() 실패 시 에러 핸들링 미흡 | 디스크 full 등에서 데이터 손실 | save() 실패 시 retry + alert 추가 | 디스크 full 시뮬레이션 후 에러 핸들링 확인 | Med |
| **C.12** | **Low** | `hybrid_chatbot_agent.py:126-127` | temperature 0.7 고정 | 동일 질문 재현성 낮음, A/B 테스트 어려움 | temperature를 config 기반으로 조정 가능하게 | temperature=0 설정 후 동일 질문 10회 결과 비교 | High |

---

## D. DEBUGGING PLAN (승인 전 vs 승인 후)

### D.1 승인 전 (정적 검증)

| # | Check Item | Method | Expected | Status |
|---|------------|--------|----------|--------|
| D.1.1 | Apify Actor ID 일치 | `grep "junglee/amazon-bestsellers"` | 1개 파일에서 정확히 발견 | ✅ 확인됨 |
| D.1.2 | KG RelationType 정의 완전성 | `relations.py` 23개 enum 검증 | 모든 사용처에서 정의된 타입만 사용 | ✅ 확인됨 |
| D.1.3 | RAG 문서 경로 일치 | `docs/guides/` 내 4개 파일 존재 확인 | Type D 문서 4개 | 확인 필요 |
| D.1.4 | config/brands.json 스키마 | 필수 필드 (name, segment, country_of_origin) | 모든 브랜드 엔트리 완전 | 확인 필요 |
| D.1.5 | 환경변수 문서화 | CLAUDE.md 내 env vars 목록 | 모든 필수 변수 문서화 | ✅ 확인됨 |

### D.2 승인 후 (동적 검증) - Mini/Normal/Regression Run

#### Stage 1: Mini-Run (단일 입력)

```bash
# 실행 명령 (초안)
python -c "
import asyncio
from src.core.batch_workflow import BatchWorkflow

async def mini_run():
    wf = BatchWorkflow(use_hybrid=True)
    result = await wf.run_daily_workflow(categories=['lip_care'])
    print(json.dumps(result, indent=2, ensure_ascii=False))

    # Chatbot 1 질문
    chat_result = await wf.chat('LANEIGE Lip Care SoS는?')
    print(chat_result)

asyncio.run(mini_run())
"
```

**기대 산출물**:
- `result['steps']['crawl']['result']['total_products']` > 0
- `result['steps']['insight']['result']['daily_insight']` 비어있지 않음
- `chat_result['sources']` 7개 타입 중 3개 이상 포함
- `data/dashboard_data.json` 생성됨

**실패 시 중단 조건**: crawl step 실패 → 전체 중단, 로그 수집

#### Stage 2: Normal-Run (대표 입력 3개, 7일)

```bash
# 실행 명령 (초안)
python -c "
import asyncio
from src.core.batch_workflow import BatchWorkflow

async def normal_run():
    wf = BatchWorkflow(use_hybrid=True)
    categories = ['lip_care', 'skin_care', 'face_powder']
    result = await wf.run_daily_workflow(categories=categories)

    # 챗봇 5개 질문 실행
    questions = [
        'LANEIGE SoS 트렌드는?',
        'COSRX vs LANEIGE 비교',
        'Lip Care 시장 집중도',
        '오늘 랭킹 변동 원인',
        '경쟁사 대응 전략'
    ]
    for q in questions:
        await wf.chat(q)

asyncio.run(normal_run())
"
```

**기대 산출물**:
- 3개 카테고리 각각 products > 0
- KG triples 증가 (`kg_result['total_triples']`)
- 5개 질문 모두 응답 (response 비어있지 않음)
- 응답 시간 < 30초 (p95)

#### Stage 3: Regression-Run (스냅샷 비교)

```bash
# 어제 스냅샷 vs 오늘 스냅샷 비교
# data/dashboard_data_20260126.json vs data/dashboard_data_20260127.json

python scripts/diff_dashboard.py \
  --old data/dashboard_data_20260126.json \
  --new data/dashboard_data_20260127.json \
  --output diff_report.md
```

**기대 산출물**:
- SoS 변동 < 5% (비정상 데이터 감지)
- Top 10 제품 중 7개 이상 동일 (순위 안정성)
- 리포트 daily_insight 섹션 구조 동일

---

## E. REPORT PIPELINE 점검 결과 (핵심)

### E.1 리포트가 참조하는 데이터 소스 증명

| 데이터 소스 | 코드 위치 | 사용 방식 |
|------------|----------|----------|
| metrics_data | `hybrid_insight_agent.py:130-134` | execute() 파라미터로 전달 |
| crawl_data | `hybrid_insight_agent.py:130-134` | KG 업데이트 + context |
| KG facts | `hybrid_insight_agent.py:184` | `_update_knowledge_graph()` 후 조회 |
| RAG chunks | `hybrid_retriever.py` → `HybridContext.rag_chunks` | hybrid_retrieval 결과 |
| External signals | `hybrid_insight_agent.py:119-128` | Tavily/RSS/Reddit/YouTube |

### E.2 시간창/버전 명시 여부

| 항목 | 현재 상태 | 권장 |
|------|----------|------|
| 리포트 생성 시점 | `generated_at` 필드 존재 ✅ | - |
| 데이터 수집 시점 | `data_source.crawled_at` 포함 ✅ | - |
| 시간창 (time window) | 미명시 ❌ | `data_timeframe: {start, end}` 추가 |
| 데이터 버전 | 미명시 ❌ | `data_version: {kg_version, crawl_batch_id}` 추가 |

### E.3 재현성 검증

| 항목 | 현재 상태 | 권장 |
|------|----------|------|
| 동일 입력 → 동일 KPI | 불확실 (LLM temperature 영향) | KPI 계산은 deterministic, insight는 별도 |
| 캐시/스냅샷 저장 | `data/latest_crawl_result.json` 존재 ✅ | 버전별 스냅샷 추가 |
| 추정 vs 사실 분리 | templates.py에 hedging 가이드 존재 ✅ | 명시적 구조 분리 권장 |

### E.4 인용 포함 여부

- **현재**: `## 참고자료` 섹션에 numbered citations 존재
- **Gap**: Document chunk 단위 정밀 인용 없음, 페이지/섹션 레벨 인용만

---

## F. CHATBOT PIPELINE 점검 결과 (핵심)

### F.1 라우팅 규칙 (Intent → Tool)

| Intent | Query Pattern | Tool Selection | 증거 |
|--------|---------------|----------------|------|
| DIAGNOSIS | "왜", "원인", "분석" | Type A Playbook + KG | `router.py`, `hybrid_retriever.py` |
| TREND | "최근", "트렌드", "인기" | Type B Intelligence | `router.py:QueryIntent.TREND` |
| CRISIS | "문제", "대응", "위기" | Type C Response Guide | `router.py:QueryIntent.CRISIS` |
| METRIC | "SoS", "HHI", "지표" | Type D Metric Guide | `router.py:QueryIntent.METRIC` |
| GENERAL | (no keyword) | All documents | Default |

### F.2 Fallback 안전성

| 실패 상황 | 현재 처리 | 권장 |
|----------|----------|------|
| KG 조회 실패 | Silent (빈 결과) | Explicit warning + RAG only fallback |
| RAG 검색 실패 | Fallback search (무필터) ✅ | - |
| LLM 호출 실패 | Exception propagation | Graceful degradation message |
| External signal 실패 | Silent skip ❌ | Explicit "외부 신호 수집 실패" 표시 |

### F.3 재현성 (Reproducibility)

| 항목 | 현재 상태 | 영향 |
|------|----------|------|
| temperature | 0.7 (config 가능) | 동일 질문 결과 변동 |
| 대화 메모리 | 100 turns max | 긴 대화 시 truncation |
| 캐시 | RAG 5분 TTL | 짧은 간격 동일 질문 캐시 hit |

### F.4 보안 (Prompt Injection 방어)

| 위험 | 현재 상태 | 권장 |
|------|----------|------|
| System prompt 노출 | 방어 없음 ❌ | Input filter + output sanitization |
| API 키 노출 | 환경변수만 | 로그 마스킹 강화 |
| 내부 경로 노출 | 가능 ❌ | Path normalization + filter |

---

## G. ABLATION STUDY + EVALUATION HARNESS 설계 (초안)

### G.1 비교군 정의

| Config | RAG | BM25 | KG | Ontology Reasoning |
|--------|-----|------|----|--------------------|
| Baseline | ❌ | ❌ | ❌ | ❌ |
| RAG Only | ✅ | ❌ | ❌ | ❌ |
| RAG+BM25 | ✅ | ✅ | ❌ | ❌ |
| RAG+KG | ✅ | ❌ | ✅ | ❌ |
| Full Hybrid | ✅ | ✅ | ✅ | ✅ |

### G.2 리포트 평가 지표

| Metric | Definition | Measurement |
|--------|------------|-------------|
| KPI Accuracy | 수기 검증 대비 정확도 | 샘플 10개 수기 계산 vs 리포트 KPI |
| Citation Coverage | 주장 당 근거 비율 | 문장 수 / 인용 수 |
| Reproducibility | 동일 입력 동일 결과 | 3회 실행 결과 diff |
| Generation Failure Rate | 생성 실패 비율 | 실패 수 / 총 시도 |
| Generation Time | 생성 소요 시간 | p50, p95 (seconds) |
| Cost per Report | 리포트 당 API 비용 | LLM tokens × price |

### G.3 챗봇 평가 지표

| Metric | Definition | Measurement |
|--------|------------|-------------|
| Faithfulness | 답변 내용이 컨텍스트와 일치 | LLM-as-judge (0-1) |
| Groundedness | 근거 없는 주장 비율 | Hallucination detection |
| Citation Coverage | 답변 내 출처 표시 비율 | 인용 문장 / 전체 문장 |
| Answer Correctness | 골든셋 대비 정확도 | Exact/Partial match |
| p95 Latency | 95분위 응답 시간 | Milliseconds |
| Cost per Query | 질의 당 API 비용 | LLM tokens × price |

### G.4 최소 평가 하네스 스키마

```jsonl
// tests/golden/chatbot_golden.jsonl
{"query": "LANEIGE Lip Care SoS는?", "expected_facts": ["SoS"], "expected_brands": ["LANEIGE"], "expected_categories": ["lip_care"]}
{"query": "COSRX 경쟁력 분석", "expected_facts": ["SoS", "rank"], "expected_brands": ["COSRX"]}

// tests/golden/report_golden.jsonl
{"input_date": "2026-01-27", "expected_sections": ["핵심", "원인 분석", "권장 액션", "참고자료"]}
```

**평가 스크립트 (초안)**:
```python
# scripts/evaluate_golden.py
async def evaluate_chatbot():
    golden = load_jsonl("tests/golden/chatbot_golden.jsonl")
    results = []
    for case in golden:
        response = await chatbot.chat(case["query"])
        results.append({
            "query": case["query"],
            "brands_found": extract_brands(response),
            "facts_found": extract_facts(response),
            "citation_count": count_citations(response),
            "latency_ms": response["stats"]["response_time_ms"]
        })
    return compute_metrics(results, golden)
```

---

## H. IMPROVEMENT ROADMAP + APPROVAL GATES

### H.1 Quick Wins (1-2일)

| # | Task | Impact | Approval Gate |
|---|------|--------|---------------|
| H.1.1 | 로그 마스킹 강화 (API 키 패턴) | Security | `grep -r "sk-\|apify_api_" logs/` = 0 |
| H.1.2 | temperature config 노출 | Reproducibility | config 변경 후 chatbot 동작 확인 |
| H.1.3 | External signal 실패 시 명시적 표시 | UX | 실패 시뮬레이션 후 output 확인 |

### H.2 Mid-Term (1-2주)

| # | Task | Impact | Approval Gate |
|---|------|--------|---------------|
| H.2.1 | Webhook 서명검증 구현 | Security | Mock webhook + invalid signature reject |
| H.2.2 | Document chunk_id 생성 | Citation Precision | 동일 문서 다중 청크 유니크 ID 확인 |
| H.2.3 | 골든셋 10개 생성 + 평가 스크립트 | Quality | `python scripts/evaluate_golden.py` 성공 |
| H.2.4 | Apify run_id E2E 추적 | Observability | run_id로 Actor 콘솔 vs 로컬 매칭 |

### H.3 Long-Term (1-2개월)

| # | Task | Impact | Approval Gate |
|---|------|--------|---------------|
| H.3.1 | SHACL 제약 검증 구현 | Data Quality | 스키마 위반 트리플 reject 확인 |
| H.3.2 | Prompt injection 방어 레이어 | Security | Injection 테스트 10개 모두 방어 |
| H.3.3 | Vector search 활성화 + 평가 | Retrieval Quality | Ablation 결과 비교 |
| H.3.4 | Report versioning + diff tool | Reproducibility | 어제 vs 오늘 diff 자동 생성 |

---

## I. 내가 답해야 할 질문 10개 (필수)

1. **Apify Actor 실행 시 run_id를 어디에 저장하고 있는가?** (현재: 저장 안 함 → 추적 불가)

2. **KG JSON 파일 손상 시 복구 절차는?** (현재: 절차 없음)

3. **동일 질문 10회 실행 시 챗봇 응답 variation은 얼마나 되는가?** (temperature=0.7 영향 측정 필요)

4. **External signal collector 실패 시 리포트/챗봇 출력에 표시되는가?** (현재: silent skip)

5. **Report daily_insight 생성 시 사용된 LLM 토큰 수와 비용 추적이 되는가?** (현재: 불확실)

6. **COSRX가 Korean brand로 정확히 인식되는가?** (config/brands.json 확인 필요)

7. **Prompt injection 테스트 결과는?** (`{{system}}`, `ignore above`, `reveal your instructions` 등)

8. **RAG 문서 Type A/B/C/D/E 각각 몇 개 파일이 있는가?** (docs/ 디렉토리 구조 확인)

9. **KG max_triples=50000 도달 시 eviction 정책이 정상 동작하는가?** (테스트 필요)

10. **리포트 참고자료 섹션의 인용 번호 [1], [2]가 본문 내 참조와 일치하는가?** (수기 검증 필요)

---

**END OF AUDIT REPORT**

*이 감사 보고서는 코드 정적 분석 기반으로 작성되었으며, 실행 기반 검증은 승인 후 진행됩니다.*
