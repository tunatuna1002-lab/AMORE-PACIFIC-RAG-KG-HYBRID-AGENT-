# AMORE RAG-KG Hybrid Agent - 시스템 아키텍처 전체 해설

---

## 1. 프로젝트 정체성

**Amazon US LANEIGE 브랜드 경쟁력 모니터링 자율 AI 시스템**

매일 아마존 베스트셀러 Top 100을 5개 카테고리에서 크롤링하고, Knowledge Graph + RAG + Ontology 추론을 결합하여 시장 인사이트를 생성하며, AI 챗봇으로 실시간 분석을 제공합니다.

---

## 2. 전체 아키텍처 다이어그램

```
┌─────────────────────────────────────────────────────────────────────┐
│                        PRESENTATION LAYER                          │
│  ┌──────────────────────┐  ┌──────────────────────────────────┐    │
│  │  Dashboard (HTML/JS) │  │  Telegram Admin Bot              │    │
│  │  amore_unified_      │  │  /logs /status /crawl /kg /db    │    │
│  │  dashboard_v4.html   │  └──────────────────────────────────┘    │
│  └──────────┬───────────┘                                          │
│             │ REST API                                             │
├─────────────┼──────────────────────────────────────────────────────┤
│             ▼          API LAYER (FastAPI)                         │
│  ┌─────────────────────────────────────────────────────────┐       │
│  │  dashboard_api.py (Entry Point)                         │       │
│  │  ├── /api/health          GET   Healthcheck             │       │
│  │  ├── /api/data            GET   Dashboard JSON          │       │
│  │  ├── /api/v3/chat         POST  AI Chatbot              │       │
│  │  ├── /api/crawl/start     POST  Manual Crawl (API Key)  │       │
│  │  ├── /api/v4/brain/status GET   Scheduler Status        │       │
│  │  ├── /api/export/docx     POST  Async DOCX Export       │       │
│  │  ├── /api/signals/news    GET   Tavily News             │       │
│  │  └── /webhook/telegram    POST  Telegram Webhook        │       │
│  └─────────────────────┬───────────────────────────────────┘       │
├─────────────────────────┼──────────────────────────────────────────┤
│                         ▼                                          │
│              ORCHESTRATION LAYER (src/core/)                       │
│  ┌─────────────────────────────────────────────────────────┐       │
│  │              UnifiedBrain (brain.py)                     │       │
│  │     "자율 스케줄러 + 쿼리 라우터 + 오케스트레이터"         │       │
│  │                                                         │       │
│  │  ┌─────────────┐  ┌──────────────┐  ┌───────────────┐  │       │
│  │  │  Scheduler  │  │ Query Router │  │  Crawl Mgr    │  │       │
│  │  │ (22:00 KST) │  │  복잡도 판단  │  │  상태 관리     │  │       │
│  │  └──────┬──────┘  └──────┬───────┘  └───────┬───────┘  │       │
│  │         │               │                   │          │       │
│  │         ▼               ▼                   ▼          │       │
│  │  ┌─────────┐   ┌──────────────┐   ┌──────────────┐    │       │
│  │  │  Batch  │   │  ReAct Agent │   │  Alert Mgr   │    │       │
│  │  │Workflow │   │ (Self-Reflect)│   │ (Email/TG)   │    │       │
│  │  └────┬────┘   └──────┬───────┘   └──────────────┘    │       │
│  └───────┼───────────────┼─────────────────────────────────┘       │
├──────────┼───────────────┼─────────────────────────────────────────┤
│          ▼               ▼                                         │
│                    AGENT LAYER (src/agents/)                        │
│  ┌──────────────┐ ┌──────────────────┐ ┌───────────────────┐       │
│  │ CrawlerAgent │ │HybridChatbotAgent│ │HybridInsightAgent │       │
│  │  크롤링 실행  │ │ KG+RAG+Ontology │ │  전략 인사이트     │       │
│  └──────┬───────┘ │ 하이브리드 챗봇   │ │  외부신호 융합     │       │
│         │         └────────┬─────────┘ └─────────┬─────────┘       │
│         │                  │                     │                 │
│  ┌──────┴───────┐ ┌───────┴──────┐  ┌───────────┴──────┐          │
│  │ MetricsAgent │ │ StorageAgent │  │ AlertAgent       │          │
│  │ SoS/HHI/CPI │ │ SQLite+Sheet │  │ 순위변동 알림     │          │
│  └──────────────┘ └──────────────┘  └──────────────────┘          │
├────────────────────────────────────────────────────────────────────┤
│              HYBRID RETRIEVAL LAYER                                │
│                                                                    │
│  ┌─────────── RAG ───────────┐  ┌────── Ontology ──────────┐      │
│  │  src/rag/                 │  │  src/ontology/            │      │
│  │                           │  │                           │      │
│  │  HybridRetriever          │  │  KnowledgeGraph           │      │
│  │  ├─ QueryIntent 분류      │  │  ├─ Triple Store (50K)    │      │
│  │  ├─ EntityLinker          │  │  ├─ Subject/Object Index  │      │
│  │  └─ ContextBuilder        │  │  └─ JSON Persistence      │      │
│  │                           │  │                           │      │
│  │  DocumentRetriever        │  │  OntologyReasoner          │      │
│  │  ├─ 14개 참조문서          │  │  ├─ Rule-based Inference  │      │
│  │  ├─ Embedding Cache       │  │  ├─ Business Rules        │      │
│  │  └─ ChromaDB (선택)       │  │  └─ Insight Generation    │      │
│  └───────────────────────────┘  └───────────────────────────┘      │
│                    │                        │                      │
│                    └────────┬───────────────┘                      │
│                             ▼                                      │
│                   ┌─────────────────┐                              │
│                   │  HybridContext   │                              │
│                   │  ├ ontology_facts│                              │
│                   │  ├ inferences    │                              │
│                   │  ├ rag_chunks    │                              │
│                   │  └ combined_ctx  │                              │
│                   └────────┬────────┘                              │
│                            ▼                                       │
│                   ┌─────────────────┐                              │
│                   │  LLM (GPT-4.1)  │                              │
│                   │  via LiteLLM    │                              │
│                   └─────────────────┘                              │
├────────────────────────────────────────────────────────────────────┤
│                    TOOLS LAYER (src/tools/)                         │
│                                                                    │
│  ┌── Crawling ──┐  ┌── Social ──────┐  ┌── Utility ────────┐      │
│  │AmazonScraper │  │TikTok(PW)      │  │MetricCalculator   │      │
│  │ Playwright   │  │Instagram(IL)   │  │SQLiteStorage      │      │
│  │ Stealth      │  │YouTube(yt-dlp) │  │SheetsWriter       │      │
│  │ Browserforge │  │Reddit(JSON)    │  │KGBackup (7일)     │      │
│  └──────────────┘  │GoogleTrends    │  │EmailSender(SMTP)  │      │
│                    └────────────────┘  │TelegramBot        │      │
│                                        │JobQueue           │      │
│  ┌── Public Data ┐                     │ClaimExtractor     │      │
│  │관세청 수출입   │                     │ClaimVerifier      │      │
│  │식약처 화장품   │                     │ConfidenceScorer   │      │
│  └───────────────┘                     └───────────────────┘      │
├────────────────────────────────────────────────────────────────────┤
│                 DOMAIN LAYER (src/domain/) - Layer 1               │
│  ┌── Entities ──────────────┐  ┌── Interfaces (Protocol) ───────┐ │
│  │ Product, RankRecord      │  │ CrawlerAgentProtocol           │ │
│  │ Brand, BrandMetrics      │  │ StorageAgentProtocol           │ │
│  │ Category, Snapshot       │  │ KnowledgeGraphProtocol         │ │
│  │ Relation, RelationType   │  │ RetrieverProtocol              │ │
│  │ InferenceResult          │  │ LLMClientProtocol              │ │
│  │ AmoreAgentError (base)   │  │ ProductRepository              │ │
│  └──────────────────────────┘  └─────────────────────────────────┘ │
├────────────────────────────────────────────────────────────────────┤
│              INFRASTRUCTURE (src/infrastructure/)                   │
│  ┌─────────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │ ConfigManager   │  │ Bootstrap    │  │ JSON/Sheets Repo     │  │
│  │ (thresholds.json│  │ (DI Container│  │ (Persistence)        │  │
│  └─────────────────┘  └──────────────┘  └──────────────────────┘  │
├────────────────────────────────────────────────────────────────────┤
│                     STORAGE LAYER                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────────────┐   │
│  │   SQLite     │  │ Google Sheets│  │ Knowledge Graph JSON   │   │
│  │  (Primary)   │  │  (Backup)    │  │  (50K triples)         │   │
│  │ /data/*.db   │  │              │  │ /data/knowledge_graph  │   │
│  └──────────────┘  └──────────────┘  └────────────────────────┘   │
└────────────────────────────────────────────────────────────────────┘
```

---

## 3. 핵심 데이터 흐름

### 3.1 일일 자동 크롤링 파이프라인

```
22:00 KST (APScheduler)
    │
    ▼
UnifiedBrain.execute_crawl()
    │
    ▼
AmazonScraper (Playwright + Stealth + Browserforge)
    ├── Beauty & Personal Care  (Top 100)
    ├── Skin Care               (Top 100)
    ├── Lip Care                (Top 100)  ← LANEIGE Lip Sleeping Mask
    ├── Lip Makeup              (Top 100)  ← 립스틱/립글로스
    └── Face Powder             (Top 100)
    │
    ▼ 수집 데이터: ASIN, 제목, 브랜드, 순위, 가격, 평점, 리뷰수, 배지, 쿠폰
    │
    ├──▶ SQLite (Source of Truth)
    ├──▶ Google Sheets (Backup)
    │
    ▼
KnowledgeGraph 업데이트
    ├── (Brand) ─HAS_PRODUCT─▶ (ASIN)
    ├── (ASIN) ─BELONGS_TO_CATEGORY─▶ (Category)
    └── (Brand) ─COMPETES_WITH─▶ (Brand)
    │
    ▼
MetricsAgent 계산
    ├── SoS = LANEIGE 제품수 / Top N × 100
    ├── HHI = Σ(점유율²) → 시장집중도
    └── CPI = LANEIGE 평균가 / 카테고리 평균가 × 100
    │
    ▼
HybridInsightAgent
    ├── Ontology Reasoner → 규칙 기반 추론
    ├── RAG 참조문서 → 전략 가이드
    ├── 외부 신호 (Tavily 뉴스, Google Trends)
    └── LLM 합성 → 전략적 인사이트
    │
    ▼
AlertAgent → Email/Telegram 알림 (순위 ±10, SoS 변동)
```

### 3.2 챗봇 쿼리 처리 흐름

```
사용자: "LANEIGE가 경쟁사 대비 어떤 위치에 있나요?"
    │
    ▼
UnifiedBrain.process_query()
    │
    ├── 홉 카운트 판단: `router.py`가 hops=2(경쟁사 조회+지표 조회) ≥ HOP_THRESHOLD(2) → ReAct 후보  # [2026-09 사후] 키워드 판단에서 홉 카운트 라우터로 교체, `agents.use_react_agent`(기본 OFF)가 켜져 있어야 실제 실행
    │
    ▼
ReActAgent (Self-Reflection Loop, max 5회 — `max_iterations` 기본값, [post-2026-09] 정정)
    │
    │  Iteration 1:
    │  ├── Thought: "경쟁사 비교를 위해 KG에서 관계 조회 필요"
    │  ├── Action: kg_neighbors(entity="LANEIGE", predicates=["competesWith"])  # [post-2026-09] 도구명 변경
    │  ├── Observation: [COSRX, Neutrogena, CeraVe...]
    │  └── Reflection: confidence=0.5, needs_improvement=true
    │
    │  Iteration 2:
    │  ├── Thought: "SoS/HHI 지표로 정량적 비교 필요"
    │  ├── Action: get_metrics(category="lip_care")  # [post-2026-09] 도구명 변경 — 아래 참고
    │  ├── Observation: {SoS: 8.3%, HHI: 0.12, rank: 3}
    │  └── Reflection: confidence=0.85, needs_improvement=false
    │
    ▼
HybridChatbotAgent.chat()
    │
    ├── EntityExtractor
    │   ├── Brands: ["LANEIGE", "COSRX", "CeraVe"]
    │   ├── Categories: ["lip_care"]
    │   └── Metrics: ["sos", "rank", "competition"]
    │
    ├── Parallel Retrieval ─────────────────────┐
    │   ├── KnowledgeGraph.query()              │
    │   │   └── Facts: 23 triples              │
    │   │                                       │
    │   ├── OntologyReasoner.infer()            │
    │   │   └── Inferences:                    │
    │   │       "LANEIGE SoS 8.3% → 중위권"    │
    │   │       "HHI 0.12 → 분산된 시장"        │
    │   │                                       │
    │   └── DocumentRetriever.search()          │
    │       └── RAG chunks: 5개 문서 발췌       │
    │           (Embedding Cache Hit Rate: 67%) │
    │                                           │
    ├── HybridContext 구성 ◀────────────────────┘
    │   ├── ontology_facts
    │   ├── inferences
    │   ├── rag_chunks
    │   └── combined_context (unified text)
    │
    ▼
LLM (GPT-4.1-mini, temp=0.4)
    │
    ▼
Response:
    ├── answer: "LANEIGE는 Lip Care 카테고리에서..."
    ├── sources: [{title, url, relevance}]
    ├── confidence: 0.85
    └── metadata: {mode: "react", iterations: 2}
```

---

## 4. 레이어별 상세 설명

### 4.1 Core Layer - 두뇌 (`src/core/`)

| 파일 | 역할 | 핵심 메서드 |
|------|------|------------|
| `brain.py` (1,556줄) | 자율 오케스트레이터 | `process_query()`, `execute_crawl()`, `start_scheduler()` |
| `react_agent.py` (321줄) | ReAct 자기성찰 루프 | `run()` → Thought→Action→Observation→Reflection |
| `batch_workflow.py` | 일일 배치 파이프라인 | CRAWL→STORE→KG→CALCULATE→INSIGHT→EXPORT |
| `crawl_manager.py` | 크롤링 상태 관리 | `start_crawl()`, `get_status()` |
| `scheduler.py` | APScheduler 통합 | 매일 22:00 KST (UTC 13:00) |
| ~~`query_processor.py`~~ | [post-2026-09] 삭제(호출처 0건) | 질의 라우팅·신뢰도 분기는 `query_graph.py`의 QueryGraph가 맡는다 |
| `verification_pipeline.py` | 응답 검증 | Claim 추출 → 사실 검증 |
| `cache.py` | TTL 기반 캐싱 | 5분 TTL, 자동 만료 |

**UnifiedBrain - ReAct 활성화 기준 [2026-09 사후: 키워드 판단 → 홉 카운트 라우터로 교체]**

`src/core/router.py`가 질문을 단계(stage)로 분해해 `hops = len(stages)`를 세고,
`hops >= HOP_THRESHOLD(2)`면 ReAct 후보로 표시한다. 실제로 ReAct 루프를 타는지는
플래그 `agents.use_react_agent`(기본 OFF)에 달려 있다 — OFF면 판정만 `route_trace`에
남기고 기존 경로로 응답한다(OFF 상태에서 `agents.react_shadow_mode`가 켜져 있으면
그림자 실행만 한다). 단, 신뢰도가 HIGH면 홉 수와 무관하게 파이프라인이 답한다 — ReAct는 신뢰도 MEDIUM/LOW + 2홉 이상일 때만 탄다(측정용 플래그 `agents.react_bypass_confidence`, 기본 OFF, 로 이 관문을 건너뛸 수 있다). [2026-09-18 사후] 6단계 비교 결과 세 플래그 모두 기본 OFF 유지(`docs/plans/evidence-react-ontology-decisions-2026-09.md` S6-3).

<details>
<summary>Original text (kept for history, describes the pre-rework keyword-based activation)</summary>

```
Simple → HybridChatbotAgent 직접 처리
Complex → ReActAgent 활성화
  ├── "분석", "비교", "왜", "원인" 등 분석 키워드
  ├── 멀티 엔티티 질문 (3+ 브랜드/카테고리)
  └── 컨텍스트 부족 (추가 조회 필요)
```

</details>

### 4.2 Agent Layer - 전문 에이전트 (`src/agents/`)

| 에이전트 | 역할 | 입력 → 출력 |
|----------|------|------------|
| `HybridChatbotAgent` | AI 챗봇 (메인) | 쿼리 → KG+RAG+Ontology 융합 답변 |
| `HybridInsightAgent` | 전략 인사이트 | 크롤링 데이터 → 시장 분석 보고서 |
| `CrawlerAgent` | 크롤링 래퍼 | 카테고리 → Amazon Top 100 데이터 |
| `StorageAgent` | 영속화 | 데이터 → SQLite + Google Sheets |
| `MetricsAgent` | KPI 계산 | 순위 데이터 → SoS, HHI, CPI |
| `AlertAgent` | 알림 발생 | 변동 감지 → Email/Telegram |
| `PeriodInsightAgent` | 주간/월간 리포트 | 기간 데이터 → 트렌드 분석 |

### 4.3 RAG Layer - 문서 검색 (`src/rag/`)

**QueryIntent 기반 문서 라우팅:**
```
DIAGNOSIS (진단)  → 아마존 랭킹 급등 원인 분석 가이드
TREND (트렌드)    → K-뷰티 시장 트렌드 레이더
CRISIS (위기)     → 부정 이슈 조기경보 대응 프롬프트
METRIC (지표)     → Strategic Indicators Definition
GENERAL (일반)    → 전체 문서 검색
```

**14개 참조 문서 (docs/ 디렉토리):**
- Type D: 지표 정의, 해석 가이드, 조합 Playbook, 홈페이지 규칙
- Type A: 랭킹 급등 원인 분석, 변동 원인 가이드
- Type B: K-뷰티 트렌드, 미국 뷰티 레이더
- Type C: 부정 이슈 경보, 인플루언서 맵
- Type E: IR 보고서 (1Q/2Q/3Q 2025)

**Embedding Cache:**
```
MD5(query_text) → cached_embedding
├── 최대 1,000개 엔트리
├── FIFO 방식 eviction
├── Hit Rate 추적 (33%+ API 비용 절감)
└── ~6KB/entry (1000개 ≈ 6MB)
```

### 4.4 Ontology Layer - 지식 추론 (`src/ontology/`)

**Knowledge Graph 구조:**
```
Triple: (Subject, Predicate, Object)

관계 유형 (RelationType):
  HAS_PRODUCT          "LANEIGE" ──▶ "B07X3Y1B2Z"
  BELONGS_TO_CATEGORY  "B07X3Y1B2Z" ──▶ "lip_care"
  COMPETES_WITH        "LANEIGE" ◀──▶ "COSRX"
  PARENT_CATEGORY      "lip_care" ──▶ "skin_care"
  HAS_SUBCATEGORY      "skin_care" ──▶ "lip_care"
  HAS_SENTIMENT        "B07X3Y1B2Z" ──▶ "positive"
  HAS_AI_SUMMARY       "B07X3Y1B2Z" ──▶ "Best seller..."

인덱스:
  subject_index  : {entity → [관련 triples]}
  object_index   : {entity → [관련 triples]}
  predicate_index: {relation_type → [triples]}

제한: 최대 50,000 triples (중요도 기반 eviction)
```

**Ontology Reasoner - 규칙 기반 추론:**
```python
Rule: "LOW_SOS_WARNING"
  IF: brand.sos < 5%
  THEN: InsightType.RISK_ALERT
        "브랜드 점유율 위험 수준"

Rule: "HIGH_CONCENTRATION"
  IF: category.hhi > 0.25
  THEN: InsightType.MARKET_STRUCTURE
        "시장 과점 구조 감지"

Rule: "COMPETITIVE_THREAT"
  IF: competitor.rank_change < -10
  THEN: InsightType.COMPETITIVE_POSITION
        "경쟁사 급부상 경고"
```

### 4.5 Tools Layer - 수집 및 유틸 (`src/tools/`)

**Amazon Scraper - Anti-Bot 전략:**
```
Playwright (Headless Chromium)
  ├── playwright-stealth: 자동화 감지 우회
  ├── browserforge: 브라우저 핑거프린트 생성
  ├── Random User-Agent 로테이션
  ├── 지수 백오프 (8초 base × 2^retry)
  ├── Circuit Breaker (3회 실패 → 중단)
  └── 차단 감지 시 디버그 스크린샷 저장
```

**소셜 미디어 수집기 (전부 무료):**

| 플랫폼 | 라이브러리 | 방식 |
|--------|-----------|------|
| TikTok | Playwright | 브라우저 자동화 |
| Instagram | Instaloader | 공식 비공식 API |
| YouTube | yt-dlp | 메타데이터 추출 |
| Reddit | requests | JSON API (인증 불필요) |
| Google Trends | trendspyg | 트렌드 데이터 |

### 4.6 Domain Layer - Clean Architecture 핵심 (`src/domain/`)

```
src/domain/
├── entities/          # 순수 도메인 모델 (외부 의존 없음)
│   ├── product.py     # Product, RankRecord, BadgeType
│   ├── brand.py       # Brand, BrandMetrics
│   ├── market.py      # Category, Snapshot, MarketMetrics
│   └── relations.py   # Relation, RelationType, InferenceResult
│
├── interfaces/        # Protocol 기반 인터페이스
│   ├── agent.py       # CrawlerAgentProtocol
│   ├── knowledge_graph.py
│   ├── retriever.py
│   ├── scraper.py
│   └── llm_client.py
│
└── exceptions.py      # AmoreAgentError 계층
```

**의존성 규칙 (안쪽으로만):**
```
Infrastructure → Domain  ✅
Application → Domain     ✅
Domain → Application     ❌ 절대 금지
Domain → Infrastructure  ❌ 절대 금지
```

---

## 5. KPI 계산 공식

| 지표 | 공식 | 의미 |
|------|------|------|
| **SoS** (Share of Shelf) | LANEIGE 제품수 ÷ Top N × 100 | 진열대 점유율 |
| **HHI** (Herfindahl Index) | Σ(시장점유율²) | 시장 집중도 (>0.25 = 과점) |
| **CPI** (Category Price Index) | LANEIGE 평균가 ÷ 카테고리 평균가 × 100 | 가격 포지셔닝 |

---

## 6. 카테고리 계층 구조

```
Beauty & Personal Care (L0) ─── Node: beauty
├── Skin Care (L1) ─────────── Node: 11060451
│   └── Lip Care (L2) ──────── Node: 3761351
│       └── LANEIGE Lip Sleeping Mask 🎯
│
└── Makeup (L1)
    ├── Lips (L2) ──────────── Node: 11059031
    │   └── 립스틱, 립글로스
    └── Face (L2)
        └── Powder (L3) ────── Node: 11058971
```

> **Lip Care** (스킨케어 하위) ≠ **Lip Makeup** (색조 하위) - 완전히 다른 카테고리

---

## 7. 배포 아키텍처 (Railway)

```
┌─────────────────────────────────┐
│         Railway Cloud           │
│                                 │
│  ┌───────────────────────────┐  │
│  │  FastAPI Server (Uvicorn) │  │
│  │  PORT: auto-detect        │  │
│  └──────────┬────────────────┘  │
│             │                   │
│  ┌──────────▼────────────────┐  │
│  │  /data Volume (Persistent)│  │
│  │  ├── amore_data.db        │  │
│  │  ├── knowledge_graph.json │  │
│  │  └── backups/kg/          │  │
│  │      └── 7일 롤링 백업     │  │
│  └───────────────────────────┘  │
│                                 │
│  Healthcheck: /api/health       │
│  Timeout: 300s                  │
└─────────────────────────────────┘
         │
         │ HTTPS
         ▼
┌─────────────────┐  ┌──────────────┐  ┌──────────────┐
│  Dashboard      │  │ Telegram Bot │  │ Gmail SMTP   │
│  (Browser)      │  │ (Admin)      │  │ (Alerts)     │
└─────────────────┘  └──────────────┘  └──────────────┘
```

**3중 저장소:**

| 저장소 | 역할 | 위치 |
|--------|------|------|
| Railway SQLite | **Source of Truth** | `/data/amore_data.db` |
| Google Sheets | 백업 | 스프레드시트 |
| 로컬 SQLite | 개발용 | `./data/amore_data.db` |

---

## 8. 보안 레이어

```
Request → CORS → Security Headers → Rate Limit → API Key Auth → Handler
                                                      │
                                    ┌─────────────────┘
                                    │
                    ┌───────────────▼───────────────┐
                    │  Security Middleware           │
                    │  ├── X-Content-Type-Options    │
                    │  ├── X-Frame-Options           │
                    │  ├── X-XSS-Protection          │
                    │  ├── SlowAPI Rate Limiting      │
                    │  ├── API Key 마스킹 (sk-****)   │
                    │  └── Prompt Injection Guard     │
                    └───────────────────────────────┘
```

---

## 9. 기술 스택 요약

```
┌── Backend ──────────┐  ┌── AI/ML ──────────────┐  ┌── Storage ──────────┐
│ Python 3.11+        │  │ GPT-4.1-mini (LiteLLM)│  │ SQLite              │
│ FastAPI + Uvicorn   │  │ OpenAI Embeddings      │  │ Google Sheets       │
│ APScheduler         │  │ ChromaDB (Vector DB)   │  │ JSON (KG)           │
│ Pydantic v2         │  │ owlready2 (Ontology)   │  │ Railway Volume      │
└─────────────────────┘  └────────────────────────┘  └─────────────────────┘

┌── Scraping ─────────┐  ┌── Social Media ────────┐  ┌── Monitoring ──────┐
│ Playwright          │  │ Playwright (TikTok)    │  │ Telegram Bot API   │
│ playwright-stealth  │  │ Instaloader (IG)       │  │ Gmail SMTP         │
│ browserforge        │  │ yt-dlp (YouTube)       │  │ Python logging     │
│ fake-useragent      │  │ JSON API (Reddit)      │  │ APScheduler jobs   │
└─────────────────────┘  │ trendspyg (Trends)     │  └────────────────────┘
                         └────────────────────────┘

┌── Testing ──────────┐  ┌── DevOps ─────────────┐
│ pytest + pytest-cov │  │ Railway (PaaS)         │
│ pytest-asyncio      │  │ Git + pre-commit       │
│ 60% min coverage    │  │ Ruff (linter/fmt)      │
│ Golden set eval     │  │ detect-secrets         │
└─────────────────────┘  └────────────────────────┘
```

---

## 10. 디자인 패턴 정리

| 패턴 | 적용 위치 | 설명 |
|------|----------|------|
| **Clean Architecture** | 전체 | Domain→Application→Infrastructure 계층 분리 |
| **Protocol (DI)** | `src/domain/interfaces/` | 생성자 주입, 인터페이스 기반 |
| **ReAct Pattern** | `react_agent.py` | Thought→Action→Observation→Reflection |
| **Circuit Breaker** | `amazon_scraper.py` | 3회 실패 시 자동 중단 |
| **TTL Cache** | `cache.py`, `retriever.py` | 시간 기반 자동 만료 |
| **FIFO Eviction** | Embedding Cache | 최대 1000개, 오래된 것부터 제거 |
| **Observer** | AlertAgent | 변동 감지 → 알림 발행 |
| **Strategy** | QueryIntent | 의도별 다른 검색 전략 |
| **Facade** | UnifiedBrain | 복잡한 서브시스템 통합 인터페이스 |
