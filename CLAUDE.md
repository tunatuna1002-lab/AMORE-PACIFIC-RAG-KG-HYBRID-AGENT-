# CLAUDE.md

> Claude Code가 이 코드베이스 작업 시 참조하는 필수 컨텍스트

---

## 1. 프로젝트 개요

**AMORE Pacific RAG-KG Hybrid Agent** — Amazon US LANEIGE 브랜드 경쟁력 모니터링 자율 AI 시스템

- **Daily Auto-Crawling**: Amazon Best Sellers Top 100 × 5 카테고리 (22:00 KST)
- **KPI Analysis**: SoS(Share of Shelf), HHI(시장 집중도), CPI(가격경쟁력)
- **AI Chatbot**: RAG + Knowledge Graph + Ontology 하이브리드 검색
- **Insight Generation**: LLM 기반 전략적 인사이트 자동 생성
- **Alert System**: 순위 변동 감지 → 이메일/Telegram 알림

### 코드베이스 규모

| 항목 | 수치 |
|------|------|
| src/ Python 파일 | 200개 |
| src/ 코드 라인 | ~70,700 lines |
| tests/ 파일 | 111개 |
| tests/ 코드 라인 | ~22,400 lines |
| src/api/dashboard_api.py | ~3,900 lines |
| 커버리지 목표 | 60% (pytest-cov) |

---

## 2. 기술 스택

| Category | Technology |
|----------|-----------|
| Language | Python 3.11+ (로컬: 3.13.7, `python3` 사용) |
| Backend | FastAPI, Uvicorn |
| LLM | OpenAI GPT-4.1-mini via LiteLLM |
| Scraping | Playwright, playwright-stealth, browserforge, fake-useragent |
| Storage | SQLite (aiosqlite), Google Sheets API |
| RAG | ChromaDB + sentence-transformers (all-MiniLM-L6-v2) |
| Ontology | owlready2, rdflib, Rule-based Reasoner |
| NLP | spaCy (NER/Entity Linking) |
| Data | pandas, numpy, matplotlib |
| Test | pytest, pytest-asyncio, pytest-cov |
| Lint | Ruff (line-length=100, target=py311) |
| Deploy | Docker (python:3.11-slim), Railway |
| Notifications | Gmail SMTP, Telegram Bot, Resend |
| Social Media | Playwright (TikTok), Instaloader (IG), yt-dlp (YT), JSON API (Reddit) |

---

## 3. Entry Points

| 파일 | 역할 | 실행 방법 |
|------|------|-----------|
| `src/api/dashboard_api.py` | **FastAPI 메인 서버** (3,900 lines monolith) | `uvicorn src.api.dashboard_api:app --host 0.0.0.0 --port 8001 --reload` |
| `scripts/start.py` | Railway 배포용 시작 스크립트 | `python scripts/start.py` (PORT 환경변수 사용) |
| `main.py` | CLI 진입점 (크롤링 + 챗봇) | `python main.py` / `python main.py --chat` |
| `src/core/orchestrator.py` | BatchWorkflow 별칭 (하위 호환) | `from src.core.orchestrator import Orchestrator` |

### 주요 API 엔드포인트

| Method | Endpoint | Description | Auth |
|--------|----------|-------------|------|
| GET | `/api/health` | 헬스체크 | - |
| GET | `/api/data` | 대시보드 데이터 JSON | - |
| POST | `/api/v3/chat` | AI 챗봇 (권장) | - |
| POST | `/api/crawl/start` | 크롤링 시작 | API Key |
| GET | `/api/v4/brain/status` | 스케줄러 상태 | - |
| GET | `/dashboard` | 대시보드 UI (HTML) | - |

---

## 4. 프로젝트 구조

```
.
├── main.py                       # CLI 진입점
│
├── src/
│   ├── api/                      # API (FastAPI)
│   │   ├── dashboard_api.py      # FastAPI 메인 서버 (루트에서 이동)
│   │   ├── app_factory.py        # 앱 초기화
│   │   ├── routes/               # 라우트 모듈
│   │   │   ├── chat.py, crawl.py, data.py
│   │   │   ├── health.py, brain.py, export.py
│   │   │   ├── alerts.py, analytics.py
│   │   │   ├── competitors.py, deals.py
│   │   │   ├── market_intelligence.py
│   │   │   ├── signals.py, sync.py
│   │   │   └── __init__.py
│   │   ├── validators/
│   │   │   └── input_validator.py
│   │   ├── dependencies.py
│   │   └── models.py
│   │
│   ├── core/                     # 핵심 오케스트레이션
│   │   ├── brain.py              # UnifiedBrain - 자율 스케줄러
│   │   ├── react_agent.py        # ReAct Self-Reflection Agent
│   │   ├── batch_workflow.py     # 배치 워크플로우 (=Orchestrator)
│   │   ├── orchestrator.py       # BatchWorkflow 하위 호환 래퍼 (루트에서 이동)
│   │   ├── query_router.py       # 쿼리 라우팅
│   │   ├── query_processor.py    # 쿼리 처리
│   │   ├── response_pipeline.py  # 응답 파이프라인
│   │   ├── hallucination_detector.py
│   │   ├── prompt_guard.py       # 프롬프트 인젝션 방어
│   │   ├── circuit_breaker.py    # 서킷브레이커
│   │   ├── cache.py              # 캐시
│   │   ├── scheduler.py          # 크론 스케줄러
│   │   └── ... (30+ modules)
│   │
│   ├── agents/                   # AI 에이전트
│   │   ├── hybrid_chatbot_agent.py   # 하이브리드 챗봇
│   │   ├── hybrid_insight_agent.py   # 인사이트 생성
│   │   ├── crawler_agent.py          # 크롤러 에이전트
│   │   ├── alert_agent.py            # 알림 에이전트
│   │   ├── metrics_agent.py          # 메트릭 에이전트
│   │   ├── storage_agent.py          # 저장 에이전트
│   │   ├── suggestion_engine.py      # 후속 질문 생성 엔진
│   │   ├── source_provider.py        # 출처 추출 및 포매팅
│   │   ├── external_signal_manager.py # 외부 신호 수집 관리
│   │   ├── period_insight_agent.py   # 기간별 인사이트
│   │   └── true_hybrid_insight_agent.py
│   │
│   ├── rag/                      # RAG 시스템
│   │   ├── hybrid_retriever.py   # KG + RAG 통합 검색
│   │   ├── retrieval_strategy.py  # OWL + Legacy 전략 패턴
│   │   ├── confidence_fusion.py  # 다중 소스 신뢰도 융합
│   │   ├── retriever.py          # 문서 검색 + 임베딩 캐시
│   │   ├── embedding_cache.py    # 임베딩 캐시 (InMemory/SQLite)
│   │   ├── reranker.py           # 재순위화
│   │   ├── entity_linker.py      # 엔티티 링킹
│   │   ├── chunker.py            # 문서 청킹
│   │   ├── query_rewriter.py     # 쿼리 리라이팅
│   │   ├── confidence_fusion.py  # 신뢰도 융합
│   │   ├── context_builder.py    # 컨텍스트 빌더
│   │   └── router.py             # RAG 라우터
│   │
│   ├── ontology/                 # Knowledge Graph & 추론
│   │   ├── knowledge_graph.py    # Triple Store (JSON 기반)
│   │   ├── ontology_knowledge_graph.py
│   │   ├── reasoner.py           # 규칙 기반 추론 엔진
│   │   ├── owl_reasoner.py       # OWL 추론
│   │   ├── unified_reasoner.py   # 통합 추론
│   │   ├── kg_enricher.py        # KG 보강
│   │   ├── kg_query.py           # KG 쿼리
│   │   ├── kg_updater.py         # KG 업데이트
│   │   ├── schema.py             # 온톨로지 스키마
│   │   ├── category_service.py   # 카테고리 서비스
│   │   ├── sentiment_service.py  # 감성 분석 서비스
│   │   └── rules/                # 비즈니스 규칙
│   │       ├── alert_rules.py
│   │       ├── growth_rules.py
│   │       ├── market_rules.py
│   │       ├── price_rules.py
│   │       ├── ir_rules.py
│   │       └── sentiment_rules.py
│   │
│   ├── tools/                    # 도구 모음
│   │   ├── scrapers/             # 웹 스크래퍼
│   │   │   ├── amazon_scraper.py
│   │   │   ├── amazon_product_scraper.py
│   │   │   └── deals_scraper.py
│   │   ├── collectors/           # 데이터 수집기
│   │   │   ├── tiktok_collector.py
│   │   │   ├── instagram_collector.py
│   │   │   ├── youtube_collector.py
│   │   │   ├── reddit_collector.py
│   │   │   ├── google_trends_collector.py
│   │   │   ├── public_data_collector.py
│   │   │   ├── external_signal_collector.py
│   │   │   └── tavily_search.py
│   │   ├── calculators/          # 지표 계산
│   │   │   ├── metric_calculator.py   # SoS, HHI, CPI
│   │   │   ├── period_analyzer.py
│   │   │   └── exchange_rate.py
│   │   ├── intelligence/         # 시장 정보
│   │   │   ├── market_intelligence.py
│   │   │   ├── morning_brief.py
│   │   │   ├── ir_report_parser.py
│   │   │   ├── claim_extractor.py
│   │   │   ├── claim_verifier.py
│   │   │   ├── confidence_scorer.py
│   │   │   ├── insight_verifier.py
│   │   │   └── source_manager.py
│   │   ├── exporters/            # 내보내기
│   │   │   ├── report_generator.py
│   │   │   ├── chart_generator.py
│   │   │   ├── dashboard_exporter.py
│   │   │   ├── export_handlers.py
│   │   │   └── insight_formatter.py
│   │   ├── storage/              # 저장소
│   │   │   ├── sqlite_storage.py
│   │   │   └── sheets_writer.py
│   │   ├── notifications/        # 알림
│   │   │   ├── email_sender.py
│   │   │   ├── telegram_bot.py
│   │   │   └── alert_service.py
│   │   └── utilities/            # 유틸리티
│   │       ├── kg_backup.py
│   │       ├── brand_resolver.py
│   │       ├── data_integrity_checker.py
│   │       ├── job_queue.py
│   │       └── reference_tracker.py
│   │
│   ├── domain/                   # Clean Architecture Layer 1
│   │   ├── entities/             # 도메인 엔티티
│   │   │   ├── product.py
│   │   │   ├── brand.py
│   │   │   ├── market.py
│   │   │   ├── brain_models.py
│   │   │   └── relations.py
│   │   ├── interfaces/           # 프로토콜/인터페이스
│   │   │   ├── agent.py, alert.py, brain.py
│   │   │   ├── chatbot.py, insight.py, retriever.py
│   │   │   ├── knowledge_graph.py, llm_client.py
│   │   │   ├── metric.py, repository.py
│   │   │   ├── scraper.py, signal.py, storage.py
│   │   │   └── brain_components.py
│   │   ├── value_objects/
│   │   └── exceptions.py
│   │
│   ├── application/              # Clean Architecture Layer 2
│   │   ├── workflows/            # 유스케이스
│   │   │   ├── chat_workflow.py
│   │   │   ├── crawl_workflow.py
│   │   │   ├── insight_workflow.py
│   │   │   ├── alert_workflow.py
│   │   │   └── batch_workflow.py
│   │   ├── services/
│   │   │   └── query_analyzer.py
│   │   └── orchestrators/
│   │
│   ├── adapters/                 # Clean Architecture Layer 3
│   │   ├── agents/
│   │   ├── presenters/
│   │   └── rag/
│   │
│   ├── infrastructure/           # Clean Architecture Layer 4
│   │   ├── bootstrap.py
│   │   ├── container.py          # DI 컨테이너
│   │   ├── feature_flags.py      # Feature flag 시스템
│   │   ├── config/
│   │   │   └── config_manager.py
│   │   └── persistence/
│   │       ├── json_repository.py
│   │       └── sheets_repository.py
│   │
│   ├── memory/                   # 대화 메모리
│   │   ├── conversation_memory.py
│   │   ├── session.py
│   │   ├── context.py
│   │   └── history.py
│   │
│   ├── monitoring/               # 모니터링
│   │   ├── logger.py             # AgentLogger
│   │   ├── metrics.py
│   │   ├── rag_metrics.py
│   │   └── tracer.py
│   │
│   └── shared/                   # 공유 유틸
│       ├── constants.py
│       └── llm_client.py
│
├── config/                       # 설정 파일
│   ├── thresholds.json           # 시스템 설정 + 카테고리 URL
│   ├── category_hierarchy.json   # Amazon 카테고리 트리
│   ├── competitors.json          # 경쟁사 정보
│   ├── tracked_competitors.json
│   ├── brands.json               # 브랜드 매핑
│   ├── asin_brand_mapping.json
│   ├── entities.json             # 엔티티 정의
│   ├── rules.json                # 비즈니스 규칙
│   ├── retrieval_weights.json    # RAG 가중치
│   └── public_apis.json          # 공공 API 설정
│
├── prompts/                      # 프롬프트 템플릿
│   ├── chat_system.txt
│   ├── insight_generation.txt
│   ├── query_router.txt
│   ├── metrics.json
│   ├── version_manager.py
│   ├── registry.py               # 프롬프트 중앙 관리
│   └── components/               # 프롬프트 컴포넌트
│
├── dashboard/                    # 프론트엔드
│   ├── amore_unified_dashboard_v4.html  # 메인 대시보드
│   └── test_chat.html
│
├── eval/                         # 평가 프레임워크
│   ├── cli.py, runner.py, loader.py
│   ├── schemas.py, regression.py, report.py
│   ├── cost_tracker.py
│   ├── judge/                    # LLM Judge
│   ├── metrics/                  # 평가 메트릭 (L1~L5)
│   ├── validators/               # 검증기
│   └── data/                     # 골든셋
│       ├── golden/
│       └── examples/
│
├── tests/                        # 테스트
│   ├── conftest.py               # 공통 fixture
│   ├── unit/                     # 단위 테스트 (레이어별)
│   │   ├── agents/, api/, application/
│   │   ├── core/, domain/, infrastructure/
│   │   ├── memory/, monitoring/
│   │   ├── ontology/, prompts/
│   │   ├── rag/, shared/, tools/
│   │   └── __init__.py
│   ├── eval/                     # 평가 테스트 (semantic, metrics, regression)
│   ├── integration/              # 통합 테스트
│   ├── adversarial/              # 적대적 테스트 (prompt injection)
│   └── golden/
│
├── scripts/                      # 운영 스크립트
│   ├── start.py                  # Railway 배포용 시작 (루트에서 이동)
│   ├── start_dashboard.command   # macOS 원클릭 시작 (루트에서 이동)
│   ├── sync_from_railway.py
│   ├── sync_sheets_to_sqlite.py
│   ├── sync_to_railway.py
│   ├── evaluate_golden.py
│   ├── export_dashboard.py
│   ├── run_evaluation.py
│   └── ...
│
├── examples/                     # 예제 스크립트
│   ├── react_agent_demo.py
│   ├── confidence_fusion_demo.py
│   ├── conversation_memory_demo.py
│   └── ...
│
├── data/                         # 런타임 데이터 (gitignored)
│   ├── amore_data.db             # 메인 SQLite DB
│   ├── knowledge_graph.json      # KG Triple Store
│   ├── dashboard_data.json       # 캐시된 대시보드 데이터
│   ├── chroma/                   # ChromaDB 벡터 스토어
│   ├── market_intelligence/      # 시장 정보 (signals, youtube, ir 등)
│   └── ...
│
├── docs/                         # 문서
│   ├── analysis/                 # 의존성 분석, 정리 계획
│   ├── architecture/             # 아키텍처 설계
│   ├── guides/                   # 가이드
│   ├── reports/                  # 분석 리포트
│   ├── research/                 # 리서치 자료
│   ├── security/                 # 보안 감사
│   ├── diagrams/
│   ├── plans/
│   ├── refactoring/
│   └── dev/                      # 개발 노트
│
├── logs/                         # 로그 (gitignored)
├── static/fonts/                 # 웹 폰트 + AMOREPACIFIC CI 폰트
│
├── pyproject.toml                # 프로젝트 설정 (pytest, ruff, coverage)
├── requirements.txt              # Python 의존성
├── Dockerfile                    # Docker 빌드 (python:3.11-slim + Playwright)
├── railway.toml                  # Railway 배포 설정
├── .pre-commit-config.yaml       # Pre-commit 훅
└── .env.example                  # 환경변수 템플릿
```

---

## 5. 모니터링 카테고리 (Amazon BSR)

```
Beauty & Personal Care (L0)
├── Skin Care (L1)
│   └── Lip Care (L2)  ← LANEIGE Lip Sleeping Mask
└── Makeup (L1)
    ├── Lip Makeup (L2) ← 립스틱, 립글로스
    └── Face Makeup (L2)
        └── Face Powder (L3)
```

| 카테고리 | Node ID | Level | Parent | 모니터링 |
|----------|---------|-------|--------|----------|
| Beauty & Personal Care | `beauty` | 0 | - | O |
| Skin Care | `11060451` | 1 | beauty | O |
| Lip Care | `3761351` | 2 | skin_care | O |
| Lip Makeup | `11059031` | 2 | makeup | O |
| Face Powder | `11058971` | 3 | face_makeup | O |

> **주의**: Lip Care(스킨케어)와 Lip Makeup(색조)은 **다른** 카테고리.
> LANEIGE Lip Sleeping Mask → Lip Care (Skin Care 하위)

---

## 6. 개발 명령어

```bash
# 서버 실행
uvicorn src.api.dashboard_api:app --host 0.0.0.0 --port 8001 --reload

# 테스트 (python3 사용)
python3 -m pytest tests/ -v                    # 전체 (커버리지 포함)
python3 -m pytest tests/unit/domain/ -v        # Domain 레이어만
python3 -m pytest tests/ -m "not slow" -v      # 느린 테스트 제외

# 골든셋 평가
python3 scripts/evaluate_golden.py --verbose

# KG 백업
python3 -m src.tools.utilities.kg_backup backup
python3 -m src.tools.utilities.kg_backup list
python3 -m src.tools.utilities.kg_backup restore 2026-01-27

# 데이터 동기화
python3 scripts/sync_from_railway.py           # Railway → 로컬
python3 scripts/sync_sheets_to_sqlite.py       # Sheets → SQLite

# 린팅
ruff check src/ --fix
ruff format src/
```

---

## 7. 환경 변수

```bash
# 필수
OPENAI_API_KEY=sk-...

# 서버
API_KEY=...                        # 보호 엔드포인트 인증
AUTO_START_SCHEDULER=true          # 스케줄러 자동 시작

# Google Sheets
GOOGLE_SPREADSHEET_ID=...
GOOGLE_SHEETS_CREDENTIALS_JSON=...

# LLM
LLM_TEMPERATURE_CHAT=0.4
LLM_TEMPERATURE_INSIGHT=0.6

# 외부 신호
TAVILY_API_KEY=tvly-...            # 뉴스 (월 1,000건 무료)
GNEWS_API_KEY=...                  # GNews (일 100건 무료)
DATA_GO_KR_API_KEY=...             # 관세청/식약처

# 알림
SMTP_SERVER=smtp.gmail.com         # Gmail SMTP
SMTP_PORT=587
SENDER_EMAIL=...
SENDER_PASSWORD=...                # Gmail 앱 비밀번호
ALERT_RECIPIENTS=...
TELEGRAM_BOT_TOKEN=...
TELEGRAM_ADMIN_CHAT_ID=...
```

---

## 8. Clean Architecture

```
src/
├── domain/           # Layer 1: Entities + Interfaces (외부 의존 없음)
├── application/      # Layer 2: Use Cases / Workflows
├── adapters/         # Layer 3: Interface Adapters
└── infrastructure/   # Layer 4: Frameworks & Drivers
```

### Import 규칙 (의존성: 안쪽으로만)

| From → To | 허용 |
|-----------|------|
| domain → (nothing) | O |
| application → domain | O |
| adapters → domain, application | O |
| infrastructure → domain, application | O |
| **domain → application/infrastructure** | X |

### DI 패턴

```python
# Bad: 구체 클래스 직접 import
from src.agents.crawler_agent import CrawlerAgent

# Good: Protocol 기반 DI
from src.domain.interfaces.agent import CrawlerAgentProtocol
class MyWorkflow:
    def __init__(self, crawler: CrawlerAgentProtocol):
        self.crawler = crawler
```

---

## 9. 핵심 모듈 참조

| 모듈 | 경로 | 역할 |
|------|------|------|
| DashboardAPI | `src/api/dashboard_api.py` | FastAPI 메인 서버 |
| Orchestrator | `src/core/orchestrator.py` | BatchWorkflow 하위 호환 래퍼 |
| UnifiedBrain | `src/core/brain.py` | 자율 스케줄러 + ReAct 통합 |
| ReActAgent | `src/core/react_agent.py` | Self-Reflection (복잡한 질문) |
| BatchWorkflow | `src/core/batch_workflow.py` | 배치 워크플로우 (=Orchestrator) |
| HybridChatbot | `src/agents/hybrid_chatbot_agent.py` | AI 챗봇 |
| HybridInsight | `src/agents/hybrid_insight_agent.py` | 인사이트 생성 |
| AlertAgent | `src/agents/alert_agent.py` | 순위 변동 알림 |
| SuggestionEngine | `src/agents/suggestion_engine.py` | 후속 질문 생성 엔진 |
| SourceProvider | `src/agents/source_provider.py` | 출처 추출 및 포매팅 |
| ExternalSignalManager | `src/agents/external_signal_manager.py` | 외부 신호 수집 관리 |
| HybridRetriever | `src/rag/hybrid_retriever.py` | RAG + KG 통합 검색 |
| RetrievalStrategy | `src/rag/retrieval_strategy.py` | OWL + 인텐트 기반 전략 패턴 |
| ConfidenceFusion | `src/rag/confidence_fusion.py` | 다중 소스 신뢰도 융합 엔진 |
| Retriever | `src/rag/retriever.py` | 문서 검색 + 임베딩 캐시 |
| EmbeddingCache | `src/rag/embedding_cache.py` | 임베딩 캐시 (InMemory/SQLite) |
| KnowledgeGraph | `src/ontology/knowledge_graph.py` | Triple Store (JSON) |
| UnifiedReasoner | `src/ontology/unified_reasoner.py` | 통합 추론 엔진 (OWL + Rules) |
| PromptRegistry | `prompts/registry.py` | 프롬프트 중앙 관리 |
| FeatureFlags | `src/infrastructure/feature_flags.py` | Feature flag 시스템 (ENV > JSON > default) |
| MetricCalculator | `src/tools/calculators/metric_calculator.py` | SoS, HHI, CPI |
| AmazonScraper | `src/tools/scrapers/amazon_scraper.py` | Playwright 크롤러 |
| KGBackup | `src/tools/utilities/kg_backup.py` | KG 백업 (7일 롤링) |
| EmailSender | `src/tools/notifications/email_sender.py` | Gmail SMTP |
| TelegramBot | `src/tools/notifications/telegram_bot.py` | Telegram 알림 |
| AgentLogger | `src/monitoring/logger.py` | 구조화 로깅 |
| Container | `src/infrastructure/container.py` | DI 컨테이너 |

---

## 10. 데이터 저장소

| 저장소 | 위치 | 역할 |
|--------|------|------|
| SQLite (Railway) | `/data/amore_data.db` | Source of Truth |
| SQLite (로컬) | `./data/amore_data.db` | 개발용 |
| Google Sheets | 스프레드시트 | 백업 |
| KG JSON | `data/knowledge_graph.json` | Triple Store |
| ChromaDB | `data/chroma/` | 벡터 스토어 |
| Dashboard JSON | `data/dashboard_data.json` | 캐시 |

### KG 백업 정책
- 위치: `data/backups/kg/`
- 주기: 일 1회 (크롤링 완료 후)
- 보관: 7일 롤링

---

## 11. 코드 컨벤션

- **Async-First**: 모든 I/O 작업은 `async/await`
- **Type Hints**: 모든 함수에 파라미터 + 반환 타입 힌트 필수
- **Pydantic Models**: 데이터 구조는 `BaseModel` 사용
- **Ruff**: line-length=100, target=py311 (`E501` 무시)
- **TDD**: RED → GREEN → REFACTOR
- **테스트 경로**: `tests/unit/{layer}/test_*.py`
- **테스트 환경 분리**: `.env.test` 사용

---

## 12. 배포 (Railway)

| 항목 | 값 |
|------|-----|
| 프로젝트 | splendid-harmony |
| 빌드 | Dockerfile (python:3.11-slim + Playwright Chromium) |
| Healthcheck | `/api/health` (300초 타임아웃) |
| Volume | `/data` (SQLite + KG) |
| 포트 | `PORT` 환경변수 (기본 8001) |
| 재시작 정책 | on_failure (최대 3회) |

---

## 13. 디자인 시스템 (AMOREPACIFIC)

| 색상 | HEX | 용도 |
|------|-----|------|
| Pacific Blue | `#001C58` | 헤더, 사이드바, 주요 CTA |
| Amore Blue | `#1F5795` | 강조, 링크 |
| Gray | `#7D7D7D` | 보조 텍스트 |
| White | `#FFFFFF` | 배경, 카드 |

---

## 14. 컨텍스트 관리 규칙

1. 토큰 부족을 이유로 작업을 일찍 중단하지 마세요. 컨텍스트는 자동 compact됩니다.
2. Phase별 관련 파일 5-20개만 로드하세요. 전체 코드베이스를 읽지 마세요.
3. 리서치나 탐색 작업은 서브에이전트에 위임하세요.
4. 미완료 작업은 `docs/dev/FUTURE_WORK.md`를 참조하세요.
5. 리팩토링 성과는 `docs/REFACTORING_RESULTS.md`를 참조하세요.
