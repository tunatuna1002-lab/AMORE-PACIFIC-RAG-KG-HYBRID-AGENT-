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
| src/ Python 파일 | 238개 |
| src/ 코드 라인 | ~74,900 lines |
| tests/ 파일 | 228개 |
| tests/ 코드 라인 | ~79,200 lines |
| src/api/dashboard_api.py | 195 lines (진입점, 라우트는 routes/ 분리) |
| 커버리지 게이트 | 75% (`fail_under`, pyproject + CI) |

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
| `src/api/dashboard_api.py` | **FastAPI 메인 서버** (진입점, 라우트는 `src/api/routes/` 12개 모듈) | `uvicorn src.api.dashboard_api:app --host 0.0.0.0 --port 8001 --reload` |
| `scripts/start.py` | Railway 배포용 시작 스크립트 | `python scripts/start.py` (PORT 환경변수 사용) |
| `main.py` | CLI 진입점 (크롤링 + 챗봇) | `python main.py` / `python main.py --chat` |
| `src/application/workflows/batch_workflow.py` | 일일 배치 파이프라인 | `BatchWorkflow.run_daily_workflow()` |

### 주요 API 엔드포인트

| Method | Endpoint | Description | Auth |
|--------|----------|-------------|------|
| GET | `/api/health` | 헬스체크 | - |
| GET | `/api/data` | 대시보드 데이터 JSON | - |
| POST | `/api/v4/chat` | AI 챗봇 (권장, 스트리밍: `/api/v4/chat/stream`) | API Key |
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
│   │   ├── dashboard_api.py      # FastAPI 메인 서버 (195줄, 진입점)
│   │   ├── app_factory.py        # 앱 초기화
│   │   ├── dashboard_shape.py    # 대시보드 응답 형태 어댑터
│   │   ├── dependencies.py       # deps/ 전체를 재수출하는 파사드
│   │   ├── deps/                 # 의존성 주입 (dependencies.py 에서 분할)
│   │   │   ├── auth.py, session.py, audit.py
│   │   │   ├── data.py, suggestions.py, providers.py
│   │   ├── routes/               # 라우트 모듈 13개
│   │   │   ├── chat.py, crawl.py, data.py (113줄)
│   │   │   ├── health.py, brain.py, export.py (346줄)
│   │   │   ├── alerts.py (737줄), analytics.py (195줄)
│   │   │   ├── competitors.py, deals.py
│   │   │   ├── market_intelligence.py, signals.py, sync.py
│   │   ├── middleware/           # csrf.py, security_headers.py
│   │   ├── validators/           # input_validator.py
│   │   └── models.py
│   │
│   ├── core/                     # 핵심 오케스트레이션
│   │   ├── brain.py              # UnifiedBrain — 자율 에이전트 코어
│   │   ├── brain_scheduler.py    # 스케줄 루프 (brain 에서 분리)
│   │   ├── query_graph.py        # **단일 챗 경로** (run / run_stream)
│   │   ├── graph_state.py        # QueryGraph 상태
│   │   ├── intent.py             # 의도 분류 (단일 출처)
│   │   ├── query_router.py       # is_compound 판정만 유지
│   │   ├── react_agent.py        # ReAct Self-Reflection (기본 off)
│   │   ├── state_manager.py      # 시스템 상태 단일 출처
│   │   ├── response_pipeline.py, verification_pipeline.py
│   │   ├── hallucination_detector.py, confidence.py
│   │   ├── prompt_guard.py       # 프롬프트 인젝션 방어
│   │   ├── circuit_breaker.py, cache.py
│   │   ├── alert_manager.py, crawl_manager.py
│   │   ├── context_gatherer.py, decision_maker.py
│   │   ├── explainability.py, tool_coordinator.py, tools.py
│   │   └── models.py
│   │
│   ├── agents/                   # AI 에이전트
│   │   ├── base_hybrid_agent.py  # 공통 베이스
│   │   ├── hybrid_chatbot_agent.py
│   │   ├── hybrid_insight_agent.py, period_insight_agent.py
│   │   ├── crawler_agent.py, alert_agent.py
│   │   ├── metrics_agent.py, storage_agent.py
│   │   ├── suggestion_engine.py, source_provider.py
│   │   └── external_signal_manager.py
│   │
│   ├── rag/                      # RAG 시스템 (Phase 4 에서 분할)
│   │   ├── hybrid_retriever.py   # KG + RAG 통합 파사드 (665줄)
│   │   ├── retriever.py          # 문서 검색 파사드 (676줄)
│   │   ├── fusion/               # rrf.py, weighted.py, hybrid_search.py
│   │   ├── vector_index.py, bm25_index.py, search_cache.py
│   │   ├── document_registry.py, document_loader.py
│   │   ├── chunker.py, section_chunker.py
│   │   ├── kg_facts.py, kg_edges.py        # KG 사실·엣지 수집
│   │   ├── context_builder.py, context_render.py, hybrid_context.py
│   │   ├── query_expansion.py, query_rewriter.py
│   │   ├── selfrag_gate.py, relevance_grader.py, reranker.py
│   │   ├── retrieval_strategy.py, confidence_fusion.py
│   │   ├── embedding_cache.py, entity_linker.py
│   │   ├── legacy_intent.py      # (deprecated shim — core/intent.py 사용)
│   │   ├── rag_kg_extractor.py, templates.py, router.py
│   │
│   ├── ontology/                 # Knowledge Graph & 추론
│   │   ├── knowledge_graph.py    # Triple Store (JSON 기반)
│   │   ├── tbox.py               # OWL TBox 정의
│   │   ├── builder.py            # ABox 구축
│   │   ├── materializer.py       # 추론 결과 물질화 (provenance)
│   │   ├── inference_context.py  # 추론 컨텍스트
│   │   ├── thresholds.py         # config/thresholds.json 단일 출처
│   │   ├── reasoner.py           # 규칙 기반 추론
│   │   ├── owl_reasoner.py       # OWL 추론 (**배치 전용**)
│   │   ├── kg_enricher.py, kg_query.py, kg_updater.py, kg_iri.py
│   │   ├── category_service.py, sentiment_service.py
│   │   └── rules/                # alert, growth, market, price, ir, sentiment
│   │
│   ├── tools/                    # 도구 모음
│   │   ├── scrapers/             # amazon_scraper.py, deals_scraper.py
│   │   ├── collectors/           # google_trends, public_data,
│   │   │                         #   external_signal, tavily_search
│   │   ├── calculators/          # metric_calculator.py (SoS/HHI/CPI),
│   │   │                         #   period_analyzer.py
│   │   ├── intelligence/         # market_intelligence, morning_brief,
│   │   │                         #   newsletter, ir_report_parser,
│   │   │                         #   claim_extractor/verifier,
│   │   │                         #   confidence_scorer, insight_verifier,
│   │   │                         #   source_manager
│   │   ├── exporters/
│   │   │   ├── report/           # design, docx, pptx, pdf, facade
│   │   │   ├── report_generator.py, chart_generator.py
│   │   │   ├── dashboard_exporter.py, export_handlers.py
│   │   │   └── insight_formatter.py
│   │   ├── storage/              # sqlite_storage.py, sheets_writer.py
│   │   ├── notifications/        # email_sender, telegram_bot, alert_service
│   │   └── utilities/            # kg_backup, brand_resolver,
│   │                             #   data_integrity_checker, job_queue,
│   │                             #   reference_tracker
│   │
│   ├── domain/                   # Clean Architecture Layer 1
│   │   ├── entities/             # product, brand, market, alert,
│   │   │                         #   brain_models, relations
│   │   ├── interfaces/           # Protocol 정의 13개
│   │   ├── value_objects/        # retrieval_result.py
│   │   ├── brand.py              # 브랜드 정규화·해석 (도메인 승격)
│   │   └── exceptions.py
│   │
│   ├── application/              # Clean Architecture Layer 2
│   │   ├── workflows/
│   │   │   ├── batch_workflow.py # 일일 배치 (크롤→저장→KG→지표→인사이트→알림→내보내기)
│   │   │   └── chat_workflow.py
│   │   └── services/             # API 라우트에서 추출한 서비스 12개
│   │       ├── dashboard_data_service.py, sql_rows.py
│   │       ├── date_range.py, category_names.py
│   │       ├── query_analyzer.py, export_service.py
│   │       ├── alert_service.py, analytics_service.py
│   │       ├── sos_trend_service.py, historical_service.py
│   │       ├── brand_matrix.py, external_signals_service.py
│   │
│   ├── infrastructure/           # Clean Architecture Layer 4
│   │   ├── bootstrap.py, container.py, feature_flags.py
│   │   ├── config/config_manager.py
│   │   └── persistence/          # json_repository.py, sheets_repository.py
│   │
│   ├── memory/                   # conversation_memory, session, context, history
│   ├── monitoring/               # logger(AgentLogger), metrics, rag_metrics, tracer
│   └── shared/                   # constants, llm_client, units, parsing
│
├── config/                       # 설정 파일
│   ├── thresholds.json           # **임계값 단일 출처** + 카테고리 URL
│   ├── category_hierarchy.json   # Amazon 카테고리 트리
│   ├── competitors.json, tracked_competitors.json
│   ├── brands.json, asin_brand_mapping.json, entities.json
│   ├── retrieval_weights.json    # freshness / max_context_items
│   └── public_apis.json
│
├── prompts/                      # 프롬프트 템플릿
│   ├── registry.py               # 프롬프트 중앙 관리
│   ├── agents/chatbot_system.txt
│   ├── agents/variants/          # 프롬프트 실험 변형
│   ├── components/, metrics.json, version_manager.py
│
├── dashboard/                    # 프론트엔드
│   └── amore_unified_dashboard_v4.html
│
├── eval/                         # 평가 프레임워크
│   ├── cli.py, runner.py, loader.py, schemas.py
│   ├── regression.py, report.py, cost_tracker.py
│   ├── judge/, metrics/, validators/
│   ├── baselines/replay/         # 골든셋 record/replay 기록 (gitignored 아님)
│   └── data/golden/, data/examples/
│
├── tests/                        # 테스트 (228 파일)
│   ├── conftest.py               # .env.test 격리 + 아웃바운드 소켓 차단 + 싱글턴 리셋
│   ├── unit/                     # 레이어별 단위 테스트
│   │   └── test_import_graph.py  # **순환 0 / 역방향 0 정적 검증**
│   ├── characterization/         # 동작 고정(특성화) 테스트
│   ├── eval/                     # 평가 + 골든셋 재생 게이트
│   ├── integration/, adversarial/, golden/
│
├── scripts/                      # 운영 스크립트
│   ├── start.py                  # Railway 배포용 시작
│   ├── record_golden_replay.py   # 골든셋 기록 (OPENAI_API_KEY 필요)
│   ├── sync_from_railway.py, sync_sheets_to_sqlite.py
│   └── evaluate_golden.py, export_dashboard.py, ...
│
├── data/                         # 런타임 데이터 (gitignored)
│   ├── amore_data.db, knowledge_graph.json
│   ├── dashboard_data.json, chroma/, market_intelligence/
│
├── docs/                         # 문서
│   ├── plans/                    # 리팩토링 계획 + 인계 프롬프트
│   ├── analysis/, architecture/, guides/, reports/
│   ├── research/, security/, diagrams/, refactoring/
│   └── dev/FUTURE_WORK.md        # 미해결 항목
│
├── pyproject.toml                # pytest, ruff, coverage(fail_under=75)
├── requirements.txt
├── Dockerfile                    # python:3.11-slim + Playwright Chromium
├── railway.toml
└── .env.example
```

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
├── application/      # Layer 2: Use Cases / Workflows + Services
├── infrastructure/   # Layer 3: Frameworks & Drivers (DI, 설정, 영속화)
└── api/ core/ agents/ rag/ ontology/ tools/   # Layer 4: 진입점·구현
```

> `src/adapters/` 는 빈 패키지여서 삭제됐다. 어댑터 역할은 `api/`(요청·응답)와
> `infrastructure/persistence/`(저장소)가 나눠 맡는다.

### Import 규칙 (의존성: 안쪽으로만)

| 규칙 | 상태 |
|------|------|
| `src` 모듈 간 **최상위 import 순환** | 0 (SCC 크기 1) |
| `domain` → `src.domain` · `src.shared` 외 | X |
| `application` → `src.api` | X |
| `tools` · `rag` · `ontology` → `src.api` | X |

> `tests/unit/test_import_graph.py` 가 `ast` 파싱으로 정적 검증한다(실행 없음).
> **최상위 import 만** 센다 — 함수 본문 안의 지연 import 는 이 코드베이스가 순환을
> 의도적으로 끊는 방식이라 실패시키지 않고 별도로 수집해 보고만 한다.

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
| DashboardAPI | `src/api/dashboard_api.py` | FastAPI 메인 서버 (진입점) |
| AppFactory | `src/api/app_factory.py` | 앱 초기화 |
| BatchWorkflow | `src/application/workflows/batch_workflow.py` | 일일 배치 파이프라인 |
| ChatWorkflow | `src/application/workflows/chat_workflow.py` | 챗 유스케이스 |
| UnifiedBrain | `src/core/brain.py` | 자율 에이전트 코어 |
| BrainScheduler | `src/core/brain_scheduler.py` | 스케줄 루프 (brain 에서 분리) |
| QueryGraph | `src/core/query_graph.py` | **단일 챗 경로** (스트림·비스트림 공통) |
| ReActAgent | `src/core/react_agent.py` | Self-Reflection (`ENABLE_REACT_AGENT`, 기본 off) |
| StateManager | `src/core/state_manager.py` | 시스템 상태 단일 출처 |
| HybridChatbot | `src/agents/hybrid_chatbot_agent.py` | AI 챗봇 |
| HybridInsight | `src/agents/hybrid_insight_agent.py` | 인사이트 생성 |
| AlertAgent | `src/agents/alert_agent.py` | 순위 변동 알림 |
| SuggestionEngine | `src/agents/suggestion_engine.py` | 후속 질문 생성 |
| SourceProvider | `src/agents/source_provider.py` | 출처 추출·포매팅 |
| HybridRetriever | `src/rag/hybrid_retriever.py` | RAG + KG 통합 검색 (파사드, 665줄) |
| Retriever | `src/rag/retriever.py` | 문서 검색 파사드 (676줄) |
| RetrievalStrategy | `src/rag/retrieval_strategy.py` | OWL + 인텐트 기반 전략 패턴 |
| ConfidenceFusion | `src/rag/confidence_fusion.py` | 다중 소스 신뢰도 융합 |
| ContextBuilder | `src/rag/context_builder.py` | 컨텍스트 조립 + 출처 등록 |
| KnowledgeGraph | `src/ontology/knowledge_graph.py` | Triple Store (JSON) |
| OntologyReasoner | `src/ontology/reasoner.py` | 규칙 기반 추론 |
| OWLReasoner | `src/ontology/owl_reasoner.py` | OWL 추론 (**배치 전용**) |
| Materializer | `src/ontology/materializer.py` | 추론 결과를 KG 에 물질화 (provenance 포함) |
| Thresholds | `src/ontology/thresholds.py` | `config/thresholds.json` 단일 출처 접근 |
| PromptRegistry | `prompts/registry.py` | 프롬프트 중앙 관리 |
| FeatureFlags | `src/infrastructure/feature_flags.py` | Feature flag (ENV > JSON > default) |
| Container | `src/infrastructure/container.py` | DI 컨테이너 |
| MetricCalculator | `src/tools/calculators/metric_calculator.py` | SoS, HHI, CPI |
| AmazonScraper | `src/tools/scrapers/amazon_scraper.py` | Playwright 크롤러 |
| KGBackup | `src/tools/utilities/kg_backup.py` | KG 백업 (7일 롤링) |
| EmailSender | `src/tools/notifications/email_sender.py` | Gmail SMTP |
| TelegramBot | `src/tools/notifications/telegram_bot.py` | Telegram 알림 |
| AgentLogger | `src/monitoring/logger.py` | 구조화 로깅 |
| units | `src/shared/units.py` | 퍼센트 ↔ 분수 변환 (단위 경계 단일 지점) |

> **단위 규약**: `MetricCalculator.calculate_sos` 와 `share_of_shelf`(저장·API)는 **퍼센트**,
> 온톨로지·KG 메타데이터·규칙은 **분수(0~1)**. 변환은 `src/shared/units.py` 한 곳에서만.

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
- **로깅**: 에이전트·워크플로우는 `AgentLogger`, 라이브러리 모듈은
  `logging.getLogger(__name__)`. CLI(`__main__`)와 docstring 예시의 `print` 는 유지

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
