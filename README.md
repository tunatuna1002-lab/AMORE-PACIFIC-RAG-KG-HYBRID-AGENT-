# AMORE Pacific RAG-Ontology Hybrid Agent

> Amazon US 베스트셀러를 매일 수집해 LANEIGE 브랜드의 경쟁력을 지표·챗봇·인사이트·알림으로 보여 주는 에이전트

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 무엇을 하나

| 기능 | 내용 |
|------|------|
| 일일 크롤링 | 매일 22:00 KST, Amazon Best Sellers 5개 카테고리 × Top 100 (Playwright + stealth) |
| KPI 계산 | SoS(Share of Shelf), HHI(시장 집중도), CPI(카테고리 평균가 대비 가격 지수, 100 기준) |
| AI 챗봇 | 문서·지식 그래프(KG)·크롤 DB 수치·온톨로지 사실을 증거 카드로 모아 답한다 (`POST /api/v4/chat`) |
| 인사이트·리포트 | LLM 기반 인사이트, DOCX 리포트·Excel 내보내기 (`/api/export/*`) |
| 알림 | 순위·SoS 급변 시 이메일(Gmail SMTP)·Telegram |

모니터링 카테고리: Beauty & Personal Care(`beauty`, L0) · Skin Care(`11060451`, L1) · Lip Care(`3761351`, L2) · Lip Makeup(`11059031`, L2) · Face Powder(`11058971`, L3). LANEIGE Lip Sleeping Mask는 Lip Makeup이 아니라 **Lip Care**(Skin Care 하위)다.

## Quick Start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # 런타임 + pytest
pip install -r requirements-dev.txt      # 선택: owlready2 (OWL 내보내기·Pellet 교차 검증 스크립트용)
playwright install chromium

# .env (최소)
OPENAI_API_KEY=sk-...
API_KEY=your-api-key                     # 보호 엔드포인트(챗봇·크롤 시작) 인증
AUTO_START_SCHEDULER=true                # 미설정 시 false → 22:00 자동 크롤 안 함

# API 서버 + 대시보드
uvicorn src.api.dashboard_api:app --host 0.0.0.0 --port 8001 --reload
# → http://localhost:8001/dashboard

# CLI
python3 main.py                          # 일일 워크플로우 1회 (크롤 → 저장 → 지표)
python3 main.py --categories lip_care    # 일부 카테고리만
python3 main.py --chat                   # 대화형 챗봇
python3 main.py --dry-run                # Google Sheets 저장 없이 실행
```

### 주요 API

| Method | Endpoint | 설명 | 인증 |
|--------|----------|------|------|
| GET | `/api/health` | 헬스체크 | - |
| GET | `/api/data` | 대시보드 데이터 JSON | - |
| GET | `/dashboard` | 대시보드 UI | - |
| POST | `/api/v4/chat` | 챗봇 (권장 경로, SSE: `/api/v4/chat/stream`) | API Key |
| POST | `/api/chat` | 챗봇 v1 (`HybridChatbotAgent`) | API Key |
| POST | `/api/crawl/start` | 크롤링 시작 | API Key |
| GET | `/api/v4/brain/status` | 스케줄러 상태, ReAct 활성 여부(`components`), 온톨로지 상태(`ontology`) | - |

## 아키텍처

### 데이터 흐름 (배치)

```
Amazon Best Sellers (5 카테고리 × Top 100, 22:00 KST)
  → CrawlerAgent (Playwright)
  → StorageAgent: Google Sheets 먼저, SQLite 다음 (병행 저장, 자동 동기화는 Sheets→SQLite 단방향)
  → MetricCalculator (SoS·HHI·CPI) → KG 갱신 (data/knowledge_graph.json) → 알림
```

SQLite(`data/amore_data.db`, Railway는 `/data/amore_data.db`)가 읽기 정본이다. exporter·지표·API는 SQLite를 읽는다.

### 질의 경로 (`/api/v4/chat` → `UnifiedBrain` → `QueryGraph`)

```
GUARD → CACHE_CHECK → GATHER_CONTEXT → ASSESS_CONFIDENCE
   ├─ HIGH                    → GENERATE_RESPONSE
   ├─ UNKNOWN                 → CLARIFICATION (되묻기)
   ├─ MEDIUM/LOW + 1홉        → DECIDE (DecisionMaker) → EXECUTE_TOOL → GENERATE_RESPONSE
   └─ MEDIUM/LOW + 2홉 이상   → REACT_AGENT (플래그 OFF면 DECIDE 경로)
GENERATE_RESPONSE → 수치 검증(annotate) → OUTPUT_GUARD → DONE
```

`GATHER_CONTEXT`는 `HybridRetriever.retrieve()`가 맡는다 (`src/rag/hybrid_retriever.py`).

1. **엔티티 연결**: 브랜드·카테고리·제품·지표 추출 (`entity_linker.py`). 온톨로지 플래그 ON이면 브랜드 등록부 사전을 더해 쓴다.
2. **온톨로지 질의 해석**: 그룹 → 소속 브랜드 전개, 세그먼트·원산지 등 정적 사실 카드 (`ontology_context.py`, 원본 `config/ontology/*.json` + `config/category_hierarchy.json`, 로더 `src/ontology/ontology.py`).
3. **KG 사실 조회**: JSON Triple Store (`src/ontology/knowledge_graph.py`).
4. **DB 지표 사실**: 크롤 SQLite의 날짜가 붙은 수치 (`metric_facts.py`).
5. **규칙 추론**: 위 사실을 증거 카드로 받아 규칙을 판정 (`src/ontology/reasoner.py` + `rule_contracts.py`, 규칙 37개).
6. **문서 검색**: ChromaDB Dense(`text-embedding-3-small`) + BM25 RRF, 추론 결과로 질의 확장.
7. **증거 카드 조립·선별** → 프롬프트 렌더링 (`evidence_assembly.py`, `evidence_renderer.py`, 모델 `src/domain/entities/evidence.py`).

그 뒤 단계:

- **신뢰도** (`src/core/confidence.py`): 엔티티 충족도 0.60 + 카드 종류 충족 0.40, 검색 분포는 ±0.05 동점 가르기. HIGH ≥ 0.95, MEDIUM ≥ 0.61, LOW ≥ 0.60, 그 아래는 UNKNOWN.
- **DecisionMaker** (`src/core/decision_maker.py`): MEDIUM/LOW일 때 네이티브 function calling으로 도구 1개를 고르거나 바로 답한다. 도구는 `src/core/tool_registry.py`의 5종(`resolve_entity`·`kg_neighbors`·`get_metrics`·`apply_rules`·`search_docs`)이고 ReAct와 공유한다.
- **수치 검증** (`src/core/numeric_verifier.py`): 답변 속 수치를 인용 카드와 대조한다. 기본 `annotate`는 결과를 메타데이터에 기록만 하고 답변은 바꾸지 않는다. ReAct·v1 경로는 이 단계를 거치지 않는다.

**표현 주의**: 런타임 추론은 Python 규칙 엔진과 JSON 온톨로지 로더(Python 폐포)다. OWL 추론기는 서비스에 연결된 적이 없어 삭제했고, OWL은 개발용 내보내기·Pellet 교차 검증(`scripts/export_ontology_owl.py`, `scripts/check_ontology_owl.py`)에만 쓴다. 규칙 37개 중 조사 시점에 발화 가능한 규칙은 13개였다(이력·감성·IR 입력 부족, [`docs/analysis/ontology-review-2026-09-18.md`](docs/analysis/ontology-review-2026-09-18.md) §3.3).

### 레이어

Clean Architecture: `src/domain`(엔티티·프로토콜) → `src/application`(워크플로우) → `src/adapters` → `src/infrastructure`(DI 컨테이너·설정·기능 플래그). 에이전트·RAG·온톨로지·도구는 `src/agents`, `src/rag`, `src/ontology`, `src/tools`에 있다. 전체 구조는 [`CLAUDE.md`](CLAUDE.md) §4.

## 기능 플래그

우선순위는 `ENV(FF_{SECTION}_{KEY})` > `config/feature_flags.json` > 코드 기본값이다 (`src/infrastructure/feature_flags.py`). 아래 "기본"은 저장소의 `config/feature_flags.json` 값이다.

| 플래그 | 기본 | 의미 | 근거 |
|--------|------|------|------|
| `ontology.use_class_reasoning` | **ON** | 온톨로지 질의 확장·정적 사실 카드·등록부 사전 (코드 기본값은 False) | OA-10, 측정 후 켬 |
| `agents.use_react_agent` | OFF | MEDIUM/LOW + 2홉 이상 질문을 ReAct 루프(최대 5회)로 처리 | S6-3, 비교 후 OFF 유지 |
| `agents.react_shadow_mode` | OFF | 답은 파이프라인이 만들고 ReAct는 기록만 | S6-3 |
| `agents.react_bypass_confidence` | OFF | 측정용: HIGH 관문을 건너뛰고 ReAct 진입 | S6-3 |
| `response.numeric_verification_mode` | `annotate` | `off` / `annotate`(기록만) / `enforce`(불일치 수치 표시) | S6-4, enforce 보류 |
| `kg.write_validation` | `warn` | KG 쓰기 검증 `off` / `warn`(로그만) / `enforce`(차단) | OA-7 |
| `reasoner.enabled` | ON | 규칙 추론 on/off. 옛 이름 `reasoner.use_owl_reasoner` (별칭, 1회 경고) | §O6-3 |
| `reasoner.use_unified_reasoner` | ON | 이것 또는 `reasoner.enabled`가 켜져 있으면 규칙을 판정한다 | — |
| `kg.enabled` | ON | KG 조회 on/off. 옛 이름 `ontology.use_ontology_kg` (별칭) | §O6-3 |
| `retriever.use_db_metric_facts` | ON | 크롤 DB 수치를 증거 카드로 싣기 | — |
| `retriever.use_confidence_fusion` | ON | 다중 소스 신뢰도 융합 | — |
| `retriever.use_reranker` | OFF | 재순위화 (코드 기본값은 True, JSON이 끔) | — |
| `agents.use_query_rewriter` / `agents.use_external_signals` | ON | 질의 재작성 / 외부 신호(뉴스 등) 사용 | — |
| `prompts.use_centralized_prompts` | ON | `prompts/registry.py`에서 프롬프트 로드 | — |
| `cache.use_sqlite_embedding_cache` | OFF | 임베딩 캐시를 SQLite에 영속화 | — |
| `router.use_llm_fallback` | OFF | 라우터 LLM 폴백 | — |

결정 문서: S6-x는 [`docs/plans/evidence-react-ontology-decisions-2026-09.md`](docs/plans/evidence-react-ontology-decisions-2026-09.md), OA-x는 [`docs/plans/ontology-activation-decisions-2026-09.md`](docs/plans/ontology-activation-decisions-2026-09.md), §O6-3은 [`docs/experiments/ontology_activation_2026-09.md`](docs/experiments/ontology_activation_2026-09.md).

## 평가와 측정 결과

골든셋 + 유형별 시험지(numeric·relation·rule·multihop)를 LLM judge로 채점한다. 판정은 "평균 차이가 노이즈 기준 이상이고 반복 실행 범위가 겹치지 않을 때만 차이 있음"이다.

| 실험 | 비교 | 대표 결과 | 문서 |
|------|------|-----------|------|
| 온톨로지 플래그 (O7) | 54문항(multihop+relation) × 3회, OFF → ON | 종합 0.692 → 0.745, L3 골드 엣지 recall(canonical) 0.301 → 0.582, 근거성 0.867 → 0.968, 수치 정확도 0.601 → 0.733. 나머지 29문항 회귀 없음 | [`ontology_activation_2026-09.md`](docs/experiments/ontology_activation_2026-09.md) §O7·§요약 |
| 온톨로지 플래그 (O7) | rule 42문항 × 2회 | 규칙 정답 일치율 0.781 → 0.906 | 같은 문서 §O7-5 |
| 온톨로지 비용 | 같은 54문항 | 프롬프트 카드 58.1 → 69.9장/문항, 파이프라인 비용 +9%, 지연 차이 없음 | 같은 문서 §O7-8 |
| 증거 카드 (2단계) | 233문항 × 3회, 기준선 → 2단계 | numeric 수치 정확도 0.000 → 0.700, 전체 종합 0.656 → 0.698. 대가: 답변 입력 토큰 ×4.5, 지연 +30% | [`evidence_pipeline_2026-09.md`](docs/experiments/evidence_pipeline_2026-09.md) 2단계 |
| 규칙 추론 (3단계) | rule 42문항, 규칙 on vs off × 3회 | 규칙 정답 일치율 0.469 → 0.779. 반면 judge 종합 점수는 규칙 off가 0.022 높음 | 같은 문서 3단계 |
| KG 제거 | 130문항 × 3회, full vs KG off | KG off 시 근거성 0.717 → 0.541. 일부는 채점 컨텍스트 영향(교차 채점으로 분해). 관련성·토큰 F1·수치 정확도는 차이 없음 | [`kg_ablation_2026-09.md`](docs/experiments/kg_ablation_2026-09.md) |

ReAct는 6단계 비교에서 켜기 조건을 채우지 못했다(관련성 하락, 비용 +46%, 지연 +13%) → S6-3.

### 평가 실행

```bash
# v4 = 대시보드 Brain 경로 (기본 target v1 = /api/chat)
.venv/bin/python -m eval.cli run --dataset eval/data/golden/laneige_golden_v2.jsonl --target v4 \
  --data-as-of 2026-08-31 --judge llm --semantic-similarity --concurrency 4

python3 scripts/evaluate_golden.py --verbose   # 기존 골든셋 스크립트
```

`eval.cli`에는 `run` 외에 `compare`·`set-baseline`·`portfolio`·`ablation` 하위 명령이 있다 (`eval/cli.py`). OpenAI API 비용이 든다.

## 테스트

```bash
python3 -m pytest tests/ -q --no-cov          # 전체
python3 -m pytest tests/unit/ -v              # 단위 테스트만
python3 -m pytest tests/ -m "not slow" -v     # 느린 테스트 제외
```

- 최신 전체 실행: **6,361 passed / 8 skipped / 0 failed** (2026-09-18, 로컬 전체 실행).
- 테스트 환경 변수는 `ENV_FILE`(기본 `.env.test`)로 분리한다 (`tests/conftest.py`).
- CI(`.github/workflows/test.yml`): Python 3.11에서 Ruff(실패 허용) → 단위 테스트 + 커버리지 → API 키가 있을 때 통합 테스트(실패 허용), 별도 job으로 Bandit·pip-audit. `main`의 `b7f7450` 실행은 통과했다.
- 커버리지는 측정하지만 강제하지 않는다 (`pyproject.toml` `fail_under = 0`).
- CI는 `requirements-dev.txt`를 설치하지 않아 Pellet 교차 검증 테스트는 skip된다.

## 배포 (Railway / Docker)

- `Dockerfile`: `python:3.11-slim` + Playwright Chromium + 한글 폰트, `CMD ["python", "scripts/start.py"]`.
- `scripts/start.py`: 온톨로지 원본을 먼저 로드(형식 오류면 종료) → Chroma 색인 빌드(실패해도 계속) → `PORT` 환경변수(기본 8001)로 uvicorn 시작.
- `railway.toml`: healthcheck `/api/health`(300초), 재시작 `on_failure` 최대 3회. Volume `/data`에 SQLite·KG를 둔다.

```bash
docker build -t amore-agent .
docker run -p 8001:8001 -e OPENAI_API_KEY=sk-... -e API_KEY=... amore-agent
```

| 환경 변수 | 용도 |
|-----------|------|
| `OPENAI_API_KEY` | 필수 (LLM·임베딩) |
| `API_KEY`, `AUTO_START_SCHEDULER` | 보호 엔드포인트 인증, 스케줄러 자동 시작 |
| `ALLOWED_HOSTS` | TrustedHost 허용 목록 (기본 `localhost,127.0.0.1,.railway.app`). Railway healthcheck는 내부 IP로 오므로 `*` 필요 |
| `GOOGLE_SHEETS_SPREADSHEET_ID` / `GOOGLE_SPREADSHEET_ID`, `GOOGLE_SHEETS_CREDENTIALS_JSON` | Sheets 저장. 모듈마다 ID 변수 이름이 다르다(`sheets_writer.py`는 앞, `config_manager.py`·`sheets_repository.py`는 뒤) |
| `TAVILY_API_KEY`, `GNEWS_API_KEY`, `DATA_GO_KR_API_KEY` | 외부 신호(뉴스·관세청/식약처), 선택 |
| `SMTP_SERVER`, `SMTP_PORT`, `SENDER_EMAIL`, `SENDER_PASSWORD`, `ALERT_RECIPIENTS` | 이메일 알림, 선택 |
| `TELEGRAM_BOT_TOKEN`, `TELEGRAM_ADMIN_CHAT_ID` | Telegram 알림, 선택 |

운영 명령:

```bash
python3 scripts/sync_from_railway.py              # Railway → 로컬
python3 scripts/sync_sheets_to_sqlite.py          # Sheets → SQLite
python3 -m src.tools.utilities.kg_backup backup   # KG 백업 (data/backups/kg/, 7일 롤링)
python3 -m src.tools.utilities.kg_backup list
```

## 문서 지도

| 문서 | 내용 |
|------|------|
| [`CLAUDE.md`](CLAUDE.md), [`AGENTS.md`](AGENTS.md) | 개발 컨텍스트, 전체 디렉토리 구조, 모듈 참조 |
| [`docs/portfolio/amore_architecture_evidence.md`](docs/portfolio/amore_architecture_evidence.md) | 주장별 근거(과장 금지 항목 포함) |
| [`docs/analysis/ontology-review-2026-09-18.md`](docs/analysis/ontology-review-2026-09-18.md) | 온톨로지 현황 검토 (규칙 발화 가능성, KG 품질) |
| [`docs/experiments/ontology_activation_2026-09.md`](docs/experiments/ontology_activation_2026-09.md) | 온톨로지 작동 O0~O7 측정 |
| [`docs/experiments/evidence_pipeline_2026-09.md`](docs/experiments/evidence_pipeline_2026-09.md) | 증거 카드·규칙 추론·ReAct 0~6단계 |
| [`docs/experiments/kg_ablation_2026-09.md`](docs/experiments/kg_ablation_2026-09.md) | KG·규칙 추론 제거 실험 |
| [`docs/experiments/eval_v4_baseline_2026-09-17.md`](docs/experiments/eval_v4_baseline_2026-09-17.md) | v4 경로 평가 기준선 |
| [`docs/plans/evidence-react-ontology-decisions-2026-09.md`](docs/plans/evidence-react-ontology-decisions-2026-09.md), [`docs/plans/ontology-activation-decisions-2026-09.md`](docs/plans/ontology-activation-decisions-2026-09.md), [`docs/plans/risk-remediation-decisions-2026-09-17.md`](docs/plans/risk-remediation-decisions-2026-09-17.md) | 결정 기록 |
| [`docs/dev/FUTURE_WORK.md`](docs/dev/FUTURE_WORK.md) | 남은 일 (9.9·9.10이 최신) |
| [`docs/REFACTORING_RESULTS.md`](docs/REFACTORING_RESULTS.md) | 2026-02 리팩토링 결과 |
| [`docs/guides/react_agent_guide.md`](docs/guides/react_agent_guide.md), [`docs/embedding_cache_guide.md`](docs/embedding_cache_guide.md), [`docs/AMOREPACIFIC_DESIGN_SYSTEM.md`](docs/AMOREPACIFIC_DESIGN_SYSTEM.md) | 모듈·디자인 가이드 |

## 한계와 남은 일

자세한 목록은 [`docs/dev/FUTURE_WORK.md`](docs/dev/FUTURE_WORK.md) 9.9·9.10.

- **프롬프트 카드 수가 한계선(~70장/문항)에 닿았다.** 온톨로지 ON에서 평균 69.9장. 증거 선별을 좁히기 전에는 신뢰도 신호의 변별력도 낮다(5단계 게이트에서 HIGH·LOW 실패율 차이 없음).
- **ReAct는 구현돼 있지만 기본 OFF다.** 신뢰도 관문 뒤에 있어 2홉 질문 대부분이 진입하지 않고, 토큰 예산(12,000)이 실제 사용량보다 작다.
- **수치 검증기는 기록만 한다(annotate).** 인용 파싱 결함(`[M-a], [M-b]`)과 계산값 처리 때문에 enforce는 보류했다. ReAct·v1 경로에는 적용되지 않는다.
- **스크레이퍼 브랜드 오귀속**: 일부 제품에 가짜 브랜드(`unknown`·`fresh`·`chi` 등)가 붙어 KG `competesWith`에 섞인다. 조회 쪽에서 placeholder 브랜드만 거른다.
- **날짜 없는 KG 수치 엣지**가 남아 있다(평가 스냅샷 기준 333건). 날짜를 지어내지 않기로 했고, 이 때문에 `kg.write_validation=enforce`를 켜지 못한다.
- **KG 효과는 일부만 입증됐다.** KG 제거 시 근거성 하락은 확인했지만 일부는 채점 방식 영향이며, 관련성·수치 정확도 차이는 없었다.

## 변경 이력 요약

- 2026-01: 크롤러·KG·규칙 추론·하이브리드 검색·대시보드 초기 구축.
- 2026-02: 10-스프린트 로드맵(모놀리스 분해, 순환 의존성 제거, DI, 보안) — [`docs/REFACTORING_RESULTS.md`](docs/REFACTORING_RESULTS.md), [`docs/plans/roadmap-progress.md`](docs/plans/roadmap-progress.md).
- 2026-08: 사실 검증 감사, 평가 하네스 사이클 — [`docs/experiments/`](docs/experiments/).
- 2026-09: 증거 카드 파이프라인, 규칙 추론 연결, 신뢰도 재설계, 온톨로지 JSON 원본화·OWL 모듈 삭제, 온톨로지 플래그 기본 ON — 위 실험·결정 문서.

## 라이선스

MIT License
