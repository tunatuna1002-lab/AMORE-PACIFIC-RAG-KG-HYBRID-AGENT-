# AMORE Pacific RAG-KG 하이브리드 에이전트

> **RAG (Retrieval-Augmented Generation)**와 **Knowledge Graph** 기술을 결합한 Amazon 마켓플레이스 분석 및 인사이트 생성을 위한 지능형 AI 에이전트 시스템

[English README](./README.md)

---

## 개요

이 프로젝트는 **AMORE Pacific의 LANEIGE 브랜드**를 위해 Amazon US 마켓플레이스에서 실시간 경쟁 인텔리전스를 제공하기 위해 개발되었습니다. 시스템은 매일 Amazon 베스트셀러 순위를 크롤링하고, 전략적 KPI를 계산하며, 하이브리드 AI 접근법을 통해 실행 가능한 인사이트를 생성합니다.

### 왜 RAG + Knowledge Graph 하이브리드인가?

기존 RAG 시스템은 관련 문서를 검색하지만 **구조화된 추론 능력**이 부족합니다. 순수 Knowledge Graph 접근법은 강력한 추론 능력을 갖지만 **자연어 이해**에 한계가 있습니다. 우리의 하이브리드 접근법은 두 가지를 결합합니다:

```
┌─────────────────────────────────────────────────────────────────┐
│                    사용자 질문                                    │
└─────────────────────┬───────────────────────────────────────────┘
                      │
          ┌───────────▼───────────┐
          │    질문 라우터         │
          │  (의도 분류)           │
          └───────────┬───────────┘
                      │
        ┌─────────────┴─────────────┐
        │                           │
┌───────▼───────┐         ┌────────▼────────┐
│  RAG 파이프라인│         │  Knowledge Graph │
│               │         │                 │
│ • 문서 검색   │         │ • 엔티티 관계   │
│ • 시맨틱 검색 │         │ • 온톨로지 추론 │
│ • 컨텍스트    │         │ • 비즈니스 규칙 │
│   구성        │         │                 │
└───────┬───────┘         └────────┬────────┘
        │                           │
        └─────────────┬─────────────┘
                      │
          ┌───────────▼───────────┐
          │    LLM 오케스트레이터  │
          │   (응답 생성)          │
          └───────────┬───────────┘
                      │
          ┌───────────▼───────────┐
          │    구조화된 응답       │
          │  + 실행 가능한 인사이트│
          └───────────────────────┘
```

---

## 주요 기능

### 1. 멀티 에이전트 아키텍처
- **Crawler Agent**: Amazon 베스트셀러 순위 스크래핑 (카테고리당 Top 100)
- **Storage Agent**: 버전 관리와 함께 Google Sheets에 데이터 저장
- **Metrics Agent**: 10개 이상의 전략적 KPI 계산 (SoS, HHI, CPI 등)
- **Insight Agent**: AI 기반 일일 인사이트 생성
- **Chatbot Agent**: 대화 메모리를 갖춘 대화형 Q&A
- **Alert Agent**: 임계값 모니터링 및 알림 발송

### 2. RAG 시스템
- **Document Retriever**: 비즈니스 문서에 대한 시맨틱 검색
- **Query Router**: 의도 분류 (정의, 해석, 분석 등)
- **Context Builder**: 질문 유형에 따른 동적 컨텍스트 조립
- **Hybrid Retriever**: 키워드와 시맨틱 검색 결합

### 3. Knowledge Graph & 온톨로지
- **엔티티 유형**: Brand, Product, Category, Competitor
- **메트릭 온톨로지**: ProductMetrics, BrandMetrics, MarketMetrics
- **비즈니스 규칙 엔진**: 설정 가능한 임계값 기반 규칙
- **Reasoner**: 인사이트 도출을 위한 추론 엔진

### 4. 대시보드 & API
- **FastAPI 백엔드**: 다중 버전 RESTful API (v1, v2, v3)
- **인터랙티브 대시보드**: KPI 실시간 시각화
- **DOCX 내보내기**: 전문적인 인사이트 리포트 생성
- **Audit Trail**: 모든 챗봇 상호작용 완전 로깅

---

## 프로젝트 구조

```
AMORE-RAG-ONTOLOGY-HYBRID AGENT/
├── src/                          # 소스 코드
│   ├── agents/                   # AI 에이전트
│   │   ├── crawler_agent.py      # Amazon 스크래핑
│   │   ├── storage_agent.py      # Google Sheets 연동
│   │   ├── metrics_agent.py      # KPI 계산
│   │   ├── insight_agent.py      # AI 인사이트 생성
│   │   ├── chatbot_agent.py      # 대화형 AI
│   │   ├── alert_agent.py        # 임계값 모니터링
│   │   ├── hybrid_insight_agent.py   # RAG+KG 하이브리드 인사이트
│   │   └── hybrid_chatbot_agent.py   # RAG+KG 하이브리드 챗봇
│   │
│   ├── core/                     # 코어 오케스트레이션
│   │   ├── unified_orchestrator.py   # 메인 오케스트레이터
│   │   ├── llm_orchestrator.py   # LLM 조정
│   │   ├── brain.py              # 의사결정
│   │   ├── rules_engine.py       # 비즈니스 규칙
│   │   ├── simple_chat.py        # 단순화된 채팅 서비스
│   │   └── crawl_manager.py      # 백그라운드 크롤링 관리
│   │
│   ├── rag/                      # RAG 컴포넌트
│   │   ├── retriever.py          # 문서 검색
│   │   ├── router.py             # 질문 분류
│   │   ├── context_builder.py    # 컨텍스트 조립
│   │   ├── hybrid_retriever.py   # 하이브리드 검색
│   │   └── templates.py          # 프롬프트 템플릿
│   │
│   ├── ontology/                 # Knowledge Graph
│   │   ├── schema.py             # 엔티티 정의
│   │   ├── knowledge_graph.py    # 그래프 연산
│   │   ├── reasoner.py           # 추론 엔진
│   │   ├── business_rules.py     # 규칙 정의
│   │   └── relations.py          # 관계 유형
│   │
│   ├── memory/                   # 세션 & 컨텍스트
│   │   ├── session.py            # 세션 관리
│   │   ├── history.py            # 실행 이력
│   │   └── context.py            # 컨텍스트 추적
│   │
│   ├── monitoring/               # 관측성
│   │   ├── logger.py             # 구조화된 로깅
│   │   ├── tracer.py             # 실행 트레이싱
│   │   └── metrics.py            # 품질 메트릭
│   │
│   └── tools/                    # 외부 연동
│       ├── amazon_scraper.py     # Amazon API 래퍼
│       ├── sheets_writer.py      # Google Sheets API
│       ├── dashboard_exporter.py # JSON 내보내기
│       ├── metric_calculator.py  # KPI 수식
│       └── email_sender.py       # 알림 발송
│
├── dashboard/                    # 프론트엔드
│   ├── amore_unified_dashboard_v4.html
│   └── test_chat.html
│
├── docs/                         # 문서화
│   ├── architecture/             # 시스템 설계
│   │   ├── LLM_ORCHESTRATOR_DESIGN.md
│   │   └── *.xml (다이어그램)
│   └── guides/                   # 비즈니스 가이드
│       ├── Strategic Indicators Definition.md
│       ├── Metric Interpretation Guide.md
│       └── Indicator Combination Playbook.md
│
├── config/                       # 설정
│   └── thresholds.json           # 알림 임계값
│
├── data/                         # 런타임 데이터
│   ├── dashboard_data.json       # 대시보드 상태
│   └── knowledge_graph.json      # 영속화된 KG
│
├── main.py                       # CLI 진입점
├── dashboard_api.py              # FastAPI 서버
├── orchestrator.py               # 워크플로우 오케스트레이터
└── export_dashboard.py           # 데이터 내보내기 스크립트
```

---

## 개발 여정

### Phase 1: 기반 구축 (에이전트 & 크롤링)
Amazon 베스트셀러 데이터를 크롤링하기 위한 간단한 에이전트 아키텍처로 시작했습니다. 각 에이전트는 단일 책임 원칙에 따라 설계되었습니다:
- Crawler가 원시 HTML을 가져옴
- Parser가 구조화된 데이터를 추출
- Storage가 Google Sheets에 저장

### Phase 2: 분석 (메트릭 & KPI)
이커머스 분석 모범 사례를 기반으로 전략적 KPI 계산을 추가했습니다:
- **SoS (Share of Shelf)**: Top 100에서 브랜드 가시성
- **HHI (Herfindahl-Hirschman Index)**: 시장 집중도
- **CPI (Competitive Position Index)**: 상대적 포지셔닝
- **Volatility**: 시간에 따른 순위 안정성

### Phase 3: AI 통합 (LLM + RAG)
자연어 인사이트를 위해 OpenAI GPT 모델을 통합했습니다:
- 메트릭에서 일일 인사이트 생성
- 대화형 Q&A를 위한 챗봇
- 비즈니스 문서에 응답을 근거하기 위한 RAG

### Phase 4: Knowledge Graph (온톨로지)
구조화된 지식 표현을 추가했습니다:
- 엔티티-관계 모델링
- 비즈니스 규칙 추론
- RAG + KG를 결합한 하이브리드 추론

### Phase 5: 프로덕션 강화
- 다중 버전 API (v1 레거시, v2 오케스트레이터, v3 단순화)
- 백그라운드 크롤링 관리
- Audit Trail 로깅
- DOCX 리포트 생성

---

## 설치

### 사전 요구사항
- Python 3.10+
- Google Cloud 자격 증명 (Sheets API용)
- OpenAI API 키

### 설정

```bash
# 레포지토리 클론
git clone https://github.com/tunatuna1002-lab/AMORE-PACIFIC-RAG-KG-HYBRID-AGENT-.git
cd AMORE-PACIFIC-RAG-KG-HYBRID-AGENT-

# 가상 환경 생성
python -m venv venv
source venv/bin/activate  # Windows: `venv\Scripts\activate`

# 의존성 설치
pip install -r requirements.txt

# 환경 설정
cp .env.example .env
# .env 파일에 API 키 입력
```

### 환경 변수

```env
OPENAI_API_KEY=sk-...
GOOGLE_SHEETS_SPREADSHEET_ID=...
GOOGLE_APPLICATION_CREDENTIALS=./credentials.json
```

---

## 사용법

### 일일 워크플로우 (배치 처리)
```bash
# 전체 일일 워크플로우 실행
python main.py

# 특정 카테고리만 실행
python main.py --categories lip_care face_moisturizer

# 드라이 런 (Google Sheets 쓰기 없음)
python main.py --dry-run
```

### 인터랙티브 챗봇
```bash
python main.py --chat
```

### 대시보드 API 서버
```bash
# FastAPI 서버 시작
python dashboard_api.py

# 또는 uvicorn 사용
uvicorn dashboard_api:app --host 0.0.0.0 --port 8001 --reload
```

### 대시보드 데이터 내보내기
```bash
python export_dashboard.py
```

---

## API 엔드포인트

| 엔드포인트 | 메소드 | 설명 |
|----------|--------|-------------|
| `/api/data` | GET | 대시보드 데이터 조회 |
| `/api/chat` | POST | 챗봇 (v1 - RAG 전용) |
| `/api/v2/chat` | POST | 챗봇 (v2 - 통합 오케스트레이터) |
| `/api/v3/chat` | POST | 챗봇 (v3 - 단순화) |
| `/api/v2/stats` | GET | 오케스트레이터 통계 |
| `/api/crawl/status` | GET | 크롤링 상태 |
| `/api/crawl/start` | POST | 수동 크롤링 시작 |
| `/api/export/docx` | POST | DOCX 리포트 생성 |

---

## 전략적 KPI

| KPI | 설명 | 수식 |
|-----|-------------|---------|
| **SoS** | 점유율 (Share of Shelf) | Top 100 내 브랜드 제품 수 / 100 |
| **HHI** | 시장 집중도 | Σ(시장 점유율²) |
| **CPI** | 경쟁 포지션 지수 | 가중 순위 점수 |
| **Volatility** | 순위 안정성 | 순위 변동의 표준편차 |
| **Top10 Count** | 프리미엄 가시성 | Top 10 내 제품 수 |
| **Avg Rank** | 평균 순위 | 브랜드 제품 평균 순위 |

---

## 기술 스택

- **백엔드**: Python, FastAPI, asyncio
- **LLM**: OpenAI GPT-4.1-mini (LiteLLM 경유)
- **RAG**: 시맨틱 검색을 갖춘 커스텀 리트리버
- **Knowledge Graph**: JSON 영속화를 갖춘 커스텀 구현
- **스토리지**: Google Sheets API
- **스크래핑**: BeautifulSoup, httpx
- **프론트엔드**: HTML/CSS/JS (바닐라)

---

## 아키텍처 원칙

1. **에이전트 기반 설계**: 각 에이전트는 단일 책임을 가짐
2. **Think-Act-Observe 루프**: ReAct 패턴에서 영감
3. **하이브리드 인텔리전스**: 구조화된(KG) 지식과 비구조화된(RAG) 지식 결합
4. **우아한 성능 저하**: 각 컴포넌트에 대한 폴백 전략
5. **관측성**: 포괄적인 로깅 및 트레이싱

---

## 기여하기

기여를 환영합니다! PR을 제출하기 전에 기여 가이드라인을 읽어주세요.

---

## 라이선스

이 프로젝트는 AMORE Pacific을 위해 개발된 독점 소프트웨어입니다.

---

## 감사의 글

- AMORE Pacific 디지털 혁신팀
- GPT 모델을 위한 OpenAI
- Claude를 통한 개발 지원을 위한 Anthropic
