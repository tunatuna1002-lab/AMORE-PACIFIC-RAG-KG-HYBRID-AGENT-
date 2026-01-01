# AMORE-RAG-ONTOLOGY-HYBRID AGENT 최종 계획서

## 1. 프로젝트 개요

### 1.1 목표
미국 Amazon 베스트셀러에서 라네즈(LANEIGE) 및 경쟁사 제품 순위를 매일 트래킹하고,
자동화된 인사이트를 생성하는 AI 에이전트 시스템

### 1.2 핵심 기능
- Amazon 베스트셀러 Top 100 크롤링 (5개 카테고리)
- Google Sheets에 일자별/카테고리별 히스토리 저장
- 10개 전략 지표 자동 계산
- 규칙 기반 인사이트 자동 생성
- RAG 기반 챗봇 질의응답 (대시보드 연동)

### 1.3 인사이트 예시
> "라네즈 립 슬리핑 마스크, 런칭 후 2개월간 Amazon 뷰티 전체 카테고리에서 TOP 5,
> 립케어 카테고리에서 1위 5주 연속 달성"

---

## 2. 기술 스택

### 2.1 LLM 선택: GPT-4.1-mini (단독)

| 항목 | 결정 |
|------|------|
| **모델** | `openai/gpt-4.1-mini` |
| **월 예상 비용** | ~$5 |
| **선택 이유** | 성능+비용 균형, 1M 컨텍스트, 단일 모델 단순함 |

#### 선택 근거

| 비교 항목 | GPT-4.1-mini | GPT-4o-mini | Gemini Flash |
|-----------|--------------|-------------|--------------|
| 월 비용 | $5 | $2 | $1 |
| MMLU | ~84% | 82% | 77.9% |
| 컨텍스트 | **1M** | 128K | 1M |
| 복잡 분석 | 좋음 | 보통 | 보통 |

### 2.2 전체 기술 스택

| 항목 | 기술 |
|------|------|
| 에이전트 프레임워크 | Google ADK |
| LLM | OpenAI GPT-4.1-mini (via LiteLLM) |
| 크롤링 | Playwright |
| 저장소 | Google Sheets API |
| RAG | 벡터 검색 (4개 MD 파일 기반) |
| 언어 | Python 3.11+ |

---

## 3. 크롤링 대상

### 3.1 카테고리 (5개)

| 카테고리 | URL | 비고 |
|----------|-----|------|
| Beauty 전체 | `/zgbs/beauty` | 최상위 |
| Skin Care | `/zgbs/beauty/11060451` | 라네즈 주력 |
| Lip Care | `/zgbs/beauty/3761351` | 립 슬리핑 마스크 |
| Lip Makeup | `/zgbs/beauty/11059031` | 립 글로이 밤 |
| Face Powder | `/zgbs/beauty/11058971` | 쿠션 등 |

### 3.2 크롤링 범위
- **Top 100 전체** (경쟁사 분석 포함)
- **실행 주기**: 매일 1회

---

## 4. 시스템 아키텍처

### 4.1 에이전트 구조 (ADK + LiteLLM)

```
┌─────────────────────────────────────────────────────────┐
│                    Orchestrator                          │
│              (Think → Act → Observe 루프)                │
│                   [GPT-4.1-mini]                         │
└─────────────────┬───────────────────────────────────────┘
                  │ 순차 실행
    ┌─────────────┼─────────────┬─────────────┐
    ▼             ▼             ▼             ▼
┌────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
│Crawler │  │ Storage  │  │ Metrics  │  │ Insight  │
│ Agent  │  │  Agent   │  │  Agent   │  │  Agent   │
│        │  │          │  │          │  │          │
│4.1-mini│  │ 4.1-mini │  │ 4.1-mini │  │ 4.1-mini │
└────────┘  └──────────┘  └──────────┘  └──────────┘
    │             │             │             │
    ▼             ▼             ▼             ▼
┌────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
│Amazon  │  │ Google   │  │ Metric   │  │ RAG +    │
│Scraper │  │ Sheets   │  │Calculator│  │Templates │
└────────┘  └──────────┘  └──────────┘  └──────────┘


┌─────────────────────────────────────────────────────────┐
│                 Dashboard Chatbot                        │
│                   [GPT-4.1-mini]                         │
│        RAG 기반 질의응답 + 전략 분석 보고서               │
└─────────────────────────────────────────────────────────┘
```

### 4.2 에이전트 역할

| Agent | 역할 | 입력 | 출력 |
|-------|------|------|------|
| `crawler_agent` | Amazon Top 100 크롤링 | 카테고리 URL 5개 | 제품 데이터 리스트 |
| `storage_agent` | Google Sheets 저장 | 크롤링 데이터 | 저장 완료 상태 |
| `metrics_agent` | 10개 지표 계산 | Sheets 데이터 | 계산된 지표 |
| `insight_agent` | 인사이트 생성 | 지표 + RAG | 인사이트 텍스트 |
| `orchestrator` | 전체 조율 + Observe | Worker 결과 | 최종 리포트 |
| `chatbot` | 대시보드 질의응답 | 사용자 질문 | 분석/보고서 |

### 4.3 LiteLLM 연동 코드

```python
from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
import os

# API 키 설정
os.environ["OPENAI_API_KEY"] = "your-openai-api-key"

# 모든 에이전트 동일 모델 사용
model = LiteLlm(model="openai/gpt-4.1-mini")

crawler_agent = LlmAgent(model=model, name="crawler_agent", ...)
storage_agent = LlmAgent(model=model, name="storage_agent", ...)
metrics_agent = LlmAgent(model=model, name="metrics_agent", ...)
insight_agent = LlmAgent(model=model, name="insight_agent", ...)
chatbot_agent = LlmAgent(model=model, name="chatbot_agent", ...)
```

### 4.4 Orchestrator - Think, Act, Observe 루프

```python
# orchestrator.py 핵심 로직
for agent in [crawler, storage, metrics, insight]:
    # Think: 다음 행동 계획
    plan = think(current_state, agent)

    # Act: 에이전트 실행
    result = agent.execute(plan)

    # Observe: 결과 검증
    if not observe(result):
        handle_error(result)  # 재시도 또는 알림

    # 상태 업데이트
    update_state(result)
```

---

## 5. 데이터 모델

### 5.1 온톨로지 (5개 엔티티)

```
Brand ─────────────────────────────────────────┐
   │                                           │
   ▼                                           │
Product (asin, product_name, first_seen_date)  │
   │                                           │
   ├──────────────────────────────────────┐    │
   ▼                                      ▼    ▼
Category                               Snapshot (date)
                                          │
                                          ▼
                                    RankRecord
                                    - rank
                                    - price
                                    - rating
                                    - reviews_count
                                    - badge
```

### 5.2 수집 필드 (Amazon)

| 필드 | 설명 | 예시 |
|------|------|------|
| `snapshot_date` | 수집 일자 | 2025-01-15 |
| `category` | 카테고리명 | Lip Care |
| `rank` | 순위 | 1 |
| `asin` | Amazon 제품 ID | B08XYZ123 |
| `product_name` | 제품명 | LANEIGE Lip Sleeping Mask |
| `brand` | 브랜드 | LANEIGE |
| `price` | 가격 | $24.00 |
| `rating` | 평점 | 4.7 |
| `reviews_count` | 리뷰 수 | 89,234 |
| `product_url` | 상품 URL | https://amazon.com/dp/... |
| `badge` | 뱃지 | Best Seller, Amazon's Choice |
| `first_seen_date` | 최초 발견일 | 2025-01-01 |
| `launch_date` | 실제 런칭일 (선택) | 2024-11-15 |

### 5.3 런칭일 처리 (하이브리드 방식)

| 상황 | 처리 |
|------|------|
| 신규 발견 제품 | `first_seen_date` 자동 기록 |
| 주요 제품 (라네즈) | 상세 페이지에서 `launch_date` 1회 수집 |
| 기존 제품 | `first_seen_date = null` |

**인사이트 표현:**
- 런칭일 있을 때: "런칭 후 2개월간 TOP 5"
- 런칭일 없을 때: "최근 60일간 TOP 5 유지"

---

## 6. 전략 지표 (10개)

### 6.1 Level 1: Market & Brand

| 지표 | 정의 | 산출식 |
|------|------|--------|
| **SoS** | 브랜드 노출 점유율 | (브랜드 제품 수 / Top N) × 100 |
| **HHI** | 시장 집중도 | Σ(각 브랜드 SoS)² |
| **Brand Avg Rank** | 브랜드 평균 순위 | Σ(제품 순위) / 제품 수 |

### 6.2 Level 2: Category & Price

| 지표 | 정의 | 산출식 |
|------|------|--------|
| **CPI** | 가격 포지션 지수 | (브랜드 평균 가격 / 카테고리 평균) × 100 |
| **Churn Rate** | 순위 교체율 | (신규 진입 + 이탈) / 100 |
| **Avg Rating Gap** | 평점 격차 | 브랜드 평균 평점 - 카테고리 평균 |

### 6.3 Level 3: Product & Risk

| 지표 | 정의 | 산출식 |
|------|------|--------|
| **Rank Volatility** | 순위 변동성 | StdDev(최근 7일 순위) |
| **Rank Shock** | 순위 급변 | \|오늘 순위 - 어제 순위\| ≥ N |
| **Streak Days** | 연속 체류일 | Top N 연속 진입 일수 |
| **Rating Trend** | 평점 추세 | Slope(평점 이동평균) |

---

## 7. 임계값 설정 (thresholds.json)

```json
{
  "ranking": {
    "top_n_tiers": [3, 5, 10, 20, 50, 100],
    "significant_drop": 5,
    "significant_rise": 10
  },
  "streak": {
    "weekly_highlight": 5,
    "monthly_highlight": 30
  },
  "monitoring": {
    "new_product_watch_days": 60,
    "trend_analysis_window": 7
  },
  "competition": {
    "gap_alert": 3,
    "laneige_market_share_warning": 0.1
  },
  "brand_health": {
    "sos_change_up": 1.0,
    "sos_change_down": -1.0,
    "hhi_concentrated": 0.25
  },
  "price_quality": {
    "cpi_premium": 100,
    "rating_gap_warning": 0
  }
}
```

---

## 8. RAG 시스템

### 8.1 데이터 소스 (4개 MD 파일)

| 파일 | 용도 |
|------|------|
| `Strategic Indicators Definition.md` | 지표 정의/산출식 |
| `Metric Interpretation Guide.md` | 지표 해석 가이드 |
| `Indicator Combination Playbook.md` | 지표 조합 해석 |
| `Home Page Insight Rules.md` | 인사이트 생성 규칙/톤 |

### 8.2 RAG 라우팅

| 질의 유형 | 참조 문서 |
|----------|----------|
| "SoS가 뭐야?" | Strategic Indicators Definition |
| "순위 급락 해석?" | Metric Interpretation Guide |
| "CPI 높고 평점 낮으면?" | Indicator Combination Playbook |
| "오늘 요약 문구?" | Home Page Insight Rules |

### 8.3 Fallback 처리
- 질의 의도 파악 실패 시 → 사용자에게 재질문
- 예: "어떤 지표에 대해 알고 싶으신가요? (SoS, HHI, CPI 등)"

### 8.4 챗봇 질의 예시

```
사용자: "라네즈 립 슬리핑 마스크의 최근 3개월 성과를 경쟁사 대비 분석하고,
        4가지 지표를 결합해 전략적 제언을 포함한 보고서를 작성해줘"

GPT-4.1-mini 응답:
- Sheets 데이터 검색 (1M 컨텍스트 활용)
- RAG로 지표 해석 가이드 참조
- 구조화된 전략 보고서 생성
```

---

## 9. Memory & Observability

### 9.1 Memory 계층 (docx 문서 기반 보완)

| 모듈 | 역할 |
|------|------|
| `session.py` | 현재 실행 세션 상태 (진행 중인 작업, 중간 결과) |
| `history.py` | 과거 실행 히스토리 (성공/실패 기록) |
| `context.py` | 에이전트 간 공유 컨텍스트 |

### 9.2 Observability 계층 (Agent Quality 문서 기반)

| 모듈 | 역할 | 예시 |
|------|------|------|
| `logger.py` | 이벤트 로그 | "crawler_agent 시작", "100개 제품 수집 완료" |
| `tracer.py` | 실행 궤적 추적 | 각 단계 입력/출력/소요시간 |
| `metrics.py` | 품질 메트릭 | 성공률, 평균 실행 시간, 에러율 |

### 9.3 도구 문서화 (Agent Tools & MCP 문서 기반)

```python
# 예시: tools/amazon_scraper.py
def scrape_bestsellers(category_url: str) -> dict:
    """
    Amazon 베스트셀러 Top 100 크롤링

    Args:
        category_url: Amazon 카테고리 URL

    Returns:
        {
            "products": [...],  # 최대 100개
            "count": 100,
            "category": "Lip Care",
            "snapshot_date": "2025-01-15"
        }

    Errors:
        - "BLOCKED": Amazon 차단됨 - IP 변경 또는 대기 필요
        - "TIMEOUT": 응답 없음 - 재시도 권장
        - "PARSE_ERROR": HTML 구조 변경 - 파서 업데이트 필요
    """
```

---

## 10. 최종 폴더 구조

```
AMORE-RAG-ONTOLOGY-HYBRID AGENT/
├── main.py                        # CLI 진입점
├── orchestrator.py                # Root Agent (Think-Act-Observe)
│
├── agents/
│   ├── __init__.py
│   ├── crawler_agent.py           # Amazon 크롤링
│   ├── storage_agent.py           # Google Sheets 저장
│   ├── metrics_agent.py           # 지표 계산
│   ├── insight_agent.py           # 인사이트 생성
│   └── chatbot_agent.py           # 대시보드 챗봇
│
├── tools/
│   ├── __init__.py
│   ├── amazon_scraper.py          # Playwright 크롤러
│   ├── sheets_writer.py           # Google Sheets API
│   └── metric_calculator.py       # 지표 계산 로직
│
├── rag/
│   ├── __init__.py
│   ├── router.py                  # 질의 라우팅 + Fallback
│   ├── retriever.py               # MD 문서 검색
│   └── templates.py               # 응답 템플릿
│
├── ontology/
│   ├── __init__.py
│   └── schema.py                  # 5개 엔티티 정의
│
├── memory/
│   ├── __init__.py
│   ├── session.py                 # 현재 세션 상태
│   ├── history.py                 # 실행 히스토리
│   └── context.py                 # 공유 컨텍스트
│
├── monitoring/
│   ├── __init__.py
│   ├── logger.py                  # 로그 기록
│   ├── tracer.py                  # 실행 궤적
│   └── metrics.py                 # 품질 메트릭
│
├── config/
│   └── thresholds.json            # 임계값 설정
│
├── docs/
│   ├── Strategic Indicators Definition.md
│   ├── Metric Interpretation Guide.md
│   ├── Indicator Combination Playbook.md
│   └── Home Page Insight Rules.md
│
├── output/
│   └── amore_unified_dashboard_v3.html
│
├── requirements.txt
├── .env                           # API 키 (OPENAI_API_KEY)
└── README.md
```

---

## 11. 구현 순서

| 순서 | 작업 | 산출물 |
|------|------|--------|
| 1 | 프로젝트 구조 + ADK/LiteLLM 세팅 | 폴더 구조, requirements.txt, .env |
| 2 | Playwright Amazon 크롤러 | `amazon_scraper.py` |
| 3 | Google Sheets 연동 | `sheets_writer.py` |
| 4 | 임계값 설정 | `thresholds.json` |
| 5 | 온톨로지 스키마 | `schema.py` |
| 6 | 지표 계산기 | `metric_calculator.py` |
| 7 | 인사이트 생성 (규칙 기반) | `insight_agent.py` |
| 8 | RAG 라우팅 + Fallback | `router.py`, `retriever.py` |
| 9 | 챗봇 템플릿 + 안전장치 | `templates.py` |
| 10 | Memory 계층 | `session.py`, `history.py`, `context.py` |
| 11 | Monitoring 계층 | `logger.py`, `tracer.py`, `metrics.py` |
| 12 | ADK Agent 연결 (LiteLLM) | 5개 Worker + Orchestrator |
| 13 | 대시보드 챗봇 연동 | `chatbot_agent.py` |

---

## 12. 비용 예측

### 12.1 월간 예상 비용

| 항목 | 수량 | 비용 |
|------|------|------|
| GPT-4.1-mini (파이프라인) | 30회/월 | ~$2 |
| GPT-4.1-mini (챗봇) | 1,500건/월 | ~$3 |
| Google Sheets API | 무료 티어 | $0 |
| **총 월 비용** | | **~$5** |

### 12.2 비용 대비 성능

| 항목 | 값 |
|------|-----|
| 월 비용 | ~$5 |
| MMLU 성능 | ~84% |
| 컨텍스트 | 1M 토큰 |
| 복잡 분석 | 지원 |

---

## 13. 참고 문서

### 에이전트 설계 참고 (docx)
- `_Introduction to Agents_ 핵심 정리.docx` → Think-Act-Observe 루프
- `Agent Quality 핵심요약.docx` → Observability 계층
- `Agent Tools & Interoperability with MCP 핵심요약.docx` → 도구 문서화
- `Context Engineering_ Sessions & Memory 핵심정리.docx` → Memory 계층

### RAG 데이터 소스 (md)
- `Strategic Indicators Definition.md`
- `Metric Interpretation Guide.md`
- `Indicator Combination Playbook.md`
- `Home Page Insight Rules.md`

---

## 14. 핵심 결정 요약

| 항목 | 결정 | 근거 |
|------|------|------|
| LLM | GPT-4.1-mini 단독 | 성능+비용 균형, 1M 컨텍스트 |
| 프레임워크 | ADK + LiteLLM | Google 생태계 + OpenAI 모델 |
| 크롤링 | Top 100 전체 | 경쟁사 분석 필요 |
| 저장소 | Google Sheets | 협업/공유 용이 |
| 런칭일 | 하이브리드 방식 | first_seen_date + launch_date |
| 임계값 | JSON 설정 파일 | 유연한 조정 |
| 챗봇 | 단일 모델 | 라우팅 복잡도 제거 |

---

*최종 업데이트: 2025-12-28*
*LLM: GPT-4.1-mini (OpenAI via LiteLLM)*
