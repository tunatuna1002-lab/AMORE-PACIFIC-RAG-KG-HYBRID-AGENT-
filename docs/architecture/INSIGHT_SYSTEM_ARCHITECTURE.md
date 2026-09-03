# 인사이트 시스템 아키텍처

> 버전: v2026.01.26
> 작성일: 2026-01-26

## 개요

LANEIGE Amazon US 마켓 인사이트 시스템은 **4-Layer 데이터 아키텍처**를 기반으로 "왜?(Why?)"에 답하는 인과관계 중심 인사이트를 생성합니다.

### 기존 문제점
- 단순 숫자 나열: "SoS 2.5%, 순위 3위"
- 인과관계 부재: "왜 순위가 올랐는지" 설명 없음
- 외부 요인 무시: Amazon 데이터만 분석

### 해결 방안
- 4-Layer 계층적 분석으로 거시→미시 원인 추적
- 출처 기반 신뢰도 보장 ([1], [2] 인용)
- 자동화된 데이터 수집 파이프라인

---

## 4-Layer 데이터 아키텍처

```
┌─────────────────────────────────────────────────────────────┐
│  Layer 4: 거시경제/무역 (Macro)                              │
│  ├─ 관세청 수출입통계 (HS 3304 화장품)                        │
│  ├─ 환율 동향                                                │
│  └─ 관세/무역 정책                                           │
├─────────────────────────────────────────────────────────────┤
│  Layer 3: 산업/기업 (Industry)                               │
│  ├─ 아모레퍼시픽 IR 실적 (분기별)                             │
│  ├─ 식약처 기능성화장품 등록 현황                              │
│  └─ 경쟁사 동향                                              │
├─────────────────────────────────────────────────────────────┤
│  Layer 2: 소비자 트렌드 (Consumer)                           │
│  ├─ Reddit (r/SkincareAddiction, r/AsianBeauty)              │
│  ├─ TikTok 바이럴 트렌드                                     │
│  ├─ 뷰티 전문매체 (Allure, Byrdie, WWD)                      │
│  └─ Google Trends 검색 관심도                                │
├─────────────────────────────────────────────────────────────┤
│  Layer 1: Amazon 성과 (Performance)                          │
│  ├─ 일일 크롤링 데이터 (Top 100 × 5 카테고리)                  │
│  ├─ SoS, HHI, CPI 지표                                       │
│  └─ 순위 변동, 가격, 리뷰                                     │
└─────────────────────────────────────────────────────────────┘
```

### Layer별 데이터 소스

| Layer | 데이터 소스 | 수집 방법 | 비용 | 신뢰도 |
|-------|------------|----------|------|--------|
| **Layer 4** | 관세청 수출입통계 | data.go.kr API | 무료 | 0.95 |
| **Layer 3** | 아모레퍼시픽 IR | Predefined + PDF 파싱 | 무료 | 1.00 |
| **Layer 3** | 식약처 기능성화장품 | data.go.kr API | 무료 | 0.95 |
| **Layer 2** | Reddit | PRAW API | 무료 | 0.70 |
| **Layer 2** | 뷰티 매체 RSS | RSS 피드 | 무료 | 0.90 |
| **Layer 2** | Google Trends | pytrends | 무료 | 0.80 |
| **Layer 1** | Amazon | Playwright 크롤링 | 무료 | 0.95 |

---

## 인사이트 생성 프로세스

### 1. 데이터 수집 (Daily 22:00 KST)

```python
# UnifiedBrain 스케줄러
async def _execute_scheduled_task(task_name):
    # 1. Amazon 크롤링
    result = await crawl_manager.start_crawl()

    # 2. Market Intelligence 수집 (Layer 2-4)
    mi_result = await brain.collect_market_intelligence()
```

### 2. Market Intelligence Engine

```python
class MarketIntelligenceEngine:
    async def collect_all_layers(self):
        """모든 레이어 병렬 수집"""
        tasks = [
            self._collect_layer_4(),  # 거시경제
            self._collect_layer_3(),  # 산업/기업
            self._collect_layer_2(),  # 소비자 트렌드
        ]
        await asyncio.gather(*tasks)
```

### 3. HybridInsightAgent 통합

```python
class HybridInsightAgent:
    async def execute(self, metrics_data, crawl_data):
        # 1. 온톨로지 추론
        hybrid_context = await self.hybrid_retriever.retrieve(query)

        # 2. 외부 신호 수집
        external_signals = await self._collect_external_signals()

        # 3. Market Intelligence 수집
        market_intelligence = await self._collect_market_intelligence()

        # 4. LLM 인사이트 생성 (4-Layer 템플릿)
        daily_insight = await self._generate_daily_insight(
            hybrid_context, metrics_data, crawl_summary,
            external_signals, market_intelligence
        )
```

---

## 인사이트 출력 템플릿

```markdown
# LANEIGE Amazon US 일일 인사이트

## 📌 오늘의 핵심
[가장 중요한 변화 + 원인 연결]
예: "Lip Sleeping Mask 순위 상승은 Q3 Americas 매출 +6.9% 성장[2]과
최근 TikTok 바이럴[3]의 복합 효과로 판단됩니다."

## 🔍 원인 분석 (Why?)

### Layer 4: 거시경제/무역
• [관세청 수출입 데이터 기반 분석] [1]
• [환율/관세 영향 분석]

### Layer 3: 산업/기업 동향
• [아모레퍼시픽 IR 실적 기반 분석] [2]
• [브랜드 전략/캠페인 영향]

### Layer 2: 소비자 트렌드
• [Reddit/SNS 트렌드 분석] [3]
• [뷰티 매체 보도 내용]

### Layer 1: Amazon 성과
• [순위 변동, SoS, 가격 등 핵심 지표]
• [경쟁사 동향]

## ⚠️ 주의 사항
• [리스크 또는 모니터링 필요 사항]

## 💡 권장 액션
1. [즉시 실행] 구체적 액션 1
2. [모니터링] 구체적 액션 2
3. [검토 필요] 구체적 액션 3

## 📚 참고자료
[1] 관세청, 품목별 수출입통계, 2025.12
[2] 아모레퍼시픽, "3Q 2025 Earnings Release", 2025.11
[3] Reddit r/SkincareAddiction, "LANEIGE lip mask...", 2026.01
```

---

## 출처 관리 시스템

### 신뢰도 점수 (Reliability Score)

| 출처 유형 | 신뢰도 | 예시 |
|----------|--------|------|
| IR 공시 | 1.00 | 아모레퍼시픽 분기 실적 |
| 정부 공식 | 0.95 | 관세청, 식약처 |
| 학술/연구 | 0.90 | 논문, 시장조사 보고서 |
| 전문 매체 | 0.85 | Allure, WWD, Byrdie |
| 뉴스 | 0.70 | 일반 뉴스 기사 |
| SNS | 0.50 | Reddit, TikTok |

### 출처 우선순위

참고자료 섹션 생성 시 우선순위:
1. Market Intelligence (Layer 4 → Layer 3 → Layer 2)
2. External Signals (Reddit, TikTok)
3. RAG Documents (내부 가이드)
4. Knowledge Graph (온톨로지 추론)

### InsightSourceBuilder 사용법

```python
from src.tools.source_manager import InsightSourceBuilder

builder = InsightSourceBuilder()

# 출처 추가
builder.add_source(
    publisher="관세청",
    title="품목별 수출입통계",
    date="2025.12",
    source_type="government",
    reliability_score=0.95
)

# 인사이트에 인용 삽입
text = "화장품 대미 수출이 12% 증가했습니다"
cited_text = builder.cite(text, source_index=1)
# → "화장품 대미 수출이 12% 증가했습니다[1]"

# 참고자료 섹션 생성
references = builder.build_references()
```

---

## API 엔드포인트

### Market Intelligence API

| Method | Endpoint | 설명 |
|--------|----------|------|
| GET | `/api/market-intelligence/status` | 데이터 수집 상태 |
| GET | `/api/market-intelligence/layers` | 4-Layer 데이터 조회 |
| POST | `/api/market-intelligence/collect` | 수동 데이터 수집 트리거 |
| GET | `/api/market-intelligence/insight` | 생성된 인사이트 조회 |
| GET | `/api/insights/sources` | 출처 정보 조회 |

### 사용 예시

```bash
# Layer 데이터 조회
curl http://localhost:8001/api/market-intelligence/layers

# 응답 예시
{
  "layer_4": {
    "trade_data": {
      "export_to_us": 1500000000,
      "yoy_change": 12.5
    }
  },
  "layer_3": {
    "ir_data": {
      "americas_revenue_yoy": 6.9,
      "operating_margin": 8.2
    }
  },
  "layer_2": {
    "reddit_trends": [...],
    "rss_articles": [...]
  }
}
```

---

## 파일 구조

```
src/tools/
├── market_intelligence.py      # MarketIntelligenceEngine (통합 엔진)
├── public_data_collector.py    # 관세청/식약처 API
├── ir_report_parser.py         # IR 보고서 파싱
├── external_signal_collector.py # Reddit/RSS 수집
├── source_manager.py           # 출처 관리
├── google_trends_collector.py  # Google Trends (Phase 1)
└── youtube_collector.py        # YouTube 리뷰 (Phase 2)

src/agents/
└── hybrid_insight_agent.py     # 인사이트 생성 에이전트

src/core/
└── brain.py                    # 스케줄러 통합
```

---

## 환경변수 설정

```bash
# 필수
OPENAI_API_KEY=sk-...

# 공공데이터 API (선택 - 없으면 Predefined 데이터 사용)
PUBLIC_DATA_API_KEY=...
# 또는
DATA_GO_KR_API_KEY=...

# 유료 API (선택)
TAVILY_API_KEY=...              # 뉴스 검색 (~$10/월)
YOUTUBE_API_KEY=...             # YouTube Data API (무료 할당량)
```

---

## Graceful Degradation

API 키가 없어도 시스템은 정상 동작합니다:

| 상황 | 동작 |
|------|------|
| 공공데이터 API 키 없음 | Predefined 데이터 사용 (IR, 수출입 통계) |
| Reddit API 실패 | RSS 피드만 수집 |
| 외부 API 타임아웃 | 캐시된 데이터 사용 |

```python
# Graceful degradation 예시
async def _collect_layer_4(self):
    try:
        data = await self.public_data.fetch_trade_data()
    except Exception:
        # Fallback to predefined data
        data = self._get_predefined_trade_data()
    return data
```

---

## 작성 원칙 (LLM 프롬프트)

1. **인과관계 중심**: "A가 발생했다" → "A는 B 때문에 발생한 것으로 판단된다"
2. **출처 필수 인용**: 모든 사실 주장에 [1], [2] 형태로 출처 인용
3. **계층적 분석**: Layer 4(거시) → Layer 1(Amazon)으로 원인-결과 연결
4. **정량적 표현**: "증가" 대신 "+12%", "많음" 대신 "2,400 업보트"
5. **가설적 표현**: 확실하지 않은 내용은 "~로 판단됩니다", "~가능성이 있습니다" 사용

---

## 테스트

```bash
# 단위 테스트
python -m pytest tests/unit/tools/ -v

# 통합 테스트
python -m pytest tests/integration/ -v

# 전체 테스트 (82개)
python -m pytest tests/ -v
```

---

## 향후 계획

### Phase 3 (선택)
- [ ] Tavily API 연동 (뉴스 검색 강화)
- [ ] YouTube Data API 연동 (리뷰 분석)

### Phase 4 (고도화)
- [ ] 인과관계 자동 추출 (NLP)
- [ ] 가설 검증 엔진
- [ ] 예측 모델 통합
