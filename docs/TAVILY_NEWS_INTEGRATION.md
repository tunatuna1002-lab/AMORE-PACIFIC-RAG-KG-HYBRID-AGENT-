# Tavily 뉴스 API 통합 가이드

> 버전: v2026.01.27
> 작성자: AI Agent

## 개요

AMORE RAG 시스템에 Tavily Search API를 통합하여 실시간 뷰티 산업 뉴스를 수집하고, 이를 인사이트 보고서 및 챗봇 응답의 **실제 출처/근거**로 활용합니다.

## 아키텍처

```
┌─────────────────────────────────────────────────────────────┐
│                    External News Sources                     │
├─────────────────────────────────────────────────────────────┤
│  Tavily API  │   RSS Feeds   │   Reddit API  │   YouTube    │
│  (실시간 뉴스) │  (전문 매체)   │  (소비자 트렌드) │  (리뷰 영상)  │
└──────┬───────────────┬──────────────┬──────────────┬────────┘
       │               │              │              │
       └───────────────┴──────────────┴──────────────┘
                              │
                              ▼
              ┌───────────────────────────────┐
              │   ExternalSignalCollector     │
              │   (src/tools/external_signal_ │
              │    collector.py)              │
              └───────────────┬───────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
        ▼                     ▼                     ▼
┌───────────────┐   ┌─────────────────┐   ┌───────────────┐
│ PeriodInsight │   │ HybridChatbot   │   │ Export API    │
│ Agent         │   │ Agent           │   │ (analyst-     │
│ (보고서 생성)  │   │ (실시간 응답)    │   │  report)      │
└───────────────┘   └─────────────────┘   └───────────────┘
```

## 핵심 파일

| 파일 | 역할 |
|------|------|
| `src/tools/tavily_search.py` | Tavily API 클라이언트 (신뢰도 기반 검색) |
| `src/tools/external_signal_collector.py` | 통합 외부 신호 수집기 |
| `src/agents/period_insight_agent.py` | 기간별 분석 보고서 생성 (뉴스 출처 인용) |
| `src/agents/hybrid_chatbot_agent.py` | 챗봇 응답 (뉴스 컨텍스트 포함) |
| `src/api/routes/export.py` | 보고서 내보내기 API |

## 설정

### 환경변수

```bash
# .env 파일
TAVILY_API_KEY=tvly-xxx...  # Tavily API 키 (필수)
ENABLE_TAVILY_SEARCH=true   # 활성화 여부 (기본: true)
```

### Tavily API 키 발급

1. https://tavily.com 접속
2. 회원가입 후 API 키 발급
3. 무료 플랜: 1,000 검색/월

## 검색 최적화

### 검색 쿼리 전략

```python
# 브랜드별 쿼리 (정확 매칭)
f'"{brand}" beauty news'
f'"{brand}" Amazon skincare bestseller'

# 토픽별 쿼리
f"{topic} news"

# 기본 K-Beauty 쿼리
"K-Beauty skincare trends"
"LANEIGE lip sleeping mask"
"Korean beauty Amazon bestseller"
"Amorepacific beauty news"
```

### 검색 기간 설정

| 용도 | 기간 | 이유 |
|------|------|------|
| 챗봇 실시간 응답 | 14일 | 최신 트렌드 반영 |
| 분석 보고서 | 14-30일 | 분석 기간에 맞춤 |
| 브랜드 언급 검색 | 30일 | 장기 트렌드 파악 |

### 신뢰 소스 (46개)

#### Tier 1: 최고 신뢰도 (0.88-0.95)
- allure.com, wwd.com, beautyindependent.com
- cosmeticsdesign.com, cosmeticsdesign-asia.com
- premiumbeautynews.com, cosmeticsbusiness.com

#### Tier 2: 높은 신뢰도 (0.80-0.95)
- reuters.com, bloomberg.com, forbes.com
- businessinsider.com, cnbc.com

#### Tier 3: 중간 신뢰도 (0.68-0.80)
- vogue.com, elle.com, harpersbazaar.com
- byrdie.com, refinery29.com, glamour.com

#### Tier 4: 한국/아시아 매체 (0.78-0.82)
- koreaherald.com, kedglobal.com
- koreatimes.co.kr, en.yna.co.kr

## 사용 예시

### 1. PeriodInsightAgent - 보고서 생성

```python
from src.agents.period_insight_agent import PeriodInsightAgent
from src.tools.period_analyzer import PeriodAnalyzer
from src.tools.external_signal_collector import ExternalSignalCollector

# 1. 기간 분석
analyzer = PeriodAnalyzer()
analysis = await analyzer.analyze("2026-01-20", "2026-01-27")

# 2. 외부 신호 수집
collector = ExternalSignalCollector()
await collector.initialize()
signals = await collector.fetch_tavily_news(
    brands=["LANEIGE", "COSRX"],
    topics=["K-Beauty"],
    days=14
)

# 3. 인사이트 생성 (뉴스를 실제 출처로 활용)
agent = PeriodInsightAgent()
report = await agent.generate_report(
    analysis,
    external_signals={"signals": signals}
)

# 보고서에 뉴스 기사가 인용됨:
# "Allure에 따르면, K-Beauty 트렌드가 2026년에도 지속될 전망이다 [Allure, 2026-01-25]"
```

### 2. HybridChatbotAgent - 챗봇 응답

```python
from src.agents.hybrid_chatbot_agent import HybridChatbotAgent

agent = HybridChatbotAgent()
result = await agent.chat("LANEIGE 최근 뉴스 알려줘")

# 응답 예시:
# "최근 LANEIGE 관련 주요 뉴스입니다:
#  1. Cosmetics Business에 따르면, LANEIGE가 UK 시장 공략을 강화하고 있습니다 [2026-01-27]
#  2. Allure에서 'Best Korean Skin-Care Products'로 LANEIGE 제품이 선정되었습니다 [2026-01-26]
#
#  ---
#  📚 출처 및 참고자료:
#  1. 📰 **Laneige to tap into hydration heritage to conquer UK beauty** (신뢰도: 88%)
#     - 출처: cosmeticsbusiness.com
#     - 날짜: 2026-01-27
#     - URL: https://www.cosmeticsbusiness.com/..."
```

### 3. Export API - 애널리스트 보고서

```bash
curl -X POST "http://localhost:8001/api/export/analyst-report" \
  -H "Content-Type: application/json" \
  -d '{
    "start_date": "2026-01-20",
    "end_date": "2026-01-27",
    "include_external_signals": true
  }'
```

보고서 "5. 외부 신호 분석" 섹션에 실제 뉴스 기사가 인용됨:

```markdown
## 5. 외부 신호 분석

### 5.1 업계 뉴스 동향
■ Cosmetics Business(2026-01-27)에 따르면, LANEIGE가 hydration heritage를 활용하여
  영국 뷰티 시장 공략을 강화하고 있다.
■ Allure(2026-01-26)는 '2026 K-Beauty Trends' 기사에서 Glass Skin 트렌드의
  지속을 전망하며 LANEIGE를 주요 브랜드로 언급했다.

### 5.2 시사점
■ LANEIGE의 글로벌 확장 전략이 가속화되고 있으며, 특히 유럽 시장에서의
  포지셔닝 강화가 주목됨
■ K-Beauty 트렌드의 지속으로 Amazon US에서의 경쟁력 유지 전망
```

## 데이터 흐름

```
[Tavily API 호출]
       │
       ▼
[TavilySearchResult 변환]
  - title, url, content
  - score (Tavily 관련성)
  - reliability_score (매체 신뢰도)
       │
       ▼
[ExternalSignal 변환]
  - signal_id, source, tier
  - title, content, url
  - published_at, collected_at
  - relevance_score, metadata
       │
       ├──────────────────────────────┐
       │                              │
       ▼                              ▼
[PeriodInsightAgent]           [HybridChatbotAgent]
  - LLM 프롬프트에 뉴스 전달      - 출처 섹션에 뉴스 표시
  - "XXX 매체에 따르면..." 형식   - 신뢰도/관련도 점수 표시
  - URL, 날짜 명시               - 📰 아이콘으로 구분
```

## 비용 추정

| 항목 | 계산 | 월 비용 |
|------|------|--------|
| 챗봇 (일 100회) | 100 × 5 쿼리 × 30일 = 15,000 | 무료 플랜 초과 시 $10/월 |
| 보고서 (일 1회) | 1 × 6 쿼리 × 30일 = 180 | 무료 플랜 내 |
| **합계** | - | **$0-10/월** |

## 트러블슈팅

### 1. Tavily API 키 미설정

```
WARNING: TAVILY_API_KEY not configured. Tavily search will be disabled.
```

**해결**: `.env` 파일에 `TAVILY_API_KEY` 추가

### 2. 뉴스가 수집되지 않음

```python
# 디버깅
from src.tools.tavily_search import TavilySearchClient

client = TavilySearchClient()
print(f"is_enabled: {client.is_enabled()}")
print(f"api_key present: {bool(client.api_key)}")
```

### 3. 신뢰도 낮은 소스만 반환

`TRUSTED_SOURCES`에 해당 도메인이 없으면 기본 신뢰도 0.7 적용.
필요시 `tavily_search.py`의 `TRUSTED_SOURCES` 딕셔너리에 추가.

## 향후 개선 계획

1. **캐싱**: 동일 쿼리 결과 캐싱으로 API 비용 절감
2. **YouTube API 통합**: 영상 리뷰 데이터 수집
3. **감성 분석**: 뉴스 기사의 긍정/부정 톤 분석
4. **알림 시스템**: 주요 뉴스 발생 시 Slack/Email 알림

---

## 변경 이력

| 날짜 | 버전 | 변경 내용 |
|------|------|----------|
| 2026-01-27 | v1.0 | 초기 통합 완료 |
| 2026-01-27 | v1.1 | 검색 키워드 최적화, 신뢰 소스 확장 (30→46개) |
| 2026-01-27 | v1.2 | 인사이트 에이전트 출처 인용 기능 추가 |
