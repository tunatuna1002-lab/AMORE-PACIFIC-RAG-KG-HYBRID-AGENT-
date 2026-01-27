# CLAUDE.md

> Claude Code가 이 코드베이스 작업 시 참조하는 필수 컨텍스트

---

## 1. 프로젝트 개요

**AMORE Pacific RAG-KG Hybrid Agent** - Amazon US에서 LANEIGE 브랜드 경쟁력을 모니터링하는 자율 AI 시스템

### 핵심 기능
- **Daily Auto-Crawling**: Amazon Top 100 × 5 카테고리 (22:00 KST)
- **KPI Analysis**: SoS, HHI, CPI
- **AI Chatbot**: RAG + KG + Ontology 하이브리드
- **Insight Generation**: LLM 기반 전략적 인사이트

### 모니터링 카테고리
1. Beauty & Personal Care
2. Skin Care
3. Lip Care
4. Lip Makeup
5. Face Powder

---

## 2. 기술 스택

| Category | Technology |
|----------|-----------|
| Backend | Python 3.11+, FastAPI, Uvicorn |
| LLM | OpenAI GPT-4.1-mini via LiteLLM |
| Scraping | Playwright, playwright-stealth, browserforge |
| Storage | SQLite, Google Sheets |
| RAG | ChromaDB + OpenAI Embeddings |
| Ontology | owlready2, Rule-based Reasoner |
| Test | pytest, pytest-cov (60% 최소 커버리지) |
| Social Media | Playwright (TikTok), Instaloader (IG), yt-dlp (YT) |
| Public Data | 관세청 수출입통계, 식약처 기능성화장품 API |

---

## 3. 프로젝트 구조

```
├── dashboard_api.py             # FastAPI 메인 엔트리
├── orchestrator.py              # 배치 워크플로우 오케스트레이터
├── src/
│   ├── core/                    # 스케줄링 & 오케스트레이션
│   │   └── brain.py             # UnifiedBrain - 자율 스케줄러
│   ├── agents/                  # AI 에이전트
│   │   ├── hybrid_chatbot_agent.py
│   │   ├── hybrid_insight_agent.py
│   │   └── crawler_agent.py
│   ├── ontology/                # Knowledge Graph & 추론
│   │   ├── knowledge_graph.py   # Triple Store
│   │   └── reasoner.py          # Ontology 추론 엔진
│   ├── rag/                     # RAG 시스템
│   │   ├── hybrid_retriever.py  # KG + RAG 통합 검색
│   │   └── retriever.py         # 문서 검색
│   ├── tools/                   # 유틸리티
│   │   ├── amazon_scraper.py    # Playwright 크롤러
│   │   ├── kg_backup.py         # KG 백업 관리
│   │   ├── metric_calculator.py # KPI 계산
│   │   ├── tiktok_collector.py  # TikTok 수집 (Playwright)
│   │   ├── instagram_collector.py # Instagram 수집 (Instaloader)
│   │   ├── youtube_collector.py # YouTube 수집 (yt-dlp)
│   │   ├── reddit_collector.py  # Reddit 수집 (JSON API)
│   │   ├── google_trends_collector.py # Google Trends
│   │   └── public_data_collector.py # 공공데이터 API
│   ├── domain/                  # Clean Architecture Layer 1
│   │   ├── entities/
│   │   └── interfaces/
│   ├── application/             # Clean Architecture Layer 2
│   │   └── workflows/
│   └── infrastructure/          # Clean Architecture Layer 4
├── dashboard/                   # 프론트엔드
│   └── amore_unified_dashboard_v4.html
├── tests/                       # 테스트
│   ├── golden/                  # 골든셋 테스트
│   └── conftest.py
└── docs/                        # 문서
    └── guides/                  # RAG 참조 문서
```

---

## 4. 개발 명령어

### 서버 실행
```bash
uvicorn dashboard_api:app --host 0.0.0.0 --port 8001 --reload
```

### 테스트
```bash
python -m pytest tests/ -v                    # 전체 테스트 (커버리지 포함)
python -m pytest tests/unit/domain/ -v        # Domain 레이어만
open coverage_html/index.html                 # 커버리지 리포트
python scripts/evaluate_golden.py --verbose   # 골든셋 평가
```

### KG 백업
```bash
python -m src.tools.kg_backup backup          # 수동 백업
python -m src.tools.kg_backup list            # 백업 목록
python -m src.tools.kg_backup restore 2026-01-27  # 복원
```

### 데이터 동기화
```bash
python scripts/sync_from_railway.py           # Railway → 로컬
python scripts/sync_sheets_to_sqlite.py       # Sheets → SQLite
```

---

## 5. 환경 변수

```bash
# 필수
OPENAI_API_KEY=sk-...

# 선택 - 서버 설정
API_KEY=...                        # API 인증
AUTO_START_SCHEDULER=true          # 스케줄러 자동 시작

# 선택 - Google Sheets
GOOGLE_SPREADSHEET_ID=...          # Google Sheets ID
GOOGLE_SHEETS_CREDENTIALS_JSON=... # 서비스 계정 JSON

# 선택 - LLM 설정
LLM_TEMPERATURE_CHAT=0.4           # 챗봇 temperature
LLM_TEMPERATURE_INSIGHT=0.6        # 인사이트 temperature

# 선택 - 뉴스/외부 신호 (무료 티어)
TAVILY_API_KEY=tvly-...            # Tavily 뉴스 (월 1,000건 무료)
GNEWS_API_KEY=...                  # GNews (일 100건 무료)

# 선택 - 공공데이터 (완전 무료)
DATA_GO_KR_API_KEY=...             # 관세청/식약처 API

# 선택 - 이메일 알림 (Gmail SMTP, 무료)
SMTP_SERVER=smtp.gmail.com         # Gmail SMTP 서버
SMTP_PORT=587                      # TLS 포트
SENDER_EMAIL=your@gmail.com        # 발신자 Gmail
SENDER_PASSWORD=xxxx xxxx xxxx xxxx # Gmail 앱 비밀번호 (16자리)
ALERT_RECIPIENTS=alert@email.com   # 수신자 (쉼표로 복수 가능)
```

---

## 6. Clean Architecture

### 레이어 구조 (의존성: 안쪽으로만)

```
src/
├── domain/           # Layer 1: Entities (외부 의존 없음)
├── application/      # Layer 2: Use Cases
├── adapters/         # Layer 3: Interface Adapters
└── infrastructure/   # Layer 4: Frameworks & Drivers
```

### Import 규칙

| From → To | 허용 |
|-----------|------|
| domain → (nothing) | ✅ |
| application → domain | ✅ |
| adapters → domain, application | ✅ |
| infrastructure → domain, application | ✅ |
| **domain → application/infrastructure** | ❌ |
| **infrastructure → adapters** | ❌ |

### DI 패턴

```python
# ❌ Bad
from src.agents.crawler_agent import CrawlerAgent
class MyWorkflow:
    def __init__(self):
        self.crawler = CrawlerAgent()

# ✅ Good
from src.domain.interfaces.agent import CrawlerAgentProtocol
class MyWorkflow:
    def __init__(self, crawler: CrawlerAgentProtocol):
        self.crawler = crawler
```

---

## 7. TDD 워크플로우

1. **🔴 RED**: 테스트 먼저 작성 (`tests/unit/{layer}/test_*.py`)
2. **🟢 GREEN**: 최소 구현으로 테스트 통과
3. **🔵 REFACTOR**: 코드 정리 (테스트 유지)

### 테스트 환경 분리

```bash
# .env.test 사용 (자동 로드)
ENV_FILE=.env.test python -m pytest tests/
```

---

## 8. API 엔드포인트

| Method | Endpoint | Description | Auth |
|--------|----------|-------------|------|
| GET | `/api/health` | 헬스 체크 | - |
| GET | `/api/data` | 대시보드 데이터 | - |
| POST | `/api/v3/chat` | AI 챗봇 (권장) | - |
| POST | `/api/crawl/start` | 크롤링 시작 | API Key |
| GET | `/api/v4/brain/status` | 스케줄러 상태 | - |

---

## 9. 데이터 저장소

### 3중 저장소 구조

| 저장소 | 위치 | Source of Truth |
|--------|------|-----------------|
| Railway SQLite | `/data/amore_data.db` | ✅ Yes |
| Google Sheets | 스프레드시트 | 백업 |
| 로컬 SQLite | `./data/amore_data.db` | 개발용 |

### KG 백업 정책

- **위치**: `data/backups/kg/` (Railway: `/data/backups/kg/`)
- **주기**: 일 1회 (크롤링 완료 후)
- **보관**: 7일 롤링

---

## 10. 디자인 시스템 (AMOREPACIFIC)

| 색상 | HEX | 용도 |
|------|-----|------|
| **Pacific Blue** | `#001C58` | 헤더, 사이드바, 주요 CTA |
| **Amore Blue** | `#1F5795` | 강조, 링크, 보조 버튼 |
| **Gray** | `#7D7D7D` | 보조 텍스트, 비활성 |
| **White** | `#FFFFFF` | 배경, 카드 |

```css
:root {
    --pacific-blue: #001C58;
    --amore-blue: #1F5795;
    --text-secondary: #7D7D7D;
}
```

---

## 11. 주요 모듈 참조

| 모듈 | 파일 | 역할 |
|------|------|------|
| UnifiedBrain | `src/core/brain.py` | 자율 스케줄러 |
| KnowledgeGraph | `src/ontology/knowledge_graph.py` | Triple Store (Railway Volume 자동 연결) |
| HybridRetriever | `src/rag/hybrid_retriever.py` | RAG + KG + Ontology 통합 |
| HybridChatbotAgent | `src/agents/hybrid_chatbot_agent.py` | AI 챗봇 |
| KGBackupManager | `src/tools/kg_backup.py` | KG 백업 관리 (7일 보관) |

### 소셜 미디어 수집기 (v2026.01.27)

| 모듈 | 파일 | 기술 | 비용 |
|------|------|------|------|
| TikTokCollector | `src/tools/tiktok_collector.py` | Playwright | 무료 |
| InstagramCollector | `src/tools/instagram_collector.py` | Instaloader | 무료 |
| YouTubeCollector | `src/tools/youtube_collector.py` | yt-dlp | 무료 |
| RedditCollector | `src/tools/reddit_collector.py` | JSON API | 무료 |
| GoogleTrendsCollector | `src/tools/google_trends_collector.py` | trendspyg/pytrends | 무료 |
| PublicDataCollector | `src/tools/public_data_collector.py` | 관세청/식약처 | 무료 |

### 사용 예시

```python
# TikTok
from src.tools.tiktok_collector import TikTokCollector
collector = TikTokCollector()
posts = await collector.search_hashtag("laneige", limit=50)

# Instagram
from src.tools.instagram_collector import InstagramCollector
collector = InstagramCollector()
posts = await collector.search_kbeauty(limit=100)

# YouTube
from src.tools.youtube_collector import YouTubeCollector
collector = YouTubeCollector()
videos = await collector.search("LANEIGE review", limit=20)

# Reddit
from src.tools.reddit_collector import RedditCollector
collector = RedditCollector()
posts = await collector.search("LANEIGE", subreddit="AsianBeauty")
```

---

## 12. 코드 컨벤션

### Async-First
```python
async def crawl_category(self, category: str) -> List[Product]:
    async with async_playwright() as p:
        browser = await p.chromium.launch()
```

### Pydantic Models
```python
class Product(BaseModel):
    asin: str
    title: str
    brand: str
    rank: int
    price: Optional[float] = None
```

### Type Hints
모든 함수에 파라미터 및 반환 타입 힌트 필수

---

## 13. E2E 감사 체크리스트

### Security
- [ ] API Key 로그 마스킹 (`sk-` 패턴)
- [ ] Prompt injection 방어 (시스템 프롬프트 노출 방지)

### Data Integrity
- [ ] KG JSON 검증 (auto_load 시)
- [ ] 크롤링 실패 시 stale data warning

### 알려진 이슈

| ID | 이슈 | 상태 |
|----|------|------|
| C.1 | Webhook 서명검증 미구현 | 향후 적용 |
| C.6 | chunk_id 부재 | 향후 적용 |
| C.8 | SHACL 제약 검증 미구현 | Low Priority |

---

## 14. 구현 완료 내역

### 2026-01-27 (v3) - 이메일 알림 시스템

| 항목 | 파일 |
|------|------|
| AlertAgent-Brain 통합 | `src/core/brain.py` |
| Gmail SMTP 발송 | `src/tools/email_sender.py` |
| 알림 조건 (순위 ±10, SoS 변동) | `src/agents/alert_agent.py` |

**동작 흐름:**
```
크롤링 → 순위 변동 감지 → AlertAgent → EmailSender → Gmail SMTP → 수신자
```

**테스트 완료:** 2026-01-27 23:01 KST

### 2026-01-27 (v2) - 소셜 미디어 수집기

| 항목 | 파일 |
|------|------|
| TikTok 수집기 | `src/tools/tiktok_collector.py` |
| Instagram 수집기 | `src/tools/instagram_collector.py` |
| YouTube 수집기 | `src/tools/youtube_collector.py` |
| Reddit 수집기 | `src/tools/reddit_collector.py` |
| Google Trends 업데이트 | `src/tools/google_trends_collector.py` (trendspyg 지원) |

### 2026-01-27 (v1)

| 항목 | 파일 |
|------|------|
| KG Railway Volume 연결 | `src/ontology/knowledge_graph.py` |
| KG 자동 백업 (7일) | `src/tools/kg_backup.py` |
| 테스트 환경 분리 | `tests/conftest.py`, `.env.test` |
| 외부 신호 실패 경고 | `src/agents/hybrid_insight_agent.py` |
| 골든셋 평가 스크립트 | `scripts/evaluate_golden.py` |
| 커버리지 측정 환경 | `pyproject.toml`, `pytest.ini` |

### 미구현 (향후 작업)

| 항목 | 우선순위 |
|------|----------|
| SHACL 제약 검증 | Low |
| Webhook 서명검증 | Medium |
| Document chunk_id | Medium |
| Prompt injection 방어 | High |
| 아마존 리뷰 감성분석 | Medium |
| ~~이메일 알림 통합~~ | ~~High~~ → **완료 (v3)** |
