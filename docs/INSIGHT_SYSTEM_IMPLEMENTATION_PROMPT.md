# 인사이트 시스템 고도화 구현 프롬프트

> 이 프롬프트를 새 대화 세션에 복사하여 사용하세요.

---

## 프롬프트 (복사해서 사용)

```
당신은 화장품 산업 분석 시스템을 개발하는 시니어 백엔드 엔지니어입니다.

## 프로젝트 개요

AMORE Pacific RAG-KG Hybrid Agent 프로젝트의 인사이트 시스템을 고도화합니다.
현재 시스템은 Amazon US 베스트셀러 데이터를 수집하지만, "숫자 나열"에 그치고 있습니다.
목표는 "왜?"와 "앞으로 어떻게?"를 답하는 진짜 인사이트 시스템입니다.

## 핵심 철학

```
아마존 랭킹 = 결과 (종속변수)
시장 동향/외부 요인 = 원인 (독립변수)
```

## 구현할 4-Layer 데이터 아키텍처

Layer 4: 거시경제 & 무역 (관세청 수출입 API, 환율, 관세 뉴스)
Layer 3: 산업 & 기업 (아모레퍼시픽 IR, 증권사 리포트, 전문기관)
Layer 2: 소비자 트렌드 (Reddit, TikTok, 뷰티 매체 RSS)
Layer 1: Amazon 데이터 (현재 시스템 - 순위, 가격, 리뷰)

## 구현 우선순위 (무료 소스 우선)

### Phase 1: 무료 인프라 (P0)
1. 관세청 수출입통계 API 연동 (HS Code 3304)
2. RSS 피드 수집기 (Cosmetics Design Asia, KEDGlobal, WWD)
3. 기존 Reddit API 활성화
4. 아모레퍼시픽 IR PDF 파서
5. 출처 관리 시스템

### Phase 2: 데이터 파이프라인
1. Layer별 데이터 수집기 구현
2. 일일 스케줄러 연동
3. PDF 다운로더 (KPMG, KDI)

### Phase 3: 유료 API (필요시)
1. Tavily API 연동 (~$10/월)
2. 검색 쿼리 엔진

### Phase 4: 인사이트 엔진
1. 인과관계 분석 로직
2. 가설 생성 엔진
3. 출처 자동 삽입
4. 지식그래프 스키마 확장

## 기술 스택

- Python 3.11+, FastAPI
- 공공데이터 API (data.go.kr) - 무료
- RSS 파싱: feedparser
- PDF 파싱: PyPDF2
- 웹 스크래핑: BeautifulSoup, Playwright

## 주요 API 엔드포인트

### 관세청 수출입통계
```python
url = "https://apis.data.go.kr/1220000/natImexpTrdStt/getNatImexpTrdSttList"
params = {"hsSgn": "3304", "type": "json"}  # 화장품 HS Code
```

### 식약처 기능성화장품
```python
url = "http://apis.data.go.kr/1471000/FtnltCosmRptPrdlstInfoService/getRptPrdlstInq"
```

## 출처 표시 형식

인사이트에 반드시 출처를 [1], [2] 형식으로 표시하고,
문서 끝에 참고자료 섹션을 추가합니다.

```markdown
K-Beauty 수출이 +12% 증가했습니다 [1].
아모레퍼시픽 Americas 매출은 +6.9% 성장 [2].

## 참고자료
[1] 관세청, 품목별 수출입통계, 2025.01
[2] 아모레퍼시픽 IR, "3Q 2025 Earnings Release", 2025.11.06
```

## 수정할 파일 목록

| 파일 | 변경 내용 |
|------|----------|
| src/tools/market_intelligence.py | 신규 - 외부 데이터 수집기 |
| src/tools/public_data_collector.py | 신규 - 공공데이터 API |
| src/tools/ir_report_parser.py | 신규 - IR PDF 파서 |
| src/tools/source_manager.py | 신규 - 출처 관리 |
| src/agents/hybrid_insight_agent.py | 프롬프트 개선 |
| config/public_apis.json | 신규 - API 설정 |

## 참고 문서

프로젝트 루트의 다음 파일들을 먼저 읽어주세요:
1. docs/INSIGHT_SYSTEM_PLAN_v3.md - 상세 계획안
2. CLAUDE.md - 프로젝트 컨텍스트
3. src/tools/external_signal_collector.py - 기존 RSS/Reddit 코드

## 예산 제약

- 월 $50 이하
- 무료 API 우선 사용
- Tavily/Serper는 ~$15/월로 제한

## 시작 지시

1. 먼저 docs/INSIGHT_SYSTEM_PLAN_v3.md를 읽어주세요
2. 기존 external_signal_collector.py 코드를 확인해주세요
3. Phase 1부터 순차적으로 구현을 시작해주세요
4. 각 단계마다 테스트 코드도 작성해주세요

구현을 시작해주세요.
```

---

## 단축 프롬프트 (빠른 시작용)

```
docs/INSIGHT_SYSTEM_PLAN_v3.md 계획안을 읽고 인사이트 시스템 고도화를 구현해줘.

핵심:
- 아마존 랭킹(결과)과 시장 동향(원인)을 연결하는 시스템
- 무료 API 우선 (관세청, 식약처, RSS)
- 출처 표시 필수 [1], [2] 형식
- Phase 1 (무료 인프라)부터 시작

먼저 계획안을 읽고, 기존 external_signal_collector.py를 확인한 후 구현 시작해줘.
```

---

## 특정 Phase만 구현할 때

### Phase 1만 구현
```
docs/INSIGHT_SYSTEM_PLAN_v3.md의 Phase 1 (무료 인프라 구축)만 구현해줘.

구현할 것:
1. src/tools/public_data_collector.py - 관세청/식약처 API
2. RSS 피드 수집기 확장 (Cosmetics Design Asia, KEDGlobal)
3. src/tools/source_manager.py - 출처 관리 시스템

기존 코드:
- src/tools/external_signal_collector.py (RSS/Reddit 코드 있음)
```

### Phase 4만 구현 (인사이트 엔진)
```
docs/INSIGHT_SYSTEM_PLAN_v3.md의 Phase 4 (인사이트 엔진)를 구현해줘.

구현할 것:
1. src/agents/hybrid_insight_agent.py 프롬프트 개선
2. 인과관계 분석 로직 (Layer 1-4 데이터 연결)
3. 출처 자동 삽입 기능

출처 형식:
- 본문에 [1], [2] 인용
- 끝에 참고자료 섹션
- 신뢰도 점수: IR(1.0) > 공공데이터(0.95) > 전문기관(0.9) > 뉴스(0.7)
```

---

## 아모레퍼시픽 IR 데이터 활용 프롬프트

```
아모레퍼시픽 3Q 2025 IR 데이터를 인사이트에 연동해줘.

IR 소스: https://www.apgroup.com/int/en/investors/amorepacific-corporation/ir-reports/quarterly-results/quarterly-results.html

핵심 데이터 (3Q 2025):
- 전체 매출: 1,016.9B KRW (+4.1% YoY)
- Americas: 156.8B KRW (+6.9% YoY)
- Amazon Prime Day: 매출 2배 성장
- LANEIGE: Tracckr Brand Viral Index 2위
- Mise-en-scène: US Amazon Fragrance 1위

이 데이터를 인사이트 생성 시 자동으로 참조하도록 구현해줘.
```

---

## 디버깅/확인용 프롬프트

```
인사이트 시스템 구현 상태를 확인해줘.

체크리스트:
1. [ ] public_data_collector.py 존재 여부
2. [ ] 관세청 API 연동 테스트
3. [ ] RSS 피드 수집 동작 여부
4. [ ] 출처 관리 시스템 동작 여부
5. [ ] hybrid_insight_agent.py 프롬프트 개선 여부
6. [ ] DOCX 출력에 출처 표시 여부

각 항목을 확인하고 미완료 항목이 있으면 구현해줘.
```
