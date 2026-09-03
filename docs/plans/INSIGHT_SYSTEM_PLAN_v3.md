# 인사이트 시스템 고도화 계획안 v3

> 화장품 산업 분석가 관점 - "결과(아마존 랭킹)"의 "원인(시장 동향)"을 연결하는 시스템
>
> **v3 업데이트**: 한국 공공데이터 API + 전문기관/애널리스트 소스 + 아모레퍼시픽 IR 데이터 추가

---

## 핵심 철학

```
아마존 랭킹 = 결과 (종속변수)
시장 동향/외부 요인 = 원인 (독립변수)

"숫자 나열"이 아니라 "왜?" + "앞으로 어떻게?"를 답해야 진짜 인사이트
```

---

## 시스템 요구사항

| 항목 | 설정 |
|------|------|
| 타겟 시장 | 미국 (Amazon US) |
| 데이터 수집 | API 자동 수집 (Tavily/Serper) + 무료 공공 API |
| 업데이트 주기 | 일일 |
| API 예산 | 월 $50 이하 |
| SNS 범위 | TikTok, Instagram, Reddit, YouTube, X 전체 |

---

## 데이터 레이어 아키텍처

```
┌─────────────────────────────────────────────────────────────┐
│                    📊 Market Intelligence Engine             │
├─────────────────────────────────────────────────────────────┤
│  Layer 4: 거시경제 & 무역 정책                               │
│  ├─ 트럼프 관세 정책 (USTR, 무역 뉴스)                       │
│  ├─ 환율 동향 (USD/KRW)                                     │
│  ├─ FDA 규제 변화                                           │
│  ├─ 해상 물류비 (SCFI 지수)                                  │
│  └─ 관세청 수출입통계 API (HS Code 3304)                     │
├─────────────────────────────────────────────────────────────┤
│  Layer 3: 산업 & 기업 분석                                   │
│  ├─ 아모레퍼시픽 IR (분기 실적 발표)                         │
│  ├─ 증권사 리포트 (메리츠-박종대, 상상인-김혜미)              │
│  ├─ K-Beauty 수출 통계 (KITA, 관세청)                        │
│  ├─ 산업 뉴스 (WWD, Cosmetics Design Asia, KEDGlobal)       │
│  └─ 전문기관 (KCII, KHIDI, KDI, KPMG)                       │
├─────────────────────────────────────────────────────────────┤
│  Layer 2: 소비자 트렌드 & SNS                                │
│  ├─ TikTok (해시태그 조회수, 바이럴 제품)                     │
│  ├─ Instagram (인플루언서 언급, 스폰서드 포스트)              │
│  ├─ Reddit (r/SkincareAddiction, r/AsianBeauty)             │
│  ├─ YouTube (리뷰어 센티먼트, 조회수)                        │
│  ├─ X/Twitter (실시간 버즈)                                  │
│  └─ 뷰티 전문 매체 (Allure, Byrdie, Refinery29, The Cut)    │
├─────────────────────────────────────────────────────────────┤
│  Layer 1: Amazon 데이터 (현재 시스템)                        │
│  ├─ 순위, 가격, 리뷰                                         │
│  ├─ AI Review Highlights                                    │
│  └─ 프로모션, 딜, 쿠폰                                       │
└─────────────────────────────────────────────────────────────┘
             ↓
    ┌───────────────────┐
    │  Insight Engine   │
    │  (인과관계 분석)   │
    │  + 출처 표시      │
    └───────────────────┘
```

---

## 아모레퍼시픽 IR 데이터 (1차 소스)

### IR 보고서 접근

| 항목 | 정보 |
|------|------|
| **URL** | https://www.apgroup.com/int/en/investors/amorepacific-corporation/ir-reports/quarterly-results/quarterly-results.html |
| **업데이트** | 분기별 (1월, 4월, 7월, 11월) |
| **형식** | PDF (Earnings Release) |
| **언어** | 영어 |

### 3Q 2025 실적 요약 (2025.11.6 발표)

#### 전체 실적
| 지표 | 2024 3Q | 2025 3Q | YoY |
|------|---------|---------|-----|
| **매출** | 977.2B KRW | 1,016.9B KRW | **+4.1%** |
| **영업이익** | 65.2B KRW | 91.9B KRW | **+41.0%** |
| **순이익** | 37.2B KRW | 68.2B KRW | **+83.6%** |
| **영업이익률** | 6.7% | 9.0% | +2.3%p |

#### 지역별 매출
| 지역 | 2024 3Q | 2025 3Q | YoY |
|------|---------|---------|-----|
| **국내** | 534.5B | 556.6B | +4.1% |
| **Americas** | 146.6B | 156.8B | **+6.9%** |
| **EMEA** | 54.5B | 52.7B | -3.2% |
| **Greater China** | 97.6B | 106.0B | +8.5% |
| **Other Asia** | 129.6B | 125.4B | -3.3% |

#### 국내 브랜드별 매출 비중 (2025 3Q)
| 브랜드 | 비중 | 전년 동기 |
|--------|------|----------|
| **Sulwhasoo** | 28% | 28% |
| **HERA** | 15% | 15% |
| **AESTURA** | 7% | 5% ⬆️ |
| **MISE EN SCÈNE** | 7% | 6% ⬆️ |
| **LANEIGE** | 6% | 6% |
| **IOPE** | 5% | 5% |
| **ILLIYOON** | 4% | 3% ⬆️ |
| **RYO** | 3% | 3% |

#### Americas 핵심 인사이트 (아마존 US 연관)

**LANEIGE**:
- 'Next-Gen Hydration' 캠페인으로 스킨케어 (Cream Skin, Water Bank) 매출 증가
- Lip Sleeping Mask 신에디션 (Baskin Robbins, Strawberry Shortcake) 출시
- 인플루언서 콜라보 마이크로드라마 'Beauty and the Beat' 런칭
- Tracckr Brand Viral Index 8월 **2위** 기록

**COSRX**:
- 'Peptide Collagen Hydrogel Eye Patch' TikTok Shop 매출 급증
- 신규 성장 모멘텀 확보

**Aestura, Hanyul**:
- 미국 오프라인 확대 + 캐나다 Sephora 신규 진출
- Aestura: 피부과 의사/인플루언서 이벤트로 더마 카테고리 신뢰도 강화
- 'Atobarrier 365 Cream' 강력한 판매 모멘텀

**Amazon Prime Day**:
- 매출 2배 성장
- Illiyoon, Mise-en-scène 강력한 성과
- Mise-en-scène 'Perfect Hair Serum' US Amazon **Fragrance 카테고리 1위**

---

## 무료 공공데이터 API (data.go.kr)

| API | 엔드포인트 | 데이터 | 비용 |
|-----|-----------|--------|------|
| **식약처 기능성화장품** | `apis.data.go.kr/1471000/FtnltCosmRptPrdlstInfoService` | 제품명, 제조사, 판매사 | 무료 |
| **관세청 수출입통계** | `apis.data.go.kr/1220000/natImexpTrdStt` | HS Code 3304 화장품 수출입 | 무료 |
| **KOSIS 지표정보** | `apis.data.go.kr/15057227` | 산업 시계열 통계 | 무료 |

---

## 전문기관 데이터 (웹/PDF)

| 기관 | 제공 데이터 | 수집 방법 | 업데이트 |
|------|-----------|----------|---------|
| **아모레퍼시픽 IR** | 분기 실적, 지역별/브랜드별 성과 | PDF 다운로드 | 분기별 |
| **대한화장품산업연구원(KCII)** | 국가별 시장 동향, 원료 트렌드 | 웹 스크래핑/PDF | 월간 |
| **한국보건산업진흥원(KHIDI)** | 보건산업통계집, 화장품 생산/유통 | KHISS 웹 다운로드 | 연간 |
| **한국무역협회(KITA)** | K-Beauty 수출 현황, 유망 시장 | K-STAT CSV 다운로드 | 월간 |
| **KDI 경제정보센터** | 화장품 산업 백서, 정책 자료 | PDF 다운로드 | 비정기 |
| **삼정KPMG 경제연구원** | 화장품산업 9대 트렌드, M&A 동향 | PDF 다운로드 | 연간 |

---

## 증권사 애널리스트 리포트

| 증권사 | 담당 애널리스트 | 강점 | 접근 방법 |
|--------|---------------|------|----------|
| **메리츠증권** | 박종대 | K-뷰티 글로벌 경쟁력, 구조적 변화 | Telegram @meritz_research |
| **상상인증권** | 김혜미 | 수출 데이터 기반 시장 예측 | 홈페이지 리서치 |
| **DB금융투자** | 허제나 | 기업별 밸류에이션, 투자 전략 | 홈페이지 리서치 |

### 시장 트렌드 전문가

| 전문가/기관 | 전문 분야 |
|------------|----------|
| **천계성 (트렌디어 대표)** | 데이터/AI 기반 K-뷰티 트렌드 분석 |
| **대한화장품산업연구원** | 해외 진출 가이드, 현지화 전략 |

---

## 글로벌 뷰티 전문 매체

| 매체 | URL | 활용 | 수집 방법 |
|------|-----|------|----------|
| **WWD** | wwd.com | Amazon Prime Day 딜, 글로벌 뷰티 트렌드 | RSS |
| **Cosmetics Design Asia** | cosmeticsdesign-asia.com | 아시아 화장품 산업 분석, 신제품 | RSS |
| **KEDGlobal** | kedglobal.com/beauty-cosmetics | 한국경제 영어판 K-Beauty 뉴스 | RSS |
| **The Cut** | thecut.com | 뷰티 리뷰 & 트렌드 | 웹 |
| **Korea Herald** | koreaherald.com | K-Beauty 글로벌 뉴스 | RSS |
| **코스인코리아** | cosinkorea.com | 국내 화장품 산업 뉴스 | RSS |
| **코스모닝** | cosmorning.com | 국내 화장품 산업 뉴스 | RSS |

---

## 확장된 API 전략 (월 $50 예산)

### Tier 1: 무료 공공 API (비용 $0)

```python
PUBLIC_APIS = {
    "mfds_cosmetics": {
        "url": "http://apis.data.go.kr/1471000/FtnltCosmRptPrdlstInfoService/getRptPrdlstInq",
        "params": {"type": "json", "numOfRows": 100},
        "update": "daily",
        "use": "제품 정보 → Knowledge Graph 엔티티"
    },
    "customs_trade": {
        "url": "https://apis.data.go.kr/1220000/natImexpTrdStt/getNatImexpTrdSttList",
        "params": {"hsSgn": "3304"},  # 화장품 HS Code
        "update": "monthly",
        "use": "수출입 통계 → Layer 4 거시경제"
    },
    "kosis_stats": {
        "url": "https://apis.data.go.kr/15057227",
        "update": "monthly",
        "use": "산업 통계 → 시계열 트렌드"
    }
}
```

### Tier 2: 유료 뉴스/검색 API (~$15/월)

| API | 용도 | 일일 호출 | 월간 비용 |
|-----|------|----------|----------|
| **Tavily** | 글로벌 뉴스 검색 | 35회 | ~$10 |
| **Serper** | Google 뉴스 백업 | 10회 | ~$5 |

### Tier 3: 웹 스크래핑/수동 (비용 $0)

| 대상 | 도구 | 용도 |
|------|------|------|
| 아모레퍼시픽 IR | PDF 파서 | 분기 실적 데이터 |
| 네이버 금융 | BeautifulSoup | 증권사 리포트 |
| K-STAT (KITA) | Selenium | 무역통계 CSV |
| KHISS | Playwright | 보건산업통계 |
| KPMG PDF | PyPDF2 | 연간 산업 전망 |

### 예산 요약

| 항목 | 월간 비용 |
|------|----------|
| 공공 API | $0 |
| Tavily + Serper | ~$15 |
| OpenAI (인사이트 생성) | ~$10 |
| **합계** | **~$25** |
| **여유 예산** | **$25** |

---

## 출처 표시 시스템

### 인용 형식

```markdown
## 오늘의 인사이트

LANEIGE Lip Sleeping Mask가 Top4를 유지하는 가운데,
아모레퍼시픽 3Q 2025 실적에서 Americas 매출이 +6.9% 성장했습니다 [1].
특히 Amazon Prime Day에서 매출이 2배 증가했으며 [1],
TikTok에서 #LipMask 해시태그가 520만 조회를 기록하며
소비자 관심이 지속되고 있습니다 [2].

MEDICUBE의 급성장(SoS +1.2%p)은 최근 미국 진출 마케팅 강화와
일치합니다 [3]. 다만 원/달러 환율이 1,450원을 넘어서면서
가격 경쟁력 압박이 예상됩니다 [4].

---
## 참고자료
[1] 아모레퍼시픽 IR, "3Q 2025 Earnings Release", 2025.11.06
    https://www.apgroup.com/int/en/investors/...
[2] TikTok Creative Center, #LipMask 해시태그 분석, 2025.01.25 기준
[3] KEDGlobal, "K-Beauty brands shine at Amazon Prime Day", 2025.07.21
    https://www.kedglobal.com/beauty-cosmetics/...
[4] 한국은행, 환율 동향, 2025.01.26 기준
```

### 출처 데이터 구조

```python
@dataclass
class Source:
    id: str                    # "[1]"
    title: str                 # 기사/리포트 제목
    publisher: str             # 매체명
    date: str                  # 발행일
    url: Optional[str]         # 링크 (있으면)
    source_type: str           # ir, news, research, sns, government
    reliability_score: float   # 신뢰도 점수 (0-1)
```

### 출처 신뢰도 점수

| 소스 유형 | 신뢰도 | 예시 |
|----------|--------|------|
| **IR/공시** | 1.0 | 아모레퍼시픽 Earnings Release |
| **공공데이터** | 0.95 | 관세청 수출입통계, KOSIS |
| **전문기관** | 0.9 | KCII, KHIDI, KPMG |
| **증권사 리포트** | 0.85 | 메리츠증권, 상상인증권 |
| **전문 매체** | 0.8 | WWD, Cosmetics Design Asia |
| **일반 뉴스** | 0.7 | KEDGlobal, Korea Herald |
| **SNS** | 0.5 | Reddit, TikTok |

---

## 지식그래프 스키마 확장

현재 엔티티 외 추가:

| 엔티티 타입 | 출처 | 관계 |
|-----------|------|------|
| `EarningsReport` | 아모레퍼시픽 IR | `Company -[releases]-> EarningsReport` |
| `Regulation` | 식약처 API | `Product -[regulated_by]-> Regulation` |
| `TradeFlow` | 관세청 API | `Country -[exports_to]-> Country` |
| `MarketTrend` | KPMG 리포트 | `Trend -[affects]-> Category` |
| `MediaMention` | RSS 피드 | `Brand -[mentioned_in]-> Article` |
| `AnalystReport` | 증권사 | `Report -[covers]-> Brand` |
| `ExportStat` | KITA | `Brand -[exports_to]-> Market` |
| `Campaign` | IR 데이터 | `Brand -[runs]-> Campaign` |

---

## 새로운 인사이트 템플릿

```markdown
# LANEIGE Amazon US 일일 인사이트
> 생성일: 2025-01-26 | 분석 기간: 2025.01.19 ~ 2025.01.26

---

## 📌 오늘의 핵심 (1문장)
[가장 중요한 변화/발견 + 원인 연결]

---

## 📊 Amazon 성과 데이터
### 순위 변동
- Lip Sleeping Mask: 4위 유지 (7일 평균 4.2위)
- Holiday Gift Set: 25→67위 (-42위, Velocity -14)

### KPI
- SoS: 1.7% (전주 대비 -0.3%p)
- Top10 제품: 2개

---

## 🔍 원인 분석 (Why?)

### Layer 4: 거시경제/무역
- **환율**: USD/KRW 1,438원 (전주 대비 +12원) [1]
- **관세**: 트럼프 행정부 K-Beauty 관세 검토 보도 [2]
- **수출통계**: 화장품 대미 수출 +12% YoY (관세청) [3]
- **영향**: 가격 경쟁력 소폭 약화 가능성 (hypothesis)

### Layer 3: 산업/기업 동향
- **아모레퍼시픽 IR**: 3Q 2025 Americas +6.9%, Amazon Prime Day 2배 성장 [4]
- **경쟁사**: MEDICUBE 미국 마케팅 강화 [5]
- **영향**: 전체 K-Beauty 성장세 지속, 경쟁 심화

### Layer 2: SNS/소비자 트렌드
- **TikTok**: #LipSleepingMask 520만 조회 (+15% WoW) [6]
- **Reddit**: r/SkincareAddiction 긍정 리뷰 12건/주
- **인플루언서**: @skincarebyhyram 언급 없음 (모니터링)
- **영향**: 소비자 관심 유지, 바이럴 모멘텀 지속

### Layer 1: Amazon 내부 요인
- **프로모션**: Gift Set 쿠폰 종료 (01/20) → 급락 원인 추정
- **경쟁사 딜**: COSRX Lightning Deal 진행 중
- **리뷰**: Lip Sleeping Mask 신규 리뷰 +47건/주

---

## 🎯 원인 가설 매트릭스

| 현상 | 가설 | 근거 | 검증 방법 |
|------|------|------|----------|
| Gift Set 급락 | 쿠폰 종료 | 01/20 쿠폰 만료 확인 | 쿠폰 히스토리 |
| Gift Set 급락 | 시즌 종료 | 1월 Gift 수요 감소 | 카테고리 전체 추이 |
| MEDICUBE 상승 | 마케팅 강화 | 뉴스 보도 [5] | 광고 지출 확인 |

---

## 🔮 전망 (What's Next?)

### 단기 (1-2주)
- Gift Set: 쿠폰 재개 없으면 70위권 안착 예상
- Lip Sleeping Mask: Top5 유지 전망 (SNS 모멘텀 지속)

### 중기 (1-3개월)
- **리스크**: 관세 불확실성 → 가격 인상 압박 가능
- **기회**: K-Beauty 전체 성장세 → 카테고리 확장 기회
- **IR 시사점**: 4Q 2025 예상 - Americas 성장 지속 전망 (신제품 출시)

---

## ✅ 다음 7일 액션

| 우선순위 | 액션 | 담당 | 기한 | 측정 지표 |
|---------|------|------|------|----------|
| P0 | Gift Set 쿠폰 재개 검토 | 마케팅 | D+2 | 순위 회복 |
| P1 | MEDICUBE 상세 벤치마크 | 전략 | D+5 | 가격/프로모션 비교 |
| P2 | 관세 동향 모니터링 | 무역 | 지속 | 정책 변화 |

---

## 📚 참고자료

[1] 한국은행, 환율 동향, 2025.01.26
[2] Reuters, "Trump administration reviews K-Beauty tariffs", 2025.01.22
    https://reuters.com/...
[3] 관세청, 품목별 수출입통계 (HS Code 3304), 2025.01
[4] 아모레퍼시픽 IR, "3Q 2025 Earnings Release", 2025.11.06
    https://www.apgroup.com/int/en/investors/...
[5] KEDGlobal, "K-Beauty brands shine at Amazon Prime Day", 2025.07.21
    https://www.kedglobal.com/...
[6] TikTok Creative Center, 해시태그 분석, 2025.01.26 기준

---
_본 리포트는 AI 분석 시스템에 의해 생성되었습니다.
투자 결정 전 전문가 상담을 권장합니다._
```

---

## 구현 로드맵 (무료 소스 우선)

### Phase 1: 무료 인프라 구축 (3-4일) ⭐ 최우선
1. **공공데이터 API 연동** (식약처, 관세청, KOSIS) - 비용 $0
2. **RSS 피드 수집기** (Cosmetics Design Asia, KEDGlobal, WWD, 코스인코리아)
3. **기존 Reddit API 활성화** (이미 코드 존재)
4. **아모레퍼시픽 IR PDF 파서** (분기별 자동 다운로드)
5. 출처 관리 시스템 구축

### Phase 2: 무료 데이터 파이프라인 (4-5일)
1. Layer 4: **관세청 API** → 월별 수출입 통계
2. Layer 3: **RSS 피드** → 글로벌 뷰티 뉴스
3. Layer 3: **IR 데이터** → 분기 실적 연동
4. Layer 2: **Reddit API** → 소비자 트렌드
5. **PDF 다운로더** (KPMG, KDI 연간 리포트)
6. 일일 스케줄러 연동

### Phase 3: 유료 API 추가 (2-3일)
1. **Tavily API 연동** (~$10/월)
2. 검색 쿼리 엔진 구현
3. 증권사 리포트 웹 스크래핑 (네이버 금융)

### Phase 4: 인사이트 엔진 (5-7일)
1. 인과관계 분석 로직
2. 가설 생성 엔진
3. 전망 생성 로직
4. 출처 자동 삽입
5. **지식그래프 스키마 확장**

### Phase 5: 프롬프트 & 출력 (3-5일)
1. 새 인사이트 템플릿 적용
2. DOCX 리포트 업데이트
3. 대시보드 연동

---

## 구현 우선순위 매트릭스

| 우선순위 | 소스 | 비용 | ROI | 구현 난이도 |
|---------|------|------|-----|-----------|
| **P0** | 아모레퍼시픽 IR PDF | $0 | ⭐⭐⭐⭐ | 보통 |
| **P0** | 관세청 수출입 API | $0 | ⭐⭐⭐ | 쉬움 |
| **P0** | RSS 피드 (글로벌 매체) | $0 | ⭐⭐⭐ | 쉬움 |
| **P0** | Reddit API | $0 | ⭐⭐⭐ | 쉬움 (기존 코드) |
| **P1** | 식약처 화장품 API | $0 | ⭐⭐ | 쉬움 |
| **P1** | KOSIS 통계 API | $0 | ⭐⭐ | 보통 |
| **P2** | Tavily API | ~$10/월 | ⭐⭐⭐ | 쉬움 |
| **P2** | 증권사 리포트 스크래핑 | $0 | ⭐⭐⭐ | 보통 |
| **P3** | KPMG/KDI PDF 파서 | $0 | ⭐⭐ | 보통 |
| **P3** | SimilarWeb (수동) | $0 | ⭐ | 수동 입력 |

---

## 수정 파일 목록

| 파일 | 변경 내용 |
|------|----------|
| `src/tools/market_intelligence.py` | 신규 - 외부 데이터 수집기 |
| `src/tools/public_data_collector.py` | **신규** - 공공데이터 API 연동 |
| `src/tools/ir_report_parser.py` | **신규** - 아모레퍼시픽 IR PDF 파서 |
| `src/tools/analyst_report_scraper.py` | **신규** - 증권사 리포트 스크래핑 |
| `src/tools/source_manager.py` | 신규 - 출처 관리 시스템 |
| `src/agents/hybrid_insight_agent.py` | 프롬프트 개선 + 외부 데이터 연동 |
| `src/rag/context_builder.py` | 외부 컨텍스트 통합 |
| `src/ontology/knowledge_graph.py` | **수정** - 새 엔티티 타입 추가 |
| `dashboard_api.py` | 새 API 엔드포인트 + DOCX 개선 |
| `config/search_queries.json` | 신규 - 검색 키워드 설정 |
| `config/public_apis.json` | **신규** - 공공 API 설정 |
| `config/ir_sources.json` | **신규** - IR 소스 설정 |

---

## 검증 방법

1. **정확도**: 외부 소스가 실제 순위 변동과 연관되는지 검증
2. **활용도**: 마케팅팀 피드백 - 인사이트가 실제 의사결정에 도움되는지
3. **출처 신뢰도**: 인용된 출처의 정확성 주기적 검토
4. **IR 연동**: 분기 실적 발표 후 인사이트에 자동 반영되는지 확인

---

## 참고 링크

### 아모레퍼시픽 IR
- [Quarterly Results](https://www.apgroup.com/int/en/investors/amorepacific-corporation/ir-reports/quarterly-results/quarterly-results.html)

### 공공데이터 & 통계
- [공공데이터 포털](https://www.data.go.kr/)
- [식약처 기능성화장품 API](https://www.data.go.kr/data/15095680/openapi.do)
- [관세청 수출입통계 API](https://www.data.go.kr/data/15100475/openapi.do)
- [KOSIS 국가통계포털](https://kosis.kr/)

### 전문기관 & 연구원
- [대한화장품산업연구원](https://www.kcii.re.kr/)
- [KHISS 보건산업통계](https://khiss.go.kr/menu?menuId=MENU00319)
- [한국무역협회 K-STAT](https://stat.kita.net/)
- [KDI 경제정보센터](https://eiec.kdi.re.kr/)
- [삼정KPMG 경제연구원](https://kpmg.com/kr/ko/home/services/special-service/eri.html)
- [메리츠증권 Telegram](https://t.me/s/meritz_research)

### 글로벌 뷰티 전문 매체
- [WWD (Women's Wear Daily)](https://wwd.com/)
- [Cosmetics Design Asia](https://www.cosmeticsdesign-asia.com/)
- [KEDGlobal Beauty](https://www.kedglobal.com/beauty-cosmetics)
- [The Cut](https://www.thecut.com/)

### 웹 트래픽 분석
- [SimilarWeb](https://www.similarweb.com/)

---

## 참고 인사이트 (수집된 외부 소스)

### KEDGlobal: 2025 아마존 프라임 데이 성과

> **출처**: kedglobal.com (2025.07.21)

| 순위 | 브랜드 | 점유율 | 비고 |
|------|--------|--------|------|
| 1위 | **MEDICUBE** (APR) | 9.3% | K-Beauty 1위 |
| 9위 | **LANEIGE** (아모레퍼시픽) | 3.0% | Lip Sleeping Mask 립케어 1위 |
| 10위 | **BIODANCE** | - | - |
| 10위 밖 | COSRX | - | 전년 1위 → 급락 (반복 마케팅, 낮은 할인율) |

### Cosmetics Design Asia: LANEIGE Gen Z 전략

> **출처**: cosmeticsdesign-asia.com (2024.03.19)

**신제품**: Bouncy and Firm Sleeping Mask (3번째 슬리핑 마스크)

**핵심 성분**:
- 펩타이드 + Peony & Collagen Complex
- 5가지 히알루론산
- Hydro-melt Glow Capsules
- 녹차 락토바실러스

**마케팅 전략**:
- "슬로우 에이징" (Pre-aging care) 메시지
- Gen Z 타겟: 20대부터 피부 노화 관리 시작
- 앰버서더: Sydney Sweeney (할리우드 배우)

**성과 지표**:
- Lip Glowy Balm: 미국에서 **6초마다 1개 판매**
- 2024년 30주년 기념 Water Bank 라인 재출시
