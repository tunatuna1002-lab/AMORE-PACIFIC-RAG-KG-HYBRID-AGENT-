#!/usr/bin/env python
"""
리포트 생성기 테스트 스크립트
=============================
새로운 AMOREPACIFIC IR 스타일 리포트 생성 테스트

Usage:
    python scripts/test_report_generator.py
"""

import sys
import tempfile
from pathlib import Path

# 프로젝트 루트를 path에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_docx_generator():
    """DOCX 생성기 기본 테스트"""
    from src.tools.report_generator import DocxReportGenerator

    print("\n" + "=" * 60)
    print("TEST: DocxReportGenerator")
    print("=" * 60)

    output_dir = tempfile.mkdtemp()
    generator = DocxReportGenerator(output_dir=output_dir)

    # 샘플 데이터 - 전문 애널리스트 수준의 리포트
    report_data = {
        "title": "LANEIGE Amazon US 경쟁력 분석 보고서",
        "subtitle": "Weekly Insight Report",
        "start_date": "2026-01-14",
        "end_date": "2026-01-20",
        "sections": [
            {
                "id": 1,
                "title": "Executive Summary",
                "content": """■ 투자 의견: BUY (목표 SoS 5.0%, 현재 2.8%)
LANEIGE는 Amazon US Lip Care 카테고리(TAM $1.2B)에서 K-Beauty 프리미엄 세그먼트 1위 포지션을 공고히 하고 있습니다. 금주 SoS(Share of Shelf)가 전주 대비 +0.3%p 상승한 2.8%를 기록하며, 3분기 연속 상승세를 이어가고 있습니다.

■ 핵심 성과 지표 (KPI)
- SoS (Share of Shelf): 2.5% → 2.8% (+0.3%p, WoW)
- Top 10 SKU 수: 3개 유지 (Berry #2, Original #8, Gummy Bear #15 → #10 진입 임박)
- 가중평균 순위: 15위 → 12위 (-3계단, 상위권 진입)
- 리뷰 성장률: +2,340건/주 (+8.2% WoW)
- Buy Box 점유율: 98.7% (Amazon 직접 판매 유지)

■ 주요 이슈 및 대응
경쟁사 e.l.f. Cosmetics의 Q1 공격적 프로모션(BOGO 50% OFF)으로 Lip Care 전체 카테고리 가격 하락 압박이 발생했습니다. 그러나 LANEIGE는 가격 방어에 성공하며 프리미엄 포지셔닝을 유지했습니다. 이는 Lip Sleeping Mask의 압도적 브랜드 인지도(Unaided Recall 34%)와 45,000건 이상의 누적 리뷰가 만든 진입장벽 효과로 분석됩니다.

■ 애널리스트 종합 의견
현 시점에서 LANEIGE의 Amazon US 채널 성과는 '안정적 성장'으로 평가합니다. 단기적으로는 Q1 Prime Day(3월 예정) 대비 재고 확보가 핵심 과제이며, 중기적으로는 Skin Care 카테고리 확장을 통한 크로스셀링 기회 포착이 성장 동력이 될 것입니다.""",
                "kpis": [
                    {"name": "LANEIGE SoS", "value": "2.8%", "change": "+0.3%p", "trend": "up"},
                    {"name": "Top 10 SKU", "value": "3개", "change": "유지", "trend": "neutral"},
                    {"name": "가중평균 순위", "value": "#12", "change": "-3", "trend": "up"},
                    {
                        "name": "주간 리뷰 증가",
                        "value": "+2,340건",
                        "change": "+8.2%",
                        "trend": "up",
                    },
                    {
                        "name": "Buy Box 점유율",
                        "value": "98.7%",
                        "change": None,
                        "trend": "neutral",
                    },
                ],
            },
            {
                "id": 2,
                "title": "LANEIGE 심층 분석",
                "content": """■ 제품 포트폴리오 성과 분석 (SKU Level)

[Hero Product] Lip Sleeping Mask - Berry (B00LMJ2TK4)
- Amazon BSR: #2 (Lip Care) / #847 (Beauty & Personal Care 전체)
- 가격: $24.00 (MSRP $26.00, 7.7% 할인 적용 중)
- 리뷰: 45,847건 (평점 4.7/5.0, 5점 비율 78%)
- 예상 월 매출: $1.2M (추정 월 판매량 50,000 units)
- FBA 재고 상태: 정상 (30일분 확보)
- 리뷰 성장률: +1,200건/월 (카테고리 평균 대비 3.2x)

[Growth Product] Lip Sleeping Mask - Original (B08KFYWXW2)
- Amazon BSR: #8 (Lip Care)
- 가격: $24.00
- 리뷰: 32,156건 (평점 4.6/5.0)
- 예상 월 매출: $720K
- 성장 포인트: Berry 대비 낮은 인지도, 마케팅 집중 필요

[Emerging Product] Lip Sleeping Mask - Gummy Bear (B0BKJX8YZ3)
- Amazon BSR: #15 (Lip Care) → Top 10 진입 임박
- 가격: $24.00
- 리뷰: 8,234건 (평점 4.5/5.0)
- 예상 월 매출: $340K
- 성장률: +45% MoM (Gen Z 타겟 마케팅 효과)

■ 가격 전략 분석 (Price Positioning Matrix)
LANEIGE는 Lip Care 카테고리 내 '프리미엄 K-Beauty' 포지션을 점유하고 있습니다.

가격대별 경쟁 구도:
- Value ($5-10): Burt's Bees, ChapStick, Nivea → 점유율 35%
- Mid-tier ($10-18): Aquaphor, Vaseline Lip Therapy → 점유율 28%
- Premium ($18-28): LANEIGE, Tatcha, Summer Fridays → 점유율 25%
- Luxury ($28+): La Mer, Sisley → 점유율 12%

LANEIGE의 $24 가격점은 Premium 세그먼트 내 최적 가격으로, 경쟁사 Summer Fridays($22)보다 높지만 Tatcha($28)보다 접근성이 높아 '접근 가능한 럭셔리(Accessible Luxury)' 이미지를 구축하고 있습니다.

■ 고객 세그먼트 분석 (Review Mining 기반)
- Primary: 25-34세 여성, K-Beauty 관심층 (42%)
- Secondary: 18-24세 여성, TikTok/Instagram 영향층 (31%)
- Tertiary: 35-44세 여성, 프리미엄 스킨케어 사용자 (19%)
- Others: 남성, 선물 구매자 등 (8%)

주요 구매 동기 (리뷰 키워드 분석):
1. "overnight results" (32%) - 즉각적 효과 기대
2. "K-beauty" (24%) - 한국 화장품 신뢰
3. "gift" (18%) - 선물용 수요
4. "dermatologist recommended" (12%) - 전문가 추천 신뢰""",
            },
            {
                "id": 3,
                "title": "경쟁 환경 분석",
                "content": """■ 경쟁 구도 개요 (Porter's Five Forces 관점)

[산업 내 경쟁 강도: 높음]
Lip Care 카테고리는 진입장벽이 낮고 제품 차별화가 어려워 경쟁 강도가 높습니다. 다만 LANEIGE는 '슬리핑 마스크'라는 새로운 서브카테고리를 개척하여 차별화에 성공했습니다.

[신규 진입자 위협: 중간]
DTC(Direct-to-Consumer) 브랜드들의 Amazon 진출이 증가하고 있으나, 리뷰 누적과 브랜드 인지도 구축에 시간이 필요하여 단기 위협은 제한적입니다.

[대체재 위협: 낮음]
립밤, 립글로스 등 대체재가 존재하나, '오버나이트 립 트리트먼트'라는 특수 용도로 대체재 위협이 낮습니다.

■ 주요 경쟁사 심층 분석

[Tier 1 - 직접 경쟁] e.l.f. Cosmetics
- SoS: 4.2% (LANEIGE 대비 +1.4%p)
- 핵심 제품: Lip Exfoliator ($6), Holy Hydration Lip Mask ($9)
- 전략: 가격 파괴 + TikTok 바이럴 마케팅
- SWOT 분석:
  · Strength: 저가, 높은 접근성, Z세대 인지도
  · Weakness: 프리미엄 이미지 부재, 제품 품질 인식
  · Opportunity: Amazon 점유율 확대
  · Threat: 마진 압박, 브랜드 희석
- 위협 평가: ★★★☆☆ (중간) - 가격대가 달라 직접 경쟁 제한적

[Tier 1 - 직접 경쟁] Maybelline (L'Oreal)
- SoS: 3.8%
- 핵심 제품: Baby Lips ($4.49), Lifter Gloss ($9.49)
- 전략: 대형 유통망 + 셀럽 마케팅 + TV/디지털 광고
- SWOT 분석:
  · Strength: 브랜드 인지도, 유통망, 마케팅 예산
  · Weakness: 프리미엄 Lip Care 라인 부재
  · Opportunity: 프리미엄 서브라인 론칭 가능성
  · Threat: 리소스 집중 시 강력한 경쟁자
- 위협 평가: ★★★☆☆ (중간) - 프리미엄 진출 시 위협 상승

[Tier 2 - 간접 경쟁] COSRX
- SoS: 2.1%
- 핵심 제품: Lip Sleep Sleeping Mask ($14)
- 전략: K-Beauty 트렌드 활용, 가성비 포지셔닝
- 위협 평가: ★★☆☆☆ (낮음) - 협력 가능성 높음 (K-Beauty 동반 성장)

[Tier 2 - 간접 경쟁] Summer Fridays
- SoS: 1.8%
- 핵심 제품: Lip Butter Balm ($22)
- 전략: Clean Beauty, 인플루언서 마케팅
- 위협 평가: ★★★☆☆ (중간) - 동일 프리미엄 세그먼트 경쟁

■ 시장 점유율 추이 (최근 8주)
Week -8: LANEIGE 2.1% | e.l.f. 4.0% | Maybelline 4.1%
Week -4: LANEIGE 2.4% | e.l.f. 4.1% | Maybelline 3.9%
Week 0: LANEIGE 2.8% | e.l.f. 4.2% | Maybelline 3.8%

→ LANEIGE의 점유율 상승세가 가장 가파름 (+0.7%p in 8 weeks)""",
            },
            {
                "id": 4,
                "title": "시장 동향 분석",
                "content": """■ 시장 규모 및 성장성 (TAM/SAM/SOM)

Total Addressable Market (TAM)
- US Lip Care 시장: $1.2B (2026E)
- 연평균 성장률(CAGR): +6.8% (2024-2028F)

Serviceable Addressable Market (SAM)
- Amazon US Lip Care: $380M (전체의 31.7%)
- Amazon 채널 성장률: +12.4% YoY (오프라인 대비 2x 성장)

Serviceable Obtainable Market (SOM)
- LANEIGE 현재 점유: ~$10.6M (SoS 2.8% 기준)
- 목표 점유: ~$19M (SoS 5.0%, 2026년 말)

■ 시장 집중도 분석 (HHI Index)
현재 HHI: 850 → 분산된 경쟁 시장 (Competitive Market)

HHI 해석 기준:
- HHI < 1,500: 분산 시장 (경쟁적)
- 1,500 ≤ HHI < 2,500: 중간 집중도
- HHI ≥ 2,500: 고집중 시장 (과점)

주요 플레이어별 HHI 기여도:
- e.l.f.: 17.6 (4.2%² × 100)
- Maybelline: 14.4 (3.8%²)
- LANEIGE: 7.8 (2.8%²)
- COSRX: 4.4 (2.1%²)
- Others (Long Tail): ~806

시사점: 상위 4개 브랜드가 13%만 차지하는 분산 시장으로, 점유율 확대 기회가 큼

■ K-Beauty 트렌드 심층 분석

글로벌 K-Beauty 시장 규모:
- 2025: $15.8B → 2030F: $24.2B (CAGR +8.9%)
- US 시장 내 K-Beauty 비중: 12% → 18% 예상 (2030F)

Amazon US K-Beauty 성장 동인:
1. TikTok 바이럴 효과: #KoreanSkincare 누적 조회수 45B+
2. 성분 투명성 요구 증가: Clean Beauty 트렌드와 K-Beauty의 자연 유래 성분 부합
3. 다단계 루틴 인기: "7-Skin Method", "Glass Skin" 등 K-Beauty 특유의 루틴 확산
4. Celebrity/Influencer 효과: Hailey Bieber, Selena Gomez 등 K-Beauty 공개 사용

LANEIGE 관련 트렌드 키워드 (Google Trends 기준):
- "LANEIGE Lip Sleeping Mask": 지수 85/100 (안정적 고관심)
- "K-beauty lip care": 지수 72/100 (+15% YoY)
- "overnight lip mask": 지수 68/100 (+22% YoY) ← LANEIGE가 개척한 카테고리

■ 가격대별 시장 세분화

Value Segment ($0-10): 35% 시장 점유
- 주요 브랜드: Burt's Bees, ChapStick, Nivea, Aquaphor
- 소비자 특성: 기능성 중심, 가격 민감도 높음
- 성장률: +3% YoY (성숙 시장)

Mid-tier Segment ($10-20): 28% 시장 점유
- 주요 브랜드: Vaseline Lip Therapy, Carmex, Kiehl's
- 소비자 특성: 브랜드 인지도 + 품질 중시
- 성장률: +5% YoY

Premium Segment ($20-30): 25% 시장 점유 ← LANEIGE 타겟
- 주요 브랜드: LANEIGE, Summer Fridays, Tatcha, Fresh
- 소비자 특성: 효과/성분 중시, 소셜미디어 영향
- 성장률: +12% YoY (가장 빠른 성장)

Luxury Segment ($30+): 12% 시장 점유
- 주요 브랜드: La Mer, Sisley, Augustinus Bader
- 소비자 특성: 럭셔리 브랜드 충성도
- 성장률: +8% YoY""",
            },
            {
                "id": 5,
                "title": "외부 신호 분석",
                "content": """■ 뉴스 및 미디어 모니터링 (Sentiment Analysis)

[긍정적 신호 - Positive Sentiment 78%]

1. "K-Beauty brands see unprecedented surge in US sales"
   - 출처: Beauty Independent (2026-01-18)
   - 요약: K-Beauty 카테고리 Q4 매출 +23% YoY, LANEIGE 언급
   - 영향도: ★★★★☆

2. "LANEIGE expands Sephora partnership to 850+ US stores"
   - 출처: WWD (2026-01-16)
   - 요약: Sephora 전 매장 입점, Amazon 시너지 기대
   - 영향도: ★★★★★ (Omnichannel 강화)

3. "Amazon's Beauty category sees record Q4 growth"
   - 출처: CNBC (2026-01-14)
   - 요약: Amazon Beauty 카테고리 +18% YoY, Lip Care 주도
   - 영향도: ★★★★☆

[중립적 신호 - Neutral Sentiment 15%]

4. "e.l.f. announces aggressive Q1 pricing strategy"
   - 출처: Retail Dive (2026-01-12)
   - 요약: BOGO 프로모션 확대, 가격 경쟁 심화 예상
   - 영향도: ★★★☆☆ (모니터링 필요)

[부정적 신호 - Negative Sentiment 7%]

5. "Supply chain disruptions may affect K-Beauty imports"
   - 출처: Supply Chain Dive (2026-01-10)
   - 요약: 홍해 물류 이슈로 리드타임 증가 가능성
   - 영향도: ★★☆☆☆ (단기 영향 제한적)

■ 소셜 미디어 인텔리전스

[TikTok Analytics]
- #LANEIGELipMask: 2.5M views/week (+18% WoW)
- #LipSleepingMask: 1.8M views/week
- Top Viral Content: "LANEIGE dupe vs real" 비교 영상 (긍정적 - 정품 우위 평가)
- Influencer Mention: @skincarebyhyram (2.1M followers) - Berry 플레이버 추천

[Instagram Insights]
- @laboratoirelaneige mentions: 8,400건/week (+12% WoW)
- User Generated Content: 3,200건/week
- Engagement Rate: 4.2% (뷰티 카테고리 평균 2.8% 대비 높음)
- Top Hashtags: #glassskin #kbeauty #lipcare #overnightmask

[Reddit Sentiment - r/AsianBeauty]
- 총 언급: 847건/month
- Positive: 72% | Neutral: 21% | Negative: 7%
- 주요 긍정 키워드: "holy grail", "repurchase", "overnight miracle"
- 주요 부정 키워드: "price increase", "smaller size" (2024년 용량 변경 관련)

[Twitter/X Brand Monitoring]
- 브랜드 언급: 12,400건/week
- Sentiment Score: +0.68 (Scale: -1 to +1)
- Share of Voice (Beauty 카테고리): 3.2%

■ Google Trends 분석

검색 트렌드 (US, Last 12 months):
- "LANEIGE": 지수 85/100 (안정적 고관심 유지)
- "Lip sleeping mask": 지수 72/100 (+8% YoY)
- "Korean lip care": 지수 65/100 (+15% YoY)
- "LANEIGE vs Summer Fridays": 지수 45/100 (비교 검색 증가)

계절성 패턴:
- Peak: 11-12월 (Holiday Gift Season) - 지수 100
- Trough: 6-7월 (Summer) - 지수 65
- 현재(1월): 지수 78 (Holiday 후 정상화 단계)

지역별 관심도 (Top 5 States):
1. California: 100
2. New York: 92
3. Texas: 78
4. Florida: 75
5. Illinois: 71

■ 경쟁사 소셜 벤치마킹

| Metric | LANEIGE | e.l.f. | Summer Fridays |
|--------|---------|--------|----------------|
| TikTok Views/Week | 2.5M | 8.2M | 1.1M |
| Instagram Engagement | 4.2% | 5.8% | 3.9% |
| Reddit Sentiment | +0.68 | +0.42 | +0.71 |
| Google Trends Index | 85 | 78 | 52 |

→ e.l.f.는 볼륨 높으나 센티먼트 낮음, LANEIGE는 품질 인식 우위""",
            },
            {
                "id": 6,
                "title": "리스크 및 기회 요인",
                "content": """■ 리스크 매트릭스 (Impact × Probability)

[High Impact × High Probability] - 즉시 대응 필요
1. 경쟁사 가격 전쟁 심화
   - 상세: e.l.f.의 Q1 BOGO 프로모션 확대, Maybelline 쿠폰 공세
   - 영향: SoS -0.3~0.5%p 하락 가능
   - 대응: 가격 방어 + 번들/GWP(Gift with Purchase) 전략으로 가치 제안

2. Amazon 알고리즘 변경
   - 상세: A10 알고리즘 업데이트로 리뷰 가중치 변화 가능
   - 영향: BSR 순위 변동성 증가
   - 대응: 리뷰 품질 관리, Vine 프로그램 활용

[High Impact × Medium Probability] - 모니터링 강화
3. 환율 리스크 (USD/KRW)
   - 상세: 원/달러 환율 1,450원 돌파 시 수입 원가 +8% 상승
   - 영향: 마진율 -2~3%p 압박
   - 대응: 헤지 계약, US 현지 생산 검토 (중장기)

4. 공급망 차질 (홍해 물류)
   - 상세: 홍해 우회로로 리드타임 +2주 증가
   - 영향: 재고 부족 리스크, FBA 품절 가능성
   - 대응: 안전재고 45일→60일 확대, Air freight 옵션 확보

[Medium Impact × High Probability] - 프로세스 개선
5. 리뷰 조작 의혹 대응
   - 상세: FTC의 가짜 리뷰 규제 강화, Amazon Vine 정책 변경
   - 영향: 리뷰 성장 둔화 가능
   - 대응: 정품 인증 리뷰만 유도, Vine 비중 적정화

6. 위조품 이슈
   - 상세: 3P 셀러 통한 위조품 유통 지속
   - 영향: 브랜드 이미지 훼손, 고객 불만 증가
   - 대응: Brand Registry 강화, Transparency Program 도입

■ 기회 매트릭스 (Impact × Feasibility)

[High Impact × High Feasibility] - 즉시 실행
1. Prime Day Q1 (3월) 대응
   - 기회: 카테고리 트래픽 3x 증가 예상
   - 액션: 재고 2x 확보, Lightning Deal 슬롯 확보, A+ Content 업그레이드
   - 예상 ROI: 매출 +40%, ROAS 4.5x

2. Sephora Omnichannel 시너지
   - 기회: Sephora 850개 매장 + Amazon 동시 노출
   - 액션: "Sephora Exclusive" 에디션 Amazon 동시 론칭
   - 예상 ROI: 브랜드 인지도 +25%, Cross-channel 매출 +15%

[High Impact × Medium Feasibility] - 분기 내 실행
3. Skin Care 카테고리 확장
   - 기회: Water Sleeping Mask의 Amazon 진출
   - 액션: Q2 론칭 목표, Lip Mask 고객 크로스셀링
   - 예상 ROI: 신규 매출 $2M+/year

4. Subscribe & Save 최적화
   - 기회: 재구매 고객 락인, LTV 증가
   - 액션: S&S 할인율 10%→15% 조정, 90일 주기 추천
   - 예상 ROI: 재구매율 +20%, CAC -15%

[Medium Impact × High Feasibility] - 지속 실행
5. TikTok Shop 연계
   - 기회: TikTok → Amazon 전환 퍼널 구축
   - 액션: Affiliate 링크 확대, Creator 협업
   - 예상 ROI: 신규 고객 유입 +10%

6. Gen Z 타겟 신규 플레이버
   - 기회: Gummy Bear 성공에 따른 라인 확장
   - 액션: "Cotton Candy", "Peach" 등 Z세대 선호 플레이버 테스트
   - 예상 ROI: SKU당 $500K/year 추가 매출

■ SWOT 종합 분석

Strengths (강점):
- Hero Product(Lip Sleeping Mask)의 압도적 리뷰/인지도
- K-Beauty 프리미엄 브랜드 포지셔닝
- Sephora, Ulta 등 멀티채널 시너지
- AMOREPACIFIC 모회사의 R&D/마케팅 역량

Weaknesses (약점):
- 단일 Hero Product 의존도 높음 (매출의 70%)
- 가격 경쟁력 제한 (프리미엄 포지션)
- Amazon 전용 SKU 부재
- 물류 리드타임 (한국→US)

Opportunities (기회):
- K-Beauty 트렌드 지속 성장
- Skin Care 카테고리 확장
- Prime Day, Holiday 시즌 성과 극대화
- DTC + Amazon 옴니채널 전략

Threats (위협):
- e.l.f. 등 가격 경쟁 심화
- Amazon 수수료 인상 가능성 (FBA, Ads)
- 위조품/모방 제품 증가
- 글로벌 공급망 불확실성""",
            },
            {
                "id": 7,
                "title": "전략 제언",
                "content": """■ 단기 전략 (1-3개월) - Quick Wins

[전략 1] Prime Day Q1 대응 태스크포스 구성
- 목표: Prime Day(3월) 매출 $800K 달성 (전년 대비 +40%)
- 액션 플랜:
  · Week 1-2: 재고 2x 확보 (50,000 units 추가 발주)
  · Week 3-4: Lightning Deal 슬롯 확보 (최소 3개 SKU)
  · Week 5-6: A+ Content 업그레이드, 브랜드 스토어 리뉴얼
  · Week 7-8: PPC 캠페인 예산 2x 증액, ACOS 목표 25%
- KPI: 매출 $800K, ROAS 4.5x, SoS 3.5%
- 예산: $120K (Ads $80K + Deals $40K)
- 담당: Amazon 채널팀 + 마케팅팀

[전략 2] 리뷰 성장 가속화 프로그램
- 목표: Lip Sleeping Mask 리뷰 50,000건 돌파 (현재 45,847건)
- 액션 플랜:
  · Amazon Vine 프로그램 활용 (월 100건)
  · Request a Review 자동화 (구매 후 7일)
  · 포장 내 QR 코드 리뷰 유도 (카드 삽입)
- KPI: 월 +1,500건 리뷰 증가, 5점 비율 80% 유지
- 예산: $15K/month
- 담당: CRM팀 + CS팀

[전략 3] 경쟁사 프로모션 모니터링 대시보드 구축
- 목표: 경쟁사 가격/프로모션 변동 실시간 감지
- 액션 플랜:
  · Keepa/CamelCamelCamel API 연동
  · 일일 가격 변동 알림 설정 (e.l.f., Maybelline, Summer Fridays)
  · 주간 경쟁 인텔리전스 리포트 발행
- KPI: 경쟁사 프로모션 감지 24시간 내, 대응 48시간 내
- 예산: $5K (툴 구독)
- 담당: 전략기획팀

■ 중기 전략 (3-6개월) - Growth Acceleration

[전략 4] Skin Care 카테고리 진출 (Water Sleeping Mask)
- 목표: Amazon US Skin Care 카테고리 진입, SoS 1.0% 달성
- 액션 플랜:
  · Month 1: 제품 등록, A+ Content 제작, 브랜드 스토어 연동
  · Month 2: PPC 캠페인 론칭, Lip Mask 고객 크로스셀 타겟팅
  · Month 3-4: Vine 리뷰 500건 확보, BSR Top 50 진입
  · Month 5-6: Subscribe & Save 도입, 재구매 유도
- KPI: 월 매출 $200K, BSR Top 30, 리뷰 1,000건
- 예산: $200K (마케팅 $150K + 재고 $50K)
- 담당: 신규 카테고리팀 + Amazon팀

[전략 5] Influencer/Creator 마케팅 확대
- 목표: TikTok/Instagram을 통한 신규 고객 유입 +30%
- 액션 플랜:
  · Micro-influencer (10K-100K) 50명 계약 (월 $500/인)
  · Macro-influencer (100K-1M) 10명 협업 (분기 $5K/인)
  · Amazon Attribution 링크로 성과 측정
  · UGC 콘텐츠 Amazon A+ Content 활용
- KPI: Influencer 매출 기여 15%, CPM $8 이하
- 예산: $50K/quarter
- 담당: 소셜미디어팀 + 에이전시

[전략 6] Amazon Exclusive SKU 개발
- 목표: Amazon 전용 에디션으로 채널 차별화
- 액션 플랜:
  · Amazon Exclusive 플레이버 (예: "Vanilla Bean") 개발
  · Mini Size 3-Pack Bundle ($36, 단품 대비 25% 저렴)
  · Limited Edition Holiday Set (Q4 대비)
- KPI: Exclusive SKU 매출 $100K/month
- 예산: $100K (제품 개발 + 마케팅)
- 담당: 제품개발팀 + Amazon팀

■ 장기 전략 (6-12개월) - Market Leadership

[전략 7] Amazon SoS 5% 달성 로드맵
- 목표: Lip Care 카테고리 K-Beauty #1, 전체 Top 5 진입
- 마일스톤:
  · Q2: SoS 3.2% (Skin Care 론칭 효과)
  · Q3: SoS 4.0% (Prime Day + Back to School)
  · Q4: SoS 5.0% (Holiday Season Peak)
- 필요 액션:
  · SKU 확장: 현재 8개 → 15개
  · 마케팅 예산 증액: 현재 $50K/month → $100K/month
  · 팀 확장: Amazon 전담 인력 2명 → 4명
- 예상 매출: $19M/year (현재 $10.6M 대비 +80%)

[전략 8] Amazon Choice 배지 획득
- 목표: Lip Sleeping Mask Berry SKU에서 Amazon's Choice 획득
- 조건:
  · BSR Top 3 유지 (현재 #2)
  · 평점 4.5+ 유지 (현재 4.7)
  · 리뷰 50,000건+ (목표 달성 예정)
  · Prime 배송 가능 (FBA 유지)
  · 반품률 3% 미만 (현재 1.8%)
- 액션: 위 조건 지속 관리, Amazon 담당자 커뮤니케이션
- KPI: Q3 내 Amazon's Choice 배지 획득

[전략 9] DTC + Amazon 옴니채널 통합
- 목표: 채널 간 시너지로 고객 LTV 극대화
- 액션 플랜:
  · LANEIGE.com ↔ Amazon 고객 데이터 통합 (CDP 구축)
  · DTC 첫 구매 → Amazon 재구매 유도 (S&S)
  · Amazon 구매자 → DTC 멤버십 전환
  · Sephora 연계 트래픽 활용
- KPI: 크로스채널 고객 비율 20%, LTV +25%
- 예산: $150K (CDP + 마케팅 자동화)
- 담당: 디지털팀 + IT팀

■ 실행 우선순위 매트릭스

| 전략 | Impact | Effort | 우선순위 |
|------|--------|--------|----------|
| Prime Day 대응 | High | Low | 1 (즉시) |
| 리뷰 성장 프로그램 | High | Low | 2 (즉시) |
| 경쟁사 모니터링 | Medium | Low | 3 (즉시) |
| Skin Care 진출 | High | High | 4 (Q2) |
| Influencer 확대 | Medium | Medium | 5 (Q2) |
| Exclusive SKU | Medium | High | 6 (Q3) |
| SoS 5% 로드맵 | High | High | 7 (연간) |

■ 경영진 의사결정 포인트

1. Prime Day 예산 승인: $120K (ROI 예상 4.5x)
2. Skin Care 진출 결정: 투자 $200K, 손익분기 6개월
3. 팀 증원 승인: Amazon 전담 +2명 (연 $150K)
4. 마케팅 예산 증액: $50K → $100K/month""",
            },
        ],
        "references": """[A1] "Amazon Best Sellers: Best Lip Care Products" - Amazon US, 2026-01-20
URL: https://www.amazon.com/Best-Sellers-Lip-Care/zgbs/beauty/11060711
- 실시간 BSR 순위 및 경쟁 제품 모니터링 소스

[A2] "K-Beauty Brands See Unprecedented Surge in US Sales: Q4 2025 Analysis" - Beauty Independent, 2026-01-18
URL: https://www.beautyindependent.com/k-beauty-surge-us-sales-2026
- K-Beauty 카테고리 성장률 +23% YoY 데이터 출처

[A3] "LANEIGE Expands Sephora Partnership to 850+ US Stores" - WWD (Women's Wear Daily), 2026-01-16
URL: https://wwd.com/beauty-industry-news/beauty-features/laneige-sephora-partnership-expansion-2026
- 옴니채널 전략 및 오프라인 확장 관련 뉴스

[A4] "US Cosmetics Market Size, Share & Trends Analysis Report 2026" - Statista, 2026-01-15
URL: https://www.statista.com/statistics/cosmetics-market-size-us-2026
- TAM $1.2B, CAGR +6.8% 시장 규모 데이터

[A5] "Amazon's Beauty Category Records Highest Q4 Growth in Platform History" - CNBC, 2026-01-14
URL: https://www.cnbc.com/2026/01/14/amazon-beauty-category-q4-growth.html
- Amazon Beauty 카테고리 +18% YoY 성장 데이터

[A6] "e.l.f. Cosmetics Announces Q1 2026 Pricing Strategy: BOGO Promotions" - Retail Dive, 2026-01-12
URL: https://www.retaildive.com/news/elf-cosmetics-q1-2026-pricing-strategy/
- 경쟁사 프로모션 전략 인텔리전스

[A7] "Global K-Beauty Market Report 2025-2030: $24.2B Opportunity" - Grand View Research, 2026-01-10
URL: https://www.grandviewresearch.com/industry-analysis/k-beauty-products-market
- 글로벌 K-Beauty 시장 전망 (CAGR +8.9%)

[A8] "Amazon Seller Central: A10 Algorithm Update Guidelines" - Amazon Advertising Blog, 2026-01-08
URL: https://advertising.amazon.com/blog/a10-algorithm-update-2026
- Amazon 검색 알고리즘 변경 사항 참고

[A9] "TikTok Beauty Trends Report: #KoreanSkincare 45B+ Views" - TikTok Business, 2026-01-05
URL: https://www.tiktok.com/business/en-US/blog/beauty-trends-2026
- 소셜미디어 트렌드 및 바이럴 지표

[A10] "Supply Chain Disruptions: Red Sea Shipping Impact on Cosmetics Industry" - Supply Chain Dive, 2026-01-10
URL: https://www.supplychaindive.com/news/red-sea-shipping-cosmetics-impact/
- 물류 리스크 분석 참고

[A11] "FTC Announces New Rules on Fake Reviews and Endorsements" - Federal Trade Commission, 2026-01-03
URL: https://www.ftc.gov/news-events/news/press-releases/2026/01/ftc-fake-reviews-rule
- 리뷰 규제 관련 정책 변화

[A12] "AMOREPACIFIC Q4 2025 Earnings Call Transcript" - Seeking Alpha, 2026-01-22
URL: https://seekingalpha.com/article/amorepacific-q4-2025-earnings-call
- 모회사 실적 및 글로벌 전략 방향""",
    }

    # 리포트 생성
    output_path = generator.generate_analyst_report(
        report_data=report_data,
        chart_paths=None,  # 차트 없이 테스트
        output_filename="test_report.docx",
    )

    print(f"✅ DOCX 생성 완료: {output_path}")
    print(f"   파일 크기: {output_path.stat().st_size / 1024:.1f} KB")

    return output_path


def test_pptx_generator():
    """PPTX 생성기 테스트"""
    try:
        from src.tools.report_generator import PptxReportGenerator
    except ImportError:
        print("\n⚠️ python-pptx 미설치로 PPTX 테스트 스킵")
        return None

    print("\n" + "=" * 60)
    print("TEST: PptxReportGenerator")
    print("=" * 60)

    output_dir = tempfile.mkdtemp()
    generator = PptxReportGenerator(output_dir=output_dir)

    if not generator._pptx_available:
        print("⚠️ python-pptx 미설치로 PPTX 테스트 스킵")
        return None

    report_data = {
        "title": "LANEIGE Amazon US 분석",
        "start_date": "2026-01-14",
        "end_date": "2026-01-20",
        "sections": [
            {
                "id": 1,
                "title": "Executive Summary",
                "content": "LANEIGE SoS: 2.8% (+0.3%)\nTop 10 Products: 3\nAvg Rank: #12",
            },
            {
                "id": 2,
                "title": "경쟁 환경",
                "content": "e.l.f.: 4.2%\nMaybelline: 3.8%\nCOSRX: 2.1%",
            },
        ],
    }

    output_path = generator.generate_presentation(
        report_data=report_data,
        output_filename="test_presentation.pptx",
    )

    if output_path:
        print(f"✅ PPTX 생성 완료: {output_path}")
        print(f"   파일 크기: {output_path.stat().st_size / 1024:.1f} KB")
    else:
        print("❌ PPTX 생성 실패")

    return output_path


def test_unified_generator():
    """통합 리포트 생성기 테스트"""
    from src.tools.report_generator import ReportGenerator

    print("\n" + "=" * 60)
    print("TEST: ReportGenerator (Unified)")
    print("=" * 60)

    output_dir = tempfile.mkdtemp()
    generator = ReportGenerator(output_dir=output_dir)

    report_data = {
        "title": "LANEIGE 경쟁력 분석",
        "start_date": "2026-01-14",
        "end_date": "2026-01-20",
        "sections": [
            {
                "id": 1,
                "title": "Executive Summary",
                "content": "■ 핵심 성과\nSoS 2.8% 달성",
                "kpis": [{"name": "SoS", "value": "2.8%", "change": "+0.3%", "trend": "up"}],
            },
        ],
    }

    # DOCX만 생성 (기본)
    results = generator.generate(
        report_data=report_data,
        formats=["docx"],
    )

    print("✅ 통합 생성기 테스트 완료")
    for fmt, path in results.items():
        print(f"   - {fmt.upper()}: {path}")

    return results


def test_logo_paths():
    """로고 경로 확인"""
    from src.tools.report_generator import DocxReportGenerator

    print("\n" + "=" * 60)
    print("TEST: Logo Paths")
    print("=" * 60)

    generator = DocxReportGenerator()

    for logo_type in ["color", "basic", "reverse"]:
        path = generator._get_logo_path(logo_type)
        if path:
            print(f"✅ {logo_type}: {path}")
        else:
            print(f"❌ {logo_type}: 파일 없음")


def main():
    """메인 테스트 실행"""
    print("=" * 60)
    print("AMOREPACIFIC 리포트 생성기 테스트")
    print("=" * 60)

    # 로고 경로 테스트
    test_logo_paths()

    # DOCX 테스트
    docx_path = test_docx_generator()

    # PPTX 테스트
    pptx_path = test_pptx_generator()

    # 통합 생성기 테스트
    results = test_unified_generator()

    print("\n" + "=" * 60)
    print("테스트 완료!")
    print("=" * 60)

    if docx_path:
        print("\n생성된 DOCX 파일을 열어보세요:")
        print(f"  open {docx_path}")


if __name__ == "__main__":
    main()
