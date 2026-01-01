# Home Page Insight Rules
Home 페이지는 분석 화면이 아니라, 마케터가 매일 3–5분 내에
**상황을 파악하고 다음 행동(L1/L2/L3)으로 이동**할 수 있도록 돕는 “요약 허브”이다.

본 문서는 Home에서 표시되는 **요약 문구/상태 배지/Action Queue/추천 이동**을
일관되게 생성하기 위한 룰을 정의한다.

> 원칙: Home은 **새로운 원인·결론을 생성하지 않는다.**
> Home은 이미 계산된 지표를 **요약·정렬·안내**만 한다.

---

## 0. 적용 범위 및 입력 데이터

### 0.1 입력 데이터 (Home에서 접근 가능)
- 계산된 지표 값(예: SoS, HHI, Brand Avg Rank, CPI, Avg Rating Gap, Churn Rate, Rank Shock, Rank Volatility, Streak Days, Rating Trend)
- 선택된 컨텍스트(brand, market, date range)
- 이벤트 후보 목록(예: Rank Shock 발생 제품 리스트)

### 0.2 Home에서 금지되는 작업
- 지표 재계산
- 원인 확정(재고/광고/품질/계절 등 단정)
- 매출/판매량/ROI 추정 또는 예측
- “정답/결론” 형태의 단정 문장

---

## 1. Home 구성 요소별 출력 규칙

## 1.1 Daily Insight Summary (오늘의 한 줄 요약)

### 목표
- 1문장으로 “현재 상태”를 요약
- **리스크가 있으면 리스크 우선**, 없으면 “안정” 중심

### 문장 톤
- 반드시 완곡 표현 사용:
  - “~일 수 있습니다”, “~가능성이 있습니다”, “~신호로 해석될 수 있습니다”
- 금지 표현:
  - “원인은 ~입니다”, “확실히 ~입니다”, “반드시 ~해야 합니다”

### 생성 우선순위(Top-down)
1) **이상 신호(Alerts)**가 존재하면 → 이상 신호 기반 문장
2) **가격·품질 불일치**가 존재하면 → 포지셔닝 리스크 문장
3) **시장 구조**가 강하게 치우치면 → 시장 맥락 문장
4) 그 외 → “전반적으로 안정” 문장

### 템플릿(예시)
- 이상 신호 우선:
  - “오늘은 일부 상품에서 순위 급변 신호가 감지되어, 단기 모니터링이 필요할 수 있습니다.”
- 가격·품질 불일치:
  - “프리미엄 가격 포지션 대비 평점 경쟁력이 낮아, 가격-품질 인식 간 불일치 가능성을 점검해볼 수 있습니다.”
- 시장 맥락:
  - “시장 집중도가 높아 경쟁 구도가 고착화된 편으로, 단기 변동성보다는 구조적 흐름을 함께 고려할 필요가 있습니다.”
- 안정:
  - “주요 지표가 큰 변동 없이 유지되어 전반적으로 안정적인 흐름으로 보입니다.”

> 근거 문서: `Metric_Interpretation_Guide.md`, `Indicator_Combination_Playbook.md`

---

## 1.2 Brand Health Status (브랜드 건강 상태 배지/카드)

### 목표
- 수치 대신 “상태”로 요약
- 3개 카드(권장): 브랜드 노출, 시장 구조, 이상 신호

### 상태 정의 (권장 라벨)
- 브랜드 노출 상태(SoS 중심):
  - Stable / Up / Down
- 시장 구조(HHI 중심):
  - Competitive / Concentrated
- 이상 신호:
  - None / Detected

### 임계값(프로토타입 기본값)
> 실제 운영에서는 시장/카테고리별 보정 가능. MVP에서는 고정값 사용.

- SoS 전일 대비 변화율:
  - Up: ≥ +1.0%p
  - Down: ≤ -1.0%p
  - Stable: 그 외
- HHI:
  - Concentrated: ≥ 0.25 (집중 시장)
  - Competitive: < 0.25
- Alerts:
  - Detected: Rank Shock 발생 제품 수 ≥ 1
  - None: 0

### 표시 규칙
- 상태 카드에는:
  - 라벨(Stable/Up/Down 등)
  - 짧은 설명 1줄(선택)
- 색상:
  - Up: positive
  - Down/Detected: warning
  - Stable: neutral

> 근거 문서: `Strategic_Indicators_Definition.md`, `Metric_Interpretation_Guide.md`

---

## 1.3 Action Queue (오늘의 확인 필요 항목)

### 목표
- 마케터의 다음 행동을 최소 클릭으로 유도
- “확인해야 할 것”을 **우선순위로 정렬**해 2~3개만 노출

### Action Queue 항목 유형
- Category-type → L2로 이동
- Product-type → L3로 이동

### 생성 규칙(우선순위)
1) Rank Shock 발생 제품이 있으면:
   - “순위 급변 상품 {N}개 확인 필요” (→ L3)
2) 가격·품질 불일치 신호가 있으면:
   - “가격 대비 평점 열위 카테고리 점검 필요” (→ L2)
3) 시장 변동성(Churn Rate)이 높으면:
   - “시장 변동성 상승: 신규 진입/이탈 흐름 점검” (→ L2)
4) 그 외:
   - “오늘은 주요 경고 신호가 없습니다” (이 경우 클릭 항목 없이 안내만 표시 가능)

### 표기 방식
- 각 항목은:
  - 제목(짧게)
  - 보조 설명 1줄(선택)
  - 이동 대상(L2/L3) 라벨 표시(예: “Go to L3”)

> 근거 문서: `Indicator_Combination_Playbook.md`

---

## 1.4 Market Pulse Snapshot (시장 분위기 요약)

### 목표
- “성과 변화가 시장 때문인지” 맥락 제공

### 입력 지표(권장)
- HHI (시장 집중도)
- Churn Rate (시장 역동성)

### 출력 규칙(문장 1개 + 배지 1~2개)
- HHI가 높고 Churn Rate가 낮으면:
  - “시장 구조가 고착화된 편이며 단기 변동은 제한적일 수 있습니다.”
- HHI가 낮고 Churn Rate가 높으면:
  - “경쟁이 분산되어 있고 변동성이 높아 트렌드 변화에 민감할 수 있습니다.”
- 그 외:
  - “시장 구조와 변동성 지표를 함께 보며 추이를 모니터링할 필요가 있습니다.”

> 근거 문서: `Metric_Interpretation_Guide.md`

---

## 1.5 Price vs Quality Signal (가격·품질 리스크 요약)

### 목표
- CPI와 Avg Rating Gap의 조합을 “리스크” 관점으로 1줄 요약
- **문제 없으면 ‘정상’으로 처리**(과대 경고 금지)

### 기본 규칙(조합)
- CPI > 100 & Avg Rating Gap < 0:
  - “프리미엄 가격 대비 평점 열위 신호 감지”
- CPI < 100 & Avg Rating Gap > 0:
  - “가성비 경쟁력 신호(가격 대비 만족도 우위)”
- 그 외:
  - “가격·품질 신호가 혼재되어 추가 확인이 필요할 수 있습니다.”

### 표시 규칙
- 리스크형 문구는 warning 배지
- 긍정 신호는 positive 배지
- 혼재/중립은 neutral 배지

> 근거 문서: `Indicator_Combination_Playbook.md`, `Metric_Interpretation_Guide.md`

---

## 1.6 Issue Preview (Top 이슈 미리보기)

### 목표
- Home에서 “최상위 3개”만 미리 노출하고 상세는 L3에서 확인

### 구성(권장)
- Rank Shock Top 3 (제품명 + 변화폭/날짜)
- Streak Days Top 3 (제품명 + 연속 체류일)

### 정렬 규칙
- Rank Shock 리스트: 하락 폭 큰 순(또는 severity 높은 순)
- Streak 리스트: 연속 체류일 큰 순

### 클릭 동작
- 리스트 아이템 클릭 → L3(Product View)로 이동
- 이동 시 해당 제품이 강조되도록(선택) query 상태/하이라이트 적용

> 근거 문서: `Metric_Interpretation_Guide.md`

---

## 1.7 Next Best Step (추천 다음 행동)

### 목표
- 사용자가 “어디로 가야 할지” 고민하지 않게 2개 버튼 제공

### 버튼 규칙(조건부 라벨)
- 가격·품질 리스크가 있으면:
  - “가격/품질 진단 보러가기 → L2”
- Rank Shock가 있으면:
  - “이상 신호 상품 보러가기 → L3”
- 둘 다 없으면:
  - “브랜드/시장 현황 보러가기 → L1”
  - “카테고리 현황 점검 → L2” (선택)

> 근거 문서: `Indicator_Combination_Playbook.md`

---

## 1.8 Chat Entry CTA (Home 전용 챗봇 진입)

### 목표
- Home 요약을 기반으로 “해석”만 요청하도록 유도

### UI 문구
- 버튼: “오늘 상황을 어떻게 해석하면 좋을까요?”
- 안내: “지표 해석과 판단 보조만 제공합니다(결론·예측·원인 확정 없음).”

### 챗봇에 전달할 컨텍스트(권장)
- 현재 선택한 brand/market/date range
- Home에서 생성된:
  - Daily Insight Summary
  - Brand Health Status(3개)
  - Action Queue(최대 3개)

### 금지
- 챗봇이 지표를 재계산하게 하는 지시
- “왜 떨어졌어?”에 대해 원인을 단정하는 응답

> 근거 문서: `Metric_Interpretation_Guide.md`

---

## 2. 출력 안전장치(Guardrails)

### 2.1 단정 방지
- 항상 가능성 표현 사용
- 최소 1개의 주의 문구를 포함(선택):
  - “단기 노이즈일 가능성도 있어 추이 확인이 필요합니다.”

### 2.2 과대 경고 방지
- Alert가 없으면 “문제 없음”을 명시해 사용자 불안 최소화
- 리스크 문구는 **조건 충족 시에만** 노출

### 2.3 정보 과밀 방지
- Home에서 노출하는 리스트는:
  - Action Queue: 최대 3개
  - Issue Preview: 각 3개
- 상세 차트는 Home에서 금지

---

## 3. RAG 질의/참조 권장 패턴(구현 참고)

### Home 요약 생성 시
- Query 1: “CPI + Rating Gap 해석” → `Indicator_Combination_Playbook.md`
- Query 2: “시장 구조/변동성 해석” → `Metric_Interpretation_Guide.md`
- Query 3: “단정 방지 문장 템플릿” → `Metric_Interpretation_Guide.md`

> 참고: Home은 해석 근거를 문서에서 끌어오되, 최종 문장은 1줄로 제한한다.

---

## 4. 프로토타입 기본 더미 값 예시(선택)

- SoS: 15% (Stable)
- HHI: 0.30 (Concentrated)
- Brand Avg Rank: 22
- CPI: 120
- Avg Rating Gap: -0.5
- Churn Rate: 0.18
- Rank Shock count: 3
- Streak Days top: 7, 6, 5
- Rating Trend: 일부 하락

> 본 값은 UI 검증용이며 실제 값과 무관하다.
