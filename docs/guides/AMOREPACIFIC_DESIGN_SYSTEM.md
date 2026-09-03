# AMOREPACIFIC Design System

> 대시보드 UI 컴포넌트에 적용되는 AMOREPACIFIC 디자인 가이드

---

## 1. 브랜드 철학

**"단아하고 지적이며, 건강한 아름다움"**

- **단아함**: 절제된 색상, 깔끔한 여백
- **지적임**: 명확한 정보 계층, 가독성 높은 타이포그래피
- **건강한 아름다움**: 자연스러운 그라데이션, 부드러운 곡선

---

## 2. 컬러 시스템

### 프라이머리 컬러

| 이름 | HEX | CSS 변수 | 용도 |
|------|-----|----------|------|
| **Pacific Blue** | `#001C58` | `--pacific-blue` | 헤더, 사이드바, 주요 CTA, 강조 텍스트 |
| **Amore Blue** | `#1F5795` | `--amore-blue` | 보조 강조, 링크, 그라데이션 끝점 |
| **White** | `#FFFFFF` | `--ap-white` | 배경, 카드, 툴팁 |

### 세컨더리 컬러

| 이름 | HEX | CSS 변수 | 용도 |
|------|-----|----------|------|
| **Text Secondary** | `#7D7D7D` | `--text-secondary` | 보조 텍스트, 설명 |
| **Border Light** | `#E5E7EB` | `--border-light` | 테두리, 구분선 |
| **Success** | `#22C55E` | `--success` | 긍정 지표, 상승 |
| **Danger** | `#EF4444` | `--danger` | 부정 지표, 하락, 경고 |
| **Warning** | `#F59E0B` | `--warning` | 주의, 모니터링 필요 |

### CSS 변수 정의

```css
:root {
    --pacific-blue: #001C58;
    --amore-blue: #1F5795;
    --ap-white: #FFFFFF;
    --text-secondary: #7D7D7D;
    --border-light: #E5E7EB;
    --success: #22C55E;
    --danger: #EF4444;
    --warning: #F59E0B;
}
```

---

## 3. 타이포그래피

### 폰트 패밀리

```css
font-family: 'Arita Dotum', 'Noto Sans KR', -apple-system, BlinkMacSystemFont, sans-serif;
```

**폴백 체인**: Arita Dotum → Noto Sans KR → 시스템 폰트

### 폰트 웨이트

| 웨이트 | 용도 |
|--------|------|
| 300 (Light) | PACIFIC 워드마크 |
| 400 (Regular) | 본문 텍스트 |
| 500 (Medium) | 소제목 |
| 600 (SemiBold) | 강조, 헤더 |
| 700 (Bold) | AMORE 워드마크 |

### 텍스트 스타일

```css
/* 툴팁 헤더 */
.tooltip-header {
    font-weight: 600;
    font-size: 11px;
    color: var(--amore-blue);
    text-transform: uppercase;
    letter-spacing: 1px;
}

/* 본문 텍스트 */
.tooltip-content {
    font-size: 12px;
    line-height: 1.6;
    color: var(--text-secondary);
}

/* 강조 텍스트 */
.tooltip-content strong {
    color: var(--pacific-blue);
    font-weight: 600;
}
```

---

## 4. 컴포넌트 패턴

### 4.1 툴팁 (Tooltip)

모든 툴팁은 동일한 디자인 언어를 따릅니다.

#### 구조

```
┌─────────────────────────────────┐
│▌ TOOLTIP HEADER                 │  ← 대문자, letter-spacing: 1px
│  ─────────────────────────────  │  ← border-bottom 구분선
│  본문 텍스트가 여기에 표시됩니다.  │
│  **강조 텍스트**는 Pacific Blue  │
│                                 │
│  ┌─────────────────────────┐   │
│  │ TIP 박스 (선택적)        │   │  ← 배경 + 왼쪽 테두리
│  └─────────────────────────┘   │
└─────────────────────────────────┘
 ▲ 왼쪽 그라데이션 바 (4px)
```

#### CSS 구현

```css
.tooltip {
    /* 레이아웃 */
    position: absolute;
    min-width: 240px;
    max-width: 320px;
    padding: 16px 20px;
    overflow: hidden;

    /* 배경 & 테두리 */
    background: var(--ap-white);
    color: var(--pacific-blue);
    border: 1px solid var(--border-light);
    border-radius: 8px;

    /* 그림자 */
    box-shadow: 0 8px 32px rgba(0, 28, 88, 0.12);

    /* 타이포그래피 */
    font-family: 'Arita Dotum', 'Noto Sans KR', sans-serif;
    font-size: 12px;
    line-height: 1.6;

    /* 애니메이션 */
    opacity: 0;
    visibility: hidden;
    transition: all 0.2s ease;
    z-index: 10000;
}

/* 왼쪽 그라데이션 바 */
.tooltip::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    width: 4px;
    height: 100%;
    background: linear-gradient(180deg, var(--pacific-blue) 0%, var(--amore-blue) 100%);
    border-radius: 8px 0 0 8px;
}

/* 화살표 (선택적) */
.tooltip::after {
    content: '';
    position: absolute;
    bottom: 100%;
    left: 20px;
    border: 8px solid transparent;
    border-bottom-color: var(--ap-white);
    filter: drop-shadow(0 -2px 2px rgba(0, 28, 88, 0.05));
}

/* 헤더 */
.tooltip .tooltip-header {
    font-weight: 600;
    font-size: 11px;
    color: var(--amore-blue);
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 12px;
    padding-bottom: 8px;
    border-bottom: 1px solid var(--border-light);
}

/* 호버 시 표시 */
.trigger:hover .tooltip {
    opacity: 1;
    visibility: visible;
}
```

#### TIP 박스

```css
.tooltip .tip-box {
    margin-top: 12px;
    padding: 10px 12px;
    background: rgba(31, 87, 149, 0.06);
    border-radius: 6px;
    border-left: 3px solid var(--amore-blue);
}
```

---

### 4.2 배너 (Banner)

#### Daily AI Insight 배너 구조

```
┌──────────────────────────────────────────────────────────┐
│▌┌─────────────┐                                          │
│ │   AMORE     │  Daily AI Insight — LANEIGE             │
│ │   PACIFIC   │  Amazon US 내 라네즈 브랜드는...         │
│ │  ─────────  │                                          │
│ │  AI Insight │                                          │
│ └─────────────┘                                          │
└──────────────────────────────────────────────────────────┘
   ▲ 브랜드 영역      ▲ 콘텐츠 영역
   (그라데이션 배경)   (흰색 배경)
```

#### CSS 구현

```css
.insight-banner {
    background: var(--ap-white);
    border: 1px solid var(--border-light);
    border-radius: 8px;
    padding: 0;
    position: relative;
    overflow: hidden;
    display: flex;
}

/* 왼쪽 그라데이션 바 */
.insight-banner::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    width: 4px;
    height: 100%;
    background: linear-gradient(180deg, var(--pacific-blue) 0%, var(--amore-blue) 100%);
}

/* 브랜드 영역 */
.insight-brand-area {
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    padding: 20px 24px;
    background: linear-gradient(135deg, var(--pacific-blue) 0%, var(--amore-blue) 100%);
    min-width: 140px;
}

/* 워드마크 */
.insight-brand-wordmark {
    display: flex;
    flex-direction: column;
    align-items: center;
    color: white;
    font-size: 12px;
    line-height: 1.2;
}

.insight-brand-wordmark .amore {
    font-weight: 700;
    letter-spacing: 3px;
}

.insight-brand-wordmark .pacific {
    font-weight: 300;
    letter-spacing: 2px;
}
```

---

### 4.3 카드 (Card)

```css
.card {
    background: var(--ap-white);
    border-radius: 8px;
    border: 1px solid var(--border-light);
    overflow: visible;
    transition: all 0.2s ease;
    position: relative;
}

/* 호버 시 왼쪽 액센트 라인 */
.card::before {
    content: '';
    position: absolute;
    left: 0;
    top: 0;
    bottom: 0;
    width: 3px;
    background: var(--amore-blue);
    border-radius: 8px 0 0 8px;
    opacity: 0;
    transition: opacity 0.2s ease;
}

.card:hover::before {
    opacity: 1;
}
```

---

### 4.4 알림 (Alert)

```css
.alert {
    display: flex;
    align-items: flex-start;
    gap: 12px;
    padding: 14px 18px;
    background: var(--ap-white);
    border: 1px solid var(--border-light);
    border-radius: 8px;
    border-left: 4px solid var(--amore-blue);
}

/* 경고 상태 */
.alert.warning {
    border-left-color: var(--warning);
    background: rgba(245, 158, 11, 0.04);
}

/* 위험 상태 */
.alert.danger {
    border-left-color: var(--danger);
    background: rgba(239, 68, 68, 0.04);
}
```

---

### 4.5 Drawer / Modal 헤더

```css
.drawer-header {
    background: var(--ap-white);
    padding: 20px 24px;
    border-bottom: 1px solid var(--border-light);
    position: relative;
}

/* 하단 그라데이션 구분선 */
.drawer-header::after {
    content: '';
    position: absolute;
    bottom: 0;
    left: 0;
    right: 0;
    height: 3px;
    background: linear-gradient(90deg, var(--pacific-blue) 0%, var(--amore-blue) 50%, transparent 100%);
}

/* 워드마크 스타일 */
.drawer-brand-wordmark {
    display: flex;
    align-items: baseline;
    gap: 0;
}

.drawer-brand-wordmark .amore {
    color: var(--pacific-blue);
    font-weight: 600;
    font-size: 14px;
    letter-spacing: 2px;
}

.drawer-brand-wordmark .pacific {
    color: var(--amore-blue);
    font-weight: 300;
    font-size: 14px;
    letter-spacing: 1.5px;
}
```

---

## 5. 그림자 시스템

| 이름 | 값 | 용도 |
|------|---|------|
| **Shadow SM** | `0 1px 3px rgba(0, 28, 88, 0.08)` | 버튼, 입력 필드 |
| **Shadow MD** | `0 4px 12px rgba(0, 28, 88, 0.1)` | 카드 호버 |
| **Shadow LG** | `0 8px 32px rgba(0, 28, 88, 0.12)` | 툴팁, 모달, 드롭다운 |

---

## 6. 간격 시스템

| 토큰 | 값 | 용도 |
|------|---|------|
| `xs` | 4px | 아이콘과 텍스트 사이 |
| `sm` | 8px | 관련 요소 간격 |
| `md` | 12px | 섹션 내 요소 간격 |
| `lg` | 16px | 카드 패딩, 섹션 간격 |
| `xl` | 20px | 툴팁 패딩 |
| `2xl` | 24px | 카드 간 간격 |

---

## 7. 애니메이션

```css
/* 기본 전환 */
transition: all 0.2s ease;

/* 호버 효과 */
transition: opacity 0.2s ease, visibility 0.2s ease;

/* 확장/축소 */
transition: max-height 0.3s ease;
```

---

## 8. 반응형 브레이크포인트

| 이름 | 값 | 대상 |
|------|---|------|
| Mobile | `max-width: 768px` | 스마트폰 |
| Tablet | `max-width: 1024px` | 태블릿 |
| Desktop | `min-width: 1025px` | 데스크톱 |

### 모바일 툴팁 조정

```css
@media (max-width: 768px) {
    .tooltip {
        min-width: 200px;
        max-width: 280px;
        padding: 14px 16px;
        font-size: 11px;
    }
}
```

---

## 9. 접근성 고려사항

- **색상 대비**: Pacific Blue (#001C58) on White = 14.7:1 (WCAG AAA)
- **포커스 표시**: `outline: 2px solid var(--amore-blue); outline-offset: 2px;`
- **애니메이션 감소**: `@media (prefers-reduced-motion: reduce)` 지원
- **최소 터치 영역**: 44px × 44px (모바일)

---

## 10. 적용된 컴포넌트 목록

| 컴포넌트 | CSS 클래스 | 파일 위치 |
|----------|-----------|-----------|
| 상태 카드 툴팁 | `.status-card-tooltip` | `dashboard_v4.html:965` |
| 차트 타이틀 툴팁 | `.chart-title-tooltip` | `dashboard_v4.html:1149` |
| KPI 카드 툴팁 | `.kpi-tooltip` | `dashboard_v4.html:1238` |
| 테이블 헤더 툴팁 | `.th-tooltip` | `dashboard_v4.html:1343` |
| 정보 팝업 툴팁 | `.info-popup-tooltip` | `dashboard_v4.html:1438` |
| 제품명 툴팁 | `.product-name-tooltip` | `dashboard_v4.html:1858` |
| 셀렉터 툴팁 | `.selector-tooltip` | `dashboard_v4.html:3107` |
| Daily AI Insight 배너 | `.insight-banner` | `dashboard_v4.html:812` |
| AI Strategy Drawer | `.drawer-header` | `dashboard_v4.html:2489` |
| 날짜 알림 | `.missing-dates-alert` | `dashboard_v4.html:763` |

---

## 11. 버전 히스토리

| 버전 | 날짜 | 변경 내용 |
|------|------|----------|
| 1.0 | 2026-01-28 | 초기 디자인 시스템 문서화 |

---

## 참고 자료

- [AMOREPACIFIC 브랜드 가이드](https://design.amorepacific.com/)
- [Arita 폰트 다운로드](https://design.amorepacific.com/arita/)
- AMOREPACIFIC 경영실적 발표 자료 (2025 3Q)
