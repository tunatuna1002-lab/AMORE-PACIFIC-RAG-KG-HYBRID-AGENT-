# dashboard/ - 프론트엔드

## 개요

AMORE Pacific RAG-KG Hybrid Agent의 웹 대시보드 UI입니다.
FastAPI의 `StaticFiles`로 서빙되며, AMOREPACIFIC 디자인 시스템을 따릅니다.

## 파일 목록

| 파일 | 크기 | 설명 |
|------|------|------|
| `amore_unified_dashboard_v4.html` | ~25KB | 메인 대시보드 SPA |
| `test_chat.html` | ~3KB | 챗봇 테스트 페이지 |
| `AGENTS.md` | - | 현재 문서 |

## amore_unified_dashboard_v4.html

### 주요 기능

| 섹션 | 설명 |
|------|------|
| **Overview** | 전체 KPI 요약 (SoS, HHI, CPI) |
| **AI Chatbot** | RAG+KG 하이브리드 채팅 |
| **Category Analysis** | 카테고리별 상세 분석 (5개 탭) |
| **Insight** | AI 생성 전략적 인사이트 |
| **KG Visualizer** | Knowledge Graph 시각화 (D3.js) |

### 카테고리 탭

```
Beauty & Personal Care (L0)
├── Skin Care (L1)
│   └── Lip Care (L2) ← LANEIGE Lip Sleeping Mask
└── Makeup (L1)
    ├── Lip Makeup (L2) ← 립스틱, 립글로스
    └── Face > Powder (L3)
```

### 기술 스택

| 항목 | 기술 |
|------|------|
| UI Framework | Vanilla JS (no framework) |
| CSS | Custom CSS + AMOREPACIFIC 디자인 |
| Charts | Chart.js 4.4.0 |
| Graph | D3.js v7 (force-directed graph) |
| Icons | Font Awesome 6.5.1 |
| HTTP | Fetch API |

### 디자인 시스템

#### 컬러 팔레트

```css
:root {
    /* Primary */
    --pacific-blue: #001C58;      /* 헤더, 사이드바 */
    --amore-blue: #1F5795;        /* CTA, 링크 */

    /* Secondary */
    --light-gray: #F5F5F5;        /* 배경 */
    --text-secondary: #7D7D7D;    /* 보조 텍스트 */
    --border-gray: #E0E0E0;       /* 구분선 */

    /* Accent */
    --success-green: #4CAF50;     /* 상승 */
    --warning-orange: #FF9800;    /* 경고 */
    --danger-red: #F44336;        /* 하락 */
}
```

#### 타이포그래피

```css
/* 헤딩 */
h1 { font-size: 32px; font-weight: 700; }
h2 { font-size: 24px; font-weight: 600; }
h3 { font-size: 18px; font-weight: 500; }

/* 본문 */
body { font-family: 'Segoe UI', sans-serif; font-size: 14px; }
```

### API 연동

#### 엔드포인트

| Method | Endpoint | 설명 |
|--------|----------|------|
| GET | `/api/data` | 대시보드 데이터 |
| POST | `/api/v3/chat` | AI 챗봇 (권장) |
| GET | `/api/v4/brain/status` | 스케줄러 상태 |
| POST | `/api/crawl/start` | 크롤링 트리거 |

#### 데이터 구조

```javascript
// GET /api/data 응답
{
    "categories": {
        "beauty": {
            "products": [...],
            "metrics": {
                "sos": 45.2,
                "hhi": 0.15,
                "cpi": 0.89
            }
        }
    },
    "latest_insight": {
        "content": "...",
        "generated_at": "2026-02-10T12:00:00"
    }
}
```

#### Fetch 예시

```javascript
// 대시보드 데이터 로드
async function loadDashboardData() {
    const response = await fetch('/api/data');
    const data = await response.json();
    updateCharts(data);
}

// AI 챗봇
async function sendChatMessage(query) {
    const response = await fetch('/api/v3/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query })
    });
    const result = await response.json();
    return result.response;
}
```

### Chart.js 설정

#### KPI 카드 (Doughnut)

```javascript
new Chart(ctx, {
    type: 'doughnut',
    data: {
        labels: ['LANEIGE', 'Others'],
        datasets: [{
            data: [sos, 100 - sos],
            backgroundColor: ['#001C58', '#E0E0E0']
        }]
    }
});
```

#### 순위 추이 (Line)

```javascript
new Chart(ctx, {
    type: 'line',
    data: {
        labels: dates,
        datasets: [{
            label: 'LANEIGE',
            data: ranks,
            borderColor: '#1F5795',
            tension: 0.4
        }]
    }
});
```

### KG Visualizer (D3.js)

#### Force-Directed Graph

```javascript
const simulation = d3.forceSimulation(nodes)
    .force("link", d3.forceLink(links).id(d => d.id))
    .force("charge", d3.forceManyBody().strength(-300))
    .force("center", d3.forceCenter(width / 2, height / 2));
```

#### 노드 타입별 색상

| 노드 타입 | 색상 | 설명 |
|----------|------|------|
| `Product` | `#001C58` | 제품 |
| `Brand` | `#1F5795` | 브랜드 |
| `Category` | `#4CAF50` | 카테고리 |
| `Metric` | `#FF9800` | 지표 |

## test_chat.html

챗봇 기능만 분리한 간단한 테스트 페이지입니다.

### 용도

- 챗봇 API 단독 테스트
- 디버깅 및 응답 시간 측정
- 프롬프트 엔지니어링 실험

### 사용 예시

```bash
# 브라우저에서 열기
open dashboard/test_chat.html
```

## 서버 설정

### FastAPI Static Mount

```python
# dashboard_api.py
from fastapi.staticfiles import StaticFiles

app.mount("/", StaticFiles(directory="dashboard", html=True), name="dashboard")
```

### 로컬 실행

```bash
uvicorn dashboard_api:app --host 0.0.0.0 --port 8001 --reload
```

접속: `http://localhost:8001/amore_unified_dashboard_v4.html`

## 반응형 디자인

### 브레이크포인트

```css
/* Mobile */
@media (max-width: 768px) {
    .sidebar { display: none; }
    .main-content { margin-left: 0; }
}

/* Tablet */
@media (min-width: 769px) and (max-width: 1024px) {
    .sidebar { width: 200px; }
}

/* Desktop */
@media (min-width: 1025px) {
    .sidebar { width: 250px; }
}
```

## 사용자 경험

### 로딩 상태

```javascript
function showLoading() {
    document.getElementById('loading-spinner').style.display = 'block';
}

function hideLoading() {
    document.getElementById('loading-spinner').style.display = 'none';
}
```

### 에러 핸들링

```javascript
try {
    const data = await fetch('/api/data');
} catch (error) {
    showError('데이터를 불러오는데 실패했습니다.');
}
```

### 통화 전환 (USD ↔ KRW)

```javascript
let currentCurrency = 'USD';

function toggleCurrency() {
    currentCurrency = currentCurrency === 'USD' ? 'KRW' : 'USD';
    updateAllPrices();
}
```

## 주의사항

1. **CORS**: FastAPI에서 CORS 설정 필요 (개발 환경)
2. **캐싱**: 브라우저 캐시 문제 시 강력 새로고침 (Cmd+Shift+R)
3. **API Key**: 프론트엔드에 하드코딩 금지
4. **Responsive**: 모바일 테스트 필수
5. **차트 업데이트**: destroy() 후 재생성하여 메모리 릭 방지

## 성능 최적화

### 이미지 Lazy Loading

```html
<img src="placeholder.png" data-src="actual.png" loading="lazy">
```

### 차트 애니메이션 최소화

```javascript
// 빠른 렌더링
options: {
    animation: { duration: 300 }
}
```

### Debounce (검색 입력)

```javascript
let debounceTimer;
input.addEventListener('input', (e) => {
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(() => search(e.target.value), 300);
});
```

## 접근성 (a11y)

```html
<!-- ARIA 라벨 -->
<button aria-label="Close modal">×</button>

<!-- 키보드 내비게이션 -->
<div tabindex="0" role="button" onkeypress="handleEnter">Click me</div>

<!-- Alt 텍스트 -->
<img src="logo.png" alt="AMOREPACIFIC Logo">
```

## 참고

- [Chart.js 문서](https://www.chartjs.org/docs/latest/)
- [D3.js 문서](https://d3js.org/)
- `dashboard_api.py` - FastAPI 서버
- `CLAUDE.md` - 디자인 시스템 컬러 코드
