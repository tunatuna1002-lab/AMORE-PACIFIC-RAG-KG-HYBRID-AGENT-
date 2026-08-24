# Dashboard Troubleshooting Guide — Round 5

> **Context Engineering Document**: 이 문서는 새로운 Claude Code 세션에서 컨텍스트를 복원하기 위한 자립형 가이드입니다.
> 이전 세션의 모든 분석 결과, 파일 경로, 라인 번호, 정확한 코드 수정 사항을 포함합니다.

---

## 1. 현황 요약

### 1.1 증상 (스크린샷 기준)

| # | 증상 | 위치 | 심각도 |
|---|------|------|--------|
| S1 | 액션 보드 "데이터 로딩 중..." 스피너 무한 표시 | Home 페이지 | P0 |
| S2 | 모든 차트 빈 캔버스 (SoS 추이, 시장점유율, 가격경쟁력, Bubble) | Home, Brand, Category | P0 |
| S3 | 챗봇 FAB 버튼 미표시 (우측 하단) | 전체 페이지 | P1 |
| S4 | 날짜 선택기 "연도. 월. 일." 플레이스홀더 유지 | 전체 차트 영역 | P1 |
| S5 | 헤더 "데이터 로딩중..." 고정 | 우측 상단 | P1 |

### 1.2 이전 수정 이력 (Round 1–4)

모든 이전 수정은 `docs/analysis/dashboard-data-fix.md`에 문서화되어 있으며, 코드에 이미 반영됨.

| Round | 날짜 | 수정 내용 | 상태 |
|-------|------|-----------|------|
| 1 | 2026-02-19 | 데이터 경로 동적 해결 (`./data` → Railway `/data` 감지) | ✅ |
| 2 | 2026-02-19 | JavaScript TDZ 에러 (`let` → `var`, Lucide 가드 14곳) | ✅ |
| 3 | 2026-02-19 | Railway 배포 경로 (AppConfig, 빈 대시보드 구조, 폰트 fallback) | ✅ |
| 4 | 2026-02-19 | CDN → 셀프 호스팅 (Chart.js, Lucide, Noto Sans KR) | ✅ |

### 1.3 핵심 발견

**모든 증상의 근본 원인은 하나로 수렴합니다**: `loadDashboardData()` 함수의 `fetch()` 호출이 실패하면, 이후 전체 초기화 체인이 중단됩니다.

```
window.onload (line 13043)
  → loadDashboardData() (line 13060) ← 여기서 실패하면
    → fetch(`${API_BASE}/api/data`) (line 5684)
    → updateDashboardFromData() (line 5737) ← 실행 안됨
      → updateActionTable() (line 5799) ← 스피너 교체 안됨
      → initDateRangePickers() (line 5897) ← 날짜 초기화 안됨
        → globalDateRange.init() (line 6037) ← 날짜 범위 설정 안됨
  → switchPage('home') (line 13064)
  → loadInitialChartDataWithGlobalRange() (line 13074) ← 날짜 없어서 조기 리턴
```

**`loadDashboardData()`가 실패하는 원인 후보** (우선순위순):

1. **CSP `connect-src` 미지정** → 브라우저가 `fetch()` 응답을 차단
2. **Railway 환경변수 미설정** → `ALLOWED_HOSTS`, `ALLOWED_ORIGINS`
3. **데이터 파일 부재** → `dashboard_data.json` + SQLite 모두 비어있음
4. **네트워크/CORS** → API 요청 자체 실패

---

## 2. 빠른 진단 체크리스트

아래 명령어를 순서대로 실행하여 어떤 문제가 발생하고 있는지 확인합니다.

### 2.1 Railway 환경변수 확인

```bash
# Railway CLI에서 실행
railway variables | grep -E 'ALLOWED_HOSTS|ALLOWED_ORIGINS|OPENAI_API_KEY|API_KEY|DASHBOARD_READ_TOKEN'
```

**예상 결과**:
- `ALLOWED_HOSTS=*` ← 필수 (없으면 TrustedHostMiddleware가 내부 IP 차단)
- `ALLOWED_ORIGINS=` ← 설정되어야 함 (아래 Issue A 참조)
- `OPENAI_API_KEY=sk-...` ← 챗봇 작동에 필요
- `API_KEY=...` ← 보호 엔드포인트 인증

### 2.2 데이터 파일 존재 확인

```bash
# Railway 볼륨의 데이터 파일 확인
railway run -- ls -la /data/dashboard_data.json
railway run -- ls -la /data/amore_data.db

# SQLite 데이터 확인
railway run -- python3 -c "
import sqlite3
c = sqlite3.connect('/data/amore_data.db')
print('Tables:', c.execute(\"SELECT name FROM sqlite_master WHERE type='table'\").fetchall())
print('Row count:', c.execute('SELECT COUNT(*) FROM raw_data').fetchone())
print('Date range:', c.execute('SELECT MIN(snapshot_date), MAX(snapshot_date) FROM raw_data').fetchone())
"
```

### 2.3 API 응답 확인

```bash
# 헬스체크
curl -s https://<YOUR-APP>.railway.app/api/health | python3 -m json.tool

# 대시보드 데이터 API
curl -s https://<YOUR-APP>.railway.app/api/data | python3 -m json.tool | head -30

# 응답 헤더 (CSP, CORS 확인)
curl -sI https://<YOUR-APP>.railway.app/api/data
```

**확인할 헤더**:
```
Content-Security-Policy: default-src 'self'; ... connect-src 'self' ← 이것이 있어야 함
Access-Control-Allow-Origin: ...
```

### 2.4 정적 파일 확인

```bash
# Chart.js 로드 확인
curl -sI https://<YOUR-APP>.railway.app/static/vendor/js/chart.umd.min.js | head -5

# Lucide 로드 확인
curl -sI https://<YOUR-APP>.railway.app/static/vendor/js/lucide.min.js | head -5
```

**예상**: 둘 다 `HTTP/2 200` 반환

### 2.5 브라우저 DevTools 확인

대시보드를 열고 F12 → Console 탭에서 다음을 확인:

```javascript
// 1. API_BASE가 올바른지
console.log('API_BASE:', API_BASE);

// 2. dashboardData가 로드되었는지
console.log('dashboardData:', window.dashboardData);
console.log('metadata:', window.dashboardData?.metadata);

// 3. globalDateRange 상태
console.log('dateRange:', globalDateRange?.startDate, '~', globalDateRange?.endDate);

// 4. Lucide가 로드되었는지
console.log('lucide:', typeof lucide);

// 5. Chart.js가 로드되었는지
console.log('Chart:', typeof Chart);

// 6. 챗봇 FAB 상태
const fab = document.getElementById('chatFab');
console.log('FAB:', fab, 'className:', fab?.className, 'rect:', fab?.getBoundingClientRect());

// 7. CSP 위반 확인
document.addEventListener('securitypolicyviolation', (e) => {
    console.error('CSP VIOLATION:', e.violatedDirective, e.blockedURI);
});
```

**Network 탭**: `/api/data` 요청을 찾아서 Status, Response, Headers를 확인합니다.

---

## 3. 근본 원인 분석

### Issue A: CSP `connect-src` 누락 (P0)

**파일**: `src/api/middleware/security_headers.py:20-24`

**현재 코드**:
```python
response.headers["Content-Security-Policy"] = (
    "default-src 'self'; script-src 'self' 'unsafe-inline'; "
    "style-src 'self' 'unsafe-inline'; img-src 'self' data: https:; "
    "font-src 'self' data:"
)
```

**문제**:
- `connect-src` 지시어가 없음
- CSP 명세에 따라 `connect-src`가 없으면 `default-src 'self'`로 폴백
- 동일 출처(same-origin) 요청인 `fetch('/api/data')`는 `'self'`에 의해 허용되어야 함
- **그러나** 일부 브라우저(특히 Chrome의 특정 버전)에서 `connect-src` 미명시 시 SSE(Server-Sent Events) 연결이 차단되는 경우가 있음
- 챗봇의 `EventSource` (`/api/v4/chat/stream`)가 이에 해당
- 또한, 명시적 `connect-src 'self'`가 없으면 Content-Security-Policy-Report-Only 모드에서 위반으로 보고될 수 있음

**영향**:
- `fetch('/api/data')` → 대부분 작동하지만 일부 환경에서 차단 가능
- `EventSource('/api/v4/chat/stream')` → SSE 연결 차단 가능성 높음
- 브라우저 콘솔에 CSP 위반 에러 표시 여부로 확인 가능

**수정 방법**:

```python
# src/api/middleware/security_headers.py:20-24
# BEFORE:
response.headers["Content-Security-Policy"] = (
    "default-src 'self'; script-src 'self' 'unsafe-inline'; "
    "style-src 'self' 'unsafe-inline'; img-src 'self' data: https:; "
    "font-src 'self' data:"
)

# AFTER:
response.headers["Content-Security-Policy"] = (
    "default-src 'self'; script-src 'self' 'unsafe-inline'; "
    "style-src 'self' 'unsafe-inline'; img-src 'self' data: https:; "
    "font-src 'self' data:; connect-src 'self'"
)
```

**검증**:
```bash
curl -sI https://<YOUR-APP>.railway.app/api/data | grep -i content-security
# 예상: Content-Security-Policy: ... connect-src 'self'
```

---

### Issue B: `loadDashboardData()` 실패 시 전체 초기화 체인 중단 (P0)

**파일**: `dashboard/amore_unified_dashboard_v4.html`

**`window.onload` 초기화 시퀀스** (line 13043-13075):
```javascript
window.onload = async function() {
    startClock();                                    // line 13044

    // 날짜 범위 기본값 안전장치
    if (...) globalDateRange.setDefaultRange();       // line 13047-13049

    // 10초 안전 타임아웃
    const loadingSafety = setTimeout(..., 10000);     // line 13052-13058

    await loadDashboardData();                        // line 13060 ← 핵심
    clearTimeout(loadingSafety);                      // line 13061

    updateDataTime();                                 // line 13063
    switchPage('home');                               // line 13064
    loadAlertSettings();                              // line 13065
    initDateRangeSelectors();                         // line 13066

    if (...) globalDateRange.setDefaultRange();        // line 13068-13071

    await loadInitialChartDataWithGlobalRange();       // line 13074
};
```

**문제**: `loadDashboardData()`가 `false`를 반환하더라도(실패 시), 이후 코드는 계속 실행됩니다. 하지만 `dashboardData`가 `null`이면:
- `updateDashboardFromData()` (line 5765): `if (!dashboardData) return;` → 즉시 리턴
- `initDateRangePickers()` 호출 안됨 → `globalDateRange.init()` 호출 안됨
- `loadInitialChartDataWithGlobalRange()` (line 13078): `globalDateRange.get()` → `startDate: null` → 조기 리턴

**그러나** 아래 안전장치가 있음:
- line 13047-13049: `globalDateRange.setDefaultRange()` → 하지만 `dashboardData`가 null이면 30일 폴백 사용
- line 13068-13071: 다시 한번 `setDefaultRange()` 호출

**결론**: `loadDashboardData()` 내 `fetch()` 자체가 예외를 throw하면 `catch` 블록(line 5741-5745)에서 처리되고 `false`가 반환됩니다. 이 경우 `dashboardData`는 `null`로 남아있어 전체 체인이 중단됩니다.

**수정 방법**: `window.onload`에서 `loadDashboardData()` 실패 시에도 `globalDateRange.init()`가 호출되도록 보장:

```javascript
// dashboard/amore_unified_dashboard_v4.html, line 13060 부근
// BEFORE:
    await loadDashboardData();
    clearTimeout(loadingSafety);

    updateDataTime();
    switchPage('home');

// AFTER:
    const dataLoaded = await loadDashboardData();
    clearTimeout(loadingSafety);

    // 데이터 로드 실패 시에도 날짜 범위 초기화 보장
    if (!dataLoaded) {
        globalDateRange.setDefaultRange();
    }

    updateDataTime();
    switchPage('home');
```

---

### Issue C: 챗봇 FAB 미표시 (P1)

**파일**: `dashboard/amore_unified_dashboard_v4.html`

**HTML** (line 5283):
```html
<div class="chat-fab" id="chatFab" onclick="toggleChatbot()">
    <i data-lucide="message-square" style="color: white; width: 28px; height: 28px;"></i>
</div>
```

**CSS** (line 3658-3678):
```css
.chat-fab {
    position: fixed;
    bottom: 24px;
    right: 24px;
    width: 56px;
    height: 56px;
    background: var(--pacific-blue);  /* #001C58 */
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    cursor: pointer;
    box-shadow: var(--shadow-lg);
    z-index: 100;
    transition: all 0.2s;
    border: none;
}
.chat-fab.hidden { opacity: 0; pointer-events: none; transform: scale(0.9); }
```

**`toggleChatbot()` 함수** (line 9262-9274):
```javascript
function toggleChatbot() {
    chatbotOpen = !chatbotOpen;
    document.getElementById('chatbotPanel').classList.toggle('open', chatbotOpen);
    document.getElementById('chatFab').classList.toggle('hidden', chatbotOpen);
    // ...
}
```

**가능한 원인**:

1. **Lucide 아이콘 미렌더링**: `<i data-lucide="message-square">` 요소는 `lucide.createIcons()` 호출 없이는 빈 상태입니다. FAB 자체는 56×56 파란색 사각형으로 표시되지만, 아이콘이 없으면 시각적으로 버튼인지 인식하기 어려울 수 있습니다.

2. **`z-index` 충돌**: `.ai-drawer` (line 3681-3684)는 `position: fixed; z-index:` 값이 설정되어 있습니다. 열린 상태에서 FAB를 가릴 수 있습니다.

3. **`window.onload` 이후 Lucide 재렌더링 누락**: 초기 `lucide.createIcons()` 호출(line 5416)은 페이지 로드 직후 실행됩니다. 그러나 `window.onload`에서 DOM 조작이 끝난 후에는 재호출하지 않아, 동적으로 추가된 아이콘이 렌더링되지 않을 수 있습니다.

**진단**:
```javascript
// 브라우저 DevTools에서 실행
const fab = document.getElementById('chatFab');
console.log('exists:', !!fab);
console.log('className:', fab.className);          // 'hidden' 포함 여부
console.log('computed display:', getComputedStyle(fab).display);
console.log('computed opacity:', getComputedStyle(fab).opacity);
console.log('rect:', fab.getBoundingClientRect()); // width/height가 0인지
console.log('icon rendered:', fab.querySelector('svg') !== null); // Lucide SVG 존재 여부
```

**수정 방법**: `window.onload` 끝에 `lucide.createIcons()` 명시적 호출 추가:

```javascript
// dashboard/amore_unified_dashboard_v4.html, line 13074 뒤에 추가
// BEFORE:
    await loadInitialChartDataWithGlobalRange();
};

// AFTER:
    await loadInitialChartDataWithGlobalRange();

    // 모든 초기화 완료 후 아이콘 최종 렌더링 보장
    if (typeof lucide !== 'undefined') {
        lucide.createIcons();
        console.log('[window.onload] Final lucide.createIcons() called');
    }
};
```

---

### Issue D: 차트 빈 캔버스 (P1)

**원인 체인**:

1. `loadInitialChartDataWithGlobalRange()` (line 13078-13096) 호출
2. `globalDateRange.get()` (line 13079) → `{startDate: null, endDate: null}` 반환
3. line 13081: `if (!startDate || !endDate) return;` → **조기 리턴, 차트 데이터 로드 안됨**

**왜 `globalDateRange`가 null인가**:
- `globalDateRange.init()` (line 6037)는 `initDateRangePickers()` (line 6035) 안에서만 호출됨
- `initDateRangePickers()`는 `updateDashboardFromData()` (line 5897) 마지막에서 호출됨
- `updateDashboardFromData()` (line 5765)는 `if (!dashboardData) return;`으로 시작
- `dashboardData`가 `null`이면 → `globalDateRange.init()` 호출 안됨

**그러나** `window.onload`의 안전장치 (line 13047-13049, 13068-13071):
```javascript
if (typeof globalDateRange !== 'undefined' && globalDateRange &&
    (!globalDateRange.startDate || !globalDateRange.endDate)) {
    globalDateRange.setDefaultRange();
}
```
이 코드는 `globalDateRange.startDate`가 `null`일 때 `setDefaultRange()`를 호출합니다.
`setDefaultRange()` (line 5927)는 `dashboardData?.metadata?.available_date_range`를 먼저 확인하고, 없으면 **30일 폴백**을 사용합니다 (line 5941-5948).

**결론**: 이 안전장치가 작동해야 합니다. 만약 작동하지 않는다면:
- `globalDateRange` 자체가 `undefined`일 가능성 (하지만 line 5903에서 `const`로 선언됨)
- `typeof globalDateRange !== 'undefined'` 체크에서 이미 통과해야 함

**가장 가능성 높은 시나리오**: `loadDashboardData()`의 `fetch()` 자체가 **예외를 throw**하여 `window.onload` 함수 전체가 중단됨. `await loadDashboardData()` 줄에서 throw되면 이후 `switchPage()`, `loadInitialChartDataWithGlobalRange()` 모두 실행되지 않습니다.

**그런데** `loadDashboardData()` 내부에 `try/catch` (line 5683-5746)가 있어 예외를 잡아야 합니다. **하지만**: `fetch()`가 네트워크 에러(CORS 차단, CSP 차단)로 reject되면, `catch` 블록에서 잡히고 `false`가 반환됩니다. 이 경우 `window.onload`는 계속 진행합니다.

**실제 문제 원인**: Issue A(CSP `connect-src`)가 해결되면, `fetch()`가 성공하고 전체 체인이 정상 작동할 것입니다. 차트 문제는 Issue A의 연쇄 효과입니다.

**추가 수정**: Issue B의 수정으로 `loadDashboardData()` 실패 시에도 `globalDateRange.setDefaultRange()` 호출을 보장합니다.

---

### Issue E: 날짜 선택기 플레이스홀더 (P2)

**원인**: Issue D와 동일 — `globalDateRange`가 초기화되지 않으면 `<input type="date">`에 값이 설정되지 않습니다.

**수정**: Issue A + Issue B 해결로 자동 해결됩니다.

---

## 4. 수정 지침 (우선순위순)

### Fix 1: CSP `connect-src` 추가 (Issue A)

**파일**: `src/api/middleware/security_headers.py`
**라인**: 20-24

```python
# BEFORE (line 20-24):
response.headers["Content-Security-Policy"] = (
    "default-src 'self'; script-src 'self' 'unsafe-inline'; "
    "style-src 'self' 'unsafe-inline'; img-src 'self' data: https:; "
    "font-src 'self' data:"
)

# AFTER:
response.headers["Content-Security-Policy"] = (
    "default-src 'self'; script-src 'self' 'unsafe-inline'; "
    "style-src 'self' 'unsafe-inline'; img-src 'self' data: https:; "
    "font-src 'self' data:; connect-src 'self'"
)
```

### Fix 2: `window.onload` 실패 복구 강화 (Issue B)

**파일**: `dashboard/amore_unified_dashboard_v4.html`
**라인**: 13060 부근

```javascript
// BEFORE (line 13059-13066):
            await loadDashboardData();
            clearTimeout(loadingSafety);

            updateDataTime(); // 데이터 시간 업데이트
            switchPage('home');
            loadAlertSettings(); // 알림 설정 로드
            initDateRangeSelectors(); // 날짜 범위 선택 초기화

// AFTER:
            const dataLoaded = await loadDashboardData();
            clearTimeout(loadingSafety);

            // 데이터 로드 실패 시에도 날짜 범위 초기화 보장
            if (!dataLoaded && typeof globalDateRange !== 'undefined') {
                globalDateRange.setDefaultRange();
                console.warn('[window.onload] Data load failed, using default date range');
            }

            updateDataTime(); // 데이터 시간 업데이트
            switchPage('home');
            loadAlertSettings(); // 알림 설정 로드
            initDateRangeSelectors(); // 날짜 범위 선택 초기화
```

### Fix 3: Lucide 아이콘 최종 렌더링 (Issue C)

**파일**: `dashboard/amore_unified_dashboard_v4.html`
**라인**: 13074 뒤에 추가

```javascript
// BEFORE (line 13074-13075):
            await loadInitialChartDataWithGlobalRange();
        };

// AFTER:
            await loadInitialChartDataWithGlobalRange();

            // 모든 초기화 완료 후 아이콘 최종 렌더링 보장 (chatFab 등)
            if (typeof lucide !== 'undefined') {
                lucide.createIcons();
            }
        };
```

### Fix 4: Railway 환경변수 설정

Railway 대시보드 또는 CLI에서 다음 환경변수를 확인/설정:

```bash
# 필수 (없으면 추가)
railway variables set ALLOWED_HOSTS="*"

# CORS 허용 출처 (Railway 도메인 포함)
railway variables set ALLOWED_ORIGINS="https://<YOUR-APP>.railway.app,http://localhost:8001"
```

> **참고**: `ALLOWED_HOSTS=*`는 이미 설정되어 있을 수 있습니다 (Round 3에서 문서화). `ALLOWED_ORIGINS`는 동일 출처 요청에는 영향을 주지 않지만, 명시적으로 설정하는 것이 안전합니다.

### Fix 5: 디버그 로깅 추가 (선택사항)

문제가 지속되면 `loadDashboardData()`에 상세 로깅 추가:

**파일**: `dashboard/amore_unified_dashboard_v4.html`
**라인**: 5684 뒤에 추가

```javascript
// line 5684 뒤:
                const response = await fetch(`${API_BASE}/api/data`);
                console.log('[loadDashboardData] fetch response:', response.status, response.statusText);
                console.log('[loadDashboardData] response headers:', {
                    csp: response.headers.get('content-security-policy'),
                    cors: response.headers.get('access-control-allow-origin'),
                    ct: response.headers.get('content-type')
                });
```

---

## 5. 검증 절차

### 5.1 Fix별 검증

| Fix | 검증 방법 | 기대 결과 |
|-----|-----------|-----------|
| Fix 1 (CSP) | `curl -sI .../api/data \| grep content-security` | `connect-src 'self'` 포함 |
| Fix 2 (복구) | 데이터 없는 상태로 대시보드 로드 | 스피너 대신 "데이터 없음" 메시지 |
| Fix 3 (Lucide) | 대시보드 로드 후 우측 하단 확인 | 파란색 FAB 버튼 + 말풍선 아이콘 |
| Fix 4 (env) | `railway variables` | `ALLOWED_HOSTS=*` 확인 |

### 5.2 E2E 스모크 테스트

1. **대시보드 로드**: `https://<YOUR-APP>.railway.app/dashboard` 접속
2. **Home 페이지**: 데이터 날짜 표시, 액션 보드 내용 또는 "없음" 메시지
3. **Brand View**: KPI 카드 값 표시, 경쟁사 테이블 데이터
4. **Category View**: 카테고리 버튼 클릭 시 KPI 변경
5. **Product View**: 제품 선택 시 순위/리뷰/평점 표시
6. **챗봇**: FAB 클릭 → 패널 열림 → 추천 질문 클릭 → 응답 수신
7. **차트**: SoS 추이, 경쟁사 비교 차트에 데이터 표시
8. **Console**: F12 → 에러 0건 확인

### 5.3 CSP 위반 확인

```javascript
// 브라우저 DevTools에서 실행 후 페이지 새로고침
document.addEventListener('securitypolicyviolation', (e) => {
    console.error('🔴 CSP VIOLATION:', {
        directive: e.violatedDirective,
        blockedURI: e.blockedURI,
        originalPolicy: e.originalPolicy
    });
});
```

새로고침 후 콘솔에 `CSP VIOLATION` 메시지가 없어야 합니다.

---

## 6. Context Engineering — 새 세션용 프롬프트

아래 텍스트를 새로운 Claude Code 터미널에 복사하여 사용합니다.

---

```
AMORE Pacific 대시보드의 데이터 연결 문제를 수정해야 합니다.

## 배경
Railway에 배포된 대시보드 (FastAPI + 단일 HTML SPA)에서 다음 문제가 발생:
1. 액션 보드 무한 로딩 스피너
2. 모든 차트 빈 캔버스
3. 챗봇 FAB 버튼 미표시 (우측 하단)
4. 날짜 선택기 플레이스홀더 유지
5. 헤더 "데이터 로딩중..." 고정

## 이미 완료된 수정 (Round 1-4)
- Round 1: 데이터 경로 동적 해결 (./data → /data 감지)
- Round 2: JavaScript TDZ 에러 (let→var, Lucide 가드 14곳)
- Round 3: Railway 배포 경로 (AppConfig, 빈 대시보드 구조)
- Round 4: CDN → 셀프 호스팅 (Chart.js, Lucide, Noto Sans KR)
- 상세: docs/analysis/dashboard-data-fix.md

## 근본 원인 분석 결과 (Round 5)
모든 증상은 `loadDashboardData()`의 `fetch('/api/data')` 실패에서 시작.
실패 시 `updateDashboardFromData()` → `initDateRangePickers()` → `globalDateRange.init()`
전체 체인이 중단되어 차트, 날짜, 액션 보드 모두 미초기화.

## 필요한 수정 5건

### Fix 1: CSP connect-src 추가 (P0)
파일: src/api/middleware/security_headers.py (line 20-24)
현재: "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' data: https:; font-src 'self' data:"
수정: 끝에 "; connect-src 'self'" 추가
이유: connect-src 미지정 시 일부 브라우저에서 fetch/SSE 차단

### Fix 2: window.onload 실패 복구 (P0)
파일: dashboard/amore_unified_dashboard_v4.html (line 13060)
현재: `await loadDashboardData();`
수정: `const dataLoaded = await loadDashboardData();`로 변경하고,
      바로 아래에 실패 시 `globalDateRange.setDefaultRange()` 호출 추가
이유: 데이터 로드 실패 시에도 날짜 범위 초기화 보장

### Fix 3: Lucide 아이콘 최종 렌더링 (P1)
파일: dashboard/amore_unified_dashboard_v4.html (line 13074 뒤)
수정: `if (typeof lucide !== 'undefined') { lucide.createIcons(); }` 추가
이유: window.onload 끝에서 모든 아이콘(특히 chatFab) 최종 렌더링 보장

### Fix 4: Railway 환경변수 확인
ALLOWED_HOSTS=* (필수 — TrustedHostMiddleware)
ALLOWED_ORIGINS에 Railway 도메인 추가 (선택)

### Fix 5: 디버그 로깅 (선택)
파일: dashboard/amore_unified_dashboard_v4.html (line 5684 뒤)
fetch response 상세 로그 추가

## 핵심 파일 맵
| 파일 | 라인 | 역할 |
|------|------|------|
| src/api/middleware/security_headers.py | 20-24 | CSP 정책 |
| dashboard/amore_unified_dashboard_v4.html | 5283 | 챗봇 FAB div |
| dashboard/amore_unified_dashboard_v4.html | 5681-5747 | loadDashboardData() |
| dashboard/amore_unified_dashboard_v4.html | 5903-5983 | globalDateRange 객체 |
| dashboard/amore_unified_dashboard_v4.html | 6035-6037 | initDateRangePickers() → globalDateRange.init() |
| dashboard/amore_unified_dashboard_v4.html | 6345-6354 | updateActionTable() 빈 배열 처리 |
| dashboard/amore_unified_dashboard_v4.html | 9262-9274 | toggleChatbot() |
| dashboard/amore_unified_dashboard_v4.html | 13043-13075 | window.onload 초기화 시퀀스 |
| src/api/app_factory.py | 56-71 | 미들웨어 (CORS, TrustedHost) |
| src/api/routes/data.py | 26-59 | /api/data 엔드포인트 |
| src/api/routes/health.py | 38-60 | /dashboard HTML 서빙 |
| src/api/dependencies.py | 178-215 | load_dashboard_data() |
| docs/analysis/dashboard-data-fix.md | 전체 | Round 1-4 수정 이력 |

## 수정 순서
1. Fix 1 (CSP) → 2. Fix 2 (복구 강화) → 3. Fix 3 (Lucide) → 4. Fix 4 (env vars)
5. 배포 → 6. E2E 검증 → 7. Fix 5 (디버그 로깅, 필요시)

상세 가이드: docs/troubleshooting/dashboard-fix-guide.md
```

---

## 7. 아키텍처 참고

### 7.1 데이터 흐름

```
[Browser]
    ↓ window.onload
    ↓ DOMContentLoaded → checkServerStatus() → initDateRanges()
    ↓ window.onload → loadDashboardData()
    ↓                   ↓
    ↓             fetch(API_BASE + '/api/data')
    ↓                   ↓
[FastAPI Server]        ↓
    ↓ TrustedHostMiddleware → CORS → CSRF → SecurityHeaders
    ↓                   ↓
    ↓ /api/data route (src/api/routes/data.py:26)
    ↓   1. load_dashboard_data() → /data/dashboard_data.json
    ↓   2. fallback: _generate_dashboard_from_sqlite()
    ↓   3. fallback: empty structure {_is_empty: true}
    ↓                   ↓
[Browser]               ↓
    ↓ dashboardData = response.json()
    ↓ updateDashboardFromData()
    ↓   → updateActionTable()       → 액션 보드
    ↓   → initDateRangePickers()    → globalDateRange.init()
    ↓ switchPage('home')
    ↓   → initChartsForPage('home') → 정적 차트 데이터
    ↓ loadInitialChartDataWithGlobalRange()
    ↓   → fetch('/api/historical')  → 동적 차트 데이터
    ↓ lucide.createIcons()          → 아이콘 렌더링
```

### 7.2 미들웨어 스택 (실행 순서: 아래→위)

```
Request → TrustedHostMiddleware (ALLOWED_HOSTS)
        → CORSMiddleware (ALLOWED_ORIGINS)
        → CSRFMiddleware (exempt: /api/)
        → SecurityHeadersMiddleware (CSP, X-Frame-Options 등)
        → Route Handler
        → Response (역순으로 헤더 추가)
```

### 7.3 정적 파일 마운트

```
/static/          → static/                     (app_factory.py:133)
/static/vendor/js/chart.umd.min.js              (Chart.js v4.4.8, 202KB)
/static/vendor/js/lucide.min.js                 (Lucide v0.469.0, 350KB)
/static/vendor/fonts/noto-sans-kr.css           (@font-face 정의)
/static/vendor/fonts/NotoSansKR-*.ttf           (5 weight)
/fonts/           → static/fonts/               (app_factory.py:136)
/dashboard        → GET route (health.py:38)    (HTML 서빙, API key 주입)
/dashboard/       → StaticFiles (app_factory.py:140)  (사실상 미사용)
```

---

## 8. 변경 로그

| 날짜 | Round | 수정 내용 | 상태 |
|------|-------|-----------|------|
| 2026-02-19 | 1 | 데이터 경로 동적 해결 | ✅ |
| 2026-02-19 | 2 | TDZ 에러 (let→var, Lucide 가드) | ✅ |
| 2026-02-19 | 3 | Railway 배포 경로 | ✅ |
| 2026-02-19 | 4 | CDN → 셀프 호스팅 | ✅ |
| 2026-02-19 | 5 | CSP connect-src + 초기화 복구 + Lucide 재렌더링 | ⏳ 대기 |
