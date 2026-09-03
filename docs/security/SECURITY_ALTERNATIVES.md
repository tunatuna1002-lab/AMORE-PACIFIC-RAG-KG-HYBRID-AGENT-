# 보안 강화 대안책

> 회사 내부 검토 후 적용 여부 결정 필요

**작성일:** 2026-01-28
**현재 적용:** 속도 제한 강화 (IP당 10회/분)

---

## 현재 상태

| 항목 | 상태 |
|------|------|
| 채팅 API 인증 | 없음 (속도 제한만) |
| 속도 제한 | IP당 10회/분 |
| 네트워크 제한 | 없음 (공개) |

---

## 대안 1: Cloudflare Access (권장)

**개요:** 네트워크 레벨에서 인증 요구. 코드 수정 없음.

**장점:**
- 무료 (50명까지)
- 코드 수정 불필요
- 회사 이메일 도메인으로 제한 가능 (@amorepacific.com)
- 한 번 인증하면 일정 기간 유지

**단점:**
- Cloudflare 의존성
- DNS 설정 필요

**적용 방법:**
1. Cloudflare 계정 생성
2. 도메인 DNS를 Cloudflare로 이전
3. Access > Applications에서 서비스 URL 등록
4. Policy 설정: 이메일 도메인 `@amorepacific.com` 허용

**사용자 경험:**
- 첫 접속 시 이메일 입력 → 인증 코드 수신 → 입력
- 이후 30일간 자동 로그인

**비용:** 무료 (50명 이하)

---

## 대안 2: Google OAuth 로그인

**개요:** 회사 Google Workspace 계정으로 로그인

**장점:**
- 회사 계정만 허용 가능
- 사용자 친숙한 "Google로 로그인" 버튼
- 세션 관리 자동화

**단점:**
- 코드 수정 필요 (대시보드 + API)
- Google Cloud Console 설정 필요

**필요 패키지:**
```bash
pip install python-jose[cryptography] passlib[bcrypt]
```

**적용 방법:**
1. Google Cloud Console에서 OAuth 2.0 클라이언트 생성
2. FastAPI에 OAuth2 엔드포인트 추가
3. 대시보드에 로그인 UI 추가
4. 허용 도메인 설정: `hd=amorepacific.com`

**예상 개발 기간:** 1-2일

---

## 대안 3: IP 화이트리스트

**개요:** 회사 IP 주소만 접근 허용

**장점:**
- 매우 간단
- 코드 수정 최소

**단점:**
- 회사 고정 IP 필요
- 재택근무 시 접근 불가
- VPN 필요할 수 있음

**적용 방법 (FastAPI Middleware):**
```python
ALLOWED_IPS = ["123.45.67.89", "98.76.54.32"]  # 회사 IP

@app.middleware("http")
async def ip_whitelist(request: Request, call_next):
    client_ip = request.client.host
    if client_ip not in ALLOWED_IPS:
        return JSONResponse(status_code=403, content={"detail": "Access denied"})
    return await call_next(request)
```

---

## 대안 4: Railway Private Network

**개요:** Railway 내부 네트워크에서만 접근

**장점:**
- Railway 설정만으로 완료
- 코드 수정 불필요

**단점:**
- Railway 내부 서비스에서만 접근 가능
- 외부 대시보드 접근 불가

**적용 방법:**
1. Railway Dashboard → Settings → Networking
2. "Private Network Only" 활성화

**주의:** 대시보드도 Railway 내부에서 접근해야 함

---

## 대안 5: HTTP Basic Auth (간단)

**개요:** 브라우저 기본 팝업으로 아이디/비밀번호 입력

**장점:**
- 매우 간단한 구현
- 추가 UI 불필요

**단점:**
- 사용자 경험 나쁨
- 세션 관리 없음 (매번 입력)

**적용 방법:**
```python
from fastapi import Depends
from fastapi.security import HTTPBasic, HTTPBasicCredentials
import secrets

security = HTTPBasic()

def verify_credentials(credentials: HTTPBasicCredentials = Depends(security)):
    correct_username = secrets.compare_digest(credentials.username, "admin")
    correct_password = secrets.compare_digest(credentials.password, os.getenv("DASHBOARD_PASSWORD"))
    if not (correct_username and correct_password):
        raise HTTPException(status_code=401, headers={"WWW-Authenticate": "Basic"})
    return credentials.username
```

---

## 비교 표

| 대안 | 보안 수준 | 사용자 경험 | 구현 난이도 | 비용 |
|------|----------|------------|------------|------|
| Cloudflare Access | 높음 | 좋음 | 쉬움 | 무료 |
| Google OAuth | 높음 | 좋음 | 중간 | 무료 |
| IP 화이트리스트 | 중간 | 제한적 | 쉬움 | 무료 |
| Railway Private | 높음 | 제한적 | 쉬움 | 무료 |
| HTTP Basic Auth | 낮음 | 나쁨 | 쉬움 | 무료 |

---

## 권장 순서

1. **단기 (현재):** 속도 제한 강화 ✅ 적용됨
2. **중기:** Cloudflare Access 또는 Google OAuth
3. **장기:** 필요시 추가 보안 레이어

---

## 결정 시 고려사항

- [ ] 재택근무자 접근 필요 여부
- [ ] 회사 고정 IP 존재 여부
- [ ] Google Workspace 사용 여부
- [ ] 도메인 DNS 변경 가능 여부
- [ ] 예상 사용자 수
