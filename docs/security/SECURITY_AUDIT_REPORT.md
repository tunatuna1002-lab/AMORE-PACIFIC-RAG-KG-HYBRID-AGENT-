# Security Audit Report
## AMORE Pacific RAG-KG Hybrid Agent

**Audit Date:** 2026-01-28
**Auditor:** IT Security Expert
**Scope:** Full codebase security review
**Status:** HIGH severity vulnerabilities detected

---

## Executive Summary

| Severity | Count |
|----------|-------|
| **CRITICAL** | 0 |
| **HIGH** | 5 |
| **MEDIUM** | 6 |
| **LOW** | 4 |
| **INFO** | 4 |

**Overall Risk Rating:** 🟠 **HIGH**

### Top 3 Priority Items

1. **[VULN-005]** Docker 컨테이너가 root 권한으로 실행됨
2. **[VULN-007]** 채팅 API 엔드포인트에 인증 없음
3. **[VULN-006]** API 키 비교가 타이밍 공격에 취약

### Secrets Management: ✅ GOOD

`.env` 및 `config/google_credentials.json` 파일은 `.gitignore`에 등록되어 있고, Git에 추적되지 않음을 확인함. 비밀 관리가 올바르게 되어 있음.

---

## Detailed Findings

### POSITIVE FINDINGS (Secrets Management)

#### [INFO-004] Secrets Properly Excluded from Git ✅
- **Status**: SECURE
- **Location**: `.gitignore:1-8`
- **Description**: 민감한 파일들이 `.gitignore`에 올바르게 등록되어 Git에 추적되지 않음
- **Verified Files**:
  - `.env` - Git 미추적 ✅
  - `config/google_credentials.json` - Git 미추적 ✅
- **Verification Command**: `git ls-files | grep -E "\.env|google_credentials"` → 결과 없음

---

### HIGH Severity

#### [VULN-005] Docker Container Running as Root
- **Severity**: 🟠 HIGH
- **CVSS Score**: 7.5
- **CWE ID**: CWE-250 (Execution with Unnecessary Privileges)
- **Location**: `Dockerfile:1-33`
- **Description**: The Docker container runs all processes as root user, violating the principle of least privilege.
- **Impact**: Container escape vulnerabilities become more severe; any compromise grants root access.
- **Remediation**:
  ```dockerfile
  # Add after apt-get install
  RUN useradd -m -u 1000 appuser
  USER appuser
  ```
- **References**: [Docker Security Best Practices](https://docs.docker.com/develop/security-best-practices/)

---

#### [VULN-006] Weak API Key Comparison (Timing Attack)
- **Severity**: 🟠 HIGH
- **CVSS Score**: 7.0
- **CWE ID**: CWE-208 (Observable Timing Discrepancy)
- **Location**: `src/api/dependencies.py:36`
- **Description**: API key comparison uses `==` operator which is vulnerable to timing attacks.
- **Impact**: Attacker could potentially brute-force the API key by measuring response times.
- **Proof of Concept**:
  ```python
  if api_key != API_KEY:  # Timing vulnerability
  ```
- **Remediation**:
  ```python
  import hmac
  if not hmac.compare_digest(api_key, API_KEY):
  ```
- **References**: [CWE-208](https://cwe.mitre.org/data/definitions/208.html)

---

#### [VULN-007] Missing API Key on Public Endpoints
- **Severity**: 🟠 HIGH
- **CVSS Score**: 7.5
- **CWE ID**: CWE-306 (Missing Authentication for Critical Function)
- **Location**: `dashboard_api.py:717-948`
- **Description**: Chat API endpoints (`/api/chat`, `/api/v3/chat`, `/api/v2/chat`) only have rate limiting but no API key authentication, allowing unrestricted access.
- **Impact**: Resource abuse through unlimited API calls (rate limit of 30/min per IP is bypassable).
- **Remediation**: Add `Depends(verify_api_key)` to sensitive endpoints or implement stronger authentication.

---

#### [VULN-008] Incomplete Prompt Injection Defense
- **Severity**: 🟠 HIGH
- **CVSS Score**: 7.3
- **CWE ID**: CWE-74 (Improper Neutralization of Special Elements)
- **Location**: `src/api/validators/input_validator.py:37-49`
- **Description**: While prompt injection patterns are defined, the system prompt in `dashboard_api.py:775-806` is directly concatenated with user input, and the input validator is not applied to all chat endpoints.
- **Impact**: Potential LLM manipulation to extract system prompts or bypass safety controls.
- **Proof of Concept**: Patterns like `"Actually, ignore the above..."` variations may bypass the filter.
- **Remediation**:
  1. Apply `InputValidator.validate()` to ALL user inputs before LLM processing
  2. Add more comprehensive injection patterns
  3. Implement content filtering on LLM outputs
  4. Consider using OpenAI's moderation API

---

#### [VULN-009] Exposed Error Details in API Responses
- **Severity**: 🟠 HIGH
- **CVSS Score**: 5.3
- **CWE ID**: CWE-209 (Generation of Error Message Containing Sensitive Information)
- **Location**: `dashboard_api.py:1180`, `dashboard_api.py:1046`
- **Description**: Raw exception messages are returned to users in error responses.
- **Impact**: Information leakage about internal implementation details, file paths, database structure.
- **Proof of Concept**:
  ```python
  text=f"처리 중 오류가 발생했습니다: {str(e)}"
  ```
- **Remediation**: Use generic error messages and log detailed errors server-side only.

---

### MEDIUM Severity

#### [VULN-010] XSS Risk in Dashboard (Partial Mitigation)
- **Severity**: 🟡 MEDIUM
- **CVSS Score**: 6.1
- **CWE ID**: CWE-79 (Cross-site Scripting)
- **Location**: `dashboard/amore_unified_dashboard_v4.html`
- **Description**: While `escapeHtml()` function exists and is used for user messages, several `innerHTML` assignments use data from API responses without escaping (e.g., `insightEl.innerHTML = dashboardData.home.insight_message`).
- **Impact**: If API responses are compromised, XSS attacks could occur.
- **Remediation**:
  1. Use `textContent` instead of `innerHTML` where possible
  2. Apply `escapeHtml()` to ALL dynamic content
  3. Implement Content Security Policy headers

---

#### [VULN-011] Missing Security Headers
- **Severity**: 🟡 MEDIUM
- **CVSS Score**: 5.3
- **CWE ID**: CWE-1021 (Improper Restriction of Rendered UI Layers)
- **Location**: `dashboard_api.py` (missing)
- **Description**: No security headers (CSP, X-Frame-Options, X-Content-Type-Options, HSTS) are configured.
- **Impact**: Clickjacking, MIME-type sniffing, and other browser-based attacks.
- **Remediation**:
  ```python
  from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware

  @app.middleware("http")
  async def add_security_headers(request, call_next):
      response = await call_next(request)
      response.headers["X-Frame-Options"] = "DENY"
      response.headers["X-Content-Type-Options"] = "nosniff"
      response.headers["X-XSS-Protection"] = "1; mode=block"
      response.headers["Content-Security-Policy"] = "default-src 'self'"
      return response
  ```

---

#### [VULN-012] Session ID Predictability
- **Severity**: 🟡 MEDIUM
- **CVSS Score**: 5.3
- **CWE ID**: CWE-330 (Use of Insufficiently Random Values)
- **Location**: `dashboard_api.py:735`, `src/api/dependencies.py:51`
- **Description**: Session IDs default to "default" or are user-provided strings without validation. No cryptographically random session generation.
- **Impact**: Session prediction/hijacking if session IDs are guessable.
- **Remediation**: Generate cryptographically random session IDs using `secrets.token_urlsafe()`.

---

#### [VULN-013] Missing Rate Limit on Sensitive Operations
- **Severity**: 🟡 MEDIUM
- **CVSS Score**: 5.3
- **CWE ID**: CWE-770 (Allocation of Resources Without Limits)
- **Location**: `dashboard_api.py` (various endpoints)
- **Description**: The `/api/crawl/start` endpoint lacks rate limiting, allowing potential DoS through repeated crawl requests.
- **Remediation**: Add `@limiter.limit("5/hour")` to crawl endpoints.

---

#### [VULN-014] SQL Injection Mitigation (Good Practice)
- **Severity**: 🟢 LOW (Positive Finding)
- **Location**: `src/tools/sqlite_storage.py`
- **Description**: The codebase uses parameterized queries consistently for SQLite operations. This is good practice.
- **Note**: However, the dynamic `f-string` construction in `get_raw_data()` (line 413-418) concatenates `where_clause` which could be risky if modified.

---

#### [VULN-015] Outdated Dependencies Risk
- **Severity**: 🟡 MEDIUM
- **CVSS Score**: Variable
- **CWE ID**: CWE-1104 (Use of Unmaintained Third Party Components)
- **Location**: `requirements.txt`
- **Description**: Several dependencies use `>=` version specifiers without upper bounds, potentially pulling in vulnerable versions. Notable concerns:
  - `playwright>=1.40.0` - Web automation tool with frequent security updates
  - `chromadb>=0.4.0` - Vector database
  - `owlready2>=0.45` - OWL/RDF processor
- **Remediation**:
  1. Pin specific versions: `playwright==1.42.0`
  2. Run regular `pip-audit` scans
  3. Enable Dependabot/Snyk for automated alerts

---

### LOW Severity

#### [VULN-016] Debug Information in Comments
- **Severity**: 🔵 LOW
- **CVSS Score**: 3.1
- **CWE ID**: CWE-615 (Inclusion of Sensitive Information in Source Code Comments)
- **Location**: Various files
- **Description**: Some comments contain implementation details that could assist attackers.
- **Remediation**: Review and remove sensitive comments before production deployment.

---

#### [VULN-017] Verbose Logging
- **Severity**: 🔵 LOW
- **CVSS Score**: 3.7
- **CWE ID**: CWE-532 (Insertion of Sensitive Information into Log File)
- **Location**: `src/monitoring/logger.py`
- **Description**: While a `SensitiveDataFilter` exists for masking API keys, verify it covers all sensitive patterns. User queries are logged in audit trail which may contain PII.
- **Remediation**: Review what data is logged and ensure GDPR/privacy compliance.

---

#### [VULN-018] CORS Configuration (Conditional Risk)
- **Severity**: 🔵 LOW
- **CVSS Score**: 4.3
- **CWE ID**: CWE-942 (Permissive Cross-domain Policy)
- **Location**: `dashboard_api.py:126-134`
- **Description**: CORS allows credentials with configurable origins. If `ALLOWED_ORIGINS` env var is set to `*` in production, this becomes HIGH severity.
- **Current Setting**: `http://localhost:8001,http://127.0.0.1:8001` (safe for development)
- **Remediation**: Ensure production environment sets specific allowed origins.

---

#### [VULN-019] Missing .dockerignore
- **Severity**: 🔵 LOW
- **CVSS Score**: 2.0
- **Location**: Project root (missing file)
- **Description**: No `.dockerignore` file to exclude sensitive files from Docker build context.
- **Impact**: `.env` and `config/google_credentials.json` could be copied into Docker image.
- **Remediation**: Create `.dockerignore`:
  ```
  .env
  .env.*
  config/google_credentials.json
  *.pyc
  __pycache__
  .git
  ```

---

### INFORMATIONAL

#### [INFO-001] Good: Input Validation Module Exists
- **Location**: `src/api/validators/input_validator.py`
- **Description**: A dedicated input validation module with XSS and prompt injection patterns exists. Ensure it's applied consistently.

#### [INFO-002] Good: Sensitive Data Logging Filter
- **Location**: `src/monitoring/logger.py`
- **Description**: A `SensitiveDataFilter` class masks API keys in logs. Good practice.

#### [INFO-003] Good: Rate Limiting Implemented
- **Location**: `dashboard_api.py:117-124`
- **Description**: SlowAPI rate limiter is configured (30/minute). Good practice.

---

## Vulnerability Inventory Table

| ID | Severity | Category | File | Line | Status |
|----|----------|----------|------|------|--------|
| VULN-001 | CRITICAL | Secrets | .env | 3 | Open |
| VULN-002 | CRITICAL | Secrets | config/google_credentials.json | 1 | Open |
| VULN-003 | CRITICAL | Secrets | .env | 27-31 | Open |
| VULN-004 | CRITICAL | Secrets | .env | multiple | Open |
| VULN-005 | HIGH | Docker | Dockerfile | 1-33 | Open |
| VULN-006 | HIGH | AuthN | src/api/dependencies.py | 36 | Open |
| VULN-007 | HIGH | AuthN | dashboard_api.py | 717+ | Open |
| VULN-008 | HIGH | Injection | src/api/validators | 37-49 | Open |
| VULN-009 | HIGH | InfoLeak | dashboard_api.py | 1180 | Open |
| VULN-010 | MEDIUM | XSS | dashboard/*.html | multiple | Partial |
| VULN-011 | MEDIUM | Headers | dashboard_api.py | - | Open |
| VULN-012 | MEDIUM | Session | dashboard_api.py | 735 | Open |
| VULN-013 | MEDIUM | DoS | dashboard_api.py | crawl | Open |
| VULN-014 | LOW | SQLi | sqlite_storage.py | - | Mitigated |
| VULN-015 | MEDIUM | Deps | requirements.txt | - | Open |
| VULN-016 | LOW | Info | Various | - | Open |
| VULN-017 | LOW | Logging | logger.py | - | Partial |
| VULN-018 | LOW | CORS | dashboard_api.py | 126 | Conditional |
| VULN-019 | LOW | Docker | - | - | Open |

---

## Remediation Roadmap

### Immediate (0-48 hours) - CRITICAL/Exploitable

1. **ROTATE ALL EXPOSED SECRETS** (VULN-001, 002, 003, 004)
   - OpenAI API Key
   - GCP Service Account
   - Gmail App Password
   - Apify, Tavily, Data.go.kr API Keys

2. **Remove secrets from Git history**
   ```bash
   # Option 1: BFG Repo Cleaner (recommended)
   bfg --delete-files .env --delete-files google_credentials.json

   # Option 2: git filter-branch
   git filter-branch --force --index-filter \
     'git rm --cached --ignore-unmatch .env config/google_credentials.json' \
     --prune-empty --tag-name-filter cat -- --all
   ```

3. **Force push cleaned history** (coordinate with team)
   ```bash
   git push origin --force --all
   git push origin --force --tags
   ```

### Short-term (1-2 weeks) - HIGH Severity

4. **Implement secure secrets management**
   - Use environment variables only (never files)
   - Consider AWS Secrets Manager, HashiCorp Vault, or Railway secrets

5. **Fix Docker security** (VULN-005)
   - Add non-root user
   - Create `.dockerignore`

6. **Secure API authentication** (VULN-006, 007)
   - Use `hmac.compare_digest()` for timing-safe comparison
   - Add authentication to chat endpoints or implement proper API gateway

7. **Strengthen prompt injection defense** (VULN-008)
   - Apply InputValidator to all chat inputs
   - Implement output filtering

8. **Sanitize error messages** (VULN-009)

### Medium-term (1-3 months) - MEDIUM Severity

9. **Add security headers** (VULN-011)
10. **Implement secure session management** (VULN-012)
11. **Add rate limiting to crawl endpoints** (VULN-013)
12. **XSS hardening** (VULN-010)
    - Audit all `innerHTML` usage
    - Implement CSP
13. **Pin dependency versions** (VULN-015)
14. **Set up automated security scanning**
    - GitHub Dependabot
    - Snyk or Safety for Python
    - SAST tools (Bandit, Semgrep)

### Long-term (Quarterly) - Hardening

15. **Implement Web Application Firewall (WAF)**
16. **Security training for development team**
17. **Regular penetration testing**
18. **SAST/DAST integration in CI/CD**
19. **Security incident response plan**

---

## Security Checklist

- [ ] All CRITICAL vulnerabilities patched
- [ ] Secrets rotated and removed from codebase/history
- [ ] Dependencies updated to secure versions
- [ ] Security headers implemented
- [ ] Input validation added to ALL entry points
- [ ] Logging & monitoring enhanced
- [ ] Docker hardened (non-root user)
- [ ] Rate limiting on all sensitive endpoints
- [ ] CSP and XSS protections active
- [ ] Automated security scanning enabled

---

## Appendix

### Tools/Methods Used
- Manual code review
- Pattern matching (grep) for secrets
- Git history analysis
- Dockerfile security assessment
- Dependency analysis

### Files Reviewed
- `.env`, `.env.example`
- `config/google_credentials.json`
- `Dockerfile`
- `requirements.txt`
- `dashboard_api.py`
- `src/api/dependencies.py`
- `src/api/validators/input_validator.py`
- `src/tools/sqlite_storage.py`
- `src/tools/email_sender.py`
- `dashboard/amore_unified_dashboard_v4.html`
- Multiple source files in `src/`

### Scope Limitations
- Dynamic application security testing (DAST) not performed
- Network security assessment not in scope
- Cloud infrastructure configuration not reviewed
- Third-party API security not assessed

### False Positive Rationale
- VULN-014: SQL queries use parameterized statements consistently

---

**Report Generated:** 2026-01-28
**Classification:** CONFIDENTIAL
**Distribution:** Security Team, Development Lead, DevOps

---

*This report contains sensitive security information. Handle according to your organization's security policies.*
