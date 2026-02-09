# src/api - API Layer

## OVERVIEW

FastAPI routes with versioned endpoints, API key auth, rate limiting, and middleware.

## ROUTE MODULES

| Module | Prefix | Key Endpoints |
|--------|--------|---------------|
| chat.py | `/chat`, `/v2/chat`, `/v3/chat`, `/v4/chat` | AI chatbot (versioned) |
| brain.py | `/api/v4/brain` | Scheduler control, status, mode |
| crawl.py | `/api/crawl` | Crawl status, start |
| data.py | `/api/data`, `/api/historical` | Dashboard data |
| deals.py | `/api/deals` | Deals list, summary, scrape |
| export.py | `/api/export` | DOCX, Excel, analyst reports |
| signals.py | `/api/signals` | External signals (RSS, Reddit, Tavily) |
| alerts.py | `/api/alerts`, `/api/v3/alert-settings` | Alert management |

## VERSIONING

| Endpoint Type | Version | Path |
|---------------|---------|------|
| Chat | v1-v4 | `/api/chat`, `/api/v2/chat`, `/api/v3/chat`, `/api/v4/chat` |
| Brain | v4 | `/api/v4/brain/*` |
| Alerts Settings | v3 | `/api/v3/alert-settings` |
| Others | unversioned | `/api/data`, `/api/export`, etc. |

## AUTH PATTERNS

| Location | Method | Protected Endpoints |
|----------|--------|---------------------|
| dashboard_api.py | `hmac.compare_digest` | Chat, crawl start |
| dependencies.py | Simple equality | Brain scheduler, deals scrape, alerts save |

Header: `X-API-Key`

## MIDDLEWARE

- **CORS**: Configurable `ALLOWED_ORIGINS`
- **Security Headers**: X-Content-Type-Options, X-Frame-Options, X-XSS-Protection
- **Rate Limiting**: SlowAPI with per-endpoint limits on chat

## WARNINGS

⚠️ **dashboard_api.py** has inline endpoints duplicating route modules:
- `/api/chat` inline vs `chat.py`
- `/api/data` inline vs `data.py`
- `/api/crawl/*` inline vs `crawl.py`

Use route modules for new endpoints; inline endpoints are for backward compatibility.

## ANTI-PATTERNS

- **NEVER** add new inline endpoints to dashboard_api.py (use route modules)
- **NEVER** skip API key verification on write operations
- **NEVER** use simple string comparison for auth (use hmac.compare_digest)
- **NEVER** expose internal errors in API responses
