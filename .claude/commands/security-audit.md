# Security Audit for AMORE RAG Agent

Perform a comprehensive security audit on changed or specified files.

## Usage
```
/project:security-audit [path]
```

## Focus Areas

### 1. Scraping Security (`src/tools/`)
- Rate limiting implementation in amazon_scraper.py
- User-agent rotation and fingerprint randomization
- Credential/API key leakage in scraper code
- Playwright stealth configuration

### 2. API Security (`src/api/`)
- API key validation in routes
- SQL injection patterns in SQLite queries
- CORS configuration review
- Input validation on endpoints
- Rate limiting (slowapi) configuration

### 3. Authentication & Secrets
- Hardcoded credentials in source files
- .env file exposure risk
- Google Sheets credential handling
- OpenAI API key masking in logs

### 4. Data Privacy
- PII in crawled product data
- KG backup security (data/backups/)
- Railway environment isolation
- Session data handling

### 5. Dependency Security
- Known vulnerabilities in requirements.txt
- Outdated packages with CVEs

## Audit Checklist

- [ ] No hardcoded API keys or passwords
- [ ] All user inputs sanitized
- [ ] SQL queries use parameterized statements
- [ ] Sensitive data masked in logs
- [ ] CORS properly configured
- [ ] Rate limiting active on public endpoints
- [ ] Dependencies up-to-date

## Run This Audit

Analyze the specified files for security issues. Report findings with severity levels:
- **CRITICAL**: Immediate action required
- **HIGH**: Should be fixed before deployment
- **MEDIUM**: Should be addressed in next sprint
- **LOW**: Nice to have improvements

$ARGUMENTS
