# Data Collection Policy

## Overview

This document describes the data collection practices of the AMORE-RAG-ONTOLOGY-HYBRID AGENT project, including compliance with platform policies and legal requirements.

---

## Amazon Best Sellers Data Collection

### Purpose
- Educational and research analysis of publicly available e-commerce ranking data
- Market trend analysis for academic purposes
- Non-commercial demonstration of AI-powered analytics

### Data Collected
| Data Type | Source | Personal Data | Sensitive |
|-----------|--------|---------------|-----------|
| Product Name | Public listing | No | No |
| ASIN (Product ID) | Public listing | No | No |
| Price | Public listing | No | No |
| Rating | Public aggregate | No | No |
| Review Count | Public aggregate | No | No |
| Rank Position | Public listing | No | No |
| Category | Public listing | No | No |

### Compliance Measures

#### 1. robots.txt Compliance
- **Status**: Compliant
- **Verification**: Amazon's robots.txt does not explicitly disallow `/zgbs` (Best Sellers) paths for general crawlers
- **Last Checked**: 2026-01-18

```
# Amazon robots.txt analysis
# Path: /zgbs/* (Best Sellers)
# Result: Not explicitly disallowed
```

#### 2. Rate Limiting
- **Frequency**: Maximum 1 crawl per day
- **Time**: 06:00 KST (21:00 UTC previous day)
- **Delay**: Minimum 2 seconds between requests
- **Categories**: 5 categories × 100 products = 500 products/day

#### 3. Technical Safeguards
```python
# Implemented in src/tools/amazon_scraper.py
- Random User-Agent rotation
- Request delay (2+ seconds)
- Headless browser (Playwright)
- Block detection and graceful handling
- No authentication bypass attempts
```

#### 4. Data Retention
- Data stored in Google Sheets for historical analysis
- Local JSON cache for dashboard display
- No indefinite storage of raw HTML
- Aggregated metrics only

---

## Alternative Data Sources (Recommended)

For production use, consider these official alternatives:

### Amazon Product Advertising API (PA-API)
- **URL**: https://affiliate-program.amazon.com/
- **Benefits**: Official API, stable, compliant
- **Limitations**: Requires affiliate account, rate limits apply
- **Cost**: Free (with affiliate agreement)

### Third-Party Data Providers
- Jungle Scout API
- Helium 10 API
- Keepa API

---

## Legal Disclaimer

1. **Educational Use**: This project is developed for educational and research purposes as part of an academic competition.

2. **No Warranty**: The data collection methods described are for demonstration only. Users deploying this system should conduct their own legal review.

3. **Platform Terms**: Users are responsible for reviewing and complying with Amazon's Terms of Service and any applicable platform policies.

4. **No Personal Data**: This system does not collect, process, or store any personal information. All data is publicly available product information.

5. **Non-Commercial**: Collected data is not intended for commercial redistribution.

---

## Contact

For questions about data collection practices, please contact the project maintainers.

---

## References

- Amazon robots.txt: https://www.amazon.com/robots.txt
- Amazon Conditions of Use: https://www.amazon.com/gp/help/customer/display.html?nodeId=508088
- PA-API Documentation: https://webservices.amazon.com/paapi5/documentation/

Last updated: 2026-01-18
