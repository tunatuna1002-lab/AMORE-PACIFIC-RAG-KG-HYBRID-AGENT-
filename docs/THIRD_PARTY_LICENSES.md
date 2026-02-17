# Third-Party Licenses

This document lists all third-party libraries and resources used in this project along with their licenses.

## Core Dependencies

### Web Framework & Server

| Library | Version | License | URL |
|---------|---------|---------|-----|
| FastAPI | >=0.104.0 | MIT | https://github.com/tiangolo/fastapi |
| Uvicorn | >=0.24.0 | BSD-3-Clause | https://github.com/encode/uvicorn |
| Starlette | (via FastAPI) | BSD-3-Clause | https://github.com/encode/starlette |

### AI & Machine Learning

| Library | Version | License | URL |
|---------|---------|---------|-----|
| OpenAI Python | >=1.30.0 | Apache-2.0 | https://github.com/openai/openai-python |
| LiteLLM | >=1.40.0 | MIT | https://github.com/BerriAI/litellm |
| Sentence-Transformers | >=2.2.0 | Apache-2.0 | https://github.com/UKPLab/sentence-transformers |
| ChromaDB | >=0.4.0 | Apache-2.0 | https://github.com/chroma-core/chroma |

### Data Processing

| Library | Version | License | URL |
|---------|---------|---------|-----|
| Pandas | >=2.0.0 | BSD-3-Clause | https://github.com/pandas-dev/pandas |
| NumPy | >=1.24.0 | BSD-3-Clause | https://github.com/numpy/numpy |
| Pydantic | >=2.0.0 | MIT | https://github.com/pydantic/pydantic |

### Google APIs

| Library | Version | License | URL |
|---------|---------|---------|-----|
| google-api-python-client | >=2.100.0 | Apache-2.0 | https://github.com/googleapis/google-api-python-client |
| google-auth-httplib2 | >=0.2.0 | Apache-2.0 | https://github.com/googleapis/google-auth-library-python-httplib2 |
| google-auth-oauthlib | >=1.2.0 | Apache-2.0 | https://github.com/googleapis/google-auth-library-python-oauthlib |

### Web Scraping & HTTP

| Library | Version | License | URL |
|---------|---------|---------|-----|
| Playwright | >=1.40.0 | Apache-2.0 | https://github.com/microsoft/playwright-python |
| HTTPX | >=0.25.0 | BSD-3-Clause | https://github.com/encode/httpx |
| AioHTTP | >=3.9.0 | Apache-2.0 | https://github.com/aio-libs/aiohttp |

### Document Processing

| Library | Version | License | URL |
|---------|---------|---------|-----|
| python-docx | >=1.1.0 | MIT | https://github.com/python-openxml/python-docx |
| openpyxl | >=3.1.0 | MIT | https://github.com/theorchard/openpyxl |

### Utilities

| Library | Version | License | URL |
|---------|---------|---------|-----|
| python-dotenv | >=1.0.0 | BSD-3-Clause | https://github.com/theskumar/python-dotenv |
| structlog | >=23.1.0 | Apache-2.0 | https://github.com/hynek/structlog |
| python-dateutil | >=2.8.2 | Apache-2.0 & BSD-3-Clause | https://github.com/dateutil/dateutil |

---

## External APIs & Services

### OpenAI API
- **Provider**: OpenAI, Inc.
- **Model Used**: gpt-4o-mini
- **Terms of Service**: https://openai.com/policies/terms-of-use
- **Pricing**: Pay-per-use (https://openai.com/pricing)
- **License**: Proprietary API, Apache-2.0 client library

### Google Sheets API
- **Provider**: Google LLC
- **Terms of Service**: https://cloud.google.com/terms
- **Pricing**: Free tier available, quota-based
- **License**: Apache-2.0 client library

### Hugging Face Models
- **Model**: all-MiniLM-L6-v2
- **Author**: Microsoft Research
- **License**: Apache-2.0
- **Model Card**: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2

---

## Data Sources

### Amazon Best Sellers
- **Source**: Amazon.com Best Sellers pages
- **URL Pattern**: https://www.amazon.com/Best-Sellers-*/zgbs/*
- **Data Type**: Publicly available product rankings
- **Usage**: Educational/research purposes
- **Compliance**:
  - robots.txt: `/zgbs` path not explicitly disallowed for general crawlers
  - Rate limiting: 1 request per day per category
  - Delay: 2+ seconds between requests

**Important Notes**:
- This project collects publicly available ranking data for educational and research purposes only
- No personal user data is collected
- Commercial redistribution of collected data is not permitted
- Users should review Amazon's Terms of Service before deploying

---

## License Compatibility

All libraries used in this project have licenses that are compatible with the MIT License:

- **MIT**: Permissive, allows commercial use
- **Apache-2.0**: Permissive, patent protection included
- **BSD-3-Clause**: Permissive, similar to MIT

The combined work is distributed under the MIT License.

---

## Contact

For any licensing questions or concerns, please open an issue in the project repository.

Last updated: 2026-01-18
