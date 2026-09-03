# External Signal Failure Warnings

## Overview

As of v2026.01.27, the system now provides **explicit warning messages** when external signal collectors fail to load or execute, instead of silently skipping them.

## Problem Statement

Previously, when external signal collectors (Google Trends, YouTube, Tavily, RSS, Reddit) were unavailable due to:
- Missing API keys
- Import errors (missing dependencies)
- API quota limits
- Network issues

The system would **silently skip** these data sources, leaving users unaware that:
1. Some data sources were not used in the analysis
2. The insights might be incomplete
3. There was an issue that needed attention

## Solution

### 1. Import-Time Logging

When collectors fail to import, warnings are logged immediately:

```python
# hybrid_insight_agent.py
try:
    from src.tools.google_trends_collector import GoogleTrendsCollector
    GOOGLE_TRENDS_AVAILABLE = True
except ImportError as e:
    _logger.warning(f"GoogleTrendsCollector not available - Google Trends signals will be skipped: {e}")
    GOOGLE_TRENDS_AVAILABLE = False
```

**Log Output:**
```
[2026-01-27 10:30:15] [WARNING] GoogleTrendsCollector not available - Google Trends signals will be skipped: No module named 'pytrends'
[2026-01-27 10:30:15] [WARNING] YouTubeCollector not available - YouTube signals will be skipped: No module named 'google.auth'
```

### 2. Runtime Detection

During execution, the system checks which collectors failed:

```python
def _get_failed_signal_collectors(self) -> List[str]:
    """사용 불가능한 외부 신호 수집기 목록 반환"""
    failed = []

    if not GOOGLE_TRENDS_AVAILABLE:
        failed.append("Google Trends")

    if not YOUTUBE_AVAILABLE:
        failed.append("YouTube")

    # ... additional checks

    return failed
```

### 3. User-Facing Warnings

#### A. Insight Reports

When generating daily insights, failed collectors are explicitly mentioned:

```markdown
# LANEIGE Amazon US 일일 인사이트

## 📌 오늘의 핵심
...

## ⚠️ 주의 사항
...

> ⚠️ **외부 신호 수집 실패**: Google Trends, YouTube
> *(위 데이터 소스는 현재 사용할 수 없습니다. 분석은 나머지 데이터를 기반으로 수행되었습니다.)*
```

#### B. Chatbot Responses

Chatbot responses also display warnings:

```markdown
LANEIGE의 Lip Sleeping Mask는 Lip Care 카테고리에서 4위를 기록하고 있습니다...

> ⚠️ **외부 신호 수집 실패**: External Signals (Tavily/RSS/Reddit)
> *(위 데이터 소스는 현재 사용할 수 없습니다. 응답은 나머지 데이터를 기반으로 생성되었습니다.)*

---
**📚 출처 및 참고자료:**
...
```

## Monitored Collectors

The system tracks the following external signal collectors:

| Collector | Purpose | Failure Detection |
|-----------|---------|-------------------|
| **Google Trends** | Search volume trends | Import-time + Runtime |
| **YouTube** | Video reviews & engagement | Import-time + Runtime |
| **Tavily News** | Real-time news search | Runtime only |
| **RSS Feeds** | Industry publications (Allure, WWD, etc.) | Runtime only |
| **Reddit** | Consumer discussions | Runtime only |
| **Market Intelligence** | Multi-layer economic data | Runtime only |

## Implementation Details

### Modified Files

1. **`src/agents/hybrid_insight_agent.py`**
   - Added import-time logging for Google Trends and YouTube
   - Added `_get_failed_signal_collectors()` method
   - Modified `_generate_daily_insight()` to include warning section

2. **`src/agents/hybrid_chatbot_agent.py`**
   - Added `_get_failed_signal_collectors()` method
   - Modified response formatting to include inline warnings

### Code Flow

```
Import Time:
├─ Try import GoogleTrendsCollector
│  └─ If fails → Log warning + Set GOOGLE_TRENDS_AVAILABLE = False
├─ Try import YouTubeCollector
│  └─ If fails → Log warning + Set YOUTUBE_AVAILABLE = False

Runtime (Insight Generation):
├─ Call _collect_external_signals()
├─ Call _get_failed_signal_collectors()
│  ├─ Check GOOGLE_TRENDS_AVAILABLE
│  ├─ Check YOUTUBE_AVAILABLE
│  ├─ Try import ExternalSignalCollector
│  └─ Try import MarketIntelligenceEngine
├─ Generate insight with LLM
└─ Append warning section if failed_signals is not empty

Runtime (Chatbot):
├─ Call _collect_external_signals()
├─ Call _get_failed_signal_collectors()
├─ Generate response with LLM
└─ Insert warning before sources section if failed_signals is not empty
```

## Testing

Run the test script to verify functionality:

```bash
python test_failed_signals.py
```

Expected output:
- ✅ All collectors available → No warnings
- ⚠️ Collectors unavailable → Explicit warnings displayed

## User Benefits

1. **Transparency**: Users know exactly which data sources were used
2. **Debuggability**: Clear logs help identify configuration issues
3. **Trust**: Explicit warnings about missing data build trust
4. **Actionability**: Users can fix issues (add API keys, install dependencies)

## Common Causes & Solutions

### Google Trends Unavailable

**Cause:**
```
ImportError: No module named 'pytrends'
```

**Solution:**
```bash
pip install pytrends
```

### YouTube Unavailable

**Cause:**
```
ImportError: No module named 'google.auth'
```

**Solution:**
```bash
pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib
```

### Tavily News Unavailable

**Cause:**
```
Environment variable TAVILY_API_KEY not set
```

**Solution:**
```bash
export TAVILY_API_KEY=tvly-...
```

Or in `.env`:
```
TAVILY_API_KEY=tvly-...
```

### Market Intelligence Unavailable

**Cause:**
```
ImportError: No module named 'src.tools.market_intelligence'
```

**Solution:**
```bash
# Ensure all dependencies are installed
pip install -r requirements.txt
```

## Related Documentation

- [External Signal Collector](EXTERNAL_SIGNAL_COLLECTOR.md)
- [Tavily News Integration](TAVILY_NEWS_INTEGRATION.md)
- [Market Intelligence System](INSIGHT_SYSTEM_ARCHITECTURE.md)

## Changelog

### v2026.01.27 - Initial Implementation
- Added import-time logging for Google Trends and YouTube
- Added `_get_failed_signal_collectors()` method
- Added warning sections to insight reports and chatbot responses
- Created test script `test_failed_signals.py`
