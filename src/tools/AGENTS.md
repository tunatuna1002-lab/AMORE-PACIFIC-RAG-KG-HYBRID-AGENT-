# src/tools - Utilities & Collectors

## OVERVIEW

Scrapers, social media collectors, external data fetchers, backup services, and metric calculators.

## KEY MODULES

| Module | File | Role |
|--------|------|------|
| AmazonScraper | `amazon_scraper.py` | Playwright + stealth, AWS WAF evasion |
| TikTokCollector | `tiktok_collector.py` | Playwright page eval |
| InstagramCollector | `instagram_collector.py` | Instaloader wrapper |
| YouTubeCollector | `youtube_collector.py` | yt-dlp wrapper |
| RedditCollector | `reddit_collector.py` | Reddit JSON API |
| GoogleTrendsCollector | `google_trends_collector.py` | trendspyg/pytrends |
| ExternalSignalCollector | `external_signal_collector.py` | RSS + Reddit + Tavily aggregator |
| PublicDataCollector | `public_data_collector.py` | data.go.kr APIs |
| KGBackupService | `kg_backup.py` | 7-day rolling KG backups |
| MetricCalculator | `metric_calculator.py` | SoS, HHI, CPI |
| EmailSender | `email_sender.py` | Gmail SMTP |

## ASYNC PATTERNS

| Tool | Pattern | Reason |
|------|---------|--------|
| AmazonScraper | Native async | Playwright async API |
| TikTokCollector | Native async | Playwright async API |
| InstagramCollector | ThreadPoolExecutor | Instaloader is sync |
| YouTubeCollector | ThreadPoolExecutor | yt-dlp is sync |
| RedditCollector | Native async | aiohttp |
| GoogleTrendsCollector | ThreadPoolExecutor | trendspyg/pytrends sync |
| PublicDataCollector | Native async | aiohttp |

## SCRAPER PATTERNS

```python
# AmazonScraper lifecycle
async with AmazonScraper() as scraper:
    products = await scraper.crawl_category("lip_care")
```

- Circuit breaker for repeated failures
- Exponential backoff on WAF blocks
- Debug screenshots saved to `logs/crawler_screenshots/`

## ANTI-PATTERNS

- **NEVER** use sync requests in async context (wrap with ThreadPoolExecutor)
- **NEVER** hardcode rate limits (use config)
- **NEVER** skip stealth context for Amazon scraping
- **NEVER** store credentials in collector classes
