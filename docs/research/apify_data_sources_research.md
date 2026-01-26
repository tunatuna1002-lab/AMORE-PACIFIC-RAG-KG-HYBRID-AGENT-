# Apify 데이터 소스 조사 결과

> 조사일: 2026-01-26
> 상태: 보류 (추후 구현 예정)

---

## 요약

Apify 무료 크레딧 ($5/월)으로 다음 데이터 수집 가능:

| 소스 | 월 비용 | 우선순위 |
|------|---------|----------|
| YouTube | $0 (10K 무료) | ⭐⭐⭐⭐⭐ |
| Pinterest | $0 (20K 무료) | ⭐⭐⭐⭐ |
| TikTok | ~$0.12 | ⭐⭐⭐⭐⭐ |
| Instagram | ~$0.08 | ⭐⭐⭐⭐ |
| Google Trends | $0 (trendspyg) | ⭐⭐⭐⭐ |
| Amazon Best Sellers | ~$1.50 | ⭐⭐⭐ (기존 대체) |

**총 예상 비용: $0 ~ $2/월 (무료 크레딧 내)**

---

## 1. 소셜 미디어 스크래퍼

### TikTok
- **Actor**: [clockworks/tiktok-scraper](https://apify.com/clockworks/tiktok-scraper)
- **가격**: $0.30/1,000 posts
- **사용자**: 119K, 평점 4.6
- **수집 가능**: 해시태그, 비디오, 프로필, 조회수, 좋아요

```python
run_input = {
    "hashtags": ["LANEIGE", "KBeauty", "LipSleepingMask"],
    "resultsPerHashtag": 50
}
```

### Instagram
- **Actor**: [apify/instagram-scraper](https://apify.com/apify/instagram-scraper)
- **가격**: $0.25/1,000 posts
- **사용자**: 171K, 평점 4.7
- **주의**: 안티봇 매우 강력

### YouTube
- **Actor**: [streamers/youtube-scraper](https://apify.com/streamers/youtube-scraper)
- **가격**: 무료 크레딧으로 10,000 결과 가능
- **수집 가능**: 제목, 조회수, 좋아요, 댓글수, 자막

```python
run_input = {
    "searchKeywords": ["LANEIGE review", "Lip Sleeping Mask review"],
    "maxResults": 50,
    "sortBy": "date"
}
```

### Pinterest
- **Actor**: [epctex/pinterest-scraper](https://apify.com/epctex/pinterest-scraper)
- **가격**: $0.25/1,000 pins
- **무료**: 20,000 pins 가능!
- **활용**: 뷰티 트렌드 예측 (Pinterest Predicts 88% 정확도)

### X/Twitter
- **Actor**: [apidojo/tweet-scraper](https://apify.com/apidojo/tweet-scraper)
- **가격**: ⚠️ 무료 플랜 $40/1K (사실상 사용 불가)
- **대안**: Google Alerts로 대체 권장

---

## 2. 커머스 스크래퍼

### Amazon Best Sellers (기존 대체 가능)
- **Actor**: [junglee/amazon-bestsellers](https://apify.com/junglee/amazon-bestsellers)
- **가격**: $0.10/1,000 결과
- **장점**:
  - 속도 8배 향상 (80분 → 10분)
  - 차단 리스크 최소화
  - 유지보수 불필요
- **단점**: 브랜드 인식 후처리 필요

### Sephora
- **Actor**: [autofacts/sephora](https://apify.com/autofacts/sephora/api)
- **수집 가능**: 가격, 재고, 리뷰, 프로모션

### Ulta
- **Actor**: [mscraper/ulta-scraper](https://apify.com/mscraper/ulta-scraper/api)
- **수집 가능**: 가격, 프로모션

---

## 3. 무료 데이터 소스 (API 키 불필요)

### Google Trends
- **라이브러리**: [trendspyg](https://github.com/flack0x/trendspyg) (pytrends 대체)
- **설치**: `pip install trendspyg`

```python
from trendspyg import TrendReq

pytrends = TrendReq()
pytrends.build_payload(
    kw_list=["LANEIGE", "COSRX", "Korean skincare"],
    geo="US"
)
interest_over_time = pytrends.interest_over_time()
```

### Google Alerts
- **라이브러리**: [google-alerts](https://github.com/9b/google-alerts)
- **설치**: `pip install google-alerts`
- **주의**: 별도 이메일 계정 권장 (보안)

### RSS Feeds (뷰티 뉴스)
| 소스 | RSS URL |
|------|---------|
| Allure | allure.com/feed/rss |
| WWD Beauty | wwd.com/beauty-industry-news/feed |
| Byrdie | byrdie.com/rss |
| Cosmetics Business | cosmeticsbusiness.com/rss |
| Premium Beauty News | premiumbeautynews.com/rss |

---

## 4. 환경 변수 설정

`.env` 파일에 추가:

```bash
# Apify
APIFY_API_TOKEN=apify_api_xxxxxxxxxxxxxxxxxxxxxxxx

# Google Alerts (선택사항 - 별도 계정 권장)
GOOGLE_ALERTS_EMAIL=alerts@example.com
GOOGLE_ALERTS_PASSWORD=xxxxxxxx
```

---

## 5. 비용 최적화 전략 ($5/월 내)

| 소스 | 빈도 | 수집량 | 예상 비용 |
|------|------|--------|-----------|
| YouTube | 주 1회 | 200 videos | $0 |
| Pinterest | 주 1회 | 500 pins | $0 |
| TikTok | 주 2회 | 200 posts | ~$0.12 |
| Instagram | 주 1회 | 300 posts | ~$0.08 |
| Google Trends | 일 1회 | 무제한 | $0 |
| RSS Feeds | 일 1회 | 무제한 | $0 |
| **합계** | | | **~$0.20/월** |

---

## 6. 구현 우선순위

### Phase 1 (즉시 구현 가능 - 무료)
1. Google Trends (trendspyg)
2. RSS Feeds 확장
3. YouTube Scraper

### Phase 2 (Apify 무료 크레딧)
4. Pinterest Scraper
5. TikTok Scraper
6. Instagram Scraper

### Phase 3 (기존 시스템 대체 검토)
7. Amazon Best Sellers → Apify 하이브리드

---

## 7. 참고 링크

### Apify
- [Apify Store](https://apify.com/store)
- [Apify Pricing](https://apify.com/pricing)
- [Apify Free Tier Guide](https://use-apify.com/docs/what-is-apify/apify-free-plan)

### 소셜 미디어
- [TikTok Creative Center](https://ads.tiktok.com/business/creativecenter/pc/en)
- [Pinterest Predicts 2026](https://professionalbeauty.co.uk/pinterest-predicts-2026-beauty-wellness-trends)

### 도구
- [trendspyg GitHub](https://github.com/flack0x/trendspyg)
- [google-alerts GitHub](https://github.com/9b/google-alerts)
- [Top 100 Beauty RSS Feeds](https://rss.feedspot.com/beauty_rss_feeds/)

---

## 8. 코드 스니펫

### Apify 기본 사용법

```python
from apify_client import ApifyClient
import os

client = ApifyClient(os.getenv("APIFY_API_TOKEN"))

# TikTok 해시태그 스크래핑
run = client.actor("clockworks/tiktok-hashtag-scraper").call(
    run_input={
        "hashtags": ["LANEIGE", "KBeauty"],
        "resultsPerHashtag": 50
    }
)

for item in client.dataset(run["defaultDatasetId"]).iterate_items():
    print(item)
```

### 통합 수집기 구조 (추후 구현)

```python
# src/tools/social_trend_collector.py

class SocialTrendCollector:
    """TikTok/Instagram/YouTube/Pinterest 통합 수집기"""

    def __init__(self):
        self.apify_client = ApifyClient(os.getenv("APIFY_API_TOKEN"))

    async def fetch_tiktok_trends(self, hashtags: List[str]) -> List[Dict]:
        """TikTok 해시태그 트렌드"""
        pass

    async def fetch_youtube_reviews(self, keywords: List[str]) -> List[Dict]:
        """YouTube 리뷰 영상"""
        pass

    async def fetch_pinterest_trends(self, keywords: List[str]) -> List[Dict]:
        """Pinterest 트렌드 핀"""
        pass

    async def fetch_google_trends(self, keywords: List[str]) -> pd.DataFrame:
        """Google Trends 검색 관심도"""
        pass
```

---

## 변경 이력

| 날짜 | 변경 내용 |
|------|----------|
| 2026-01-26 | 초기 조사 완료, 문서 작성 |
